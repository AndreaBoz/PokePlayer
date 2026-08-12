import asyncio
from typing import Any

import time
import os

import numpy as np
import random
import torch
import supersuit as ss
from scipy import stats
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from poke_env.battle import AbstractBattle
from poke_env.environment import SinglesEnv
from poke_env.player import (
    DefaultBattleOrder,
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.poke_player.obs_space_helper_v1_7 import *

BATTLE_FORMAT = "gen4randombattle"
N_FEATURES = 318

def compute_ci95(values):
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, mean, mean
    sem = np.std(values, ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    margin = t_crit * sem
    return mean, mean - margin, mean + margin


# IMPOSTAZIONE DEI SEED
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            net_arch=[256, 256],
            features_extractor_class=FeaturesExtractor,
        )

    def forward(self, obs, deterministic=False):
        self._mask = obs["action_mask"]
        return super().forward(obs, deterministic)

    def evaluate_actions(self, obs, actions):
        self._mask = obs["action_mask"]
        return super().evaluate_actions(obs, actions)

    def _get_action_dist_from_latent(self, latent_pi):
        action_logits = self.action_net(latent_pi)
        mask = torch.where(self._mask == 1, 0, float("-inf"))
        return self.action_dist.proba_distribution(action_logits + mask)


class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=N_FEATURES)

    def forward(self, obs):
        return obs["observation"]      


class PolicyPlayer(Player):
    policy: ActorCriticPolicy | None

    def __init__(
        self, policy: ActorCriticPolicy | None = None, *args: Any, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.policy = policy

    def choose_move(self, battle: AbstractBattle):
        if battle.wait:
            return DefaultBattleOrder()

        features = self.embed_battle(battle)
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=np.float32)

        with torch.no_grad():
            obs_dict = {
                "observation": torch.as_tensor(features, device=self.policy.device).unsqueeze(0),
                "action_mask": torch.as_tensor(mask, device=self.policy.device).unsqueeze(0),
            }
            action, _, _ = self.policy.forward(obs_dict)

        action = action.cpu().numpy()[0]
        return SinglesEnv.action_to_order(action, battle)

    @staticmethod
    def embed_battle(battle: AbstractBattle):

        moves_base_power, moves_dmg_multiplier = get_moves_base_power_and_dmg_multiplier(battle)
        fainted_mon_player = get_fainted_ratio(battle.team)
        fainted_mon_opponent = get_fainted_ratio(battle.opponent_team)
        our_hp = get_mon_hp_fraction(battle.active_pokemon)
        opp_hp = get_mon_hp_fraction(battle.opponent_active_pokemon)
        threat_level = get_opponent_type_threat(battle)
        entry_hazards = get_entry_hazards(battle)
        weather = get_weather(battle)
        team_statuses = get_full_team_statuses(battle)
        volatile_statuses = get_active_volatile_statuses(battle)
        active_boosts = get_active_boosts(battle)
        speed_comparison = get_speed_comparison(battle)
        team_matchups = get_team_matchups(battle)
        opp_evaluation = get_opp_evaluation(battle)
        turn_progress = get_turn_progress(battle)

        features = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_player, fainted_mon_opponent],
                [our_hp, opp_hp],
                [threat_level],
                entry_hazards,
                weather,
                team_statuses,
                active_boosts,
                volatile_statuses,
                speed_comparison,
                team_matchups,
                opp_evaluation,
                [turn_progress],
            ],
            dtype=np.float32,
        )
        return features


REWARD_MODES = ("shaped", "sparse")


class SelfPlayEnv(SinglesEnv):
    def __init__(self, reward_mode: str = "shaped", **kwargs):
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode deve essere uno tra {REWARD_MODES}, ricevuto: {reward_mode!r}")
        super().__init__(**kwargs)
        self.reward_mode = reward_mode
        self.observation_spaces = {
            agent: Box(-1.0, 4.0, shape=(N_FEATURES,), dtype=np.float32)
            for agent in self.possible_agents
        }

    def calc_reward(self, battle) -> float:
        if self.reward_mode == "sparse":
            return self.reward_computing_helper(battle, victory_value=30.0)
        return self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=1.0,
            status_value=0.5, victory_value=30.0,
        )

    def embed_battle(self, battle: AbstractBattle):
        return PolicyPlayer.embed_battle(battle)

def train(seed: int = 42, reward_mode: str = "shaped"):
    # Setup
    seed_start = time.time()

    set_global_seed(seed)
    num_envs = 7
    env = SelfPlayEnv(
        reward_mode=reward_mode, battle_format=BATTLE_FORMAT, log_level=40, open_timeout=None
    )

    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    vec_env = ss.concat_vec_envs_v1(
        vec_env, num_vec_envs=num_envs, num_cpus=num_envs,
        base_class="stable_baselines3",
    )

    ppo = PPO(
        MaskedActorCriticPolicy,
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        gamma=0.99,
        n_epochs=4,
        ent_coef=0.01,
        device="cpu",
        seed=None,
        tensorboard_log="./tb_logs",
    )

    os.makedirs("checkpoints", exist_ok=True)

    CHECKPOINT_EVERY = 200_000
    total_timesteps = 10_000_000
    n_chunks = total_timesteps // CHECKPOINT_EVERY

    for i in range(n_chunks):
        chunk_t0 = time.time()
        ppo.learn(
            CHECKPOINT_EVERY, reset_num_timesteps=False,
            tb_log_name=f"seed_{seed}_{reward_mode}", progress_bar=True,
        )
        ppo.save(f"checkpoints/seed{seed}_{reward_mode}_chunk{i}")
        print(f"[chunk {i+1}/{n_chunks}] {CHECKPOINT_EVERY} step in {time.time()-chunk_t0:.1f}s "
              f"(totale finora: {(time.time()-seed_start)/60:.1f} min)")

    vec_env.close()
    print(f"[Seed {seed}, reward_mode={reward_mode}] Training totale completato in "f"{(time.time() - seed_start)/60:.1f} min")

    # Valutazione
    agent = PolicyPlayer(
        policy=ppo.policy, battle_format=BATTLE_FORMAT, max_concurrent_battles=10
    )
    opponents: list[Player] = [
        c(battle_format=BATTLE_FORMAT, max_concurrent_battles=1)
        for c in [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]
    ]
    asyncio.run(agent.battle_against(*opponents, n_battles=100))

    results = {}
    print(f"\n--- Win rates vs bots (Seed {seed}, reward_mode={reward_mode}) ---")
    for opp_cls, opp in zip(
        [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer], opponents
    ):
        win_rate = round(100 * opp.n_lost_battles / opp.n_finished_battles)
        results[opp_cls.__name__] = win_rate     # era: results[opp.username]
        print(f"{opp.username}: {win_rate}%")

    for opp in opponents:
        print(f"{opp.username}: {opp.n_finished_battles} battaglie completate su 100 richieste")
    if opp.n_finished_battles < 100:
        print("  ATTENZIONE: alcune battaglie non sono state completate")

    return results


if __name__ == "__main__":
    seeds = [40, 50]
    reward_mode = "shaped"  # "shaped" oppure "sparse"
    all_results = []

    for s in seeds:
        print(f"\n--------------- STARTING RUN WITH SEED {s} (reward_mode={reward_mode}) ---------------")
        res = train(seed=s, reward_mode=reward_mode)
        all_results.append(res)

    print("\n--------------- RISULTATI FINALI AGGREGATI ---------------")
    for bot_name in all_results[0].keys():
        bot_win_rates = [run[bot_name] for run in all_results]
        mean_wr = np.mean(bot_win_rates)
        std_wr = np.std(bot_win_rates)
        mean_ci, ci_low, ci_high = compute_ci95(bot_win_rates)
        print(f"{bot_name}: {mean_wr:.2f}% ± {std_wr:.2f}% (dev. std., n={len(bot_win_rates)}) "
              f"| CI 95%: [{ci_low:.2f}%, {ci_high:.2f}%]")
