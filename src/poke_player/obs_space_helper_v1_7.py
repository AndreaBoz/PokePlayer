import numpy as np

from poke_env.battle import AbstractBattle
from poke_env.data import GenData
from poke_env.battle import Weather
from poke_env.battle.status import Status

from functools import lru_cache

@lru_cache(maxsize=None)
def _get_type_chart(gen: int):
    return GenData.from_gen(gen).type_chart

@lru_cache(maxsize=None)
def _cached_damage_multiplier(attacking_type, target_type_1, target_type_2, gen: int) -> float:
    #Utilzzo di cache per ottimizzare i cicli di ricerca all'interno di type_chart.

    type_chart = _get_type_chart(gen)
    return attacking_type.damage_multiplier(
        target_type_1, target_type_2, type_chart=type_chart
    )

def get_moves_base_power_and_dmg_multiplier(battle: AbstractBattle):

    moves_base_power = -np.ones(4, dtype=np.float32)
    moves_dmg_multiplier = np.zeros(4, dtype=np.float32)

    active_mon = battle.active_pokemon
    opp_mon = battle.opponent_active_pokemon

    if active_mon is not None:
        # Iteriamo sulle mosse effettive del Pokémon (fino a 4)
        for i, move in enumerate(active_mon.moves.values()):
            if i >= 4:
                break 
            # Potenza base normalizzata su 150.0 per mantenere i valori <= 1.0. (mosse come autodistruzione vengono cappate a 150)
            moves_base_power[i] = min(move.base_power, 150) / 150.0

            # Calcolo del moltiplicatore contro i tipi dell'avversario attivo
            if opp_mon is not None and opp_mon.type_1 is not None:
                moves_dmg_multiplier[i] = _cached_damage_multiplier(
                    move.type, opp_mon.type_1, opp_mon.type_2, battle.gen
                ) / 4.0

    return moves_base_power, moves_dmg_multiplier

def get_fainted_ratio(team_dict: dict) -> float:
    if not team_dict:
        return 0.0
    return len([mon for mon in team_dict.values() if mon.fainted]) / 6.0

def get_mon_hp_fraction(mon) -> float:
    #Restituisce la frazione di HP attuali del Pokémon [0.0, 1.0].
    return mon.current_hp_fraction if mon is not None else 0.0

def get_opponent_type_threat(battle: AbstractBattle) -> float:
    our_mon = battle.active_pokemon
    opp_mon = battle.opponent_active_pokemon

    if our_mon is None or opp_mon is None or our_mon.type_1 is None:
        return 0.25 # Valore neutro (1.0 / 4.0)

    threats = [
        _cached_damage_multiplier(opp_type, our_mon.type_1, our_mon.type_2, battle.gen)
        for opp_type in [opp_mon.type_1, opp_mon.type_2]
        if opp_type is not None
    ]
    max_threat = max(threats) if threats else 1.0

    # Normalizziamo su 4.0 per ottenere un valore compreso tra [0.0 e 1.0]
    return max_threat / 4.0

def get_entry_hazards(battle: AbstractBattle) -> np.ndarray:
    """
    Estrae il vettore degli Entry Hazard:
    - [0] Player Stealth Rock  (0.0 o 1.0)
    - [1] Player Spikes        (0.0, 0.33, 0.66, 1.0 -> 0-3 layer)
    - [2] Player Toxic Spikes  (0.0, 0.5, 1.0 -> 0-2 layer)
    - [3] Opponent Stealth Rock(0.0 o 1.0)
    - [4] Opponent Spikes      (0.0, 0.33, 0.66, 1.0)
    - [5] Opponent Toxic Spikes(0.0, 0.5, 1.0)
    """

    hazards = np.zeros(6, dtype=np.float32)

    our_side = battle.side_conditions

    # Stealth Rock (binario)
    if "pointedstones" in our_side or "stealthrock" in our_side:
        hazards[0] = 1.0

    # Spikes (max 3 layer)
    if "spikes" in our_side:
        hazards[1] = min(our_side["spikes"], 3) / 3.0

    # Toxic Spikes (max 2 layer)
    if "toxicspikes" in our_side:
        hazards[2] = min(our_side["toxicspikes"], 2) / 2.0

    opp_side = battle.opponent_side_conditions

    # Stealth Rock (binario)
    if "pointedstones" in opp_side or "stealthrock" in opp_side:
        hazards[3] = 1.0

    # Spikes (max 3 layer)
    if "spikes" in opp_side:
        hazards[4] = min(opp_side["spikes"], 3) / 3.0

    # Toxic Spikes (max 2 layer)
    if "toxicspikes" in opp_side:
        hazards[5] = min(opp_side["toxicspikes"], 2) / 2.0

    return hazards

def get_weather(battle: AbstractBattle) -> np.ndarray:
    """
    Estrae il vettore del meteo:
    - [0] Sun (SUNNYDAY)
    - [1] Rain (RAINDANCE)
    - [2] Sandstorm (SANDSTORM)
    - [3] Hail (HAIL)
    - [4] Fog / Unknown Weather

    Se nessun meteo è attivo, restituisce un vettore di zeri.
    """
    weather_vec = np.zeros(5, dtype=np.float32)

    if not battle.weather:
        return weather_vec

    w_str = str(battle.weather).upper()

    if "SUNNYDAY" in w_str or "SUN" in w_str:
        weather_vec[0] = 1.0

    elif "RAINDANCE" in w_str or "RAIN" in w_str:
        weather_vec[1] = 1.0

    elif "SANDSTORM" in w_str or "SAND" in w_str:
        weather_vec[2] = 1.0

    elif "HAIL" in w_str:
        weather_vec[3] = 1.0

    elif "FOG" in w_str or "UNKNOWN" in w_str:
        weather_vec[4] = 1.0

    return weather_vec

def _get_status(pokemon) -> np.ndarray:
    """
    Estrae un vettore di 6 elementi per lo stato non volatile di un Pokémon:
    - [0] Burn (BRN)
    - [1] Freeze (FRZ)
    - [2] Paralysis (PAR)
    - [3] Poison (PSN)
    - [4] Bad Poison / Toxic (TOX)
    - [5] Sleep (SLP)
    """
    status_vec = np.zeros(6, dtype=np.float32)

    if pokemon is None or not pokemon.status:
        return status_vec

    status_str = str(pokemon.status).upper()

    if "BRN" in status_str:
        status_vec[0] = 1.0
    elif "FRZ" in status_str:
        status_vec[1] = 1.0
    elif "PAR" in status_str:
        status_vec[2] = 1.0
    elif "PSN" in status_str:
        status_vec[3] = 1.0
    elif "TOX" in status_str:
        status_vec[4] = 1.0
    elif "SLP" in status_str:
        status_vec[5] = 1.0

    return status_vec

def get_full_team_statuses(battle: AbstractBattle) -> np.ndarray:
    """
    Estrae il vettore delle condizioni di stato non volatili per TUTTI i Pokémon di player e opponent:
    - I 6 Pokémon del Giocatore (6 float per Pokémon) = 36 elementi
    - I 6 Pokémon dell'Avversario (6 float per Pokémon) = 36 elementi
    """
    team_statuses = []

    # 1. Team del Giocatore
    player_team = list(battle.team.values())
    for i in range(6):
        if i < len(player_team):
            team_statuses.append(_get_status(player_team[i]))
        else:
            team_statuses.append(np.zeros(6, dtype=np.float32))

    # 2. Team Avversario
    opp_team = list(battle.opponent_team.values())
    for i in range(6):
        if i < len(opp_team):
            team_statuses.append(_get_status(opp_team[i]))
        else:
            # Pokémon NON ancora visto: inseriamo un vettore di 6 zeri (pari al numero di status possibili per singolo pokemon) per mantenere la dimensione del vettore fissa.
            team_statuses.append(np.zeros(6, dtype=np.float32))

    return np.concatenate(team_statuses)

def _get_boosts(pokemon) -> np.ndarray:
    """
    Estrae un vettore a 7 elementi normalizzati [-1.0, 1.0] per i boost del Pokémon:
    - [0]: Atk
    - [1]: Def
    - [2]: SpA
    - [3]: SpD
    - [4]: Spe
    - [5]: Accuracy
    - [6]: Evasion
    """
    boosts_vec = np.zeros(7, dtype=np.float32)

    if pokemon is None:
        return boosts_vec

    # Mappatura ordinata delle 7 statistiche
    stats_order = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

    for i, stat in enumerate(stats_order):
        raw_boost = pokemon.boosts.get(stat, 0)
        # Normalizzazione da [-6, 6] a [-1.0, 1.0]
        boosts_vec[i] = raw_boost / 6.0

    return boosts_vec

def get_active_boosts(battle: AbstractBattle) -> np.ndarray:
    """
    Restituisce la concatenazione dei boost dei Pokémon attivi (14 float totali):
    - [0-6]:  Pokémon attivo del Giocatore
    - [7-13]: Pokémon attivo dell'Avversario
    """
    player_boosts = _get_boosts(battle.active_pokemon)
    opp_boosts = _get_boosts(battle.opponent_active_pokemon)

    return np.concatenate([player_boosts, opp_boosts])

def _get_volatile_status(pokemon) -> np.ndarray:
    """
    Estrae un vettore di 6 float per le condizioni volatili del Pokémon:
    - [0]: Confusion
    - [1]: Drowsy (Yawn)
    - [2]: Infatuated (Attract)
    - [3]: Leech Seed
    - [4]: Trapped
    - [5]: Must Recharge
    """
    volatile_vec = np.zeros(6, dtype=np.float32)
    if pokemon is None:
        return volatile_vec

    if pokemon.effects:
        effects_str = {str(eff).lower() for eff in pokemon.effects.keys()}

        # Confusion
        if any("confusion" in eff for eff in effects_str):
            volatile_vec[0] = 1.0

        # Drowsy / Yawn
        if any("yawn" in eff for eff in effects_str):
            volatile_vec[1] = 1.0

        # Infatuated / Attract
        if any("attract" in eff for eff in effects_str):
            volatile_vec[2] = 1.0

        # Leech Seed
        if any("leech" in eff for eff in effects_str):
            volatile_vec[3] = 1.0

        # Trapped
        trapped_keywords = ["trapped", "mean_look", "block", "spider_web", "arena_trap", "magnet_pull"]
        if any(any(kw in eff for kw in trapped_keywords) for eff in effects_str):
            volatile_vec[4] = 1.0

    # Indice 5: Must Recharge (non può agire)
    if getattr(pokemon, "must_recharge", False):
        volatile_vec[5] = 1.0

    return volatile_vec

def get_active_volatile_statuses(battle: AbstractBattle) -> np.ndarray:
    # Concatena i vettori degli stati volatili per i Pokémon attivi in campo (10 float totali):

    player_volatile = _get_volatile_status(battle.active_pokemon)
    opp_volatile = _get_volatile_status(battle.opponent_active_pokemon)

    # Concateniamo in un unico array di 12 float32
    return np.concatenate([player_volatile, opp_volatile])


def _get_boost_multiplier(stage: int) -> float:
    #Restituisce il moltiplicatore di statistica basato sullo stadio [-6, +6].
    if stage >= 0:
        return (2.0 + stage) / 2.0
    else:
        return 2.0 / (2.0 + abs(stage))

def _get_effective_speed(pokemon, battle: AbstractBattle) -> float:
    """
    Calcola la velocità reale/effettiva di un Pokémon considerando:
    - Statistica base / stimata
    - Modificatori dei boost (-6 a +6)
    - Paralisi (Gen 4: 0.25x)
    - Strumenti (Choice Scarf)
    - Abilità + Meteo (Swift Swim, Chlorophyll)
    """
    if pokemon is None:
        return 0.0

    # 1. Recupero della velocità base/stimata
    # Se le stat non sono del tutto calcolate (es. avversario sconosciuto), usiamo 300 come fallback medio
    raw_speed = 300.0
    if hasattr(pokemon, "stats") and pokemon.stats and pokemon.stats.get("spe") is not None:
        raw_speed = float(pokemon.stats["spe"])
    elif hasattr(pokemon, "base_stats") and pokemon.base_stats:
        # Stima approssimativa a livello 100 se conosciamo solo la base stat:
        # IV=31, EV=85 (floor(85/4)=21), costante +5 della formula ufficiale,
        # natura neutra -> floor((2*base+31+21)*100/100+5) = 2*base+57
        base_spe = pokemon.base_stats.get("spe", 80)
        raw_speed = float(base_spe * 2 + 31 + 21 + 5)

    # 2. Applicazione Boost (-6 a +6)
    spe_boost = pokemon.boosts.get("spe", 0)
    speed = raw_speed * _get_boost_multiplier(spe_boost)

    # 3. Stato Non-Volatile (Paralisi in Gen 4 = 0.25x)
    if pokemon.status == Status.PAR:
        speed *= 0.25

    # 4. Strumento (Choice Scarf)
    if pokemon.item == "choicescarf":
        speed *= 1.5

    # 5. Abilità Meteo
    ability = getattr(pokemon, "ability", "") or ""
    weather_str = str(battle.weather).lower() if battle and battle.weather else ""

    if ability == "swift-swim" or ability == "swiftswim":
        if "rain" in weather_str:
            speed *= 2.0
    elif ability == "chlorophyll":
        if "sun" in weather_str:
            speed *= 2.0

    return speed

def get_speed_comparison(battle: AbstractBattle) -> np.ndarray:
    """
    Restituisce un vettore a 3 elementi (considerando che la velocità max teorica in Gen 4 è 1000):
    - [0]: Velocità effettiva del giocatore / 1000.0
    - [1]: Velocità effettiva dell'avversario / 1000.0
    - [2]: Flag Outspeed (1.0 se player > opp, 0.0 se opp > player, 0.5 tie)
    """
    player_speed = _get_effective_speed(battle.active_pokemon, battle)
    opp_speed = _get_effective_speed(battle.opponent_active_pokemon, battle)

    player_speed_norm = min(player_speed / 1000.0, 1.0)
    opp_speed_norm = min(opp_speed / 1000.0, 1.0)

    # Flag di priorità/Outspeed
    if player_speed > opp_speed:
        outspeed_flag = 1.0
    elif opp_speed > player_speed:
        outspeed_flag = 0.0
    else:
        outspeed_flag = 0.5

    return np.array([player_speed_norm, opp_speed_norm, outspeed_flag], dtype=np.float32)

def get_team_matchups(battle: AbstractBattle) -> np.ndarray:
    """
    Estrae il matchup per tutti i 6 slot fissi della squadra (24 float totali: 6 slot x 4 info).

    Info per ogni slot (4 float):
    - [0]: Moltiplicatore offensivo dei tipi vs avversario
    - [1]: Resistenza difensiva vs tipi avversari
    - [2]: HP Fraction del Pokémon [0.0 - 1.0]
    - [3]: Flag is_active (1.0 se è il Pokémon attivo, 0.0 se è in panchina/svenuto)
    """

    opp_mon = battle.opponent_active_pokemon
    active_mon = battle.active_pokemon
    team_vec = np.zeros((6, 4), dtype=np.float32)

    if opp_mon is None or opp_mon.type_1 is None:
        return team_vec.flatten()

    opp_types = [t for t in [opp_mon.type_1, opp_mon.type_2] if t is not None]

    # Lista fissa dei membri della squadra (fino a 6)
    team_mons = list(battle.team.values())

    for i in range(6):
        if i < len(team_mons):
            mon = team_mons[i]
            if mon and not mon.fainted:
                mon_types = [t for t in [mon.type_1, mon.type_2] if t is not None]

                # Offensivo: Miglior moltiplicatore di danno che il tipo del Pokemon alleato può offrire contro opp.
                off_mults = [
                    _cached_damage_multiplier(t, opp_mon.type_1, opp_mon.type_2, battle.gen)
                    for t in mon_types
                ]
                max_off = max(off_mults) if off_mults else 1.0

                # Difensivo: Miglior moltiplicatore di danno che il tipo del Pokemon avversario può offrire contro player.
                def_mults = [
                    _cached_damage_multiplier(opp_t, mon.type_1, mon.type_2, battle.gen)
                    for opp_t in opp_types
                ]

                max_def_taken = max(def_mults) if def_mults else 1.0

                # Facciamo valere l'immunità il doppio di una doppia resistenza
                def_score = 8.0 if max_def_taken == 0 else (1.0 / max_def_taken)

                # Flag per identificare se questo specifico slot è in campo
                is_active = 1.0 if mon == active_mon else 0.0

                team_vec[i] = [
                    min(max_off / 4.0, 1.0),      # L'attacco viene normalizzato su 4.0 (massimo x4 efficacia).
                    min(def_score / 8.0, 1.0),    # La difesa viene normalizzata su 8.0 per separare immunità e doppia resistenza.
                    float(mon.current_hp_fraction),
                    is_active,
                ]

    return team_vec.flatten()

def get_opp_evaluation(battle: AbstractBattle) -> np.ndarray:
    """
    Estrae la matrice delle mosse rivelate dell'avversario contro la squadra del giocatore.
    Dimensione: 6 slot avversari * 4 mosse * (6 matchup + 1 PP ratio) = 168 float totali.
    """
    matrix = np.zeros((6, 4, 7), dtype=np.float32)

    player_team = list(battle.team.values())
    opp_team = list(battle.opponent_team.values())

    player_types = [
        (mon.type_1, mon.type_2) if mon and mon.type_1 is not None else None
        for mon in player_team
    ]

    for i in range(6):
        if i < len(opp_team):
            opp_mon = opp_team[i]
            # Consideriamo le mosse solo se il Pokémon è rivelato e ha mosse note
            if opp_mon and opp_mon.moves:
                opp_moves = list(opp_mon.moves.values())
                for m_idx in range(4):
                    if m_idx < len(opp_moves):
                        move = opp_moves[m_idx]
                        move_type = move.type

                        # 1. Moltiplicatore di danno della mossa vs i 6 Pokémon del player
                        for p_idx in range(6):
                            if p_idx < len(player_types) and player_types[p_idx] is not None:
                                type_1, type_2 = player_types[p_idx]
                                mult = _cached_damage_multiplier(
                                    move_type, type_1, type_2, battle.gen,
                                )
                                matrix[i, m_idx, p_idx] = min(mult / 4.0, 1.0)

                        # 2. Frazione PP rimanenti [0.0 - 1.0]
                        max_pp = getattr(move, "max_pp", 0) or 1
                        current_pp = getattr(move, "current_pp", max_pp)
                        matrix[i, m_idx, 6] = max(0.0, min(current_pp / float(max_pp), 1.0))

    return matrix.flatten()

def get_turn_progress(battle: AbstractBattle) -> float:
    
    #Turno corrente della battaglia, normalizzato in [0.0, 1.0].

    MAX_TURN = 100  # cap di normalizzazione oltre il quale si tronca a 1.0.
    return min(battle.turn / MAX_TURN, 1.0)
