# PokePlayer

Agente basato su Reinforcement Learning (PPO con action masking) per battaglie Pokémon in formato `gen4randombattle`, tramite [poke-env](https://github.com/hsahovic/poke-env) e [Pokémon Showdown](https://github.com/smogon/pokemon-showdown).

- `pokeRL_v1_7.py`: script di training e valutazione (self-play, reward configurabile shaped/sparse, rete 256x256, 318 feature in osservazione).
- `obs_space_helper_v1_7.py`: funzioni di estrazione del vettore delle osservazioni a partire dallo stato di battaglia.
- `Pokeplayer.pdf`: formalizzazione e primi risultati sperimentali.

## Requisiti

- Python 3.13
- Node.js (per il server Pokémon Showdown)

Dipendenze Python:
```
poke-env==0.12.1
stable-baselines3
supersuit
gymnasium
torch
numpy
scipy
```

Installazione in un virtualenv:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install poke-env==0.12.1 stable-baselines3 supersuit gymnasium torch numpy scipy
```

## Setup del server Pokémon Showdown

Il training richiede un'istanza locale del server, usata come simulatore di battaglia.

```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
```

Nel file `config/config.js`, impostare:
```js
simulator: 7
network: 2
```
Questi sono i valori ritenuti ottimali su Macbook air M1 con 8 core e 8 GB di ram.

Avvio del server:
```bash
node pokemon-showdown start --no-security
```

## Esecuzione

Con il server Showdown attivo e il virtualenv attivato:
```bash
python pokeRL_v1_7.py
```
Seed e `reward_mode` ("shaped" o "sparse") si impostano nel blocco `if __name__ == "__main__":` in fondo allo script. Ogni run salva un checkpoint ogni 200.000 step in `checkpoints/` e le curve di training in `tb_logs/` (visualizzabili con `tensorboard --logdir tb_logs`).
