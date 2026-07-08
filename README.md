# Monopoly RL

Train a Monopoly-playing AI **from scratch** with self-play reinforcement learning,
then sit down and play against it in a pygame GUI.

The agent learns the entire game — buying, auctions, building, trading, jail, and
cash management — with **no hand-written strategy**. Every decision comes from a
policy trained purely on the reward signal of winning games.

The two halves of the project are deliberately coupled by one engineering rule:
**the environment the agent trains in and the environment you play it in are the
same game, with the same valuations.** A single shared encoder
([`engine/observation.py`](engine/observation.py)) computes the observation, the
legal moves, and every economic valuation for *both* the training env and the GUI —
so the agent you trained is exactly the agent you play against.

---

## Quick start

Requires **Python 3.12**.

```bash
# 1. Play against the trained agent (only pygame needed)
pip install -r requirements.txt
python play_gui.py
```

In the setup screen each of the four seats can be **Human**, **AI** (the trained
model), or **FP** (a hand-crafted heuristic bot). Set one seat to AI and the rest to
FP to *watch* the trained agent play the benchmark bots.

```bash
# 2. Train your own agent (adds the RL stack: gymnasium, stable-baselines3, torch)
pip install -r requirements-rl.txt
python -m training.train_selfplay --timesteps 15000000 --n-envs 32 --fp-prob 0.3
```

The game **engine itself is pure standard library**; `pygame` is needed only for
the GUI, and the RL dependencies only for training.

---

## How it works

### The engine is a reinforcement-learning environment

[`engine/game.py`](engine/game.py) is a complete, rules-accurate Monopoly engine
(turns, rent, auctions, trades, bankruptcy). [`engine/rl_env.py`](engine/rl_env.py)
wraps it as a Gymnasium environment that steps at **decision points** and decodes a
flat action into an engine move.

### One source of truth: `ObsEncoder`

[`engine/observation.py`](engine/observation.py) is the heart of the project. Bound
to a live game, it is the single place that computes:

- the **observation vector** the policy sees,
- the **legal-action mask** for each decision phase,
- every economic **valuation** — tile value, trade value, bid value, balancing
  cash, "completes my set", "denies an opponent", rent exposure, and more.

Both the training env and the GUI's AI ([`ui/ai_player.py`](ui/ai_player.py)) import
and use this same encoder, guaranteeing train/play parity.

### Observation & action spaces

- **Observation:** a `262`-dim float vector — per-player cash/position/jail state,
  per-tile ownership/mortgage/development, per-group monopoly progress,
  completes-my-set / completes-an-opponent's-set flags, expected-profit-per-turn
  features, and a landing-frequency prior.
- **Actions:** a flat `Discrete(211)` space split into phase bands
  ([`engine/constants.py`](engine/constants.py)): jail, buy/decline, manage
  (build / mortgage / unmortgage / **trade proposals** / end turn), liquidate,
  auction bids, and trade accept/reject. Illegal actions are masked every step, so
  the policy only ever chooses among legal moves.
- Trade proposals use a **target + cash-tier** scheme: the agent picks which tile to
  acquire and a cash tier (0.75× / 1.0× / 1.25× the engine's balancing amount); the
  engine chooses the give-tile and computes the balancing cash.

> **Obs/action shape is locked.** Any change to the observation dimension or action
> count requires a from-scratch retrain — saved models are shape-locked to
> `262 / 211`. Reward-coefficient and heuristic-only changes do not change the shape.

### Reward design

The reward ([`engine/rewards.py`](engine/rewards.py)) has a **shaped** and a
**sparse** mode:

- **Dense shaping** is potential-based on the agent's *relative* net-worth advantage
  (my net worth − mean opponent net worth), so it telescopes and leaves the terminal
  reward undistorted. On top of that sit one-time bonuses for acquiring tiles,
  building houses, being **first** to a set, and **denying** an opponent a set, plus
  a per-step **solvency penalty** that keeps a rent-sized cash cushion so the agent
  does not spend itself into bankruptcy.
- **Terminal reward** ∈ `[-1, +1]`: bankruptcy `−1`, sole survivor `+1`, otherwise
  net-worth rank among the survivors.

Every tunable coefficient lives in one frozen dataclass,
[`engine/config.py`](engine/config.py)`::RewardConfig`, so a model's exact training
knobs can be serialized and swept without touching globals.

### Training: self-play against a strong benchmark

Training ([`training/train_selfplay.py`](training/train_selfplay.py)) is **MaskablePPO**
(from `sb3-contrib`) in a 4-player game. Each episode samples opponents from an
[`OpponentPool`](training/selfplay.py):

- frozen **snapshots of the agent itself** (self-play),
- a trivial engine baseline, and
- the **FP-A/B/C heuristic trio** ([`training/baselines.py`](training/baselines.py)) —
  state-aware bots (modeled on Bonjour et al., 2021) that actually bid, build, and
  trade toward monopolies while keeping a cash buffer. They differ only in which
  colour groups they prioritize, and they are the **meaningful benchmark**; the
  trivial baseline is too weak to be diagnostic.

---

## Repository layout

```
engine/            The game-as-an-RL-environment (the core)
  game.py          Pure Monopoly rules engine
  rl_env.py        Gymnasium env: decision-point stepping + action decode
  observation.py   ObsEncoder — single source of truth (obs, masks, valuations)
  rewards.py       Reward shaping (net-worth advantage + terminal + solvency)
  config.py        RewardConfig — every tunable coefficient in one dataclass
  constants.py     Phase ids and the flat action-space layout (211 actions)

models/            Plain game objects: Board, Tile, Card, Deck, Player
data/              Static board + card-deck definitions
tools/             Board generation helper

training/
  train_selfplay.py  Main entry: MaskablePPO self-play loop + CLI
  selfplay.py        OpponentPool (self snapshots, baseline, FP bots)
  baselines.py       FP-A/B/C hand-crafted heuristic opponents

validation/
  evaluate.py        Play a model N games vs a chosen opponent; win-rate dashboard
  simulate.py        Deeper strategic diagnostics (monopoly race, trades, bids)
  board_visits.py    Landing-frequency prior used as an observation feature

ui/
  app.py             The pygame GUI (human plays here)
  ai_player.py       Drives AI/FP seats using the SAME ObsEncoder
  board_layout.py    Board rendering geometry

training_pipeline.py Chains train -> evaluate -> simulate in one command
play_gui.py          Launch the human-vs-AI GUI
```

---

## Commands

```bash
# Play against the trained agent (GUI)
python play_gui.py

# Train from scratch (clear the snapshot pool first for a clean run)
rm -f runs/sp_pool/*.zip
python -m training.train_selfplay --timesteps 15000000 --n-envs 32 --fp-prob 0.3

# Evaluate vs the strong FP trio (the meaningful benchmark)
PYTHONPATH=. python -m validation.evaluate runs/monopoly_ppo --episodes 200 --opponent fp

# Evaluate vs the trivial baseline (add --stochastic to avoid phantom timeouts)
PYTHONPATH=. python -m validation.evaluate runs/monopoly_ppo --episodes 200 --stochastic

# Deeper strategic diagnostics (monopoly race, trade accept/reject, bid health)
PYTHONPATH=. python -m validation.simulate runs/monopoly_ppo --games 100

# Full pipeline: train -> evaluate(fp) -> simulate
python training_pipeline.py --n-envs 32 --episodes 200 --games 100
```

The deployed model is `runs/monopoly_ppo.zip` (`262`-dim obs, `211` actions). It
loads and runs in both evaluation and the GUI.

---

## Design notes & gotchas

- **Two trade-valuation paths must stay in sync.** Trades are valued both by
  `ObsEncoder._trade_value` (env / GUI-proposed) and by `GUIAIDecider.evaluate_trade`
  (human-proposes-to-AI). When you change how trades are valued, change both.
- **Obs/action shape changes ⇒ from-scratch retrain.** All saved models are locked to
  `262 / 211`.
- **Auction economics are a standing constraint.** Auction bidding (`_bid_value`) is
  deliberately kept at the plain sticker premium even though *trade* valuations prize
  monopolies far higher; the health metric `mean_set_completion_bid_ratio` (from
  `simulate.py`) should stay near retail. Do not "fix" auctions with the trade
  multiplier.
- **Deterministic eval understates the trivial-baseline matchup** (argmax end-turn
  loops → phantom timeouts); use `--stochastic` there, but not against FP.

---

## Requirements

| File | Purpose | Key packages |
|---|---|---|
| `requirements.txt` | Run the GUI | `pygame` |
| `requirements-rl.txt` | Train / evaluate | `gymnasium`, `stable_baselines3`, `sb3_contrib`, `torch`, `numpy` |
| `requirements-dev.txt` | Development | `pytest`, `coverage`, `ruff` |

`torch` wheels are platform/CUDA-specific — if the pinned build does not match your
machine, install torch per <https://pytorch.org/get-started/> first, then
`pip install -r requirements-rl.txt` for the rest.
