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
# 1. Play against the trained agent
#    (the GUI loads a MaskablePPO policy, so it needs the RL stack too)
pip install -r requirements.txt -r requirements-rl.txt
python play_gui.py
```

In the setup screen each of the four seats can be **Human**, **AI** (the trained
model), or **FP** (a hand-crafted heuristic bot). Set one seat to AI and the rest to
FP to *watch* the trained agent play the benchmark bots.

```bash
# 2. Train your own agent
python -m training.train_selfplay --timesteps 15000000 --n-envs 32 --fp-prob 0.3

# 3. Run the tests
pip install -r requirements-dev.txt && pytest
```

The game **engine itself is pure standard library** — only the GUI (`pygame`) and the
policy (`torch` / `sb3-contrib`) add dependencies on top.

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
  cash, "completes my set", "denies an opponent", rent exposure, and more,
- and **the trade rules**: `build_offer()` constructs every proposal (which tile to
  hand over, how much cash), `accepts()` decides every offer. One implementation
  each, called by the training env and the GUI alike.

Both the training env and the GUI's AI ([`ui/ai_player.py`](ui/ai_player.py)) import
and use this same encoder, guaranteeing train/play parity. That guarantee is
enforced by a test ([`tests/test_trade_parity.py`](tests/test_trade_parity.py)) that
puts both paths on the same board and asserts they build the identical offer — it
was written because they *didn't*: a comment claiming the two were "byte-identical"
was wrong by 3× on a contested trade.

### Observation & action spaces

- **Observation:** a `265`-dim float vector — per-player cash/position/jail state,
  per-tile ownership/mortgage/development, per-group monopoly progress,
  completes-my-set / completes-an-opponent's-set flags, expected-profit-per-turn
  features, a landing-frequency prior, **the money the current decision is for**
  (the debt in a forced liquidation, the standing bid in an auction) and whether
  the player could even raise it, and **the clock** (how far into the game we are).
  Unbounded dollar features are log-compressed (`observation.squash`) so a
  late-game balance cannot drown out the rest of the board.
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
> `265 / 211`. Reward-coefficient and heuristic-only changes do not change the shape.

### Reward design

The reward ([`engine/rewards.py`](engine/rewards.py)) has a **shaped** and a
**sparse** mode:

- **Dense shaping** is potential-based on the agent's *relative* net-worth advantage
  (my net worth − mean opponent net worth): `γΦ(s') − Φ(s)`, with **`Φ = 0` at a real
  terminal**, so it telescopes away and what survives is the actual win. Two details
  earn their keep. The mean is taken over a **fixed** denominator (every opponent the
  game started with, a bankrupt one being worth exactly 0) — a survivors-only mean
  *jumps* when a player is eliminated, and knocking out the poorest opponent used to
  pay the agent a **negative** reward for a strictly good outcome. And zeroing `Φ` at
  the end is what stops the last step paying out the agent's whole net worth
  (a `+5..+12` spike that dwarfed the `±1` saying who won, training it to *end rich*
  rather than to *win*).
- On top of that sit one-time bonuses for acquiring tiles, building houses, being
  **first** to a set, and **denying** an opponent a set, plus a **solvency penalty**
  that keeps a rent-sized cash cushion so the agent does not spend itself into
  bankruptcy. It is charged **once per turn**, not per decision: the number of
  decisions in a turn is the agent's own choice, so a per-decision drag could be
  shrunk by simply doing less.
- **Terminal reward** ∈ `[-1, +1]`: bankruptcy `−1`, sole survivor `+1`, otherwise
  net-worth rank among the survivors. A **turn-cap timeout is a truncation, not a
  termination**: the game did not really end, so no final payoff is invented and the
  learner bootstraps `V(s')` instead.

Every tunable coefficient lives in one frozen dataclass,
[`engine/config.py`](engine/config.py)`::RewardConfig`, injectable into the env
(`MonopolyEnv(cfg=...)`) so it can be swept without touching globals — and written to
a `*.meta.json` sidecar beside every saved model, together with the git SHA, so a
checkpoint can always be tied back to the economics it learned under. Evaluation and
the GUI read that sidecar rather than applying today's defaults to an old model.

### Training: self-play against a strong benchmark

Training ([`training/train_selfplay.py`](training/train_selfplay.py)) is **MaskablePPO**
(from `sb3-contrib`) in a 4-player game, on a `256×256` ReLU MLP. Each episode samples
an opponent roster from an [`OpponentPool`](training/selfplay.py):

- frozen **snapshots of the agent itself** (self-play), drawn **independently per
  seat** — so the agent meets an old self and a recent self in the same game rather
  than three copies of one opponent;
- a trivial engine baseline, and
- the **FP-A/B/C heuristic trio** ([`training/baselines.py`](training/baselines.py)) —
  state-aware bots (modeled on Bonjour et al., 2021) that actually bid, build, and
  trade toward monopolies while keeping a cash buffer. They differ only in which
  colour groups they prioritize, and they are the **meaningful benchmark**; the
  trivial baseline is too weak to be diagnostic. When FP is drawn it is drawn as the
  *whole trio*, since that is the roster the agent is measured against.

The snapshot pool keeps a **spread across training history** rather than the newest N:
the earliest snapshot is a permanent anchor, and eviction thins the most crowded
stretch of the timeline. Keeping only recent snapshots leaves the agent playing
near-copies of its current self, which is how self-play cycles instead of improving.

TensorBoard gets win rate **split by opponent type** (`win_rate_vs_fp` /
`_vs_snapshot` / `_vs_baseline` — one blended number cannot tell you which bar you
cleared), plus `timeout_rate` and `illegal_rate` (masked-out actions the env had to
clamp; it should stay at 0).

---

## Repository layout

```
engine/            The game-as-an-RL-environment (the core)
  game.py          Pure Monopoly rules engine
  rl_env.py        Gymnasium env: decision-point stepping + action decode
  observation.py   ObsEncoder — single source of truth (obs, masks, valuations,
                   and the one trade builder / accept rule)
  rewards.py       Reward shaping (net-worth advantage + terminal + solvency)
  config.py        RewardConfig — every tunable coefficient in one dataclass,
                   plus the model metadata sidecar
  constants.py     Phase ids and the flat action-space layout (211 actions)

models/            Plain game objects: Board, Tile, Card, Deck, Player
data/              Static board + card-deck definitions, and board_visits.json
                   (the landing-frequency table the observation is built on)
tools/             Board generation helper
tests/             pytest suite: engine rules, mask legality, reward invariants,
                   opponent-pool eviction, and env↔GUI trade parity

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
# Run the tests (no PYTHONPATH needed -- see pyproject.toml)
pytest

# Play against the trained agent (GUI)
python play_gui.py

# Train from scratch (clear the snapshot pool first for a clean run)
rm -f runs/sp_pool/*.zip
python -m training.train_selfplay --timesteps 15000000 --n-envs 32 --fp-prob 0.3

# Sweep a reward coefficient (RewardConfig is injectable; nothing to monkeypatch)
python -m training.train_selfplay --solvency-penalty-coef 0.05 --save-path runs/sweep_005

# Evaluate vs the strong FP trio (the meaningful benchmark)
PYTHONPATH=. python -m validation.evaluate runs/monopoly_ppo --episodes 200 --opponent fp

# Evaluate vs the trivial baseline (add --stochastic to avoid phantom timeouts)
PYTHONPATH=. python -m validation.evaluate runs/monopoly_ppo --episodes 200 --stochastic

# Deeper strategic diagnostics (monopoly race, trade accept/reject, bid health)
PYTHONPATH=. python -m validation.simulate runs/monopoly_ppo --games 100

# Full pipeline: train -> evaluate(fp) -> simulate
python training_pipeline.py --n-envs 32 --episodes 200 --games 100
```

The deployed model is `runs/monopoly_ppo.zip` (`265`-dim obs, `211` actions), beside
its `runs/monopoly_ppo.meta.json` sidecar recording the `RewardConfig`, the CLI args
and the git SHA it was trained with. It loads and runs in both evaluation and the GUI.

---

## Design notes & gotchas

- **Trade valuation lives in exactly one place.** `ObsEncoder.build_offer()` and
  `ObsEncoder.accepts()`. There used to be four implementations (two offer builders,
  three accept rules) kept in step by comment discipline, and they had already
  drifted apart. Add trade logic there, not in the env or the UI.
- **Obs/action shape changes ⇒ from-scratch retrain.** All saved models are locked to
  `265 / 211`. The opponent pool checks the shape and skips stale snapshots.
- **The landing-frequency table is part of the observation.** `data/board_visits.json`
  scales every traffic-derived feature and valuation. It is tracked static data, and
  a missing table now fails loudly instead of silently substituting a uniform prior
  (which quietly changes what the policy sees).
- **Auction economics are a standing constraint.** Auction bidding (`_bid_value`) is
  deliberately kept at the plain sticker premium even though *trade* valuations prize
  monopolies far higher; the health metric `mean_set_completion_bid_ratio` (from
  `simulate.py`) should stay near retail. Do not "fix" auctions with the trade
  multiplier.
- **Analytics observe, they don't patch.** `simulate.py` used to monkey-patch the
  env's private methods to count trades and bankruptcies, so any refactor silently
  broke it. The engine now reports these (`Game.on_bankrupt`,
  `MonopolyEnv.on_trade_offer`); subscribe to a hook rather than overwriting a method.
- **Deterministic eval understates the trivial-baseline matchup** (argmax end-turn
  loops → phantom timeouts); use `--stochastic` there, but not against FP.
- **`simulate.py`'s "monopolies ever completed" is cumulative** (a set counts even if
  later traded away, and re-counts for each owner). "Monopolies held at game end" is
  the number you see on the board.

---

## Requirements

| File | Purpose | Key packages |
|---|---|---|
| `requirements.txt` | Run the GUI | `pygame` |
| `requirements-rl.txt` | Train / evaluate | `gymnasium`, `stable_baselines3`, `sb3_contrib`, `torch`, `numpy`, `matplotlib` |
| `requirements-dev.txt` | Development | `pytest`, `coverage`, `ruff` |

`torch` wheels are platform/CUDA-specific — if the pinned build does not match your
machine, install torch per <https://pytorch.org/get-started/> first, then
`pip install -r requirements-rl.txt` for the rest.
