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
- every economic **valuation** — tile value, trade value, bid value, rent exposure,
  "completes my set", and more,
- and **the trade rule**: `accepts()` decides every offer, for every seat, in
  training and in the GUI alike.

Group-level economics — what a monopoly is worth, and what a *share* of one is worth
— live next door in [`engine/valuation.py`](engine/valuation.py) (`SetValuer`,
exposed as `encoder.sets`), because the reward reads them too. The trade heuristic
itself is [`engine/trade.py`](engine/trade.py).

Both the training env and the GUI's AI ([`ui/ai_player.py`](ui/ai_player.py)) use
this same encoder and the same `TradeEngine`, guaranteeing train/play parity. It is
enforced by a test ([`tests/test_trade_parity.py`](tests/test_trade_parity.py)),
written because the two paths *had* drifted: a comment claiming they were
"byte-identical" was wrong by 3× on a contested trade.

### Trading is heuristic, not learned

The policy does not trade. Every offer is built and judged by
[`engine/trade.py`](engine/trade.py), from the valuations above.

This follows the **hybrid** architecture of [Bonjour et al.
2021](https://arxiv.org/abs/2103.00683) — a fixed heuristic for the rare,
valuation-heavy trade decisions and DRL for everything else — which beat their
pure-DRL agent **91.65% to 69.95%** (DDQN: 76.91% vs 47.41%). Their reasoning holds
here: trade decisions are too rare to learn a good valuation from, and a valuation is
something you can just write down. Ours is considerably richer than the paper's
(which is a plain net-worth difference with a 1.5→2.0 monopoly bonus).

It was also, empirically, not working. The learned responder accepted **~85% of every
offer put to it**, gave away completed sets for junk, moved **$58,747 of trade cash
per game** on a board where you start with $1,500, and in one measured game
ping-ponged a single tile **424 times**.

The engine builds **multi-tile packages** both ways (`trade_max_package`, default 3)
— it can ask for the two tiles it needs and pay in three it doesn't. The old 84-action
trade band could not name such an offer, which mattered: when a partner holds two of a
three-tile group, *no* one-for-one swap can ever complete the set.

### Observation & action spaces

- **Observation:** a `261`-dim float vector — per-player cash/position/jail state,
  per-tile ownership/mortgage/development, per-group monopoly progress,
  completes-my-set / completes-an-opponent's-set flags, expected-profit-per-turn
  features, a landing-frequency prior, **the money the current decision is for**
  (the debt in a forced liquidation, the standing bid in an auction) and whether
  the player could even raise it, and **the clock** (how far into the game we are).
  Unbounded dollar features are log-compressed (`observation.squash`) so a
  late-game balance cannot drown out the rest of the board.
- **Actions:** a flat `Discrete(125)` space split into phase bands
  ([`engine/constants.py`](engine/constants.py)): jail, buy/decline, manage
  (build / mortgage / unmortgage / end turn), liquidate, and auction bids. Illegal
  actions are masked every step, so the policy only ever chooses among legal moves.
- **There are no trade actions.** Trading used to own 86 of 211 ids — an 84-id
  proposal band plus accept/reject — and `PHASE_TRADE_RESPOND` was a decision phase.
  Both are gone.

> **Obs/action shape is locked.** Any change to the observation dimension or action
> count requires a from-scratch retrain — saved models are shape-locked to
> `261 / 125`. Reward-coefficient and heuristic-only changes do not change the shape.

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
- Net worth prices a set by **how good it is and how much of it you hold**: each group
  contributes `group price × strength × f(k/n)` (see the gotchas), so losing the orange
  set costs far more shaped reward than losing the utilities, and holding two of three
  oranges is worth real money. It used to count only groups owned *outright* — 2/3 of
  orange scored exactly zero, so the policy had no reason to protect a nearly-finished
  set and no way to learn why the trade heuristic protects one. A *complete* set scores
  exactly what it did before (`f(1) = 1`), so the reward scale is unchanged.
  `set_strength_reward_weight` scales the strength tilt — `0` makes every set flat, a
  clean ablation baseline.
- On top of that sit one-time bonuses for acquiring tiles, building houses, being
  **first** to a set, and **denying** an opponent a set (the last two also weighted by
  set strength), plus a **solvency penalty** that keeps a rent-sized cash cushion so
  the agent does not spend itself into bankruptcy. It is charged **once per turn**,
  not per decision: the number of decisions in a turn is the agent's own choice, so a
  per-decision drag could be shrunk by simply doing less.
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
  observation.py   ObsEncoder — single source of truth (obs, masks, tile
                   valuations, and the one accept rule)
  valuation.py     SetValuer — what a monopoly, and a share of one, is worth
                   (per-set strength, completion curve, game stage)
  trade.py         The trade heuristic: packages, bargaining, settlement.
                   Trading is NOT a policy decision — see the README section
  rewards.py       Reward shaping (net-worth advantage + terminal + solvency)
  config.py        RewardConfig — every tunable coefficient in one dataclass,
                   plus the model metadata sidecar
  constants.py     Phase ids and the flat action-space layout (125 actions)

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

game_logs/           Transcript of each GUI game played (git-ignored)
```

---

## Commands

```bash
# Run the tests (no PYTHONPATH needed -- see pyproject.toml)
pytest

# Play against the trained agent (GUI)
python play_gui.py

# Every GUI game writes a transcript to game_logs/ (one file per game, named by
# start time and seed, and saved even if you quit part-way).
python play_gui.py --log-dir ""      # ... unless you turn it off

# Train from scratch (clear the snapshot pool first for a clean run)
rm -f runs/sp_pool/*.zip
python -m training.train_selfplay --timesteps 15000000 --n-envs 32 --fp-prob 0.3

# Sweep a reward coefficient (RewardConfig is injectable; nothing to monkeypatch)
python -m training.train_selfplay --solvency-penalty-coef 0.05 --save-path runs/sweep_005

# The trade-economics knobs (see the trade gotchas below before touching these).
# set_progress_exponent and trade_denial_weight interact -- sweep them together.
python -m training.train_selfplay --trade-denial-weight 0.5 --trade-monopoly-mult 3.0 \
    --set-progress-exponent 2.5 --set-strength-clamp 0.15 2.0 \
    --set-quality-clamp 0.75 1.25 --set-strength-reward-weight 1.0 \
    --stage-inflation-weight 0.5 --stage-inflation-cap 2.0 --trade-max-package 3

# Evaluate vs the strong FP trio (the meaningful benchmark)
PYTHONPATH=. python -m validation.evaluate runs/monopoly_ppo --episodes 200 --opponent fp

# Evaluate vs the trivial baseline (add --stochastic to avoid phantom timeouts)
PYTHONPATH=. python -m validation.evaluate runs/monopoly_ppo --episodes 200 --stochastic

# Deeper strategic diagnostics (monopoly race, trade accept/reject, bid health)
PYTHONPATH=. python -m validation.simulate runs/monopoly_ppo --games 100

# Full pipeline: train -> evaluate(fp) -> simulate
python training_pipeline.py --n-envs 32 --episodes 200 --games 100
```

The deployed model is `runs/monopoly_ppo.zip` (`261`-dim obs, `125` actions), beside
its `runs/monopoly_ppo.meta.json` sidecar recording the `RewardConfig`, the CLI args
and the git SHA it was trained with. It loads and runs in both evaluation and the GUI.

---

## Design notes & gotchas

- **Trade logic lives in exactly one place.** `engine/trade.py` builds every offer,
  `ObsEncoder.accepts()` judges every offer, `engine/valuation.py` prices every set.
  There used to be four implementations (two offer builders, three accept rules) kept
  in step by comment discipline, and they had already drifted. Add trade logic there,
  not in the env or the UI.
- **Set value is priced per *group*, not per tile — it is not additive.** Two tiles of
  one group are worth more together than apart, and giving an orange away while taking
  another orange is a wash. `SetValuer.swap_delta()` compares each affected group's
  position before and after the *whole* package; summing per-tile marginals
  double-counts in both directions. Measured while it did: the engine paid **$2,000
  for a swap that left both sides exactly where they started** — the "exchanges for no
  apparent reason" symptom, which turned out to be arithmetic, not strategy.
- **Blocking must be worth less than completing, or nothing can trade.**
  `trade_denial_weight` is the single most load-bearing trade knob. At `1.0` a tile is
  worth the same to the player blocking with it as to the player who needs it, every
  swap is **zero-sum**, there are no gains from trade, and `accepts()` correctly
  refuses essentially every offer: monopolies stop forming and games drag to the turn
  cap. It sits at `0.5` so a bargaining range exists. Pinned by
  `test_zero_sum_denial_kills_every_deal`, because the failure is *silent* — nothing
  errors, sets just quietly stop forming. Split from `denial_value_weight`, which
  governs auction blocking only.
- **…which is exactly why the no-gift rule cannot be left to the valuation.** The same
  discount that makes trade possible means handing over a set-completer *costs the
  giver a fraction of what it hands the taker* — pure joint surplus, and a greedy
  package builder hunts precisely for that. Measured without the rule: agents handed
  each other **completed sets to buy single tiles**, 47 trades a game. So
  `TradeEngine._may_hand_over` bars it outright, unless the deal completes a set for us
  too and the set we gain is at least as strong as the one we give (a genuine
  set-for-set). The discount is right for *pricing*; it is not a licence to arm your
  opponent. Tiles from the group you are buying into are excluded from the payment for
  the same class of reason — they are a wash, so they sorted to the *top* of the
  surplus order and crowded out the junk that would actually have paid.
- **Sets are not equal, and cost is half of why.** `SetValuer.strength` blends
  *traffic × full-development rent* with *that rent per dollar of capital the set
  needs*, each normalised to a mean of ~1.0 and clamped (`set_quality_clamp` on the
  cost tilt, `set_strength_clamp` on the blend) — so it **redistributes**
  `trade_monopoly_mult` across sets rather than changing the overall scale. Landing at
  **orange 1.73 > red 1.49 > yellow 1.37 > dark_blue 1.28 > green 1.21 > pink 1.05 >
  railroad 0.83 > light_blue 0.82 > brown/utility at the 0.15 floor**. The cost term is
  load-bearing: on earning power alone **green outranks orange**, because green has the
  highest hotel rent on the board — and costs $3,000 to develop, with the worst payback
  of any street set. Getting that backwards is a Monopoly-strategy error, not a
  rounding one. One definition, shared by trades and the reward.
- **The completion curve is what killed junk-for-monopoly.** Set value scales by
  `f(k, n) = (k/n) ** set_progress_exponent`, so every share of a group is worth
  something. It replaced a step function that paid a set premium **only** at exactly
  one-tile-from-complete: a player holding 2/3 of orange priced St. James at its **$227
  sticker — the same as with 1/3 of the set — and would sell it for a brown and $200**,
  while the identical tile was worth **$4,937** once the set was whole. A 21.7× cliff
  with nothing underneath it, which is precisely how a human traded junk to the AI for
  monopolies. It is now a ~2× step. Pinned by `TestNoCliff`.
- **Obs/action shape changes ⇒ from-scratch retrain.** All saved models are locked to
  `261 / 125`. The opponent pool checks the shape and skips stale snapshots.
- **The landing-frequency table is part of the observation.** `data/board_visits.json`
  scales every traffic-derived feature and valuation. It is tracked static data, and
  a missing table now fails loudly instead of silently substituting a uniform prior
  (which quietly changes what the policy sees).
- **Auction economics are a standing constraint.** Auction bidding (`_bid_value`) is
  deliberately kept at the plain sticker premium even though *trade* valuations prize
  monopolies far higher; the health metric `mean_set_completion_bid_ratio` (from
  `simulate.py`) should stay near retail. Do not "fix" auctions with the trade
  multiplier. This is why the denial weight is **two knobs**: `denial_value_weight`
  (auctions, `1.0`) and `trade_denial_weight` (trades, `0.5`). Set strength and the
  completion curve are likewise trade/reward-only and deliberately not applied to
  `_bid_value`.
- **The stage multiplier is trade-only, on purpose.** Set values scale with the
  board's cash inflation (`stage()`: mean live balance vs. the $1,500 start, capped) —
  everyone laps Go, so a fixed set price gets cheap in real terms and the agent would
  sell a monopoly for what was a fortune on turn 5. But the **reward must not** do
  this: a cash-inflating term in `Φ` makes the shaping potential drift upward and pays
  the agent for making everyone richer. Trades price against today's board; net worth
  is absolute. Pinned by `test_the_reward_is_stage_free`.
  There is a real tension here worth knowing about: cash inflation argues set prices
  should *rise* late, while a shrinking rent horizon argues their EV *falls*. Only
  inflation is modelled. If late-game trading stalls, `stage_inflation_cap` is the knob.
- **Trade health is measurable without a model**, since trading is pure heuristic —
  which makes it cheap to check before spending compute on a retrain. A healthy run
  (25 games, random legal actions elsewhere): **~4 accepted trades/game, one tile
  changing hands ≤ ~5 times, ~$1.8k of trade cash/game, zero self-set-breaks, zero
  monopoly gifts.** Compare the old code: 85% accept rate, 424 trades of one tile,
  $58,747/game. Note the accept rate is ~100% by construction now and means nothing —
  the engine only proposes deals it has already checked will clear.
- **Analytics observe, they don't patch.** `simulate.py` used to monkey-patch the
  env's private methods to count trades and bankruptcies, so any refactor silently
  broke it. The engine now reports these (`Game.on_bankrupt`,
  `MonopolyEnv.on_trade_offer`); subscribe to a hook rather than overwriting a method.
- **Deterministic eval understates the trivial-baseline matchup** (argmax end-turn
  loops → phantom timeouts); use `--stochastic` there, but not against FP.
- **`simulate.py`'s "monopolies ever completed" is cumulative** (a set counts even if
  later traded away, and re-counts for each owner). "Monopolies held at game end" is
  the number you see on the board. The gap between the two is the **churn metric**: a
  winner completing 6.8 sets but holding 3.65 means half of them were traded away
  again. Healthy is ever ≈ held (currently 2.00 vs 1.97).
- **Game length and first-monopoly tempo are noise-dominated below ~100 games.**
  Seed choice swamps them: the *same* config measured 416 turns on seeds 0–29 and 256
  turns on seeds 100–125, and a `trade_monopoly_mult` sweep at n=12 ranked 1.5 worse
  than both 1.0 and 2.0. Do not tune against these at small n — it is how you end up
  chasing a coefficient that was never the cause. The trade-volume metrics (hundreds
  of samples per game) are stable and can be trusted much sooner.

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
