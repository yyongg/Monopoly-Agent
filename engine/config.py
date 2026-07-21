"""Tunable reward / valuation knobs, gathered into one :class:`RewardConfig`.

Every coefficient that shapes how the agent values money, tiles, sets, and
acquisitions lives here -- previously scattered as ~20 module-level constants in
``rl_env`` plus a few class attributes on the env and the GUI decider. Collecting
them into one dataclass means:

* the UI (and any tool) imports *one* object instead of a growing tuple,
* hyperparameter sweeps construct ``RewardConfig(build_bonus_coef=...)`` instead
  of monkeypatching module globals,
* the exact knobs a model was trained with can be serialised into its checkpoint
  metadata.

The module-level constants below are kept (equal to the dataclass defaults) so
existing ``from engine.rl_env import TRADE_INCOME_WEIGHT`` style imports keep
working; new code should read fields off a ``RewardConfig`` instance.
"""

import json
import os
import subprocess
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class RewardConfig:
    """All reward-shaping and valuation coefficients for one training run."""

    # -- Net-worth valuation (reward) --------------------------------------
    # An owned, unmortgaged property is booked at ``price * (1 +
    # acquisition_premium)`` so *buying* is a small net gain rather than
    # net-worth-neutral; without it the policy collapses to "always decline".
    acquisition_premium: float = 0.5
    # Each fully-owned colour group adds ``monopoly_bonus * group list price`` to
    # net worth, so the tile that finishes a set is a big net-worth jump.
    monopoly_bonus: float = 1.0

    # -- Trade / bid valuation ---------------------------------------------
    # Weight on the blocking (deny-opponent) term in *auction* bidding
    # (``_bid_value``). Kept at 1.0: auction economics are a standing constraint.
    denial_value_weight: float = 1.0
    # Weight on the blocking term in *trade* valuation (``_trade_value``), split
    # from the auction weight above so it can be tuned without touching auctions.
    #
    # This knob is what makes trade possible at all. At 1.0 a tile is worth
    # exactly as much to the holder (who blocks) as to the acquirer (who
    # progresses), so the swap is **zero-sum** and there are no gains from trade --
    # ``accepts()`` then correctly refuses essentially every offer, monopolies
    # stop forming, and games drag to the turn cap. Below 1.0 blocking is worth
    # less than completing, which opens a bargaining range: a deal clears at a
    # price both sides gain from, while ``accepts()`` still refuses a giveaway.
    trade_denial_weight: float = 0.5
    # Turns of expected rent folded into a tile's trade value (list price +
    # trade_income_weight * landing-traffic * nominal rent).
    trade_income_weight: float = 3.0
    # A completed monopoly earns *far* more over a game than its group sticker
    # price, so in *trade* valuations (only -- auction bidding via _bid_value is
    # deliberately left at the sticker premium to keep auction economics near
    # retail) a set is priced at this multiple of its base ``monopoly_bonus *
    # group_price`` value. The dollar scale of the whole set-value model
    # (``engine.valuation.SetValuer.monopoly_value``); the per-set tilt on top of
    # it is ``set_strength_clamp`` / ``set_quality_clamp`` below.
    trade_monopoly_mult: float = 3.0
    # Largest package (tiles per side) the heuristic trade engine will assemble.
    # The engine's ``execute_trade`` takes arbitrary lists both ways; this only
    # bounds the greedy search in ``engine.trade``. A human can still stack more.
    trade_max_package: int = 3
    # Extra weight, in trade valuations, on completing/denying a set that would
    # be the *game's first* monopoly (none owned by anyone yet): the set value is
    # scaled by ``1 + trade_first_monopoly_weight``. Being first to a set (or
    # denying the opponent that) is a decisive tempo edge, so the agent should
    # both refuse to sell into it and pay up to secure it.
    trade_first_monopoly_weight: float = 1.0

    # -- One-time shaped bonuses -------------------------------------------
    denial_bonus_coef: float = 0.5     # reward for taking an opponent's last tile,
    #                                    as a fraction of the denied set's value
    # -- Monopoly-race shaping (being FIRST to a set, and denying that) ----
    # One-time bonus when the agent completes the *game's first* monopoly (no
    # set owned by anyone yet), as a fraction of the completed set's value.
    first_monopoly_bonus_coef: float = 0.5
    # Early-completion multiplier on that bonus: 1 + tempo_weight * tempo, where
    # tempo decays 1 -> 0 over ``first_monopoly_tempo_turns`` turns. Rewards a
    # fast race to the first set, not just eventually completing one.
    first_monopoly_tempo_weight: float = 1.5
    first_monopoly_tempo_turns: float = 50.0
    # Extra weight on the denial bonus when the tile taken would have completed
    # the *game's first* monopoly (no set owned by anyone yet): mult = 1 + weight.
    first_denial_weight: float = 0.5
    # Reward for acquiring an unowned tile from the bank, scaled by expected
    # income. Buying on landing earns the full coef; an auction win earns the
    # smaller auction coef. See the extended note that used to sit in rl_env.
    acquisition_bonus_coef: float = 3.0
    auction_acquisition_bonus_coef: float = 1.0
    # Price-scaled preference for buying on landing (never on an auction win),
    # sized (c >= 1.0) to beat the decline-then-snipe-the-auction exploit.
    buy_preference_coef: float = 1.30
    # Reward for placing a house/hotel, scaled by the extra rent it adds (rent
    # jump * landing traffic), tilted toward cheap-to-develop groups.
    build_bonus_coef: float = 0.5

    # -- Solvency / liquidity ----------------------------------------------
    # Net worth prices a dollar of cash exactly like a dollar of property, and
    # acquisition_premium makes converting cash into assets a net gain -- so
    # nothing counters the agent spending itself broke. These two knobs add a
    # per-step penalty for letting cash fall below a cushion sized to the
    # board's live rent threat (expected rent outflow per board round, from
    # opponent-owned developed tiles). The penalty is 0 above the cushion and
    # rises linearly to solvency_penalty_coef at zero cash, making liquidity
    # itself valuable and directly countering the self-bankruptcy failure mode.
    # solvency_penalty_coef is the primary tuning knob: too small and the agent
    # still plays broke; too large and it hoards cash instead of developing.
    solvency_cushion_turns: float = 3.0
    solvency_penalty_coef: float = 0.02

    # -- Cost-normalised strategy signals ----------------------------------
    # House cost at which the build tilt (ref / house_cost) is 1.0 -- the groups'
    # harmonic-mean house cost, so the *average* build bonus is unchanged and
    # only its distribution across colours shifts.
    build_roi_ref_house_cost: float = 95.0

    # -- Per-set strength: value a monopoly by its real earning power ------
    # See engine.valuation.SetValuer. A per-group multiplier blending two terms,
    # each normalised by its own mean across the 10 groups so the *average* group
    # scores ~1.0 -- it redistributes trade_monopoly_mult across sets rather than
    # changing the overall scale:
    #
    #   power(G)      = sum over the group of traffic x full-development rent
    #   efficiency(G) = power(G) / (group price + cost of a hotel on every tile)
    #   strength(G)   = clamp(norm(power) * clamp(norm(efficiency), *set_quality_clamp))
    #
    # The efficiency term is what makes this match real play. Ranking on power
    # alone put **green above orange** -- green has the highest hotel rent but
    # costs $3,000 to develop and has the worst payback of any street set. The
    # cost tilt flips that: orange 1.73 > red 1.49 > yellow > dark_blue >
    # green 1.21 > pink > railroad > light_blue > brown/utility (both at the
    # floor). Utilities score 0.04 raw -- a 2-tile $300 "monopoly" is not a
    # monopoly, and pricing it like the orange set is what had the agents
    # ping-ponging sets for cash.
    set_quality_clamp: tuple = (0.75, 1.25)   # clamp on the efficiency tilt
    set_strength_clamp: tuple = (0.15, 2.0)   # clamp on the blended strength
    # How strongly the *reward's* set net worth (and the one-time first-monopoly
    # / denial bonuses) tilt by set strength: effective strength is
    # 1 + w*(strength - 1). w=0 recovers flat behaviour exactly (a safe ablation
    # baseline); w=1 is full strength. Trade valuation always uses full strength;
    # this knob only governs the reward shaping.
    set_strength_reward_weight: float = 1.0
    # How set value grows with how much of the group you hold:
    # ``f(k, n) = (k/n) ** set_progress_exponent``, so a group you own k of n of
    # is worth ``monopoly_value(G) * f(k, n)``.
    #
    # This exponent is the fix for the junk-for-monopoly exploit. The old
    # valuation gave a set premium **only** when a tile was exactly one away from
    # completing, so a player 2/3 of the way to orange priced St. James at
    # sticker ($227) and would sell it for a brown + $200 -- while the same tile
    # was worth $4,937 once the set was complete. A 21.7x cliff with nothing in
    # between. The curve replaces that step with a gradient (now ~2x): >1 keeps
    # completion the biggest single jump, while every rung below it still costs
    # real money. 1.0 would price each tile of a set identically; very high
    # values re-create the cliff.
    set_progress_exponent: float = 2.5

    # -- Game stage: cash inflates, so property costs more cash later ------
    # Every player collects $200 a lap, so late-game balances dwarf the $1,500
    # start and a flat set price gets cheap in real terms. Trade set values scale
    # by ``clamp(1 + w * (mean live balance / 1500 - 1), 1.0, cap)``: $1,500 ->
    # 1.00, $2,600 -> 1.37, $4,800+ -> 2.00 at the defaults.
    #
    # Measured cash on the board, not the turn count: it is what "inflation"
    # actually means here, and it self-calibrates (a poor, stalled game does not
    # inflate). Deliberately **not** applied to the reward -- see
    # engine.rewards._set_net_worth.
    stage_inflation_weight: float = 0.5
    stage_inflation_cap: float = 2.0

    # -- Observation scaling -----------------------------------------------
    # Divides the expected-profit-per-turn dollar figure into feature range.
    profit_scale: float = 1000.0


# Default instance whose fields back the module-level constants below.
DEFAULT_REWARD_CONFIG = RewardConfig()

# Backward-compatible module-level names (equal to the defaults). Re-exported
# from engine.rl_env so existing imports keep resolving.
ACQUISITION_PREMIUM = DEFAULT_REWARD_CONFIG.acquisition_premium
MONOPOLY_BONUS = DEFAULT_REWARD_CONFIG.monopoly_bonus
DENIAL_VALUE_WEIGHT = DEFAULT_REWARD_CONFIG.denial_value_weight
TRADE_DENIAL_WEIGHT = DEFAULT_REWARD_CONFIG.trade_denial_weight
TRADE_INCOME_WEIGHT = DEFAULT_REWARD_CONFIG.trade_income_weight
DENIAL_BONUS_COEF = DEFAULT_REWARD_CONFIG.denial_bonus_coef
FIRST_MONOPOLY_BONUS_COEF = DEFAULT_REWARD_CONFIG.first_monopoly_bonus_coef
FIRST_MONOPOLY_TEMPO_WEIGHT = DEFAULT_REWARD_CONFIG.first_monopoly_tempo_weight
FIRST_MONOPOLY_TEMPO_TURNS = DEFAULT_REWARD_CONFIG.first_monopoly_tempo_turns
FIRST_DENIAL_WEIGHT = DEFAULT_REWARD_CONFIG.first_denial_weight
ACQUISITION_BONUS_COEF = DEFAULT_REWARD_CONFIG.acquisition_bonus_coef
AUCTION_ACQUISITION_BONUS_COEF = DEFAULT_REWARD_CONFIG.auction_acquisition_bonus_coef
BUY_PREFERENCE_COEF = DEFAULT_REWARD_CONFIG.buy_preference_coef
BUILD_BONUS_COEF = DEFAULT_REWARD_CONFIG.build_bonus_coef
BUILD_ROI_REF_HOUSE_COST = DEFAULT_REWARD_CONFIG.build_roi_ref_house_cost
SET_QUALITY_CLAMP = DEFAULT_REWARD_CONFIG.set_quality_clamp
SET_STRENGTH_CLAMP = DEFAULT_REWARD_CONFIG.set_strength_clamp
SET_STRENGTH_REWARD_WEIGHT = DEFAULT_REWARD_CONFIG.set_strength_reward_weight
SET_PROGRESS_EXPONENT = DEFAULT_REWARD_CONFIG.set_progress_exponent
STAGE_INFLATION_WEIGHT = DEFAULT_REWARD_CONFIG.stage_inflation_weight
STAGE_INFLATION_CAP = DEFAULT_REWARD_CONFIG.stage_inflation_cap
TRADE_MAX_PACKAGE = DEFAULT_REWARD_CONFIG.trade_max_package
PROFIT_SCALE = DEFAULT_REWARD_CONFIG.profit_scale


# --- Run metadata: tying a saved model to the knobs it was trained with -----
# ``model.save(path)`` writes only the weights, so a checkpoint used to carry no
# record of the coefficients (or the code) behind it -- and these defaults have
# churned across commits. The sidecar below travels with the zip so evaluation
# and the GUI can rebuild the *training* economics instead of silently applying
# today's defaults to yesterday's model.

def config_sidecar_path(model_path):
    """The metadata file that pairs with a saved model zip."""
    base = model_path[:-4] if model_path.endswith(".zip") else model_path
    return base + ".meta.json"


def _git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5,
                             cwd=os.path.dirname(os.path.abspath(__file__)))
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def save_run_metadata(model_path, cfg, args=None):
    """Writes ``<model>.meta.json``: the reward config, the CLI args, the git SHA."""
    meta = {
        "reward_config": asdict(cfg),
        "git_sha": _git_sha(),
        "args": vars(args) if args is not None else None,
    }
    path = config_sidecar_path(model_path)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return path


def load_run_config(model_path):
    """The :class:`RewardConfig` a model was trained with, or the current
    defaults when the model predates the sidecar (with no way to know better)."""
    try:
        with open(config_sidecar_path(model_path)) as f:
            saved = json.load(f)["reward_config"]
    except (OSError, ValueError, KeyError):
        return RewardConfig()
    known = {f.name: f.type for f in fields(RewardConfig)}
    kwargs = {}
    for name, value in saved.items():
        if name not in known:
            continue  # a coefficient this version of the code no longer has
        kwargs[name] = tuple(value) if isinstance(value, list) else value
    return RewardConfig(**kwargs)
