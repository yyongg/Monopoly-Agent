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

from dataclasses import dataclass


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
    denial_value_weight: float = 1.0   # weight on the blocking (deny-opponent) term
    # Turns of expected rent folded into a tile's trade value (list price +
    # trade_income_weight * landing-traffic * nominal rent).
    trade_income_weight: float = 3.0
    # A completed monopoly earns *far* more over a game than its group sticker
    # price, so in *trade* valuations (only -- auction bidding via _bid_value is
    # deliberately left at the sticker premium to keep auction economics near
    # retail) we prize completing/denying a set at this multiple of its base
    # ``monopoly_bonus * group_price`` value. Stops opponents cheaply buying a
    # set-completer out of the agent for a small cash premium.
    trade_monopoly_mult: float = 3.0
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
    # Set-completion tilt: avg group dev-ROI / set_roi_ref, clamped so it nudges
    # rather than dominates the sticker/traffic value in a trade.
    set_roi_ref: float = 1.3
    set_quality_clamp: tuple = (0.75, 1.25)

    # -- Observation scaling -----------------------------------------------
    # Divides the expected-profit-per-turn dollar figure into feature range.
    profit_scale: float = 1000.0

    # -- GUI trade heuristic (ui/ai_player.GUIAIDecider) -------------------
    # Value of a freshly completed monopoly beyond its tiles' list price, as a
    # multiple of the group price, when judging a trade.
    set_bonus: float = 1.0
    # Extra value the AI places on a tile it already owns when deciding to part
    # with it (list price * (1 + keep_premium)), so it demands a real premium.
    keep_premium: float = 0.5


# Default instance whose fields back the module-level constants below.
DEFAULT_REWARD_CONFIG = RewardConfig()

# Backward-compatible module-level names (equal to the defaults). Re-exported
# from engine.rl_env so existing imports keep resolving.
ACQUISITION_PREMIUM = DEFAULT_REWARD_CONFIG.acquisition_premium
MONOPOLY_BONUS = DEFAULT_REWARD_CONFIG.monopoly_bonus
DENIAL_VALUE_WEIGHT = DEFAULT_REWARD_CONFIG.denial_value_weight
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
SET_ROI_REF = DEFAULT_REWARD_CONFIG.set_roi_ref
SET_QUALITY_CLAMP = DEFAULT_REWARD_CONFIG.set_quality_clamp
PROFIT_SCALE = DEFAULT_REWARD_CONFIG.profit_scale
SET_BONUS = DEFAULT_REWARD_CONFIG.set_bonus
KEEP_PREMIUM = DEFAULT_REWARD_CONFIG.keep_premium
