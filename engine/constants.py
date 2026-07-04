"""Structural constants for the RL interface: decision phases, the flat action
id layout, and the auction bid grid.

These are pure identifiers (no tunable behaviour -- see :mod:`engine.config` for
the reward/valuation knobs). They are re-exported from :mod:`engine.rl_env` so
existing ``from engine.rl_env import PHASE_BUY, A_TRADE, ...`` imports keep
working.
"""

# --- Decision phases -------------------------------------------------------
PHASE_JAIL = 0
PHASE_BUY = 1
PHASE_MANAGE = 2
PHASE_LIQUIDATE = 3
PHASE_TERMINAL = 4
PHASE_AUCTION = 5        # submit a sealed bid for a property up for auction
PHASE_TRADE_RESPOND = 6  # accept or reject a trade offered by another player
NUM_PHASES = 7

# --- Flat action ids -------------------------------------------------------
A_PAY_JAIL = 0
A_USE_CARD = 1
A_ROLL_JAIL = 2
A_BUY = 3
A_DECLINE = 4
A_END_MANAGE = 5
A_BUILD = 6          # A_BUILD + i, for ownable tile i (0..27)
A_SELL = 34          # A_SELL + i
A_MORTGAGE = 62      # A_MORTGAGE + i
A_UNMORTGAGE = 90    # A_UNMORTGAGE + i
NUM_OWNABLE = 28
NUM_GROUPS = 10      # monopoly groups: 8 street colors + railroads + utilities
# Trade proposals: ``A_TRADE + i * NUM_TRADE_TIERS + tier`` proposes acquiring
# ownable tile ``i`` at a cash tier. The engine computes a balancing cash figure
# (``ObsEncoder._balancing_cash``); the tier scales it by ``TRADE_CASH_TIERS`` so
# the agent controls how sweet the offer is -- lowball, fair, or generous -- the
# way the paper's discretised buy/sell trades did.
A_TRADE = 118
NUM_TRADE_TIERS = 3
TRADE_CASH_TIERS = [0.75, 1.0, 1.25]
_TRADE_BAND = NUM_OWNABLE * NUM_TRADE_TIERS  # 84
A_TRADE_ACCEPT = A_TRADE + _TRADE_BAND       # 202: accept a trade offered to me
A_TRADE_REJECT = A_TRADE_ACCEPT + 1          # 203: reject a trade offered to me
A_AUCTION_PASS = A_TRADE_REJECT + 1          # 204: bid nothing in the auction
A_AUCTION_BID = A_AUCTION_PASS + 1           # A_AUCTION_BID + k: bid BID_FRACTIONS[k]
NUM_BID_LEVELS = 6

# Auction bid buckets, each a multiple of the property's *value to the bidder*
# (``ObsEncoder._bid_value``: list price, boosted when the tile completes the
# bidder's set or blocks an opponent's). Scaling by value -- not raw list price
# -- lets the agent pay a premium for a pivotal tile while staying conservative
# on ordinary ones. The absolute bid is still bounded by ``BID_CEILING_MULT`` *
# list price and by cash (see ``MonopolyEnv._make_bid_hook``).
BID_FRACTIONS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
BID_CEILING_MULT = 4.0   # hard ceiling on any bid, as a multiple of list price

NUM_ACTIONS = A_AUCTION_BID + NUM_BID_LEVELS  # 211


def trade_action(tile_index, tier):
    """Flat action id proposing to acquire ownable tile ``tile_index`` at cash
    ``tier`` (0..NUM_TRADE_TIERS-1). Inverse of :func:`decode_trade_action`."""
    return A_TRADE + tile_index * NUM_TRADE_TIERS + tier


def decode_trade_action(action):
    """Splits a trade-band action id into ``(tile_index, tier)``. Assumes
    ``A_TRADE <= action < A_TRADE + NUM_OWNABLE * NUM_TRADE_TIERS``."""
    offset = action - A_TRADE
    return offset // NUM_TRADE_TIERS, offset % NUM_TRADE_TIERS
