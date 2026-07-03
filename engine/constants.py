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
A_TRADE = 118        # A_TRADE + i: propose to acquire ownable tile i by trade
NUM_OWNABLE = 28
NUM_GROUPS = 10      # monopoly groups: 8 street colors + railroads + utilities
A_TRADE_ACCEPT = 146     # accept a trade offered to me (PHASE_TRADE_RESPOND)
A_TRADE_REJECT = 147     # reject a trade offered to me
A_AUCTION_PASS = 148     # bid nothing in the current auction
A_AUCTION_BID = 149      # A_AUCTION_BID + k: bid BID_FRACTIONS[k] * bid-value
NUM_BID_LEVELS = 6

# Auction bid buckets, each a multiple of the property's *value to the bidder*
# (``ObsEncoder._bid_value``: list price, boosted when the tile completes the
# bidder's set or blocks an opponent's). Scaling by value -- not raw list price
# -- lets the agent pay a premium for a pivotal tile while staying conservative
# on ordinary ones. The absolute bid is still bounded by ``BID_CEILING_MULT`` *
# list price and by cash (see ``MonopolyEnv._make_bid_hook``).
BID_FRACTIONS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
BID_CEILING_MULT = 4.0   # hard ceiling on any bid, as a multiple of list price

NUM_ACTIONS = A_AUCTION_BID + NUM_BID_LEVELS  # 155
