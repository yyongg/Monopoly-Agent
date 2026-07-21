"""Structural constants for the RL interface: decision phases, the flat action
id layout, and the auction bid grid.

These are pure identifiers (no tunable behaviour -- see :mod:`engine.config` for
the reward/valuation knobs). They are re-exported from :mod:`engine.rl_env` so
existing ``from engine.rl_env import PHASE_BUY, A_BUILD, ...`` imports keep
working.
"""

# --- Decision phases -------------------------------------------------------
# Trading is **not** here: it is resolved entirely by the heuristic in
# :mod:`engine.trade`, never put to the policy. See the action-id note below.
PHASE_JAIL = 0
PHASE_BUY = 1
PHASE_MANAGE = 2
PHASE_LIQUIDATE = 3
PHASE_TERMINAL = 4
PHASE_AUCTION = 5        # submit a sealed bid for a property up for auction
NUM_PHASES = 6

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
# There are no trade actions. Trading used to own 86 of 211 action ids -- an
# 84-id proposal band (tile x cash tier) plus accept/reject -- and the policy
# was very bad at it: it accepted ~85% of every offer put to it, gave away sets
# for junk, and ping-ponged one tile 424 times in a single game.
#
# Trades are now decided entirely by the heuristic in :mod:`engine.trade`, so
# those ids are gone rather than dead-masked. That follows Bonjour et al. 2021
# (arxiv 2103.00683), whose *hybrid* agent -- fixed heuristic for the rare
# accept-trade decision, DRL for the rest -- beat their pure-DRL agent 91.65% to
# 69.95%: a decision this rare and this valuation-heavy costs sample complexity
# in the policy and returns nothing. It also frees offers from the action space,
# so the heuristic can assemble multi-tile packages the 84-id band could not name.
A_AUCTION_PASS = 118                # bid nothing in the auction
A_AUCTION_BID = A_AUCTION_PASS + 1  # A_AUCTION_BID + k: bid BID_FRACTIONS[k]
NUM_BID_LEVELS = 6

# Auction bid buckets, each a multiple of the property's *value to the bidder*
# (``ObsEncoder._bid_value``: list price, boosted when the tile completes the
# bidder's set or blocks an opponent's). Scaling by value -- not raw list price
# -- lets the agent pay a premium for a pivotal tile while staying conservative
# on ordinary ones. The absolute bid is still bounded by ``BID_CEILING_MULT`` *
# list price and by cash (see ``MonopolyEnv._make_bid_hook``).
BID_FRACTIONS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
BID_CEILING_MULT = 4.0   # hard ceiling on any bid, as a multiple of list price

NUM_ACTIONS = A_AUCTION_BID + NUM_BID_LEVELS  # 125
