"""Gym-style reinforcement-learning environment for the Monopoly engine.

``MonopolyEnv`` wraps :class:`engine.game.Game` and exposes the standard
Gymnasium API (``reset`` / ``step``) so an RL agent can play one seat. It is a
*decision-point* environment: each call to ``step`` supplies a single action for
the one decision the controlled player currently faces (escape jail, buy a
property, a property-management action, or raising cash under a shortfall).
Between the controlled player's decisions the environment plays out everything
that needs no choice from them -- dice, rent, cards, and the opponents' entire
turns -- automatically.

Opponents play the engine baseline by default, but can be driven by a policy
(``opponent_policy`` / ``opponent_provider``) so the same network plays every
seat -- the basis for self-play. Observations are perspective-relative (the
acting seat is always relative player 0), which is what lets one policy control
any seat.

Why a worker thread
-------------------
A Monopoly turn makes several decisions, some of them *nested* deep inside the
engine: landing on an unowned property offers a purchase, and a Chance card such
as "Advance to Boardwalk" can trigger a second purchase offer from inside
``resolve_tile``. Rather than re-implement the engine as a coroutine, the real
game loop runs on a background thread; the controlled player's decision hooks
(``decide_purchase`` / ``on_shortfall``) and the management phases block on a
queue, handing control back to ``step``. This reuses the engine verbatim, so the
RL view of the rules can never drift from headless play.

Strict ping-pong between the two threads (worker produces one request then
blocks; ``step`` consumes it, reads state while the worker is blocked, then
unblocks the worker) means only one thread ever touches game state at a time, so
no locking is required.

Action space (``Discrete(154)``)
--------------------------------
A single flat action id, interpreted against the per-step ``action_mask`` in
``info`` (and :meth:`MonopolyEnv.legal_actions`). Property-targeted actions are
indexed by the 28 ownable tiles in board order::

    0  PAY_JAIL          pay the $50 fine to leave jail
    1  USE_JAIL_CARD     spend a Get Out of Jail Free card
    2  ROLL_JAIL         roll for doubles to leave jail
    3  BUY               buy the property just landed on
    4  DECLINE           decline the offered property (it then goes to auction)
    5  END_MANAGE        finish managing / stop liquidating
    6  +i  BUILD i       build a house/hotel on ownable tile i
    34 +i  SELL i        sell a house/hotel from tile i
    62 +i  MORTGAGE i    mortgage tile i
    90 +i  UNMORTGAGE i  lift the mortgage on tile i
    118+i  TRADE i       propose to acquire ownable tile i from its owner (only
                         legal when acquiring i completes a monopoly for the
                         agent); the engine builds the balancing give + cash
                         offer and the partner accepts/rejects it.
    146    TRADE_ACCEPT  accept a trade another player has offered me
    147    TRADE_REJECT  reject a trade another player has offered me
    148    AUCTION_PASS  bid nothing in the current auction
    149+k  AUCTION_BID k bid BID_FRACTIONS[k] * price for the auctioned property

Illegal actions (those not set in the mask) are clamped to a safe default rather
than raising, and reported via ``info["illegal"]``.
"""

import json
import os
import queue
import random
import threading

import numpy as np

from engine.game import Game
from models.board import Board
from models.player import Player
from models.tiles.properties.street_property import StreetProperty
from models.tiles.properties.railroad import Railroad
from models.tiles.properties.utility import Utility
from data.board_tiles import build_board_tiles
from data.decks import build_chance_deck, build_community_deck

try:  # Gymnasium is optional; the env works standalone without it.
    from gymnasium import Env as _GymEnv
    from gymnasium.spaces import Box, Discrete
    _HAS_GYM = True
except ImportError:  # pragma: no cover - exercised only without gymnasium
    _HAS_GYM = False
    _GymEnv = object

    class Discrete:
        """Minimal stand-in for ``gymnasium.spaces.Discrete``."""

        def __init__(self, n):
            self.n = int(n)

        def sample(self):
            return int(np.random.randint(self.n))

    class Box:
        """Minimal stand-in for ``gymnasium.spaces.Box``."""

        def __init__(self, low, high, shape, dtype=np.float32):
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype


# --- Decision phases -------------------------------------------------------
PHASE_JAIL = 0
PHASE_BUY = 1
PHASE_MANAGE = 2
PHASE_LIQUIDATE = 3
PHASE_TERMINAL = 4
PHASE_AUCTION = 5        # submit a sealed bid for a property up for auction
PHASE_TRADE_RESPOND = 6  # accept or reject a trade offered by another player
NUM_PHASES = 7

# --- Action layout ---------------------------------------------------------
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
# (``_bid_value``: list price, boosted when the tile completes the bidder's set
# or blocks an opponent's). Scaling by value -- not raw list price -- lets the
# agent pay a premium for a pivotal tile while staying conservative on ordinary
# ones. The absolute bid is still bounded by ``BID_CEILING_MULT`` * list price
# and by cash (see ``_make_bid_hook``).
BID_FRACTIONS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
BID_CEILING_MULT = 3.0   # hard ceiling on any bid, as a multiple of list price
DENIAL_VALUE_WEIGHT = 1.0  # weight on the blocking (deny-opponent) value term
DENIAL_BONUS_COEF = 0.5  # one-time reward for taking an opponent's last tile,
#                          as a fraction of that denied set's shaped value
# One-time reward for acquiring an unowned property from the bank, scaled by the
# tile's *expected income* (landing traffic x nominal rent) so busy, high-rent
# tiles are worth chasing while junk earns almost nothing. This breaks the
# passive "decline everything and let it go to auction" equilibrium that
# self-play falls into.
#
# Buying on landing earns the *full* bonus; winning the same tile at auction
# earns only ``AUCTION_ACQUISITION_BONUS_COEF``. The gap keeps the agent
# contesting auctions (rather than conceding them) while preferring to buy
# outright.
#
# The income-scaled gap alone, though, is far too small to stop the agent
# declining everything: the shaped net worth books an owned tile at ``price x
# (1 + ACQUISITION_PREMIUM)`` regardless of what was paid, so acquiring *cheaply*
# is an instant net-worth windfall -- and an early-game auction (few contested
# bidders, opening bid ~10% of list) is the cheap route. That windfall scales
# with *price* (~0.5-1.0 x price / 1000 of reward), which dwarfs the income-scaled
# bonus gap (~a few x income / 1000). So buying on landing also earns a
# *price-scaled* preference bonus (``BUY_PREFERENCE_COEF`` x price / 1000), paid
# only on a direct buy and never on an auction win.
#
# Sizing it (this matters -- earlier values that were "near ACQUISITION_PREMIUM"
# were provably too weak). Comparing buy-at-list against decline-then-snipe the
# auction at price ``A``, in /1000 reward units:
#   buy   = 0.5*P (net-worth telescoping) + 3*income + c*P
#   snipe = (1.5*P - A)  (telescoping)    + 1*income
#   buy - snipe = (c - 1)*P + A + 2*income
# So buying only out-scores a *free* snipe (A -> 0) once ``c >= 1.0`` -- at the
# old c = 0.5 the bonus covered barely half the list-price gap, which is why the
# agent kept declining. We set c = 1.25: comfortably past break-even so buying
# wins for any realistic auction price with margin to spare for the policy to
# latch onto, without inflating it so far the agent over-buys into rent it can't
# cover. Raise it for an even stronger buy bias, lower toward 1.0 to tighten it
# (below 1.0 reopens the decline-then-snipe exploit).
ACQUISITION_BONUS_COEF = 3.0
AUCTION_ACQUISITION_BONUS_COEF = 1.0
BUY_PREFERENCE_COEF = 1.40
NUM_ACTIONS = A_AUCTION_BID + NUM_BID_LEVELS  # 155

# --- Landing-frequency prior ----------------------------------------------
# How often each board tile is landed on, precomputed by
# ``validation/board_visits.py`` (a movement-only Monte-Carlo). Fed into the
# observation so the agent can value a high-traffic property above a quiet one
# of the same price: a tile landed on twice as often earns ~twice the rent.
# Stored as a share of all landings (sums to ~1 over the 40 tiles); the
# observation multiplies by the board size to express it as a "traffic vs even"
# multiple (1.0 == average). If the table is missing the env falls back to a
# uniform prior, so training still runs (every tile just looks average).
LANDING_FREQ_PATH = os.path.join("runs", "board_visits.json")
_LAND_FREQ_CACHE = None


def load_landing_frequencies(path=LANDING_FREQ_PATH):
    """Returns ``{board_pos: landing_share}`` from the saved visit table.

    Cached after first load. On a missing/unreadable file returns ``{}``, which
    callers treat as a uniform prior (see ``base_traffic``)."""
    global _LAND_FREQ_CACHE
    if _LAND_FREQ_CACHE is not None:
        return _LAND_FREQ_CACHE
    freqs = {}
    try:
        with open(path) as f:
            data = json.load(f)
        for t in data["tiles"]:
            freqs[int(t["pos"])] = float(t["frequency"])
    except (OSError, ValueError, KeyError):
        freqs = {}
    _LAND_FREQ_CACHE = freqs
    return freqs


def base_rent(prop):
    """A nominal single-ownership rent for ``prop``, used as a scale-normalised
    "rent it can collect" observation feature (exact value is unimportant)."""
    if isinstance(prop, StreetProperty):
        return float(prop.rent_table[0])
    if isinstance(prop, Railroad):
        return 25.0            # nominal one-railroad rent
    return 28.0               # utility: 4x an average roll of 7


# Sentinels passed between the worker thread and the env.
_TERMINAL = "__terminal__"
_ABORT = "__abort__"


class _Abort(Exception):
    """Raised inside the worker thread to tear it down on ``reset``/``close``."""


class MonopolyEnv(_GymEnv):
    """Single-agent Gymnasium environment for one seat of a Monopoly game.

    Args:
        seat (int): Index of the player the agent controls (0-based).
        num_players (int): Total players in the game (controlled + opponents).
        names (list[str] | None): Optional player names; defaults to colours.
        max_turns (int): Turn cap after which an episode is truncated, so two
            survivors who never bankrupt each other still terminate.
        reward_mode (str): ``"shaped"`` (net-worth change each decision plus a
            decisive terminal outcome) or ``"sparse"`` (only the terminal
            outcome). The terminal outcome is win/loss by net-worth rank so that
            turn-cap timeouts are still decisive -- see :meth:`_terminal_reward`.
        seed (int | None): Seed for the dice RNG (also settable via ``reset``).
        opponent_policy (callable | None): A fixed opponent policy
            ``(observation, action_mask_bool) -> action_index`` used to drive
            *every* opponent seat. ``None`` keeps the engine baseline (opponents
            play their whole turn via ``Game.step``). Observations handed to the
            policy are from that opponent's own perspective (it always sees
            itself as relative player 0), so a single policy can play any seat.
        opponent_provider (callable | None): A zero-arg callable sampled at each
            ``reset`` that returns an opponent policy (or ``None`` for baseline).
            Used for self-play, where opponents are sampled from a snapshot pool
            each episode. Takes precedence over ``opponent_policy``.
    """

    metadata = {"render_modes": []}

    # Shaped-reward premium on an owned, unmortgaged property above its cash
    # price. A property earns rent, so it is worth more than the cash paid for
    # it; without this premium buying is exactly net-worth-neutral (reward 0)
    # and the policy collapses to "always decline". With it, buying yields a
    # small positive shaped reward (premium * price / 1000), so acquisition is
    # encouraged while the terminal -1 still punishes reckless over-buying.
    ACQUISITION_PREMIUM = 0.5

    # Extra shaped value for *completing a monopoly*, as a multiple of the
    # group's total list price. A full set unlocks houses and multiplied rent,
    # so it is worth far more than its individual tiles: completing one gives a
    # large net-worth jump (bonus * group price / 1000 reward), which teaches
    # the agent to finish sets -- and makes mortgaging a spare property to
    # afford the set-completing tile clearly worth the ~10% mortgage interest.
    MONOPOLY_BONUS = 1.0

    def __init__(self, seat=None, num_players=4, names=None, max_turns=1000,
                 reward_mode="shaped", seed=None, opponent_policy=None,
                 opponent_provider=None):
        if seat is not None and not 0 <= seat < num_players:
            raise ValueError("seat must be in range(num_players)")
        self._seat_fixed = seat  # None means pick a random seat each episode
        self.seat = seat if seat is not None else 0  # overwritten at reset
        self.num_players = num_players
        self._names = names or ["Red", "Blue", "Green", "Yellow"][:num_players]
        if len(self._names) != num_players:
            raise ValueError("names must have num_players entries")
        self.max_turns = max_turns
        self.reward_mode = reward_mode
        self._seed = seed

        # Opponent control: a fixed policy, a per-episode provider, and the
        # policy chosen for the current episode (None == engine baseline). The
        # decider map routes each seat's decisions to the agent (queue) or to an
        # opponent policy (synchronous), and is rebuilt each reset.
        self._opponent_policy_fixed = opponent_policy
        self._opponent_provider = opponent_provider
        self._opponent_policy = None
        self._deciders = {}

        # The 28 ownable tiles, fixed by board order, give every property a
        # stable action/observation index for the agent.
        self._ownable_template = [
            t for t in build_board_tiles()
            if isinstance(t, (StreetProperty, Railroad, Utility))
        ]
        assert len(self._ownable_template) == NUM_OWNABLE
        assert len(self._build_groups(self._ownable_template)) == NUM_GROUPS

        self.action_space = Discrete(NUM_ACTIONS)
        obs_len = (5 * num_players
                   + NUM_OWNABLE * (num_players + 3)
                   + NUM_PHASES + 1
                   + 3               # trade context: recv tile, give tile, cash
                   + 2               # context flags: completes mine / opp's set
                   + NUM_GROUPS * 2  # per-group progress: mine / max-opp frac
                   + 3)              # context prop economics: price, rent, traffic
        self.observation_space = Box(
            low=-1.0, high=np.inf, shape=(obs_len,), dtype=np.float32)
        self._obs_len = obs_len

        # Cross-thread channels and per-episode worker state.
        self._req_q = queue.Queue()
        self._resp_q = queue.Queue()
        self._thread = None
        self._done = True
        self._seeded = False       # whether the RNG has been seeded once
        self._rng = random.Random()  # per-env RNG for dice and deck order
        self._cur_phase = PHASE_TERMINAL
        self._cur_mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        self._cur_prop = None      # property offered (BUY/AUCTION) or liquidated
        self._cur_amount = 0       # shortfall amount (LIQUIDATE)
        self._cur_trade = None     # offer being judged (PHASE_TRADE_RESPOND)
        self._traded_this_manage = set()  # trade targets tried this MANAGE phase
        self._prev_advantage = 0.0  # last relative-advantage potential (shaping)
        self._pending_bonus = 0.0   # denial reward queued by the on_acquire hook
        self._truncated = False
        self.game = None
        self.ownable = []

    # -- Gymnasium API ------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        """Starts a fresh episode and returns ``(observation, info)``."""
        # Seed the RNG (Gymnasium's ``self.np_random``); on the very first reset
        # fall back to the constructor seed so MonopolyEnv(seed=...) is
        # reproducible without an explicit reset(seed=...).
        if seed is None and not self._seeded:
            seed = self._seed
        if _HAS_GYM:
            super().reset(seed=seed)
        elif seed is not None or not self._seeded:
            self.np_random = np.random.default_rng(seed)
        self._seeded = True
        if self._seat_fixed is None:
            self.seat = int(self.np_random.integers(0, self.num_players))

        self._shutdown_worker()
        self._build_game()

        # Pick this episode's opponent policy (a provider, if any, is sampled
        # fresh each episode for self-play; otherwise the fixed policy).
        if self._opponent_provider is not None:
            self._opponent_policy = self._opponent_provider()
        else:
            self._opponent_policy = self._opponent_policy_fixed
        self._wire_deciders()

        controlled = self.game.players[self.seat]
        # Shortfalls are dispatched by payer: the agent and any policy-driven
        # opponents may liquidate; baseline opponents fall through to bankruptcy.
        self.game.on_shortfall = self._on_shortfall
        # Credit a denial bonus whenever a tile changes hands to the agent.
        self.game.on_acquire = self._on_acquire

        self._done = False
        self._truncated = False
        self._prev_advantage = (self._net_worth(controlled)
                                - self._mean_opp_networth())
        self._pending_bonus = 0.0
        self._thread = threading.Thread(target=self._run_game, daemon=True)
        self._thread.start()
        obs, _, _, _, info = self._pump_until_request()
        return obs, info

    def step(self, action):
        """Applies one action and advances to the next decision.

        Returns the Gymnasium 5-tuple
        ``(observation, reward, terminated, truncated, info)``.
        """
        if self._done:
            raise RuntimeError("step() called on a finished episode; reset() first")
        self._resp_q.put(int(action))
        return self._pump_until_request()

    def close(self):
        """Tears down the worker thread if an episode is still running."""
        self._shutdown_worker()

    def legal_actions(self):
        """Returns the indices of actions legal for the current decision."""
        return np.flatnonzero(self._cur_mask).tolist()

    @property
    def action_mask(self):
        """The current 0/1 legal-action mask (length ``NUM_ACTIONS``)."""
        return self._cur_mask.copy()

    def action_masks(self):
        """Boolean legal-action mask for the current decision.

        This is the method name sb3-contrib's maskable utilities look for
        (``MaskablePPO``); it mirrors :attr:`action_mask` as a bool array.
        """
        return self._cur_mask.astype(bool)

    def set_opponent_policy(self, policy):
        """Sets a fixed opponent policy used from the next ``reset`` onward.

        ``policy`` is a callable ``(observation, action_mask_bool) -> action``
        or ``None`` to restore baseline opponents. Clears any provider.
        """
        self._opponent_policy_fixed = policy
        self._opponent_provider = None

    # -- Per-seat decision routing ------------------------------------------
    def _wire_deciders(self):
        """Builds the seat -> decider map for the current episode.

        The agent seat is always routed to :meth:`_agent_decide` (which blocks
        for an external action via the queue). When an opponent policy is set,
        every other seat is routed to :meth:`_policy_decide` (synchronous, no
        queue). Each routed seat also gets its ``decide_purchase`` hook wired so
        nested purchase offers (e.g. from a Chance "Advance to" card) reach the
        same decider. Seats with no decider keep the engine baseline.
        """
        self._deciders = {self.seat: self._agent_decide}
        if self._opponent_policy is not None:
            for s in range(self.num_players):
                if s != self.seat:
                    self._deciders[s] = self._make_policy_decider(s)
        for s, decide in self._deciders.items():
            self.game.players[s].decide_purchase = self._make_buy_hook(s, decide)
            self.game.players[s].decide_bid = self._make_bid_hook(s, decide)

    def _make_policy_decider(self, seat):
        """Returns a decider bound to ``seat`` that calls the opponent policy."""
        return lambda phase, prop=None, amount=0: self._policy_decide(
            seat, phase, prop, amount)

    def _make_buy_hook(self, seat, decide):
        """Returns a ``decide_purchase(prop)`` hook routed through ``decide``.

        If the decider chooses to buy but is short on cash, it runs a
        liquidation sub-phase (mortgage / sell) to reach the price before the
        engine finalizes the purchase -- this is how the agent affords a
        set-completing tile. It declines if it cannot or will not cover it.
        """
        player = self.game.players[seat]

        def hook(prop):
            if decide(PHASE_BUY, prop) != A_BUY:
                return False
            while player.balance < prop.price:
                if not self._has_liquidation_options(player):
                    return False
                action = decide(PHASE_LIQUIDATE, prop, prop.price)
                if action == A_END_MANAGE:
                    return False
                self._apply_manage_action(player, action)
            return True

        return hook

    def _make_bid_hook(self, seat, decide):
        """Returns a ``decide_bid(prop, min_bid)`` hook routed through ``decide``.

        Each ascending-auction round the decider picks a bid bucket for ``prop``;
        the hook reads it as the seat's valuation ceiling -- a fraction of the
        tile's *value to this bidder* (``_bid_value``), bounded by
        ``BID_CEILING_MULT`` * list price and by cash. It matches the round's
        ``min_bid`` while that ceiling covers it, and otherwise passes -- so the
        agent stays in the bidding until the price climbs past what its chosen
        bucket is worth. ``A_AUCTION_PASS`` (or an unaffordable bucket) drops out.
        """
        player = self.game.players[seat]

        def hook(prop, min_bid=0):
            action = decide(PHASE_AUCTION, prop)
            k = action - A_AUCTION_BID
            if 0 <= k < NUM_BID_LEVELS:
                value = self._bid_value(player, prop)
                ceiling = min(int(round(BID_FRACTIONS[k] * value)),
                              int(round(BID_CEILING_MULT * prop.price)),
                              player.balance)
                if min_bid <= 0:
                    return ceiling
                return min_bid if ceiling >= min_bid else 0
            return 0  # A_AUCTION_PASS or anything unexpected

        return hook

    def _group_price(self, grp):
        """Total list price of every tile in a monopoly group."""
        return sum(t.price for t in grp)

    def _bid_value(self, player, prop):
        """``prop``'s value to ``player`` for auction bidding: its list price,
        boosted by the group's value when winning it completes ``player``'s own
        set, or (weighted) when it would complete an opponent's set -- so the
        ceiling reflects how pivotal the tile is, not just its sticker price."""
        value = float(prop.price)
        grp = self._group_of.get(id(prop))
        if grp is None:
            return value
        if self._completes_monopoly_for(player, prop):
            value += self.MONOPOLY_BONUS * self._group_price(grp)
        elif any(o is not player and not o.bankrupt
                 and self._completes_monopoly_for(o, prop)
                 for o in self.game.players):
            value += (DENIAL_VALUE_WEIGHT * self.MONOPOLY_BONUS
                      * self._group_price(grp))
        return value

    def _expected_income(self, prop):
        """A proxy for the rent ``prop`` will earn its owner: how often it is
        landed on (traffic vs an average tile) times its nominal rent."""
        uniform = 1.0 / self._board_size
        traffic = self._land_freq.get(prop.pos, uniform) * self._board_size
        return traffic * base_rent(prop)

    def _on_acquire(self, player, prop, source="trade"):
        """``Game.on_acquire`` hook: ``prop`` just transferred to ``player``.

        ``source`` is "buy" (bought on landing), "auction" (won at auction), or
        "trade" (player-to-player). For the controlled agent, queue two one-time
        shaped bonuses:

        * **Acquisition** -- when the tile was taken fresh from the bank, a reward
          scaled by its expected income, so the agent actively acquires properties
          instead of dumping them to auction. Buying on landing earns the full
          ``ACQUISITION_BONUS_COEF``; an auction win earns the smaller
          ``AUCTION_ACQUISITION_BONUS_COEF`` -- so buying the tile it wants
          outright beats declining it and sniping the cheaper auction, while
          winning is still worth more than conceding to an opponent. Trades don't
          earn it (they aren't the source of the passivity problem and net worth
          already prices them).
        * **Denial** -- when the tile was an opponent's last-missing piece,
          blocking their monopoly (any source). The completion test ignores
          ``prop``'s new owner, so evaluating it after the transfer is correct.
        """
        if player is not self.game.players[self.seat]:
            return
        if source in ("buy", "auction"):
            coef = (ACQUISITION_BONUS_COEF if source == "buy"
                    else AUCTION_ACQUISITION_BONUS_COEF)
            self._pending_bonus += coef * self._expected_income(prop) / 1000.0
            if source == "buy":
                # Price-scaled premium for buying on landing (never on an auction
                # win), sized to cancel the cheap-auction net-worth windfall so
                # buying at list beats declining-to-snipe. See BUY_PREFERENCE_COEF.
                self._pending_bonus += BUY_PREFERENCE_COEF * prop.price / 1000.0
        grp = self._group_of.get(id(prop))
        if grp is None:
            return
        if any(o is not player and not o.bankrupt
               and self._completes_monopoly_for(o, prop)
               for o in self.game.players):
            self._pending_bonus += (DENIAL_BONUS_COEF * self.MONOPOLY_BONUS
                                    * self._group_price(grp) / 1000.0)

    # -- Game construction --------------------------------------------------
    def _build_game(self):
        players = [Player(name) for name in self._names]
        board = Board(build_board_tiles())
        game = Game(players, board, build_chance_deck(), build_community_deck())

        # Give this env a private RNG, seeded from the Gymnasium RNG, so each
        # episode differs yet the whole run is reproducible from the seed, and
        # parallel envs diverge. The engine otherwise draws dice and shuffles
        # decks from the global ``random`` module, which would couple envs that
        # share a process; route both through this generator instead.
        episode_seed = int(self.np_random.integers(0, 2 ** 31 - 1))
        self._rng = random.Random(episode_seed)
        game.roll_dice = self._make_roll_dice(game)
        self._rng.shuffle(game.chance_deck.cards)
        self._rng.shuffle(game.community_deck.cards)

        self.game = game
        self.ownable = [
            t for t in board.tiles
            if isinstance(t, (StreetProperty, Railroad, Utility))
        ]
        self._prop_index = {id(p): i for i, p in enumerate(self.ownable)}
        # Monopoly groups (each street colour, all railroads, all utilities),
        # precomputed once for the completed-set reward bonus and trade logic.
        self._groups = self._build_groups(self.ownable)
        self._group_of = {id(t): grp for grp in self._groups for t in grp}
        # Landing-frequency prior (share per board pos) and the board size, used
        # to expose each tile's "traffic vs even" multiple in the observation.
        self._land_freq = load_landing_frequencies()
        self._board_size = board.length

    @staticmethod
    def _build_groups(ownable):
        """Groups ownable tiles into the sets that form a monopoly, returned in
        a stable order (by group key) so the per-group observation features line
        up identically between the env and the GUI mirror."""
        groups = {}
        for t in ownable:
            if isinstance(t, StreetProperty):
                key = ("street", t.color)
            elif isinstance(t, Railroad):
                key = ("railroad", "")
            else:
                key = ("utility", "")
            groups.setdefault(key, []).append(t)
        return [groups[k] for k in sorted(groups)]

    def _make_roll_dice(self, game):
        """Returns a ``roll_dice`` bound to this env's private RNG."""
        rng = self._rng

        def roll_dice():
            game.last_dice = (rng.randint(1, 6), rng.randint(1, 6))
            return game.last_dice

        return roll_dice

    # -- Worker thread: the real game loop ----------------------------------
    def _run_game(self):
        """Runs the full game on the worker thread until the episode ends.

        Each seat is played by :meth:`_play_turn` when it has a decider (the
        agent, or a policy-driven opponent); seats with no decider use the
        engine's own ``Game.step`` baseline. Exits when the agent is out, the
        game is over, or the turn cap is hit, then signals the env.
        """
        g = self.game
        controlled = g.players[self.seat]
        turns = 0
        try:
            while (not g.is_over() and not controlled.bankrupt
                   and turns < self.max_turns):
                seat = g.current_player
                if seat in self._deciders:
                    self._play_turn(g.players[seat], self._deciders[seat])
                else:
                    g.step()  # baseline opponent, whole turn
                turns += 1
            self._truncated = turns >= self.max_turns and not g.is_over()
        except _Abort:
            return
        # Signal the end of the episode to whoever is waiting in step().
        self._req_q.put(_TERMINAL)

    def _play_turn(self, player, decide):
        """Plays one turn for ``player``, mirroring ``Game.step``.

        ``decide(phase, prop, amount) -> action`` supplies every choice (jail
        action, pre/post-roll management, and the nested purchase / shortfall
        hooks). For the agent it blocks on the queue; for an opponent policy it
        runs synchronously. The turn is advanced exactly once on every exit
        path, matching the engine.
        """
        g = self.game

        if player.in_jail:
            choice = self._jail_choice(decide)
            result = g.handle_jail_turn(player, choice)
            if result in ("jailed", "freed"):
                g.advance_turn()
                return
            if result == "moved":
                g.resolve_tile(player)
                g.advance_turn()
                return
            # "released": paid / used a card, now take a normal turn.

        self._manage_phase(player, decide)  # pre-roll

        while True:
            _, _, is_double, sent_to_jail = g.roll_once(player)
            if sent_to_jail:  # third double
                g.advance_turn()
                return
            g.resolve_tile(player)
            if player.in_jail:  # a Go-To-Jail tile/card ended the turn
                player.double_count = 0
                g.advance_turn()
                return
            if player.bankrupt:
                player.double_count = 0
                g.advance_turn()
                return
            if not is_double:
                break

        self._manage_phase(player, decide)  # post-roll
        player.double_count = 0
        g.advance_turn()

    def _manage_phase(self, player, decide):
        """Lets ``player`` issue management actions until it chooses to stop."""
        # A trade target is offered at most once per MANAGE phase; re-proposing a
        # rejected trade changes no state, so without this the phase could spin
        # forever on a repeatedly-declined offer.
        self._traded_this_manage = set()
        while True:
            action = decide(PHASE_MANAGE)
            if action == A_END_MANAGE:
                return
            self._apply_manage_action(player, action)
            if player.bankrupt:
                return

    def _jail_choice(self, decide):
        action = decide(PHASE_JAIL)
        if action == A_PAY_JAIL:
            return "pay"
        if action == A_USE_CARD:
            return "card"
        return "roll"

    def _on_shortfall(self, payer, amount):
        """``Game.on_shortfall`` hook, dispatched by payer.

        The agent and policy-driven opponents may sell houses, mortgage, or
        trade until they cover the debt or give up; baseline opponents (no
        decider) fall through to bankruptcy as in headless play.
        """
        seat = self.game.players.index(payer)
        decide = self._deciders.get(seat)
        if decide is None:
            return
        while payer.balance < amount:
            if not self._has_liquidation_options(payer):
                return  # nothing left to raise; bankruptcy will follow
            action = decide(PHASE_LIQUIDATE, None, amount)
            if action == A_END_MANAGE:
                return  # chose to stop; bankruptcy will follow
            self._apply_manage_action(payer, action)

    # -- Deciders: agent (queue) vs opponent policy (synchronous) -----------
    def _agent_decide(self, phase, prop=None, amount=0):
        """The agent's decider: posts a request and blocks for ``step``.

        Records the current decision (phase/property/mask, all from the agent's
        own perspective) so ``step`` can build the observation, then validates
        the returned action against the mask -- clamping an illegal action to a
        safe default rather than letting it corrupt state.
        """
        self._cur_phase = phase
        self._cur_prop = prop
        self._cur_amount = amount
        self._cur_mask = self._legal_mask(phase, prop, self.seat)
        self._req_q.put((phase, prop, amount))
        action = self._resp_q.get()
        if action == _ABORT:
            raise _Abort()
        self._illegal = not (0 <= action < NUM_ACTIONS and self._cur_mask[action])
        if self._illegal:
            action = self._safe_default(phase)
        return action

    def _policy_decide(self, seat, phase, prop=None, amount=0):
        """An opponent's decider: queries the opponent policy synchronously.

        Builds the mask and observation from ``seat``'s own perspective and
        clamps an illegal/invalid policy action to a safe default.
        """
        mask = self._legal_mask(phase, prop, seat)
        obs = self._encode_obs(seat, phase, prop)
        action = int(self._opponent_policy(obs, mask.astype(bool)))
        if not (0 <= action < NUM_ACTIONS and mask[action]):
            action = self._safe_default(phase)
        return action

    def _safe_default(self, phase):
        if phase == PHASE_JAIL:
            return A_ROLL_JAIL
        if phase == PHASE_BUY:
            return A_DECLINE
        if phase == PHASE_AUCTION:
            return A_AUCTION_PASS
        if phase == PHASE_TRADE_RESPOND:
            return A_TRADE_REJECT
        return A_END_MANAGE  # MANAGE / LIQUIDATE: do nothing further

    # -- Applying actions ---------------------------------------------------
    def _apply_manage_action(self, player, action):
        """Carries out a build/sell/mortgage/unmortgage/trade action.

        The engine methods re-validate every move, so an action that slipped
        through the mask simply does nothing.
        """
        g = self.game
        if A_BUILD <= action < A_BUILD + NUM_OWNABLE:
            g.build_house(self.ownable[action - A_BUILD], player)
        elif A_SELL <= action < A_SELL + NUM_OWNABLE:
            g.sell_house(self.ownable[action - A_SELL], player)
        elif A_MORTGAGE <= action < A_MORTGAGE + NUM_OWNABLE:
            g.mortgage_property(self.ownable[action - A_MORTGAGE], player)
        elif A_UNMORTGAGE <= action < A_UNMORTGAGE + NUM_OWNABLE:
            g.unmortgage_property(self.ownable[action - A_UNMORTGAGE], player)
        elif A_TRADE <= action < A_TRADE + NUM_OWNABLE:
            i = action - A_TRADE
            self._traded_this_manage.add(i)
            self._attempt_trade(player, self.ownable[i])

    def _has_liquidation_options(self, player):
        """Whether ``player`` has any house to sell or property to mortgage."""
        g = self.game
        for prop in player.properties:
            if isinstance(prop, StreetProperty) and prop.can_sell_house(g, player):
                return True
            if prop.can_mortgage(g, player):
                return True
        return False

    def _raisable_cash(self, player):
        """Cash ``player`` could raise beyond its balance by selling every
        house and mortgaging every property -- an upper bound used to decide
        whether a not-yet-affordable purchase is within reach."""
        total = 0
        for prop in player.properties:
            if prop.mortgaged:
                continue
            total += prop.mortgage_value
            if isinstance(prop, StreetProperty):
                total += prop.houses * (prop.house_cost() // 2)
        return total

    def _can_afford(self, player, price):
        """Whether ``player`` can pay ``price`` now or after liquidating."""
        return (player.balance >= price
                or player.balance + self._raisable_cash(player) >= price)

    # -- Targeted trading ---------------------------------------------------
    def _prop_value(self, prop):
        """List-price value of a property, discounted for an unpaid mortgage.

        Mirrors ``ui/ai_player.GUIAIDecider._property_value`` so headless trades
        and GUI trades value tiles the same way.
        """
        if prop.mortgaged:
            return float(prop.price - prop.unmortgage_cost)
        return float(prop.price)

    def _completes_monopoly_for(self, player, target):
        """Whether acquiring ``target`` would complete a monopoly for ``player``
        (they already own every other tile in ``target``'s group)."""
        grp = self._group_of.get(id(target))
        if grp is None:
            return False
        return all(t.owner is player for t in grp if t is not target)

    def _can_propose_trade(self, initiator, target):
        """Whether ``initiator`` may propose a trade to acquire ``target``.

        Trades are restricted to set-completing acquisitions from a solvent
        opponent: the tile must be tradeable (no buildings in its group),
        acquiring it must finish a monopoly for the initiator, and the initiator
        must have a tile it can hand over in return.
        """
        owner = target.owner
        if owner is None or owner is initiator or owner.bankrupt:
            return False
        if not self.game.can_trade_property(target):
            return False
        if not self._completes_monopoly_for(initiator, target):
            return False
        return self._choose_give_tile(initiator, owner, target) is not None

    def _choose_give_tile(self, initiator, partner, target):
        """Picks the tile ``initiator`` offers ``partner`` in a trade for
        ``target`` (or ``None`` if it has nothing suitable to give).

        Prefers parting with a *spare* -- the only tile the initiator owns in its
        group, so the trade breaks no progress -- and, among spares, one that
        helps the partner complete a set (likelier to be accepted); otherwise the
        cheapest tradeable tile outside the target's group.
        """
        g = self.game
        target_group = self._group_of.get(id(target))

        def count(owner, group):
            return sum(1 for t in group if t.owner is owner)

        tradeable = [p for p in initiator.properties
                     if g.can_trade_property(p)
                     and self._group_of.get(id(p)) is not target_group]
        if not tradeable:
            return None
        spares = [p for p in tradeable
                  if count(initiator, self._group_of[id(p)]) == 1]
        pool = spares or tradeable
        helpful = [p for p in pool
                   if count(partner, self._group_of[id(p)])
                   == len(self._group_of[id(p)]) - 1]
        final = helpful or pool
        return min(final, key=lambda p: p.price)

    def _balancing_cash(self, initiator, partner, give, receive):
        """Cash the initiator pays the partner to balance a trade.

        The initiator covers the list-value gap (the received tile completes its
        set, so it is usually worth more than the tile given up) plus a premium
        for prying loose a set-completing tile. Never negative -- in the T1 model
        only the initiator pays, so the partner is never asked for cash.
        """
        recv_val = sum(self._prop_value(t) for t in receive)
        give_val = sum(self._prop_value(t) for t in give)
        grp = self._group_of.get(id(receive[0])) if receive else None
        set_price = sum(t.price for t in grp) if grp else 0
        premium = int(round(0.25 * set_price))
        return max(0, int(round(recv_val - give_val)) + premium)

    def _formula_trade_ok(self, partner, gain, lose, cash):
        """Baseline partner's accept rule: take the deal if it is non-negative by
        list value (``cash`` received plus tiles gained minus tiles given up)."""
        value = float(cash)
        value += sum(self._prop_value(p) for p in gain)
        value -= sum(self._prop_value(p) for p in lose)
        return value >= 0

    def _attempt_trade(self, initiator, target):
        """Builds and offers a set-completing trade for ``target``.

        The engine constructs the give-tile + balancing cash; the partner's
        decider (agent via the queue, opponent policy synchronously, or a
        baseline via :meth:`_formula_trade_ok`) accepts or rejects. On acceptance
        the swap runs through :meth:`Game.execute_trade`.
        """
        g = self.game
        partner = target.owner
        if not self._can_propose_trade(initiator, target):
            return
        give = self._choose_give_tile(initiator, partner, target)
        if give is None:
            return
        receive = [target]
        cash = min(self._balancing_cash(initiator, partner, [give], receive),
                   initiator.balance)
        if cash < 0:
            return

        partner_seat = g.players.index(partner)
        decide = self._deciders.get(partner_seat)
        if decide is not None:
            # From the partner's view they receive ``give`` plus ``cash`` and
            # part with ``target``; expose that as the response observation.
            self._cur_trade = {"recv": give, "give": target, "cash": cash}
            action = decide(PHASE_TRADE_RESPOND, target, cash)
            self._cur_trade = None
            accept = action == A_TRADE_ACCEPT
        else:
            accept = self._formula_trade_ok(partner, [give], receive, cash)

        if accept:
            g.execute_trade(initiator, partner, [give], receive, cash)

    # -- Legal-action masks -------------------------------------------------
    def _legal_mask(self, phase, prop, seat):
        mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        g = self.game
        player = g.players[seat]

        if phase == PHASE_JAIL:
            mask[A_ROLL_JAIL] = 1
            if player.balance >= 50:
                mask[A_PAY_JAIL] = 1
            if player.jail_cards:
                mask[A_USE_CARD] = 1
            return mask

        if phase == PHASE_BUY:
            mask[A_DECLINE] = 1
            # Offer BUY whenever the price is reachable -- either in cash now or
            # by mortgaging / selling first (the buy hook runs that liquidation
            # before finalizing), so the agent can afford a set-completing tile.
            if prop is not None and self._can_afford(player, prop.price):
                mask[A_BUY] = 1
            return mask

        if phase == PHASE_AUCTION:
            # Pass is always allowed; a bid bucket is legal when its dollar
            # amount is covered by cash (auctions are settled in cash only).
            mask[A_AUCTION_PASS] = 1
            if prop is not None:
                for k, frac in enumerate(BID_FRACTIONS):
                    if player.balance >= int(round(frac * prop.price)):
                        mask[A_AUCTION_BID + k] = 1
            return mask

        if phase == PHASE_TRADE_RESPOND:
            mask[A_TRADE_ACCEPT] = 1
            mask[A_TRADE_REJECT] = 1
            return mask

        # MANAGE and LIQUIDATE share the property-action masks.
        for i, p in enumerate(self.ownable):
            if p.owner is not player:
                continue
            if isinstance(p, StreetProperty):
                if phase == PHASE_MANAGE and p.can_build_house(g, player):
                    mask[A_BUILD + i] = 1
                # Selling houses is only offered during forced LIQUIDATE, for
                # the same reason as mortgaging: allowing it during voluntary
                # MANAGE let the agent sell<->rebuild houses on a monopoly.
                if phase == PHASE_LIQUIDATE and p.can_sell_house(g, player):
                    mask[A_SELL + i] = 1
            # Mortgaging is only offered during forced LIQUIDATE (raising cash
            # the player actually needs). Allowing it during voluntary MANAGE
            # let the agent mortgage-flip a property it just bought (and
            # oscillate mortgage<->unmortgage), so it is masked out there.
            if phase == PHASE_LIQUIDATE and p.can_mortgage(g, player):
                mask[A_MORTGAGE + i] = 1
            if phase == PHASE_MANAGE:
                if p.can_unmortgage(g, player):
                    mask[A_UNMORTGAGE + i] = 1

        # Trade proposals target tiles owned by *other* players: offer A_TRADE+i
        # for each opponent tile whose acquisition would complete a monopoly for
        # this player (and hasn't already been tried this MANAGE phase).
        if phase == PHASE_MANAGE:
            for i, p in enumerate(self.ownable):
                if i not in self._traded_this_manage \
                        and self._can_propose_trade(player, p):
                    mask[A_TRADE + i] = 1

        mask[A_END_MANAGE] = 1  # always allowed to stop
        return mask

    # -- Observation & reward ----------------------------------------------
    def _pump_until_request(self):
        """Waits for the next decision request (or the terminal signal)."""
        item = self._req_q.get()
        if item == _TERMINAL:
            self._done = True
            self._cur_phase = PHASE_TERMINAL
            self._cur_mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
            self._cur_prop = None
            obs = self._encode_obs(self.seat, PHASE_TERMINAL, None)
            reward = self._reward(terminal=True)
            return obs, reward, True, self._truncated, self._info(terminal=True)

        phase, prop, amount = item
        self._cur_phase, self._cur_prop, self._cur_amount = phase, prop, amount
        obs = self._encode_obs(self.seat, phase, prop)
        reward = self._reward(terminal=False)
        return obs, reward, False, False, self._info()

    def _encode_obs(self, perspective, phase, prop):
        """Builds the flat float32 observation from ``perspective``'s view.

        Players are ordered starting at ``perspective`` (so the acting seat is
        always relative player 0) and property owners are encoded by relative
        seat. This makes the observation perspective-invariant, so one policy
        can play any seat -- the basis for self-play.
        """
        g = self.game
        n = self.num_players
        parts = []

        # Per-player block (acting seat first): balance, position, jail flag,
        # #cards, bankrupt.
        for k in range(n):
            p = g.players[(perspective + k) % n]
            parts.extend([
                p.balance / 1500.0,
                p.pos / 39.0,
                1.0 if p.in_jail else 0.0,
                float(len(p.jail_cards)),
                1.0 if p.bankrupt else 0.0,
            ])

        # Per-property block: owner one-hot relative to perspective (slot 0 ==
        # "mine", slot n == unowned), mortgaged, houses.
        for p in self.ownable:
            owner_onehot = [0.0] * (n + 1)
            if p.owner is None:
                owner_onehot[n] = 1.0
            else:
                rel = (g.players.index(p.owner) - perspective) % n
                owner_onehot[rel] = 1.0
            parts.extend(owner_onehot)
            parts.append(1.0 if p.mortgaged else 0.0)
            parts.append(getattr(p, "houses", 0) / 5.0)

        # Phase one-hot and the context property index (BUY/AUCTION/LIQUIDATE),
        # or -1 when no single property is in play.
        phase_onehot = [0.0] * NUM_PHASES
        phase_onehot[phase] = 1.0
        parts.extend(phase_onehot)
        if prop is not None:
            parts.append(self._prop_index[id(prop)] / NUM_OWNABLE)
        else:
            parts.append(-1.0)

        # Trade context (PHASE_TRADE_RESPOND): the tile this player would
        # receive, the tile it would give up, and the net cash it gets, all from
        # its own perspective; -1/-1/0 outside a trade response.
        tc = self._cur_trade
        if phase == PHASE_TRADE_RESPOND and tc is not None:
            recv, give = tc["recv"], tc["give"]
            parts.append(self._prop_index[id(recv)] / NUM_OWNABLE
                         if recv is not None else -1.0)
            parts.append(self._prop_index[id(give)] / NUM_OWNABLE
                         if give is not None else -1.0)
            parts.append(tc["cash"] / 1500.0)
        else:
            parts.extend([-1.0, -1.0, 0.0])

        # Set-awareness. Two context flags for the property in play: does
        # acquiring it complete *my* set, and is it an opponent's last-missing
        # tile (completing it would finish *their* set)?
        me = g.players[perspective]
        opponents = [g.players[(perspective + k) % n] for k in range(1, n)]
        parts.append(1.0 if prop is not None
                     and self._completes_monopoly_for(me, prop) else 0.0)
        parts.append(1.0 if prop is not None and any(
            not o.bankrupt and self._completes_monopoly_for(o, prop)
            for o in opponents) else 0.0)

        # Per-group progress: how much of each monopoly group I own, and the
        # most any single opponent owns -- general set-awareness for bidding,
        # building, and trading.
        for grp in self._groups:
            size = len(grp)
            parts.append(sum(1 for t in grp if t.owner is me) / size)
            parts.append(max((sum(1 for t in grp if t.owner is o)
                              for o in opponents), default=0) / size)

        # Economics of the property in play (the tile being bought / auctioned /
        # traded for): its price, a nominal rent, and how often it is landed on
        # relative to an average tile (1.0 == even). Together these let the agent
        # weigh traffic against cost and rent when valuing an acquisition; 0s
        # when no single property is in play.
        econ = prop
        if econ is None and phase == PHASE_TRADE_RESPOND and tc is not None:
            econ = tc.get("recv")
        if econ is not None:
            uniform = 1.0 / self._board_size
            parts.append(econ.price / 400.0)
            parts.append(base_rent(econ) / 50.0)
            parts.append(self._land_freq.get(econ.pos, uniform) * self._board_size)
        else:
            parts.extend([0.0, 0.0, 0.0])

        return np.asarray(parts, dtype=np.float32)

    def _net_worth(self, player):
        """Net worth used for reward shaping.

        An owned, unmortgaged property is valued at ``price * (1 +
        ACQUISITION_PREMIUM)``: above the cash price, reflecting the rent it
        earns, so *buying* a property is a small net gain (cash ``-price`` but
        property ``+price*(1+premium)``) rather than net-worth-neutral. Without
        that premium buying scores reward 0 and the policy collapses to "always
        decline".

        A mortgaged property is valued at ``price - unmortgage_cost`` (its worth
        once the outstanding mortgage debt is cleared). Combined with the
        ``mortgage_value`` cash a mortgage pays out, this makes forced
        mortgaging during liquidation a clear net-worth loss -- the agent only
        does it when it genuinely needs the cash.
        """
        total = float(player.balance)
        for prop in player.properties:
            total += (prop.price - prop.unmortgage_cost) if prop.mortgaged \
                else prop.price * (1.0 + self.ACQUISITION_PREMIUM)
            if isinstance(prop, StreetProperty):
                # Value houses above cost (same premium as properties): they
                # multiply rent, so *building* should be a small net gain, not
                # the net-worth loss that valuing them at half-cost produced --
                # which would teach the agent never to build.
                total += (prop.houses * prop.house_cost()
                          * (1.0 + self.ACQUISITION_PREMIUM))
        # Reward holding *complete* sets: each fully-owned group adds a bonus,
        # so the tile that finishes a monopoly is a big net-worth jump.
        total += self.MONOPOLY_BONUS * self._owned_monopoly_value(player)
        return total

    def _owned_monopoly_value(self, player):
        """Total list price of the monopoly groups ``player`` fully owns."""
        total = 0.0
        for tiles in self._groups:
            if all(t.owner is player for t in tiles):
                total += sum(t.price for t in tiles)
        return total

    def _decisive_winner(self):
        """The winner even when no one was bankrupted (a turn-cap timeout).

        A real end has a sole survivor; on a timeout with several survivors the
        richest (by shaped net worth) is declared the winner. This makes every
        episode conclusive -- without trading and with the GO salary, games
        often reach the turn cap with everyone still solvent, which otherwise
        leaves the agent with no win/loss signal on those games.
        """
        survivors = self.game.active_players()
        if not survivors:
            return None
        if len(survivors) == 1:
            return survivors[0]
        return max(survivors, key=self._net_worth)

    def _terminal_reward(self):
        """Decisive end-of-game reward in ``[-1, +1]``.

        Bankruptcy is the worst outcome (-1); a sole survivor wins outright
        (+1). On a timeout the agent is ranked among the surviving players by
        net worth and mapped linearly to ``[-1, +1]`` (leader +1, worst survivor
        -1), so being ahead at the cap is rewarded and every game is decisive.
        Ties in net worth count as neither ahead nor behind.
        """
        controlled = self.game.players[self.seat]
        if controlled.bankrupt:
            return -1.0
        survivors = self.game.active_players()
        if len(survivors) <= 1:
            return 1.0  # sole survivor: outright win
        my_nw = self._net_worth(controlled)
        others = [p for p in survivors if p is not controlled]
        beat = sum(self._net_worth(p) < my_nw for p in others)
        lost = sum(self._net_worth(p) > my_nw for p in others)
        return (beat - lost) / (len(survivors) - 1)

    def _reward(self, terminal):
        controlled = self.game.players[self.seat]
        reward = 0.0
        if self.reward_mode == "shaped":
            # Shape on *relative* advantage (my net worth minus the mean
            # opponent's), not absolute net worth. This way an opponent
            # completing a set raises the baseline and costs me reward -- so the
            # agent is pushed to spend to block it -- while finishing my own set
            # still pays. It telescopes over the episode, so the decisive
            # terminal reward is unaffected.
            adv = self._net_worth(controlled) - self._mean_opp_networth()
            reward += (adv - self._prev_advantage) / 1000.0
            self._prev_advantage = adv
            # One-time bonus for snatching an opponent's last-missing tile
            # (denying their monopoly); credited by the on_acquire hook.
            reward += self._pending_bonus
            self._pending_bonus = 0.0
        if terminal:
            reward += self._terminal_reward()
        return reward

    def _mean_opp_networth(self):
        """Mean net worth of the controlled seat's non-bankrupt opponents (0 if
        none remain), the baseline for the relative shaped reward."""
        controlled = self.game.players[self.seat]
        others = [self._net_worth(p) for p in self.game.players
                  if p is not controlled and not p.bankrupt]
        return sum(others) / len(others) if others else 0.0

    def _info(self, terminal=False):
        info = {
            "action_mask": self._cur_mask.copy(),
            "phase": self._cur_phase,
            "current_player": self.game.current_player,
            "illegal": getattr(self, "_illegal", False),
        }
        if terminal:
            winner = self._decisive_winner()
            info["winner"] = winner.name if winner is not None else None
            info["won"] = winner is self.game.players[self.seat]
            info["timeout"] = self._truncated
        return info

    # -- Teardown -----------------------------------------------------------
    def _shutdown_worker(self):
        """Aborts a running worker thread and drains the channels."""
        if self._thread is not None and self._thread.is_alive():
            self._resp_q.put(_ABORT)
            self._thread.join(timeout=5.0)
        self._thread = None
        self._done = True
        # Drain any stale items so the next episode starts clean.
        for q in (self._req_q, self._resp_q):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
