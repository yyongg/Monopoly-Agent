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

Action space (``Discrete(211)``)
--------------------------------
A single flat action id, interpreted against the per-step ``action_mask`` in
``info`` (and :meth:`MonopolyEnv.legal_actions`). Property-targeted actions are
indexed by the 28 ownable tiles in board order::

    0  PAY_JAIL              pay the $50 fine to leave jail
    1  USE_JAIL_CARD         spend a Get Out of Jail Free card
    2  ROLL_JAIL             roll for doubles to leave jail
    3  BUY                   buy the property just landed on
    4  DECLINE               decline the offered property (goes to auction)
    5  END_MANAGE            finish managing / stop liquidating
    6  +i   BUILD i          build a house/hotel on ownable tile i
    34 +i   SELL i           sell a house/hotel from tile i
    62 +i   MORTGAGE i       mortgage tile i
    90 +i   UNMORTGAGE i     lift the mortgage on tile i
    118+3i+t TRADE i,t       propose to acquire tile i from its owner at cash
                             tier t in {0,1,2} (0.75/1.0/1.25x the balancing
                             cash). Legal when acquiring i completes the agent's
                             monopoly OR denies the owner one they are cornering;
                             the engine builds the give + tiered cash offer.
    202    TRADE_ACCEPT      accept a trade another player has offered me
    203    TRADE_REJECT      reject a trade another player has offered me
    204    AUCTION_PASS      bid nothing in the current auction
    205+k  AUCTION_BID k     bid BID_FRACTIONS[k] * value for the auctioned tile

Illegal actions (those not set in the mask) are clamped to a safe default rather
than raising, and reported via ``info["illegal"]``.
"""

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

# Structural ids, tuning knobs, the observation encoder, and reward shaping live
# in dedicated modules; re-export the public names so existing
# ``from engine.rl_env import PHASE_BUY, MonopolyEnv, ...`` imports keep working.
from engine.constants import (
    PHASE_JAIL, PHASE_BUY, PHASE_MANAGE, PHASE_LIQUIDATE, PHASE_TERMINAL,
    PHASE_AUCTION, PHASE_TRADE_RESPOND, NUM_PHASES,
    A_PAY_JAIL, A_USE_CARD, A_ROLL_JAIL, A_BUY, A_DECLINE, A_END_MANAGE,
    A_BUILD, A_SELL, A_MORTGAGE, A_UNMORTGAGE, A_TRADE,
    A_TRADE_ACCEPT, A_TRADE_REJECT, A_AUCTION_PASS, A_AUCTION_BID,
    NUM_OWNABLE, NUM_GROUPS, NUM_BID_LEVELS, NUM_ACTIONS, NUM_TRADE_TIERS,
    TRADE_CASH_TIERS, BID_FRACTIONS, BID_CEILING_MULT, decode_trade_action,
)
from engine.config import (
    RewardConfig, DEFAULT_REWARD_CONFIG,
    ACQUISITION_PREMIUM, MONOPOLY_BONUS, DENIAL_VALUE_WEIGHT, TRADE_INCOME_WEIGHT,
    DENIAL_BONUS_COEF, ACQUISITION_BONUS_COEF, AUCTION_ACQUISITION_BONUS_COEF,
    BUY_PREFERENCE_COEF, BUILD_BONUS_COEF, BUILD_ROI_REF_HOUSE_COST,
    SET_ROI_REF, SET_QUALITY_CLAMP, PROFIT_SCALE, SET_BONUS, KEEP_PREMIUM,
)
from engine.observation import (
    ObsEncoder, observation_length, load_landing_frequencies, base_rent,
    build_groups, safe_default, LANDING_FREQ_PATH,
)
from engine.rewards import RewardMixin

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


# Sentinels passed between the worker thread and the env.
_TERMINAL = "__terminal__"
_ABORT = "__abort__"


class _Abort(Exception):
    """Raised inside the worker thread to tear it down on ``reset``/``close``."""


class MonopolyEnv(RewardMixin, _GymEnv):
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
        opponent_policy (callable | object | list | dict | None): The opponent
            spec driving the non-agent seats. It may be a plain
            ``(observation, action_mask_bool) -> action_index`` callable (a
            network policy, applied to every opponent seat), a **state-aware**
            object with a ``decide(seat, phase, prop, amount, mask)`` method (and
            optional ``bind(game, ownable)``) such as a hand-crafted baseline
            agent, a **list/tuple** of policies dealt round-robin across the
            opponent seats (heterogeneous opponents, e.g. an FP-A/B/C trio), or a
            **dict** ``{seat: policy}`` for explicit per-seat control. ``None``
            keeps the engine baseline (opponents play via ``Game.step``).
            Network-policy observations are from that opponent's own perspective
            (it always sees itself as relative player 0), so one policy plays any
            seat.
        opponent_provider (callable | None): A zero-arg callable sampled at each
            ``reset`` that returns an opponent spec (any form accepted by
            ``opponent_policy``, or ``None`` for baseline). Used for self-play,
            where opponents are sampled from a snapshot pool each episode. Takes
            precedence over ``opponent_policy``.
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
        self._opponent_policies = {}  # resolved per opponent seat, each episode
        self._deciders = {}

        # The 28 ownable tiles, fixed by board order, give every property a
        # stable action/observation index for the agent.
        self._ownable_template = [
            t for t in build_board_tiles()
            if isinstance(t, (StreetProperty, Railroad, Utility))
        ]
        assert len(self._ownable_template) == NUM_OWNABLE
        assert len(build_groups(self._ownable_template)) == NUM_GROUPS

        self.action_space = Discrete(NUM_ACTIONS)
        obs_len = observation_length(num_players)
        self.observation_space = Box(
            low=-1.0, high=np.inf, shape=(obs_len,), dtype=np.float32)
        self._obs_len = obs_len
        self.cfg = RewardConfig()
        self.encoder = ObsEncoder(self.cfg)

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
        self.encoder._cur_trade = None     # offer being judged (PHASE_TRADE_RESPOND)
        self.encoder._traded_this_manage = set()  # trade targets tried this MANAGE phase
        self._prev_advantage = 0.0  # last relative-advantage potential (shaping)
        self._pending_bonus = 0.0   # denial reward queued by the on_acquire hook
        self._turn = 0              # turns played this episode (tempo shaping)
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

        # Pick this episode's opponent spec (a provider, if any, is sampled
        # fresh each episode for self-play; otherwise the fixed policy) and
        # resolve it to a per-opponent-seat policy map.
        chosen = (self._opponent_provider() if self._opponent_provider is not None
                  else self._opponent_policy_fixed)
        self._opponent_policies = self._resolve_opponents(chosen)
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
    def _resolve_opponents(self, chosen):
        """Maps the sampled opponent spec to a ``{opponent_seat: policy}`` dict.

        ``chosen`` may be:

        * ``None`` -- engine baseline on every opponent seat (empty map);
        * a single policy -- a plain ``(obs, mask) -> action`` callable or a
          state-aware object (see :meth:`_policy_decide`) -- applied to *every*
          opponent seat;
        * a list/tuple -- a pool dealt round-robin across the opponent seats
          (so ``[FP_A, FP_B, FP_C]`` fills the three non-agent seats regardless
          of which seat the agent drew);
        * a dict ``{absolute_seat: policy}`` -- explicit per-seat control; a seat
          mapped to ``None`` (or omitted) keeps the baseline.

        The agent's own seat is always skipped.
        """
        seats = [s for s in range(self.num_players) if s != self.seat]
        if chosen is None:
            return {}
        if isinstance(chosen, dict):
            return {s: chosen[s] for s in seats
                    if chosen.get(s) is not None}
        if isinstance(chosen, (list, tuple)):
            return {s: chosen[k % len(chosen)] for k, s in enumerate(seats)
                    if chosen[k % len(chosen)] is not None}
        return {s: chosen for s in seats}

    def _wire_deciders(self):
        """Builds the seat -> decider map for the current episode.

        The agent seat is always routed to :meth:`_agent_decide` (which blocks
        for an external action via the queue). Each opponent seat that has a
        resolved policy is routed to :meth:`_policy_decide` (synchronous, no
        queue); a state-aware opponent is ``bind``-ed to the fresh game first.
        Each routed seat also gets its ``decide_purchase`` / ``decide_bid`` hooks
        wired so nested purchase / auction offers reach the same decider. Seats
        with no policy keep the engine baseline.
        """
        self._deciders = {self.seat: self._agent_decide}
        for s, policy in self._opponent_policies.items():
            if hasattr(policy, "bind"):
                policy.bind(self.game, self.ownable)
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
                if not self.encoder._has_liquidation_options(player):
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
                value = self.encoder._bid_value(player, prop)
                ceiling = min(int(round(BID_FRACTIONS[k] * value)),
                              int(round(BID_CEILING_MULT * prop.price)),
                              player.balance)
                if min_bid <= 0:
                    return ceiling
                return min_bid if ceiling >= min_bid else 0
            return 0  # A_AUCTION_PASS or anything unexpected

        return hook

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
        # Bind the observation encoder (obs, valuations, masks) to this game.
        self.encoder.bind(game, self.ownable)

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
                self._turn = turns  # current turn, read by tempo shaping
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
        self.encoder._traded_this_manage = set()
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
            if not self.encoder._has_liquidation_options(payer):
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
        self._cur_mask = self.encoder._legal_mask(phase, prop, self.seat)
        self._req_q.put((phase, prop, amount))
        action = self._resp_q.get()
        if action == _ABORT:
            raise _Abort()
        self._illegal = not (0 <= action < NUM_ACTIONS and self._cur_mask[action])
        if self._illegal:
            action = self._safe_default(phase)
        return action

    def _policy_decide(self, seat, phase, prop=None, amount=0):
        """An opponent's decider: queries that seat's policy synchronously.

        Two kinds of policy are supported and dispatched by duck typing:

        * a **state-aware** opponent (has ``.decide``) is handed the full
          decision context ``decide(seat, phase, prop, amount, mask)`` -- used by
          the hand-crafted baseline agents, which read live game state;
        * a plain **network policy** ``(obs, mask) -> action`` is handed the
          encoded observation from ``seat``'s perspective (self-play snapshots).

        Either way an illegal/invalid action is clamped to a safe default.
        """
        policy = self._opponent_policies[seat]
        mask = self.encoder._legal_mask(phase, prop, seat)
        if hasattr(policy, "decide"):
            action = int(policy.decide(seat, phase, prop, amount,
                                       mask.astype(bool),
                                       offer=self.encoder._cur_trade))
        else:
            obs = self.encoder._encode_obs(seat, phase, prop)
            action = int(policy(obs, mask.astype(bool)))
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
            prop = self.ownable[action - A_BUILD]
            before = prop.houses
            g.build_house(prop, player)
            # Reward the controlled agent for actually placing a house (the
            # engine re-validates, so a masked-through build may be a no-op).
            if prop.houses > before and player is self.game.players[self.seat]:
                self._pending_bonus += self._build_bonus(prop)
        elif A_SELL <= action < A_SELL + NUM_OWNABLE:
            g.sell_house(self.ownable[action - A_SELL], player)
        elif A_MORTGAGE <= action < A_MORTGAGE + NUM_OWNABLE:
            g.mortgage_property(self.ownable[action - A_MORTGAGE], player)
        elif A_UNMORTGAGE <= action < A_UNMORTGAGE + NUM_OWNABLE:
            g.unmortgage_property(self.ownable[action - A_UNMORTGAGE], player)
        elif A_TRADE <= action < A_TRADE + NUM_OWNABLE * NUM_TRADE_TIERS:
            i, tier = decode_trade_action(action)
            self.encoder._traded_this_manage.add(i)
            self._attempt_trade(player, self.ownable[i], tier)

    def _attempt_trade(self, initiator, target, tier=1):
        """Builds and offers a trade to acquire ``target`` at cash ``tier``.

        The engine constructs the give-tile + a balancing cash figure, scaled by
        ``TRADE_CASH_TIERS[tier]`` (lowball / fair / generous). The partner's
        decider (agent via the queue, opponent policy synchronously, or a
        baseline via :meth:`_formula_trade_ok`) accepts or rejects. On acceptance
        the swap runs through :meth:`Game.execute_trade`.
        """
        g = self.game
        partner = target.owner
        if not self.encoder._can_propose_trade(initiator, target):
            return
        give = self.encoder._choose_give_tile(initiator, partner, target)
        if give is None:
            return
        receive = [target]
        balancing = self.encoder._balancing_cash(initiator, partner, [give], receive)
        raw = int(round(balancing * TRADE_CASH_TIERS[tier]))
        # Positive: we pay the partner (clamp to our balance). Negative: we
        # request cash in a mutual set-for-set, so the partner pays -- clamp the
        # request to *their* balance so the swap can actually settle.
        if raw >= 0:
            cash = min(raw, initiator.balance)
        else:
            cash = -min(-raw, partner.balance)

        partner_seat = g.players.index(partner)
        decide = self._deciders.get(partner_seat)
        if decide is not None:
            # From the partner's view they receive ``give`` plus ``cash`` and
            # part with ``target``; expose that as the response observation.
            self.encoder._cur_trade = {"recv": give, "give": target, "cash": cash}
            action = decide(PHASE_TRADE_RESPOND, target, cash)
            self.encoder._cur_trade = None
            accept = action == A_TRADE_ACCEPT
        else:
            accept = self.encoder._formula_trade_ok(partner, [give], receive, cash)

        if accept:
            g.execute_trade(initiator, partner, [give], receive, cash)

    # -- Legal-action masks -------------------------------------------------
    # -- Observation & reward ----------------------------------------------
    def _pump_until_request(self):
        """Waits for the next decision request (or the terminal signal)."""
        item = self._req_q.get()
        if item == _TERMINAL:
            self._done = True
            self._cur_phase = PHASE_TERMINAL
            self._cur_mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
            self._cur_prop = None
            obs = self.encoder._encode_obs(self.seat, PHASE_TERMINAL, None)
            reward = self._reward(terminal=True)
            return obs, reward, True, self._truncated, self._info(terminal=True)

        phase, prop, amount = item
        self._cur_phase, self._cur_prop, self._cur_amount = phase, prop, amount
        obs = self.encoder._encode_obs(self.seat, phase, prop)
        reward = self._reward(terminal=False)
        return obs, reward, False, False, self._info()

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
