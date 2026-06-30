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

Action space (``Discrete(146)``)
--------------------------------
A single flat action id, interpreted against the per-step ``action_mask`` in
``info`` (and :meth:`MonopolyEnv.legal_actions`). Property-targeted actions are
indexed by the 28 ownable tiles in board order::

    0  PAY_JAIL          pay the $50 fine to leave jail
    1  USE_JAIL_CARD     spend a Get Out of Jail Free card
    2  ROLL_JAIL         roll for doubles to leave jail
    3  BUY               buy the property just landed on
    4  DECLINE           decline the offered property
    5  END_MANAGE        finish managing / stop liquidating
    6  +i  BUILD i       build a house/hotel on ownable tile i
    34 +i  SELL i        sell a house/hotel from tile i
    62 +i  MORTGAGE i    mortgage tile i
    90 +i  UNMORTGAGE i  lift the mortgage on tile i
    118+i  TRADE i       (disabled) the agent does not trade -- these actions
                         are never legal, so the agent never initiates a trade.
                         The slots are retained only to keep the action-space
                         size (and thus the policy head) stable; trade decisions
                         in the GUI are handled by a separate valuation formula.

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
NUM_PHASES = 5

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
A_TRADE = 118        # A_TRADE + i
NUM_OWNABLE = 28
NUM_ACTIONS = A_TRADE + NUM_OWNABLE  # 146

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
            terminal win/loss bonus) or ``"sparse"`` (only the terminal +1/-1).
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

        self.action_space = Discrete(NUM_ACTIONS)
        obs_len = (5 * num_players
                   + NUM_OWNABLE * (num_players + 3)
                   + NUM_PHASES + 1)
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
        self._cur_prop = None      # property offered (BUY) or being liquidated
        self._cur_amount = 0       # shortfall amount (LIQUIDATE)
        self._prev_networth = 0.0
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

        self._done = False
        self._truncated = False
        self._prev_networth = self._net_worth(controlled)
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
            self.game.players[s].decide_purchase = self._make_buy_hook(decide)

    def _make_policy_decider(self, seat):
        """Returns a decider bound to ``seat`` that calls the opponent policy."""
        return lambda phase, prop=None, amount=0: self._policy_decide(
            seat, phase, prop, amount)

    def _make_buy_hook(self, decide):
        """Returns a ``decide_purchase(prop)`` hook routed through ``decide``."""
        return lambda prop: decide(PHASE_BUY, prop) == A_BUY

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
        # A_TRADE actions are never legal (the agent does not trade), so they
        # never reach here.

    def _has_liquidation_options(self, player):
        """Whether ``player`` has any house to sell or property to mortgage."""
        g = self.game
        for prop in player.properties:
            if isinstance(prop, StreetProperty) and prop.can_sell_house(g, player):
                return True
            if prop.can_mortgage(g, player):
                return True
        return False

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
            if prop is not None and player.balance >= prop.price:
                mask[A_BUY] = 1
            return mask

        # MANAGE and LIQUIDATE share the property-action masks.
        for i, p in enumerate(self.ownable):
            if p.owner is not player:
                continue
            if isinstance(p, StreetProperty):
                if phase == PHASE_MANAGE and p.can_build_house(g, player):
                    mask[A_BUILD + i] = 1
                if p.can_sell_house(g, player):
                    mask[A_SELL + i] = 1
            if p.can_mortgage(g, player):
                mask[A_MORTGAGE + i] = 1
            if phase == PHASE_MANAGE:
                if p.can_unmortgage(g, player):
                    mask[A_UNMORTGAGE + i] = 1
                # No A_TRADE bit: the agent never initiates trades.

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

        # Phase one-hot and the context property index (BUY/LIQUIDATE), or -1.
        phase_onehot = [0.0] * NUM_PHASES
        phase_onehot[phase] = 1.0
        parts.extend(phase_onehot)
        if prop is not None:
            parts.append(self._prop_index[id(prop)] / NUM_OWNABLE)
        else:
            parts.append(-1.0)

        return np.asarray(parts, dtype=np.float32)

    def _net_worth(self, player):
        """Liquidation-style net worth used for reward shaping.

        A mortgaged property is valued at ``price - unmortgage_cost`` (its worth
        once the outstanding mortgage debt is cleared) rather than at its
        mortgage value. Because mortgaging also pays the owner ``mortgage_value``
        in cash, this makes the round trip cost exactly the ~10% mortgage
        interest -- so the agent sees a real (if small) net-worth penalty for
        mortgage-flipping a property it just bought, instead of mortgaging being
        net-worth-neutral and therefore "free".
        """
        total = float(player.balance)
        for prop in player.properties:
            total += (prop.price - prop.unmortgage_cost) if prop.mortgaged \
                else prop.price
            if isinstance(prop, StreetProperty):
                total += prop.houses * (prop.house_cost() // 2)
        return total

    def _reward(self, terminal):
        controlled = self.game.players[self.seat]
        reward = 0.0
        if self.reward_mode == "shaped":
            nw = self._net_worth(controlled)
            reward += (nw - self._prev_networth) / 1000.0
            self._prev_networth = nw
        if terminal:
            if self.game.winner() is controlled:
                reward += 1.0
            elif controlled.bankrupt:
                reward -= 1.0
        return reward

    def _info(self, terminal=False):
        info = {
            "action_mask": self._cur_mask.copy(),
            "phase": self._cur_phase,
            "current_player": self.game.current_player,
            "illegal": getattr(self, "_illegal", False),
        }
        if terminal:
            winner = self.game.winner()
            info["winner"] = winner.name if winner is not None else None
            info["won"] = winner is self.game.players[self.seat]
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
