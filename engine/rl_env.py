"""Gym-style reinforcement-learning environment for the Monopoly engine.

``MonopolyEnv`` wraps :class:`engine.game.Game` and exposes the standard
Gymnasium API (``reset`` / ``step``) so an RL agent can play one seat against
baseline opponents. It is a *decision-point* environment: each call to ``step``
supplies a single action for the one decision the controlled player currently
faces (escape jail, buy a property, a property-management action, or raising
cash under a shortfall). Between the controlled player's decisions the
environment plays out everything that needs no choice from them -- dice, rent,
cards, and the opponents' entire turns -- automatically.

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
    118+i  TRADE i       sell tile i to a willing opponent at list price

Illegal actions (those not set in the mask) are clamped to a safe default rather
than raising, and reported via ``info["illegal"]``.
"""

import queue
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
except ImportError:  # pragma: no cover - exercised only without gymnasium
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
    """

    metadata = {"render_modes": []}

    def __init__(self, seat=0, num_players=4, names=None, max_turns=1000,
                 reward_mode="shaped", seed=None):
        if not 0 <= seat < num_players:
            raise ValueError("seat must be in range(num_players)")
        self.seat = seat
        self.num_players = num_players
        self._names = names or ["Red", "Blue", "Green", "Yellow"][:num_players]
        if len(self._names) != num_players:
            raise ValueError("names must have num_players entries")
        self.max_turns = max_turns
        self.reward_mode = reward_mode
        self._seed = seed

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
        self._shutdown_worker()
        if seed is not None:
            self._seed = seed
        self._build_game()

        controlled = self.game.players[self.seat]
        # Only the controlled player's decisions are routed to the agent; the
        # opponents keep the engine's baseline policy (buy-if-affordable, roll
        # in jail, no liquidation), exactly as in headless play.
        controlled.decide_purchase = self._on_decide_purchase
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

    # -- Game construction --------------------------------------------------
    def _build_game(self):
        players = [Player(name) for name in self._names]
        board = Board(build_board_tiles())
        game = Game(players, board, build_chance_deck(), build_community_deck())
        if self._seed is not None:
            # Game.roll_dice draws from the stdlib random module.
            import random
            random.seed(self._seed)
        self.game = game
        self.ownable = [
            t for t in board.tiles
            if isinstance(t, (StreetProperty, Railroad, Utility))
        ]
        self._prop_index = {id(p): i for i, p in enumerate(self.ownable)}

    # -- Worker thread: the real game loop ----------------------------------
    def _run_game(self):
        """Runs the full game on the worker thread until the episode ends.

        Opponent turns use the engine's own ``step`` (baseline policy); the
        controlled player's turn is driven here so the agent can be consulted at
        each decision point. Exits when the controlled player is out, the game
        is over, or the turn cap is hit, then signals the env.
        """
        g = self.game
        controlled = g.players[self.seat]
        turns = 0
        try:
            while (not g.is_over() and not controlled.bankrupt
                   and turns < self.max_turns):
                if g.current_player == self.seat:
                    self._controlled_turn(controlled)
                else:
                    g.step()  # baseline opponent, whole turn
                turns += 1
            self._truncated = turns >= self.max_turns and not g.is_over()
        except _Abort:
            return
        # Signal the end of the episode to whoever is waiting in step().
        self._req_q.put(_TERMINAL)

    def _controlled_turn(self, player):
        """Plays one turn for the controlled player, mirroring ``Game.step``.

        Decision points (jail action, pre/post-roll management, and the nested
        purchase / shortfall hooks) block for an agent action. The turn is
        advanced exactly once on every exit path, matching the engine.
        """
        g = self.game

        if player.in_jail:
            choice = self._jail_choice(player)
            result = g.handle_jail_turn(player, choice)
            if result in ("jailed", "freed"):
                g.advance_turn()
                return
            if result == "moved":
                g.resolve_tile(player)
                g.advance_turn()
                return
            # "released": paid / used a card, now take a normal turn.

        self._manage_phase(player)  # pre-roll

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

        self._manage_phase(player)  # post-roll
        player.double_count = 0
        g.advance_turn()

    def _manage_phase(self, player):
        """Lets the agent issue management actions until it chooses to stop."""
        while True:
            action = self._ask(PHASE_MANAGE)
            if action == A_END_MANAGE:
                return
            self._apply_manage_action(player, action)
            if player.bankrupt:
                return

    # -- Decision hooks (called on the worker thread) -----------------------
    def _jail_choice(self, player):
        action = self._ask(PHASE_JAIL)
        if action == A_PAY_JAIL:
            return "pay"
        if action == A_USE_CARD:
            return "card"
        return "roll"

    def _on_decide_purchase(self, prop):
        """``Player.decide_purchase`` override for the controlled player."""
        action = self._ask(PHASE_BUY, prop=prop)
        return action == A_BUY

    def _on_shortfall(self, payer, amount):
        """``Game.on_shortfall`` hook: only the agent gets to liquidate.

        Opponents have no shortfall hook behaviour (they fall through to
        bankruptcy as in headless play). The agent may sell houses, mortgage,
        or trade until it covers the debt or gives up.
        """
        if payer is not self.game.players[self.seat]:
            return
        while payer.balance < amount:
            if not self._has_liquidation_options(payer):
                return  # nothing left to raise; bankruptcy will follow
            action = self._ask(PHASE_LIQUIDATE, prop=None, amount=amount)
            if action == A_END_MANAGE:
                return  # agent chose to stop; bankruptcy will follow
            self._apply_manage_action(payer, action)

    def _ask(self, phase, prop=None, amount=0):
        """Hands a decision to the env and blocks for the agent's action.

        Posts the request, waits for ``step`` to push a response, then validates
        it against the legal mask -- clamping an illegal action to a safe
        default (decline / roll / stop) rather than letting it corrupt state.
        """
        self._cur_phase = phase
        self._cur_prop = prop
        self._cur_amount = amount
        self._cur_mask = self._legal_mask(phase, prop)
        self._req_q.put((phase, prop, amount))
        action = self._resp_q.get()
        if action == _ABORT:
            raise _Abort()
        self._illegal = not (0 <= action < NUM_ACTIONS and self._cur_mask[action])
        if self._illegal:
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
        elif A_TRADE <= action < A_TRADE + NUM_OWNABLE:
            self._do_trade(player, self.ownable[action - A_TRADE])

    def _do_trade(self, seller, prop):
        """Sells ``prop`` to a willing opponent at its list price.

        A simplified trade primitive: the agent puts one property on the market
        and the engine finds an opponent who benefits and can pay list price.
        Cash flows to the seller (``cash = -price`` in ``execute_trade``'s
        initiator-pays-partner convention). Acquiring properties or negotiating
        price is left as a future extension.
        """
        buyer = self._find_trade_buyer(seller, prop)
        if buyer is not None:
            self.game.execute_trade(seller, buyer, [prop], [], -prop.price)

    def _find_trade_buyer(self, seller, prop):
        """Returns the opponent most likely to buy ``prop``, or ``None``.

        A buyer must be solvent enough to pay list price and gain something:
        completing a colour group / set is preferred over merely extending one.
        Returns ``None`` when no opponent benefits, which also masks the trade
        action out.
        """
        g = self.game
        if not g.can_trade_property(prop):
            return None
        best, best_score = None, 0
        for other in g.players:
            if other is seller or other.bankrupt or other.balance < prop.price:
                continue
            score = self._trade_appeal(prop, other)
            if score > best_score:
                best, best_score = other, score
        return best

    def _trade_appeal(self, prop, buyer):
        """Heuristic appeal of ``prop`` to ``buyer`` (0 = no interest)."""
        if isinstance(prop, StreetProperty):
            group = prop.color_group(self.game)
            owned = sum(1 for t in group if t.owner is buyer)
            if owned == 0:
                return 0
            # Completing the set is worth far more than extending it.
            return 10 if owned == len(group) - 1 else owned
        same_type = sum(
            1 for p in buyer.properties if isinstance(p, type(prop)))
        return same_type + 1  # railroads/utilities: always some scale value

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
    def _legal_mask(self, phase, prop):
        mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        g = self.game
        player = g.players[self.seat]

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
                if self._find_trade_buyer(player, p) is not None:
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
            obs = self._encode_obs()
            reward = self._reward(terminal=True)
            return obs, reward, True, self._truncated, self._info(terminal=True)

        phase, prop, amount = item
        self._cur_phase, self._cur_prop, self._cur_amount = phase, prop, amount
        obs = self._encode_obs()
        reward = self._reward(terminal=False)
        return obs, reward, False, False, self._info()

    def _encode_obs(self):
        """Builds the flat float32 observation for the current state."""
        g = self.game
        n = self.num_players
        parts = []

        # Per-player block: balance, position, jail flag, #cards, bankrupt.
        for p in g.players:
            parts.extend([
                p.balance / 1500.0,
                p.pos / 39.0,
                1.0 if p.in_jail else 0.0,
                float(len(p.jail_cards)),
                1.0 if p.bankrupt else 0.0,
            ])

        # Per-property block: owner one-hot (incl. unowned), mortgaged, houses.
        for p in self.ownable:
            owner_onehot = [0.0] * (n + 1)
            if p.owner is None:
                owner_onehot[n] = 1.0
            else:
                owner_onehot[g.players.index(p.owner)] = 1.0
            parts.extend(owner_onehot)
            parts.append(1.0 if p.mortgaged else 0.0)
            parts.append(getattr(p, "houses", 0) / 5.0)

        # Phase one-hot and the context property index (BUY/LIQUIDATE), or -1.
        phase_onehot = [0.0] * NUM_PHASES
        phase_onehot[self._cur_phase] = 1.0
        parts.extend(phase_onehot)
        if self._cur_prop is not None:
            parts.append(self._prop_index[id(self._cur_prop)] / NUM_OWNABLE)
        else:
            parts.append(-1.0)

        return np.asarray(parts, dtype=np.float32)

    def _net_worth(self, player):
        """Liquidation-style net worth used for reward shaping."""
        total = float(player.balance)
        for prop in player.properties:
            total += prop.mortgage_value if prop.mortgaged else prop.price
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
