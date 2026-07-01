"""AI player for the GUI: drives decisions using a trained MaskablePPO model.

``GUIAIDecider`` wraps a model and exposes simple per-decision methods
(``jail_choice``, ``purchase_decision``, ``manage_loop``, ``liquidate_loop``)
that the GUI can call instead of showing a human prompt. The obs encoding and
legal-mask logic mirror ``MonopolyEnv`` exactly so the model sees the same
inputs it was trained on.

The AI never *initiates* trades (trade actions are masked out, matching the
trained policy). When a human proposes a trade to the AI, ``evaluate_trade``
scores the offer with a dollar-valued formula and accepts only if it comes out
ahead -- see that method for the formula.
"""

import numpy as np

from engine.rl_env import (
    PHASE_JAIL, PHASE_BUY, PHASE_MANAGE, PHASE_LIQUIDATE,
    NUM_PHASES, NUM_ACTIONS, NUM_OWNABLE,
    A_PAY_JAIL, A_USE_CARD, A_ROLL_JAIL,
    A_BUY, A_DECLINE, A_END_MANAGE,
    A_BUILD, A_SELL, A_MORTGAGE, A_UNMORTGAGE,
)
from models.tiles.properties.street_property import StreetProperty
from models.tiles.properties.railroad import Railroad
from models.tiles.properties.utility import Utility


class GUIAIDecider:
    """Drives one AI player's decisions in the GUI using a trained model.

    Call ``bind(game, ownable)`` once after the board is built, then use the
    public methods at each decision point during the turn loop.
    """

    def __init__(self, num_players, model, deterministic=False, log=None):
        self.num_players = num_players
        self.model = model
        self.deterministic = deterministic
        # Callback used to report each AI move to the GUI game log. Defaults to
        # a no-op so the decider also works headless; ``MonopolyApp`` sets it to
        # ``add_log`` so the player can see what the AI did.
        self.log = log or (lambda message: None)
        self.game = None
        self.ownable = []
        self._prop_index = {}

    def bind(self, game, ownable):
        """Attach to a live game instance. Must be called before any decisions."""
        self.game = game
        self.ownable = ownable
        self._prop_index = {id(p): i for i, p in enumerate(ownable)}

    # --- Public decision methods ---

    def jail_choice(self, player):
        """Returns "pay", "card", or "roll" for a jailed player."""
        seat = self.game.players.index(player)
        action = self._decide(seat, PHASE_JAIL)
        if action == A_PAY_JAIL:
            return "pay"
        if action == A_USE_CARD:
            return "card"
        return "roll"

    def purchase_decision(self, player, prop):
        """Returns True to buy ``prop``, False to decline."""
        seat = self.game.players.index(player)
        action = self._decide(seat, PHASE_BUY, prop)
        bought = action == A_BUY
        if bought:
            self.log(f"{player.name} [AI] bought {prop.name} for ${prop.price}.")
        else:
            self.log(f"{player.name} [AI] declined {prop.name}.")
        return bought

    def manage_loop(self, player):
        """Issues management actions until the model chooses END_MANAGE."""
        seat = self.game.players.index(player)
        for _ in range(50):
            action = self._decide(seat, PHASE_MANAGE)
            if action == A_END_MANAGE:
                return
            self._apply_manage_action(player, action)
            if player.bankrupt:
                return

    def liquidate_loop(self, player, amount):
        """Liquidates assets until ``player`` covers ``amount`` or gives up."""
        seat = self.game.players.index(player)
        for _ in range(50):
            if player.balance >= amount:
                return
            if not self._has_liquidation_options(player):
                return
            action = self._decide(seat, PHASE_LIQUIDATE)
            if action == A_END_MANAGE:
                return
            self._apply_manage_action(player, action)

    # --- Internal helpers ---

    def _decide(self, seat, phase, prop=None):
        obs = self._encode_obs(seat, phase, prop)
        mask = self._legal_mask(phase, prop, seat)
        action, _ = self.model.predict(
            obs, action_masks=mask.astype(bool), deterministic=self.deterministic)
        action = int(action)
        if not (0 <= action < NUM_ACTIONS and mask[action]):
            action = _safe_default(phase)
        return action

    def _encode_obs(self, perspective, phase, prop):
        g = self.game
        n = self.num_players
        parts = []
        for k in range(n):
            p = g.players[(perspective + k) % n]
            parts.extend([
                p.balance / 1500.0,
                p.pos / 39.0,
                1.0 if p.in_jail else 0.0,
                float(len(p.jail_cards)),
                1.0 if p.bankrupt else 0.0,
            ])
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
        phase_onehot = [0.0] * NUM_PHASES
        phase_onehot[phase] = 1.0
        parts.extend(phase_onehot)
        if prop is not None and id(prop) in self._prop_index:
            parts.append(self._prop_index[id(prop)] / NUM_OWNABLE)
        else:
            parts.append(-1.0)
        return np.asarray(parts, dtype=np.float32)

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

        for i, p in enumerate(self.ownable):
            if p.owner is not player:
                continue
            if isinstance(p, StreetProperty):
                if phase == PHASE_MANAGE and p.can_build_house(g, player):
                    mask[A_BUILD + i] = 1
                # Selling houses is only offered during forced LIQUIDATE, for
                # the same reason as mortgaging: allowing it during voluntary
                # MANAGE let the AI sell<->rebuild houses on a monopoly.
                if phase == PHASE_LIQUIDATE and p.can_sell_house(g, player):
                    mask[A_SELL + i] = 1
            # Mortgaging is only offered during forced LIQUIDATE (raising cash
            # the player actually needs). Allowing it during voluntary MANAGE
            # let the AI mortgage-flip a property it just bought (and oscillate
            # mortgage<->unmortgage), so it is masked out there.
            if phase == PHASE_LIQUIDATE and p.can_mortgage(g, player):
                mask[A_MORTGAGE + i] = 1
            if phase == PHASE_MANAGE:
                if p.can_unmortgage(g, player):
                    mask[A_UNMORTGAGE + i] = 1
                # No A_TRADE bit: the AI never initiates trades.

        mask[A_END_MANAGE] = 1
        return mask

    def _apply_manage_action(self, player, action):
        g = self.game
        if A_BUILD <= action < A_BUILD + NUM_OWNABLE:
            prop = self.ownable[action - A_BUILD]
            if g.build_house(prop, player):
                self.log(f"{player.name} [AI] built on {prop.name} "
                         f"(now {prop.houses}).")
        elif A_SELL <= action < A_SELL + NUM_OWNABLE:
            prop = self.ownable[action - A_SELL]
            if g.sell_house(prop, player):
                self.log(f"{player.name} [AI] sold a house on {prop.name}.")
        elif A_MORTGAGE <= action < A_MORTGAGE + NUM_OWNABLE:
            prop = self.ownable[action - A_MORTGAGE]
            if g.mortgage_property(prop, player):
                self.log(f"{player.name} [AI] mortgaged {prop.name} "
                         f"for ${prop.mortgage_value}.")
        elif A_UNMORTGAGE <= action < A_UNMORTGAGE + NUM_OWNABLE:
            prop = self.ownable[action - A_UNMORTGAGE]
            cost = prop.unmortgage_cost
            if g.unmortgage_property(prop, player):
                self.log(f"{player.name} [AI] lifted the mortgage on "
                         f"{prop.name} for ${cost}.")
        # A_TRADE is never legal: the AI does not initiate trades.

    # --- Trade evaluation (responding to a human's proposal) ---------------

    # How much a freshly completed monopoly is worth beyond the raw price of its
    # tiles, as a multiple of the group's total list price. A full set roughly
    # doubles a group's value (it unlocks houses and multiplied rent), so a
    # gained set adds ~1x its price on top of the tiles themselves, and handing
    # one to the opponent costs the same.
    SET_BONUS = 1.0

    def evaluate_trade(self, me, other, gain, lose, cash_delta):
        """Estimates the dollar value to ``me`` of a trade and decides on it.

        From ``me``'s perspective the trade hands ``me`` the properties in
        ``gain`` and ``cash_delta`` dollars, and takes away the properties in
        ``lose`` (which go to ``other``). The value is::

            value =  cash_delta
                   + sum(property_value(p) for p in gain)
                   - sum(property_value(p) for p in lose)
                   + SET_BONUS * (group price of any monopoly this completes
                                  for ``me``, minus any it breaks for ``me``)
                   - SET_BONUS * (group price of any monopoly this hands to
                                  ``other``, minus any it strips from ``other``)

        where a property is valued at its list price, less the outstanding
        unmortgage cost if it is mortgaged. ``me`` accepts only when the value
        is strictly positive (and it can afford any cash it owes).

        Returns ``(accepted: bool, value: float)``.
        """
        # Can't accept a trade whose cash ``me`` cannot cover.
        if cash_delta < 0 and me.balance < -cash_delta:
            return False, float("-inf")

        value = float(cash_delta)
        value += sum(self._property_value(p) for p in gain)
        value -= sum(self._property_value(p) for p in lose)

        # Strategic set synergy. Only ``me`` and ``other`` change holdings.
        value += self.SET_BONUS * self._set_swing(me, gain, lose)
        value -= self.SET_BONUS * self._set_swing(other, lose, gain)

        return value > 0, value

    def _property_value(self, prop):
        """List-price value of a property, discounted for an unpaid mortgage."""
        if prop.mortgaged:
            return float(prop.price - prop.unmortgage_cost)
        return float(prop.price)

    def _ownable_groups(self):
        """Groups the ownable tiles into sets: each street colour, all
        railroads, all utilities -- the units that form a monopoly."""
        groups = {}
        for t in self.ownable:
            if isinstance(t, StreetProperty):
                key = ("street", t.color)
            elif isinstance(t, Railroad):
                key = ("railroad", None)
            else:
                key = ("utility", None)
            groups.setdefault(key, []).append(t)
        return groups

    def _set_swing(self, player, gained, lost):
        """Net group-price value of the monopolies ``player`` gains minus those
        it loses, if it were to receive ``gained`` and give up ``lost``."""
        gained_ids = {id(t) for t in gained}
        lost_ids = {id(t) for t in lost}

        def owns_after(tile):
            if id(tile) in gained_ids:
                return True
            if id(tile) in lost_ids:
                return False
            return tile.owner is player

        swing = 0.0
        for tiles in self._ownable_groups().values():
            before = all(t.owner is player for t in tiles)
            after = all(owns_after(t) for t in tiles)
            if after and not before:
                swing += sum(t.price for t in tiles)   # completed a monopoly
            elif before and not after:
                swing -= sum(t.price for t in tiles)   # broke a monopoly
        return swing

    def _has_liquidation_options(self, player):
        g = self.game
        for prop in player.properties:
            if isinstance(prop, StreetProperty) and prop.can_sell_house(g, player):
                return True
            if prop.can_mortgage(g, player):
                return True
        return False


def _safe_default(phase):
    if phase == PHASE_JAIL:
        return A_ROLL_JAIL
    if phase == PHASE_BUY:
        return A_DECLINE
    return A_END_MANAGE
