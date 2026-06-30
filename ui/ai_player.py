"""AI player for the GUI: drives decisions using a trained MaskablePPO model.

``GUIAIDecider`` wraps a model and exposes simple per-decision methods
(``jail_choice``, ``purchase_decision``, ``manage_loop``, ``liquidate_loop``)
that the GUI can call instead of showing a human prompt. The obs encoding and
legal-mask logic mirror ``MonopolyEnv`` exactly so the model sees the same
inputs it was trained on.
"""

import numpy as np

from engine.rl_env import (
    PHASE_JAIL, PHASE_BUY, PHASE_MANAGE, PHASE_LIQUIDATE,
    NUM_PHASES, NUM_ACTIONS, NUM_OWNABLE,
    A_PAY_JAIL, A_USE_CARD, A_ROLL_JAIL,
    A_BUY, A_DECLINE, A_END_MANAGE,
    A_BUILD, A_SELL, A_MORTGAGE, A_UNMORTGAGE, A_TRADE,
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
                if p.can_sell_house(g, player):
                    mask[A_SELL + i] = 1
            if p.can_mortgage(g, player):
                mask[A_MORTGAGE + i] = 1
            if phase == PHASE_MANAGE:
                if p.can_unmortgage(g, player):
                    mask[A_UNMORTGAGE + i] = 1
                if self._find_trade_buyer(player, p) is not None:
                    mask[A_TRADE + i] = 1

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
        elif A_TRADE <= action < A_TRADE + NUM_OWNABLE:
            self._do_trade(player, self.ownable[action - A_TRADE])

    def _do_trade(self, seller, prop):
        buyer = self._find_trade_buyer(seller, prop)
        if buyer is not None and self.game.execute_trade(
                seller, buyer, [prop], [], -prop.price):
            self.log(f"{seller.name} [AI] sold {prop.name} to "
                     f"{buyer.name} for ${prop.price}.")

    def _find_trade_buyer(self, seller, prop):
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
        if isinstance(prop, StreetProperty):
            group = prop.color_group(self.game)
            owned = sum(1 for t in group if t.owner is buyer)
            if owned == 0:
                return 0
            return 10 if owned == len(group) - 1 else owned
        return sum(1 for p in buyer.properties if isinstance(p, type(prop))) + 1

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
