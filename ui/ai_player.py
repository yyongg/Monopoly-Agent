"""AI player for the GUI: drives decisions using a trained MaskablePPO model.

``GUIAIDecider`` wraps a model and exposes simple per-decision methods
(``jail_choice``, ``purchase_decision``, ``manage_loop``, ``liquidate_loop``)
that the GUI can call instead of showing a human prompt. Observation encoding,
tile/trade valuations, and legal masks all come from a shared
:class:`~engine.observation.ObsEncoder` (``self.encoder``) -- the same object the
training env uses -- so the model can never be fed inputs that drift from what it
was trained on.

The AI never *initiates* trades via the model (trade actions are masked out,
matching the trained policy) but does propose set-completing trades heuristically
(:meth:`_attempt_trade`). When a human proposes a trade to the AI,
``evaluate_trade`` scores the offer with a dollar-valued formula and accepts only
if it comes out ahead -- see that method for the formula.
"""

import numpy as np

from engine.constants import (
    PHASE_JAIL, PHASE_BUY, PHASE_MANAGE, PHASE_LIQUIDATE,
    PHASE_AUCTION, PHASE_TRADE_RESPOND,
    NUM_ACTIONS, NUM_OWNABLE, NUM_BID_LEVELS, BID_FRACTIONS, BID_CEILING_MULT,
    A_PAY_JAIL, A_USE_CARD, A_ROLL_JAIL,
    A_BUY, A_DECLINE, A_END_MANAGE,
    A_BUILD, A_SELL, A_MORTGAGE, A_UNMORTGAGE, A_TRADE,
    A_TRADE_REJECT, A_AUCTION_PASS, A_AUCTION_BID,
)
from engine.observation import ObsEncoder


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
        # Observation encoder: the single shared implementation of obs
        # encoding, tile/trade valuations, and legal masks (also used by the
        # training env), so the GUI can never drift from what the policy was
        # trained on. The app sets ``trade_arbiter`` to resolve trades this AI
        # proposes to other players.
        self.encoder = ObsEncoder()
        self.trade_arbiter = None

    def bind(self, game, ownable):
        """Attach to a live game instance. Must be called before any decisions."""
        self.game = game
        self.ownable = ownable
        self.encoder.bind(game, ownable)

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
        """Returns True to buy ``prop``, False to decline.

        If the model chooses to buy but is short on cash (e.g. to complete a
        set), it first mortgages / sells to reach the price; it declines if it
        can't raise enough.
        """
        seat = self.game.players.index(player)
        action = self._decide(seat, PHASE_BUY, prop)
        if action != A_BUY:
            self.log(f"{player.name} [AI] declined {prop.name}.")
            return False
        if player.balance < prop.price:
            self.log(f"{player.name} [AI] is raising cash to buy {prop.name}.")
            self.liquidate_loop(player, prop.price)
            if player.balance < prop.price:
                self.log(f"{player.name} [AI] couldn't raise enough for "
                         f"{prop.name}; declined.")
                return False
        self.log(f"{player.name} [AI] bought {prop.name} for ${prop.price}.")
        return True

    def bid_choice(self, player, prop, min_bid=0):
        """Returns this AI's bid for one ascending-auction round (0 passes).

        Mirrors the env's bid hook: the model picks a bucket read as a valuation
        ceiling -- a fraction of the tile's *value to this bidder*
        (``_bid_value``), bounded by ``BID_CEILING_MULT`` * list price and by
        cash. The AI matches the round's ``min_bid`` while that ceiling covers
        it, otherwise it drops out.
        """
        seat = self.game.players.index(player)
        action = self._decide(seat, PHASE_AUCTION, prop)
        k = action - A_AUCTION_BID
        if 0 <= k < NUM_BID_LEVELS:
            value = self.encoder._bid_value(player, prop)
            ceiling = min(int(round(BID_FRACTIONS[k] * value)),
                          int(round(BID_CEILING_MULT * prop.price)),
                          player.balance)
            if min_bid <= 0:
                return ceiling
            if ceiling >= min_bid:
                self.log(f"{player.name} [AI] bids ${min_bid} for {prop.name}.")
                return min_bid
        return 0

    def manage_loop(self, player):
        """Issues management actions until the model chooses END_MANAGE."""
        seat = self.game.players.index(player)
        # A trade target is proposed at most once per manage pass (re-proposing a
        # rejected offer changes nothing); mirrors the env's guard.
        self.encoder._traded_this_manage = set()
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
            if not self.encoder._has_liquidation_options(player):
                return
            action = self._decide(seat, PHASE_LIQUIDATE)
            if action == A_END_MANAGE:
                return
            self._apply_manage_action(player, action)

    # --- Internal helpers ---

    def _decide(self, seat, phase, prop=None):
        obs = self.encoder._encode_obs(seat, phase, prop)
        mask = self.encoder._legal_mask(phase, prop, seat)
        action, _ = self.model.predict(
            obs, action_masks=mask.astype(bool), deterministic=self.deterministic)
        action = int(action)
        if not (0 <= action < NUM_ACTIONS and mask[action]):
            action = _safe_default(phase)
        return action

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
            i = action - A_TRADE
            self.encoder._traded_this_manage.add(i)
            self._attempt_trade(player, self.ownable[i])

    # --- Trade initiation (proposing a set-completing trade) ---------------

    def _attempt_trade(self, initiator, target):
        """Builds a set-completing offer and asks the partner (via the app's
        arbiter, or the valuation formula when none is set)."""
        partner = target.owner
        if not self.encoder._can_propose_trade(initiator, target):
            return
        give = self.encoder._choose_give_tile(initiator, partner, target)
        if give is None:
            return
        receive = [target]

        # Never propose a trade we would ourselves reject. ``evaluate_trade`` is
        # what decides every offer *we* receive, and it is linear in cash (slope
        # -1), so our value of this swap paying nothing is the most cash we could
        # add and still come out ahead. If even paying nothing is a loss for us
        # (the tile we'd give is worth more to us than the one we'd get), the
        # deal is bad regardless of price and we don't propose it. Otherwise we
        # cap the cash we offer at that break-even, so the identical deal handed
        # back to us always clears -- the two valuation paths can no longer
        # disagree.
        _, self_value = self.evaluate_trade(initiator, partner, receive, [give], 0)
        if self_value <= 0:
            return
        break_even = int(np.ceil(self_value)) - 1  # max cash keeping value > 0
        cash = min(self.encoder._balancing_cash(initiator, partner, [give], receive),
                   break_even, initiator.balance)
        if cash < 0:
            return
        if self.trade_arbiter is not None:
            accepted = self.trade_arbiter(
                initiator, partner, [give], receive, cash)
        else:  # headless fallback: the partner's own valuation formula
            accepted, _ = self.evaluate_trade(
                partner, initiator, [give], receive, cash)
        if not accepted:
            self.log(f"{partner.name} declined {initiator.name} [AI]'s trade.")
            return
        if self.game.execute_trade(initiator, partner, [give], receive, cash):
            self.log(f"{initiator.name} [AI] traded {give.name} + ${cash} to "
                     f"{partner.name} for {target.name}.")

    # --- Trade evaluation (responding to a human's proposal) ---------------

    def evaluate_trade(self, me, other, gain, lose, cash_delta):
        """Estimates the dollar value to ``me`` of a trade and decides on it.

        From ``me``'s perspective the trade hands ``me`` the properties in
        ``gain`` and ``cash_delta`` dollars, and takes away the properties in
        ``lose`` (which go to ``other``). The value is::

            value =  cash_delta
                   + sum(prop_value(p) for p in gain)
                   - sum(prop_value(p) * (1 + cfg.keep_premium) for p in lose)
                   + cfg.trade_income_weight * (expected income gained - lost)
                   + cfg.set_bonus * (group price of any monopoly this completes
                                      for ``me``, minus any it breaks for ``me``)
                   - cfg.set_bonus * (group price of any monopoly this hands to
                                      ``other``, minus any it strips from ``other``)

        where a property is valued at its list price, less the outstanding
        unmortgage cost if it is mortgaged. On top of that each tile carries a
        few turns of its expected rent (landing traffic * nominal rent), so a
        busy tile is worth more to take and dearer to give up than a quiet one.
        Property the AI gives up also carries the ``KEEP_PREMIUM``, reflecting
        the rent and set potential lost, so it will not surrender a tile for a
        marginal cash gain. ``me`` accepts only when the value is strictly
        positive (and it can afford any cash it owes).

        Returns ``(accepted: bool, value: float)``.
        """
        # Can't accept a trade whose cash ``me`` cannot cover.
        if cash_delta < 0 and me.balance < -cash_delta:
            return False, float("-inf")

        value = float(cash_delta)
        value += sum(self.encoder._prop_value(p) for p in gain)
        value -= sum(self.encoder._prop_value(p) * (1.0 + self.encoder.cfg.keep_premium)
                     for p in lose)

        # Traffic/rent: value a few turns of each tile's expected income.
        value += self.encoder.cfg.trade_income_weight * sum(self.encoder._expected_income(p) for p in gain)
        value -= self.encoder.cfg.trade_income_weight * sum(self.encoder._expected_income(p) for p in lose)

        # Strategic set synergy. Only ``me`` and ``other`` change holdings.
        value += self.encoder.cfg.set_bonus * self._set_swing(me, gain, lose)
        value -= self.encoder.cfg.set_bonus * self._set_swing(other, lose, gain)

        return value > 0, value

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
        for tiles in self.encoder._groups:
            before = all(t.owner is player for t in tiles)
            after = all(owns_after(t) for t in tiles)
            # Weight a completed/broken set by its development ROI, so gaining an
            # efficient money group (orange/light-blue) counts for more than a
            # sticker-equal but sluggish one -- consistent with _trade_value.
            set_price = sum(t.price for t in tiles) * self.encoder._set_quality(tiles)
            if after and not before:
                swing += set_price   # completed a monopoly
            elif before and not after:
                swing -= set_price   # broke a monopoly
        return swing

def _safe_default(phase):
    if phase == PHASE_JAIL:
        return A_ROLL_JAIL
    if phase == PHASE_BUY:
        return A_DECLINE
    if phase == PHASE_AUCTION:
        return A_AUCTION_PASS
    if phase == PHASE_TRADE_RESPOND:
        return A_TRADE_REJECT
    return A_END_MANAGE
