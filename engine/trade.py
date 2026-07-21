"""The trade heuristic: who offers what to whom, and for how much.

Trading is **not** a policy decision. It used to be: 86 of the 211 action ids
were trade proposals plus accept/reject, and the learned policy was disastrous at
it -- it accepted ~85% of every offer put to it, handed over completed sets for
junk, bled $58,747 of trade cash per game (players start with $1,500), and in one
measured game ping-ponged a single tile 424 times. Trades are now resolved here,
by valuation, for every seat alike.

That split follows Bonjour et al. 2021 (arxiv 2103.00683), whose *hybrid* agent
-- a fixed heuristic for the rare, valuation-heavy trade decisions and DRL for
the rest -- beat their pure-DRL agent 91.65% to 69.95%. Their reasoning applies
here exactly: a decision this rare produces too few training samples to learn a
good valuation, while a valuation is something we can simply *write down* (see
:mod:`engine.valuation`).

**The bargaining range.** Both sides' acceptance is linear in cash with slope 1
(:meth:`ObsEncoder.accepts` is ``trade_delta >= 0``), so for a fixed package of
tiles the deal is fully described by two numbers::

    partner accepts    <=>  cash >= -partner_value_at_zero_cash
    initiator accepts  <=>  cash <=  initiator_value_at_zero_cash

A deal therefore exists **iff the two sides' values sum to something positive** --
iff there are real gains from trade. This is why ``trade_denial_weight`` must
stay below 1.0: at 1.0 a tile is worth the same to the player blocking with it as
to the player who wants it, every swap sums to exactly zero, and nothing can ever
clear. The engine prices at the **midpoint** of that range, so both sides strictly
gain and no deal turns on a rounding error.
"""

import math

from engine.observation import TradeOffer


class TradeEngine:
    """Proposes and settles trades for one live game.

    Holds an :class:`~engine.observation.ObsEncoder` for the valuations
    (``_trade_value``) and the one acceptance rule (``accepts``), so a trade the
    GUI's AI would take is exactly a trade the training env's AI would take.
    """

    def __init__(self, encoder, on_offer=None, arbiter=None):
        self.encoder = encoder
        # Announced per proposal: (initiator, partner, offer, accepted, completes).
        self.on_offer = on_offer
        # Decides acceptance instead of :meth:`ObsEncoder.accepts` when set, as
        # ``arbiter(initiator, partner, give, receive, cash) -> bool``. The GUI
        # uses it to put an AI's offer to a *human* partner, who is under no
        # obligation to be rational. ``None`` (training, headless) means every
        # partner answers with the valuation.
        self.arbiter = arbiter

    @property
    def game(self):
        return self.encoder.game

    @property
    def cfg(self):
        return self.encoder.cfg

    # -- Driving ------------------------------------------------------------
    def run_round(self, initiator):
        """One round of trading for ``initiator``: at most one deal with each
        solvent opponent, best deal first.

        At most one deal per partner per round is the churn backstop. A trade
        strictly raises both sides' valuation, so a *cycle* of trades cannot keep
        paying out -- but cash moves and the board's stage multiplier drifts, and
        the old code's unbounded re-proposing is what produced the 424-trade
        ping-pong. One deal per partner bounds it structurally.
        """
        g = self.game
        for partner in list(g.players):
            if partner is initiator or partner.bankrupt or initiator.bankrupt:
                continue
            offer = self.best_offer(initiator, partner)
            if offer is None:
                continue
            self._settle(initiator, partner, offer)

    def _settle(self, initiator, partner, offer):
        """Puts ``offer`` to ``partner`` and executes it if they accept."""
        give, receive, cash = offer
        if self.arbiter is not None:
            accepted = bool(self.arbiter(initiator, partner, list(give),
                                         list(receive), cash))
        else:
            accepted, _ = self.encoder.accepts(partner, give, receive, cash,
                                               partner=initiator)
        completes = self._completes_set(initiator, receive)
        executed = bool(accepted and self.game.execute_trade(
            initiator, partner, list(give), list(receive), cash))
        if self.on_offer is not None:
            self.on_offer(initiator, partner, offer, executed, completes)
        return executed

    # -- Building an offer --------------------------------------------------
    def best_offer(self, initiator, partner):
        """The best deal ``initiator`` can put to ``partner``, or ``None``.

        Grows the **receive** side one tile at a time -- most wanted first, by
        marginal set value -- and prices each package, keeping whichever clears
        for the most gain. Multi-tile matters: when a partner holds two of a
        three-tile group, no one-for-one swap can ever complete the set, which the
        old 84-action trade band could not express.
        """
        wanted = [t for t in partner.properties
                  if self.game.can_trade_property(t)]
        wanted.sort(key=lambda t: self.encoder.sets.marginal(t, initiator),
                    reverse=True)

        best, best_gain = None, 0.0
        package = []
        for tile in wanted[:self.cfg.trade_max_package]:
            # Stop extending once the tiles stop being worth anything as set
            # progress -- past that we are just buying an opponent's junk.
            if package and self.encoder.sets.marginal(tile, initiator) <= 0:
                break
            package = package + [tile]
            offer = self._price(initiator, partner, package)
            if offer is None:
                continue
            gain = self.encoder.trade_delta(initiator, offer.receive, offer.give,
                                            offer.cash, partner=partner)
            if gain > best_gain:
                best, best_gain = offer, gain
        return best

    def _price(self, initiator, partner, receive):
        """The cheapest package + cash that buys ``receive``, or ``None``.

        Tries cash alone first, then adds give-tiles one at a time and takes the
        **first** package that clears -- never hand over more than the deal needs.
        Candidates are ordered by the surplus each creates (worth to the partner
        minus worth to us): offer what is cheap to you and dear to them, which is
        the paper's rule for what to put on the table.

        Tiles from a group we are *buying into* are excluded outright. Trading an
        orange away to get an orange is a wash by construction, and it crowds out
        the tiles that would actually pay for the deal.
        """
        wanted_groups = {id(g[0]) for g in self.encoder.sets._affected_groups(receive)}

        def elsewhere(tile):
            grp = self.encoder.sets.group_of(tile)
            return grp is None or id(grp[0]) not in wanted_groups

        pool = [t for t in initiator.properties
                if self.game.can_trade_property(t) and elsewhere(t)]
        pool.sort(key=lambda t: (self.encoder._trade_value(t, partner)
                                 - self.encoder._trade_value(t, initiator)),
                  reverse=True)

        offer = self._settle_cash(initiator, partner, [], receive)
        if offer is not None:
            return offer

        give = []
        for tile in pool:
            if len(give) >= self.cfg.trade_max_package:
                break
            if not self._may_hand_over(initiator, partner, give + [tile], receive):
                continue
            give = give + [tile]
            offer = self._settle_cash(initiator, partner, give, receive)
            if offer is not None:
                return offer
        return None

    def _may_hand_over(self, initiator, partner, package, receive):
        """Whether ``package`` may be offered -- i.e. whether it is acceptable for
        the tiles in it to end up with ``partner``.

        Handing over what **finishes the partner's set** is barred unless the deal
        finishes a set for us too, and the best set we gain is at least as strong
        as the best one we hand over: a genuine set-for-set, the one trade where
        arming a rival pays for itself.

        Judged on the whole package, not tile by tile -- two browns are each
        harmless alone and a monopoly together.

        This rule cannot be left to the valuation, and it is worth saying why.
        ``trade_denial_weight`` must sit below 1.0 or blocking is worth exactly
        what completing is worth, every swap is zero-sum, and nothing trades at
        all. But that same discount means gifting a set *costs the giver only a
        fraction of what it hands the taker* -- pure joint surplus, which is
        precisely what a greedy package builder hunts for. Measured without this
        rule: agents handed each other completed sets to buy single tiles, 47
        trades a game. The discount is right for pricing; it is not a licence to
        arm your opponent.
        """
        sets = self.encoder.sets
        gifted = [g for g in sets._affected_groups(package)
                  if self._completes_for(partner, g, package)]
        if not gifted:
            return True
        mine = [g for g in sets._affected_groups(receive)
                if self._completes_for(initiator, g, receive)]
        if not mine:
            return False
        return (max(sets.monopoly_value(g) for g in mine)
                >= max(sets.monopoly_value(g) for g in gifted))

    @staticmethod
    def _completes_for(player, grp, incoming):
        """Whether acquiring all of ``incoming`` finishes ``grp`` for ``player``."""
        ids = {id(t) for t in incoming}
        return all(t.owner is player or id(t) in ids for t in grp)

    def _settle_cash(self, initiator, partner, give, receive):
        """Prices a fixed pair of packages, or ``None`` when no cash clears it.

        See the module docstring for the bargaining range. ``cash`` is signed:
        positive means the initiator pays, negative means it is *paid* (which
        happens in a mutual set-for-set, where the tile it hands over is worth
        more to the partner than the one it takes).
        """
        enc = self.encoder
        # Each side's value of the tiles alone, before any cash changes hands.
        self_value = enc.trade_delta(initiator, receive, give, 0, partner=partner)
        partner_value = enc.trade_delta(partner, give, receive, 0, partner=initiator)
        if self_value + partner_value <= 0:
            return None      # no gains from trade: no cash can satisfy both

        lo = -partner_value  # least the initiator can pay and be accepted
        hi = self_value      # most it can pay and still come out ahead
        cash = int(round((lo + hi) / 2.0))            # split the surplus
        cash = max(int(math.ceil(lo)), min(cash, int(math.floor(hi))))

        cash = self._afford(initiator, partner, receive, cash)
        if cash is None or cash < lo or cash > hi:
            return None
        return TradeOffer(list(give), list(receive), cash)

    def _afford(self, initiator, partner, receive, cash):
        """Clamps ``cash`` to what the paying side actually has, or ``None``.

        The initiator will not spend the liquid cushion that keeps it solvent
        (:meth:`ObsEncoder._cash_reserve`, the same cushion the solvency reward
        sizes) on a speculative trade -- but it *will* spend all of it to complete
        a monopoly, which is worth far more than the cushion protects against.
        """
        if cash < 0:
            return -min(-cash, int(partner.balance))
        ceiling = int(initiator.balance)
        if not self._completes_set(initiator, receive):
            reserve = self.encoder._cash_reserve(initiator)
            ceiling = min(ceiling, int(initiator.balance - reserve))
        if ceiling < 0:
            return None
        return min(cash, ceiling)

    def _completes_set(self, player, receive):
        """Whether acquiring every tile of ``receive`` finishes a colour group for
        ``player`` that it does not already hold."""
        incoming = {id(t) for t in receive}
        for tile in receive:
            grp = self.encoder.sets.group_of(tile)
            if grp is None:
                continue
            if all(t.owner is player or id(t) in incoming for t in grp):
                return True
        return False
