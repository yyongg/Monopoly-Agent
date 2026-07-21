"""Group-level economics: what a monopoly is worth, and what a *share* of one is
worth.

:class:`SetValuer` is the single model of set value in the project. The trade
heuristic (:mod:`engine.trade`, through ``ObsEncoder._trade_value``) and the
reward (:mod:`engine.rewards`) both read it, so "orange is worth more than the
utilities" resolves to the same number in the negotiation, in the net worth the
policy is trained on, and in the GUI.

Two ideas carry the whole module:

**Strength** -- a per-group multiplier for how good the set actually is, blending
absolute earning power (traffic x full-development rent) with capital efficiency
(that rent per dollar it costs to buy and build). The efficiency term is what
makes the ranking match real play: on earning power alone *green* outranks
*orange*, because green has the highest hotel rent -- but it costs $3,000 to
develop and has the worst payback of any street set. Cost-adjusted, orange comes
out on top where it belongs.

**Completion** -- how set value grows with the fraction of the group you hold,
``f(k, n) = (k/n) ** set_progress_exponent``. This replaced a step function that
priced a set *only* when a tile was exactly one away from completing it. Under
the old rule a player holding 2/3 of orange valued St. James at its $227 sticker
and would sell it for a brown and $200, while the same tile was worth $4,937 the
moment the set was complete: a 21.7x cliff with nothing underneath it. That is
what let a human trade junk for monopolies. The curve prices every rung.

A tile's worth to a player is then its **marginal** contribution -- what the
group position is worth with it, minus what it is worth without it -- which is
one expression for what used to be three special cases (completes my set, denies
an opponent's set, and the first-monopoly premium).
"""

from engine.config import RewardConfig
from models.tiles.properties.railroad import Railroad
from models.tiles.properties.street_property import StreetProperty

# What every player starts with (``models.player.Player.balance``). The stage
# multiplier measures cash inflation against it.
STARTING_BALANCE = 1500.0


def peak_rent(prop):
    """The rent ``prop`` collects when its group is *fully developed* -- the
    earning power that makes a monopoly worth holding. Streets: the hotel row of
    the rent table. Railroads: the four-railroad rent (25 * 2**4 = 400, the
    engine's doubling table). Utilities: both owned, at an average roll of 7
    (10 * 7 = 70)."""
    if isinstance(prop, StreetProperty):
        return float(prop.rent_table[-1])
    if isinstance(prop, Railroad):
        return 400.0          # 25 * 2**4: rent per railroad when all four are held
    return 70.0               # utility: 10x an average roll of 7 (both owned)


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _mean(values):
    return (sum(values) / len(values)) if values else 0.0


class SetValuer:
    """Values monopoly groups for one live game. See the module docstring.

    Construct with a :class:`~engine.config.RewardConfig`, then :meth:`bind` to a
    game. ``bind`` takes the traffic function rather than an
    :class:`~engine.observation.ObsEncoder` so this module stays free of the
    observation layer (which imports *it*).
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or RewardConfig()
        self.game = None
        self.groups = []
        self._group_of = {}
        self._strength_of = {}   # id(tile) -> its group's strength

    def bind(self, game, groups, traffic):
        """Attach to a live game. ``groups`` is the monopoly grouping (see
        ``engine.observation.build_groups``); ``traffic`` is a callable giving a
        tile's landing frequency as a multiple of an average tile."""
        self.game = game
        self.groups = groups
        self._group_of = {id(t): grp for grp in groups for t in grp}
        self._compute_strengths(traffic)
        return self

    # -- Per-set strength (static for a board: computed once per bind) -------
    def _compute_strengths(self, traffic):
        """Precomputes each group's strength, cached by tile id.

        Blends two terms, each normalised by its own mean across the groups so
        the *average* group scores ~1.0 -- making strength a redistribution of
        ``trade_monopoly_mult`` rather than a change to the overall scale:

        * **power** -- traffic x full-development rent, summed over the group:
          what the set earns when built out.
        * **efficiency** -- that power per dollar of capital the set needs (list
          price + a hotel on every tile), clamped by ``set_quality_clamp`` so it
          tilts the ranking without dominating it.

        The blend is renormalised to mean 1.0 again and clamped by
        ``set_strength_clamp``.
        """
        power = [sum(traffic(t) * peak_rent(t) for t in grp) for grp in self.groups]
        capital = [self.group_price(grp) + self._development_cost(grp)
                   for grp in self.groups]
        efficiency = [p / c if c else 0.0 for p, c in zip(power, capital)]

        power_mean, eff_mean = _mean(power), _mean(efficiency)
        lo_q, hi_q = self.cfg.set_quality_clamp
        blend = []
        for p, e in zip(power, efficiency):
            p_n = (p / power_mean) if power_mean else 1.0
            e_n = (e / eff_mean) if eff_mean else 1.0
            blend.append(p_n * _clamp(e_n, lo_q, hi_q))

        blend_mean = _mean(blend)
        lo_s, hi_s = self.cfg.set_strength_clamp
        self._strength_of = {}
        for grp, b in zip(self.groups, blend):
            strength = _clamp(b / blend_mean, lo_s, hi_s) if blend_mean else 1.0
            for tile in grp:
                self._strength_of[id(tile)] = strength

    @staticmethod
    def _development_cost(grp):
        """What a hotel on every tile of ``grp`` costs (a hotel is five house
        purchases). Zero for railroads and utilities, which never build."""
        return sum(5 * t.house_cost() for t in grp
                   if isinstance(t, StreetProperty))

    @staticmethod
    def group_price(grp):
        """Total list price of every tile in a monopoly group."""
        return float(sum(t.price for t in grp))

    def strength(self, grp):
        """``grp``'s strength multiplier (see :meth:`_compute_strengths`): ~1.0
        for an average set, 1.73 for orange, at the floor for brown/utility.
        Falls back to 1.0 for an unrecognised group."""
        return self._strength_of.get(id(grp[0]), 1.0) if grp else 1.0

    def group_of(self, prop):
        """The monopoly group ``prop`` belongs to, or ``None``."""
        return self._group_of.get(id(prop))

    # -- Game stage ---------------------------------------------------------
    def stage(self):
        """Cash-inflation multiplier on set value.

        Every player collects $200 a lap, so balances drift far above the $1,500
        start and a fixed set price gets cheaper in real terms as the game runs
        long -- the agent would sell a monopoly for what was a fortune on turn 5
        and is pocket change on turn 80. Scales with the mean live balance,
        weighted by ``stage_inflation_weight`` and capped by
        ``stage_inflation_cap``.

        Measured from cash actually on the board rather than the turn count: that
        is what inflation *is* here, and it self-calibrates (a poor, stalled game
        does not inflate). Never below 1.0 -- a poor board should not make sets
        cheap.
        """
        w = self.cfg.stage_inflation_weight
        if w <= 0.0:
            return 1.0
        live = [p for p in self.game.players if not p.bankrupt]
        if not live:
            return 1.0
        mean_balance = _mean([p.balance for p in live])
        return _clamp(1.0 + w * (mean_balance / STARTING_BALANCE - 1.0),
                      1.0, self.cfg.stage_inflation_cap)

    # -- Set value ----------------------------------------------------------
    def any_monopoly_exists(self, exclude=None):
        """Whether any solvent player already owns a complete colour group
        (optionally ignoring ``exclude``). Prices the *first*-monopoly tempo
        premium, mirroring the first-monopolist reward in :mod:`engine.rewards`."""
        for grp in self.groups:
            if grp is exclude:
                continue
            owner = grp[0].owner
            if (owner is not None and not owner.bankrupt
                    and all(t.owner is owner for t in grp)):
                return True
        return False

    def monopoly_value(self, grp, stage=True):
        """Dollar value of owning **all** of ``grp``, for trade purposes.

        Its sticker price is the dollar anchor; :meth:`strength` tilts that by
        how good the set really is; the game's *first* monopoly is a decisive
        tempo edge and so is worth more to secure -- or to refuse to sell into.

        ``stage=False`` drops the cash-inflation multiplier: the reward wants a
        set's worth in absolute terms, not against today's board (see
        :meth:`engine.rewards.RewardMixin._set_net_worth`).
        """
        value = (self.cfg.monopoly_bonus * self.group_price(grp)
                 * self.strength(grp) * self.cfg.trade_monopoly_mult)
        if not self.any_monopoly_exists(exclude=grp):
            value *= 1.0 + self.cfg.trade_first_monopoly_weight
        if stage:
            value *= self.stage()
        return value

    def completion(self, owned, size):
        """``f(k, n)``: the fraction of a set's value that holding ``owned`` of
        its ``size`` tiles is worth. ``(k/n) ** set_progress_exponent`` -- 0 with
        none of it, 1.0 with all of it, convex in between so completing is the
        biggest single step while every tile below it still costs real money.

        The exponent is the fix for the junk-for-monopoly exploit; see the module
        docstring and ``RewardConfig.set_progress_exponent``.
        """
        if size <= 0:
            return 0.0
        frac = _clamp(owned / float(size), 0.0, 1.0)
        return frac ** self.cfg.set_progress_exponent

    def held_fraction(self, player, grp):
        """How much of ``grp``'s value ``player``'s current holding is worth."""
        owned = sum(1 for t in grp if t.owner is player)
        return self.completion(owned, len(grp))

    def _step(self, grp, holder, prop):
        """The fraction of ``grp``'s value that ``prop`` adds for ``holder``,
        counting ``holder``'s other tiles only -- so the answer does not depend on
        who holds ``prop`` right now."""
        size = len(grp)
        without = sum(1 for t in grp if t is not prop and t.owner is holder)
        return self.completion(without + 1, size) - self.completion(without, size)

    def marginal(self, prop, player):
        """What ``prop`` is worth to ``player`` as a piece of its colour group:
        the group position's value **with** it minus its value **without** it.

        Deliberately independent of whether ``player`` currently holds ``prop``,
        so the tile is priced the same whether they are buying it or being asked
        to give it up -- the two sides of one negotiation. At one-tile-from-
        complete this is the big completion jump the old step-function valuation
        gave; below that it now returns real money instead of zero.

        Single-tile only. To price a *package* use :meth:`swap_delta` -- set value
        is not additive across tiles of the same group, and summing marginals
        double-counts in both directions (see that method).
        """
        grp = self._group_of.get(id(prop))
        if grp is None:
            return 0.0
        return self.monopoly_value(grp) * self._step(grp, player, prop)

    def denial(self, prop, owner):
        """What ``prop`` is worth to the opponent it would help most -- the
        blocking value of keeping it away from them, *unweighted*.

        The **max** over opponents, not the sum: only one of them can end up with
        it. Callers weight this by ``trade_denial_weight``, which must stay below
        1.0 or the tile is worth the same to both sides, the swap is zero-sum, and
        no trade can ever clear (see ``RewardConfig.trade_denial_weight``).
        """
        grp = self._group_of.get(id(prop))
        if grp is None:
            return 0.0
        best = 0.0
        for other in self.game.players:
            if other is owner or other.bankrupt:
                continue
            best = max(best, self._step(grp, other, prop))
        return self.monopoly_value(grp) * best

    # -- Package valuation --------------------------------------------------
    def _affected_groups(self, tiles):
        """The distinct colour groups ``tiles`` touch, keyed so each appears once."""
        found = {}
        for tile in tiles:
            grp = self._group_of.get(id(tile))
            if grp is not None:
                found[id(grp[0])] = grp
        return list(found.values())

    def _best_opponent_share(self, grp, player, extra=(), minus=(), holder=None):
        """The largest share of ``grp`` any live opponent of ``player`` holds.

        ``extra`` / ``minus`` adjust ``holder``'s count, modelling the state
        *after* a swap: tiles ``player`` hands over land with ``holder``, tiles it
        takes come out of ``holder``'s hands. When ``holder`` is None the counts
        are read straight off the board.
        """
        extra_ids = {id(t) for t in extra}
        minus_ids = {id(t) for t in minus}
        size = len(grp)
        best = 0.0
        for other in self.game.players:
            if other is player or other.bankrupt:
                continue
            owned = sum(1 for t in grp if t.owner is other)
            if holder is None or other is holder:
                owned += sum(1 for t in grp if id(t) in extra_ids)
                owned -= sum(1 for t in grp if id(t) in minus_ids)
            best = max(best, self.completion(owned, size))
        return best

    def swap_delta(self, player, gain, lose, partner=None):
        """The change in ``player``'s **set** value from a swap: it receives every
        tile in ``gain`` (from ``partner``) and hands over every tile in ``lose``
        (to ``partner``).

        Priced per affected *group*, comparing the position before and after the
        whole package -- not as a sum of per-tile marginals, which is wrong
        whenever a package touches one group twice. Summing double-counted badly
        in both directions: giving away one orange to receive another scored as
        two independent completion jumps rather than the wash it is, and taking
        two tiles of a group scored ``2 * f(1)`` instead of ``f(2)``. That
        arithmetic produced trades that moved thousands of dollars and changed
        nothing.

        Includes the blocking term: what the swap does for whichever opponent it
        helps most, weighted by ``trade_denial_weight``. ``partner`` is who the
        tiles move to and from; ``None`` means the blocking change is attributed
        to whichever opponent it would most help.
        """
        total = 0.0
        for grp in self._affected_groups(list(gain) + list(lose)):
            size = len(grp)
            value = self.monopoly_value(grp)

            mine_before = sum(1 for t in grp if t.owner is player)
            mine_after = (mine_before
                          + sum(1 for t in gain if self._group_of.get(id(t)) is grp)
                          - sum(1 for t in lose if self._group_of.get(id(t)) is grp))
            total += value * (self.completion(mine_after, size)
                              - self.completion(mine_before, size))

            # Blocking: tiles we hand over strengthen the partner, tiles we take
            # weaken them.
            before = self._best_opponent_share(grp, player)
            after = self._best_opponent_share(grp, player, extra=lose, minus=gain,
                                              holder=partner)
            total -= self.cfg.trade_denial_weight * value * (after - before)
        return total
