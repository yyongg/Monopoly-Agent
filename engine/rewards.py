"""Reward shaping for :class:`engine.rl_env.MonopolyEnv`.

``RewardMixin`` holds everything that turns game state into the scalar reward:
the shaped net-worth valuation, the one-time acquisition / denial / build
bonuses, and the decisive terminal reward. It is mixed into ``MonopolyEnv`` and
uses the env's per-episode state (``seat``, ``reward_mode``, ``gamma``,
``_pending_bonus``, ``_prev_potential``) and its
:class:`~engine.observation.ObsEncoder`
(``self.encoder``) for tile/set valuations -- so reward and observation share one
definition of a tile's worth.
"""

from models.tiles.properties.street_property import StreetProperty


class RewardMixin:
    """Reward-shaping methods for the Monopoly env (see module docstring)."""

    # -- Net worth ----------------------------------------------------------
    def _net_worth(self, player):
        """Net worth used for reward shaping.

        An owned, unmortgaged property is valued at ``price * (1 +
        acquisition_premium)`` -- above the cash price, reflecting the rent it
        earns -- so *buying* is a small net gain rather than net-worth-neutral.
        A mortgaged property is valued at ``price - unmortgage_cost``; combined
        with the ``mortgage_value`` cash a mortgage pays out, forced mortgaging
        is a clear net-worth loss the agent only takes when it needs the cash.
        """
        premium = self.cfg.acquisition_premium
        total = float(player.balance)
        for prop in player.properties:
            total += (prop.price - prop.unmortgage_cost) if prop.mortgaged \
                else prop.price * (1.0 + premium)
            if isinstance(prop, StreetProperty):
                # Value houses above cost (same premium as properties): they
                # multiply rent, so building should be a small net gain.
                total += prop.houses * prop.house_cost() * (1.0 + premium)
        # Reward holding *complete* sets: each fully-owned group adds a bonus,
        # so the tile that finishes a monopoly is a big net-worth jump.
        total += self.cfg.monopoly_bonus * self._owned_monopoly_value(player)
        return total

    def _effective_set_strength(self, grp):
        """The shared per-set strength (:meth:`ObsEncoder._set_strength`) tilted
        by ``set_strength_reward_weight`` (w): ``1 + w*(strength - 1)``. ``w=0``
        recovers the old flat behaviour (every set weighted 1.0); ``w=1`` is full
        strength. Read by the owned-monopoly net worth and the one-time
        first-monopoly / denial bonuses, so the reward prizes a high-traffic money
        set above a cheap one -- the same ordering trades use."""
        w = self.cfg.set_strength_reward_weight
        return 1.0 + w * (self.encoder._set_strength(grp) - 1.0)

    def _owned_monopoly_value(self, player):
        """Strength-weighted value of the monopoly groups ``player`` fully owns:
        each set's total list price times its :meth:`_effective_set_strength`, so
        holding (and thus losing) a strong set moves net worth -- and the shaped
        reward -- more than a weak one of equal sticker price."""
        total = 0.0
        for tiles in self.encoder._groups:
            if all(t.owner is player for t in tiles):
                total += (sum(t.price for t in tiles)
                          * self._effective_set_strength(tiles))
        return total

    # -- One-time shaped bonuses -------------------------------------------
    def _build_bonus(self, prop):
        """Shaped bonus for the house/hotel just added to ``prop``: the rent the
        new house adds (its rent-table jump, weighted by landing traffic and a
        cost tilt so a build dollar is rewarded by the rent it returns). Called
        after :meth:`Game.build_house` has incremented ``prop.houses``."""
        h = prop.houses
        rent_gain = float(prop.rent_table[h] - prop.rent_table[h - 1])
        roi_tilt = self.cfg.build_roi_ref_house_cost / prop.house_cost()
        return (self.cfg.build_bonus_coef * self.encoder._traffic(prop)
                * rent_gain * roi_tilt / 1000.0)

    def _on_acquire(self, player, prop, source="trade"):
        """``Game.on_acquire`` hook: ``prop`` just transferred to ``player``.

        For the controlled agent, queue two one-time shaped bonuses:

        * **Acquisition** -- when taken fresh from the bank, scaled by expected
          income so the agent actively acquires instead of dumping to auction.
          Buying on landing earns the full ``acquisition_bonus_coef``; an auction
          win earns the smaller ``auction_acquisition_bonus_coef``. Trades earn
          nothing (net worth already prices them).
        * **Denial** -- when the tile was an opponent's last-missing piece,
          blocking their monopoly (any source).
        """
        if player is not self.game.players[self.seat]:
            return
        enc = self.encoder
        if source in ("buy", "auction"):
            coef = (self.cfg.acquisition_bonus_coef if source == "buy"
                    else self.cfg.auction_acquisition_bonus_coef)
            self._pending_bonus += coef * enc._expected_income(prop) / 1000.0
            if source == "buy":
                # Price-scaled premium for buying on landing (never on an auction
                # win), sized to cancel the cheap-auction net-worth windfall.
                self._pending_bonus += (self.cfg.buy_preference_coef
                                        * prop.price / 1000.0)
        grp = enc._group_of.get(id(prop))
        if grp is None:
            return

        # First-to-a-set: I just completed this group and no monopoly exists
        # anywhere yet -> reward being the game's first monopolist, weighted up
        # the earlier it happens (a fast race, not just eventual completion).
        if all(t.owner is player for t in grp):
            if self._count_complete_sets(exclude=grp) == 0:
                horizon = max(1.0, self.cfg.first_monopoly_tempo_turns)
                tempo = max(0.0, 1.0 - self.game.turn / horizon)
                self._pending_bonus += (
                    self.cfg.first_monopoly_bonus_coef * self.cfg.monopoly_bonus
                    * enc._group_price(grp) / 1000.0
                    * self._effective_set_strength(grp)
                    * (1.0 + self.cfg.first_monopoly_tempo_weight * tempo))
            return

        # Denial: I took an opponent's last-missing tile. Weight it up when the
        # tile would have completed the *game's first* monopoly (no set owned by
        # anyone yet), so stopping an opponent from being first to a set is worth
        # more than a block after the race is already under way.
        if any(o is not player and not o.bankrupt
               and enc._completes_monopoly_for(o, prop)
               for o in self.game.players):
            mult = (1.0 + self.cfg.first_denial_weight
                    if self._count_complete_sets() == 0 else 1.0)
            self._pending_bonus += (self.cfg.denial_bonus_coef
                                    * self.cfg.monopoly_bonus
                                    * enc._group_price(grp) / 1000.0 * mult
                                    * self._effective_set_strength(grp))

    def _count_complete_sets(self, exclude=None):
        """Number of colour groups fully owned by a single non-bankrupt player,
        ignoring ``exclude`` -- used to test whether *any other* set already
        exists (i.e. whether a just-completed or just-denied set is the first)."""
        count = 0
        for tiles in self.encoder._groups:
            if tiles is exclude:
                continue
            owner = tiles[0].owner
            if owner is not None and not owner.bankrupt \
                    and all(t.owner is owner for t in tiles):
                count += 1
        return count

    # -- Terminal reward ----------------------------------------------------
    def _decisive_winner(self):
        """The winner even when no one was bankrupted (a turn-cap timeout): the
        sole survivor, or on a timeout the richest survivor by shaped net worth,
        so every episode is conclusive."""
        survivors = self.game.active_players()
        if not survivors:
            return None
        if len(survivors) == 1:
            return survivors[0]
        return max(survivors, key=self._net_worth)

    def _terminal_reward(self):
        """Decisive end-of-game reward in ``[-1, +1]``: bankruptcy -1, sole
        survivor +1, otherwise the agent's net-worth rank among survivors mapped
        linearly to ``[-1, +1]``."""
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

    def _rent_exposure(self, player):
        """Expected rent ``player`` pays per board round -- the board's rent
        threat that sizes the liquid cushion. Defined once on the shared
        :class:`~engine.observation.ObsEncoder` (also used by the trade surplus
        cap), so reward and trade valuations agree on the cushion."""
        return self.encoder._rent_exposure(player)

    def _solvency_penalty(self, player):
        """Per-step drag for holding less liquid cash than the board's rent
        threat warrants. Zero once cash covers ``solvency_cushion_turns`` rounds
        of expected rent outflow; rises linearly to ``solvency_penalty_coef`` as
        cash falls to zero. Net worth alone prices cash and property alike, so
        without this the agent spends itself broke to bank shaped acquisition
        reward and only survives against opponents too weak to punish it."""
        coef = self.cfg.solvency_penalty_coef
        if coef <= 0.0:
            return 0.0
        cushion = self.cfg.solvency_cushion_turns * self._rent_exposure(player)
        if cushion <= 0.0:
            return 0.0
        deficit = max(0.0, cushion - player.balance) / cushion  # 0 .. 1
        return coef * deficit

    def _potential(self):
        """The shaping potential: the agent's *relative* net-worth advantage
        (mine minus the mean opponent's), in thousands. Relative, not absolute,
        so an opponent completing a set costs the agent reward."""
        controlled = self.game.players[self.seat]
        return (self._net_worth(controlled) - self._mean_opp_networth()) / 1000.0

    def _reward(self, terminal, truncated=False):
        """The step reward. ``terminal`` marks the last step of the episode;
        ``truncated`` says it ended because the *turn cap* cut off a live game
        rather than because the game actually finished.

        The distinction matters twice over. A truncated episode stops at an
        ordinary state, so its potential is real (not zero) and the learner
        bootstraps ``V(s')`` there -- adding a made-up final payoff on top would
        count the rest of the game twice.
        """
        real_end = terminal and not truncated
        controlled = self.game.players[self.seat]
        reward = 0.0
        if self.reward_mode == "shaped":
            # Potential-based shaping, done properly: ``gamma * phi(s') -
            # phi(s)``, with **phi = 0 at a true terminal**. Both details matter.
            #
            # The discount factor has to be here or the shaping is not the
            # policy-invariant transform it claims to be (Ng et al.). And zeroing
            # the potential at the end is what stops the *last* step paying out
            # the agent's whole accumulated net worth: ``_mean_opp_networth``
            # collapses to 0 once the last opponent is bankrupt, so the old code
            # emitted a +5..+12 spike there, dwarfing the +-1 that actually says
            # who won. The agent was being trained to end the game rich, not to
            # win it. Telescoped over the episode, what survives now is the
            # terminal reward -- the shaping only redistributes it in time.
            phi = 0.0 if real_end else self._potential()
            reward += self.gamma * phi - self._prev_potential
            self._prev_potential = phi

            reward += self._pending_bonus
            self._pending_bonus = 0.0

            # Solvency: make liquidity itself valuable, so converting cash into
            # assets is not "free" when it leaves the agent unable to absorb a
            # bad landing. Charged once per *turn*, not once per decision: the
            # number of decisions in a turn is the agent's own choice, so a
            # per-decision drag could be shrunk by simply doing less -- a
            # gradient toward passivity exactly when it needs to act.
            if self.game.turn != self._penalty_turn:
                self._penalty_turn = self.game.turn
                reward -= self._solvency_penalty(controlled)
        if real_end:
            reward += self._terminal_reward()
        return reward

    def _mean_opp_networth(self):
        """Mean net worth of the controlled seat's opponents.

        Averaged over a **fixed** denominator -- every opponent the game started
        with -- not just the survivors. A bankrupt player is worth exactly 0
        (``declare_bankrupt`` zeroes their balance and strips their property), so
        this is the same number while everyone is alive and, unlike a
        survivors-only mean, it does not *jump* when someone is eliminated.

        With the old survivors-only mean, knocking out the poorest opponent
        raised the average of those left, which lowered the agent's advantage and
        paid it a **negative** shaped reward for a strictly good outcome.
        """
        controlled = self.game.players[self.seat]
        others = [self._net_worth(p) for p in self.game.players
                  if p is not controlled]
        return sum(others) / len(others) if others else 0.0
