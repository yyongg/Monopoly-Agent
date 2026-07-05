"""Reward shaping for :class:`engine.rl_env.MonopolyEnv`.

``RewardMixin`` holds everything that turns game state into the scalar reward:
the shaped net-worth valuation, the one-time acquisition / denial / build
bonuses, and the decisive terminal reward. It is mixed into ``MonopolyEnv`` and
uses the env's per-episode state (``seat``, ``reward_mode``, ``_pending_bonus``,
``_prev_advantage``) and its :class:`~engine.observation.ObsEncoder`
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

    def _owned_monopoly_value(self, player):
        """Total list price of the monopoly groups ``player`` fully owns."""
        total = 0.0
        for tiles in self.encoder._groups:
            if all(t.owner is player for t in tiles):
                total += sum(t.price for t in tiles)
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
                tempo = max(0.0, 1.0 - self._turn / horizon)
                self._pending_bonus += (
                    self.cfg.first_monopoly_bonus_coef * self.cfg.monopoly_bonus
                    * enc._group_price(grp) / 1000.0
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
                                    * enc._group_price(grp) / 1000.0 * mult)

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
        """Expected rent ``player`` pays per board round: for each tile owned by
        a live opponent, landing traffic times the rent it collects *as
        developed now*. A proxy for how hard the board can hit the agent's cash
        -- the size its liquid cushion should cover so a bad landing does not
        bankrupt it."""
        enc = self.encoder
        total = 0.0
        for prop in enc.ownable:
            owner = prop.owner
            if owner is None or owner is player or owner.bankrupt:
                continue
            total += enc._traffic(prop) * enc._developed_rent(prop)
        return total

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

    def _reward(self, terminal):
        controlled = self.game.players[self.seat]
        reward = 0.0
        if self.reward_mode == "shaped":
            # Shape on *relative* advantage (my net worth minus the mean
            # opponent's), not absolute net worth, so an opponent completing a
            # set costs me reward. It telescopes over the episode, leaving the
            # decisive terminal reward unaffected.
            adv = self._net_worth(controlled) - self._mean_opp_networth()
            reward += (adv - self._prev_advantage) / 1000.0
            self._prev_advantage = adv
            reward += self._pending_bonus
            self._pending_bonus = 0.0
            # Solvency: make liquidity itself valuable so converting cash into
            # assets is not "free" when it leaves the agent unable to absorb a
            # bad landing (the direct counter to self-bankruptcy vs the FP bots).
            reward -= self._solvency_penalty(controlled)
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
