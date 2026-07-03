"""Per-game statistics tracker for the pygame UI.

Instrumented over a single live ``Game`` via its optional hooks
(``on_auction_end``, ``on_acquire``) plus a wrapper around ``declare_bankrupt``,
this collects the same kind of end-of-game analytics that ``validation/simulate.py``
reports across many self-play games -- but for the one game the user just played,
so the UI can show a post-game summary screen.

Tracked:
    * game length (turns),
    * final property holdings per player,
    * per-player first-monopoly tempo (turn a set was first completed through play),
    * auctions won per player,
    * *blocking* acquisitions -- a readable list of moves that denied an opponent
      their last-missing tile for a set (buy, auction, or trade),
    * mean winning auction bid split by set-completers vs. non-set-completers.

As in ``simulate.py``, properties inherited by bankrupting an opponent are excluded
from the *first-monopoly* tempo (a set only "counts" if completed through play);
the displayed property counts are the raw current holdings the player can see on
the board.
"""

from models.tiles.properties.street_property import StreetProperty
from models.tiles.properties.railroad import Railroad
from models.tiles.properties.utility import Utility


def _group_label(group):
    """A short human label for a monopoly group (a list of tiles)."""
    tile = group[0]
    if isinstance(tile, StreetProperty):
        return f"{tile.color.replace('_', ' ').title()} set"
    if isinstance(tile, Railroad):
        return "Railroads"
    return "Utilities"


def _build_groups(ownable):
    """Groups ownable tiles into the sets that form a monopoly (mirrors
    ``MonopolyEnv._build_groups``)."""
    groups = {}
    for t in ownable:
        if isinstance(t, StreetProperty):
            key = ("street", t.color)
        elif isinstance(t, Railroad):
            key = ("railroad", "")
        else:
            key = ("utility", "")
        groups.setdefault(key, []).append(t)
    return [groups[k] for k in sorted(groups)]


class GameStats:
    """Collects analytics for one live game by listening on its hooks."""

    def __init__(self, game):
        self.game = game
        ownable = [t for t in game.board.tiles
                   if isinstance(t, (StreetProperty, Railroad, Utility))]
        self.groups = _build_groups(ownable)
        self._group_of = {id(t): grp for grp in self.groups for t in grp}

        names = [p.name for p in game.players]
        self.turns = 0
        self.auctions_won = {n: 0 for n in names}
        self.first_monopoly_turn = {n: None for n in names}
        self._ever_monopolies = {n: set() for n in names}
        # Readable records of set-denying acquisitions.
        self.blocks = []            # list of dicts: by, victim, tile, set, source
        # Winning auction bids, split by whether the winner completed their set.
        self.set_completion_bids = []   # list of (bid, price)
        self.non_completion_bids = []   # list of (bid, price)
        # Tiles held as inheritance from a bankrupted opponent -- excluded from
        # the "completed through play" monopoly tempo.
        self._inherited_ids = set()

        self._install_hooks()

    # -- hook wiring ------------------------------------------------------

    def _install_hooks(self):
        self.game.on_auction_end = self._on_auction_end
        self.game.on_acquire = self._on_acquire
        orig_declare = self.game.declare_bankrupt

        def tracking_declare(player, creditor=None):
            if creditor is not None and not creditor.bankrupt:
                for prop in player.properties:
                    self._inherited_ids.add(id(prop))   # estate passes on
            else:
                for prop in player.properties:
                    self._inherited_ids.discard(id(prop))  # back to the bank
            orig_declare(player, creditor)

        self.game.declare_bankrupt = tracking_declare

    def _completes_for(self, player, prop, group):
        """Whether ``player`` owns every tile in ``group`` except ``prop``."""
        return all(t.owner is player for t in group if t is not prop)

    def _on_auction_end(self, prop, winner, bid):
        if winner is None:
            return
        self.auctions_won[winner.name] += 1
        group = self._group_of.get(id(prop))
        # Ownership has already transferred to ``winner`` at this point.
        completed = group is not None and self._completes_for(winner, prop, group)
        (self.set_completion_bids if completed
         else self.non_completion_bids).append((bid, prop.price))

    def _on_acquire(self, player, prop, source="trade"):
        # A legitimate (re)acquisition clears any inherited flag on the tile.
        self._inherited_ids.discard(id(prop))
        group = self._group_of.get(id(prop))
        if group is None:
            return
        # Blocking: some solvent opponent owned every other tile in the group,
        # so taking this one denied them the set.
        for o in self.game.players:
            if o is player or o.bankrupt:
                continue
            if self._completes_for(o, prop, group):
                self.blocks.append({
                    "by": player.name,
                    "victim": o.name,
                    "tile": prop.name,
                    "set": _group_label(group),
                    "source": source,
                })

    # -- per-turn snapshot ------------------------------------------------

    def snapshot(self, turn):
        """Records any newly-completed monopolies at ``turn`` (call each turn)."""
        self.turns = turn
        for p in self.game.players:
            if p.bankrupt:
                continue
            for grp in self.groups:
                if all(t.owner is p and id(t) not in self._inherited_ids
                       for t in grp):
                    label = _group_label(grp)
                    if label not in self._ever_monopolies[p.name]:
                        self._ever_monopolies[p.name].add(label)
                        if self.first_monopoly_turn[p.name] is None:
                            self.first_monopoly_turn[p.name] = turn

    # -- reporting --------------------------------------------------------

    @staticmethod
    def _mean_bid(bids):
        """``(mean_bid, mean_ratio)`` for a list of ``(bid, price)`` pairs, or
        ``(None, None)`` if empty."""
        if not bids:
            return None, None
        mean_bid = sum(b for b, _ in bids) / len(bids)
        ratios = [b / p for b, p in bids if p]
        mean_ratio = sum(ratios) / len(ratios) if ratios else None
        return mean_bid, mean_ratio

    def summary(self):
        """A plain-data summary dict for the UI to render."""
        g = self.game
        winner = g.winner()
        sc_bid, sc_ratio = self._mean_bid(self.set_completion_bids)
        nc_bid, nc_ratio = self._mean_bid(self.non_completion_bids)

        players = []
        for p in g.players:
            players.append({
                "name": p.name,
                "bankrupt": p.bankrupt,
                "properties": len(p.properties),
                "monopolies": sorted(self._ever_monopolies[p.name]),
                "first_monopoly_turn": self.first_monopoly_turn[p.name],
                "auctions_won": self.auctions_won[p.name],
            })

        return {
            "turns": self.turns,
            "winner": winner.name if winner else None,
            "winner_first_monopoly_turn": (
                self.first_monopoly_turn[winner.name] if winner else None),
            "players": players,
            "total_auctions_won": sum(self.auctions_won.values()),
            "blocks": list(self.blocks),
            "set_completion_bid": sc_bid,
            "set_completion_ratio": sc_ratio,
            "set_completion_count": len(self.set_completion_bids),
            "non_completion_bid": nc_bid,
            "non_completion_ratio": nc_ratio,
            "non_completion_count": len(self.non_completion_bids),
        }
