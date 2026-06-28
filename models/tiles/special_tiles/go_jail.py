"""Class for go to jail tile."""

from models.tile import Tile

class GoJail(Tile):
    """Go to jail tile for board. Inherits from Tile class."""

    def __init__(self,pos):
        super().__init__("Go to Jail", pos)

    def on_land(self, game, player):
        game.send_to_jail(player)
