"""Class for jail tile."""

from models.tile import Tile

class Jail(Tile):
    """Jail tile for board. Inherits from Tile class."""

    def __init__(self,pos):
        super().__init__("Jail", pos)

    def on_land(self, game, player):
        pass
