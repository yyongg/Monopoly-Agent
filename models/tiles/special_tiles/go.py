"""Class for go tile."""

from models.tile import Tile

class Go(Tile):
    """Go tile for board. Inherits from Tile class."""

    def __init__(self,pos):
        super().__init__("GO", pos)

    def on_land(self, game, player):
        pass
