"""Class for Free Parking card tile."""

from models.tile import Tile

class ChanceCard(Tile):
    """Free Parking tile for board. Inherits from Tile class."""

    def __init__(self,pos):
        super().__init__("Free Parking", pos)

    def on_land(self, game, player):
        pass
