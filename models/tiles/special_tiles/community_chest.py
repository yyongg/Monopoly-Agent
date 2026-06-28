"""Class for commmunity chest card tile."""

from models.tile import Tile

class CommunityChest(Tile):
    """Community Chest tile for board. Inherits from Tile class."""

    def __init__(self,pos):
        super().__init__("Community Chest", pos)

    def on_land(self, game, player):
        game.chance_deck.draw(game, player)
