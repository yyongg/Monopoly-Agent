"""Tiles class"""

class Tile:
    """
    All tiles will inherit from this class.
    """
    def __init__(self, name, pos):
        """
        Initialize tile class

        self.name (str): Name of the tile
        self.pos (int): Position of the tile on the board
        """
        self.name = name
        self.pos = pos

    def on_land(self, game, player):
        """
        Defines what happens when a player lands here"

        Args:
            game (type[Game]): Game class
            player (type[Player]): Player class
        """

        raise NotImplementedError
