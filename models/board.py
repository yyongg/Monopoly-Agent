"""Board class for the tiles on the board in order"""

class Board:
    """Class for the board"""

    def __init__(self, tiles):
        self.tiles = tiles
        self.length = len(tiles)

    def get_tile(self, pos):
        """
        Returns the tile at input position

        Args:
            pos (int): Position on the board
        """
        return self.tiles[pos]
