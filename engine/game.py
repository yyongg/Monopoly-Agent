"""Create a new game"""

import random

class Game:
    """Class to define the game."""
    def __init__(self,players,board):
        self.players = players
        self.board = board
        self.current_player = 0

    def roll_dice(self):
        """Rolls two 6-sided die and sums the results."""
        return random.randint(1,6) + random.randint(1,6)

    def step(self):
        """Plays out one Monopoly turn for one player."""

        # define player to move in this step
        player = self.players[self.current_player]

        # roll dice
        roll = self.roll_dice()
        player.move(roll)

        # gets the tile the player is on and take action on tile
        tile = self.board.get_tile(player.pos)
        tile.on_land(self,player)

        # change action to next player
        self.current_player = (self.current_player + 1) % len(self.players)
