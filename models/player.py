"""Player class"""

class Player:
    """
    Sets up each player
    """
    def __init__(self,name):
        self.name = name
        self.balance = 1500
        self.pos = 0
        self.properties = []

    def move(self, steps):
        """
        Method to move player on the board.

        Args:
            steps (int): Number of steps forward the player will take. Take mod of
            board length to account for circular board.
        """
        self.pos = (self.pos + steps) % 40
