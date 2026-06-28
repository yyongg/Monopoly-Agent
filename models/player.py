"""Player class"""

class Player:
    """
    Sets up each player
    """
    def __init__(self,name):
        self.name = name
        self.balance = 1500
        self.pos = 0
        self.bankrupt = False

        # inventory
        self.properties = []
        self.cards = []

        # jail properties
        self.in_jail = False
        self.jail_turns = 0
        self.get_out_of_jail_cards = 0

        # number of doubles on a turn
        self.double_count = 0

    def move(self, steps):
        """
        Method to move player on the board.

        Args:
            steps (int): Number of steps forward the player will take. Take mod of
            board length to account for circular board.
        """
        self.pos = (self.pos + steps) % 40
