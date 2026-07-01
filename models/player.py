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

        # jail properties
        self.in_jail = False
        self.jail_turns = 0
        self.jail_cards = []  # held Get Out of Jail Free cards

        # number of doubles on a turn
        self.double_count = 0

    def decide_purchase(self, prop):
        """
        Decides whether to buy an unowned property offered to this player.

        This is the natural hook for an RL policy: the default baseline buys any
        property the player can afford, and an agent overrides it to make its
        own choice.

        Args:
            prop (type[Property]): The property being offered.

        Returns:
            bool: True to buy, False to decline.
        """
        return self.balance >= prop.price

    def decide_bid(self, prop):
        """
        Decides how much to bid for a property that has gone to auction.

        Like :meth:`decide_purchase`, this is a hook an RL policy overrides. The
        default baseline never bids (returns 0), leaving auctions to be contested
        by policy-driven seats; the engine clamps any bid to the player's cash.

        Args:
            prop (type[Property]): The property being auctioned.

        Returns:
            int: The player's sealed bid (0 to pass on the auction).
        """
        return 0

    def move(self, steps):
        """
        Method to move player on the board.

        Args:
            steps (int): Number of steps forward the player will take. Take mod of
            board length to account for circular board.

        Returns:
            bool: True if the move passed (or landed on) GO, so the caller can
            award the GO salary. Backward moves never count as passing GO.
        """
        new_pos = self.pos + steps
        self.pos = new_pos % 40
        return steps > 0 and new_pos >= 40
