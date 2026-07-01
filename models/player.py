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

    def decide_bid(self, prop, min_bid=0):
        """
        Decides whether to bid in an ascending auction for a property.

        Called once per bidding round by ``Game.run_auction``: ``min_bid`` is the
        smallest bid that beats the current standing bid. Return a value
        ``>= min_bid`` to raise (taking the lead) or anything below it (e.g. 0)
        to drop out. Like :meth:`decide_purchase`, this is a hook an RL policy
        overrides; the default baseline never bids, leaving auctions to be
        contested by policy-driven seats. The engine clamps any bid to the
        player's cash.

        Args:
            prop (type[Property]): The property being auctioned.
            min_bid (int): The minimum bid needed to beat the standing bid.

        Returns:
            int: A bid ``>= min_bid`` to stay in, or below it to pass.
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
