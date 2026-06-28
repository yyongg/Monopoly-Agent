"""Class for the community chest and chance decks."""

import random

class Deck:
    """Class for a deck"""
    def __init__(self,cards):
        self.cards = cards[:]
        random.shuffle(self.cards)

    def draw(self):
        """Draw a card from the deck"""
        return self.cards.pop(0)

    def return_card(self, card):
        """Return a drawn card to the bottom of the pile"""
        self.cards.append(card)

    def shuffle(self):
        """Shuffle the deck"""
        random.shuffle(self.cards)
