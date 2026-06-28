"""Factories that build the standard Chance and Community Chest decks.

Cards requiring board features that are not yet implemented (nearest
railroad/utility, per-house repairs) are intentionally omitted; add them once
that logic exists. Board positions assume the standard Monopoly layout.
"""

from models.deck import Deck
from models.card import (
    MoneyCard,
    PerPlayerCard,
    GoToJailCard,
    AdvanceToCard,
    MoveCard,
    GetOutOfJailCard,
)

# Standard board positions referenced by movement cards.
GO = 0
ST_CHARLES_PLACE = 11
ILLINOIS_AVENUE = 24
READING_RAILROAD = 5
BOARDWALK = 39


def build_chance_deck():
    """Returns a shuffled Deck of Chance cards."""
    cards = [
        AdvanceToCard("Advance to GO (Collect $200)", GO),
        AdvanceToCard("Advance to Illinois Avenue", ILLINOIS_AVENUE),
        AdvanceToCard("Advance to St. Charles Place", ST_CHARLES_PLACE),
        AdvanceToCard("Take a trip to Reading Railroad", READING_RAILROAD),
        AdvanceToCard("Take a walk on the Boardwalk", BOARDWALK),
        MoveCard("Go back 3 spaces", -3),
        GoToJailCard("Go to Jail. Go directly to Jail, do not pass GO"),
        MoneyCard("Bank pays you dividend of $50", 50),
        MoneyCard("Your building loan matures. Collect $150", 150),
        MoneyCard("Speeding fine $15", -15),
        PerPlayerCard("Elected Chairman of the Board. Pay each player $50", -50),
        GetOutOfJailCard(),
    ]
    return Deck(cards)


def build_community_deck():
    """Returns a shuffled Deck of Community Chest cards."""
    cards = [
        AdvanceToCard("Advance to GO (Collect $200)", GO),
        GoToJailCard("Go to Jail. Go directly to jail, do not pass GO"),
        MoneyCard("Bank error in your favor. Collect $200", 200),
        MoneyCard("Doctor's fee. Pay $50", -50),
        MoneyCard("From sale of stock you get $50", 50),
        MoneyCard("Holiday fund matures. Collect $100", 100),
        MoneyCard("Income tax refund. Collect $20", 20),
        MoneyCard("Life insurance matures. Collect $100", 100),
        MoneyCard("Pay hospital fees of $100", -100),
        MoneyCard("Pay school fees of $50", -50),
        MoneyCard("Receive $25 consultancy fee", 25),
        MoneyCard("You inherit $100", 100),
        PerPlayerCard("It is your birthday. Collect $10 from every player", 10),
        GetOutOfJailCard(),
    ]
    return Deck(cards)
