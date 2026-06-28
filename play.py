"""Main file"""

from engine.game import Game
from models.board import Board
from models.player import Player
from data.decks import build_chance_deck, build_community_deck
from data.board_tiles import build_board_tiles

# Safety cap so two surviving players who never bankrupt each other still end.
MAX_TURNS = 5000


def start_game():
    """Runs a full self-playing game to completion and reports the result."""
    print("Starting new game...")

    # initialize players, board, chance & community decks
    players = [Player('Red'), Player('Blue'), Player('Green'), Player('Yellow')]
    board = Board(build_board_tiles())
    chance_deck = build_chance_deck()
    community_deck = build_community_deck()

    # initialize game
    game = Game(players, board, chance_deck, community_deck)

    # play turns until one player is left or the turn cap is reached
    turns = 0
    while not game.is_over() and turns < MAX_TURNS:
        game.step()
        turns += 1

    winner = game.winner()
    if winner is not None:
        print(f"{winner.name} wins after {turns} turns!")
    else:
        print(f"No winner after {turns} turns. Standings:")

    for player in players:
        status = "bankrupt" if player.bankrupt else f"${player.balance}"
        print(f"  {player.name}: {status} ({len(player.properties)} properties)")


if __name__ == "__main__":
    start_game()
