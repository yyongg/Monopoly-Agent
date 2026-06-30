"""Headless entry points for the Monopoly engine.

Two ways to drive the game without the pygame UI:

* ``baseline_game()`` runs a fully self-playing game where every player uses the
  engine's built-in baseline decisions (buy any affordable property, always roll
  in jail, no building or trading). Useful as a quick smoke test of the rules.

* ``rl_demo()`` drives the Gym-style :class:`engine.rl_env.MonopolyEnv` with a
  random *legal* agent, showing how an RL agent interacts with the game: observe,
  read the action mask, pick a legal action, step. Swap ``RandomAgent`` for a
  trained policy to evaluate it.

Run ``python play.py`` for the RL demo, or ``python play.py baseline`` for the
self-playing baseline game.
"""

import sys

import numpy as np

from engine.game import Game
from engine.rl_env import MonopolyEnv
from models.board import Board
from models.player import Player
from data.decks import build_chance_deck, build_community_deck
from data.board_tiles import build_board_tiles

# Safety cap so two surviving players who never bankrupt each other still end.
MAX_TURNS = 5000


def baseline_game():
    """Runs a full self-playing game to completion and reports the result."""
    print("Starting new baseline game...")

    players = [Player('Red'), Player('Blue'), Player('Green'), Player('Yellow')]
    board = Board(build_board_tiles())
    game = Game(players, board, build_chance_deck(), build_community_deck())

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


class RandomAgent:
    """Baseline RL agent: picks a uniformly random *legal* action.

    The action mask in ``info["action_mask"]`` marks the actions valid for the
    current decision; a real policy would consume the observation instead of
    ignoring it. This is the minimal example of the env's interaction contract.
    """

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()

    def act(self, observation, action_mask):
        legal = np.flatnonzero(action_mask)
        return int(self.rng.choice(legal))


def rl_demo(episodes=20, seat=0):
    """Plays ``episodes`` games with a RandomAgent and reports win rate."""
    print(f"Running {episodes} RL episodes "
          f"(RandomAgent controlling seat {seat})...")

    env = MonopolyEnv(seat=seat)
    agent = RandomAgent()
    wins = 0
    returns = []

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.act(obs, info["action_mask"])
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        returns.append(total_reward)
        if info.get("won"):
            wins += 1
        outcome = "won" if info.get("won") else f"winner={info.get('winner')}"
        print(f"  episode {ep + 1:>3}: return={total_reward:+.2f}  {outcome}")

    env.close()
    print(f"\nRandomAgent won {wins}/{episodes} "
          f"({100 * wins / episodes:.0f}%), "
          f"mean return {np.mean(returns):+.2f}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "baseline":
        baseline_game()
    else:
        rl_demo()


if __name__ == "__main__":
    main()
