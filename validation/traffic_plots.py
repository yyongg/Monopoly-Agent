"""Visualise board landing traffic and traffic-derived value from the table
produced by ``validation.board_visits`` (``runs/board_visits.json``).

Produces, under ``--out`` (default ``runs/traffic_analysis/``):

  1. ``board_heatmap.png``     -- the 40 tiles laid out as the board ring,
                                  shaded by landing share (the "most popular
                                  tiles" at a glance).
  2. ``landing_frequency.png`` -- every tile ranked by landing share, as a
                                  multiple of an even 1/40 board.
  3. ``expected_income.png``   -- traffic x base rent per tile: exactly the
                                  ``MonopolyEnv._expected_income`` feature the
                                  agent uses (traffic x nominal single rent).
  4. ``traffic_x_hotel.png``   -- traffic x full-hotel rent: what a developed
                                  tile is expected to earn, i.e. why the build
                                  bonus favours busy colour groups.
  5. ``buy_yield.png``         -- (traffic x base rent) / price, as a percent:
                                  expected rent per dollar of purchase price --
                                  which tiles are cheap for what they earn (the
                                  cheap sets shine here, unlike the absolute
                                  charts above).
  6. ``development_roi.png``   -- traffic x (hotel rent - base rent) per dollar
                                  sunk into houses (5 x house cost): rent gained
                                  per build dollar -- why the orange/red belt is
                                  the money set. Streets only (only streets build).

These last two divide by cost, so they answer "best value", where charts 3-4
answer "most earning power". Both matter; neither is used by the agent directly
(cost already enters the agent via net-worth reward and the trade base price).

Usage:
    PYTHONPATH=. python -m validation.traffic_plots
    PYTHONPATH=. python -m validation.traffic_plots --json runs/board_visits.json
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine.observation import base_rent
from data.board_tiles import build_board_tiles
from models.tiles.properties.street_property import StreetProperty

# Group -> colour for the bars, so streets read as their board colour.
GROUP_COLOR = {
    "brown": "#8B5A2B", "light_blue": "#AEE0F5", "pink": "#D93A96",
    "orange": "#F7941E", "red": "#ED1B24", "yellow": "#FEF200",
    "green": "#1FA84A", "dark_blue": "#0072BB",
}
TYPE_COLOR = {"railroad": "#333333", "utility": "#9AA0A6"}


def _tile_color(t):
    if t["type"] == "street":
        return GROUP_COLOR.get(t["color"], "#BBBBBB")
    return TYPE_COLOR.get(t["type"], "#DDDDDD")


def _ring_cell(pos):
    """(row, col) of board position ``pos`` on an 11x11 grid ring, GO at the
    bottom-right corner and numbering counter-clockwise like a real board."""
    if pos <= 10:
        return 10, 10 - pos            # bottom edge, right -> left
    if pos <= 20:
        return 20 - pos, 0             # left edge, bottom -> top
    if pos <= 30:
        return 0, pos - 20             # top edge, left -> right
    return pos - 30, 10                # right edge, top -> bottom


def _short(name):
    return (name[:12] + "…") if len(name) > 13 else name


def board_heatmap(tiles, path):
    grid = np.full((11, 11), np.nan)
    for t in tiles:
        r, c = _ring_cell(t["pos"])
        grid[r, c] = t["frequency"] * 100.0

    fig, ax = plt.subplots(figsize=(11, 11))
    im = ax.imshow(grid, cmap="YlOrRd", vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    for t in tiles:
        r, c = _ring_cell(t["pos"])
        ax.text(c, r - 0.22, _short(t["name"]), ha="center", va="center",
                fontsize=6.5, wrap=True)
        ax.text(c, r + 0.2, f"{t['frequency'] * 100:.2f}%", ha="center",
                va="center", fontsize=7, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Landing share per tile (% of all landings)", fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.6, label="% of landings")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def landing_frequency(tiles, board_size, path):
    ts = sorted(tiles, key=lambda t: t["frequency"], reverse=True)
    names = [t["name"] for t in ts]
    mult = [t["frequency"] * board_size for t in ts]  # vs an even 1/40 board
    colors = [_tile_color(t) for t in ts]

    fig, ax = plt.subplots(figsize=(10, 12))
    y = np.arange(len(ts))
    ax.barh(y, mult, color=colors, edgecolor="#00000033")
    ax.axvline(1.0, color="k", ls="--", lw=1, label="even (1/40)")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("landing traffic  (x an average tile)")
    ax.set_title("How often each tile is landed on")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _value_chart(tiles, values, title, xlabel, path, top=None):
    pairs = sorted(zip(tiles, values), key=lambda p: p[1], reverse=True)
    if top:
        pairs = pairs[:top]
    ts = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    colors = [_tile_color(t) for t in ts]

    fig, ax = plt.subplots(figsize=(10, max(6, 0.32 * len(ts))))
    y = np.arange(len(ts))
    ax.barh(y, vals, color=colors, edgecolor="#00000033")
    ax.set_yticks(y)
    ax.set_yticklabels([t["name"] for t in ts], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", default=os.path.join("runs", "board_visits.json"))
    parser.add_argument("--out", default=os.path.join("runs", "traffic_analysis"))
    parser.add_argument("--top", type=int, default=22,
                        help="tiles to show in the value charts")
    args = parser.parse_args()

    with open(args.json) as f:
        data = json.load(f)
    tiles = data["tiles"]
    board_size = data["meta"]["board_size"]
    os.makedirs(args.out, exist_ok=True)

    # Map each ownable tile to its engine object so rent uses the same
    # ``base_rent`` / rent_table the agent sees.
    by_pos = {t.pos: t for t in build_board_tiles()}

    ownable, exp_income, exp_hotel, buy_yield = [], [], [], []
    dev_tiles, dev_roi = [], []
    for t in tiles:
        obj = by_pos.get(t["pos"])
        if not hasattr(obj, "price"):        # not an ownable tile
            continue
        traffic = t["frequency"] * board_size
        ownable.append(t)
        income = traffic * base_rent(obj)
        exp_income.append(income)
        hotel_rent = obj.rent_table[-1] if isinstance(obj, StreetProperty) \
            else base_rent(obj)
        exp_hotel.append(traffic * hotel_rent)
        # Buy yield: rent per $ of price, as a percent (cost-adjusted).
        buy_yield.append(100.0 * income / obj.price if obj.price else 0.0)
        # Development ROI: extra rent from a full hotel per $ sunk into houses
        # (a hotel is 5 house-purchases). Streets only -- only streets build.
        if isinstance(obj, StreetProperty):
            build_cost = 5 * obj.house_cost()
            rent_gain = traffic * (obj.rent_table[-1] - base_rent(obj))
            dev_tiles.append(t)
            dev_roi.append(rent_gain / build_cost if build_cost else 0.0)

    board_heatmap(tiles, os.path.join(args.out, "board_heatmap.png"))
    landing_frequency(tiles, board_size,
                      os.path.join(args.out, "landing_frequency.png"))
    _value_chart(ownable, exp_income,
                 "Traffic × base rent  (the agent's _expected_income feature)",
                 "expected rent per opponent pass  ($ · traffic)",
                 os.path.join(args.out, "expected_income.png"), top=args.top)
    _value_chart(ownable, exp_hotel,
                 "Traffic × full-hotel rent  (developed earning power)",
                 "expected hotel rent  ($ · traffic)",
                 os.path.join(args.out, "traffic_x_hotel.png"), top=args.top)
    _value_chart(ownable, buy_yield,
                 "Buy yield: (traffic × base rent) / price  (best value buys)",
                 "expected rent per $ of price  (%, · traffic)",
                 os.path.join(args.out, "buy_yield.png"), top=args.top)
    _value_chart(dev_tiles, dev_roi,
                 "Development ROI: traffic × rent gain per $ of houses  "
                 "(a hotel = 5 house costs)",
                 "extra hotel rent per build $  (· traffic)",
                 os.path.join(args.out, "development_roi.png"), top=args.top)

    print(f"wrote 6 figures -> {args.out}/")
    # A compact top-10 table to stdout as well.
    top10 = sorted(tiles, key=lambda t: t["frequency"], reverse=True)[:10]
    print(f"\n{'tile':<22}{'share':>8}{'vs even':>9}")
    for t in top10:
        print(f"{t['name']:<22}{t['frequency']*100:>7.2f}%"
              f"{t['frequency']*board_size:>8.2f}x")


if __name__ == "__main__":
    main()
