"""Generates a clean, minimalist Monopoly board image (no property names).

The board is drawn purely from the game's own tile data and the measured tile
geometry in ``ui.board_layout`` (the same EDGES the UI uses to place tokens), so
the generated image lines up exactly with where the app draws pieces.

Run from the project root:

    python tools/generate_board.py

It writes ``assets/board_minimal.png``. The board is rendered at 2x and
downscaled for clean, anti-aliased edges. Re-run it to regenerate the asset.
"""

import math
import os
import sys

# Allow running directly (``python tools/generate_board.py``) from any cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Headless: no window needed to draw onto an off-screen surface.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402  (import after setting the SDL env vars)

from ui.board_layout import EDGES, tile_grid, interior_offset  # noqa: E402
from ui.app import GROUP_COLORS, RAILROAD_COLOR  # noqa: E402
from data.board_tiles import build_board_tiles  # noqa: E402
from models.tiles.properties.street_property import StreetProperty  # noqa: E402
from models.tiles.properties.railroad import Railroad  # noqa: E402
from models.tiles.properties.utility import Utility  # noqa: E402
from models.tiles.special_tiles.tax import Tax  # noqa: E402
from models.tiles.special_tiles.chance_card import ChanceCard  # noqa: E402
from models.tiles.special_tiles.community_chest import CommunityChest  # noqa: E402
from models.tiles.special_tiles.go import Go  # noqa: E402
from models.tiles.special_tiles.jail import Jail  # noqa: E402
from models.tiles.special_tiles.go_jail import GoJail  # noqa: E402
from models.tiles.special_tiles.free_parking import FreeParking  # noqa: E402

# Render at 2x for anti-aliasing, then smoothscale down to OUT.
OUT = 894
SCALE = 2
PX = OUT * SCALE

# Minimalist flat palette: lots of warm white, thin lines, flat accents.
BASE = (250, 249, 245)        # tile + outer surface
CENTER = (245, 244, 239)      # interior play area
LINE = (206, 205, 198)        # thin grid lines between tiles
FRAME = (54, 54, 52)          # thin outer / inner frames
ICON = (96, 96, 92)           # neutral icon stroke
CHANCE = (244, 149, 31)       # orange accent
CHEST = (0, 112, 186)         # blue accent
UTILITY = (176, 176, 170)     # muted utility band


def f(frac):
    """Board fraction -> pixel on the 2x surface."""
    return frac * PX


def tile_box(pos):
    """Returns the (x, y, w, h) pixel rect of a tile on the 2x surface."""
    row, col = tile_grid(pos)
    x0, x1 = f(EDGES[col]), f(EDGES[col + 1])
    y0, y1 = f(EDGES[row]), f(EDGES[row + 1])
    return x0, y0, x1 - x0, y1 - y0


def draw_band(surf, pos, color):
    """Draws the street/railroad color band along a tile's inner edge."""
    x, y, w, h = tile_box(pos)
    dx, dy = interior_offset(pos)
    depth = (h if dy else w) * 0.30
    if dy < 0:      # bottom edge -> band on top
        rect = pygame.Rect(x, y, w, depth)
    elif dy > 0:    # top edge -> band on bottom
        rect = pygame.Rect(x, y + h - depth, w, depth)
    elif dx > 0:    # left edge -> band on right
        rect = pygame.Rect(x + w - depth, y, depth, h)
    else:           # right edge -> band on left
        rect = pygame.Rect(x, y, depth, h)
    pygame.draw.rect(surf, color, rect)
    pygame.draw.rect(surf, LINE, rect, max(1, SCALE))


def tile_center(pos):
    x, y, w, h = tile_box(pos)
    return x + w / 2, y + h / 2


def draw_bolt(surf, cx, cy, r):
    """A minimalist lightning bolt (electric company)."""
    pts = [(cx + 0.15 * r, cy - r), (cx - 0.5 * r, cy + 0.15 * r),
           (cx - 0.02 * r, cy + 0.15 * r), (cx - 0.2 * r, cy + r),
           (cx + 0.55 * r, cy - 0.2 * r), (cx + 0.08 * r, cy - 0.2 * r)]
    pygame.draw.polygon(surf, ICON, pts)


def draw_drop(surf, cx, cy, r):
    """A minimalist water droplet (water works)."""
    pygame.draw.circle(surf, ICON, (int(cx), int(cy + 0.25 * r)), int(r * 0.62), max(2, SCALE * 2))
    pygame.draw.polygon(surf, ICON, [(cx, cy - r), (cx - 0.5 * r, cy + 0.1 * r),
                                     (cx + 0.5 * r, cy + 0.1 * r)])


def draw_diamond(surf, cx, cy, r, fill=False):
    pts = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    pygame.draw.polygon(surf, ICON, pts, 0 if fill else max(2, SCALE * 2))


def draw_cross(surf, cx, cy, r):
    """Railroad-crossing style X."""
    w = max(3, SCALE * 3)
    pygame.draw.line(surf, ICON, (cx - r, cy - r), (cx + r, cy + r), w)
    pygame.draw.line(surf, ICON, (cx + r, cy - r), (cx - r, cy + r), w)


def draw_bars(surf, cx, cy, r):
    """Jail bars."""
    box = pygame.Rect(cx - r, cy - r, 2 * r, 2 * r)
    pygame.draw.rect(surf, ICON, box, max(2, SCALE * 2), border_radius=int(r * 0.2))
    for i in range(1, 4):
        bx = box.left + box.width * i / 4
        pygame.draw.line(surf, ICON, (bx, box.top), (bx, box.bottom), max(2, SCALE * 2))


def draw_arrow(surf, cx, cy, r, angle):
    """A simple arrow used for GO / Go-To-Jail, pointing at ``angle`` radians."""
    tip = (cx + r * math.cos(angle), cy + r * math.sin(angle))
    tail = (cx - r * math.cos(angle), cy - r * math.sin(angle))
    pygame.draw.line(surf, ICON, tail, tip, max(3, SCALE * 3))
    for da in (math.radians(150), math.radians(-150)):
        bx = tip[0] + r * 0.55 * math.cos(angle + da)
        by = tip[1] + r * 0.55 * math.sin(angle + da)
        pygame.draw.line(surf, ICON, tip, (bx, by), max(3, SCALE * 3))


def main():
    pygame.init()
    surf = pygame.Surface((PX, PX))
    surf.fill(BASE)

    # Interior play area + inner frame (the empty center of the board).
    inner = pygame.Rect(f(EDGES[1]), f(EDGES[1]),
                        f(EDGES[10]) - f(EDGES[1]), f(EDGES[10]) - f(EDGES[1]))
    pygame.draw.rect(surf, CENTER, inner)

    tiles = build_board_tiles()
    for tile in tiles:
        pos = tile.pos
        x, y, w, h = tile_box(pos)
        rect = pygame.Rect(round(x), round(y), round(w), round(h))
        pygame.draw.rect(surf, BASE, rect)
        pygame.draw.rect(surf, LINE, rect, max(1, SCALE))
        cx, cy = tile_center(pos)
        r = min(w, h) * 0.20

        if isinstance(tile, StreetProperty):
            draw_band(surf, pos, GROUP_COLORS[tile.color])
        elif isinstance(tile, Railroad):
            draw_band(surf, pos, RAILROAD_COLOR)
            draw_cross(surf, cx, cy, r * 0.8)
        elif isinstance(tile, Utility):
            draw_band(surf, pos, UTILITY)
            if "Electric" in tile.name:
                draw_bolt(surf, cx, cy, r)
            else:
                draw_drop(surf, cx, cy, r)
        elif isinstance(tile, Tax):
            draw_diamond(surf, cx, cy, r, fill="Luxury" in tile.name)
        elif isinstance(tile, ChanceCard):
            pygame.draw.circle(surf, CHANCE, (int(cx), int(cy)), int(r))
        elif isinstance(tile, CommunityChest):
            box = pygame.Rect(0, 0, 2 * r, 2 * r)
            box.center = (cx, cy)
            pygame.draw.rect(surf, CHEST, box, border_radius=int(r * 0.25))
        elif isinstance(tile, Go):
            draw_arrow(surf, cx, cy, r * 1.1, math.radians(180))
        elif isinstance(tile, Jail):
            draw_bars(surf, cx, cy, r)
        elif isinstance(tile, GoJail):
            draw_arrow(surf, cx, cy, r * 1.1, math.radians(135))
        elif isinstance(tile, FreeParking):
            pygame.draw.circle(surf, ICON, (int(cx), int(cy)), int(r), max(2, SCALE * 2))

    # Frames: thin lines around the whole ring and around the center.
    outer = pygame.Rect(f(EDGES[0]), f(EDGES[0]),
                        f(EDGES[11]) - f(EDGES[0]), f(EDGES[11]) - f(EDGES[0]))
    pygame.draw.rect(surf, FRAME, outer, max(2, SCALE * 2))
    pygame.draw.rect(surf, FRAME, inner, max(2, SCALE * 2))

    out = pygame.transform.smoothscale(surf, (OUT, OUT))
    dest = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "assets", "board_minimal.png")
    pygame.image.save(out, dest)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
