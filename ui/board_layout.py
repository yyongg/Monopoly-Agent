"""Maps board positions (0-39) to pixel coordinates on the board image.

The board is an 11x11 ring of tiles, but the printed tiles are NOT uniform: the
play area is inset by a margin and the four corner squares are noticeably larger
than the regular edge tiles. Using a naive ``board_px / 11`` grid therefore puts
tokens slightly off their tiles (most visibly near the corners).

The constants below are measured from the actual board image (assets/board.jpg,
894x894) as fractions of the board's side, so token and highlight positions land
on the true printed tiles at any render size.

Position 0 (GO) sits in the bottom-right corner and positions increase
counter-clockwise: left along the bottom edge to Jail (10, bottom-left), up the
left edge to Free Parking (20, top-left), right along the top edge to Go To Jail
(30, top-right), then down the right edge back toward GO.
"""

# The board is 11 tiles per side.
GRID = 11

# Center of each of the 11 rows/columns, as a fraction of the board side.
# Measured from board.jpg; index 0 and 10 are the larger corner squares.
CENTERS = (
    0.08333, 0.17114, 0.25336, 0.33501, 0.41667, 0.49888,
    0.58110, 0.66275, 0.74441, 0.82606, 0.91501,
)

# The 12 tile boundaries (fractions of the board side) separating the 11
# rows/columns, used to draw a rectangle around a whole tile.
EDGES = (
    0.03691, 0.12975, 0.21253, 0.29418, 0.37584, 0.45749,
    0.54027, 0.62192, 0.70358, 0.78523, 0.86689, 0.96309,
)

# Average regular tile size as a fraction of the board, for small insets
# (house markers, ownership dots) that don't need exact per-tile sizing.
TILE_FRAC = 0.0817


def tile_grid(pos):
    """
    Returns the (row, col) cell of a board position in the 11x11 grid, with
    row 0 at the top and col 0 at the left.

    Args:
        pos (int): Board position 0-39.
    """
    pos %= 40
    if pos <= 10:
        # Bottom edge: GO at (10, 10) leftward to Jail at (10, 0).
        return (10, 10 - pos)
    if pos <= 20:
        # Left edge: upward from (9, 0) to Free Parking at (0, 0).
        return (10 - (pos - 10), 0)
    if pos <= 30:
        # Top edge: rightward from (0, 1) to Go To Jail at (0, 10).
        return (0, pos - 20)
    # Right edge: downward from (1, 10) to (9, 10).
    return (pos - 30, 10)


def tile_center(pos, board_x, board_y, board_px):
    """
    Returns the (x, y) pixel center of a board position's tile, using the
    measured (non-uniform) tile centers so tokens sit on the printed tile.

    Args:
        pos (int): Board position 0-39.
        board_x (int): Left pixel of the board image on screen.
        board_y (int): Top pixel of the board image on screen.
        board_px (int): On-screen pixel size of the (square) board.
    """
    row, col = tile_grid(pos)
    cx = board_x + CENTERS[col] * board_px
    cy = board_y + CENTERS[row] * board_px
    return (int(cx), int(cy))


def tile_rect(pos, board_x, board_y, board_px):
    """
    Returns the (x, y, w, h) pixel rectangle covering a whole tile, using the
    measured tile boundaries. Useful for highlighting the current tile.

    Args:
        pos (int): Board position 0-39.
        board_x (int): Left pixel of the board image on screen.
        board_y (int): Top pixel of the board image on screen.
        board_px (int): On-screen pixel size of the (square) board.
    """
    row, col = tile_grid(pos)
    x0 = board_x + EDGES[col] * board_px
    x1 = board_x + EDGES[col + 1] * board_px
    y0 = board_y + EDGES[row] * board_px
    y1 = board_y + EDGES[row + 1] * board_px
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def interior_offset(pos):
    """
    Returns a unit (dx, dy) pointing from a perimeter tile toward the board's
    interior. Used to place house markers just inside the edge a tile sits on.
    Streets never occupy the four corners, so only the four edges are handled.

    Args:
        pos (int): Board position 0-39.
    """
    pos %= 40
    if pos < 10:
        return (0, -1)   # bottom edge -> up
    if pos < 20:
        return (1, 0)    # left edge -> right
    if pos < 30:
        return (0, 1)    # top edge -> down
    return (-1, 0)       # right edge -> left


def token_offset(index, board_px):
    """
    Returns a small (dx, dy) so up to four tokens on one tile don't fully
    overlap, arranged in a 2x2 cluster by player index.

    Args:
        index (int): Player index (0-3).
        board_px (int): On-screen pixel size of the board (for scaling).
    """
    step = board_px * TILE_FRAC * 0.22
    corners = [(-step, -step), (step, -step), (-step, step), (step, step)]
    dx, dy = corners[index % len(corners)]
    return (int(dx), int(dy))
