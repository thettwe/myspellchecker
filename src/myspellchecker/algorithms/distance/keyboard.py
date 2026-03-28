"""
Keyboard layout utilities for weighted edit distance.

This module provides adjacency maps for standard Myanmar keyboard layouts
(primarily Myanmar3/Pyidaungsu) to help calculate 'fat finger' distances.
"""

from __future__ import annotations

from collections import defaultdict

# Grid representation of the standard Myanmar3 keyboard layout (Unshifted)
# We use a coordinate system to calculate distance.
# Row 0: q w e r t y u i o p [ ] \
# Row 1: a s d f g h j k l ; '
# Row 2: z x c v b n m , . /

# Mapping: Char -> (Row, Col)
MYANMAR3_LAYOUT: dict[str, tuple[int, int]] = {
    # Row 0
    "ဆ": (0, 0),
    "တ": (0, 1),
    "န": (0, 2),
    "မ": (0, 3),
    "အ": (0, 4),
    "ပ": (0, 5),
    "က": (0, 6),
    "င": (0, 7),
    "သ": (0, 8),
    "စ": (0, 9),
    "ဟ": (0, 10),
    "ဩ": (0, 11),
    "၏": (0, 12),
    # Row 1
    "\u1031": (1, 0),  # ေ (e-vowel)
    "\u103e": (1, 1),  # ှ (Ha-htoe)
    "\u102d": (1, 2),  # ိ (i-vowel)
    "\u103a": (1, 3),  # ် (Asat)
    "\u102b": (1, 4),  # ါ (aa-vowel)
    "\u1037": (1, 5),  # ့ (Aukmyit)
    "\u1032": (1, 6),  # ဲ (ai-vowel)
    "\u102f": (1, 7),  # ု (u-vowel)
    "\u1030": (1, 8),  # ူ (uu-vowel)
    "\u1038": (1, 9),  # း (Visarga)
    "ဧ": (1, 10),
    # Row 2
    "ဖ": (2, 0),
    "ထ": (2, 1),
    "ခ": (2, 2),
    "လ": (2, 3),
    "ဘ": (2, 4),
    "ည": (2, 5),
    "\u102c": (2, 6),
    ".": (2, 7),
    ",": (2, 8),
    "?": (2, 9),
}

# Shifted Layer (Same coordinates as unshifted counterparts)
# This helps capture "missed shift" errors
SHIFTED_LAYOUT: dict[str, tuple[int, int]] = {
    # Row 0
    "ဈ": (0, 0),
    "ဝ": (0, 1),
    "ဉ": (0, 2),
    "ဦ": (0, 3),
    "ဤ": (0, 4),
    "၌": (0, 5),
    "ဥ": (0, 6),
    "၍": (0, 7),
    "ဿ": (0, 8),
    "ဏ": (0, 9),
    "ဧ": (0, 10),
    "ဪ": (0, 11),
    # Row 1
    "ဗ": (1, 0),
    "\u103e": (1, 1),
    "\u102e": (1, 2),
    "\u1039": (1, 3),
    "\u103d": (1, 4),
    "\u1036": (1, 5),
    "\u1032": (1, 6),
    # Row 2
    "ဇ": (2, 0),
    "ဌ": (2, 1),
    "ဃ": (2, 2),
    "ဠ": (2, 3),
    "ယ": (2, 4),
    "\u102c": (2, 6),
}

# Combine layouts for lookup, supporting multiple keys for the same character
KEY_POSITIONS: dict[str, list[tuple[int, int]]] = defaultdict(list)

for char, pos in MYANMAR3_LAYOUT.items():
    KEY_POSITIONS[char].append(pos)

for char, pos in SHIFTED_LAYOUT.items():
    # Avoid adding duplicate positions if exactly same (e.g. if shift and unshift share key)
    if pos not in KEY_POSITIONS[char]:
        KEY_POSITIONS[char].append(pos)

# Add alternative key positions (e.g. keys mapped to multiple positions)
# ဉ (U+1009) is often on Shift+n (Row 2, Col 5) as well as Shift+e (Row 0, Col 2)
if (2, 5) not in KEY_POSITIONS["ဉ"]:
    KEY_POSITIONS["ဉ"].append((2, 5))


def get_keyboard_distance(char1: str, char2: str) -> float:
    """
    Calculate distance between two keys on the keyboard.

    Returns:
        0.0 if same character
        1.0 if adjacent (physically close)
        2.0+ if far
        If character not found, returns a default large distance (3.0).

    Note:
        If a character appears on multiple keys (e.g. ဉ), returns the
        minimum distance to any of its positions.
    """
    if char1 == char2:
        return 0.0

    if char1 not in KEY_POSITIONS or char2 not in KEY_POSITIONS:
        return 3.0

    positions1 = KEY_POSITIONS[char1]
    positions2 = KEY_POSITIONS[char2]

    min_dist = float("inf")

    for r1, c1 in positions1:
        for r2, c2 in positions2:
            # Manhattan distance on grid
            dist = abs(r1 - r2) + abs(c1 - c2)
            if dist < min_dist:
                min_dist = dist

    return float(min_dist)


def is_keyboard_adjacent(char1: str, char2: str) -> bool:
    """Check if two characters are adjacent on the keyboard."""
    return get_keyboard_distance(char1, char2) <= 1.0


# Pre-compute adjacency dictionary for efficient weighted edit distance
# Maps each character to the set of characters that are adjacent on the keyboard
KEY_ADJACENCY: dict[str, set[str]] = {}
for _char in KEY_POSITIONS:
    KEY_ADJACENCY[_char] = {
        _other
        for _other in KEY_POSITIONS
        if _other != _char and is_keyboard_adjacent(_char, _other)
    }
