"""Parser for homophones.yaml configuration.

Extracts homophone mappings (word -> set of homophones).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "parse_homophones_config",
]


def parse_homophones_config(
    config: dict[str, Any],
    *,
    homophones_map: dict[str, set[str]],
) -> None:
    """Parse homophones.yaml configuration.

    Mutates the provided container in-place.

    Args:
        config: Raw YAML dict from homophones.yaml.
        homophones_map: Dict to populate with word -> set of homophones.
    """
    homophones = config.get("homophones", {})

    for word, homophone_list in homophones.items():
        if isinstance(homophone_list, list):
            homophones_map[word] = set(homophone_list)
