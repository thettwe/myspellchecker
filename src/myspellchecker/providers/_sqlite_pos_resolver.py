"""
POS resolution fallback chain for SQLiteProvider.

Extracts the multi-step POS tag resolution logic into a standalone class
so that SQLiteProvider can delegate to it without inlining the entire
fallback chain.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

from myspellchecker.core.constants import ADVERB_SUFFIXES, NOUN_SUFFIXES
from myspellchecker.text.morphology import MorphologyAnalyzer
from myspellchecker.text.stemmer import Stemmer
from myspellchecker.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from myspellchecker.algorithms.pos_tagger_base import POSTaggerBase


class POSResolver:
    """
    Resolves POS tags for a word through a multi-step fallback chain.

    Fallback order:
    1. Direct dictionary lookup (``pos_tag`` from transformer, then ``inferred_pos``)
    2. Stemming + root lookup with suffix transformation rules
    3. POS tagger for OOV words
    4. MorphologyAnalyzer guess as last resort

    Args:
        stemmer: Stemmer instance for root-word extraction.
        pos_tagger: POS tagger (any ``POSTaggerBase`` implementation).
        morphology_analyzer: MorphologyAnalyzer for last-resort POS guessing.
    """

    def __init__(
        self,
        stemmer: "Stemmer",
        pos_tagger: "POSTaggerBase",
        morphology_analyzer: "MorphologyAnalyzer",
    ) -> None:
        """Initialize with the three fallback components.

        Args:
            stemmer: Stemmer for root-word extraction.
            pos_tagger: POS tagger (any POSTaggerBase implementation).
            morphology_analyzer: MorphologyAnalyzer for last-resort guessing.
        """
        self._stemmer = stemmer
        self._pos_tagger = pos_tagger
        self._morphology_analyzer = morphology_analyzer
        self._logger = get_logger(__name__)

    def resolve(self, cursor: sqlite3.Cursor, word: str) -> str | None:
        """
        Resolve POS tag for *word* using the full fallback chain.

        Args:
            cursor: An open ``sqlite3.Cursor`` connected to the dictionary DB.
            word: Myanmar word to look up.

        Returns:
            POS tag string (possibly pipe-separated for multi-POS), or ``None``.
        """
        pos_tag = self._get_pos_from_db(cursor, word)
        if pos_tag:
            return pos_tag

        pos_tag = self._get_pos_from_stemming(cursor, word)
        if pos_tag:
            return pos_tag

        pos_tag = self._get_pos_from_tagger(word)
        if pos_tag:
            return pos_tag

        return None

    # ------------------------------------------------------------------
    # Internal fallback steps
    # ------------------------------------------------------------------

    def _get_pos_from_db(self, cursor: sqlite3.Cursor, word: str) -> str | None:
        """
        Direct dictionary lookup for POS tag.

        Checks ``pos_tag`` first (from transformer), then falls back to
        ``inferred_pos`` (from rule-based inference) which may have more
        granular particle tags.
        """
        cursor.execute("SELECT pos_tag, inferred_pos FROM words WHERE word = ?", (word,))
        result = cursor.fetchone()
        if not result:
            return None

        pos_tag: str | None = result["pos_tag"]
        if pos_tag:
            return pos_tag

        inferred_pos: str | None = result["inferred_pos"]
        return inferred_pos if inferred_pos else None

    def _get_pos_from_stemming(self, cursor: sqlite3.Cursor, word: str) -> str | None:
        """Get POS via stemming + root lookup with suffix transformation."""
        root, suffixes = self._stemmer.stem(word)
        if root == word:
            return None

        cursor.execute("SELECT pos_tag, inferred_pos FROM words WHERE word = ?", (root,))
        root_result = cursor.fetchone()
        if not root_result:
            return None

        root_pos = root_result["pos_tag"] or root_result["inferred_pos"]
        if not root_pos:
            return None
        return self._apply_suffix_transformation(root_pos, suffixes)

    @staticmethod
    def _apply_suffix_transformation(root_pos: str, suffixes: list) -> str:
        """Apply suffix transformation rules to determine final POS."""
        for suffix in suffixes:
            if suffix in NOUN_SUFFIXES:
                return "N"
            if suffix in ADVERB_SUFFIXES:
                return "ADV"
        return root_pos

    def _get_pos_from_tagger(self, word: str) -> str | None:
        """Get POS from tagger, falling back to morphology analyzer."""
        try:
            guessed_pos = self._pos_tagger.tag_word(word)
            if guessed_pos and guessed_pos != "UNK":
                return guessed_pos
        except (ValueError, AttributeError, KeyError, RuntimeError) as e:
            self._logger.debug(f"POS tagging failed for '{word}': {e}")

        return self._get_pos_from_morphology(word)

    def _get_pos_from_morphology(self, word: str) -> str | None:
        """Get POS guess from morphology analyzer."""
        guessed_pos_tags = self._morphology_analyzer.guess_pos(word)
        if guessed_pos_tags:
            return "|".join(sorted(guessed_pos_tags))
        return None
