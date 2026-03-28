# distutils: language = c++
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: profile = False
# cython: linetrace = False
# cython: binding = True
"""
Cython-optimized Viterbi algorithm for Part-of-Speech tagging.

This module provides a high-performance HMM-based POS tagger using the
Viterbi algorithm with deleted interpolation smoothing for probability
estimation.

Key Features:
    - Trigram language model with deleted interpolation
    - Beam search optimization for large tag sets
    - OOV (Out-of-Vocabulary) word handling via morphological analysis
    - Pre-computed log probabilities for numerical stability
    - C++ unordered_map for O(1) probability lookups

Algorithm:
    Uses dynamic programming to find the most likely tag sequence:
    1. Forward pass: compute path probabilities with beam pruning
    2. Backtracking: reconstruct the optimal tag sequence

Smoothing:
    Deleted interpolation combines unigram, bigram, and trigram probabilities:
    P(tag) = λ1 * P(tag) + λ2 * P(tag|prev) + λ3 * P(tag|prev2, prev)

Performance:
    - ~5x faster than pure Python implementation
    - O(n * k^2 * beam_width) where n=words, k=tags
    - Memory efficient with beam pruning

Example:
    >>> from myspellchecker.algorithms.viterbi import CythonViterbiTagger
    >>> tagger = CythonViterbiTagger(provider, bigram_probs, trigram_probs)
    >>> tags = tagger.tag(["ကျွန်တော်", "သွား", "တယ်"])
    ['N', 'V', 'PPM']

See Also:
    - pos_tagger_viterbi.py: Python wrapper and ViterbiPOSTagger class
    - morphology.py: MorphologyAnalyzer for OOV word handling
"""

from libc.math cimport log, exp, INFINITY
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from heapq import nlargest
import warnings

from typing import Dict, List, Optional, Tuple, Set

# cimport Python built-in types for type annotations
cimport cpython.set
cimport cpython.dict
cimport cpython.list

# Need to import the Python DictionaryProvider
from myspellchecker.providers.base import DictionaryProvider
# Import MorphologyAnalyzer for OOV word tagging fallback
from myspellchecker.text.morphology import MorphologyAnalyzer

cdef class CythonViterbiTagger:
    # Expose _provider as a Python object, as DictionaryProvider is a Python class
    cdef public object _provider
    cdef public object _morphology_analyzer  # For OOV word tagging fallback
    cdef unordered_map[string, double] _cpp_bigram_probs
    cdef unordered_map[string, double] _cpp_trigram_probs
    cdef unordered_map[string, double] _cpp_unigram_probs
    cdef unordered_map[string, unordered_map[string, double]] _cpp_word_tag_probs

    cdef string _unknown_word_tag
    cdef double _min_prob
    cdef double _log_min_prob
    cdef double _emission_weight
    cdef int _beam_width
    # Deleted interpolation weights (λ1 + λ2 + λ3 = 1)
    cdef double _lambda_unigram
    cdef double _lambda_bigram
    cdef double _lambda_trigram
    # Pre-computed log(lambda) values for performance
    cdef double _log_lambda_unigram
    cdef double _log_lambda_bigram
    cdef double _log_lambda_trigram
    # Pre-computed byte constants for performance
    cdef string _pipe_separator

    def __init__(self, provider: DictionaryProvider,
                 pos_bigram_probs: Dict[Tuple[str, str], float],
                 pos_trigram_probs: Dict[Tuple[str, str, str], float],
                 unknown_word_tag: str = "UNK",
                 min_prob: float = 1e-10,
                 pos_unigram_probs: Optional[Dict[str, float]] = None,
                 word_tag_probs: Optional[Dict[str, Dict[str, float]]] = None,
                 emission_weight: float = 1.2,
                 beam_width: int = 10,
                 lambda_unigram: float = 0.1,
                 lambda_bigram: float = 0.3,
                 lambda_trigram: float = 0.6):
        """
        Initialize CythonViterbiTagger with deleted interpolation smoothing.

        Args:
            provider: DictionaryProvider for word-POS lookups
            pos_bigram_probs: Bigram transition probabilities P(t2|t1)
            pos_trigram_probs: Trigram transition probabilities P(t3|t1,t2)
            unknown_word_tag: Tag for unknown words (default: "UNK")
            min_prob: Minimum probability floor (default: 1e-10)
            pos_unigram_probs: Unigram probabilities P(t) for smoothing
            word_tag_probs: Word-tag emission probabilities
            emission_weight: Weight for emission scores (default: 1.2)
            beam_width: Beam search width for pruning (default: 10)
            lambda_unigram: Interpolation weight for unigram (default: 0.1)
            lambda_bigram: Interpolation weight for bigram (default: 0.3)
            lambda_trigram: Interpolation weight for trigram (default: 0.6)

        Note:
            lambda_unigram + lambda_bigram + lambda_trigram should equal 1.0
            for proper probability distribution. The smoothed transition
            probability uses deleted interpolation:
            P(t3|t1,t2) = λ3*P(t3|t1,t2) + λ2*P(t3|t2) + λ1*P(t3)
        """
        self._provider = provider
        # Initialize morphology analyzer for OOV word tagging fallback
        self._morphology_analyzer = MorphologyAnalyzer()
        self._unknown_word_tag = unknown_word_tag.encode('utf-8')
        self._min_prob = min_prob
        self._log_min_prob = log(min_prob)
        self._emission_weight = emission_weight
        self._beam_width = beam_width

        # Validate and store interpolation weights
        weight_sum = lambda_unigram + lambda_bigram + lambda_trigram
        if weight_sum > 0 and abs(weight_sum - 1.0) > 1e-6:
            # Warn before automatic normalization
            warnings.warn(
                f"lambda_unigram + lambda_bigram + lambda_trigram = {weight_sum:.4f} "
                f"(expected 1.0). Weights will be normalized automatically.",
                UserWarning,
                stacklevel=2,
            )
            # Normalize if not summing to 1
            lambda_unigram /= weight_sum
            lambda_bigram /= weight_sum
            lambda_trigram /= weight_sum

        # Ensure lambda values are positive to prevent log(0) domain error
        # Use min_prob as floor since it's already a safe small positive value
        lambda_unigram = max(lambda_unigram, min_prob)
        lambda_bigram = max(lambda_bigram, min_prob)
        lambda_trigram = max(lambda_trigram, min_prob)

        # Re-normalize after flooring to ensure weights sum close to 1.0
        floored_sum = lambda_unigram + lambda_bigram + lambda_trigram
        lambda_unigram = lambda_unigram / floored_sum
        lambda_bigram = lambda_bigram / floored_sum
        lambda_trigram = lambda_trigram / floored_sum

        # Re-apply floor after normalization for floating-point precision
        self._lambda_unigram = max(lambda_unigram, min_prob)
        self._lambda_bigram = max(lambda_bigram, min_prob)
        self._lambda_trigram = max(lambda_trigram, min_prob)

        # Pre-compute log(lambda) values for performance
        # Avoids repeated log() calls in hot path functions
        self._log_lambda_unigram = log(self._lambda_unigram)
        self._log_lambda_bigram = log(self._lambda_bigram)
        self._log_lambda_trigram = log(self._lambda_trigram)

        # Pre-compute byte constants for performance
        self._pipe_separator = b"|"

        # Populate C++ maps
        cdef string key
        for k_tuple, v in pos_bigram_probs.items():
            key = (k_tuple[0] + "|" + k_tuple[1]).encode('utf-8')
            self._cpp_bigram_probs[key] = log(v)

        for k_tuple, v in pos_trigram_probs.items():
            key = (k_tuple[0] + "|" + k_tuple[1] + "|" + k_tuple[2]).encode('utf-8')
            self._cpp_trigram_probs[key] = log(v)

        if pos_unigram_probs:
            for k, v in pos_unigram_probs.items():
                self._cpp_unigram_probs[k.encode('utf-8')] = log(v)

        if word_tag_probs:
            for word, tags in word_tag_probs.items():
                for tag, prob in tags.items():
                    self._cpp_word_tag_probs[word.encode('utf-8')][tag.encode('utf-8')] = log(prob)

    cdef double _get_emission_score(self, string word, string tag):
        """
        Get emission score for a word-tag pair (C++ version).
        """
        if self._cpp_word_tag_probs.count(word):
            if self._cpp_word_tag_probs[word].count(tag):
                return self._emission_weight * self._cpp_word_tag_probs[word][tag]
            else:
                return self._emission_weight * self._log_min_prob

        if self._cpp_unigram_probs.count(tag):
            return self._emission_weight * self._cpp_unigram_probs[tag]

        # Fallback: use log_min_prob (Issue #1242, parity with Python)
        return self._emission_weight * self._log_min_prob

    cdef inline double _log_sum_exp_2(self, double log_a, double log_b):
        """
        Compute log(exp(log_a) + exp(log_b)) with numerical stability.

        Uses the identity: log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
        """
        cdef double max_val, diff
        if log_a > log_b:
            max_val = log_a
            diff = log_b - log_a
        else:
            max_val = log_b
            diff = log_a - log_b

        # For very negative diff, exp(diff) ≈ 0, so result ≈ max_val
        # Guard against all -inf inputs (Issue #1241)
        if max_val == -INFINITY:
            return -INFINITY
        if diff < -30.0:
            return max_val
        return max_val + log(1.0 + exp(diff))

    cdef inline double _log_sum_exp_3(self, double log_a, double log_b, double log_c):
        """
        Compute log(exp(log_a) + exp(log_b) + exp(log_c)) with numerical stability.
        """
        cdef double max_val = log_a
        if log_b > max_val:
            max_val = log_b
        if log_c > max_val:
            max_val = log_c

        # Avoid underflow by subtracting max
        # Guard against all -inf inputs (Issue #1241)
        if max_val == -INFINITY:
            return -INFINITY
        cdef double sum_exp = exp(log_a - max_val) + exp(log_b - max_val) + exp(log_c - max_val)
        return max_val + log(sum_exp)

    cdef double _get_smoothed_bigram_prob(self, string tag_prev, string tag_curr):
        """
        Get smoothed bigram transition probability using deleted interpolation.

        P_smooth(t2|t1) = λ2 * P(t2|t1) + λ1 * P(t2)

        Returns log probability: log(λ2 * exp(log_bigram) + λ1 * exp(log_unigram))
        """
        cdef double log_bigram_prob
        cdef double log_unigram_prob
        cdef string bigram_key = tag_prev + self._pipe_separator + tag_curr

        # Get bigram probability if available (already in log space)
        if self._cpp_bigram_probs.count(bigram_key):
            log_bigram_prob = self._cpp_bigram_probs[bigram_key]
        else:
            log_bigram_prob = self._log_min_prob

        # Get unigram probability for backoff (already in log space)
        if self._cpp_unigram_probs.count(tag_curr):
            log_unigram_prob = self._cpp_unigram_probs[tag_curr]
        else:
            log_unigram_prob = self._log_min_prob

        # Deleted interpolation in log space using pre-computed log(lambda)
        # log(λ2 * P_bigram + λ1 * P_unigram)
        # = log(exp(log(λ2) + log_bigram) + exp(log(λ1) + log_unigram))
        # Use log-sum-exp for numerical stability
        return self._log_sum_exp_2(
            self._log_lambda_bigram + log_bigram_prob,
            self._log_lambda_unigram + log_unigram_prob
        )

    cdef double _get_smoothed_trigram_prob(self, string tag_prev_prev, string tag_prev, string tag_curr):
        """
        Get smoothed trigram transition probability using deleted interpolation.

        P_smooth(t3|t1,t2) = λ3 * P(t3|t1,t2) + λ2 * P(t3|t2) + λ1 * P(t3)

        Returns log probability: log(λ3*exp(log_tri) + λ2*exp(log_bi) + λ1*exp(log_uni))
        """
        cdef double log_trigram_prob
        cdef double log_bigram_prob
        cdef double log_unigram_prob
        cdef string trigram_key = tag_prev_prev + self._pipe_separator + tag_prev + self._pipe_separator + tag_curr
        cdef string bigram_key = tag_prev + self._pipe_separator + tag_curr

        # Get trigram probability if available (already in log space)
        if self._cpp_trigram_probs.count(trigram_key):
            log_trigram_prob = self._cpp_trigram_probs[trigram_key]
        else:
            log_trigram_prob = self._log_min_prob

        # Get bigram probability for backoff (already in log space)
        if self._cpp_bigram_probs.count(bigram_key):
            log_bigram_prob = self._cpp_bigram_probs[bigram_key]
        else:
            log_bigram_prob = self._log_min_prob

        # Get unigram probability for further backoff (already in log space)
        if self._cpp_unigram_probs.count(tag_curr):
            log_unigram_prob = self._cpp_unigram_probs[tag_curr]
        else:
            log_unigram_prob = self._log_min_prob

        # Deleted interpolation in log space using pre-computed log(lambda)
        # log(λ3 * P_tri + λ2 * P_bi + λ1 * P_uni)
        # = log(exp(log(λ3) + log_tri) + exp(log(λ2) + log_bi) + exp(log(λ1) + log_uni))
        # Use log-sum-exp for numerical stability
        return self._log_sum_exp_3(
            self._log_lambda_trigram + log_trigram_prob,
            self._log_lambda_bigram + log_bigram_prob,
            self._log_lambda_unigram + log_unigram_prob
        )

    cdef vector[string] _get_valid_tags(self, string word):
        """
        Get valid POS tags for a word (C++ version).

        Fallback chain:
        1. Dictionary lookup via provider
        2. Word-tag probability table
        3. Morphological analysis
        4. Unknown word tag
        """
        cdef vector[string] tags
        cdef str word_py = word.decode('utf-8')
        cdef str pos_str = self._provider.get_word_pos(word_py)

        if pos_str:
            for t in pos_str.split("|"):
                tags.push_back(t.encode('utf-8'))

        if self._cpp_word_tag_probs.count(word):
            for it in self._cpp_word_tag_probs[word]:
                # Avoid duplicates
                found = False
                for existing in tags:
                    if existing == it.first:
                        found = True
                        break
                if not found:
                    tags.push_back(it.first)

        # Morphological fallback for unknown words
        # Use morphology analyzer to guess POS before falling back to UNK
        if tags.empty() and self._morphology_analyzer is not None:
            guessed_pos_set = self._morphology_analyzer.guess_pos(word_py)
            if guessed_pos_set:
                for guessed_pos in guessed_pos_set:
                    tags.push_back(guessed_pos.encode('utf-8'))

        if tags.empty():
            tags.push_back(self._unknown_word_tag)

        return tags

    def tag_sequence(self, list words_py) -> list:
        cdef int num_words = len(words_py)
        if num_words == 0:
            return []

        cdef vector[string] words
        for w in words_py:
            words.push_back(w.encode('utf-8'))

        # Use native C++ types for DP table where possible
        cdef vector[unordered_map[string, double]] viterbi
        cdef vector[unordered_map[string, string]] backpointer
        viterbi.resize(num_words)
        backpointer.resize(num_words)

        # 1. Initialization for t = 0
        cdef vector[string] first_word_tags = self._get_valid_tags(words[0])
        cdef string tag_0, state_key, prev_state_key
        cdef double emission_score, prob, max_prob, trans_prob, prev_emission
        
        for tag_0 in first_word_tags:
            emission_score = self._get_emission_score(words[0], tag_0)
            state_key = self._unknown_word_tag + self._pipe_separator + tag_0
            viterbi[0][state_key] = emission_score
            backpointer[0][state_key] = self._unknown_word_tag

        # 2. Initialization for t = 1
        # Note on emission score computation:
        # - emission_score (for words[1]) is computed once per tag_1 (outside inner loop)
        #   since it depends only on (word[1], tag_1), not on tag_0
        # - prev_emission (for words[0]) is computed inside the loop since it varies
        #   with tag_0. This equals viterbi[0] values but is recomputed for clarity
        #   and to avoid string key lookups. This is correct per standard Viterbi.
        cdef vector[string] second_word_tags
        cdef string tag_1, best_prev_tag_0
        if num_words > 1:
            second_word_tags = self._get_valid_tags(words[1])
            for tag_1 in second_word_tags:
                max_prob = -INFINITY
                best_prev_tag_0 = b""
                # Emission for current position (word[1], tag_1) - constant for this tag_1
                emission_score = self._get_emission_score(words[1], tag_1)

                for tag_0 in first_word_tags:
                    # Use smoothed bigram probability (deleted interpolation)
                    trans_prob = self._get_smoothed_bigram_prob(tag_0, tag_1)

                    # Emission for previous position (word[0], tag_0) - varies with tag_0
                    prev_emission = self._get_emission_score(words[0], tag_0)
                    prob = prev_emission + trans_prob + emission_score
                    
                    if prob > max_prob:
                        max_prob = prob
                        best_prev_tag_0 = tag_0
                
                if not best_prev_tag_0.empty():
                    state_key = best_prev_tag_0 + self._pipe_separator + tag_1
                    viterbi[1][state_key] = max_prob
                    backpointer[1][state_key] = self._unknown_word_tag

        # 3. Recursion for t > 1
        cdef int t, idx
        cdef vector[string] curr_tags
        cdef string curr_tag, tag_prev_prev, tag_prev, best_prev_tag_prev_prev, best_prev_tag_prev
        cdef string trigram_key
        cdef double best_at_t, prune_cutoff
        
        for t in range(2, num_words):
            curr_tags = self._get_valid_tags(words[t])
            
            for curr_tag in curr_tags:
                max_prob = -INFINITY
                best_prev_tag_prev_prev = b""
                best_prev_tag_prev = b""
                emission_score = self._get_emission_score(words[t], curr_tag)
                
                for item in viterbi[t-1]:
                    state_key = item.first
                    idx = state_key.find(self._pipe_separator)
                    tag_prev_prev = state_key.substr(0, idx)
                    tag_prev = state_key.substr(idx + 1)

                    # Use smoothed trigram probability (deleted interpolation)
                    trans_prob = self._get_smoothed_trigram_prob(tag_prev_prev, tag_prev, curr_tag)

                    prob = item.second + trans_prob + emission_score
                    
                    if prob > max_prob:
                        max_prob = prob
                        best_prev_tag_prev_prev = tag_prev_prev
                        best_prev_tag_prev = tag_prev
                
                if not best_prev_tag_prev.empty():
                    state_key = best_prev_tag_prev + self._pipe_separator + curr_tag
                    viterbi[t][state_key] = max_prob
                    backpointer[t][state_key] = best_prev_tag_prev_prev

            # Beam pruning
            if viterbi[t].size() > <size_t>self._beam_width:
                best_at_t = -INFINITY
                for item in viterbi[t]:
                    if item.second > best_at_t:
                        best_at_t = item.second
                
                py_step = {k: v for k, v in viterbi[t]}
                top_items = nlargest(self._beam_width, py_step.items(), key=lambda x: x[1])
                viterbi[t].clear()
                new_bp = unordered_map[string, string]()
                for k_py, v_py in top_items:
                    viterbi[t][k_py] = v_py
                    new_bp[k_py] = backpointer[t][k_py]
                backpointer[t] = new_bp

        # 4. Termination
        if viterbi[num_words-1].empty():
            return [self._unknown_word_tag.decode('utf-8')] * num_words

        # Select best final state with tie-breaking
        # When multiple states have equal probabilities, break ties using:
        # 1. Primary: Viterbi path probability (higher is better)
        # 2. Secondary: Sum of unigram frequencies for both tags (prefer common tags)
        # 3. Tertiary: Alphabetical ordering for deterministic behavior
        cdef string best_final_state = b""
        cdef double best_prob = -INFINITY
        cdef double best_freq_sum = -INFINITY
        cdef string best_alpha_key = b""
        cdef double freq_sum, freq_prev_prev, freq_prev
        cdef string alpha_key
        max_prob = -INFINITY
        for item in viterbi[num_words-1]:
            state_key = item.first
            prob = item.second

            # Parse state_key to get tag_prev_prev and tag_prev
            idx = state_key.find(self._pipe_separator)
            tag_prev_prev = state_key.substr(0, idx)
            tag_prev = state_key.substr(idx + 1)

            # Calculate frequency sum for tie-breaking
            if self._cpp_unigram_probs.count(tag_prev_prev):
                freq_prev_prev = exp(self._cpp_unigram_probs[tag_prev_prev])
            else:
                freq_prev_prev = 0.0
            if self._cpp_unigram_probs.count(tag_prev):
                freq_prev = exp(self._cpp_unigram_probs[tag_prev])
            else:
                freq_prev = 0.0
            freq_sum = freq_prev_prev + freq_prev

            # Alphabetical key for deterministic tie-breaking
            alpha_key = tag_prev_prev + tag_prev

            # Compare using tie-breaking criteria
            if prob > best_prob:
                best_prob = prob
                best_freq_sum = freq_sum
                best_alpha_key = alpha_key
                best_final_state = state_key
            elif prob == best_prob:
                # Equal probability, use frequency sum
                if freq_sum > best_freq_sum:
                    best_freq_sum = freq_sum
                    best_alpha_key = alpha_key
                    best_final_state = state_key
                elif freq_sum == best_freq_sum:
                    # Equal frequency, use alphabetical order
                    if alpha_key < best_alpha_key:
                        best_alpha_key = alpha_key
                        best_final_state = state_key
                
        # 5. Backtracking
        cdef list best_path_tags = []
        cdef string curr_state = best_final_state
        cdef string prev_prev_tag
        cdef int remaining
        cdef bint beam_fallback = False

        for t in range(num_words - 1, 0, -1):
            idx = curr_state.find(self._pipe_separator)
            tag_prev = curr_state.substr(idx + 1)
            best_path_tags.append(tag_prev.decode('utf-8'))

            # Handle KeyError after beam pruning
            # After beam pruning, backpointer[t] may not contain curr_state
            if not backpointer[t].count(curr_state):
                # Fallback: use unknown_word_tag for remaining positions
                # At this point, tag for position t is already appended (line 548).
                # We need tags for positions 0..t-1, which is t positions.
                remaining = t
                unknown_tag_py = self._unknown_word_tag.decode('utf-8')
                for _ in range(remaining):
                    best_path_tags.append(unknown_tag_py)
                beam_fallback = True
                break

            prev_prev_tag = backpointer[t][curr_state]
            curr_state = prev_prev_tag + self._pipe_separator + curr_state.substr(0, idx)

        # Add the very first tag (only if we didn't hit beam fallback)
        if not beam_fallback:
            idx = curr_state.find(self._pipe_separator)
            best_path_tags.append(curr_state.substr(idx + 1).decode('utf-8'))

        best_path_tags.reverse()
        return best_path_tags