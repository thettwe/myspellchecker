"""
Factory for creating POS tagger instances.

This module provides a factory pattern for creating different types of POS taggers,
enabling easy switching between implementations and configuration-based initialization.

Supported Tagger Types:
- rule_based: Fast suffix-based tagging (default, no dependencies)
- transformer: Neural models from HuggingFace (requires transformers package)
- viterbi: HMM-based sequence tagging (requires provider with probability tables)
- custom: User-provided tagger class

Example:
    >>> from myspellchecker.algorithms.pos_tagger_factory import POSTaggerFactory
    >>>
    >>> # Create default rule-based tagger
    >>> tagger = POSTaggerFactory.create()
    >>>
    >>> # Create transformer tagger
    >>> tagger = POSTaggerFactory.create(
    ...     tagger_type="transformer",
    ...     model_name="chuuhtetnaing/myanmar-pos-model",
    ...     device=0
    ... )
    >>>
    >>> # Create Viterbi tagger with provider
    >>> from myspellchecker.providers import SQLiteProvider
    >>> provider = SQLiteProvider("mydict.db")
    >>> tagger = POSTaggerFactory.create(
    ...     tagger_type="viterbi",
    ...     provider=provider
    ... )
"""

from __future__ import annotations

from myspellchecker.algorithms.pos_tagger_base import POSTaggerBase


class POSTaggerFactory:
    """
    Factory for creating POS tagger instances.

    Provides static methods for creating different types of POS taggers
    with appropriate configuration and dependency checking.

    Methods:
        create: Main factory method for creating any tagger type
        create_rule_based: Convenience method for rule-based tagger
        create_transformer: Convenience method for transformer tagger
        create_viterbi: Convenience method for Viterbi tagger
        create_custom: Create tagger from custom class

    Example:
        >>> # Using main factory method
        >>> tagger = POSTaggerFactory.create("rule_based")
        >>>
        >>> # Using convenience methods
        >>> tagger = POSTaggerFactory.create_transformer(
        ...     model_name="custom/model",
        ...     device=0
        ... )
    """

    @staticmethod
    def create(tagger_type: str = "rule_based", **kwargs) -> POSTaggerBase:
        """
        Create a POS tagger of the specified type.

        This is the main factory method that routes to specific tagger
        implementations based on the tagger_type parameter.

        Args:
            tagger_type: Type of tagger to create. Options:
                        - "rule_based": Fast suffix-based tagging (default)
                        - "transformer": Neural models from HuggingFace
                        - "viterbi": HMM-based sequence tagging
                        - "custom": User-provided tagger class
            **kwargs: Additional arguments passed to tagger constructor.
                     Specific to each tagger type.

        Returns:
            POSTaggerBase instance of the requested type

        Raises:
            ValueError: If tagger_type is unknown or invalid
            ImportError: If required dependencies are not installed
                        (e.g., transformers for transformer type)

        Example:
            >>> # Rule-based tagger (default)
            >>> tagger = POSTaggerFactory.create()
            >>>
            >>> # Transformer tagger
            >>> tagger = POSTaggerFactory.create(
            ...     tagger_type="transformer",
            ...     model_name="chuuhtetnaing/myanmar-pos-model",
            ...     device=0
            ... )
            >>>
            >>> # Viterbi tagger
            >>> tagger = POSTaggerFactory.create(
            ...     tagger_type="viterbi",
            ...     provider=my_provider
            ... )
        """
        # Normalize tagger type
        tagger_type = tagger_type.lower().strip()

        if tagger_type == "rule_based":
            return POSTaggerFactory.create_rule_based(**kwargs)

        elif tagger_type == "transformer":
            return POSTaggerFactory.create_transformer(**kwargs)

        elif tagger_type == "viterbi":
            return POSTaggerFactory.create_viterbi(**kwargs)

        elif tagger_type == "custom":
            if "tagger_class" not in kwargs:
                raise ValueError(
                    "tagger_type='custom' requires 'tagger_class' parameter.\n"
                    "Example: POSTaggerFactory.create('custom', tagger_class=MyTagger)"
                )
            tagger_class = kwargs.pop("tagger_class")
            return POSTaggerFactory.create_custom(tagger_class, **kwargs)

        else:
            valid_types = ["rule_based", "transformer", "viterbi", "custom"]
            raise ValueError(
                f"Unknown tagger_type: '{tagger_type}'\n\n"
                f"Valid types: {', '.join(valid_types)}\n\n"
                f"Examples:\n"
                f"  POSTaggerFactory.create('rule_based')  # Fast, no dependencies\n"
                f"  POSTaggerFactory.create('transformer')  # High accuracy, needs transformers\n"
                f"  POSTaggerFactory.create('viterbi')  # Context-aware, needs provider\n"
            )

    @staticmethod
    def create_rule_based(
        pos_map: dict | None = None,
        use_morphology_fallback: bool = True,
        cache_size: int = 10000,
        unknown_tag: str = "UNK",
    ) -> POSTaggerBase:
        """
        Create a rule-based POS tagger.

        This tagger uses morphological suffix analysis to guess POS tags.
        It's fast, has no external dependencies, and is fork-safe.

        Performance:
            - Speed: ~100K words/second
            - Accuracy: ~70%
            - Memory: ~10MB
            - Dependencies: None

        Args:
            pos_map: Optional dictionary mapping words to sets of POS tags.
                    If provided, these will be used with priority over
                    morphological analysis.
            use_morphology_fallback: Whether to use MorphologyAnalyzer for
                                    unknown words (default: True)
            cache_size: Size of LRU cache for performance (default: 10000)
            unknown_tag: Tag to return for completely unknown words (default: "UNK")

        Returns:
            RuleBasedPOSTagger instance

        Example:
            >>> # Default configuration
            >>> tagger = POSTaggerFactory.create_rule_based()
            >>>
            >>> # With custom POS map
            >>> pos_map = {"မြန်မာ": {"n"}, "သည်": {"ppm"}}
            >>> tagger = POSTaggerFactory.create_rule_based(pos_map=pos_map)
        """
        from myspellchecker.algorithms.pos_tagger_rule import RuleBasedPOSTagger

        return RuleBasedPOSTagger(
            pos_map=pos_map,
            use_morphology_fallback=use_morphology_fallback,
            cache_size=cache_size,
            unknown_tag=unknown_tag,
        )

    @staticmethod
    def create_transformer(
        model_name: str | None = None,
        device: int = -1,
        batch_size: int = 32,
        max_length: int = 128,
        cache_dir: str | None = None,
        use_fp16: bool = True,
        use_torch_compile: bool = False,
        **kwargs,
    ) -> POSTaggerBase:
        """
        Create a transformer-based POS tagger.

        This tagger uses pre-trained neural models from HuggingFace for
        high-accuracy POS tagging. Requires the transformers package.

        Performance:
            - Speed: ~5K words/second (CPU), ~50K words/second (GPU)
            - Accuracy: ~93%
            - Memory: ~500MB for model + ~100MB buffer
            - Dependencies: transformers>=4.30.0, torch>=2.0.0

        Installation:
            pip install myspellchecker[transformers]

        Args:
            model_name: HuggingFace model ID or local path.
                       Default: "chuuhtetnaing/myanmar-pos-model"
            device: Device for inference. -1 for CPU, 0+ for GPU index.
                   Default: -1 (CPU)
            batch_size: Batch size for sequence tagging (default: 32)
            max_length: Maximum sequence length (default: 128)
            cache_dir: Directory for caching downloaded models (optional)
            **kwargs: Additional arguments passed to transformers.pipeline

        Returns:
            TransformerPOSTagger instance

        Raises:
            ImportError: If transformers package is not installed

        Example:
            >>> # Default model on CPU
            >>> tagger = POSTaggerFactory.create_transformer()
            >>>
            >>> # Custom model on GPU
            >>> tagger = POSTaggerFactory.create_transformer(
            ...     model_name="path/to/my/model",
            ...     device=0,
            ...     batch_size=64
            ... )
        """
        try:
            from myspellchecker.algorithms.pos_tagger_transformer import (
                TransformerPOSTagger,
            )
        except ImportError as e:
            raise ImportError(
                "Transformer-based POS tagging requires the 'transformers' library.\n\n"
                "Install with: pip install myspellchecker[transformers]\n\n"
                "This will install:\n"
                "  - transformers>=4.30.0\n"
                "  - torch>=2.0.0\n\n"
                "Alternatively, use the default rule-based tagger:\n"
                "  tagger = POSTaggerFactory.create('rule_based')\n"
            ) from e

        return TransformerPOSTagger(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            cache_dir=cache_dir,
            use_fp16=use_fp16,
            use_torch_compile=use_torch_compile,
            **kwargs,
        )

    @staticmethod
    def create_viterbi(
        provider,
        pos_bigram_probs: dict | None = None,
        pos_trigram_probs: dict | None = None,
        pos_unigram_probs: dict | None = None,
        unknown_tag: str = "UNK",
        min_prob: float = 1e-10,
        beam_width: int = 10,
        emission_weight: float = 1.2,
        use_morphology_fallback: bool = True,
        cache_size: int | None = None,
        **_kwargs,
    ) -> POSTaggerBase:
        """
        Create a Viterbi HMM-based POS tagger.

        This tagger uses the Viterbi algorithm with trigram HMM for
        context-aware sequence tagging. Requires a DictionaryProvider
        with POS probability tables.

        Performance:
            - Speed: ~20K words/second
            - Accuracy: ~85% (with proper probability tables)
            - Memory: ~50MB for probability tables
            - Dependencies: None (pure Python + optional Cython)

        Args:
            provider: DictionaryProvider with POS probability tables (required)
            pos_bigram_probs: Bigram transition probabilities P(tag2 | tag1).
                            If None, will attempt to load from provider.
            pos_trigram_probs: Trigram transition probabilities P(tag3 | tag1, tag2).
                             If None, will attempt to load from provider.
            pos_unigram_probs: Unigram tag probabilities P(tag).
                             If None, will attempt to load from provider.
            unknown_tag: Tag to return for unknown words (default: "UNK")
            min_prob: Minimum probability to prevent underflow (default: 1e-10)
            beam_width: Number of top states to keep per position (default: 10)
            emission_weight: Weight for emission probabilities (default: 1.2)
            use_morphology_fallback: Use MorphologyAnalyzer for OOV (default: True)
            cache_size: Accepted for config compatibility; not used by Viterbi tagger.

        Returns:
            ViterbiPOSTaggerAdapter instance

        Raises:
            ValueError: If provider is not provided

        Example:
            >>> from myspellchecker.providers import SQLiteProvider
            >>> provider = SQLiteProvider("mydict.db")
            >>>
            >>> # Default configuration (load probs from provider)
            >>> tagger = POSTaggerFactory.create_viterbi(provider=provider)
            >>>
            >>> # Custom configuration
            >>> tagger = POSTaggerFactory.create_viterbi(
            ...     provider=provider,
            ...     beam_width=15,
            ...     emission_weight=1.5
            ... )
        """
        if provider is None:
            raise ValueError(
                "Viterbi tagger requires a 'provider' parameter.\n\n"
                "Example:\n"
                "  from myspellchecker.providers import SQLiteProvider\n"
                "  provider = SQLiteProvider('mydict.db')\n"
                "  tagger = POSTaggerFactory.create_viterbi(provider=provider)\n"
            )

        from myspellchecker.algorithms.pos_tagger_viterbi import (
            ViterbiPOSTaggerAdapter,
        )

        return ViterbiPOSTaggerAdapter(
            provider=provider,
            pos_bigram_probs=pos_bigram_probs,
            pos_trigram_probs=pos_trigram_probs,
            pos_unigram_probs=pos_unigram_probs,
            unknown_tag=unknown_tag,
            min_prob=min_prob,
            beam_width=beam_width,
            emission_weight=emission_weight,
            use_morphology_fallback=use_morphology_fallback,
        )

    @staticmethod
    def create_custom(tagger_class: type[POSTaggerBase], **kwargs) -> POSTaggerBase:
        """
        Create a custom POS tagger from a user-provided class.

        The custom tagger class must inherit from POSTaggerBase and
        implement all required abstract methods.

        Args:
            tagger_class: Class that inherits from POSTaggerBase
            **kwargs: Arguments passed to the tagger class constructor

        Returns:
            Instance of the custom tagger class

        Raises:
            TypeError: If tagger_class doesn't inherit from POSTaggerBase

        Example:
            >>> class MyCustomTagger(POSTaggerBase):
            ...     def tag_word(self, word: str) -> str:
            ...         return "n"  # Simple implementation
            ...     def tag_sequence(self, words: list[str]) -> list[str]:
            ...         return ["n"] * len(words)
            ...     @property
            ...     def tagger_type(self) -> TaggerType:
            ...         return TaggerType.CUSTOM
            >>>
            >>> tagger = POSTaggerFactory.create_custom(
            ...     tagger_class=MyCustomTagger
            ... )
        """
        if not issubclass(tagger_class, POSTaggerBase):
            raise TypeError(
                f"Custom tagger class must inherit from POSTaggerBase.\n\n"
                f"Provided class: {tagger_class.__name__}\n"
                f"Base class: {tagger_class.__bases__}\n\n"
                f"Example:\n"
                f"  class MyTagger(POSTaggerBase):\n"
                f"      def tag_word(self, word: str) -> str:\n"
                f"          return 'n'\n"
                f"      def tag_sequence(self, words: list[str]) -> list[str]:\n"
                f"          return ['n'] * len(words)\n"
                f"      @property\n"
                f"      def tagger_type(self) -> TaggerType:\n"
                f"          return TaggerType.CUSTOM\n"
            )

        return tagger_class(**kwargs)
