"""Unit tests for core DI container and homophones modules."""

from unittest.mock import Mock, patch

import pytest


class TestHomophoneChecker:
    """Tests for HomophoneChecker."""

    def test_init_with_homophone_map(self):
        """Test initialization with provided homophone map."""
        from myspellchecker.core.homophones import HomophoneChecker

        homophone_map = {
            "word1": {"word2", "word3"},
            "word2": {"word1", "word3"},
        }

        checker = HomophoneChecker(homophone_map=homophone_map)
        assert checker.homophone_map == homophone_map

    @patch("myspellchecker.core.homophones.get_grammar_config")
    def test_init_with_config_path(self, mock_get_config):
        """Test initialization with config path."""
        from myspellchecker.core.homophones import HomophoneChecker

        mock_config = Mock()
        mock_config.homophones_map = {"test": {"other"}}
        mock_get_config.return_value = mock_config

        checker = HomophoneChecker(config_path="/path/to/config")

        mock_get_config.assert_called_with("/path/to/config")
        # _ensure_symmetry adds the reverse edge: "other" → {"test"}
        assert checker.homophone_map == {"test": {"other"}, "other": {"test"}}

    @patch("myspellchecker.core.homophones.get_grammar_config")
    def test_init_default(self, mock_get_config):
        """Test default initialization."""
        from myspellchecker.core.homophones import HomophoneChecker

        mock_config = Mock()
        mock_config.homophones_map = {}
        mock_get_config.return_value = mock_config

        HomophoneChecker()

        mock_get_config.assert_called_with(None)

    def test_get_homophones_found(self):
        """Test get_homophones returns candidates."""
        from myspellchecker.core.homophones import HomophoneChecker

        homophone_map = {
            "word1": {"word2", "word3"},
        }

        checker = HomophoneChecker(homophone_map=homophone_map)
        homophones = checker.get_homophones("word1")

        assert "word2" in homophones
        assert "word3" in homophones
        assert "word1" not in homophones

    def test_get_homophones_not_found(self):
        """Test get_homophones returns empty set for unknown word."""
        from myspellchecker.core.homophones import HomophoneChecker

        checker = HomophoneChecker(homophone_map={})
        homophones = checker.get_homophones("unknown")

        assert homophones == set()

    def test_get_homophones_empty_word(self):
        """Test get_homophones returns empty set for empty word."""
        from myspellchecker.core.homophones import HomophoneChecker

        checker = HomophoneChecker(homophone_map={"test": {"other"}})
        homophones = checker.get_homophones("")

        assert homophones == set()


class TestServiceContainer:
    """Tests for ServiceContainer."""

    def test_init(self):
        """Test container initialization."""
        from myspellchecker.core.di.container import ServiceContainer

        mock_config = Mock()
        container = ServiceContainer(mock_config)

        assert container._config == mock_config
        assert container._services == {}
        assert container._factories == {}
        assert container._singletons == set()

    def test_register_factory_singleton(self):
        """Test registering a singleton factory."""
        from myspellchecker.core.di.container import ServiceContainer

        mock_config = Mock()
        container = ServiceContainer(mock_config)

        factory = Mock(return_value="service_instance")
        container.register_factory("my_service", factory, singleton=True)

        assert "my_service" in container._factories
        assert "my_service" in container._singletons

    def test_register_factory_transient(self):
        """Test registering a transient factory."""
        from myspellchecker.core.di.container import ServiceContainer

        mock_config = Mock()
        container = ServiceContainer(mock_config)

        factory = Mock(return_value="service_instance")
        container.register_factory("my_service", factory, singleton=False)

        assert "my_service" in container._factories
        assert "my_service" not in container._singletons

    def test_register_factory_duplicate(self):
        """Test registering duplicate factory raises ValueError."""
        from myspellchecker.core.di.container import ServiceContainer

        mock_config = Mock()
        container = ServiceContainer(mock_config)

        factory = Mock()
        container.register_factory("my_service", factory)

        with pytest.raises(ValueError, match="already registered"):
            container.register_factory("my_service", factory)

    def test_get_singleton(self):
        """Test getting a singleton service."""
        from myspellchecker.core.di.container import ServiceContainer

        mock_config = Mock()
        container = ServiceContainer(mock_config)

        factory = Mock(return_value="service_instance")
        container.register_factory("my_service", factory, singleton=True)

        # First get creates instance
        result1 = container.get("my_service")
        assert result1 == "service_instance"
        factory.assert_called_once_with(container)

        # Second get returns cached instance
        result2 = container.get("my_service")
        assert result2 == "service_instance"
        assert factory.call_count == 1  # Factory not called again

    def test_get_transient(self):
        """Test getting a transient service creates new instance each time."""
        from myspellchecker.core.di.container import ServiceContainer

        mock_config = Mock()
        container = ServiceContainer(mock_config)

        factory = Mock(side_effect=["instance1", "instance2"])
        container.register_factory("my_service", factory, singleton=False)

        result1 = container.get("my_service")
        result2 = container.get("my_service")

        assert result1 == "instance1"
        assert result2 == "instance2"
        assert factory.call_count == 2

    def test_get_not_registered(self):
        """Test getting unregistered service raises ValueError."""
        from myspellchecker.core.di.container import ServiceContainer

        mock_config = Mock()
        container = ServiceContainer(mock_config)

        with pytest.raises(ValueError, match="not registered"):
            container.get("unknown_service")

    def test_get_factory_exception(self):
        """Test getting service when factory raises exception."""
        from myspellchecker.core.di.container import ServiceContainer

        mock_config = Mock()
        container = ServiceContainer(mock_config)

        factory = Mock(side_effect=RuntimeError("Factory error"))
        container.register_factory("my_service", factory)

        with pytest.raises(RuntimeError, match="Factory error"):
            container.get("my_service")

    def test_get_config(self):
        """Test get_config returns config."""
        from myspellchecker.core.di.container import ServiceContainer

        mock_config = Mock()
        container = ServiceContainer(mock_config)

        assert container.get_config() == mock_config

    def test_has_service_true(self):
        """Test has_service returns True for registered service."""
        from myspellchecker.core.di.container import ServiceContainer

        mock_config = Mock()
        container = ServiceContainer(mock_config)

        container.register_factory("my_service", Mock())
        assert container.has_service("my_service") is True

    def test_has_service_false(self):
        """Test has_service returns False for unregistered service."""
        from myspellchecker.core.di.container import ServiceContainer

        mock_config = Mock()
        container = ServiceContainer(mock_config)

        assert container.has_service("unknown") is False

    def test_list_services(self):
        """Test list_services returns sorted list."""
        from myspellchecker.core.di.container import ServiceContainer

        mock_config = Mock()
        container = ServiceContainer(mock_config)

        container.register_factory("zeta", Mock())
        container.register_factory("alpha", Mock())
        container.register_factory("beta", Mock())

        services = container.list_services()
        assert services == ["alpha", "beta", "zeta"]

    def test_clear_cache(self):
        """Test clear_cache removes cached instances."""
        from myspellchecker.core.di.container import ServiceContainer

        mock_config = Mock()
        container = ServiceContainer(mock_config)

        factory = Mock(side_effect=["instance1", "instance2"])
        container.register_factory("my_service", factory, singleton=True)

        # Get creates and caches
        result1 = container.get("my_service")
        assert result1 == "instance1"

        # Clear cache
        container.clear_cache()

        # Get creates new instance
        result2 = container.get("my_service")
        assert result2 == "instance2"

    def test_repr(self):
        """Test repr returns expected format."""
        from myspellchecker.core.di.container import ServiceContainer

        mock_config = Mock()
        container = ServiceContainer(mock_config)

        container.register_factory("service1", Mock())
        container.register_factory("service2", Mock())
        container.get("service1")  # Cache one

        repr_str = repr(container)
        assert "ServiceContainer" in repr_str
        assert "services=2" in repr_str
        assert "cached=1" in repr_str


class TestDIInit:
    """Tests for DI module __init__."""

    def test_imports(self):
        """Test that ServiceContainer can be imported."""
        from myspellchecker.core.di import ServiceContainer

        assert ServiceContainer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
