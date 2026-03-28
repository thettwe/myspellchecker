"""Test configuration templates for integration tests."""

# Valid YAML configuration with all fields
VALID_YAML_CONFIG = """
preset: accurate
max_edit_distance: 3
max_suggestions: 10
use_context_checker: false
use_phonetic: true
word_engine: myword

provider_config:
  cache_size: 2048
  pool_min_size: 2
  pool_max_size: 15
"""

# Valid JSON configuration
VALID_JSON_CONFIG = """{
  "preset": "fast",
  "max_edit_distance": 2,
  "max_suggestions": 5,
  "use_context_checker": true,
  "use_phonetic": false,
  "provider_config": {
    "cache_size": 1024,
    "pool_min_size": 1,
    "pool_max_size": 5
  }
}"""

# Minimal valid configuration (only preset)
MINIMAL_YAML_CONFIG = """
preset: production
"""

# Configuration with database path
CONFIG_WITH_DATABASE = """
preset: accurate
database: /path/to/custom.db
max_suggestions: 8
"""

# Configuration with production preset alias
CONFIG_WITH_PRODUCTION_PRESET = """
preset: production
max_edit_distance: 2
"""

# Configuration with development preset alias
CONFIG_WITH_DEVELOPMENT_PRESET = """
preset: development
use_context_checker: true
"""

# Invalid YAML (malformed)
INVALID_YAML_CONFIG = """
preset: [this is not valid yaml
max_suggestions: 10
"""

# Configuration with all provider settings
CONFIG_WITH_FULL_PROVIDER = """
preset: production
provider_config:
  database_path: /path/to/db.db
  cache_size: 2048
  pool_min_size: 5
  pool_max_size: 20
  pool_timeout: 30.0
  pool_max_connection_age: 7200.0
"""
