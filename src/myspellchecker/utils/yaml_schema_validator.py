"""
Optional runtime YAML schema validation.

Validates YAML rule data against JSON Schema files at load time.
Requires the ``jsonschema`` package (optional dependency).  When the
package is not installed, validation is silently skipped so the library
continues to work in minimal environments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from myspellchecker.utils.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "get_schema_path_for_yaml",
    "validate_yaml_against_schema",
]

# Resolve the schemas directory once at module level.
_SCHEMAS_DIR = Path(__file__).parent.parent / "schemas"


def validate_yaml_against_schema(
    data: dict[str, Any],
    schema_path: str | Path,
) -> list[str]:
    """Validate *data* against the JSON Schema at *schema_path*.

    Parameters
    ----------
    data:
        The parsed YAML data (a dict from ``yaml.safe_load``).
    schema_path:
        Absolute or relative path to the ``.schema.json`` file.

    Returns
    -------
    list[str]
        A list of human-readable validation error messages.
        An empty list means the data is valid (or that ``jsonschema``
        is not installed, in which case a debug message is logged).
    """
    try:
        from importlib.util import find_spec

        if find_spec("jsonschema") is None:
            raise ImportError("jsonschema not found")  # noqa: TRY301
    except ImportError:
        logger.debug(
            "jsonschema package not installed; skipping schema validation for %s",
            schema_path,
        )
        return []

    schema_path = Path(schema_path)
    if not schema_path.exists():
        logger.debug("Schema file not found: %s", schema_path)
        return []

    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load schema %s: %s", schema_path, exc)
        return [f"Schema load error: {exc}"]

    # Build a registry so that $ref: "_common.schema.json#/..." resolves
    # without network access, mirroring the approach in the test suite.
    validator = _build_validator(schema, schema_path)
    if validator is None:
        # Defensive: if registry construction failed, skip validation.
        return []

    errors: list[str] = []
    for error in validator.iter_errors(data):
        path = ".".join(str(p) for p in error.absolute_path) if error.absolute_path else "(root)"
        errors.append(f"{path}: {error.message}")

    return errors


def _build_validator(
    schema: dict[str, Any],
    schema_path: Path,
) -> Any | None:
    """Create a ``jsonschema`` validator with a ``$ref`` registry.

    Loads ``_common.schema.json`` from the same directory as *schema_path*
    and registers it so that relative ``$ref`` URIs resolve correctly.
    """
    try:
        import jsonschema
        from referencing import Registry, Resource
        from referencing.jsonschema import DRAFT7
    except ImportError:
        logger.debug("referencing package not available; skipping schema validation")
        return None

    from urllib.parse import urljoin

    # Load the common schema that most rule schemas reference.
    common_path = schema_path.parent / "_common.schema.json"
    common_schema: dict[str, Any] | None = None
    if common_path.exists():
        try:
            with open(common_path, "r", encoding="utf-8") as f:
                common_schema = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    resources: list[tuple[str, Resource]] = []  # type: ignore[type-arg]

    # Register the main schema by its $id.
    schema_id = schema.get("$id", "")
    if schema_id:
        resources.append((schema_id, Resource.from_contents(schema, default_specification=DRAFT7)))

    # Register the common schema by its $id and by the resolved relative URI.
    if common_schema is not None:
        common_resource = Resource.from_contents(common_schema, default_specification=DRAFT7)
        common_id = common_schema.get("$id", "")
        if common_id:
            resources.append((common_id, common_resource))
        if schema_id:
            resolved_common_uri = urljoin(schema_id, "_common.schema.json")
            resources.append((resolved_common_uri, common_resource))

    try:
        registry = Registry().with_resources(resources)
        validator_class = jsonschema.validators.validator_for(schema)
        return validator_class(schema, registry=registry)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to build JSON Schema validator: %s", exc)
        return None


def get_schema_path_for_yaml(yaml_filename: str) -> Path | None:
    """Map a YAML rule filename to its corresponding schema path.

    Parameters
    ----------
    yaml_filename:
        Bare filename such as ``"particles.yaml"``.

    Returns
    -------
    Path | None
        Absolute path to the schema file, or ``None`` if no matching
        schema exists on disk.
    """
    stem = Path(yaml_filename).stem  # e.g. "particles"
    schema_file = _SCHEMAS_DIR / f"{stem}.schema.json"
    if schema_file.exists():
        return schema_file
    return None
