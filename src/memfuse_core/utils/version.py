"""Version utilities for MemFuse.

This module provides utilities for reading version information from pyproject.toml.
"""

from loguru import logger
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional


def _find_pyproject_toml() -> Optional[Path]:
    """
    Find the pyproject.toml file by searching up the directory tree.

    Returns:
        Path to pyproject.toml if found, None otherwise
    """
    current_path = Path(__file__).resolve()

    # Search up the directory tree for pyproject.toml
    for parent in current_path.parents:
        pyproject_path = parent / "pyproject.toml"
        if pyproject_path.exists():
            return pyproject_path

    logger.warning("Could not find pyproject.toml file")
    return None


def _extract_version_from_data(pyproject_data: dict) -> Optional[str]:
    """
    Extract version from parsed pyproject.toml data.

    Args:
        pyproject_data: Parsed TOML data

    Returns:
        Version string if found, None otherwise
    """
    version = pyproject_data.get("tool", {}).get("poetry", {}).get("version")

    if version:
        logger.debug(f"Found project version: {version}")
        return version
    else:
        logger.warning("Version not found in pyproject.toml")
        return None


@lru_cache(maxsize=1)
def get_project_version() -> Optional[str]:
    """
    Get the project version from pyproject.toml.

    This function uses caching to avoid reading the file multiple times.

    Returns:
        The version string if found, None otherwise
    """
    try:
        pyproject_path = _find_pyproject_toml()
        if pyproject_path is None:
            return None

        # Python 3.11+ has built-in tomllib
        if sys.version_info >= (3, 11):
            import tomllib
            with open(pyproject_path, "rb") as f:
                pyproject_data = tomllib.load(f)
        else:
            # For older Python versions, fall back to toml library
            try:
                import tomli as tomllib  # type: ignore
                with open(pyproject_path, "rb") as f:
                    pyproject_data = tomllib.load(f)
            except ImportError:
                try:
                    import toml as toml_lib  # type: ignore
                    with open(pyproject_path, "r", encoding="utf-8") as f:
                        pyproject_data = toml_lib.load(f)
                except ImportError:
                    logger.warning(
                        "No TOML library available. Install tomli for Python < 3.11"
                    )
                    return None

        return _extract_version_from_data(pyproject_data)

    except Exception as e:
        logger.error(f"Error reading version from pyproject.toml: {e}")
        return None


def get_version_info() -> dict:
    """
    Get comprehensive version information.

    Returns:
        A dictionary containing version information
    """
    version = get_project_version()
    return {
        "version": version or "unknown",
        "python_version": (
            f"{sys.version_info.major}."
            f"{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        )
    }
