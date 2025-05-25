"""
Path management utilities for MemFuse.

This module provides a centralized way to manage directory creation and path operations
throughout the MemFuse framework, ensuring consistency and traceability.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class PathManager:
    """
    A centralized manager for path operations in MemFuse.

    This class provides methods for creating and managing directories,
    ensuring consistent path handling across the application.
    """

    _instance = None
    _created_dirs: Set[str] = set()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PathManager, cls).__new__(cls)
            cls._instance._created_dirs = set()
        return cls._instance

    @classmethod
    def ensure_directory(cls, directory_path: Union[str, Path], exist_ok: bool = True) -> Path:
        """
        Ensure a directory exists, creating it if necessary.

        Args:
            directory_path: Path to the directory to ensure exists
            exist_ok: If True, don't raise an error if the directory already exists

        Returns:
            Path object for the created/existing directory
        """
        path = Path(directory_path)
        path_str = str(path.absolute())

        if path_str not in cls._created_dirs:
            logger.debug(f"Creating directory: {path_str}")
            os.makedirs(path_str, exist_ok=exist_ok)
            cls._created_dirs.add(path_str)

        return path

    @classmethod
    def get_data_dir(cls, config_data_dir: Union[str, Path]) -> Path:
        """
        Get the data directory path and ensure it exists.

        Args:
            config_data_dir: Base data directory from configuration

        Returns:
            Path object for the data directory
        """
        return cls.ensure_directory(config_data_dir)

    @classmethod
    def get_user_dir(cls, data_dir: Union[str, Path], user_id: str) -> Path:
        """
        Get the user directory path and ensure it exists.

        Args:
            data_dir: Base data directory
            user_id: ID of the user

        Returns:
            Path object for the user directory
        """
        user_dir = Path(data_dir) / user_id
        return cls.ensure_directory(user_dir)

    @classmethod
    def get_store_dir(cls, data_dir: Union[str, Path], store_type: str) -> Path:
        """
        Get the store directory path and ensure it exists.

        Args:
            data_dir: Base data directory
            store_type: Type of store (vector, graph, keyword)

        Returns:
            Path object for the store directory
        """
        store_dir = Path(data_dir) / store_type
        return cls.ensure_directory(store_dir)

    @classmethod
    def get_logs_dir(cls) -> Path:
        """
        Get the logs directory path and ensure it exists.

        Returns:
            Path object for the logs directory
        """
        return cls.ensure_directory("logs")

    @classmethod
    def get_db_dir(cls, data_dir: Union[str, Path], db_name: str) -> Path:
        """
        Get the database directory path and ensure it exists.

        Args:
            data_dir: Base data directory
            db_name: Name of the database

        Returns:
            Path object for the database directory
        """
        db_dir = Path(data_dir) / "db" / db_name
        return cls.ensure_directory(db_dir)

    @classmethod
    def get_created_directories(cls) -> List[str]:
        """
        Get a list of all directories created by the PathManager.

        Returns:
            List of directory paths that have been created
        """
        return list(cls._created_dirs)
