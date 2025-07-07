#!/usr/bin/env python3
"""Test configuration loading with environment variables."""

import os
import sys
from pathlib import Path

# Add src to path (adjusted for tests/unit/ location)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from memfuse_core.utils.config import config_manager
import hydra
from omegaconf import DictConfig, OmegaConf

# Load environment variables
import dotenv
dotenv.load_dotenv()

print("Environment variables:")
print(f"POSTGRES_HOST: {os.getenv('POSTGRES_HOST')}")
print(f"POSTGRES_PORT: {os.getenv('POSTGRES_PORT')}")
print(f"POSTGRES_DB: {os.getenv('POSTGRES_DB')}")
print(f"POSTGRES_USER: {os.getenv('POSTGRES_USER')}")
print(f"POSTGRES_PASSWORD: {os.getenv('POSTGRES_PASSWORD')}")

@hydra.main(version_base=None, config_path="config", config_name="config")
def test_config(cfg: DictConfig) -> None:
    """Test configuration loading."""
    print("\nHydra configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    print("\nDatabase configuration:")
    db_config = cfg.get("database", {})
    print(f"Database type: {db_config.get('type')}")
    
    postgres_config = db_config.get("postgres", {})
    print(f"PostgreSQL host: {postgres_config.get('host')}")
    print(f"PostgreSQL port: {postgres_config.get('port')}")
    print(f"PostgreSQL database: {postgres_config.get('database')}")
    print(f"PostgreSQL user: {postgres_config.get('user')}")
    print(f"PostgreSQL password: {postgres_config.get('password')}")

if __name__ == "__main__":
    test_config()
