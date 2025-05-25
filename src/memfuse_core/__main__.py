"""
Entry point for running the MemFuse server as a module.
This allows running the server with `python -m memfuse_core`.
"""

import hydra
from omegaconf import DictConfig

from .server import run_server


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the MemFuse server with Hydra configuration.

    Args:
        cfg: Configuration from Hydra
    """
    # Run the server with the Hydra configuration directly
    run_server(cfg)


if __name__ == "__main__":
    main()
