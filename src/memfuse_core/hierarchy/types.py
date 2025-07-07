"""
Common types and enums for the memory hierarchy system.

This module contains shared types to avoid circular imports.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class WriteStrategy(Enum):
    """Strategy for writing data to memory layers."""
    PARALLEL = "parallel"      # Write to all layers simultaneously
    SEQUENTIAL = "sequential"  # Write to layers in order (M0 -> M1 -> M2)
    HYBRID = "hybrid"         # Write to M0 first, then M1 and M2 in parallel


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0
    exponential_backoff: bool = True
    retry_on_timeout: bool = True
    retry_on_error: bool = True
    
    def __post_init__(self):
        """Validate retry configuration."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.base_delay < 0:
            raise ValueError("base_delay must be non-negative")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")


@dataclass
class LayerWriteResult:
    """Result of writing data to a single memory layer."""
    success: bool
    result: Optional[str] = None
    processed_items: Optional[list] = None
    processing_time: float = 0.0
    retry_count: int = 0
    error_message: Optional[str] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.processed_items is None:
            self.processed_items = []


@dataclass
class ParallelWriteResult:
    """Result of parallel write operation across multiple layers."""
    success: bool
    layer_results: dict  # Dict[str, LayerWriteResult]
    total_processing_time: float = 0.0
    strategy_used: Optional[WriteStrategy] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.layer_results is None:
            self.layer_results = {}

    @property
    def successful_layers(self) -> list:
        """Get list of layers that succeeded."""
        return [
            layer_name for layer_name, result in self.layer_results.items()
            if result.success
        ]

    @property
    def failed_layers(self) -> list:
        """Get list of layers that failed."""
        return [
            layer_name for layer_name, result in self.layer_results.items()
            if not result.success
        ]

    @property
    def total_processed(self) -> int:
        """Get total number of processed items across all layers."""
        return sum(
            len(result.processed_items) for result in self.layer_results.values()
            if hasattr(result, 'processed_items') and result.processed_items
        )

    @property
    def total_processed_items(self) -> int:
        """Alias for total_processed for backward compatibility."""
        return self.total_processed

    @property
    def total_errors(self) -> int:
        """Get total number of errors across all layers."""
        total = 0
        for result in self.layer_results.values():
            if hasattr(result, 'errors') and result.errors:
                # ProcessingResult type
                total += len(result.errors)
            elif hasattr(result, 'error_message') and result.error_message:
                # LayerWriteResult type
                total += 1
        return total
