# Buffer Configuration Guide

## Overview

This document provides comprehensive configuration guidance for the MemFuse Buffer system, covering all components, deployment scenarios, and optimization strategies.

## Configuration File Location

**Primary Configuration**: `/config/buffer/default.yaml`

## Complete Configuration Schema

### Full Configuration Template

```yaml
# MemFuse Buffer Configuration
buffer:
  # CRITICAL CONTROL: Enable/disable entire buffer system
  enabled: true                  # true = full buffer, false = complete bypass

  # RoundBuffer configuration - Token-based FIFO buffer
  round_buffer:
    max_tokens: 800               # Token limit before transfer to HybridBuffer
    max_size: 5                   # Maximum number of rounds before forced transfer
    token_model: "gpt-4o-mini"    # Model for token counting
    enable_session_tracking: true # Track sessions for context grouping
    auto_transfer_threshold: 0.8  # Transfer when 80% of token limit reached

  # HybridBuffer configuration - Dual-format buffer with FIFO
  hybrid_buffer:
    max_size: 5                   # Maximum number of items in buffer (FIFO)
    chunk_strategy: "message"     # Chunking strategy: "message" or "contextual"
    embedding_model: "all-MiniLM-L6-v2"  # Embedding model for vector generation
    enable_auto_flush: true       # Enable automatic flushing
    auto_flush_interval: 60.0     # Auto-flush interval in seconds
    chunk_overlap: 0.1            # Overlap ratio for contextual chunking

  # QueryBuffer configuration - Multi-level cache hierarchy
  query:
    max_size: 15                  # Maximum results per query
    cache_size: 100               # Query cache size (LRU)
    default_sort_by: "score"      # Default sort field: "score" or "timestamp"
    default_order: "desc"         # Default sort order: "asc" or "desc"
    cache_ttl: 300                # Cache time-to-live in seconds
    enable_session_queries: true  # Enable session-based queries
    similarity_threshold: 0.95    # Query similarity threshold for cache hits

  # SpeculativeBuffer configuration (Future implementation)
  speculative_buffer:
    max_size: 1000                # Maximum prefetch cache size
    context_window: 5             # Number of recent items to analyze
    prediction_strategy: "semantic_similarity"  # Prediction strategy
    enable_learning: true         # Enable adaptive learning
    prediction_threshold: 0.7     # Minimum confidence for predictions

  # Token counter configuration
  token_counter:
    model: "gpt-4o-mini"          # Default model for token counting
    fallback_multiplier: 1.3      # Multiplier for word-based fallback

  # Performance settings (includes FlushManager configuration)
  performance:
    batch_write_threshold: 5      # Threshold for batch writes
    flush_interval: 60            # Auto-flush interval in seconds
    enable_async_processing: true # Enable async chunk processing
    enable_auto_flush: true       # Enable automatic flushing

    # FlushManager settings
    max_flush_workers: 3          # Maximum number of concurrent flush workers
    max_flush_queue_size: 100     # Maximum size of the flush queue
    flush_timeout: 30.0           # Default timeout for flush operations (seconds)
    flush_strategy: "hybrid"      # Flush strategy: "size_based", "time_based", "hybrid"
    retry_attempts: 3             # Number of retry attempts for failed operations
    retry_delay: 1.0              # Delay between retry attempts (seconds)

  # Monitoring and logging
  monitoring:
    enable_stats: true            # Enable statistics collection
    log_level: "INFO"             # Log level for buffer operations
    performance_tracking: true    # Track performance metrics
    health_check_interval: 30     # Health check interval in seconds
```

## Configuration Scenarios

### 1. Production High-Throughput

**Use Case**: High-volume production environment with emphasis on throughput

```yaml
buffer:
  enabled: true
  
  round_buffer:
    max_tokens: 1200              # Higher token limit for larger batches
    max_size: 8                   # More rounds for better batching
    token_model: "gpt-4o-mini"
  
  hybrid_buffer:
    max_size: 8                   # Larger buffer for better batching
    chunk_strategy: "contextual"  # Better chunking for complex content
    embedding_model: "all-MiniLM-L6-v2"
  
  query:
    cache_size: 200               # Larger cache for better hit rates
    max_size: 20                  # More results for complex queries
  
  performance:
    max_flush_workers: 5          # More workers for high throughput
    max_flush_queue_size: 200     # Larger queue for burst handling
    flush_strategy: "size_based"  # Optimize for throughput
  
  monitoring:
    log_level: "WARNING"          # Reduce log noise in production
```

### 2. Development Low-Latency

**Use Case**: Development environment with emphasis on low latency and debugging

```yaml
buffer:
  enabled: false                 # Complete bypass for direct data flow
  
  monitoring:
    log_level: "DEBUG"            # Detailed logging for debugging
    performance_tracking: true    # Track performance for optimization
```

### 3. Balanced Production

**Use Case**: Balanced production environment (recommended default)

```yaml
buffer:
  enabled: true
  
  round_buffer:
    max_tokens: 800
    max_size: 5
  
  hybrid_buffer:
    max_size: 5
    chunk_strategy: "message"
  
  query:
    cache_size: 100
    max_size: 15
  
  performance:
    max_flush_workers: 3
    flush_strategy: "hybrid"
  
  monitoring:
    log_level: "INFO"
```

### 4. Memory-Constrained Environment

**Use Case**: Limited memory environment requiring optimization

```yaml
buffer:
  enabled: true
  
  round_buffer:
    max_tokens: 400               # Smaller batches
    max_size: 3                   # Fewer rounds in memory
  
  hybrid_buffer:
    max_size: 3                   # Smaller buffer
  
  query:
    cache_size: 50                # Smaller cache
    max_size: 10                  # Fewer results
  
  performance:
    max_flush_workers: 2          # Fewer workers
    max_flush_queue_size: 50      # Smaller queue
```

### 5. High-Frequency Queries

**Use Case**: Environment with frequent, repetitive queries

```yaml
buffer:
  enabled: true
  
  query:
    cache_size: 500               # Large cache for query optimization
    cache_ttl: 600                # Longer TTL for stable content
    similarity_threshold: 0.90    # More aggressive cache matching
    max_size: 25                  # More results for complex queries
  
  performance:
    enable_async_processing: true # Async for better query performance
```

## Component-Specific Configuration

### RoundBuffer Configuration

```yaml
round_buffer:
  max_tokens: 800                 # Recommended: 400-1200 based on content size
  max_size: 5                     # Recommended: 3-10 based on memory constraints
  token_model: "gpt-4o-mini"      # Options: "gpt-4o-mini", "gpt-3.5-turbo"
  enable_session_tracking: true   # Always recommended for context
  auto_transfer_threshold: 0.8    # Recommended: 0.7-0.9
```

**Tuning Guidelines**:
- **Higher max_tokens**: Better batching, higher memory usage
- **Higher max_size**: More buffering, potential latency increase
- **Lower thresholds**: More frequent transfers, lower latency

### HybridBuffer Configuration

```yaml
hybrid_buffer:
  max_size: 5                     # Recommended: 3-10 based on throughput needs
  chunk_strategy: "message"       # "message" for simple, "contextual" for complex
  embedding_model: "all-MiniLM-L6-v2"  # Stable, efficient model
  enable_auto_flush: true         # Always recommended
  auto_flush_interval: 60.0       # Recommended: 30-120 seconds
```

**Chunk Strategy Selection**:
- **"message"**: Faster, simpler, good for chat-like content
- **"contextual"**: Better semantic coherence, higher processing cost

### QueryBuffer Configuration

```yaml
query:
  max_size: 15                    # Recommended: 10-25 based on query complexity
  cache_size: 100                 # Recommended: 50-500 based on memory and query patterns
  default_sort_by: "score"        # "score" for relevance, "timestamp" for recency
  default_order: "desc"           # "desc" for most relevant first
  cache_ttl: 300                  # Recommended: 180-600 seconds
  similarity_threshold: 0.95      # Recommended: 0.90-0.98
```

**Cache Optimization**:
- **Larger cache_size**: Better hit rates, higher memory usage
- **Lower similarity_threshold**: More cache hits, potential relevance reduction
- **Longer cache_ttl**: Better performance, potential stale data

### Performance Configuration

```yaml
performance:
  max_flush_workers: 3            # Recommended: 2-5 based on CPU cores
  max_flush_queue_size: 100       # Recommended: 50-200 based on burst patterns
  flush_timeout: 30.0             # Recommended: 15-60 seconds
  flush_strategy: "hybrid"        # "hybrid" for balanced, "size_based" for throughput
  retry_attempts: 3               # Recommended: 2-5
  retry_delay: 1.0                # Recommended: 0.5-2.0 seconds
```

**Strategy Selection**:
- **"size_based"**: Optimizes for throughput, may increase latency
- **"time_based"**: Optimizes for latency, may reduce throughput
- **"hybrid"**: Balanced approach (recommended)

## Environment-Specific Configurations

### Development Environment

```yaml
# config/buffer/development.yaml
buffer:
  enabled: false                  # Bypass for direct debugging
  monitoring:
    log_level: "DEBUG"
    performance_tracking: true
```

### Testing Environment

```yaml
# config/buffer/testing.yaml
buffer:
  enabled: true
  round_buffer:
    max_tokens: 200               # Small batches for predictable testing
    max_size: 2
  hybrid_buffer:
    max_size: 2
  query:
    cache_size: 20                # Small cache for testing
  monitoring:
    log_level: "DEBUG"
```

### Production Environment

```yaml
# config/buffer/production.yaml
buffer:
  enabled: true
  # Use balanced or high-throughput configuration
  monitoring:
    log_level: "WARNING"          # Reduce log noise
    enable_stats: true            # Enable for monitoring
```

## Configuration Validation

### Required Parameters

**Minimum Required Configuration**:
```yaml
buffer:
  enabled: true                   # Only required parameter
```

**All other parameters have sensible defaults and are optional.**

### Validation Rules

1. **enabled**: Must be boolean (true/false)
2. **max_tokens**: Must be positive integer (100-5000)
3. **max_size**: Must be positive integer (1-50)
4. **cache_size**: Must be positive integer (10-1000)
5. **flush_timeout**: Must be positive number (5.0-300.0)

### Configuration Testing

```python
# Test configuration validity
from memfuse_core.buffer.config_factory import BufferConfigManager

config = {...}  # Your configuration
config_manager = BufferConfigManager(config)
is_valid = config_manager.validate_configuration()

if not is_valid:
    print("Configuration validation failed")
```

## Performance Tuning Guidelines

### Memory Optimization

1. **Reduce buffer sizes** for memory-constrained environments
2. **Lower cache sizes** if memory usage is high
3. **Use bypass mode** for minimal memory footprint

### Latency Optimization

1. **Use bypass mode** for lowest latency
2. **Reduce max_tokens** for faster transfers
3. **Increase flush_workers** for parallel processing

### Throughput Optimization

1. **Increase buffer sizes** for better batching
2. **Use "size_based" flush strategy**
3. **Increase worker counts** for parallel processing

### Query Performance

1. **Increase cache_size** for better hit rates
2. **Tune similarity_threshold** for cache efficiency
3. **Optimize cache_ttl** based on data freshness needs

## Monitoring Configuration

### Statistics Collection

```yaml
monitoring:
  enable_stats: true              # Enable comprehensive statistics
  performance_tracking: true     # Track performance metrics
  health_check_interval: 30      # Regular health checks
```

### Log Level Guidelines

- **"DEBUG"**: Development, detailed troubleshooting
- **"INFO"**: Default, balanced information
- **"WARNING"**: Production, errors and warnings only
- **"ERROR"**: Critical issues only

## Configuration Migration

### Upgrading Configuration

When upgrading MemFuse versions:

1. **Backup current configuration**
2. **Review new configuration options**
3. **Test in development environment**
4. **Gradually roll out to production**

### Configuration Compatibility

The buffer system maintains backward compatibility:
- **Missing parameters**: Use sensible defaults
- **Deprecated parameters**: Log warnings but continue operation
- **Invalid values**: Fall back to defaults with warnings

## Related Documentation

- **[Overview](overview.md)** - Buffer system overview
- **[Bypass Mechanism](bypass_mechanism.md)** - Detailed bypass functionality
- **[Performance](performance.md)** - Performance analysis and optimization
- **[Write Buffer](write_buffer.md)** - Write path configuration details
