# MemFuse Documentation

## Overview

Comprehensive documentation for the MemFuse system architecture and implementation.

## Documentation Structure

### API Documentation (`/api/`)

REST API resources and endpoints:
- **[agents.md](api/resources/agents.md)** - Agent management
- **[knowledge.md](api/resources/knowledge.md)** - Knowledge operations  
- **[memory.md](api/resources/memory.md)** - Memory management
- **[messages.md](api/resources/messages.md)** - Message handling
- **[sessions.md](api/resources/sessions.md)** - Session management
- **[users.md](api/resources/users.md)** - User operations

### Architecture Documentation (`/architecture/`)

#### Buffer System (`/architecture/buffer/`)
- **[README.md](architecture/buffer/README.md)** - Buffer system overview
- **[overview.md](architecture/buffer/overview.md)** - Architecture overview and bypass mechanism
- **[write_buffer.md](architecture/buffer/write_buffer.md)** - WriteBuffer abstraction
- **[query_buffer.md](architecture/buffer/query_buffer.md)** - QueryBuffer and multi-level caching
- **[speculative_buffer.md](architecture/buffer/speculative_buffer.md)** - SpeculativeBuffer design
- **[configuration.md](architecture/buffer/configuration.md)** - Configuration guide
- **[bypass_mechanism.md](architecture/buffer/bypass_mechanism.md)** - Bypass functionality
- **[performance.md](architecture/buffer/performance.md)** - Performance analysis
- **[buffer_only_parameter.md](architecture/buffer/buffer_only_parameter.md)** - Query-time controls
- **[caching.md](architecture/buffer/caching.md)** - Caching strategies

#### Core Architecture
- **[chunking.md](architecture/chunking.md)** - Chunking system architecture
- **[memory.md](architecture/memory.md)** - Memory layer architecture
- **[rag.md](architecture/rag.md)** - RAG system architecture

#### PgAI Integration (`/architecture/pgai/`)
- **[overview.md](architecture/pgai/overview.md)** - PgAI integration overview
- **[multi_layer.md](architecture/pgai/multi_layer.md)** - Multi-layer processing

### Optimization Documentation (`/optimization/`)

Performance optimization guides:
- **[README.md](optimization/README.md)** - Optimization overview
- **[connection_pool.md](optimization/connection_pool.md)** - Connection pool optimization
- **[singleton.md](optimization/singleton.md)** - Singleton pattern implementation
- **[pgai_trigger.md](optimization/pgai_trigger.md)** - PgAI trigger optimization

#### Buffer Optimization (`/optimization/buffer/`)
- **[abstraction.md](optimization/buffer/abstraction.md)** - Buffer abstraction patterns
- **[flush.md](optimization/buffer/flush.md)** - Flush optimization

#### Performance Analysis (`/optimization/performance/`)
- **[README.md](optimization/performance/README.md)** - Performance overview
- **[analysis_and_plan.md](optimization/performance/analysis_and_plan.md)** - Performance analysis
- **[design.md](optimization/performance/design.md)** - Performance design
- **[implementation_guide.md](optimization/performance/implementation_guide.md)** - Implementation guide
- **[phase1_connection_pool.md](optimization/performance/phase1_connection_pool.md)** - Phase 1: Connection pool
- **[phase2_buffer_system.md](optimization/performance/phase2_buffer_system.md)** - Phase 2: Buffer system
- **[phase3_memory_layers.md](optimization/performance/phase3_memory_layers.md)** - Phase 3: Memory layers
- **[phase4_advanced.md](optimization/performance/phase4_advanced.md)** - Phase 4: Advanced features

#### Parallel Processing
- **[parallel_processing.md](optimization/parallel_processing.md)** - Parallel processing optimization

### Assets (`/assets/`)

- `logo.png` - MemFuse logo and branding assets

## Quick Start

### For Developers

1. **Buffer System**: Start with [`architecture/buffer/README.md`](architecture/buffer/README.md)
2. **API Reference**: Check [`api/resources/`](api/resources/) for endpoint documentation
3. **Performance**: Review [`optimization/README.md`](optimization/README.md) for optimization guides

### For Architects

1. **System Overview**: Read [`architecture/buffer/overview.md`](architecture/buffer/overview.md)
2. **Memory Architecture**: Review [`architecture/memory.md`](architecture/memory.md)
3. **Performance Design**: Check [`optimization/performance/design.md`](optimization/performance/design.md)

## Key Features

- **Multi-Layer Buffer System**: WriteBuffer, QueryBuffer, SpeculativeBuffer with bypass capability
- **Advanced Memory Architecture**: M0/M1/M2 memory layers with parallel processing
- **Performance Optimized**: Connection pooling, parallel processing, and intelligent caching
- **Comprehensive API**: RESTful services for all system components
- **Pluggable Storage**: Support for multiple storage backends
- **Production Ready**: Extensive configuration and monitoring capabilities

## Documentation Standards

- **Language**: All documentation in English
- **Structure**: Clear hierarchical organization with consistent formatting
- **Diagrams**: Mermaid flowcharts for architectural visualization
- **Code Examples**: Practical implementation examples
- **Configuration**: YAML configuration examples with explanations

## Contributing

1. Place documentation in appropriate subdirectories
2. Use descriptive, lowercase filenames with hyphens
3. Follow established template patterns
4. Ensure technical accuracy and test configurations
5. Update cross-references between related documents