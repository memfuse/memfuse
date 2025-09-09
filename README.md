<a id="readme-top"></a>

[![GitHub license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/Percena/MemFuse/blob/readme/LICENSE)

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://memfuse.vercel.app/">
    <img src="docs/assets/logo.png" alt="MemFuse Logo"
         style="max-width: 90%; height: auto; display: block; margin: 0 auto; padding-left: 16px; padding-right: 16px;">
  </a>
  <br />
  <br />

  <p align="center">
    <strong>MemFuse Core Services</strong>
    <br />
    The official core services for MemFuse, the open-source memory layer for LLMs.
    <br />
    <a href="https://memfuse.vercel.app/"><strong>Explore the Docs »</strong></a>
    <br />
    <br />
    <a href="https://memfuse.vercel.app/">View Demo</a>
    &middot;
    <a href="https://github.com/memfuse/memfuse/issues">Report Bug</a>
    &middot;
    <a href="https://github.com/memfuse/memfuse/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#why-memfuse">Why MemFuse?</a>
    </li>
    <li>
      <a href="#key-features">Key Features</a>
    </li>
    <li><a href="#quick-start">Quick Start</a></li>
    <li><a href="#documentation">Documentation</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#community-support">Community & Support</a></li>
  </ol>
</details>

## Why MemFuse?

Large language model applications are inherently stateless by design.
When the context window reaches its limit, previous conversations, user preferences, and critical information simply disappear.

**MemFuse** bridges this gap by providing a persistent, queryable memory layer between your LLM and storage backend, enabling AI agents to:

- **Remember** user preferences and context across sessions
- **Recall** facts and events from thousands of interactions later
- **Optimize** token usage by avoiding redundant chat history resending
- **Learn** continuously and improve performance over time

This repository contains the official server core services for seamless integration with MemFuse Client SDK. For comprehensive information about the MemFuse Client features, please visit the [MemFuse Client Python SDK](https://github.com/memfuse/memfuse-python).

## ✨ Key Features

| Category                          | What you get                                                                                                                      |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Lightning Fast**                | 52.6x performance improvement through intelligent caching, sub-10ms query responses, and 1000+ requests/second throughput        |
| **Unified Cognitive Search**      | Seamlessly combines vector, graph, and keyword search with intelligent fusion and re-ranking for superior accuracy and insights   |
| **Cognitive Memory Architecture** | Human-inspired layered memory system: M0 (raw), M1 (episodic), M2 (semantic), M3 (procedural), MSMG (meta-semantic graph)      |
| **Enterprise-Grade Performance** | Advanced caching system, memory leak detection, performance regression testing, and comprehensive monitoring                      |
| **Local-First**                   | Run the server locally or deploy with Docker/Kubernetes — no mandatory cloud dependencies or fees                                |
| **Pluggable Backends**            | Built on TimescaleDB with custom pgai implementation, compatible with Qdrant, Neo4j, Redis, and expanding backend support        |
| **Multi-Tenant Support**          | Secure isolation between users, agents, and sessions with robust scoping and access controls                                      |
| **Framework-Friendly**            | Seamless integration with LangChain, AutoGen, Vercel AI SDK, and direct OpenAI/Anthropic/Gemini/Ollama API calls                  |
| **Production-Ready Testing**      | 111 comprehensive tests with performance regression detection, memory leak monitoring, and automated error scenario validation    |
| **Enterprise Observability**      | Prometheus metrics, Grafana dashboards, Jaeger tracing, and comprehensive monitoring with automated alerting                     |
| **Apache 2.0 Licensed**           | Fully open source — fork, extend, customize, and deploy as you need                                                               |

---

## 🚀 Quick Start

### Installation

> **Note**: This repository contains the **MemFuse Core Server**. If you need to know more about the standalone Python SDK for client applications, please visit the [MemFuse Client Python SDK](https://github.com/memfuse/memfuse-python).

#### Setting Up the MemFuse Server

To set up the MemFuse server locally:

1.  Clone this repository:

    ```bash
    git clone https://github.com/memfuse/memfuse.git
    cd memfuse
    ```

2.  Verify Docker availability and pull the required database image:

    ```bash
    # Ensure Docker daemon is running first
    docker --version  # Verify Docker is available
    
    # Pull the TimescaleDB image
    docker pull timescale/timescaledb-ha:pg17
    ```

3.  Install dependencies and run the server using one of the following methods:

    **Using Poetry (Recommended)**

    ```bash
    poetry install
    poetry run python scripts/memfuse_launcher.py
    ```

    **Using pip**

    ```bash
    pip install -e .
    python scripts/memfuse_launcher.py
    ```

#### Installing the Client SDK

To use MemFuse in your applications, install the Python SDK simply from PyPI

```bash
pip install memfuse
```

### Database Requirements

MemFuse uses **TimescaleDB** as its primary database backend with a custom pgai-like implementation for advanced vector operations and automatic embedding generation.

**Prerequisites:**
- **Docker daemon must be running** (required for database startup)
- Docker and Docker Compose installed
- **TimescaleDB**: Automatically provided via `timescale/timescaledb-ha:pg17` Docker image
- **pgvector**: Vector similarity search (included in TimescaleDB image)
- **Custom pgai**: Event-driven embedding system (built-in, no external extension needed)

> **⚠️ Important**: Make sure Docker daemon is running before starting MemFuse. The service automatically starts the TimescaleDB container.

> **Note**: MemFuse implements its own pgai-like functionality and does **not** require TimescaleDB's official pgai extension.

For detailed installation instructions, configuration options, and troubleshooting tips, see the online [Installation Guide](https://memfuse.vercel.app/docs/installation).

### Basic Usage

Here's a comprehensive example demonstrating how to use the MemFuse Python SDK with OpenAI to interact with the MemFuse server:

```python
from memfuse.llm import OpenAI
from memfuse import MemFuse
import os

memfuse_client = MemFuse(
  # api_key=os.getenv("MEMFUSE_API_KEY")
  # base_url=os.getenv("MEMFUSE_BASE_URL"),
)

memory = memfuse_client.init(
  user="alice",
  # agent="agent_default",
  # session=<randomly-generated-uuid>
)

# Initialize your LLM client with the memory scope
llm_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # Your OpenAI API key
    memory=memory
)

# Make a chat completion request
response = llm_client.chat.completions.create(
    model="gpt-4o", # Or any model supported by your LLM provider
    messages=[{"role": "user", "content": "I'm planning a trip to Mars. What is the gravity there?"}]
)

print(f"Response: {response.choices[0].message.content}")
# Example Output: Response: Mars has a gravity of about 3.721 m/s², which is about 38% of Earth's gravity.
```

### Contextual Follow-up

Now, ask a follow-up question. MemFuse will automatically recall relevant context from the previous conversation:

```python
# Ask a follow-up question. MemFuse automatically recalls relevant context.
followup_response = llm_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What are some challenges of living on that planet?"}]
)

print(f"Follow-up: {followup_response.choices[0].message.content}")
# Example Output: Follow-up: Some challenges of living on Mars include its thin atmosphere, extreme temperatures, high radiation levels, and the lack of liquid water on the surface.
```

🔥 **That's it!** Every subsequent call under the same scope automatically stores notable facts to memory and retrieves them when relevant.

### Performance & Enterprise Features

MemFuse delivers enterprise-grade performance and reliability:

#### 🚀 Performance Metrics
- **52.6x faster** regex pattern caching (47.06μs → 0.89μs)
- **Sub-10ms** query response times with intelligent caching
- **1000+ requests/second** throughput on single instance
- **>90% cache hit rate** in typical workloads
- **Zero memory leaks** with automated detection and cleanup

#### 🔧 Enterprise Capabilities
- **Comprehensive Monitoring**: Prometheus metrics, Grafana dashboards, Jaeger tracing
- **Production Deployment**: Docker and Kubernetes configurations with health checks
- **High Availability**: Load balancing, failover, and backup/recovery strategies
- **Performance Testing**: Automated regression detection with 111 comprehensive tests
- **Security**: PII detection, sensitive word filtering, and content validation

#### 📊 System Architecture
```
Request → Gateway (Filters) → Buffer (Cache) → Memory (Layers) → Persistence
    ↑                                                                    ↓
    └── Monitoring ← Observability ← Performance ← Optimization ←────────┘
```

For detailed performance metrics and optimization guides, see our [Performance Documentation](docs/performance/optimization_guide.md).

### Database Management

MemFuse provides comprehensive database management tools:

```bash
# Check database status and extensions
poetry run python scripts/database_manager.py status

# Reset data (keep schema) 
poetry run python scripts/database_manager.py reset

# Validate schema and triggers
poetry run python scripts/database_manager.py validate

# Complete schema rebuild (⚠️ destroys data)
poetry run python scripts/database_manager.py recreate
```

For detailed database management, see the [Scripts Documentation](scripts/README.md).

---

## 📚 Documentation

### Core Documentation
- **[Installation Guide](https://memfuse.vercel.app/docs/installation)**: Comprehensive instructions for installing and configuring MemFuse
- **[Getting Started](https://memfuse.vercel.app/docs/quickstart)**: Step-by-step guide to integrating MemFuse into your projects
- **[API Documentation](docs/api/api_documentation.md)**: Complete REST API reference with examples
- **[System Architecture](docs/architecture/system_architecture.md)**: Detailed system design and component interactions

### Development & Operations
- **[Development Process](docs/development/development_process.md)**: Comprehensive development methodology and technical decisions
- **[Performance Optimization](docs/performance/optimization_guide.md)**: Performance tuning strategies and optimization results
- **[Production Deployment](docs/deployment/production_deployment_guide.md)**: Enterprise deployment with Docker and Kubernetes
- **[Testing Guide](docs/testing/testing_guide.md)**: Complete testing framework and best practices
- **[Troubleshooting](docs/troubleshooting/common_issues.md)**: Common issues and solutions

### Examples & Integration
- **[Examples](https://github.com/memfuse/memfuse-python/tree/main/examples)**: Sample implementations for chatbots, autonomous agents, customer support, LangChain integration, and more
- **[Scripts Documentation](scripts/README.md)**: Database management and utility scripts

---

## 🛣 Roadmap

### 📦 Phase 1 – MVP ("Fast & Transparent Core") ✅ COMPLETED

- [x] **Lightning-fast performance** — 52.6x performance improvement with intelligent caching and sub-10ms responses
- [x] **Level 0 Memory Layer** — Raw chat history storage and retrieval
- [x] **Multi-tenant support** — Secure user, agent, and session isolation
- [x] **Level 1 Memory Layer** — Semantic and episodic memory processing
- [x] **TimescaleDB Integration** — Production-ready database backend with custom pgai implementation
- [x] **Enhanced Memory Architecture** — M0 (raw), M1 (episodic), M2 (semantic), M3 (procedural), MSMG (graph) layers
- [x] **Complete REST API** — Users, Agents, Sessions, Messages, and Knowledge endpoints
- [x] **Re-ranking plugin** — LLM-powered memory relevance scoring
- [x] **Python SDK** — Complete client library for Python applications
- [x] **Benchmarks** — LongMemEval and MSC evaluation frameworks
- [x] **Enterprise Features** — Performance monitoring, memory leak detection, comprehensive testing
- [x] **Production Deployment** — Docker/Kubernetes configurations with monitoring stack
- [x] **Gateway System** — Bidirectional filtering with semantic validation and security features
- [x] **Observability** — Prometheus metrics, Grafana dashboards, Jaeger tracing

### 🧭 Phase 2 – Temporal Mastery & Quality

- [ ] **JavaScript SDK** — Client library for Node.js and browser applications
- [ ] **Multimodal memory support** — Image, audio, and video memory capabilities
- [ ] **Level 2 KG memory support** — Knowledge graph-based conceptual memory
- [ ] **Time-decay policies** — Automatic forgetting of stale information

💡 **Have an idea?** Open an issue or participate in our discussion board!

## 🤝 Community & Support

- **GitHub Discussions**: Participate in roadmap votes, RFCs, and Q&A sessions
- **Issues**: Report bugs and request new features
- **Documentation**: Comprehensive guides and API references

If MemFuse enhances your projects, please ⭐ **star the repository** — it helps the project grow and reach more developers!

## License

This MemFuse Server repository is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for complete details.
