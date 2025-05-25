<a id="readme-top"></a>

[![GitHub license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/Percena/MemFuse/blob/readme/LICENSE)

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://memfuse.vercel.app/">
    <img src="docs/assets/logo.png" alt="MemFuse Logo"
         style="max-width: 90%; height: auto; display: block; margin: 0 auto; padding-left: 16px; padding-right: 16px;" height="120">
  </a>
  <br />
  <br />

  <p align="center">
    The open-source memory layer for LLMs. Built by developers who were tired of forgetful agents.
    <br />
    <a href="https://memfuse.vercel.app/"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://memfuse.vercel.app/">View Demo</a>
    &middot;
    <a href="https://memfuse.vercel.app/">Report Bug</a>
    &middot;
    <a href="https://memfuse.vercel.app/">Request Feature</a>
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

Large-language-model apps are stateless out of the box.
Once the context window rolls over, yesterday's chat, the user's name, or that crucial fact vanishes.

**MemFuse** plugs a persistent, query-able memory layer between your LLM and a storage backend so agents can:

- remember user preferences across sessions
- recall facts & events thousands of turns later
- trim token spend instead of resending the whole chat history
- learn continuously and self-improve over time

## ✨ Key Features

| Category                          | What you get                                                                                                                     |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Lightning fast**                | Efficient buffering (write aggregation, intelligent prefetching, query caching) for rapid performance                            |
| **Unified Cognitive Search**      | Synergizes vector, graph, and keyword search with intelligent fusion & re-ranking for exceptional accuracy and diverse insights. |
| **Cognitive Memory Architecture** | Human-inspired layered memory: L0 (raw data/episodic), L1 (structured facts/semantic), and L2 (knowledge graph/conceptual).      |
| **Local-first**                   | Run the server locally or use Docker — no mandatory cloud fees                                                                   |
| **Pluggable back-ends**           | Works with Chroma, Qdrant, pgvector, Neo4j, Redis, or any custom adapter (in progress)                                           |
| **Framework-friendly**            | Drop-in providers for LangChain, AutoGen, Vercel AI SDK & raw OpenAI/Anthropic/Gemini/Ollama calls                               |
| **Apache 2.0**                    | Fully open source. Fork, extend, or ship as you like                                                                             |

---

## 🚀 Quick start

### Installation

First set up backend using Docker:

```bash
docker run --rm -p 8000:8000 \
  -e MF_STORAGE=qdrant \
  -e MF_VECTOR_URL=http://qdrant:6333 \
  ghcr.io/memfuse/memfuse:latest
```

Or start the MemFuse server locally:

```bash
# Clone the repository
git clone https://github.com/Percena/MemFuse.git
cd MemFuse

# Install and start the server with Poetry (recommended)
poetry install
poetry run memfuse-core

# Or using Python module directly
pip install -e .
python -m memfuse_core
```

Then install the Python client:

```bash
pip install memfuse
```

For detailed installation instructions, configuration options, and troubleshooting tips, see the [Installation Guide](docs/installation.md).

### Basic Usage

```python
from memfuse import MemFuse
from memfuse.llm import OpenAI

# Initialize MemFuse client
mem = MemFuse(
    api_key="MEMFUSE_API_KEY",
    base_url="http://localhost:8000"  # Connect to the local server
)

memory_scope = mem.scope(user="alice")  # user / agent / session scopes

client = OpenAI(
    api_key="OPENAI_API_KEY",
    memory_scope=memory_scope
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user",
               "content": "I'm planning a Mars mission. Remind me what gravity on Mars is?"}]
)

print(response.choices[0].message.content)
# → "Mars gravity is ~3.721 m/s² …"
```

🔥 That's it.
Every subsequent call under the same scope automatically writes notable facts to memory and fetches them when relevant.

---

## 📚 Documentation

- [Installation Guide](docs/installation.md): Detailed instructions for installing and configuring MemFuse
- [Getting Started](docs/getting-started.md): Guide to using MemFuse in your projects
- [API Reference](docs/reference): Autogenerated API documentation
- [Examples](examples/): Sample code for chat-bots, autonomous agents, customer support, LangChain integration, etc.
- [Architecture](docs/architecture.md): Overview of the MemFuse architecture and design principles

---

## 🛣 Roadmap

### 📦 Phase 1 – MVP ("Fast & Transparent Core")

- [x] Level 0 Memory Layer—raw chat history
- [x] Level 1 Memory Layer—semantic/episodic memories
- [x] User Management (CRUD)
- [x] Agent Management (CRUD)
- [x] Python SDK
- [x] Benchmarks: LongMemEval + MSC
- [x] ...and much more (👉🏻 see the roadmap board!)

### 🧭 Phase 2 – Temporal Mastery & Quality

- [ ] Level 1 Memory Layer—multimodal memory support
- [ ] Time-decay policies–automatic forgetting of stale items
- [ ] Re-ranking plugin–LLM-powered memory scoring
- [ ] JavaScript SDK

Have an idea? Open an issue or vote on the discussion board!

## 🤝 Community & Support

- Discord: join the conversation, get help, or show off your build
- GitHub Discussions: roadmap votes, RFCs, Q&A
- Twitter / X: @MemFuse — launch news & tips

If MemFuse saves you time, please ⭐ star the repo — it helps the project grow!
