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
    <strong>MemFuse Python SDK</strong>
    <br />
    The official Python client for MemFuse, the open-source memory layer for LLMs.
    <br />
    <a href="https://memfuse.vercel.app/"><strong>Explore the Docs »</strong></a>
    <br />
    <br />
    <a href="https://memfuse.vercel.app/">View Demo</a>
    &middot;
    <a href="https://github.com/Percena/MemFuse/issues">Report Bug</a>
    &middot;
    <a href="https://github.com/Percena/MemFuse/issues">Request Feature</a>
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
| **Multi-tenant support**          | Secure isolation between users, agents, and sessions with built-in scoping and access controls                                   |
| **Framework-friendly**            | Drop-in providers for LangChain, AutoGen, Vercel AI SDK & raw OpenAI/Anthropic/Gemini/Ollama calls                               |
| **Apache 2.0**                    | Fully open source. Fork, extend, or ship as you like                                                                             |

---

## 🚀 Quick start

### Installation

First, ensure you have a MemFuse server running. To set up the MemFuse server locally:

1.  Clone the [main MemFuse repository](https://github.com/memfuse/memfuse):
    ```bash
    git clone git@github.com:memfuse/memfuse.git
    cd memfuse
    ```
2.  Once in the `memfuse` directory, install its dependencies and run the server using one of the following methods:

    **Using pip:**

    ```bash
    pip install -e .
    python -m memfuse_core
    ```

    **Or using Poetry:**

    ```bash
    poetry install
    poetry run memfuse-core
    ```

Then, install the MemFuse Python SDK:

```bash
pip install memfuse
```

For detailed installation instructions, configuration options, and troubleshooting tips, see the [Installation Guide](docs/installation.md).

### Basic Usage

Here's a basic example of how to use the MemFuse Python SDK with OpenAI:

```python
from memfuse.llm import OpenAI
from memfuse import MemFuse
import os


memfuse_client = MemFuse(
  # base_url=os.getenv("MEMFUSE_BASE_URL"),
  # api_key=os.getenv("MEMFUSE_API_KEY")
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

<!-- Ask a follow-up question. MemFuse automatically recalls relevant context. -->

Now, ask a follow-up question. MemFuse will automatically recall relevant context from the previous turn:

```python
# Ask a follow-up question. MemFuse automatically recalls relevant context.
followup_response = llm_client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What are some challenges of living on that planet?"}]
)

print(f"Follow-up: {followup_response.choices[0].message.content}")
# Example Output: Follow-up: Some challenges of living on Mars include its thin atmosphere, extreme temperatures, high radiation levels, and the lack of liquid water on the surface.
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

- [x] Lightning-fast—Efficient buffering (write aggregation, intelligent prefetching, query caching) for rapid performance
- [x] Level 0 Memory Layer—raw chat history
- [x] Multi-tenant support
- [ ] Level 1 Memory Layer—semantic/episodic memories
- [x] Re-ranking plugin–LLM-powered memory scoring
- [x] Python SDK
- [x] Benchmarks: LongMemEval + MSC

### 🧭 Phase 2 – Temporal Mastery & Quality

- [ ] JavaScript SDK
- [ ] Multimodal memory support
- [ ] Level 2 KG memory support
- [ ] Time-decay policies–automatic forgetting of stale items

Have an idea? Open an issue or vote on the discussion board!

## 🤝 Community & Support

- GitHub Discussions: roadmap votes, RFCs, Q&A

If MemFuse saves you time, please ⭐ star the repo — it helps the project grow!
