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
| **Lightning Fast**                | Efficient buffering with write aggregation, intelligent prefetching, and query caching for exceptional performance                |
| **Unified Cognitive Search**      | Seamlessly combines vector, graph, and keyword search with intelligent fusion and re-ranking for superior accuracy and insights   |
| **Cognitive Memory Architecture** | Human-inspired layered memory system: M0 (raw data/episodic), M1 (structured facts/semantic), and M2 (knowledge graph/conceptual) |
| **Local-First**                   | Run the server locally or deploy with Docker — no mandatory cloud dependencies or fees                                            |
| **Pluggable Backends**            | Compatible with Chroma, Qdrant, pgvector, Neo4j, Redis, and custom adapters (expanding support)                                   |
| **Multi-Tenant Support**          | Secure isolation between users, agents, and sessions with robust scoping and access controls                                      |
| **Framework-Friendly**            | Seamless integration with LangChain, AutoGen, Vercel AI SDK, and direct OpenAI/Anthropic/Gemini/Ollama API calls                  |
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

2.  Install dependencies and run the server using one of the following methods:

    **Using Poetry (Recommended)**

    ```bash
    poetry install
    poetry run memfuse-core
    ```

    **Using pip**

    ```bash
    pip install -e .
    python -m memfuse_core
    ```

#### Installing the Client SDK

To use MemFuse in your applications, install the Python SDK simply from PyPI

```bash
pip install memfuse
```

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

---

## 📚 Documentation

- **[Installation Guide](https://memfuse.vercel.app/docs/installation)**: Comprehensive instructions for installing and configuring MemFuse
- **[Getting Started](https://memfuse.vercel.app/docs/quickstart)**: Step-by-step guide to integrating MemFuse into your projects
- **[Examples](https://github.com/memfuse/memfuse-python/tree/main/examples)**: Sample implementations for chatbots, autonomous agents, customer support, LangChain integration, and more

---

## 🛣 Roadmap

### 📦 Phase 1 – MVP ("Fast & Transparent Core")

- [x] **Lightning-fast performance** — Efficient buffering with write aggregation, intelligent prefetching, and query caching
- [x] **Level 0 Memory Layer** — Raw chat history storage and retrieval
- [x] **Multi-tenant support** — Secure user, agent, and session isolation
- [ ] **Level 1 Memory Layer** — Semantic and episodic memory processing
- [x] **Re-ranking plugin** — LLM-powered memory relevance scoring
- [x] **Python SDK** — Complete client library for Python applications
- [x] **Benchmarks** — LongMemEval and MSC evaluation frameworks

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
