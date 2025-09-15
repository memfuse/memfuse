#!/usr/bin/env python3
"""
Run LME benchmark with direct m2_semantic vector search (no SDK imports).

Features:
- Loads LME dataset from HuggingFace (configurable)
- Iterates questions and prompts an LLM to answer MCQs
- Uses SentenceTransformer embeddings to vector-search `m2_semantic.embedding`
- Retrieves top-K facts (default 10), optionally filtered by `users.name = question_id`

Env vars (DB):
- PGHOST / POSTGRES_HOST
- PGPORT / POSTGRES_PORT
- PGDATABASE / POSTGRES_DB
- PGUSER / POSTGRES_USER
- PGPASSWORD / POSTGRES_PASSWORD

Env vars (LLM):
- OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_COMPATIBLE_MODEL
- GEMINI_API_KEY, GEMINI_MODEL
- ANTHROPIC_API_KEY, ANTHROPIC_MODEL

Notes:
- This script is self-contained and does NOT import SDK helpers.
- It assumes m2_semantic uses 384-d embeddings (all-MiniLM-L6-v2).
"""

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import threading
import concurrent.futures

from loguru import logger
from dotenv import load_dotenv


def _safe_import(module_name: str, install_hint: str):
    try:
        return __import__(module_name, fromlist=["*"])
    except Exception as e:
        logger.error(f"Missing dependency '{module_name}'. {install_hint} ({e})")
        raise


def default_model_for_provider(provider: str) -> str:
    if provider == "openai":
        return os.getenv("OPENAI_COMPATIBLE_MODEL", "gpt-4o-mini")
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
    return "gpt-4o-mini"


# ----------------------------- Dataset Loading ----------------------------- #

@dataclass
class DataLoadingConfig:
    dataset_id: str
    data_file: Optional[str]
    split: str
    num_samples: int
    random_sampling: bool
    start_index: int
    load_all: bool
    question_types: Optional[List[str]]


def load_lme_dataset(cfg: DataLoadingConfig) -> List[Dict[str, Any]]:
    """Load LME-style dataset from HuggingFace datasets.

    Expects items to contain at least:
    - question_id: str
    - question: str
    - choices: List[str]
    - correct_choice_index: int
    - optional: question_type, haystack_sessions, haystack_session_ids
    """
    datasets = _safe_import(
        "datasets",
        install_hint="Install with: poetry add datasets or pip install datasets",
    )

    logger.info(
        f"Loading dataset: id={cfg.dataset_id} split={cfg.split} data_file={cfg.data_file or 'N/A'}"
    )
    if cfg.data_file:
        ds = datasets.load_dataset(cfg.dataset_id, data_files=cfg.data_file, split=cfg.split)
    else:
        ds = datasets.load_dataset(cfg.dataset_id, split=cfg.split)

    data: List[Dict[str, Any]] = list(ds)
    logger.info(f"Loaded {len(data)} records from HuggingFace")

    # Optional filter by question_types
    if cfg.question_types:
        wanted = set(cfg.question_types)
        before = len(data)
        data = [d for d in data if d.get("question_type") in wanted]
        logger.info(f"Filtered by question_types {sorted(wanted)}: {before} -> {len(data)}")

    # Sampling
    if not cfg.load_all and cfg.num_samples > 0 and len(data) > cfg.num_samples:
        if cfg.random_sampling:
            data = random.sample(data, cfg.num_samples)
        else:
            start = min(cfg.start_index, max(0, len(data) - 1))
            end = start + cfg.num_samples
            if end <= len(data):
                data = data[start:end]
            else:
                # wrap-around
                data = data[start:] + data[: end - len(data)]

    logger.info(f"Final dataset size: {len(data)}")
    return data


# --------------------------- Embedding + Database -------------------------- #

def get_db_config() -> Dict[str, Any]:
    # Defaults match simplified_memory_service
    db = {
        "host": "localhost",
        "port": 5432,
        "database": "memfuse",
        "user": "postgres",
        "password": "postgres",
    }
    env_mapping = {
        "PGHOST": "host",
        "PGPORT": "port",
        "PGDATABASE": "database",
        "PGUSER": "user",
        "PGPASSWORD": "password",
        "POSTGRES_HOST": "host",
        "POSTGRES_PORT": "port",
        "POSTGRES_DB": "database",
        "POSTGRES_USER": "user",
        "POSTGRES_PASSWORD": "password",
    }
    for env, key in env_mapping.items():
        v = os.getenv(env)
        if v is not None:
            db[key] = int(v) if key == "port" else v
    return db


def connect_db():
    psycopg2 = _safe_import(
        "psycopg2",
        install_hint="Install with: poetry add psycopg2-binary or pip install psycopg2-binary",
    )
    from psycopg2.extras import RealDictCursor  # type: ignore

    cfg = get_db_config()
    logger.info(
        f"DB: {cfg['user']}@{cfg['host']}:{cfg['port']}/{cfg['database']} (attempting connect)"
    )
    conn = psycopg2.connect(**cfg)
    conn.autocommit = True
    return conn, RealDictCursor


class Embeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        st = _safe_import(
            "sentence_transformers",
            install_hint=(
                "Install with: poetry add sentence-transformers or pip install sentence-transformers"
            ),
        )
        self.model = st.SentenceTransformer(model_name)
        self._lock = threading.Lock()

    def encode(self, text: str) -> List[float]:
        # Protect model inference in multi-threaded contexts
        with self._lock:
            v = self.model.encode([text], normalize_embeddings=False)[0]
        return v.tolist()


def query_top_facts(
    conn,
    cursor_factory,
    query_embedding: List[float],
    top_k: int = 10,
    user_name: Optional[str] = None,
    status: str = "active",
) -> List[Dict[str, Any]]:
    """Vector search top-K facts from m2_semantic by cosine distance.

    If user_name is provided, restrict results via join on users.name.
    Returns rows with: fact_id, text, confidence, similarity_score, distance, user_name
    """
    with conn.cursor(cursor_factory=cursor_factory) as cur:
        params: List[Any] = []
        if user_name:
            sql = (
                "SELECT m.fact_id, m.text, m.confidence, u.name as user_name, "
                "       (1.0 - (m.embedding <=> %s::vector) / 2.0) AS similarity_score, "
                "       (m.embedding <=> %s::vector) AS distance "
                "FROM m2_semantic m "
                "JOIN users u ON m.user_id::text = u.id "
                "WHERE m.status = %s AND u.name = %s "
                "ORDER BY m.embedding <=> %s::vector ASC "
                "LIMIT %s"
            )
            params = [query_embedding, query_embedding, status, user_name, query_embedding, top_k]
        else:
            sql = (
                "SELECT m.fact_id, m.text, m.confidence, "
                "       (1.0 - (m.embedding <=> %s::vector) / 2.0) AS similarity_score, "
                "       (m.embedding <=> %s::vector) AS distance "
                "FROM m2_semantic m "
                "WHERE m.status = %s "
                "ORDER BY m.embedding <=> %s::vector ASC "
                "LIMIT %s"
            )
            params = [query_embedding, query_embedding, status, query_embedding, top_k]

        cur.execute(sql, params)
        rows = cur.fetchall()
        return list(rows)


def query_top_chunks(
    conn,
    cursor_factory,
    query_embedding: List[float],
    top_k: int = 10,
    user_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Vector search top-K episodic chunks from m1_episodic by cosine distance.

    If user_name is provided, restrict via users.name and join on m1.user_id = users.id.
    Returns rows: chunk_id, content, similarity_score, distance
    """
    with conn.cursor(cursor_factory=cursor_factory) as cur:
        if user_name:
            sql = (
                "SELECT m.chunk_id, m.content, "
                "       (1.0 - (m.embedding <=> %s::vector) / 2.0) AS similarity_score, "
                "       (m.embedding <=> %s::vector) AS distance "
                "FROM m1_episodic m "
                "JOIN users u ON m.user_id = u.id "
                "WHERE u.name = %s "
                "ORDER BY m.embedding <=> %s::vector ASC "
                "LIMIT %s"
            )
            params = [query_embedding, query_embedding, user_name, query_embedding, top_k]
        else:
            sql = (
                "SELECT chunk_id, content, "
                "       (1.0 - (embedding <=> %s::vector) / 2.0) AS similarity_score, "
                "       (embedding <=> %s::vector) AS distance "
                "FROM m1_episodic "
                "ORDER BY embedding <=> %s::vector ASC "
                "LIMIT %s"
            )
            params = [query_embedding, query_embedding, query_embedding, top_k]

        cur.execute(sql, params)
        rows = cur.fetchall()
        return list(rows)


# ------------------------------ LLM Utilities ------------------------------ #

def build_system_prompt(n_choices: int) -> str:
    return (
        "You are given XML-structured context containing two retrieval sources: "
        "M2 semantic facts (<m2_facts>) and M1 episodic chunks (<m1_chunks>). "
        "Use M2 for concise factual grounding and M1 for conversational context and phrasing. "
        "Answer the multiple-choice question based ONLY on the provided context when possible. "
        "Respond strictly as JSON with keys index (0-" + str(n_choices - 1) + ") and reasoning. "
        "Example: {\"index\": 2, \"reasoning\": \"...\"}."
    )


def build_user_prompt(
    question: str,
    choices: List[str],
    m2_facts: List[Dict[str, Any]],
    m1_chunks: List[Dict[str, Any]],
) -> str:
    # Build XML context
    parts: List[str] = []
    parts.append("<retrieval>")

    parts.append("  <m2_facts>")
    for i, f in enumerate(m2_facts[:10], 1):
        text = f.get("text", "").strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) > 800:
            text = text[:800] + "..."
        sim = f.get("similarity_score")
        conf = f.get("confidence")
        attrs = []
        if sim is not None:
            attrs.append(f"sim=\"{sim:.3f}\"")
        if conf is not None:
            attrs.append(f"conf=\"{conf:.2f}\"")
        attrs_s = " " + " ".join(attrs) if attrs else ""
        # Escape XML special chars minimally
        safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        parts.append(f"    <fact index=\"{i}\"{attrs_s}>{safe_text}</fact>")
    parts.append("  </m2_facts>")

    parts.append("  <m1_chunks>")
    for i, c in enumerate(m1_chunks[:10], 1):
        text = c.get("content", "").strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) > 800:
            text = text[:800] + "..."
        sim = c.get("similarity_score")
        attrs = []
        if sim is not None:
            attrs.append(f"sim=\"{sim:.3f}\"")
        attrs_s = " " + " ".join(attrs) if attrs else ""
        safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        parts.append(f"    <chunk index=\"{i}\"{attrs_s}>{safe_text}</chunk>")
    parts.append("  </m1_chunks>")

    parts.append("</retrieval>")

    # Question and choices
    q = (question or "").strip()
    q_safe = q.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    parts.append("<question>")
    parts.append(f"  {q_safe}")
    parts.append("</question>")

    parts.append("<choices>")
    for i, c in enumerate(choices):
        c_text = (c or "").strip()
        c_safe = c_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        parts.append(f"  <choice index=\"{i}\">{c_safe}</choice>")
    parts.append("</choices>")

    parts.append("<instructions>")
    parts.append(
        "  Review <m2_facts> for extracted facts and <m1_chunks> for conversational details. "
        "Prioritize factual correctness. If evidence is insufficient, choose the most plausible answer and explain succinctly. "
        "Return only JSON with fields 'index' (0-based) and 'reasoning'. No other text."
    )
    parts.append("</instructions>")

    return "\n".join(parts)


def parse_json_choice(text: str, n_choices: int) -> Tuple[int, str]:
    """Extract {"index": int, "reasoning": str} from LLM output robustly."""
    # Try to find JSON block
    block = None
    code_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if code_match:
        block = code_match.group(1)
    else:
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            block = brace_match.group(0)
    if block is None:
        # Fallback: try to detect index with regex
        idx_match = re.search(r"index\s*[:=]\s*(\d+)", text, re.IGNORECASE)
        idx = int(idx_match.group(1)) if idx_match else 0
        idx = max(0, min(n_choices - 1, idx))
        return idx, text.strip()
    try:
        data = json.loads(block)
        idx = int(data.get("index", 0))
        idx = max(0, min(n_choices - 1, idx))
        reasoning = str(data.get("reasoning", ""))
        return idx, reasoning
    except Exception:
        # Last resort parsing
        idx_match = re.search(r"\b(\d+)\b", text)
        idx = int(idx_match.group(1)) if idx_match else 0
        idx = max(0, min(n_choices - 1, idx))
        return idx, text.strip()


def call_llm(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 5,
) -> str:
    provider = (provider or "openai").lower()
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            if attempt:
                time.sleep(min(2 ** (attempt - 1), 10))
            if provider == "openai":
                return _call_openai(model, system_prompt, user_prompt)
            if provider == "gemini":
                return _call_gemini(model, system_prompt, user_prompt)
            if provider == "anthropic":
                return _call_anthropic(model, system_prompt, user_prompt)
            raise ValueError(f"Unsupported provider: {provider}")
        except Exception as e:
            last_err = e
            logger.warning(f"LLM call failed (attempt {attempt+1}/{max_retries+1}): {e}")
    assert last_err
    raise last_err


def _call_openai(model: str, system_prompt: str, user_prompt: str) -> str:
    # Support modern openai client with base_url override
    try:
        from openai import OpenAI  # type: ignore

        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("MEMFUSE_LLM_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""
    except ImportError:
        # Legacy fallback
        openai = _safe_import(
            "openai",
            install_hint="Install with: poetry add openai or pip install openai",
        )
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("MEMFUSE_LLM_BASE_URL")
        openai.api_key = api_key
        if base_url:
            openai.base_url = base_url  # type: ignore
        resp = openai.ChatCompletion.create(  # type: ignore
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        return resp["choices"][0]["message"]["content"]


def _call_gemini(model: str, system_prompt: str, user_prompt: str) -> str:
    # Prefer google.generativeai; fallback to google.genai if present in env
    try:
        import google.generativeai as genai  # type: ignore

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel(model)
        contents = f"System: {system_prompt}\n\nUser: {user_prompt}"
        resp = m.generate_content(contents)
        return getattr(resp, "text", "") or ""  # type: ignore
    except ImportError:
        # Newer google.genai client
        try:
            from google.genai import Client  # type: ignore

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            client = Client(api_key=api_key)
            contents = f"System: {system_prompt}\n\nUser: {user_prompt}"
            resp = client.models.generate_content(model=model, contents=contents)
            text = ""
            if getattr(resp, "candidates", None):  # type: ignore
                cand = resp.candidates[0]
                if getattr(cand, "content", None) and getattr(cand.content, "parts", None):  # type: ignore
                    for part in cand.content.parts:
                        if getattr(part, "text", None):
                            text += part.text
            return text
        except ImportError:
            raise RuntimeError(
                "Missing google generative ai client. Install 'google-generativeai' or 'google-genai'."
            )


def _call_anthropic(model: str, system_prompt: str, user_prompt: str) -> str:
    anthropic = _safe_import(
        "anthropic",
        install_hint="Install with: poetry add anthropic or pip install anthropic",
    )
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        system=system_prompt,
        max_tokens=512,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0,
    )
    out = ""
    for part in resp.content:
        if getattr(part, "text", None):
            out += part.text
    return out


# --------------------------------- Runner --------------------------------- #

def run_single_question(
    item: Dict[str, Any],
    emb: Embeddings,
    conn,
    cursor_factory,
    provider: str,
    model: str,
    top_k: int,
    top_k_m1: int,
    enable_m2: bool,
    enable_m1: bool,
    filter_user_by_question_id: bool,
    show_facts: bool,
) -> Dict[str, Any]:
    qid = item.get("question_id", "unknown")
    qtext = item.get("question")
    choices: List[str] = item.get("choices") or []
    correct_idx = item.get("correct_choice_index")

    if not qtext or not choices or correct_idx is None:
        return {
            "question_id": qid,
            "status": "SKIPPED - Missing fields",
        }

    # Build query embedding and retrieve facts
    query_vec = emb.encode(qtext)
    user_name = qid if filter_user_by_question_id else None
    facts: List[Dict[str, Any]] = []
    if enable_m2 and conn and cursor_factory:
        try:
            facts = query_top_facts(conn, cursor_factory, query_vec, top_k=top_k, user_name=user_name)
        except Exception as e:
            logger.warning(f"Vector search failed for {qid}: {e}. Proceeding without facts.")

    # Retrieve episodic chunks (M1)
    chunks: List[Dict[str, Any]] = []
    if enable_m1 and conn and cursor_factory:
        try:
            chunks = query_top_chunks(conn, cursor_factory, query_vec, top_k=top_k_m1, user_name=user_name)
        except Exception as e:
            logger.warning(f"M1 chunk search failed for {qid}: {e}. Proceeding without chunks.")

    if show_facts:
        if enable_m2:
            logger.info(f"Top {min(top_k, len(facts))} facts (M2) for {qid}:")
            for i, f in enumerate(facts[:top_k], 1):
                _text = f.get('text', '')
                snippet = (_text[:120]).replace(chr(10), ' ')
                sim = f.get('similarity_score', 0)
                conf = f.get('confidence', 0)
                logger.info(f"  F{i:>2}. sim={sim:.3f} conf={conf:.2f} | {snippet}")
        else:
            logger.info("M2 retrieval disabled (--no-m2)")

        if enable_m1:
            logger.info(f"Top {min(top_k_m1, len(chunks))} chunks (M1) for {qid}:")
            for i, c in enumerate(chunks[: top_k_m1], 1):
                _text = c.get('content', '')
                snippet = (_text[:120]).replace(chr(10), ' ')
                sim = c.get('similarity_score', 0)
                logger.info(f"  C{i:>2}. sim={sim:.3f} | {snippet}")
        else:
            logger.info("M1 retrieval disabled (--no-m1)")

    system_prompt = build_system_prompt(len(choices))
    user_prompt = build_user_prompt(qtext, choices, facts, chunks)

    raw = call_llm(provider=provider, model=model, system_prompt=system_prompt, user_prompt=user_prompt)
    pred_idx, reasoning = parse_json_choice(raw or "", len(choices))
    is_correct = int(pred_idx) == int(correct_idx)

    # Log question and answers
    try:
        correct_text = choices[correct_idx] if 0 <= int(correct_idx) < len(choices) else ""
    except Exception:
        correct_text = ""
    try:
        model_text = choices[pred_idx] if 0 <= int(pred_idx) < len(choices) else ""
    except Exception:
        model_text = ""

    logger.info("Question: " + (qtext or "").strip())
    logger.info(f"Correct Answer [{correct_idx}]: {correct_text}")
    logger.info(f"Model Answer   [{pred_idx}]: {model_text}")

    return {
        "question_id": qid,
        "question_text": qtext,
        "model_choice_idx": pred_idx,
        "model_choice_text": choices[pred_idx] if 0 <= pred_idx < len(choices) else "",
        "correct_choice_idx": correct_idx,
        "correct_choice_text": choices[correct_idx] if 0 <= correct_idx < len(choices) else "",
        "is_correct": is_correct,
        "explanation": reasoning,
        "facts_used": len(facts),
        "chunks_used": len(chunks),
    }


def main():
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(
        description="Run LME MCQ benchmark with m2_semantic vector search (no SDK)."
    )
    # Dataset
    parser.add_argument(
        "--dataset",
        default="lme",
        choices=["msc", "lme", "locomo"],
        help="Dataset key to use defaults from scripts/config_sdk.py (default: lme)",
    )
    parser.add_argument(
        "--dataset-id",
        default=os.getenv("LME_DATASET_ID"),
        help="Override HuggingFace dataset ID. If omitted, uses scripts/config_sdk.py by --dataset key.",
    )
    parser.add_argument(
        "--data-file",
        default=os.getenv("LME_DATA_FILE"),
        help="Optional `data_files` entry for load_dataset (e.g., data/lme_s_mc10.json). If omitted, uses scripts/config_sdk.py by --dataset key.",
    )
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    parser.add_argument("--num-questions", type=int, default=10)
    parser.add_argument("--random", action="store_true", help="Random sample questions")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--load-all", action="store_true")
    parser.add_argument(
        "--question-types",
        nargs="+",
        choices=[
            "multi-session",
            "temporal-reasoning",
            "knowledge-update",
            "single-session-user",
            "single-session-assistant",
            "single-session-preference",
        ],
        help="Filter by LME question types",
    )
    # Retrieval / LLM
    parser.add_argument("--top-k", type=int, default=10, help="Top-K facts from m2_semantic")
    parser.add_argument("--top-k-m1", type=int, default=10, help="Top-K chunks from m1_episodic")
    parser.add_argument("--no-m1", action="store_true", help="Disable M1 episodic retrieval")
    parser.add_argument("--no-m2", action="store_true", help="Disable M2 semantic retrieval")
    parser.add_argument(
        "--filter-user-by-question-id",
        action="store_true",
        help="Join users to restrict facts: users.name == question_id",
    )
    parser.add_argument("--show-facts", action="store_true", help="Log retrieved facts")
    parser.add_argument(
        "--llm-provider",
        default=os.getenv("LLM_PROVIDER", "openai"),
        choices=["openai", "gemini", "anthropic"],
    )
    parser.add_argument("--model", default=None, help="Model name; defaults based on provider")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of questions to process in parallel (default: 10)")

    args = parser.parse_args()

    # Resolve model
    model = args.model or default_model_for_provider(args.llm_provider)
    logger.info(f"LLM provider={args.llm_provider} model={model}")

    # Resolve dataset id/file from scripts/config_sdk.py when not explicitly provided
    ds_id = args.dataset_id
    ds_file = args.data_file
    try:
        import config_sdk  # type: ignore

        cfg_map = getattr(config_sdk, "DATASET_CONFIGS", {})
        if args.dataset in cfg_map:
            if not ds_id:
                ds_id = cfg_map[args.dataset].get("dataset_id")
            if not ds_file:
                ds_file = cfg_map[args.dataset].get("data_file")
    except Exception as e:
        logger.warning(f"Could not import scripts/config_sdk.py for dataset defaults: {e}")

    if not ds_id:
        # Fallback hardcoded IDs if config import failed
        fallback = {
            "msc": ("Percena/msc-memfuse-mc10", "data/msc_memfuse_mc10.json"),
            "lme": ("Percena/lme-mc10", "data/lme_s_mc10.json"),
            "locomo": ("Percena/locomo-mc10", "data/locomo_mc10.json"),
        }
        ds_id, ds_file = fallback.get(args.dataset, (None, None))

    logger.info(f"Dataset key={args.dataset} dataset_id={ds_id} data_file={ds_file}")

    # Load dataset
    cfg = DataLoadingConfig(
        dataset_id=ds_id,
        data_file=ds_file,
        split=args.split,
        num_samples=args.num_questions,
        random_sampling=args.random,
        start_index=args.start_index,
        load_all=args.load_all,
        question_types=args.question_types,
    )
    try:
        dataset = load_lme_dataset(cfg)
    except Exception as e:
        logger.error(f"Failed to load dataset from HuggingFace: {e}.")
        logger.error(
            "Provide --dataset-id/--data-file or ensure scripts/config_sdk.py has proper DATASET_CONFIGS."
        )
        sys.exit(1)
    if not dataset:
        logger.error("No data loaded. Exiting.")
        sys.exit(1)

    # Connect DB and prepare embeddings
    try:
        conn, cursor_factory = connect_db()
    except Exception as e:
        logger.warning(f"DB connection failed: {e}. Will proceed without retrieval.")
        conn, cursor_factory = None, None

    try:
        emb = Embeddings()
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {e}")
        sys.exit(1)

    # Helper for thread-local DB connections
    _thread_local = threading.local()

    def get_thread_conn():
        if getattr(_thread_local, "conn", None) is None:
            try:
                _conn, _cursor_factory = connect_db()
            except Exception as e:
                logger.warning(f"Thread DB connect failed: {e}")
                _thread_local.conn = None
                _thread_local.cursor_factory = None
            else:
                _thread_local.conn = _conn
                _thread_local.cursor_factory = _cursor_factory
        return getattr(_thread_local, "conn", None), getattr(_thread_local, "cursor_factory", None)

    def process_item(i_and_item: Tuple[int, Dict[str, Any]]):
        i, item = i_and_item
        logger.info(f"=== Processing Question {i}/{len(dataset)} ===")
        t0 = time.perf_counter()
        # Acquire thread-local DB connection if any retrieval is enabled
        _conn = _cursor_factory = None
        if not args.no_m1 or not args.no_m2:
            _conn, _cursor_factory = get_thread_conn()
        try:
            res = run_single_question(
                item=item,
                emb=emb,
                conn=_conn,
                cursor_factory=_cursor_factory,
                provider=args.llm_provider,
                model=model,
                top_k=args.top_k,
                top_k_m1=args.top_k_m1,
                enable_m2=(not args.no_m2),
                enable_m1=(not args.no_m1),
                filter_user_by_question_id=args.filter_user_by_question_id,
                show_facts=args.show_facts,
            )
            dt = (time.perf_counter() - t0) * 1000
            res["elapsed_ms"] = dt
            return res
        except Exception as e:
            logger.error(f"Failed question {i}: {e}")
            return {"question_id": item.get("question_id", f"q{i}"), "status": f"FAILED: {e}"}

    # Run sequentially if low concurrency requested
    results: List[Dict[str, Any]] = []
    if (args.concurrency or 1) <= 1:
        correct = 0
        for i, item in enumerate(dataset, 1):
            res = process_item((i, item))
            if res.get("is_correct"):
                correct += 1
            results.append(res)
    else:
        # Use a ThreadPool for IO-bound LLM/API + DB
        correct = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(process_item, (i, item)) for i, item in enumerate(dataset, 1)]
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                if res.get("is_correct"):
                    correct += 1
                results.append(res)

    # Summary
    total = len(results)
    acc = (correct / total) * 100 if total > 0 else 0.0
    logger.info("=== Summary ===")
    logger.info(f"Total: {total} | Correct: {correct} | Accuracy: {acc:.1f}%")

    # Optional: write a results file next to this script
    try:
        out_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"lme_results_{int(time.time())}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Saved per-question results to: {out_path}")
    except Exception as e:
        logger.warning(f"Failed to write results file: {e}")


if __name__ == "__main__":
    main()
