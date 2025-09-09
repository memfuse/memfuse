"""Inbound and outbound filters for Gateway.

These are lightweight, synchronous filters to normalize/validate requests and
responses around the gateway pipeline. They are intentionally minimal to avoid
blocking async flows; heavy logic belongs to Guardrail or processors.
"""
from typing import Any, Dict, Protocol, Tuple, List
import re
from ..interfaces.gateway_interface import RequestContext
from ..utils.global_config_manager import get_global_config_manager


class InboundFilter(Protocol):
    def apply(self, request_data: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        ...


class ConfigRequestNormalizerInboundFilter:
    """Normalize incoming request fields using gateway.request_normalizer config.

    - Coerce query to string (fallback to empty string)
    - Clamp top_k into [min_top_k, max_top_k] with default_top_k when missing/invalid
    """

    def __init__(self) -> None:
        gcm = get_global_config_manager()
        cfg = gcm.get_section("gateway") if gcm.is_initialized() else {}
        rn = (cfg or {}).get("request_normalizer", {}) or {}
        self.enabled = bool(rn.get("enabled", False))
        self.min_top_k = int(rn.get("min_top_k", 1))
        self.max_top_k = int(rn.get("max_top_k", 50))
        self.default_top_k = int(rn.get("default_top_k", 5))

    def apply(self, request_data: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        if not self.enabled:
            return request_data
        # normalize query
        q = request_data.get("query", "")
        if not isinstance(q, str):
            q = "" if q is None else str(q)
        request_data["query"] = q
        # normalize top_k
        tk = request_data.get("top_k", self.default_top_k)
        try:
            tk = int(tk)
        except Exception:
            tk = self.default_top_k
        if tk < self.min_top_k:
            tk = self.min_top_k
        if tk > self.max_top_k:
            tk = self.max_top_k
        request_data["top_k"] = tk
        return request_data


class OutboundFilter(Protocol):
    def apply(self, response: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        ...


class NoOpInboundFilter:
    """No-op inbound filter (placeholder)."""
    def apply(self, request_data: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        return request_data


class NoOpOutboundFilter:
    """No-op outbound filter (placeholder)."""
    def apply(self, response: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        return response


class ConfigOutputRemovalFilter:
    """Remove configured fields from results (uses guardrail.output config)."""

    def __init__(self):
        gcm = get_global_config_manager()
        cfg = gcm.get_section("guardrail") if gcm.is_initialized() else {}
        output_cfg = cfg.get("output", cfg.get("output_filter", {})) or {}
        self.enabled = bool(output_cfg.get("enabled", False))
        self.fields = list(output_cfg.get("remove_fields", []) or [])

    def _remove_path(self, obj: Dict[str, Any], dotted: str) -> None:
        parts = dotted.split('.') if dotted else []
        if not parts:
            return

        def rec(cur: Any, idx: int) -> None:
            if idx >= len(parts) or cur is None:
                return
            key = parts[idx]
            is_last = (idx == len(parts) - 1)

            if isinstance(cur, dict):
                if key not in cur:
                    return
                if is_last:
                    try:
                        del cur[key]
                    except Exception:
                        pass
                else:
                    rec(cur.get(key), idx + 1)
            elif isinstance(cur, list):
                # list index, wildcard, or conditional wildcard like *{k=B}
                if key == "*" or (key.startswith("*{") and key.endswith("}")):
                    cond_key = None
                    cond_val = None
                    if key != "*":
                        inner = key[2:-1]  # inside {...}
                        if "=" in inner:
                            cond_key, cond_val = inner.split("=", 1)
                            cond_key = cond_key.strip()
                            cond_val = cond_val.strip().strip("'\"")
                    for item in cur:
                        if cond_key is not None:
                            if not isinstance(item, dict) or str(item.get(cond_key)) != cond_val:
                                continue
                        rec(item, idx + 1)
                else:
                    try:
                        i = int(key)
                    except Exception:
                        # unsupported selector; skip
                        return
                    if 0 <= i < len(cur):
                        rec(cur[i], idx + 1)
            else:
                return

        rec(obj, 0)

    def apply(self, response: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        if not self.enabled or not self.fields:
            return response
        data = response.get("data", {})
        results = data.get("results", [])
        for r in results:
            for f in self.fields:
                self._remove_path(r, f)
        return response


class ConfigPIIRedactorFilter:
    """Redact basic PII like emails/phones when guardrail.pii is enabled."""

    EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    PHONE_RE = re.compile(r"\+?(?:\d[\s-]?){10,}")

    def __init__(self):
        gcm = get_global_config_manager()
        cfg = gcm.get_section("guardrail") if gcm.is_initialized() else {}
        pii_cfg = cfg.get("pii", {}) or {}
        self.enabled = bool(pii_cfg.get("enabled", False))
        self.redact = bool(pii_cfg.get("redact", True))

    def _redact_text(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        text = self.EMAIL_RE.sub("[REDACTED:EMAIL]", text)
        text = self.PHONE_RE.sub("[REDACTED:PHONE]", text)
        return text

    def _redact_dict(self, d: Dict[str, Any]) -> None:
        for k, v in list(d.items()):
            if isinstance(v, str):
                d[k] = self._redact_text(v)
            elif isinstance(v, dict):
                self._redact_dict(v)

    def apply(self, response: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        if not self.enabled or not self.redact:
            return response
        data = response.get("data", {})
        results = data.get("results", [])
        for r in results:
            if "content" in r and isinstance(r["content"], str):
                r["content"] = self._redact_text(r["content"])  # redact in content
            if "metadata" in r and isinstance(r["metadata"], dict):
                self._redact_dict(r["metadata"])  # redact in metadata strings
        return response


class ConfigMaxContentLengthFilter:
    """Clamp/limit content field length when guardrail.length is enabled.

    Config: guardrail.length
      enabled: bool
      max_content_length: int
      suffix: string (default '...') appended when truncated
    """

    def __init__(self):
        gcm = get_global_config_manager()
        cfg = gcm.get_section("guardrail") if gcm.is_initialized() else {}
        length_cfg = cfg.get("length", {}) or {}
        self.enabled = bool(length_cfg.get("enabled", False))
        self.max_len = int(length_cfg.get("max_content_length", 0) or 0)
        self.suffix = str(length_cfg.get("suffix", "..."))

    def apply(self, response: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        if not self.enabled or self.max_len <= 0:
            return response
        data = response.get("data", {})
        results = data.get("results", [])
        for r in results:
            c = r.get("content")
            if isinstance(c, str) and len(c) > self.max_len:
                truncated = c[: self.max_len] + self.suffix
                r["content"] = truncated
                md = r.setdefault("metadata", {})
                if isinstance(md, dict):
                    md["length_truncated"] = True
        return response


class ConfigSensitiveWordFilter:
    """Mask sensitive words in content when guardrail.sensitive is enabled.

    Config: guardrail.sensitive
      enabled: bool
      words: list[str]
      mask_token: string (default '[SENSITIVE]')
      case_insensitive: bool (default True)
      recurse_metadata: bool (default False) — if True, also process metadata string values
      action: string (default 'mask') — 'mask', 'flag', or 'drop'
    """

    def __init__(self):
        gcm = get_global_config_manager()
        cfg = gcm.get_section("guardrail") if gcm.is_initialized() else {}
        s_cfg = cfg.get("sensitive", cfg.get("sensitive_words", {})) or {}
        self.enabled = bool(s_cfg.get("enabled", False))
        self.words = [w for w in (s_cfg.get("words") or []) if isinstance(w, str)]
        self.mask_token = str(s_cfg.get("mask_token", "[SENSITIVE]"))
        self.case_insensitive = bool(s_cfg.get("case_insensitive", True))
        self.recurse_metadata = bool(s_cfg.get("recurse_metadata", False))
        self.action = str(s_cfg.get("action", "mask")).lower()
        if self.action not in ("mask", "flag", "drop"):
            self.action = "mask"  # fallback to safe default

    def apply(self, response: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        if not self.enabled or not self.words:
            return response
        data = response.get("data", {})
        results = data.get("results", [])
        results_to_remove = []

        for i, r in enumerate(results):
            hit_detected = False

            # Process content field
            c = r.get("content")
            if isinstance(c, str) and c:
                new_c, content_hit = self._process_text(c)
                if content_hit:
                    hit_detected = True
                    if self.action == "mask":
                        r["content"] = new_c
                    elif self.action == "drop":
                        results_to_remove.append(i)
                        continue

            # Process metadata strings if enabled
            if self.recurse_metadata and isinstance(r.get("metadata"), dict):
                metadata_hit = self._process_metadata_recursive(r["metadata"])
                if metadata_hit:
                    hit_detected = True

            # Mark hit in metadata
            if hit_detected:
                md = r.setdefault("metadata", {})
                if isinstance(md, dict):
                    md["sensitive_hit"] = True

        # Remove flagged results (in reverse order to maintain indices)
        if self.action == "drop":
            for i in reversed(results_to_remove):
                results.pop(i)

        return response

    def _process_text(self, text: str) -> tuple[str, bool]:
        """Process text for sensitive words. Returns (processed_text, hit_detected)."""
        if not text:
            return text, False

        new_text = text
        hit_detected = False

        # Use cached regex patterns for better performance
        from .filter_cache import get_regex_cache
        regex_cache = get_regex_cache()

        for w in self.words:
            if not w:
                continue

            # Get cached compiled pattern
            pattern = regex_cache.get_word_pattern(w, self.case_insensitive)
            if pattern.search(new_text):
                hit_detected = True
                if self.action == "mask":
                    new_text = pattern.sub(self.mask_token, new_text)

        return new_text, hit_detected

    def _process_metadata_recursive(self, metadata: Dict[str, Any]) -> bool:
        """Recursively process metadata dict for sensitive words. Returns hit_detected."""
        hit_detected = False

        for key, value in metadata.items():
            if isinstance(value, str) and value:
                new_value, value_hit = self._process_text(value)
                if value_hit:
                    hit_detected = True
                    if self.action == "mask":
                        metadata[key] = new_value
            elif isinstance(value, dict):
                nested_hit = self._process_metadata_recursive(value)
                if nested_hit:
                    hit_detected = True
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, str) and item:
                        new_item, item_hit = self._process_text(item)
                        if item_hit:
                            hit_detected = True
                            if self.action == "mask":
                                value[i] = new_item
                    elif isinstance(item, dict):
                        nested_hit = self._process_metadata_recursive(item)
                        if nested_hit:
                            hit_detected = True

        return hit_detected


class ConfigToxicityAnnotatorFilter:
    """Annotate responses with a toxicity flag using simple keyword heuristic.

    Note: minimal heuristic for unit tests; not a production classifier.
    """

    KEYWORDS = {"toxic", "abuse", "insult"}

    def __init__(self):
        gcm = get_global_config_manager()
        cfg = gcm.get_section("guardrail") if gcm.is_initialized() else {}
        tox_cfg = cfg.get("toxicity", {}) or {}
        self.enabled = bool(tox_cfg.get("enabled", False))
        self.threshold = float(tox_cfg.get("threshold", 0.9))

    def _score_text(self, text: str) -> float:
        if not isinstance(text, str) or not text:
            return 0.0
        lower = text.lower()
        hits = sum(1 for k in self.KEYWORDS if k in lower)
        return min(1.0, hits / 1.0)  # any hit -> 1.0, else 0.0

    def apply(self, response: Dict[str, Any], context: RequestContext) -> Dict[str, Any]:
        if not self.enabled:
            return response
        results = response.get("data", {}).get("results", [])
        for r in results:
            score = 0.0
            if "content" in r:
                score = max(score, self._score_text(r.get("content")))
            # mark if score >= (1 - (1-threshold)) => score >= threshold (simple)
            if score >= self.threshold:
                md = r.setdefault("metadata", {})
                md["toxicity_flag"] = True
                md["toxicity_score"] = score
        return response


def build_filters_from_config(cfg: Dict[str, Any] | None) -> Tuple[List[InboundFilter], List[OutboundFilter]]:
    """Build inbound/outbound filter instances from gateway pipeline config.

    Expected cfg structure:
    {
      "pipeline": {
        "inbound": [{"name": "noop_inbound", "enabled": true}],
        "outbound": [{"name": "noop_outbound", "enabled": false}]
      }
    }
    """
    inbound: List[InboundFilter] = []
    outbound: List[OutboundFilter] = []
    if not cfg:
        return inbound, outbound

    pipeline = cfg.get("pipeline") if isinstance(cfg, dict) else None
    if not isinstance(pipeline, dict):
        return inbound, outbound

    # Registry of available filters
    inbound_registry: Dict[str, type] = {
        "noop_inbound": NoOpInboundFilter,
        "request_normalizer": ConfigRequestNormalizerInboundFilter,
    }
    outbound_registry: Dict[str, type] = {
        "noop_outbound": NoOpOutboundFilter,
        "output_remove": ConfigOutputRemovalFilter,
        "pii_redact": ConfigPIIRedactorFilter,
        "toxicity_mark": ConfigToxicityAnnotatorFilter,
        "max_length": ConfigMaxContentLengthFilter,
        "sensitive_word": ConfigSensitiveWordFilter,
        "sensitive_words": ConfigSensitiveWordFilter,
    }

    # Import composite filters dynamically to avoid circular imports
    try:
        from .composite_filters import CompositeContentFilter, ContentQualityFilter
        outbound_registry["composite_content"] = CompositeContentFilter
        outbound_registry["content_quality"] = ContentQualityFilter
    except ImportError:
        pass

    # Inbound
    for item in pipeline.get("inbound", []) or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        enabled = item.get("enabled", True)
        if not enabled or not name:
            continue
        cls = inbound_registry.get(name)
        if cls:
            inbound.append(cls())

    # Outbound
    for item in pipeline.get("outbound", []) or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        enabled = item.get("enabled", True)
        if not enabled or not name:
            continue
        cls = outbound_registry.get(name)
        if cls:
            outbound.append(cls())

    return inbound, outbound

