from __future__ import annotations

"""WebSearchAgent implementation using DuckDuckGo Instant Answer API.

No API key required. Designed to be test-friendly by allowing monkeypatching of _fetch_json.
"""

import asyncio
from typing import Any, Dict, List, Optional

import aiohttp


class WebSearchAgent:
    def __init__(self, timeout: float = 6.0) -> None:
        self.base_url = "https://api.duckduckgo.com/"
        self.timeout = timeout

    async def _fetch_json(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                return await resp.json(content_type=None)

    def _extract_results(self, data: Dict[str, Any], max_results: int) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        if not isinstance(data, dict):
            return results
        # Prefer RelatedTopics entries (which may nest)
        def _walk(items):
            for it in items or []:
                if isinstance(it, dict) and "Text" in it and "FirstURL" in it:
                    yield {
                        "title": str(it.get("Text") or "")[:200],
                        "url": str(it.get("FirstURL") or "")[:400],
                        "snippet": str(it.get("Result") or it.get("Text") or "")[:400],
                    }
                elif isinstance(it, dict) and "Topics" in it:
                    for sub in _walk(it.get("Topics") or []):
                        yield sub

        for r in _walk(data.get("RelatedTopics") or []):
            results.append(r)
            if len(results) >= max_results:
                return results

        # Fallbacks
        abs_text = str(data.get("AbstractText") or "").strip()
        if abs_text:
            results.append({
                "title": str(data.get("Heading") or "")[:200],
                "url": str(data.get("AbstractURL") or "")[:400],
                "snippet": abs_text[:400],
            })
        ans = str(data.get("Answer") or "").strip()
        if ans:
            results.append({"title": "Answer", "url": "", "snippet": ans[:400]})
        return results[:max_results]

    async def execute(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = str(payload.get("query") or payload.get("q") or "").strip()
        if not query:
            return {"error": "query required"}
        try:
            max_results = int(payload.get("max_results", 5))
        except Exception:
            max_results = 5
        params = {"q": query, "format": "json", "no_redirect": 1, "no_html": 1}
        try:
            data = await self._fetch_json(self.base_url, params=params)
            results = self._extract_results(data, max_results=max_results)
            return {"results": results, "provider": "duckduckgo"}
        except Exception as e:
            return {"error": f"web search failed: {e}"}

