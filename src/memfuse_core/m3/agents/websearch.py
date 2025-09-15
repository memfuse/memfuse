"""WebSearchAgent implementation for M3."""

from __future__ import annotations

import asyncio
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger


class WebSearchAgent:
    """Agent that performs web searches using multiple sources."""
    
    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def _close_session(self):
        """Close HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def _duckduckgo_search(self, query: str) -> Dict[str, Any]:
        """Perform DuckDuckGo search."""
        try:
            session = await self._get_session()
            params = {
                "q": query,
                "format": "json", 
                "no_redirect": 1,
                "no_html": 1
            }
            
            async with session.get("https://api.duckduckgo.com/", params=params) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
                
            # Extract results
            results = []
            
            # Process RelatedTopics
            def _walk_topics(topics):
                for topic in topics or []:
                    if isinstance(topic, dict):
                        if "Text" in topic and "FirstURL" in topic:
                            yield {
                                "title": str(topic.get("Text", ""))[:200],
                                "url": str(topic.get("FirstURL", ""))[:400], 
                                "snippet": str(topic.get("Result") or topic.get("Text", ""))[:400]
                            }
                        elif "Topics" in topic:
                            yield from _walk_topics(topic.get("Topics", []))
            
            results.extend(_walk_topics(data.get("RelatedTopics", [])))
            
            # Add abstract if available
            abstract = str(data.get("AbstractText", "")).strip()
            if abstract:
                results.append({
                    "title": str(data.get("Heading", ""))[:200],
                    "url": str(data.get("AbstractURL", ""))[:400],
                    "snippet": abstract[:400]
                })
            
            # Add answer if available
            answer = str(data.get("Answer", "")).strip()
            if answer:
                results.append({
                    "title": "Answer",
                    "url": "",
                    "snippet": answer[:400]
                })
            
            return {
                "engine": "duckduckgo",
                "results": results[:10],  # Limit results
                "total": len(results)
            }
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return {"engine": "duckduckgo", "error": str(e), "results": []}
    
    async def _arxiv_search(
        self, 
        query: str, 
        max_results: int = 10, 
        last_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform arXiv search."""
        try:
            session = await self._get_session()
            
            # Build search query - if query looks like natural language, 
            # enhance it for arXiv search
            if not any(op in query.lower() for op in ['and', 'or', 'all:', 'ti:', 'au:']):
                # Enhance natural language queries for better arXiv results
                if any(keyword in query.lower() for keyword in [
                    'memory', 'llm', 'language model', 'agent', 'rag', 'retrieval'
                ]):
                    arxiv_query = (
                        f'all:("{query}") AND '
                        'all:(memory OR "language model" OR LLM OR agent OR retrieval OR RAG)'
                    )
                else:
                    arxiv_query = f'all:("{query}")'
            else:
                arxiv_query = query
            
            params = {
                "search_query": arxiv_query,
                "start": 0,
                "max_results": max_results * 2 if last_days else max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending"
            }
            
            async with session.get("http://export.arxiv.org/api/query", params=params) as resp:
                resp.raise_for_status()
                xml_content = await resp.text()
            
            # Parse XML
            root = ET.fromstring(xml_content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            
            entries = []
            cutoff = None
            if last_days and last_days > 0:
                cutoff = datetime.now(timezone.utc) - timedelta(days=last_days)
            
            for entry in root.findall("atom:entry", ns):
                title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
                summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
                published_raw = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
                
                # Parse publication date
                pub_dt = None
                if published_raw:
                    try:
                        pub_dt = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
                    except Exception:
                        pass
                
                # Filter by date if specified
                if cutoff and pub_dt and pub_dt < cutoff:
                    continue
                
                # Get arXiv ID and construct URL
                entry_id = entry.findtext("atom:id", default="", namespaces=ns)
                arxiv_url = entry_id if entry_id.startswith("http") else f"https://arxiv.org/abs/{entry_id.split('/')[-1]}"
                
                entries.append({
                    "title": title,
                    "url": arxiv_url,
                    "snippet": summary[:400],
                    "published": published_raw,
                    "source": "arXiv"
                })
                
                if len(entries) >= max_results:
                    break
            
            return {
                "engine": "arxiv",
                "results": entries,
                "total": len(entries),
                "query_used": arxiv_query
            }
            
        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            return {"engine": "arxiv", "error": str(e), "results": []}
    
    async def execute(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web search with multiple sources."""
        query = str(payload.get("query", payload.get("q", ""))).strip()
        if not query:
            return {"error": "query required"}
        
        sources = payload.get("sources", ["duckduckgo", "arxiv"])
        max_results = int(payload.get("max_results", payload.get("max", 10)))
        last_days = payload.get("last_days")
        
        try:
            last_days = int(last_days) if last_days is not None else None
        except (ValueError, TypeError):
            last_days = None
        
        logger.info(f"WebSearchAgent executing search: query='{query}', sources={sources}")
        
        results = {}
        tasks = []
        
        # Create search tasks
        if "duckduckgo" in sources:
            tasks.append(("duckduckgo", self._duckduckgo_search(query)))
        
        if "arxiv" in sources:
            # Use custom arXiv query if provided, otherwise use the main query
            arxiv_query = payload.get("arxiv_query", query)
            tasks.append(("arxiv", self._arxiv_search(arxiv_query, max_results, last_days)))
        
        # Execute searches concurrently
        try:
            if tasks:
                search_results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
                
                for i, (source, result) in enumerate(zip([t[0] for t in tasks], search_results)):
                    if isinstance(result, Exception):
                        results[source] = {"engine": source, "error": str(result), "results": []}
                    else:
                        results[source] = result
            
            # Aggregate results
            all_results = []
            total_found = 0
            
            for source_results in results.values():
                if "results" in source_results:
                    all_results.extend(source_results["results"])
                    total_found += source_results.get("total", 0)
            
            return {
                "results": all_results[:max_results],
                "total_found": total_found,
                "sources_used": list(results.keys()),
                "source_details": results,
                "query": query
            }
            
        finally:
            # Clean up session
            await self._close_session()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._close_session()