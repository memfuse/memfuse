"""RAG (Retrieval-Augmented Generation) Service."""

import logging
from typing import List, Dict, Any, Optional

from ..services.database_service import DatabaseService
from ..services.buffer_service import BufferService
from ..services.service_factory import ServiceFactory

logger = logging.getLogger(__name__)


class RAGService:
    """RAG service for retrieval-augmented generation.
    
    This implementation provides compatibility with both memfuse_mvp style
    and the current MemFuse core architecture.
    """
    
    def __init__(
        self,
        settings=None,
        buffer_service: Optional[BufferService] = None,
        db_service: Optional[DatabaseService] = None
    ):
        """Initialize RAG service.
        
        Args:
            settings: Settings object (memfuse_mvp style) or None
            buffer_service: Buffer service for retrieval
            db_service: Database service for metadata
        """
        self.settings = settings
        self.buffer_service = buffer_service
        self.db_service = db_service
    
    async def _ensure_services(self):
        """Ensure services are available."""
        if self.db_service is None:
            self.db_service = await DatabaseService.get_instance()
        
        if self.buffer_service is None:
            # Try to get a default buffer service
            try:
                self.buffer_service = await ServiceFactory.get_buffer_service(
                    user="default",
                    agent="default", 
                    session="default"
                )
            except Exception as e:
                logger.warning(f"Failed to get buffer service: {e}")
    
    def chat(
        self,
        session_id: str,
        query: str,
        history_messages: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5
    ) -> str:
        """Synchronous chat interface for compatibility with memfuse_mvp pattern."""
        import asyncio
        
        # Handle async call in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.chat_async(session_id, query, history_messages, top_k)
        )
    
    async def chat_async(
        self,
        session_id: str,
        query: str,
        history_messages: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5
    ) -> str:
        """Chat interface with RAG retrieval.
        
        Args:
            session_id: Session ID for context
            query: User query
            history_messages: Optional message history
            top_k: Number of results to retrieve
            
        Returns:
            Generated response
        """
        try:
            await self._ensure_services()
            
            # Retrieve relevant context
            if self.buffer_service:
                try:
                    retrieval_result = await self.buffer_service.query(
                        query=query,
                        top_k=top_k,
                        session_id=session_id
                    )
                    
                    # Extract context from retrieval results
                    context_items = []
                    if (retrieval_result.get("status") == "success" and 
                        "data" in retrieval_result and 
                        "results" in retrieval_result["data"]):
                        
                        for result in retrieval_result["data"]["results"]:
                            if "content" in result:
                                context_items.append(result["content"])
                    
                    # Generate response based on context
                    if context_items:
                        context_text = "\n\n".join(context_items[:3])  # Use top 3 results
                        response = f"Based on the available information:\n\n{context_text}\n\nIn response to your query '{query}': The information above provides relevant context for your question."
                    else:
                        response = f"I don't have specific information to answer your query '{query}'. Could you provide more context or rephrase your question?"
                        
                except Exception as e:
                    logger.error(f"RAG retrieval failed: {e}")
                    response = f"I encountered an issue while searching for information about '{query}'. Please try rephrasing your question."
            else:
                response = f"RAG service is not fully configured. Unable to process query: '{query}'"
            
            return response
            
        except Exception as e:
            logger.error(f"RAG chat failed: {e}")
            return f"I'm sorry, I encountered an error while processing your query: '{query}'"
    
    async def retrieve(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents.
        
        Args:
            query: Search query
            session_id: Optional session ID for context
            top_k: Number of results to retrieve
            filters: Optional filters for retrieval
            
        Returns:
            List of retrieved documents
        """
        try:
            await self._ensure_services()
            
            if not self.buffer_service:
                logger.warning("No buffer service available for retrieval")
                return []
            
            # Perform retrieval
            retrieval_params = {
                "query": query,
                "top_k": top_k
            }
            
            if session_id:
                retrieval_params["session_id"] = session_id
            
            result = await self.buffer_service.query(**retrieval_params)
            
            # Extract results
            if (result.get("status") == "success" and 
                "data" in result and 
                "results" in result["data"]):
                return result["data"]["results"]
            else:
                logger.warning(f"Retrieval returned non-success status: {result.get('status')}")
                return []
                
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return []
    
    async def generate_with_context(
        self,
        query: str,
        context: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate response with provided context.
        
        Args:
            query: User query
            context: Retrieved context documents
            system_prompt: Optional system prompt
            
        Returns:
            Generated response
        """
        try:
            # Extract text content from context
            context_texts = []
            for item in context:
                if "content" in item:
                    context_texts.append(item["content"])
                elif "text" in item:
                    context_texts.append(item["text"])
            
            if context_texts:
                context_str = "\n\n".join(context_texts)
                response = f"Based on the following context:\n\n{context_str}\n\nAnswer: {query}"
            else:
                response = f"No context available for query: {query}"
            
            return response
            
        except Exception as e:
            logger.error(f"RAG generation with context failed: {e}")
            return f"Error generating response for query: {query}"
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the RAG service.
        
        Returns:
            Service information dictionary
        """
        return {
            "service_name": "RAGService",
            "buffer_service_available": self.buffer_service is not None,
            "db_service_available": self.db_service is not None,
            "status": "active" if (self.buffer_service and self.db_service) else "limited"
        }