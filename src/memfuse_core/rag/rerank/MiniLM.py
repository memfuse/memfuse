"""MiniLM reranker implementation for MemFuse server.

This module implements the RerankerBase interface using MiniLM cross-encoder models
from the sentence-transformers library.
"""

import time
import traceback
from typing import List, Any, Optional, Tuple, Dict, Union
import numpy as np
from loguru import logger

# Try to import sentence_transformers
try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence_transformers not available, reranking features will be limited")

from .base import RerankerBase, RerankerRegistry
from ...models.core import Item, Query


@RerankerRegistry.register("minilm")
class MiniLMReranker(RerankerBase):
    """MiniLM cross-encoder reranker implementation.
    
    This class implements the RerankerBase interface using MiniLM cross-encoder models
    from the sentence-transformers library. It supports various MiniLM cross-encoder models
    such as cross-encoder/ms-marco-MiniLM-L6-v2, etc.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        existing_model: Any = None,
        **kwargs
    ):
        """Initialize the reranker.
        
        Args:
            model_name: Name of the model to use (e.g., 'cross-encoder/ms-marco-MiniLM-L6-v2')
            existing_model: An existing CrossEncoder model instance to reuse
            **kwargs: Additional arguments
        """
        # Get configuration
        from ...utils.config import config_manager
        from omegaconf import OmegaConf

        config_dict = config_manager.get_config()
        cfg = OmegaConf.create(config_dict)
        
        # Use model from config if not specified
        if model_name is None and hasattr(cfg, 'retrieval') and hasattr(cfg.retrieval, 'rerank_model'):
            model_name = cfg.retrieval.rerank_model
        else:
            # Default model if not specified in config
            model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
            
        # Validate that this is a MiniLM model
        if not self._is_minilm_model(model_name):
            logger.warning(f"Model {model_name} may not be a MiniLM model, but will try to use it anyway")
            
        self.model_name = model_name
        
        # Use existing model if provided
        if existing_model is not None:
            self.model = existing_model
            logger.debug(f"Using existing CrossEncoder model: {model_name}")
        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            # Load the model
            try:
                logger.info(f"Loading MiniLM CrossEncoder rerank model: {model_name}")
                start_time = time.time()
                
                self.model = CrossEncoder(model_name, trust_remote_code=False)
                
                load_time = time.time() - start_time
                logger.info(f"MiniLM CrossEncoder model loaded in {load_time:.2f} seconds")
                
                # Apply FP16 optimization if configured
                if hasattr(cfg, 'retrieval') and hasattr(cfg.retrieval, 'use_fp16') and cfg.retrieval.use_fp16:
                    logger.info("Using FP16 precision for rerank model inference")
                    if hasattr(self.model.model, "half"):
                        self.model.model.half()
                        logger.info("Successfully converted rerank model to FP16 precision")
                
            except Exception as e:
                logger.error(f"Error loading MiniLM CrossEncoder model {model_name}: {e}")
                logger.debug(f"CrossEncoder model loading traceback: {traceback.format_exc()}")
                self.model = None
        else:
            logger.warning("sentence_transformers not available, reranking will not work")
            self.model = None
            
        # Store rerank configuration
        self.rerank_strategy = "rrf"  # Default strategy
        if hasattr(cfg, 'retrieval') and hasattr(cfg.retrieval, 'rerank_strategy'):
            self.rerank_strategy = cfg.retrieval.rerank_strategy
            
        # Store whether reranking is enabled by default
        self.use_rerank = False  # Default setting
        if hasattr(cfg, 'retrieval') and hasattr(cfg.retrieval, 'use_rerank'):
            self.use_rerank = cfg.retrieval.use_rerank
            
        logger.info(
            f"MiniLM rerank configuration: strategy={self.rerank_strategy}, "
            f"model={self.model_name}, enabled={self.use_rerank}"
        )
    
    def _is_minilm_model(self, model_name: str) -> bool:
        """Check if the model is a MiniLM model.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if the model is a MiniLM model, False otherwise
        """
        return "minilm" in model_name.lower()
    
    async def initialize(self) -> bool:
        """Initialize the reranker.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        return self.model is not None
    
    async def rerank(
        self,
        query: str,
        items: List[Any],
        top_k: int = 5,
        source: str = "default"
    ) -> List[Any]:
        """Rerank items based on their relevance to the query.
        
        Args:
            query: The query string
            items: List of items to rerank
            top_k: Number of top items to return
            source: Source of the items
            
        Returns:
            List of reranked items
        """
        if not self.model or not items:
            return items[:top_k] if items else []
            
        try:
            # Extract text from items
            texts = []
            for item in items:
                if hasattr(item, "content"):
                    texts.append(item.content)
                elif hasattr(item, "text"):
                    texts.append(item.text)
                elif isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, dict) and "content" in item:
                    texts.append(item["content"])
                elif isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])
                else:
                    logger.warning(f"Could not extract text from item: {item}")
                    texts.append(str(item))
                    
            # Score the items
            scores = await self.score(query, texts)
            
            # Create (item, score) pairs
            item_scores = list(zip(items, scores))
            
            # Sort by score in descending order
            item_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k items
            return [item for item, _ in item_scores[:top_k]]
            
        except Exception as e:
            logger.error(f"Error reranking items: {e}")
            logger.debug(f"Reranking traceback: {traceback.format_exc()}")
            return items[:top_k] if items else []
    
    async def score(
        self,
        query: str,
        texts: List[str]
    ) -> List[float]:
        """Score texts based on their relevance to the query.
        
        Args:
            query: The query string
            texts: List of texts to score
            
        Returns:
            List of relevance scores
        """
        if not self.model or not texts:
            return [0.0] * len(texts)
            
        try:
            # Create pairs
            pairs = [(query, text) for text in texts]
            
            # Score the pairs
            return await self.score_pairs(pairs)
            
        except Exception as e:
            logger.error(f"Error scoring texts: {e}")
            logger.debug(f"Scoring traceback: {traceback.format_exc()}")
            return [0.0] * len(texts)
    
    async def score_pairs(
        self,
        pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """Score text pairs based on their relevance.
        
        Args:
            pairs: List of (query, text) pairs to score
            
        Returns:
            List of relevance scores
        """
        if not self.model or not pairs:
            return [0.0] * len(pairs)
            
        try:
            # Score the pairs
            scores = self.model.predict(pairs)
            
            # Ensure scores is a list of floats
            if isinstance(scores, np.ndarray):
                scores = scores.tolist()
            elif not isinstance(scores, list):
                scores = [float(scores)]
                
            return scores
            
        except Exception as e:
            logger.error(f"Error scoring pairs: {e}")
            logger.debug(f"Scoring pairs traceback: {traceback.format_exc()}")
            return [0.0] * len(pairs)
