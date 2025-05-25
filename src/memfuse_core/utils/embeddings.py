"""Embedding utilities for MemFuse.

This module provides utilities for creating and managing embeddings, including
caching mechanisms for efficient embedding storage and retrieval.
"""

import os
import hashlib
import numpy as np
from typing import List, Optional, Union, Any, Dict
from collections import OrderedDict
from ..models.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_EMBEDDING_DIM
from loguru import logger
# Try to import sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Try to import openai
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import configuration
try:
    from ..utils.config import (
        get_embedding_dim,
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


# Model cache
_model_cache: Dict[str, Any] = {}


def get_model(model_name: str) -> Any:
    """Get a model by name.

    Args:
        model_name: Name of the model to load

    Returns:
        Loaded model
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            model = SentenceTransformer(model_name, trust_remote_code=True)
            _model_cache[model_name] = model
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")

    raise ValueError(f"Could not load model {model_name}")


async def create_embedding(text: str, model: Optional[str] = None) -> List[float]:
    """Create an embedding for a text.

    Args:
        text: Text to embed
        model: Model to use for embedding

    Returns:
        Embedding vector
    """
    # Get dimension from configuration if available
    dimension = None
    if CONFIG_AVAILABLE:
        try:
            dimension = get_embedding_dim(model)
        except Exception:
            dimension = DEFAULT_EMBEDDING_DIM
    else:
        dimension = DEFAULT_EMBEDDING_DIM

    if not text:
        # Return a zero vector for empty text
        return [0.0] * dimension

    # Use sentence_transformers if available
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            model_name = model or DEFAULT_EMBEDDING_MODEL
            model_instance = get_model(model_name)
            embedding = model_instance.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(
                f"Error creating embedding with sentence_transformers: {e}")

    # Use OpenAI if available
    if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        try:
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embedding with OpenAI: {e}")

    # Fallback to hash-based embedding
    return create_hash_embedding(text, dimension)


async def create_batch_embeddings(
    texts: List[str],
    model: Optional[str] = None
) -> List[List[float]]:
    """Create embeddings for multiple texts.

    Args:
        texts: Texts to embed
        model: Model to use for embedding

    Returns:
        List of embedding vectors
    """
    if not texts:
        return []

    # Get dimension from configuration if available
    dimension = None
    if CONFIG_AVAILABLE:
        try:
            dimension = get_embedding_dim(model)
        except Exception:
            dimension = DEFAULT_EMBEDDING_DIM
    else:
        dimension = DEFAULT_EMBEDDING_DIM

    # Use sentence_transformers if available
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            model_name = model or DEFAULT_EMBEDDING_MODEL
            model_instance = get_model(model_name)
            embeddings = model_instance.encode(texts)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            logger.error(
                f"Error creating batch embeddings with sentence_transformers: {e}")

    # Use OpenAI if available
    if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
        try:
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error creating batch embeddings with OpenAI: {e}")

    # Fallback to hash-based embeddings
    return [create_hash_embedding(text, dimension) for text in texts]


def create_hash_embedding(
    text: str,
    dimension: Optional[int] = None
) -> List[float]:
    """Create a hash-based embedding for a text.

    This is a fallback method when no embedding models are available.
    It creates a deterministic embedding based on the hash of the text.

    Args:
        text: Text to embed
        dimension: Dimension of the embedding (optional)

    Returns:
        Embedding vector
    """
    # Get dimension from configuration if available
    if dimension is None:
        if CONFIG_AVAILABLE:
            try:
                dimension = get_embedding_dim()
            except Exception:
                dimension = DEFAULT_EMBEDDING_DIM
        else:
            dimension = DEFAULT_EMBEDDING_DIM

    if not text:
        return [0.0] * dimension

    # Create a hash of the text
    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()

    # Convert hash bytes to a list of floats
    embedding = []
    for i in range(dimension):
        # Use modulo to cycle through the hash bytes
        byte_index = i % len(hash_bytes)
        # Convert byte to float in range [-1, 1]
        value = (hash_bytes[byte_index] / 128.0) - 1.0
        embedding.append(value)

    # Normalize the embedding
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = [v / norm for v in embedding]

    return embedding


def cosine_similarity(embedding1: Union[List[float], np.ndarray],
                    embedding2: Union[List[float], np.ndarray]) -> float:
    """Calculate cosine similarity between two embeddings.

    Args:
        embedding1: First embedding
        embedding2: Second embedding

    Returns:
        Similarity score
    """
    return calculate_similarity(embedding1, embedding2)


def cosine_similarity_matrix(embeddings1: List[Union[List[float], np.ndarray]],
                         embeddings2: List[Union[List[float], np.ndarray]]) -> np.ndarray:
    """Calculate cosine similarity matrix between two sets of embeddings.

    Args:
        embeddings1: First set of embeddings
        embeddings2: Second set of embeddings

    Returns:
        Similarity matrix
    """
    # Convert to numpy arrays if needed
    if not isinstance(embeddings1, np.ndarray):
        embeddings1 = np.array(embeddings1)
    if not isinstance(embeddings2, np.ndarray):
        embeddings2 = np.array(embeddings2)
    
    # Normalize embeddings
    embeddings1 = normalize_embeddings(embeddings1)
    embeddings2 = normalize_embeddings(embeddings2)
    
    # Calculate dot product (cosine similarity for normalized vectors)
    return np.dot(embeddings1, embeddings2.T)


def normalize_embedding(embedding: Union[List[float], np.ndarray]) -> np.ndarray:
    """Normalize a single embedding vector.

    Args:
        embedding: Embedding vector to normalize

    Returns:
        Normalized embedding vector
    """
    # Convert to numpy array if needed
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)
    
    # Calculate L2 norm
    norm = np.linalg.norm(embedding)
    
    # Avoid division by zero
    if norm > 0:
        return embedding / norm
    return embedding


def normalize_embeddings(embeddings: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """Normalize multiple embedding vectors.

    Args:
        embeddings: Embedding vectors to normalize

    Returns:
        Normalized embedding vectors
    """
    # Convert to numpy array if needed
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)
    
    # Calculate L2 norm along axis 1
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Avoid division by zero
    norms[norms == 0] = 1.0
    
    return embeddings / norms


def calculate_similarity(embedding1: Union[List[float], np.ndarray],
                         embedding2: Union[List[float], np.ndarray]) -> float:
    """Calculate cosine similarity between two embeddings.

    Args:
        embedding1: First embedding
        embedding2: Second embedding

    Returns:
        Similarity score
    """
    # Convert to numpy arrays if needed
    if isinstance(embedding1, list):
        embedding1 = np.array(embedding1)
    if isinstance(embedding2, list):
        embedding2 = np.array(embedding2)

    # Calculate norms
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    # Handle zero norms
    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Calculate cosine similarity
    return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


class EmbeddingCache:
    """LRU cache for embeddings."""

    def __init__(self, max_size: int = 10000):
        """Initialize the cache.

        Args:
            max_size: Maximum number of items to store in the cache
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __contains__(self, key: str) -> bool:
        """Check if a key is in the cache.

        Args:
            key: Key to check

        Returns:
            True if the key is in the cache, False otherwise
        """
        return key in self.cache

    def __getitem__(self, key: str) -> np.ndarray:
        """Get an item from the cache.

        Args:
            key: Key to get

        Returns:
            Cached value

        Raises:
            KeyError: If the key is not in the cache
        """
        # Move the key to the end to mark it as recently used
        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        """Set an item in the cache.

        Args:
            key: Key to set
            value: Value to set
        """
        # If the key already exists, remove it first
        if key in self.cache:
            self.cache.pop(key)

        # Add the new key-value pair
        self.cache[key] = value

        # If the cache is too large, remove the oldest item
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()

    def __len__(self) -> int:
        """Get the number of items in the cache.

        Returns:
            Number of items in the cache
        """
        return len(self.cache)
