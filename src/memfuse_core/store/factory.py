"""Store factory for MemFuse server."""

from typing import Dict, Optional, Any
import os
from loguru import logger

from ..models import StoreBackend
from ..utils.config import config_manager
from ..utils.path_manager import PathManager
from ..rag.encode.base import EncoderBase
from ..rag.encode.MiniLM import MiniLMEncoder
from .vector_store.base import VectorStore
from .graph_store.base import GraphStore
from .keyword_store.base import KeywordStore


class StoreFactory:
    """Factory for creating store instances."""

    # Class variables to store singleton instances
    _vector_store_instances: Dict[str, VectorStore] = {}
    _graph_store_instances: Dict[str, GraphStore] = {}
    _keyword_store_instances: Dict[str, KeywordStore] = {}
    _multi_path_retrieval_instances: Dict[str, Any] = {}
    _encoder_instances: Dict[str, EncoderBase] = {}

    @classmethod
    def _generate_stable_key(cls, **kwargs) -> str:
        """Generate a stable cache key that is not affected by class names or import paths.

        Args:
            **kwargs: Parameters to include in the key

        Returns:
            A stable string key for caching
        """
        # Initialize the key parts list
        key_parts = []

        # Add all parameters that should affect the cache key
        for param_name in sorted(kwargs.keys()):
            value = kwargs.get(param_name)
            if value is not None:
                key_parts.append(f"{param_name}={value}")

        # Join all parts with a separator that's unlikely to appear in the values
        return "|||".join(key_parts)

    @classmethod
    async def create_encoder(
        cls,
        model_name: Optional[str] = None,
        cache_size: Optional[int] = None,
        existing_model: Any = None,
        **kwargs
    ) -> EncoderBase:
        """Create an encoder instance.

        Args:
            model_name: Name of the model to use
            cache_size: Size of the embedding cache
            existing_model: An existing SentenceTransformer model instance to reuse
            **kwargs: Additional arguments

        Returns:
            An encoder instance
        """
        # Get configuration
        config = config_manager.get_config()

        # Use defaults from config if not provided
        model_name = model_name or config["embedding"]["model"]
        cache_size = int(cache_size or config["store"]["cache_size"])

        # Generate a stable key for caching
        stable_key = cls._generate_stable_key(
            model_name=model_name,
            cache_size=cache_size,
            has_existing_model="yes" if existing_model is not None else "no"
        )

        # Check if we already have an instance for this configuration
        if stable_key in cls._encoder_instances:
            logger.debug(f"Reusing existing encoder instance for {model_name}")
            return cls._encoder_instances[stable_key]

        # Check if we have a pre-loaded model from the model registry
        if existing_model is None:
            # Try to get the pre-loaded model from the model registry
            try:
                from ..interfaces.model_provider import ModelRegistry
                existing_model = ModelRegistry.get_embedding_model()
                if existing_model is not None:
                    logger.info(f"Using pre-loaded model: {model_name}")
            except (ImportError, AttributeError) as e:
                # If we can't import ModelRegistry or it doesn't have
                # get_embedding_model, continue without the pre-loaded model
                logger.debug(f"Could not get pre-loaded model: {e}")
                pass

        # Create a new instance - pass the model name explicitly
        encoder = MiniLMEncoder(
            model_name=model_name,
            cache_size=cache_size,
            existing_model=existing_model,
            **kwargs
        )

        # Cache the instance with the stable key
        cls._encoder_instances[stable_key] = encoder

        return encoder

    @classmethod
    async def create_vector_store(
        cls,
        backend: Optional[StoreBackend] = None,
        data_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        buffer_size: Optional[int] = None,
        cache_size: Optional[int] = None,
        encoder: Optional[EncoderBase] = None,
        existing_model: Any = None,
        **kwargs
    ) -> VectorStore:
        """Create a vector store instance.

        Args:
            backend: Storage backend to use
            data_dir: Directory to store data
            model_name: Name of the embedding model
            buffer_size: Size of the write buffer
            cache_size: Size of the query cache
            encoder: Encoder to use
            **kwargs: Additional arguments

        Returns:
            A vector store instance
        """
        # Get configuration
        config = config_manager.get_config()

        # Use defaults from config if not provided
        backend = backend or StoreBackend(config["store"]["backend"])
        data_dir = str(data_dir or config["data_dir"])
        buffer_size = int(buffer_size or config["store"]["buffer_size"])
        cache_size = int(cache_size or config["store"]["cache_size"])
        model_name = str(model_name or config["embedding"]["model"])

        # Create encoder if not provided
        if encoder is None:
            encoder = await cls.create_encoder(
                model_name=model_name,
                cache_size=cache_size,
                existing_model=existing_model
            )

        # Generate a stable key for caching
        stable_key = cls._generate_stable_key(
            data_dir=data_dir,
            backend=str(backend),
            model_name=model_name,
            buffer_size=buffer_size,
            cache_size=cache_size,
            encoder_id=id(encoder) if encoder else None
        )

        # Check if we already have an instance for this configuration
        if stable_key in cls._vector_store_instances:
            logger.debug(f"Reusing existing vector store instance for {data_dir}")
            return cls._vector_store_instances[stable_key]

        # Create user directory if it doesn't exist
        PathManager.ensure_directory(data_dir)

        # Create a new instance based on the backend
        if backend == StoreBackend.NUMPY:
            from .vector_store.numpy_store import NumpyVectorStore
            store = NumpyVectorStore(
                data_dir=data_dir,
                encoder=encoder,
                buffer_size=buffer_size,
                cache_size=cache_size,
                **kwargs
            )
        elif backend == StoreBackend.QDRANT:
            from .vector_store.qdrant_store import QdrantVectorStore

            # Get embedding dimension from config
            embedding_dim = config["embedding"]["dimension"]

            # Check if we have a specific dimension for this model in the config
            model_key = model_name.lower() if model_name else ""
            for key, model_config in config["embedding"]["models"].items():
                if key.lower() in model_key:
                    embedding_dim = model_config["dimension"]
                    break

            logger.info(
                f"Using embedding dimension: {embedding_dim}")

            store = QdrantVectorStore(
                data_dir=data_dir,
                encoder=encoder,
                embedding_dim=embedding_dim,
                cache_size=cache_size,
                buffer_size=buffer_size,
                **kwargs
            )
        elif backend == StoreBackend.SQLITE:
            from .vector_store.sqlite_store import SQLiteVectorStore
            store = SQLiteVectorStore(
                data_dir=data_dir,
                encoder=encoder,
                cache_size=cache_size,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown store backend: {backend}")

        # Initialize the store
        await store.initialize()

        # Cache the instance with the stable key
        cls._vector_store_instances[stable_key] = store

        return store

    @classmethod
    async def create_graph_store(
        cls,
        data_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        backend: Optional[StoreBackend] = None,
        encoder: Optional[EncoderBase] = None,
        existing_model: Any = None,
        **kwargs
    ) -> GraphStore:
        """Create a graph store instance.

        Args:
            data_dir: Directory to store data
            model_name: Name of the embedding model
            backend: Storage backend to use
            encoder: Encoder to use
            **kwargs: Additional arguments

        Returns:
            A graph store instance
        """
        # Get configuration
        config = config_manager.get_config()

        # Use defaults from config if not provided
        data_dir = str(data_dir or config["data_dir"])
        model_name = str(model_name or config["embedding"]["model"])
        backend = backend or StoreBackend(config["store"]["backend"])

        # Create encoder if not provided
        if encoder is None:
            encoder = await cls.create_encoder(
                model_name=model_name,
                cache_size=config["store"]["cache_size"],
                existing_model=existing_model
            )

        # Create user directory if it doesn't exist
        PathManager.ensure_directory(data_dir)

        # Generate a stable key for caching
        stable_key = cls._generate_stable_key(
            data_dir=data_dir,
            backend=str(backend),
            model_name=model_name,
            encoder_id=id(encoder) if encoder else None
        )

        # Check if we already have an instance for this configuration
        if stable_key in cls._graph_store_instances:
            logger.debug(f"Reusing existing graph store instance for {data_dir}")
            return cls._graph_store_instances[stable_key]

        # Use GraphMLStore for all backends for now
        # This will be migrated to Neo4j in the future
        from .graph_store.graphml_store import GraphMLStore
        store = GraphMLStore(
            data_dir=data_dir,
            encoder=encoder,
            buffer_size=config["store"]["buffer_size"],
            **kwargs
        )

        # Initialize the store
        await store.initialize()

        # Cache the instance with the stable key
        cls._graph_store_instances[stable_key] = store

        return store

    @classmethod
    async def create_keyword_store(
        cls,
        data_dir: Optional[str] = None,
        cache_size: Optional[int] = None,
        backend: Optional[StoreBackend] = None,
        **kwargs
    ) -> KeywordStore:
        """Create a keyword store instance.

        Args:
            data_dir: Directory to store data
            cache_size: Size of the query cache
            backend: Storage backend to use
            **kwargs: Additional arguments

        Returns:
            A keyword store instance
        """
        # Get configuration
        config = config_manager.get_config()

        # Use defaults from config if not provided
        data_dir = str(data_dir or config["data_dir"])
        cache_size = int(cache_size or config["store"]["cache_size"])
        backend = backend or StoreBackend(config["store"]["backend"])

        # Create user directory if it doesn't exist
        PathManager.ensure_directory(data_dir)

        # Generate a stable key for caching
        stable_key = cls._generate_stable_key(
            data_dir=data_dir,
            backend=str(backend),
            cache_size=cache_size
        )

        # Check if we already have an instance for this configuration
        if stable_key in cls._keyword_store_instances:
            logger.debug(f"Reusing existing keyword store instance for {data_dir}")
            return cls._keyword_store_instances[stable_key]

        # Use SQLiteKeywordStore for keyword search
        from .keyword_store.sqlite_store import SQLiteKeywordStore
        store = SQLiteKeywordStore(
            data_dir=data_dir,
            cache_size=cache_size,
            **kwargs
        )

        # Initialize the store
        await store.initialize()

        # Cache the instance with the stable key
        cls._keyword_store_instances[stable_key] = store

        return store

    @classmethod
    async def create_multi_path_retrieval(
        cls,
        data_dir: Optional[str] = None,
        vector_store: Optional[VectorStore] = None,
        graph_store: Optional[GraphStore] = None,
        keyword_store: Optional[KeywordStore] = None,
        cache_size: Optional[int] = None,
        fusion_strategy: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Create a multi-path retrieval instance.

        Args:
            data_dir: Directory to store data
            vector_store: Vector store instance
            graph_store: Graph store instance
            keyword_store: Keyword store instance
            cache_size: Size of the query cache
            fusion_strategy: Score fusion strategy to use
                ('simple', 'normalized', or 'rrf')
            **kwargs: Additional arguments

        Returns:
            A multi-path retrieval instance
        """
        # Get configuration
        cfg = config_manager.get_config()

        # Use defaults from config if not provided
        data_dir = str(data_dir or cfg["data_dir"])

        # Create stores if not provided
        if vector_store is None and cfg["store"]["multi_path"]["use_vector"]:
            vector_store = await cls.create_vector_store(
                data_dir=data_dir,
                **kwargs
            )

        if graph_store is None and cfg["store"]["multi_path"]["use_graph"]:
            graph_store = await cls.create_graph_store(
                data_dir=data_dir,
                **kwargs
            )

        if keyword_store is None and cfg["store"]["multi_path"]["use_keyword"]:
            keyword_store = await cls.create_keyword_store(
                data_dir=data_dir,
                **kwargs
            )

        # Generate a stable key for caching that doesn't depend on class names
        # Include only parameters that should affect instance identity
        stable_key = cls._generate_stable_key(
            data_dir=data_dir,
            # Include store identities if provided
            vector_store_id=id(vector_store) if vector_store else None,
            graph_store_id=id(graph_store) if graph_store else None,
            keyword_store_id=id(keyword_store) if keyword_store else None,
            cache_size=cache_size,
            fusion_strategy=fusion_strategy
        )

        logger.debug(f"Looking for multi-path retrieval with key: {stable_key}")

        # Check if we already have an instance for this configuration
        if stable_key in cls._multi_path_retrieval_instances:
            logger.info(f"Reusing existing multi-path retrieval instance for {data_dir}")
            return cls._multi_path_retrieval_instances[stable_key]

        # Get weights from configuration
        vector_weight = cfg["store"]["multi_path"]["vector_weight"]
        graph_weight = cfg["store"]["multi_path"]["graph_weight"]
        keyword_weight = cfg["store"]["multi_path"]["keyword_weight"]

        # Get fusion strategy from configuration or use default
        if fusion_strategy is None:
            # Default to "rrf" if not specified in config
            fusion_strategy = str(cfg["store"]["multi_path"].get(
                "fusion_strategy", "rrf"))

        # Import here to avoid circular imports
        from ..rag.retrieve import HybridRetrieval

        retrieval = HybridRetrieval(
            vector_store=vector_store,
            graph_store=graph_store,
            keyword_store=keyword_store,
            cache_size=cache_size or cfg["store"]["cache_size"],
            vector_weight=vector_weight,
            graph_weight=graph_weight,
            keyword_weight=keyword_weight,
            fusion_strategy=fusion_strategy
        )

        # Cache the instance with the stable key
        logger.info(f"Caching new multi-path retrieval instance with key: {stable_key}")
        cls._multi_path_retrieval_instances[stable_key] = retrieval

        return retrieval
