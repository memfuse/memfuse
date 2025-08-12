"""
Streaming Data Processing Pipeline for PgVectorScale Memory Layer.

This module implements a complete data flow processing pipeline from
streaming data to M0 storage, then to M1 chunking and embedding.
"""

import asyncio
import uuid
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
from loguru import logger

from .pgvectorscale_memory_layer import PgVectorScaleMemoryLayer
from ..interfaces.message_interface import MessageBatchList
from ..rag.chunk.base import ChunkData


class StreamingDataProcessor:
    """
    Streaming data processor that handles the complete M0 -> M1 pipeline.
    
    This processor implements the data flow described in the simplified architecture:
    1. Streaming Data -> M0 Layer (Raw Messages)
    2. M0 -> Intelligent Chunking -> M1 Layer (Chunks + Embeddings)
    3. M1 -> pgvectorscale StreamingDiskANN (High-performance search)
    """
    
    def __init__(
        self,
        user_id: str = "default_user",
        db_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 10,
        processing_delay: float = 0.1
    ):
        """Initialize the streaming data processor.
        
        Args:
            user_id: User identifier
            db_config: Database configuration
            batch_size: Number of messages to process in each batch
            processing_delay: Delay between processing batches (seconds)
        """
        self.user_id = user_id
        self.batch_size = batch_size
        self.processing_delay = processing_delay
        
        # Initialize memory layer
        self.memory_layer = PgVectorScaleMemoryLayer(
            user_id=user_id,
            db_config=db_config
        )
        
        # Processing state
        self.is_processing = False
        self.total_messages_processed = 0
        self.total_batches_processed = 0
        self.processing_errors = 0
        
        logger.info(f"StreamingDataProcessor: Initialized for user {user_id}")
    
    async def initialize(self) -> bool:
        """Initialize the processor and underlying memory layer."""
        try:
            logger.info("StreamingDataProcessor: Initializing...")
            
            # Initialize memory layer
            success = await self.memory_layer.initialize()
            if not success:
                logger.error("StreamingDataProcessor: Memory layer initialization failed")
                return False
            
            logger.info("StreamingDataProcessor: Initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"StreamingDataProcessor: Initialization failed: {e}")
            return False
    
    async def process_streaming_messages(
        self,
        messages: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a list of streaming messages through the M0 -> M1 pipeline.
        
        Args:
            messages: List of message dictionaries
            session_id: Optional session identifier
            metadata: Optional metadata
            
        Returns:
            Processing result dictionary
        """
        if not messages:
            return {"success": False, "message": "No messages to process"}
        
        if not self.memory_layer.initialized:
            await self.initialize()
        
        start_time = datetime.now()
        session_id = session_id or str(uuid.uuid4())
        
        try:
            logger.info(f"StreamingDataProcessor: Processing {len(messages)} messages for session {session_id}")
            
            # Convert messages to message batch format
            message_batch_list = self._create_message_batches(messages)
            
            # Process through memory layer (M0 -> M1 pipeline)
            write_result = await self.memory_layer.write(
                message_batch_list=message_batch_list,
                session_id=session_id,
                metadata=metadata
            )
            
            # Update processing statistics
            if write_result.success:
                self.total_messages_processed += len(messages)
                self.total_batches_processed += len(message_batch_list)
            else:
                self.processing_errors += 1
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate items written from layer results
            items_written = 0
            if write_result.layer_results:
                for layer_result in write_result.layer_results.values():
                    if isinstance(layer_result, dict) and "items_processed" in layer_result:
                        items_written += layer_result["items_processed"]

            result = {
                "success": write_result.success,
                "message": write_result.message,
                "session_id": session_id,
                "messages_processed": len(messages),
                "batches_created": len(message_batch_list),
                "items_written": items_written,
                "processing_time": processing_time,
                "metadata": write_result.metadata
            }
            
            logger.info(f"StreamingDataProcessor: Processed {len(messages)} messages in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.processing_errors += 1
            
            logger.error(f"StreamingDataProcessor: Processing failed after {processing_time:.3f}s: {e}")
            
            return {
                "success": False,
                "message": f"Processing failed: {str(e)}",
                "session_id": session_id,
                "messages_processed": 0,
                "processing_time": processing_time,
                "error": str(e)
            }
    
    async def process_streaming_batch(
        self,
        message_batch: List[Dict[str, Any]],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a single batch of streaming messages.
        
        Args:
            message_batch: Batch of message dictionaries
            session_id: Optional session identifier
            
        Returns:
            Processing result dictionary
        """
        return await self.process_streaming_messages(
            messages=message_batch,
            session_id=session_id,
            metadata={"batch_processing": True}
        )
    
    async def start_continuous_processing(
        self,
        message_generator: AsyncGenerator[Dict[str, Any], None],
        session_id: Optional[str] = None
    ) -> None:
        """Start continuous processing of streaming messages.
        
        Args:
            message_generator: Async generator that yields message dictionaries
            session_id: Optional session identifier
        """
        if self.is_processing:
            logger.warning("StreamingDataProcessor: Already processing, ignoring start request")
            return
        
        self.is_processing = True
        session_id = session_id or str(uuid.uuid4())
        current_batch = []
        
        logger.info(f"StreamingDataProcessor: Starting continuous processing for session {session_id}")
        
        try:
            async for message in message_generator:
                if not self.is_processing:
                    break
                
                current_batch.append(message)
                
                # Process batch when it reaches the configured size
                if len(current_batch) >= self.batch_size:
                    await self._process_batch_with_delay(current_batch, session_id)
                    current_batch = []
            
            # Process remaining messages
            if current_batch:
                await self._process_batch_with_delay(current_batch, session_id)
            
            logger.info("StreamingDataProcessor: Continuous processing completed")
            
        except Exception as e:
            logger.error(f"StreamingDataProcessor: Continuous processing failed: {e}")
        finally:
            self.is_processing = False
    
    async def stop_continuous_processing(self) -> None:
        """Stop continuous processing."""
        if self.is_processing:
            logger.info("StreamingDataProcessor: Stopping continuous processing...")
            self.is_processing = False
        else:
            logger.info("StreamingDataProcessor: No continuous processing to stop")
    
    async def query_processed_data(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """Query the processed data using vector similarity search.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            **kwargs: Additional query parameters
            
        Returns:
            Query result dictionary
        """
        if not self.memory_layer.initialized:
            await self.initialize()
        
        try:
            query_result = await self.memory_layer.query(query, top_k, **kwargs)
            
            return {
                "success": True,
                "query": query,
                "results": query_result.results,
                "total_count": query_result.total_count,
                "layer_sources": query_result.layer_sources,
                "metadata": query_result.metadata
            }
            
        except Exception as e:
            logger.error(f"StreamingDataProcessor: Query failed: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "results": [],
                "total_count": 0
            }
    
    def _create_message_batches(self, messages: List[Dict[str, Any]]) -> MessageBatchList:
        """Create message batches from a list of messages."""
        # For simplicity, create batches of the configured size
        batches = []
        for i in range(0, len(messages), self.batch_size):
            batch = messages[i:i + self.batch_size]
            batches.append(batch)
        
        return batches
    
    async def _process_batch_with_delay(
        self,
        batch: List[Dict[str, Any]],
        session_id: str
    ) -> None:
        """Process a batch with configured delay."""
        try:
            result = await self.process_streaming_batch(batch, session_id)
            
            if result["success"]:
                items_written = result.get('items_written', 0)
                logger.debug(f"StreamingDataProcessor: Batch processed successfully: {items_written} items")
            else:
                logger.warning(f"StreamingDataProcessor: Batch processing failed: {result['message']}")
            
            # Add processing delay
            if self.processing_delay > 0:
                await asyncio.sleep(self.processing_delay)
                
        except Exception as e:
            logger.error(f"StreamingDataProcessor: Batch processing error: {e}")
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        try:
            memory_stats = await self.memory_layer.get_stats()
            
            return {
                "total_messages_processed": self.total_messages_processed,
                "total_batches_processed": self.total_batches_processed,
                "processing_errors": self.processing_errors,
                "is_processing": self.is_processing,
                "batch_size": self.batch_size,
                "processing_delay": self.processing_delay,
                "memory_layer_stats": memory_stats
            }
            
        except Exception as e:
            logger.error(f"StreamingDataProcessor: Stats retrieval failed: {e}")
            return {
                "error": str(e),
                "total_messages_processed": self.total_messages_processed,
                "total_batches_processed": self.total_batches_processed,
                "processing_errors": self.processing_errors
            }
    
    async def close(self) -> None:
        """Close the processor and cleanup resources."""
        try:
            # Stop any ongoing processing
            await self.stop_continuous_processing()
            
            # Close memory layer
            await self.memory_layer.close()
            
            logger.info("StreamingDataProcessor: Closed successfully")
            
        except Exception as e:
            logger.error(f"StreamingDataProcessor: Close failed: {e}")


# Utility functions for creating streaming data generators

async def create_demo_message_generator(
    num_messages: int = 36,
    delay_between_messages: float = 0.1
) -> AsyncGenerator[Dict[str, Any], None]:
    """Create a demo message generator for testing.
    
    Args:
        num_messages: Number of messages to generate
        delay_between_messages: Delay between messages (seconds)
        
    Yields:
        Message dictionaries
    """
    # Demo conversation templates (similar to the pgvectorscale demo)
    conversation_templates = [
        ("user", "I want to learn Python machine learning, where should I start?"),
        ("assistant", "I recommend starting with scikit-learn, which provides rich machine learning algorithms and tools."),
        ("user", "What are the main algorithms in scikit-learn?"),
        ("assistant", "scikit-learn includes classification (SVM, Random Forest), regression (Linear, Ridge), clustering (K-means, DBSCAN), and dimensionality reduction (PCA, t-SNE)."),
        ("user", "How is deep learning different from traditional machine learning?"),
        ("assistant", "Deep learning uses neural networks with multiple layers to automatically learn feature representations, while traditional ML often requires manual feature engineering."),
        ("user", "Which deep learning framework should I choose?"),
        ("assistant", "PyTorch is great for research and experimentation, TensorFlow/Keras for production deployment, and JAX for high-performance computing."),
        ("user", "What's the typical data science project workflow?"),
        ("assistant", "The workflow typically includes: 1) Problem definition, 2) Data collection and cleaning, 3) Exploratory data analysis, 4) Feature engineering, 5) Model selection and training, 6) Evaluation and validation, 7) Deployment and monitoring."),
        ("user", "How do you handle missing data in datasets?"),
        ("assistant", "Common strategies include: deletion (listwise/pairwise), imputation (mean/median/mode), advanced imputation (KNN, iterative), or using algorithms that handle missing values natively."),
        ("user", "What are vector databases used for?"),
        ("assistant", "Vector databases store and search high-dimensional embeddings for similarity search, recommendation systems, semantic search, and RAG (Retrieval-Augmented Generation) applications."),
    ]
    
    conversation_id = str(uuid.uuid4())
    
    for i in range(num_messages):
        template_idx = i % len(conversation_templates)
        role, content = conversation_templates[template_idx]
        
        message = {
            "content": content,
            "role": role,
            "conversation_id": conversation_id,
            "sequence_number": i + 1,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "template_index": template_idx,
                "message_index": i
            }
        }
        
        yield message
        
        if delay_between_messages > 0:
            await asyncio.sleep(delay_between_messages)
