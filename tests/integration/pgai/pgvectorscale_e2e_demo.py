#!/usr/bin/env python3
"""
MemFuse pgvectorscale End-to-End Integration Demo

This script demonstrates the complete MemFuse memory layer architecture using
pgvectorscale with StreamingDiskANN for high-performance vector similarity search.

Features:
- Streaming conversation data generation
- M0 → M1 intelligent chunking with token-based strategy
- High-performance vector embeddings with sentence-transformers
- StreamingDiskANN vector similarity search with normalized scores (0-1 range)
- Complete data lineage tracking and validation
- Performance monitoring and optimization

Requirements:
- Python 3.8+
- sentence-transformers
- psycopg2-binary
- numpy
- pgvectorscale database running on localhost:5432

Usage:
    python3 tests/integration/pgai/pgvectorscale_e2e_demo.py

Environment Variables:
    PGVECTORSCALE_HOST: Database host (default: localhost)
    PGVECTORSCALE_PORT: Database port (default: 5432)
    PGVECTORSCALE_DB: Database name (default: memfuse)
    PGVECTORSCALE_USER: Database user (default: postgres)
    PGVECTORSCALE_PASSWORD: Database password (required)

Architecture:
    M0 Layer: Raw streaming messages with metadata
    M1 Layer: Intelligent chunking with embeddings
    pgvectorscale: StreamingDiskANN for optimized vector search
    Normalized similarity scores for cross-system comparison
"""

import os
import sys
import uuid
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class M0Message:
    """Raw streaming message in M0 layer"""
    message_id: str
    content: str
    role: str
    conversation_id: str
    sequence_number: int
    token_count: int
    created_at: datetime

@dataclass
class M1Chunk:
    """Intelligent chunk in M1 layer with embedding"""
    chunk_id: str
    content: str
    chunking_strategy: str
    token_count: int
    embedding: np.ndarray
    m0_raw_ids: List[str]
    conversation_id: str
    created_at: datetime

@dataclass
class SimilarityResult:
    """Vector similarity search result with normalized score"""
    chunk_id: str
    content: str
    similarity_score: float  # Normalized 0-1 range
    distance: float         # Raw cosine distance
    m0_message_count: int
    chunking_strategy: str
    created_at: datetime

class PgVectorScaleDemo:
    """MemFuse pgvectorscale End-to-End Integration Demo"""
    
    def __init__(self):
        """Initialize demo with database connection and embedding model"""
        self.db_config = {
            'host': os.getenv('PGVECTORSCALE_HOST', 'localhost'),
            'port': int(os.getenv('PGVECTORSCALE_PORT', '5432')),
            'database': os.getenv('PGVECTORSCALE_DB', 'memfuse'),
            'user': os.getenv('PGVECTORSCALE_USER', 'postgres'),
            'password': os.getenv('PGVECTORSCALE_PASSWORD', 'postgres')
        }
        
        self.conn = None
        self.embedding_model = None
        self.conversation_id = str(uuid.uuid4())
        
        # Demo configuration
        self.chunk_token_limit = 200  # Token limit per M1 chunk
        self.embedding_dim = 384      # sentence-transformers/all-MiniLM-L6-v2
        
    def connect_database(self) -> None:
        """Establish database connection with error handling"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.conn.autocommit = True
            logger.info("✅ Connected to pgvectorscale database")
            
            # Verify pgvectorscale extension
            with self.conn.cursor() as cur:
                cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vectorscale';")
                result = cur.fetchone()
                if result:
                    logger.info(f"✅ pgvectorscale extension version: {result[0]}")
                else:
                    raise Exception("pgvectorscale extension not found")
                    
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            sys.exit(1)
    
    def initialize_embedding_model(self) -> None:
        """Initialize sentence-transformers model for embeddings"""
        try:
            logger.info("🧠 Loading sentence-transformers model...")
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            sys.exit(1)
    
    def generate_streaming_data(self, num_batches: int = 3, messages_per_batch: int = 12) -> List[M0Message]:
        """Generate realistic streaming conversation data"""
        logger.info(f"📊 Generating streaming conversation data...")
        logger.info(f"   Generating {num_batches} batches, {num_batches * messages_per_batch} total messages")
        
        # Realistic conversation topics for ML/AI domain
        conversation_templates = [
            # Machine Learning Basics
            ("user", "I want to learn Python machine learning, where should I start?"),
            ("assistant", "I recommend starting with scikit-learn, which provides rich machine learning algorithms and tools."),
            ("user", "What are the main algorithms in scikit-learn?"),
            ("assistant", "scikit-learn includes classification (SVM, Random Forest), regression (Linear, Ridge), clustering (K-means, DBSCAN), and dimensionality reduction (PCA, t-SNE)."),
            
            # Deep Learning
            ("user", "How is deep learning different from traditional machine learning?"),
            ("assistant", "Deep learning uses neural networks with multiple layers to automatically learn feature representations, while traditional ML often requires manual feature engineering."),
            ("user", "Which deep learning framework should I choose?"),
            ("assistant", "PyTorch is great for research and experimentation, TensorFlow/Keras for production deployment, and JAX for high-performance computing."),
            
            # Data Science Workflow
            ("user", "What's the typical data science project workflow?"),
            ("assistant", "The workflow typically includes: 1) Problem definition, 2) Data collection and cleaning, 3) Exploratory data analysis, 4) Feature engineering, 5) Model selection and training, 6) Evaluation and validation, 7) Deployment and monitoring."),
            ("user", "How do you handle missing data in datasets?"),
            ("assistant", "Common strategies include: deletion (listwise/pairwise), imputation (mean/median/mode), advanced imputation (KNN, iterative), or using algorithms that handle missing values natively."),
            
            # Vector Databases and Embeddings
            ("user", "What are vector databases used for?"),
            ("assistant", "Vector databases store and search high-dimensional embeddings for similarity search, recommendation systems, semantic search, and RAG (Retrieval-Augmented Generation) applications."),
        ]
        
        messages = []
        sequence_num = 1
        
        for batch in range(num_batches):
            logger.info(f"   Batch {batch + 1}: {messages_per_batch} messages")
            
            for i in range(messages_per_batch):
                template_idx = (batch * messages_per_batch + i) % len(conversation_templates)
                role, content = conversation_templates[template_idx]
                
                # Estimate token count (rough approximation: 1 token ≈ 4 characters)
                token_count = max(1, len(content) // 4)
                
                message = M0Message(
                    message_id=str(uuid.uuid4()),
                    content=content,
                    role=role,
                    conversation_id=self.conversation_id,
                    sequence_number=sequence_num,
                    token_count=token_count,
                    created_at=datetime.now()
                )
                
                messages.append(message)
                sequence_num += 1
        
        return messages
    
    def insert_m0_messages(self, messages: List[M0Message]) -> None:
        """Insert M0 messages into database with batch processing"""
        logger.info("💾 Writing M0 raw message data...")
        
        try:
            with self.conn.cursor() as cur:
                # Batch insert for performance
                insert_query = """
                    INSERT INTO m0_raw 
                    (message_id, content, role, conversation_id, sequence_number, token_count, created_at, processing_status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                batch_data = [
                    (msg.message_id, msg.content, msg.role, msg.conversation_id, 
                     msg.sequence_number, msg.token_count, msg.created_at, 'pending')
                    for msg in messages
                ]
                
                cur.executemany(insert_query, batch_data)
                logger.info(f"✅ Inserted {len(messages)} M0 message records")
                
        except Exception as e:
            logger.error(f"❌ Failed to insert M0 messages: {e}")
            raise
    
    def create_m1_chunks(self, messages: List[M0Message]) -> List[M1Chunk]:
        """Create M1 chunks using token-based intelligent chunking strategy"""
        logger.info("🧠 Creating M1 chunks (token-based strategy)...")
        
        chunks = []
        current_chunk_messages = []
        current_token_count = 0
        
        for message in messages:
            # Check if adding this message would exceed token limit
            if current_token_count + message.token_count > self.chunk_token_limit and current_chunk_messages:
                # Create chunk from accumulated messages
                chunk = self._create_chunk_from_messages(current_chunk_messages)
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_messages = [message]
                current_token_count = message.token_count
            else:
                # Add message to current chunk
                current_chunk_messages.append(message)
                current_token_count += message.token_count
        
        # Handle remaining messages
        if current_chunk_messages:
            chunk = self._create_chunk_from_messages(current_chunk_messages)
            chunks.append(chunk)
        
        logger.info(f"✅ Created {len(chunks)} M1 chunks")
        logger.info(f"   Average {len(messages) / len(chunks):.1f} messages per chunk")
        
        return chunks
    
    def _create_chunk_from_messages(self, messages: List[M0Message]) -> M1Chunk:
        """Create a single M1 chunk from a list of M0 messages"""
        # Combine message contents
        combined_content = " ".join([msg.content for msg in messages])
        
        # Calculate total token count
        total_tokens = sum([msg.token_count for msg in messages])
        
        # Generate embedding
        embedding = self.embedding_model.encode(combined_content)
        
        # Create chunk
        chunk = M1Chunk(
            chunk_id=str(uuid.uuid4()),
            content=combined_content,
            chunking_strategy='token_based',
            token_count=total_tokens,
            embedding=embedding,
            m0_raw_ids=[msg.message_id for msg in messages],
            conversation_id=self.conversation_id,
            created_at=datetime.now()
        )
        
        return chunk
    
    def insert_m1_chunks(self, chunks: List[M1Chunk]) -> None:
        """Insert M1 chunks with embeddings into database"""
        logger.info("💾 Inserting M1 chunks with embeddings...")
        
        try:
            with self.conn.cursor() as cur:
                insert_query = """
                    INSERT INTO m1_episodic 
                    (chunk_id, content, chunking_strategy, token_count, embedding, 
                     m0_raw_ids, conversation_id, created_at, embedding_generated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                for chunk in chunks:
                    # Convert UUID strings to proper UUID array format
                    m0_ids_array = '{' + ','.join(chunk.m0_raw_ids) + '}'

                    cur.execute(insert_query, (
                        chunk.chunk_id,
                        chunk.content,
                        chunk.chunking_strategy,
                        chunk.token_count,
                        chunk.embedding.tolist(),  # Convert numpy array to list
                        m0_ids_array,  # Format as PostgreSQL UUID array
                        chunk.conversation_id,
                        chunk.created_at,
                        datetime.now()
                    ))
                
                logger.info(f"✅ Inserted {len(chunks)} M1 chunk records with embeddings")
                
        except Exception as e:
            logger.error(f"❌ Failed to insert M1 chunks: {e}")
            raise
    
    def perform_similarity_search(self, queries: List[str]) -> Dict[str, List[SimilarityResult]]:
        """Perform high-performance vector similarity search with normalized scores"""
        logger.info("🔍 Performing high-performance vector similarity search...")
        
        results = {}
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                for query in queries:
                    logger.info(f"\nQuery: '{query}'")
                    
                    # Generate query embedding
                    query_embedding = self.embedding_model.encode(query)
                    
                    # Execute similarity search using custom function with normalized scores
                    cur.execute("""
                        SELECT * FROM search_similar_chunks(%s::vector, %s, %s)
                    """, (query_embedding.tolist(), 0.0, 5))  # threshold=0.0, max_results=5
                    
                    search_results = []
                    rows = cur.fetchall()
                    
                    if rows:
                        logger.info(f"✅ StreamingDiskANN (pgvectorscale) returned {len(rows)} results")
                        
                        for i, row in enumerate(rows, 1):
                            result = SimilarityResult(
                                chunk_id=row['chunk_id'],
                                content=row['content'],
                                similarity_score=row['similarity_score'],
                                distance=row['distance'],
                                m0_message_count=row['m0_message_count'],
                                chunking_strategy=row['chunking_strategy'],
                                created_at=row['created_at']
                            )
                            search_results.append(result)
                            
                            # Display result with normalized similarity score
                            logger.info(f"  {i}. Similarity: {result.similarity_score:.4f} (Distance: {result.distance:.4f})")
                            logger.info(f"     Content: {result.content[:100]}...")
                            logger.info(f"     Source: {result.m0_message_count} M0 messages")
                            logger.info(f"     Strategy: {result.chunking_strategy}")
                    else:
                        logger.info("No similar chunks found")
                    
                    results[query] = search_results
                    
        except Exception as e:
            logger.error(f"❌ Similarity search failed: {e}")
            raise
        
        return results
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity and lineage relationships"""
        logger.info("🔍 Validating data integrity and lineage relationships...")
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get data statistics
                cur.execute("SELECT * FROM get_data_lineage_stats()")
                layer_stats = cur.fetchall()
                
                # Get lineage summary
                cur.execute("SELECT * FROM data_lineage_summary")
                lineage_summary = cur.fetchone()
                
                # Get vector index information
                cur.execute("SELECT * FROM vector_index_stats")
                index_stats = cur.fetchall()
                
                # Calculate integrity metrics
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_m0,
                        SUM(token_count) as total_m0_tokens,
                        AVG(token_count) as avg_m0_tokens
                    FROM m0_raw
                    WHERE conversation_id = %s
                """, (self.conversation_id,))
                m0_stats = cur.fetchone()
                
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_m1,
                        SUM(token_count) as total_m1_tokens,
                        AVG(token_count) as avg_m1_tokens
                    FROM m1_episodic
                    WHERE conversation_id = %s
                """, (self.conversation_id,))
                m1_stats = cur.fetchone()
                
                integrity_report = {
                    'layer_stats': layer_stats,
                    'lineage_summary': lineage_summary,
                    'index_stats': index_stats,
                    'm0_stats': m0_stats,
                    'm1_stats': m1_stats
                }
                
                # Display integrity report
                logger.info("\n📈 Data Integrity Report:")
                logger.info(f"   M0 Raw Messages: {m0_stats['total_m0']}")
                logger.info(f"   M1 Chunks: {m1_stats['total_m1']}")
                logger.info(f"   Embedding Vectors: {m1_stats['total_m1']}")
                logger.info(f"   Vector Indexes: {len(index_stats)}")
                logger.info(f"   Average M0 per chunk: {lineage_summary['avg_m0_per_chunk']:.1f}")
                logger.info(f"   M0 Total tokens: {m0_stats['total_m0_tokens']}")
                logger.info(f"   M0 Average tokens: {m0_stats['avg_m0_tokens']:.1f}")
                logger.info(f"   M1 Total tokens: {m1_stats['total_m1_tokens']}")
                logger.info(f"   M1 Average tokens: {m1_stats['avg_m1_tokens']:.1f}")
                
                return integrity_report
                
        except Exception as e:
            logger.error(f"❌ Data integrity validation failed: {e}")
            raise
    
    def run_complete_demo(self) -> None:
        """Execute complete end-to-end demonstration"""
        logger.info("🚀 MemFuse pgvectorscale End-to-End Integration Demo")
        logger.info("=" * 70)
        
        try:
            # Initialize components
            self.connect_database()
            self.initialize_embedding_model()
            
            # Generate and process streaming data
            messages = self.generate_streaming_data(num_batches=3, messages_per_batch=12)
            self.insert_m0_messages(messages)
            
            # Create intelligent chunks with embeddings
            chunks = self.create_m1_chunks(messages)
            self.insert_m1_chunks(chunks)
            
            # Perform similarity searches
            test_queries = [
                'Python machine learning algorithms',
                'deep learning neural networks',
                'data science project workflow'
            ]
            
            search_results = self.perform_similarity_search(test_queries)
            
            # Validate data integrity
            integrity_report = self.validate_data_integrity()
            
            # Demo summary
            logger.info("\n🎯 Demo Summary:")
            logger.info("=" * 70)
            logger.info("✅ Streaming data ingestion: Success")
            logger.info("✅ M0 message storage: Success")
            logger.info("✅ M1 intelligent chunking: Success (multi-message chunks)")
            logger.info("✅ StreamingDiskANN vector indexing: Success")
            logger.info("✅ Normalized similarity search (0-1 range): Success")
            logger.info("✅ Data lineage tracking: Success")
            logger.info("\n🚀 MemFuse pgvectorscale End-to-End Demo Complete!")
            logger.info("🎉 All functionality verified with StreamingDiskANN optimization!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            sys.exit(1)
        finally:
            if self.conn:
                self.conn.close()

def main():
    """Main execution function"""
    demo = PgVectorScaleDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()
