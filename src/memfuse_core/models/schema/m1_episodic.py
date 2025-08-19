"""M1 Episodic table schema definition."""

from typing import List
from .base import BaseSchema, ColumnDefinition, IndexDefinition, TriggerDefinition


class M1EpisodicSchema(BaseSchema):
    """Schema definition for M1 episodic memory table."""
    
    def get_table_name(self) -> str:
        return "m1_episodic"
    
    def define_columns(self) -> List[ColumnDefinition]:
        return [
            # Primary identification
            ColumnDefinition(
                name="chunk_id",
                data_type="UUID",
                primary_key=True,
                default="uuid_generate_v4()",
                comment="Unique identifier for the chunk"
            ),
            
            # Chunked content
            ColumnDefinition(
                name="content",
                data_type="TEXT",
                nullable=False,
                comment="Chunk content"
            ),
            
            # Chunking strategy metadata
            ColumnDefinition(
                name="chunking_strategy",
                data_type="VARCHAR(50)",
                nullable=False,
                default="'token_based'",
                check_constraint="chunking_strategy IN ('token_based', 'semantic', 'conversation_turn', 'hybrid')",
                comment="Strategy used for chunking"
            ),
            
            # Token analysis
            ColumnDefinition(
                name="token_count",
                data_type="INTEGER",
                nullable=False,
                default="0",
                comment="Number of tokens in the chunk"
            ),
            
            # Vector embedding (384 dimensions for sentence-transformers/all-MiniLM-L6-v2)
            ColumnDefinition(
                name="embedding",
                data_type="vector(384)",
                nullable=True,
                comment="Vector embedding of the chunk"
            ),
            
            # M0 lineage tracking
            ColumnDefinition(
                name="m0_raw_ids",
                data_type="UUID[]",
                nullable=False,
                default="'{}'",
                comment="Array of M0 message IDs that contributed to this chunk"
            ),
            ColumnDefinition(
                name="user_id",
                data_type="UUID",
                nullable=False,
                comment="User identifier"
            ),
            ColumnDefinition(
                name="session_id",
                data_type="UUID",
                nullable=False,
                comment="Session identifier"
            ),
            
            # Temporal tracking
            ColumnDefinition(
                name="created_at",
                data_type="TIMESTAMP WITH TIME ZONE",
                nullable=False,
                default="CURRENT_TIMESTAMP",
                comment="Chunk creation timestamp"
            ),
            ColumnDefinition(
                name="updated_at",
                data_type="TIMESTAMP WITH TIME ZONE",
                nullable=False,
                default="CURRENT_TIMESTAMP",
                comment="Chunk last update timestamp"
            ),
            ColumnDefinition(
                name="embedding_generated_at",
                data_type="TIMESTAMP WITH TIME ZONE",
                nullable=True,
                comment="Embedding generation timestamp"
            ),
            
            # Metadata
            ColumnDefinition(
                name="metadata",
                data_type="JSONB",
                nullable=False,
                default="'{}'::jsonb",
                comment="Additional metadata for the chunk"
            ),

            # Embedding processing status
            ColumnDefinition(
                name="needs_embedding",
                data_type="BOOLEAN",
                nullable=False,
                default="TRUE",
                comment="Whether the chunk needs embedding generation"
            ),

            # Quality metrics
            ColumnDefinition(
                name="embedding_model",
                data_type="VARCHAR(100)",
                nullable=False,
                default="'sentence-transformers/all-MiniLM-L6-v2'",
                comment="Model used for embedding generation"
            ),
            ColumnDefinition(
                name="chunk_quality_score",
                data_type="FLOAT",
                nullable=False,
                default="0.0",
                comment="Quality score of the chunk"
            ),

            # Additional metadata
            ColumnDefinition(
                name="metadata",
                data_type="JSONB",
                nullable=False,
                default="'{}'::jsonb",
                comment="Additional metadata for the chunk"
            ),
        ]
    
    def define_indexes(self) -> List[IndexDefinition]:
        return [
            # StreamingDiskANN index for optimal vector similarity search
            IndexDefinition(
                name="idx_m1_embedding_diskann",
                columns=["embedding vector_cosine_ops"],
                index_type="diskann",
                with_options={
                    "storage_layout": "memory_optimized",
                    "num_neighbors": 50,
                    "search_list_size": 100,
                    "max_alpha": 1.2,
                    "num_dimensions": 384,
                    "num_bits_per_dimension": 2
                },
                comment="StreamingDiskANN index for vector similarity search"
            ),
            
            # Additional indexes for M1 layer performance
            IndexDefinition(
                name="idx_m1_session_id",
                columns=["session_id"],
                comment="Index for session-based queries"
            ),
            IndexDefinition(
                name="idx_m1_user_id",
                columns=["user_id"],
                comment="Index for user-based queries"
            ),
            IndexDefinition(
                name="idx_m1_chunking_strategy",
                columns=["chunking_strategy"],
                comment="Index for strategy-based queries"
            ),
            IndexDefinition(
                name="idx_m1_created_at",
                columns=["created_at DESC"],
                comment="Index for temporal ordering"
            ),
            IndexDefinition(
                name="idx_m1_token_count",
                columns=["token_count"],
                comment="Index for token count queries"
            ),
            IndexDefinition(
                name="idx_m1_chunk_quality_score",
                columns=["chunk_quality_score"],
                comment="Index for quality score queries"
            ),

            # Index for embedding processing status
            IndexDefinition(
                name="idx_m1_needs_embedding",
                columns=["needs_embedding"],
                where_clause="needs_embedding = TRUE",
                comment="Index for pending embedding generation"
            ),

            # GIN index for M0 message ID arrays (lineage queries)
            IndexDefinition(
                name="idx_m1_m0_raw_ids_gin",
                columns=["m0_raw_ids"],
                index_type="gin",
                comment="GIN index for M0 lineage queries"
            ),

            # GIN index for metadata queries
            IndexDefinition(
                name="idx_m1_metadata_gin",
                columns=["metadata"],
                index_type="gin",
                comment="GIN index for metadata queries"
            ),
        ]
    
    def define_triggers(self) -> List[TriggerDefinition]:
        return [
            TriggerDefinition(
                name="m1_episodic_updated_at_trigger",
                timing="BEFORE",
                event="UPDATE",
                function_name="update_m1_episodic_updated_at"
            )
        ]
    
    def define_constraints(self) -> List[str]:
        return [
            "CONSTRAINT m1_chunks_m0_lineage_not_empty CHECK (array_length(m0_raw_ids, 1) > 0)"
        ]
    
    def get_trigger_functions_sql(self) -> List[str]:
        """Get SQL for trigger functions."""
        return [
            """
CREATE OR REPLACE FUNCTION update_m1_episodic_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
            """.strip()
        ]
