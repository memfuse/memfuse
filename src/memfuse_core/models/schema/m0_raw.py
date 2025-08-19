"""M0 Raw table schema definition."""

from typing import List
from .base import BaseSchema, ColumnDefinition, IndexDefinition, TriggerDefinition


class M0RawSchema(BaseSchema):
    """Schema definition for M0 raw messages table."""
    
    def get_table_name(self) -> str:
        return "m0_raw"
    
    def define_columns(self) -> List[ColumnDefinition]:
        return [
            # Primary identification
            ColumnDefinition(
                name="message_id",
                data_type="UUID",
                primary_key=True,
                default="uuid_generate_v4()",
                comment="Unique identifier for the message"
            ),
            
            # Content and metadata
            ColumnDefinition(
                name="content",
                data_type="TEXT",
                nullable=False,
                comment="Message content"
            ),
            ColumnDefinition(
                name="role",
                data_type="VARCHAR(20)",
                nullable=False,
                check_constraint="role IN ('user', 'assistant', 'system')",
                comment="Message role"
            ),

            # Streaming context
            ColumnDefinition(
                name="session_id",
                data_type="UUID",
                nullable=False,
                comment="Session identifier"
            ),
            ColumnDefinition(
                name="sequence_number",
                data_type="INTEGER",
                nullable=False,
                comment="Message sequence number in conversation"
            ),
            
            # Token analysis for chunking decisions
            ColumnDefinition(
                name="token_count",
                data_type="INTEGER",
                nullable=False,
                default="0",
                comment="Number of tokens in the message"
            ),
            
            # Temporal tracking
            ColumnDefinition(
                name="created_at",
                data_type="TIMESTAMP WITH TIME ZONE",
                nullable=False,
                default="CURRENT_TIMESTAMP",
                comment="Message creation timestamp"
            ),
            ColumnDefinition(
                name="updated_at",
                data_type="TIMESTAMP WITH TIME ZONE",
                nullable=False,
                default="CURRENT_TIMESTAMP",
                comment="Message last update timestamp"
            ),
            ColumnDefinition(
                name="processed_at",
                data_type="TIMESTAMP WITH TIME ZONE",
                nullable=True,
                comment="Message processing timestamp"
            ),
            
            # Processing status
            ColumnDefinition(
                name="processing_status",
                data_type="VARCHAR(20)",
                nullable=False,
                default="'pending'",
                check_constraint="processing_status IN ('pending', 'processing', 'completed', 'failed')",
                comment="Processing status"
            ),
            
            # Lineage tracking
            ColumnDefinition(
                name="chunk_assignments",
                data_type="UUID[]",
                nullable=False,
                default="'{}'",
                comment="Array of chunk IDs this message is assigned to"
            ),
        ]
    
    def define_indexes(self) -> List[IndexDefinition]:
        return [
            # Performance optimization for session queries
            IndexDefinition(
                name="idx_m0_session_sequence",
                columns=["session_id", "sequence_number"],
                unique=True,
                comment="Unique index for session sequence"
            ),

            # Index for processing status queries
            IndexDefinition(
                name="idx_m0_processing_status",
                columns=["processing_status"],
                where_clause="processing_status != 'completed'",
                comment="Index for pending/processing messages"
            ),
            
            # Index for temporal queries
            IndexDefinition(
                name="idx_m0_created_at",
                columns=["created_at DESC"],
                comment="Index for temporal ordering"
            ),
        ]
    
    def define_triggers(self) -> List[TriggerDefinition]:
        return [
            TriggerDefinition(
                name="m0_raw_updated_at_trigger",
                timing="BEFORE",
                event="UPDATE",
                function_name="update_m0_raw_updated_at"
            )
        ]
    
    def define_constraints(self) -> List[str]:
        return [
            "CONSTRAINT unique_session_sequence UNIQUE (session_id, sequence_number)"
        ]
    
    def get_trigger_functions_sql(self) -> List[str]:
        """Get SQL for trigger functions."""
        return [
            """
CREATE OR REPLACE FUNCTION update_m0_raw_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
            """.strip()
        ]
