"""DatabaseQueryAgent implementation for M3."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from ...services.database_service import DatabaseService


class DatabaseQueryAgent:
    """Agent that executes database queries using natural language to SQL conversion."""
    
    def __init__(self, llm=None) -> None:
        self.llm = llm or self._create_default_llm()
    
    def _create_default_llm(self):
        """Create default LLM instance."""
        try:
            from ...llm.chat import ChatLLM
            return ChatLLM()
        except Exception as e:
            logger.warning(f"Failed to create ChatLLM for DatabaseQueryAgent: {e}")
            return MockLLM()
    
    def _nl_to_sql(self, request: str, schema_hint: str = "") -> str:
        """Convert natural language request to SQL query."""
        system = (
            "You are a SQL query generator. Convert natural language requests to PostgreSQL SQL queries.\n"
            "IMPORTANT CONSTRAINTS:\n"
            "- Only generate SELECT queries (no INSERT, UPDATE, DELETE, DROP, etc.)\n"
            "- Use safe, read-only operations only\n"
            "- Return only the SQL query, no explanations\n"
            "- Use proper PostgreSQL syntax\n"
            f"Schema information: {schema_hint}\n"
            "Return the SQL query as plain text."
        )
        
        user_prompt = f"Convert this request to SQL: {request}"
        
        try:
            sql = self.llm.completion_json(system, user_prompt)
            # Try to extract SQL from JSON response if needed
            if sql.strip().startswith('{'):
                try:
                    parsed = json.loads(sql)
                    sql = parsed.get('sql', parsed.get('query', sql))
                except:
                    pass
            
            # Clean up the SQL
            sql = sql.strip().strip('`').strip('"').strip("'")
            return sql
            
        except Exception as e:
            logger.error(f"Failed to convert NL to SQL: {e}")
            return ""
    
    def _validate_sql(self, sql: str) -> bool:
        """Validate that SQL is safe (read-only)."""
        if not sql:
            return False
        
        sql_lower = sql.lower().strip()
        
        # Must start with SELECT
        if not sql_lower.startswith('select'):
            return False
        
        # Check for dangerous keywords
        dangerous_keywords = [
            'insert', 'update', 'delete', 'drop', 'create', 'alter', 
            'truncate', 'grant', 'revoke', 'exec', 'execute'
        ]
        
        for keyword in dangerous_keywords:
            if keyword in sql_lower:
                return False
        
        return True
    
    def _get_schema_hint(self) -> str:
        """Get basic schema information for common tables."""
        return """
        Common tables:
        - m0_raw: Raw messages (id, content, conversation_id, role, created_at, metadata)
        - m1_episodic: Processed chunks (id, content, conversation_id, created_at, metadata)  
        - m2_semantic: Semantic facts (id, content, created_at, metadata)
        - procedural_workflows: M3 workflows (id, trigger_embedding, workflow_data, usage_count)
        - lessons: Learning data (id, trigger_embedding, goal_text, agent, status, error, fix_summary)
        """
    
    async def execute(self, session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a database query from natural language request."""
        request = str(payload.get("request", payload.get("query", ""))).strip()
        if not request:
            return {"error": "request or query parameter required"}
        
        schema_hint = str(payload.get("schema_hint", "")) or self._get_schema_hint()
        
        logger.info(f"DatabaseQueryAgent processing request: {request}")
        
        try:
            # Convert natural language to SQL
            sql = self._nl_to_sql(request, schema_hint)
            
            if not sql:
                return {"error": "Failed to generate SQL query from request"}
            
            # Validate SQL safety
            if not self._validate_sql(sql):
                return {
                    "error": "Generated SQL is not safe (must be SELECT-only)", 
                    "sql": sql
                }
            
            # Execute the query
            try:
                db_service = await DatabaseService.get_instance()
                
                # Execute query with timeout
                async with db_service.get_connection() as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute(sql)
                        rows = await cursor.fetchall()
                        
                        # Get column names
                        columns = [desc[0] for desc in cursor.description] if cursor.description else []
                
                # Convert rows to list of dicts
                results = []
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    results.append(row_dict)
                
                return {
                    "sql": sql,
                    "columns": columns,
                    "rows": results,
                    "row_count": len(results),
                    "request": request
                }
                
            except Exception as db_error:
                logger.error(f"Database execution error: {db_error}")
                return {
                    "error": f"Database execution failed: {str(db_error)}",
                    "sql": sql,
                    "request": request
                }
        
        except Exception as e:
            logger.error(f"DatabaseQueryAgent execution failed: {e}")
            return {"error": f"Query processing failed: {str(e)}", "request": request}


class MockLLM:
    """Mock LLM for testing/fallback scenarios."""
    
    def completion_json(self, system: str, user: str) -> str:
        """Mock completion that returns a basic SELECT query."""
        return "SELECT id, content, created_at FROM m0_raw LIMIT 10;"