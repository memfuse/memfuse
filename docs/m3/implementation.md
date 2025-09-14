# M3 Implementation Guide

This document describes the complete M3 (Procedural Memory & Multi-Agent Orchestration) implementation that has been ported from the feat/134-m3 branch.

## Overview

M3 adds procedural memory and multi-agent orchestration capabilities to MemFuse, enabling:

1. **Task-based workflow orchestration** - Automatic decomposition of complex tasks into agent steps
2. **Procedural memory reuse** - Learning from successful workflows and reusing them for similar tasks  
3. **Lesson learning** - Capturing and applying lessons from both successful and failed executions
4. **Task-specific experience retrieval** - Querying experiences and lessons for specific tasks

## Architecture

### Core Components

#### 1. ProceduralStore (`src/memfuse_core/procedural/store.py`)
- Manages three main tables:
  - `message_workflows`: Links messages to M3 workflows
  - `procedural_memory`: Stores successful workflows for reuse
  - `procedural_lessons`: Stores lessons learned from executions
- Provides vector similarity search for workflow and lesson matching
- Handles workflow usage tracking and statistics

#### 2. Orchestrator (`src/memfuse_core/m3/orchestrator.py`)
- Main M3 workflow handler
- Implements workflow reuse logic with similarity thresholds
- Coordinates between Planner, AgentExecutor, and ProceduralStore
- Handles planning, execution, and lesson storage

#### 3. AgentExecutor (`src/memfuse_core/m3/executor.py`) 
- Executes workflow steps using available agents
- Manages context passing between steps
- Handles error recovery and logging

#### 4. Available Agents
- **RAGQueryAgent**: Uses RAG service to answer queries
- **ReportGenerationAgent**: Generates reports from data using LLM

### Database Schema

The M3 system uses three main tables:

```sql
-- Links messages to M3 workflows
CREATE TABLE message_workflows (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    workflow_id TEXT,
    step_index INT,
    tags TEXT[],
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE
);

-- Stores successful workflows for reuse
CREATE TABLE procedural_memory (
    workflow_id TEXT PRIMARY KEY,
    trigger_embedding VECTOR(384),
    trigger_pattern TEXT,
    successful_workflow JSONB NOT NULL,
    usage_count INT DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE
);

-- Stores lessons learned from executions
CREATE TABLE procedural_lessons (
    lesson_id TEXT PRIMARY KEY,
    trigger_embedding VECTOR(384),
    goal_text TEXT,
    agent TEXT,
    status TEXT CHECK (status IN ('success', 'fail')),
    error TEXT,
    fix_summary TEXT,
    working_params JSONB,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE
);
```

## API Integration

### 1. Message API Integration

The M3 system integrates with the message API to detect `task_eos` metadata:

```python
# Example message that triggers M3
{
    "role": "user",
    "content": "Complete the research analysis",
    "metadata": {
        "task": "research_analysis", 
        "task_eos": true
    }
}
```

When `task_eos: true` is detected:
1. System retrieves all messages for that task
2. Triggers M3 Orchestrator with task-scoped message history
3. Returns M3 workflow result in the response

### 2. Query API for Task-Specific Retrieval

New query endpoints for M3 functionality:

- `POST /query/experiences` - Query task-specific experiences and lessons
- `POST /query/workflows` - Query similar workflows for reuse  
- `GET /query/tasks` - List all tasks with experience statistics

### 3. Gateway Integration

The API Gateway includes M3 processing in the request pipeline:

1. **M3 Trigger Detection**: Checks for M3 metadata (`task_eos`, `workflow_name`, etc.)
2. **M3 Processing**: Routes M3 requests to the Orchestrator
3. **Response Enrichment**: Adds M3 metadata to regular query responses

## Configuration

M3 behavior is controlled through configuration files and environment variables:

### Configuration File (`config/m3/default.yaml`)

```yaml
m3:
  workflow_reuse_threshold: 0.9
  max_workflow_reuse_candidates: 5
  default_agent_timeout: 300
  max_agent_retries: 3
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  enable_workflow_reuse: true
  enable_lesson_learning: true
```

### Environment Variables

- `M3_WORKFLOW_REUSE_THRESHOLD`: Similarity threshold for workflow reuse
- `M3_AGENT_TIMEOUT`: Default timeout for agent execution
- `M3_EMBEDDING_MODEL`: Model used for embeddings
- `M3_ENABLE_WORKFLOW_REUSE`: Enable/disable workflow reuse
- `M3_ENABLE_LESSON_LEARNING`: Enable/disable lesson learning

## Usage Examples

### 1. Triggering M3 Workflows

```python
# Via message API with task_eos
await add_messages(
    session_id="session123",
    messages=[{
        "role": "user",
        "content": "Analyze market trends and create report",
        "metadata": {
            "task": "market_analysis", 
            "task_eos": True
        }
    }]
)

# Via gateway with workflow_name
await gateway.process_request({
    "user_id": "user123",
    "query": "Generate quarterly report",
    "metadata": {
        "workflow_name": "quarterly_reporting"
    }
})
```

### 2. Querying Task Experiences

```python
# Query experiences for a specific task
response = await query_task_experiences(
    session_id="session123",
    request={
        "task_name": "market_analysis",
        "query_text": "quarterly trends",
        "limit": 10
    }
)

# Response includes experiences and lessons
{
    "task_name": "market_analysis",
    "experiences": [...],
    "lessons": [...]
}
```

### 3. Workflow Reuse

When a similar task is encountered:

1. System computes embedding for the new task
2. Searches `procedural_memory` for similar workflows
3. If similarity > threshold (default 0.9), reuses existing workflow
4. Otherwise, creates new workflow plan

## Database Migration

### Running Migrations

```bash
# Check migration status
python scripts/migrate_m3.py status

# Apply all pending migrations
python scripts/migrate_m3.py migrate

# Validate schema
python scripts/migrate_m3.py validate
```

### Programmatic Migration

```python
from src.memfuse_core.migrations.m3 import initialize_m3_schema

# Initialize M3 schema
success = await initialize_m3_schema()
```

## Testing

Comprehensive tests cover all M3 components:

- **Unit Tests**: Individual component testing (`tests/m3/test_*.py`)
- **Integration Tests**: API integration tests (`tests/m3/test_api_integration.py`) 
- **End-to-End Tests**: Complete workflow tests (`tests/m3/test_e2e_workflow.py`)
- **Gateway Tests**: Gateway integration tests (`tests/m3/test_gateway_integration.py`)

```bash
# Run all M3 tests
pytest tests/m3/

# Run specific test categories
pytest tests/m3/test_orchestrator.py
pytest tests/m3/test_e2e_workflow.py
```

## Monitoring and Debugging

### Logging

M3 components provide detailed logging at various levels:

```python
# Enable debug logging for M3
import logging
logging.getLogger("src.memfuse_core.m3").setLevel(logging.DEBUG)
```

### Workflow Tracking

Each workflow execution creates debug files in the `runs/` directory:
- `input.json`: Original request
- `plan.json`: Generated workflow plan  
- `reflection.json`: Execution summary
- `report.txt`: Final result

### Database Views

M3 includes helpful database views for monitoring:

```sql
-- View workflow statistics
SELECT * FROM workflow_statistics;

-- View task performance metrics  
SELECT * FROM task_performance_metrics;
```

## Performance Considerations

### Vector Similarity Search

- Uses pgvector with DISKANN indexes for fast similarity search
- Falls back to IVFFLAT or sequential scan if DISKANN unavailable
- Embedding dimension: 384 (configurable)

### Workflow Reuse

- Similarity threshold balances reuse vs. accuracy (default 0.9)
- Usage count tracking helps identify popular workflows
- Configurable candidate limits prevent excessive search

### Lesson Learning

- Lessons stored for both success and failure cases
- Agent-specific lesson filtering improves relevance
- Configurable history limits prevent unbounded growth

## Troubleshooting

### Common Issues

1. **Vector Extension Missing**
   ```bash
   # Install pgvector extension
   CREATE EXTENSION vector;
   ```

2. **Migration Failures**
   ```bash
   # Check migration status
   python scripts/migrate_m3.py status
   
   # Validate schema
   python scripts/migrate_m3.py validate
   ```

3. **Workflow Not Triggering**
   - Verify `task_eos: true` in message metadata
   - Check M3 configuration is enabled
   - Review logs for trigger detection

4. **Poor Workflow Reuse**
   - Adjust `workflow_reuse_threshold` in config
   - Check embedding quality and similarity scores
   - Verify sufficient historical workflows exist

### Debug Commands

```python
# Check M3 configuration
from src.memfuse_core.m3.config import get_m3_config
config = get_m3_config()
print(config.to_dict())

# Validate M3 schema
from src.memfuse_core.migrations.m3 import validate_m3_schema
validation = await validate_m3_schema()
print(validation)

# Check workflow statistics
from src.memfuse_core.procedural.store import ProceduralStore
store = ProceduralStore()
stats = await store.get_task_statistics()
print(stats)
```

## Future Enhancements

The current implementation provides Phase A functionality. Future phases may include:

- **Phase B**: Advanced workflow composition and branching
- **Enhanced Agents**: More specialized agents for different domains
- **Workflow Optimization**: Automatic workflow improvement based on performance
- **Multi-Modal Support**: Integration with vision and audio processing
- **Distributed Execution**: Parallel and remote agent execution

## Contributing

When contributing to M3:

1. Run all tests: `pytest tests/m3/`
2. Update configuration if adding new parameters  
3. Add migration scripts for schema changes
4. Update this documentation for new features
5. Follow existing patterns for logging and error handling