# M3 Implementation Guide

This document summarizes the integrated M3 (Procedural Memory & Multi‑Agent Orchestration) design in this branch and its key changes versus the earlier feature branch docs.

## Gateway Wiring
- For ADD (write path): if a message carries `metadata.task` and `metadata.task_eos: true`, Gateway triggers `M3Processor` which invokes the Orchestrator and returns an M3‑shaped response. Regular message persistence proceeds and `message_workflows` gets logged.
- For QUERY (read path): standard transform pipeline runs (normalize → enrich metadata → compute scope → remove forbidden fields). When `m3.enable_query_guidance: true` and the request includes `metadata.task`, Gateway attaches compact `metadata.m3_guidance` per result summarizing reusable workflows/lessons (no change to top‑level data shape, only metadata extension).

## DB Schema (direct change, no backward compatibility)
- M0 `m0_raw`: now includes JSONB `metadata` to persist request metadata; GIN index added for metadata queries.
- M1 `m1_episodic`: now includes JSONB `metadata`; GIN index added.
- M2 `m2_semantic`: unchanged core fields (already had JSONB `metadata`).
- M3 tables: created lazily by `ProceduralStore` (`procedural_memory`, `procedural_lessons`, `message_workflows`).

## Response Contract
- `status`, `code`, `data`, `message`, `errors` are always present.
- `data` has only `results` and `total` (API layer strips internal echoes). Each result has required fields:
  - `id`, `relevance_score`, `memory_type`, `created_at`, `updated_at`, `metadata`.
  - Episodic: `content` exists, `fact` absent. Semantic: `fact={text, triples}` exists, `content` absent.
  - `metadata` contains `user_id`, `agent_id`, `session_id`, `session_name`, `scope`; `scope` derived from request `session_id`.
  - Forbidden: `level`, `retrieval`, `source`.
  - Renames: `score → relevance_score`, `type → memory_type`. `derived_from` is nested under metadata for M2.

## Configuration
See `config/m3/default.yaml`. In addition to reuse/learning toggles, this branch adds:
- `enable_query_guidance` (default false): guard query‑time guidance enrichment.

## Differences vs `feat/134-m3`
- Converged triggers: Orchestration only on write path (`task_eos`), optional guidance on read path.
- Schema tightened: mandatory metadata fields with scope logic; forbidden fields removed; renames enforced.
- M0/M1 persist request metadata, enabling lineage and future analysis.


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
