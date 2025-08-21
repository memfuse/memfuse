# Manual Testing Scripts

This directory contains manual testing scripts for MemFuse buffer system functionality.

## Available Scripts

### `test_buffer_force_flush.py`
**Purpose**: Comprehensive test for buffer force flush functionality
- Tests timeout detection based on RoundBuffer write time
- Validates graceful shutdown with proper data transfer
- Ensures no duplicate clearing warnings

**Usage**:
```bash
cd /path/to/memfuse
poetry run python tests/manual/test_buffer_force_flush.py
```

**What it tests**:
- ✅ Correct timeout detection logic (only monitors RoundBuffer write time)
- ✅ Force flush timeout triggers after configured duration
- ✅ Graceful shutdown transfers data without warnings
- ✅ No data loss during shutdown process

### `check_database_state.py`
**Purpose**: Database state inspection utility
- Directly queries database tables to verify data persistence
- Useful for debugging and verifying flush operations
- Independent tool for database consistency checking

**Usage**:
```bash
cd /path/to/memfuse
poetry run python tests/manual/check_database_state.py
```

**What it shows**:
- Database connection status
- Table contents and row counts
- Data consistency verification
- Useful for post-flush validation

## Testing Workflow

1. **Start MemFuse server**:
   ```bash
   poetry run python scripts/memfuse_launcher.py
   ```

2. **Load test data** (in another terminal):
   ```bash
   cd ../memfuse-python
   poetry run python benchmarks/load_data.py msc --num-questions 3
   ```

3. **Run force flush tests**:
   ```bash
   poetry run python tests/manual/test_buffer_force_flush.py
   ```

4. **Check database state**:
   ```bash
   poetry run python tests/manual/check_database_state.py
   ```

5. **Test graceful shutdown** (Ctrl+C the server and observe logs)

## Expected Results

### Force Flush Test Output
```
🚀 Testing Force Flush Fixes
==================================================
🧪 Testing timeout detection logic...
Initial last write time: 0
Initial has pending data: False
Timeout triggered with no data: False
Add result: success
After add - last write time: 1724284857.123
After add - has pending data: True
RoundBuffer rounds: 1
Immediate timeout check: False
Waiting 4 seconds to test timeout detection...
Timeout triggered after 4 seconds: True
✅ Timeout detection test passed!

🧪 Testing shutdown behavior...
Added test data: success
RoundBuffer has 1 rounds
Initiating shutdown...
✅ Shutdown completed successfully!

🎉 All tests passed!
✅ Timeout detection only monitors RoundBuffer write time
✅ Shutdown doesn't produce duplicate clear warnings
```

### Server Logs (Graceful Shutdown)
```
WriteBuffer: Transferring 1 remaining rounds from RoundBuffer before shutdown
RoundBuffer: Transferring 1 rounds to HybridBuffer (reason: shutdown)
HybridBuffer: Receiving 1 rounds from RoundBuffer
WriteBuffer: RoundBuffer data transferred and cleared
```

**No warnings should appear** - specifically no "Buffer cleared without transfer" messages.

## Configuration

The force flush timeout can be configured in `config/buffer/default.yaml`:

```yaml
performance:
  force_flush_timeout: 1800  # 30 minutes in seconds
```

For testing, you can create a test configuration with shorter timeout:

```yaml
performance:
  force_flush_timeout: 5  # 5 seconds for testing
```
