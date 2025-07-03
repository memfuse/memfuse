#!/usr/bin/env python3
"""
Verification script for logging encoding fix.

This script verifies that the logging service can properly handle UTF-8 characters.
"""

import asyncio
import sys
from pathlib import Path

# Add src path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from memfuse_core.services.logging_service import LoggingService
from omegaconf import DictConfig
from loguru import logger


async def verify_utf8_logging():
    """Verify UTF-8 logging functionality."""
    print("🧪 Verifying logging encoding fix...")

    # Initialize logging service
    logging_service = LoggingService()
    cfg = DictConfig({"server": {"debug": True}})

    result = await logging_service.initialize(cfg)
    if not result:
        print("❌ Logging service initialization failed")
        return False

    print("✅ Logging service initialized successfully")

    # Test key UTF-8 characters
    test_messages = [
        "用户 '张三' 登录成功",  # Chinese characters
        "🚀 系统启动 ✅",        # Emoji
        "Café ñoño",           # Accented characters
        "内存: 85% ⚠️",        # Mixed content
    ]

    print("📝 Recording test logs...")
    for msg in test_messages:
        logger.info(f"Test: {msg}")

    await asyncio.sleep(0.2)  # Wait for write

    # Verify log file
    log_file = Path("logs/memfuse_core.log")
    if not log_file.exists():
        print("❌ Log file does not exist")
        return False

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()

        # Check all test messages
        for msg in test_messages:
            if msg not in log_content:
                print(f"❌ Message not found: {msg}")
                return False

        print("✅ All UTF-8 characters recorded correctly")

    except UnicodeDecodeError as e:
        print(f"❌ UTF-8 reading failed: {e}")
        return False

    await logging_service.shutdown()
    return True


async def main():
    """Main function."""
    print("=" * 50)
    print("🔧 MemFuse Logging Encoding Fix Verification")
    print("=" * 50)

    if not Path("src/memfuse_core").exists():
        print("❌ Please run this script from MemFuse project root directory")
        return

    Path("logs").mkdir(exist_ok=True)

    try:
        success = await verify_utf8_logging()

        if success:
            print("\n🎉 Logging encoding fix verification successful!")
            print("✅ UTF-8 characters recorded correctly to log file")
            print("✅ Fix is effective, issue resolved")
        else:
            print("\n❌ Logging encoding fix verification failed")

    except Exception as e:
        print(f"\n❌ Verification error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
