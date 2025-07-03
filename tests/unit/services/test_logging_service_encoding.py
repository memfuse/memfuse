#!/usr/bin/env python3
"""
Test for logging service UTF-8 encoding fix.

This test verifies that the logging service properly handles UTF-8 encoding
to prevent log file corruption with non-ASCII characters.
"""

import asyncio
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import pytest

# Add src path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from memfuse_core.services.logging_service import LoggingService
from omegaconf import DictConfig
from loguru import logger


class TestLoggingServiceEncoding:
    """Test cases for logging service encoding functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.logs_dir = Path(self.temp_dir) / "logs"
        self.logs_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up test environment."""
        logger.remove()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @pytest.mark.asyncio
    async def test_logging_service_utf8_encoding(self):
        """Test that logging service properly configures UTF-8 encoding."""
        logging_service = LoggingService()

        with patch('memfuse_core.services.logging_service.logger') as mock_logger:
            cfg = DictConfig({"server": {"debug": True}})
            result = await logging_service.initialize(cfg)
            assert result is True

            # Verify logger.add was called with encoding parameter
            add_calls = mock_logger.add.call_args_list
            assert len(add_calls) >= 2  # console and file

            # Find file logging call and verify encoding
            file_call = next((call for call in add_calls
                            if call[0] and isinstance(call[0][0], str) and call[0][0].endswith('.log')), None)
            assert file_call is not None, "File logging call not found"

            _, kwargs = file_call
            assert kwargs.get('encoding') == 'utf-8'

    def test_utf8_characters_in_logs(self):
        """Test that UTF-8 characters are properly written to log files."""
        test_messages = [
            "Hello 世界",  # Chinese characters
            "Café ñoño",   # Spanish with accents
            "🚀 Rocket emoji",  # Emoji
            "Special chars: ©®™€£¥"  # Special symbols
        ]
        log_file = self.logs_dir / "test_utf8.log"

        # Configure logger with UTF-8 encoding
        logger.remove()
        logger.add(str(log_file), encoding="utf-8", format="{message}")

        # Write and verify test messages
        for message in test_messages:
            logger.info(message)

        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()

        for message in test_messages:
            assert message in log_content, f"Message '{message}' not found in log file"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
