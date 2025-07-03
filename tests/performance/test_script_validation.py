#!/usr/bin/env python3
"""
Validation test for test_flush_trigger.py script.

This test validates that the performance test script has correct syntax
and can generate test data without requiring a running server.
"""

import sys
import os
import uuid

# Add the performance test directory to path
sys.path.insert(0, os.path.dirname(__file__))

from test_flush_trigger import generate_test_messages


def test_generate_test_messages():
    """Test the generate_test_messages function."""
    print("🧪 Testing generate_test_messages function...")
    
    # Test with different message counts
    test_counts = [1, 5, 10, 15]
    
    for count in test_counts:
        messages = generate_test_messages(count)
        
        # Validate message count
        assert len(messages) == count, f"Expected {count} messages, got {len(messages)}"
        
        # Validate message format
        for i, message in enumerate(messages):
            assert isinstance(message, str), f"Message {i} is not a string"
            assert f"[Message {i + 1}]" in message, f"Message {i} missing proper numbering"
            assert len(message) > 50, f"Message {i} too short: {len(message)} chars"
        
        print(f"   ✓ Generated {count} messages successfully")
    
    print("✅ generate_test_messages function works correctly")


def test_message_variety():
    """Test that generated messages have variety."""
    print("\n🧪 Testing message variety...")
    
    messages = generate_test_messages(15)
    
    # Check for different content types
    tech_keywords = ["algorithm", "system", "database", "architecture", "cloud"]
    conversation_keywords = ["project", "weather", "book", "meeting", "restaurant"]
    analytical_keywords = ["correlation", "analysis", "metrics", "feedback", "revenue"]
    
    tech_count = sum(1 for msg in messages if any(kw in msg.lower() for kw in tech_keywords))
    conv_count = sum(1 for msg in messages if any(kw in msg.lower() for kw in conversation_keywords))
    anal_count = sum(1 for msg in messages if any(kw in msg.lower() for kw in analytical_keywords))
    
    print(f"   Technical messages: {tech_count}")
    print(f"   Conversational messages: {conv_count}")
    print(f"   Analytical messages: {anal_count}")
    
    # Should have variety across different types
    assert tech_count > 0, "No technical messages found"
    assert conv_count > 0, "No conversational messages found"
    assert anal_count > 0, "No analytical messages found"
    
    print("✅ Message variety test passed")


def test_unique_messages():
    """Test that messages are unique."""
    print("\n🧪 Testing message uniqueness...")
    
    messages = generate_test_messages(10)
    unique_messages = set(messages)
    
    assert len(unique_messages) == len(messages), f"Duplicate messages found: {len(messages)} total, {len(unique_messages)} unique"
    
    print(f"   ✓ All {len(messages)} messages are unique")
    print("✅ Message uniqueness test passed")


def test_script_imports():
    """Test that the script can import all required modules."""
    print("\n🧪 Testing script imports...")
    
    try:
        import requests
        print("   ✓ requests module available")
    except ImportError:
        print("   ❌ requests module not available")
        return False
    
    try:
        import time
        import uuid
        import sys
        from typing import Dict, Any, List
        print("   ✓ All standard library modules available")
    except ImportError as e:
        print(f"   ❌ Standard library import failed: {e}")
        return False
    
    print("✅ All imports successful")
    return True


def main():
    """Run all validation tests."""
    print("🚀 MemFuse Performance Test Script Validation")
    print("=" * 60)
    
    try:
        # Test imports first
        if not test_script_imports():
            print("\n❌ Import tests failed")
            return False
        
        # Test message generation
        test_generate_test_messages()
        test_message_variety()
        test_unique_messages()
        
        print("\n🎯 Validation Summary:")
        print("   ✅ Script syntax is correct")
        print("   ✅ All imports are available")
        print("   ✅ Message generation works properly")
        print("   ✅ Messages have proper variety and uniqueness")
        print("\n✅ Performance test script is ready for use!")
        print("   To run the actual performance test:")
        print("   1. Start MemFuse server: poetry run memfuse-core")
        print("   2. Run test: python tests/performance/test_flush_trigger.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
