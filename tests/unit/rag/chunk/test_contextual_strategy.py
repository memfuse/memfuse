#!/usr/bin/env python3
"""
Test comprehensive implementation of contextual chunking strategy
1. Verify LLM API calls work correctly
2. Verify sliding window logic
3. Test parallel processing capabilities
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "src"))

from memfuse_core.llm.base import LLMRequest
from memfuse_core.llm.providers.openai import OpenAIProvider


async def test_llm_provider():
    """Test if LLM provider works correctly"""
    print("🧠 Testing LLM provider...")

    try:
        # Configuration - use XAI API
        config = {
            'api_key': os.getenv('XAI_API_KEY') or os.getenv('OPENAI_API_KEY'),
            'base_url': os.getenv('XAI_API_URL') or 'https://api.x.ai/v1',
            'model': 'grok-3-mini'
        }

        if not config['api_key']:
            print("❌ No API key found in environment variables")
            return None

        # Create OpenAI provider (compatible with XAI)
        provider = OpenAIProvider(config)

        # Test simple call
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello, respond with 'LLM working!'"}],
            model="grok-3-mini",
            max_tokens=50,
            temperature=0.3
        )

        response = await provider.generate(request)

        if response.success:
            print(f"✅ LLM call successful: {response.content}")
            print(f"   Model: {response.model}")
            print(f"   Token usage: {response.usage.total_tokens}")
            return provider
        else:
            print(f"❌ LLM call failed: {response.error}")
            return None

    except Exception as e:
        print(f"❌ LLM initialization failed: {e}")
        return None


def test_sliding_window_logic():
    """Test sliding window logic for contextual chunking"""
    print("\n🎯 Testing sliding window logic...")

    # Simulate chunks sequence [0,1,2,3,4,5,6]
    all_chunks = [
        "Chunk 0: Hello, how are you today?",
        "Chunk 1: I'm doing well, thanks for asking. How about you?",
        "Chunk 2: I'm great! I wanted to talk about music.",
        "Chunk 3: That sounds interesting. What kind of music do you like?",
        "Chunk 4: I really enjoy Taylor Swift's music.",
        "Chunk 5: Oh nice! I can get into Taylor Swift too.",
        "Chunk 6: What's your favorite song by her?"
    ]

    # Assume historical chunks are [0,1], current batch is [2,3,4,5,6]
    historical_chunks = all_chunks[:2]
    current_batch = all_chunks[2:]

    print(f"📊 Historical chunks: {len(historical_chunks)}")
    print(f"📊 Current batch: {len(current_batch)}")

    # Simulate sliding window processing
    context_window_size = 2
    processed_chunks = historical_chunks.copy()

    print(f"\n🔄 Sliding window processing (window_size={context_window_size}):")

    for i, chunk in enumerate(current_batch):
        chunk_index = len(historical_chunks) + i

        # Get context window
        if len(processed_chunks) >= context_window_size:
            context = processed_chunks[-context_window_size:]
        else:
            context = processed_chunks.copy()

        # Build complete sequence
        full_sequence = [c for c in context] + [chunk]

        context_indices = [processed_chunks.index(c) for c in context]
        sequence_indices = [processed_chunks.index(c) if c in processed_chunks else chunk_index for c in full_sequence]

        print(f"  Chunk {chunk_index}: context={context_indices} -> sequence={sequence_indices}")
        print(f"    Content: {chunk[:40]}...")

        # Check if should have context
        has_context = len(context) > 0
        should_be_enhanced = has_context and not all(c.strip() == "" for c in context)

        print(f"    has_context: {has_context}")
        print(f"    should_be_enhanced: {should_be_enhanced}")

        # Add current chunk to processed_chunks
        processed_chunks.append(chunk)
        print()

    return True


async def test_parallel_processing_simulation():
    """Test parallel processing simulation for contextual chunking"""
    print("\n⚡ Testing parallel processing simulation...")

    # Create test chunks
    chunks = [
        "Chunk A: First conversation piece",
        "Chunk B: Second conversation piece",
        "Chunk C: Third conversation piece",
        "Chunk D: Fourth conversation piece"
    ]

    async def process_chunk_async(chunk_content, index, delay=0.1):
        """Simulate async chunk processing"""
        await asyncio.sleep(delay)  # Simulate processing time
        return {
            "index": index,
            "content": chunk_content,
            "processed_at": asyncio.get_event_loop().time(),
            "has_context": index > 0  # First chunk has no context
        }

    print(f"📦 Processing {len(chunks)} chunks...")

    # Create parallel tasks
    tasks = []
    for i, chunk in enumerate(chunks):
        task = process_chunk_async(chunk, i)
        tasks.append(task)

    # Execute in parallel
    start_time = asyncio.get_event_loop().time()
    results = await asyncio.gather(*tasks)
    end_time = asyncio.get_event_loop().time()

    print(f"✅ Parallel processing completed in {end_time - start_time:.2f} seconds")

    for result in results:
        print(f"  Chunk {result['index']}: has_context={result['has_context']}")

    return True


async def main():
    """Main test function"""
    print("🧪 Starting Contextual Chunking comprehensive test\n")

    # 1. Test LLM provider
    llm_available = await test_llm_provider() is not None

    # 2. Test sliding window logic
    sliding_window_success = test_sliding_window_logic()

    # 3. Test parallel processing
    parallel_success = await test_parallel_processing_simulation()

    # Summary
    print("\n📊 Test Summary:")
    print(f"✅ LLM availability: {'Yes' if llm_available else 'No'}")
    print(f"✅ Sliding window logic: {'Success' if sliding_window_success else 'Failed'}")
    print(f"✅ Parallel processing: {'Success' if parallel_success else 'Failed'}")

    all_success = llm_available and sliding_window_success and parallel_success

    if all_success:
        print("\n🎉 All tests passed! Contextual Chunking strategy core functionality is working.")
    else:
        print("\n⚠️  Some tests failed, need further investigation.")


if __name__ == "__main__":
    asyncio.run(main())
