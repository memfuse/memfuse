#!/usr/bin/env python3
"""
Test LLM API calling functionality, especially contextual chunking related features
Using XAI's grok-3-mini model
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from memfuse_core.rag.chunk.base import ChunkData
from memfuse_core.rag.chunk.contextual import ContextualChunkStrategy
from memfuse_core.llm.base import LLMRequest, LLMResponse
from memfuse_core.llm.providers.openai import OpenAIProvider
from memfuse_core.llm.prompts.manager import get_prompt_manager


async def test_xai_llm_provider():
    """Test XAI LLM provider functionality"""
    print("🧠 Testing XAI LLM provider functionality...")

    try:
        # Configuration - use XAI settings
        xai_api_key = os.getenv('XAI_API_KEY') or os.getenv('OPENAI_API_KEY')
        xai_base_url = os.getenv('XAI_API_URL') or os.getenv('OPENAI_BASE_URL')

        if not xai_api_key:
            print("❌ XAI_API_KEY or OPENAI_API_KEY environment variable not found")
            return None

        print(f"🔑 Using API Key: {xai_api_key[:10]}...")
        print(f"🌐 Using Base URL: {xai_base_url}")

        config = {
            'api_key': xai_api_key,
            'base_url': xai_base_url,
            'model': 'grok-3-mini'
        }

        # Create OpenAI provider (compatible with XAI)
        provider = OpenAIProvider(config)

        # Test simple call
        request = LLMRequest(
            messages=[{"role": "user", "content": "Hello, respond with exactly 'LLM working!'"}],
            model="grok-3-mini",
            max_tokens=50,
            temperature=0.3
        )

        print("📡 Sending test request...")
        response = await provider.generate(request)

        if response.success:
            print(f"✅ XAI LLM call successful: {response.content}")
            print(f"   Model: {response.model}")
            print(f"   Token usage: {response.usage.total_tokens}")
            return provider
        else:
            print(f"❌ XAI LLM call failed: {response.error}")
            return None

    except Exception as e:
        print(f"❌ XAI LLM initialization failed: {e}")
        return None


async def test_contextual_description_with_xai(llm_provider):
    """测试使用XAI的contextual description生成"""
    print("\n🔍 测试XAI Contextual Description生成...")
    
    if not llm_provider:
        print("❌ 无法测试 - LLM提供者不可用")
        return False
    
    try:
        # Create test data
        context_chunks = [
            ChunkData(
                content="[ASSISTANT]: Hi! How are you doing tonight?\n[USER]: I'm doing great. Just relaxing with my two dogs.",
                metadata={"chunk_id": "chunk_0", "session_id": "test"}
            ),
            ChunkData(
                content="[ASSISTANT]: Great. In my spare time I do volunteer work.\n[USER]: That's neat. What kind of volunteer work do you do?",
                metadata={"chunk_id": "chunk_1", "session_id": "test"}
            )
        ]
        
        current_chunk = "[ASSISTANT]: I work in a homeless shelter in my town.\n[USER]: Good for you. Do you like vintage cars? I've two older mustangs."
        
        print(f"📄 当前chunk: {current_chunk[:50]}...")
        print(f"🔗 上下文chunks: {len(context_chunks)}")
        
        # 使用prompt管理器生成prompt
        prompt_manager = get_prompt_manager()
        
        past_messages = "\n\n--- Previous Chunk ---\n\n".join([
            chunk.content for chunk in context_chunks
        ])
        
        prompt = prompt_manager.get_prompt(
            "contextual_chunking",
            past_messages=past_messages,
            current_messages=current_chunk,
            chunk_content=current_chunk
        )
        
        print(f"📝 生成的prompt长度: {len(prompt)}")
        
        # 创建LLM请求 - 使用grok-3-mini
        request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            model="grok-3-mini",
            max_tokens=150,
            temperature=0.3
        )
        
        print("📡 发送contextual description请求...")
        # 生成响应
        response = await llm_provider.generate(request)
        
        if response.success:
            print(f"✅ XAI Contextual description生成成功:")
            print(f"   内容: {response.content}")
            print(f"   Token使用: {response.usage.total_tokens}")
            return True
        else:
            print(f"❌ XAI Contextual description生成失败: {response.error}")
            return False
            
    except Exception as e:
        print(f"❌ 生成过程出错: {e}")
        return False


async def test_contextual_strategy_with_xai(llm_provider):
    """Test ContextualChunkStrategy integration with XAI"""
    print("\n🚀 Testing ContextualChunkStrategy integration with XAI...")

    if not llm_provider:
        print("❌ Unable to test - LLM provider unavailable")
        return False

    try:
        # Create strategy instance - using grok-3-mini
        strategy = ContextualChunkStrategy(
            enable_contextual=True,
            context_window_size=2,
            llm_provider=llm_provider,
            gpt_model="grok-3-mini"
        )
        
        print(f"✅ Strategy created successfully:")
        print(f"   enable_contextual: {strategy.enable_contextual}")
        print(f"   context_window_size: {strategy.context_window_size}")
        print(f"   llm_provider: {strategy.llm_provider is not None}")
        print(f"   gpt_model: {strategy.gpt_model}")

        # Test contextual description generation method
        context_chunks = [
            ChunkData(
                content="[ASSISTANT]: Hi! How are you doing tonight?\n[USER]: I'm doing great. Just relaxing with my two dogs.",
                metadata={"chunk_id": "chunk_0", "session_id": "test"}
            )
        ]
        
        current_chunk = "[ASSISTANT]: Great. In my spare time I do volunteer work.\n[USER]: That's neat. What kind of volunteer work do you do?"
        
        print(f"\n📄 Testing _generate_contextual_description method...")
        print("📡 Calling XAI API...")

        description = await strategy._generate_contextual_description(
            current_chunk, context_chunks
        )

        print(f"✅ XAI method call successful:")
        print(f"   Description: {description}")

        return True

    except Exception as e:
        print(f"❌ Strategy test failed: {e}")
        return False


async def test_create_enhanced_chunk_with_xai(llm_provider):
    """Test _create_enhanced_chunk_async method using XAI"""
    print("\n🔧 Testing XAI _create_enhanced_chunk_async method...")

    if not llm_provider:
        print("❌ Cannot test - LLM provider unavailable")
        return False

    try:
        # Create strategy instance
        strategy = ContextualChunkStrategy(
            enable_contextual=True,
            context_window_size=2,
            llm_provider=llm_provider,
            gpt_model="grok-3-mini"
        )
        
        # Create test data
        previous_chunks = [
            ChunkData(
                content="[ASSISTANT]: Hi! How are you doing tonight?\n[USER]: I'm doing great. Just relaxing with my two dogs.",
                metadata={"chunk_id": "chunk_0", "session_id": "test"}
            )
        ]

        chunk_content = "[ASSISTANT]: Great. In my spare time I do volunteer work.\n[USER]: That's neat. What kind of volunteer work do you do?"

        print(f"📄 Testing chunk content: {chunk_content[:50]}...")
        print(f"🔗 Historical chunks: {len(previous_chunks)}")
        print("📡 Calling XAI API to generate enhanced chunk...")

        # Call method
        enhanced_chunk = await strategy._create_enhanced_chunk_async(
            chunk_content, 0, previous_chunks, "test_session"
        )
        
        print(f"✅ XAI Enhanced chunk创建成功:")
        print(f"   has_context: {enhanced_chunk.metadata.get('has_context')}")
        print(f"   gpt_enhanced: {enhanced_chunk.metadata.get('gpt_enhanced')}")
        print(f"   context_window_size: {enhanced_chunk.metadata.get('context_window_size')}")
        print(f"   context_chunk_ids: {enhanced_chunk.metadata.get('context_chunk_ids')}")
        
        if enhanced_chunk.metadata.get('contextual_description'):
            desc = enhanced_chunk.metadata['contextual_description']
            print(f"   contextual_description: {desc}")
        
        return True
        
    except Exception as e:
        print(f"❌ XAI Enhanced chunk创建失败: {e}")
        return False


async def main():
    """Main test function"""
    print("🧪 Starting XAI LLM Contextual Chunking comprehensive test\n")
    print(f"📁 Current working directory: {os.getcwd()}")
    print(f"🔍 Checking environment variables...")
    print(f"   XAI_API_KEY: {'Set' if os.getenv('XAI_API_KEY') else 'Not set'}")
    print(f"   OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
    print(f"   XAI_API_URL: {os.getenv('XAI_API_URL', 'Not set')}")
    print()

    # 1. Test XAI LLM provider
    llm_provider = await test_xai_llm_provider()

    # 2. Test prompt manager
    try:
        prompt_manager = get_prompt_manager()
        prompt_success = True
        print("✅ Prompt manager initialization successful")
    except Exception as e:
        prompt_success = False
        print(f"❌ Prompt manager initialization failed: {e}")

    # 3. Test XAI related functionality
    if llm_provider:
        description_success = await test_contextual_description_with_xai(llm_provider)
        strategy_success = await test_contextual_strategy_with_xai(llm_provider)
        enhanced_chunk_success = await test_create_enhanced_chunk_with_xai(llm_provider)
    else:
        print("\n⚠️  Skipping XAI LLM related tests - LLM provider unavailable")
        description_success = False
        strategy_success = False
        enhanced_chunk_success = False

    # Summary
    print("\n📊 Test Summary:")
    print(f"✅ XAI LLM Provider: {'Success' if llm_provider else 'Failed'}")
    print(f"✅ Prompt Manager: {'Success' if prompt_success else 'Failed'}")
    print(f"✅ XAI Contextual Description: {'Success' if description_success else 'Failed/Skipped'}")
    print(f"✅ XAI Strategy Integration: {'Success' if strategy_success else 'Failed/Skipped'}")
    print(f"✅ XAI Enhanced Chunk: {'Success' if enhanced_chunk_success else 'Failed/Skipped'}")

    all_success = all([
        llm_provider is not None,
        prompt_success,
        description_success,
        strategy_success,
        enhanced_chunk_success
    ])

    if all_success:
        print("\n🎉 All tests passed! XAI LLM Contextual Chunking functionality is fully operational.")
        print("🚀 现在可以运行端到端测试验证contextual chunks是否正确生成！")
    else:
        print("\n⚠️  部分测试失败，需要检查配置或网络连接。")


if __name__ == "__main__":
    asyncio.run(main())
