"""Unit tests for M2 LLM integration in SimplifiedMemoryService."""

import asyncio
import uuid
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from datetime import datetime

from src.memfuse_core.services.simplified_memory_service import SimplifiedMemoryService
from src.memfuse_core.models.core import Chunk, M2Status
from src.memfuse_core.models.m2_extraction import FactExtractionResponse, ExtractedFact
from src.memfuse_core.llm.base import LLMResponse, LLMUsage


@pytest.fixture
def memory_service():
    """Create a SimplifiedMemoryService instance for testing."""
    config = {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_memfuse',
            'user': 'postgres',
            'password': 'postgres'
        }
    }
    return SimplifiedMemoryService(cfg=config, user="test_user")


@pytest.fixture
def sample_chunk():
    """Create a sample chunk for testing."""
    return {
        'chunk_id': str(uuid.uuid4()),
        'content': 'User asked about Python programming. Assistant explained list comprehensions and their benefits.',
        'user_id': str(uuid.uuid4()),
        'session_id': str(uuid.uuid4()),
        'token_count': 15,
        'created_at': datetime.now(),
        'm0_raw_ids': [str(uuid.uuid4())],
        'chunking_strategy': 'token_based',
        'metadata': {}
    }


@pytest.fixture
def sample_context_chunks():
    """Create sample context chunks for testing."""
    return [
        Chunk(
            chunk_id=str(uuid.uuid4()),
            content="Previous discussion about programming languages.",
            token_count=10,
            user_id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
            created_at=datetime.now(),
            m2_status=M2Status.COMPLETED,
            chunking_strategy="token_based",
            m0_raw_ids=[],
            metadata={}
        ),
        Chunk(
            chunk_id=str(uuid.uuid4()),
            content="User mentioned they are learning software development.",
            token_count=8,
            user_id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
            created_at=datetime.now(),
            m2_status=M2Status.COMPLETED,
            chunking_strategy="token_based",
            m0_raw_ids=[],
            metadata={}
        )
    ]


class TestBuildFactExtractionPrompt:
    """Test the _build_fact_extraction_prompt method."""
    
    def test_build_prompt_with_context(self, memory_service, sample_chunk, sample_context_chunks):
        """Test building prompt with context chunks."""
        messages = memory_service._build_fact_extraction_prompt(sample_chunk, sample_context_chunks)
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Context Chunk 1" in messages[0]["content"]
        assert "Context Chunk 2" in messages[0]["content"]
        assert sample_chunk['content'] in messages[1]["content"]
    
    def test_build_prompt_without_context(self, memory_service, sample_chunk):
        """Test building prompt without context chunks."""
        messages = memory_service._build_fact_extraction_prompt(sample_chunk, [])
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "No additional context chunks available" in messages[0]["content"]
    
    @patch('src.memfuse_core.services.simplified_memory_service.PromptManager')
    def test_build_prompt_template_error_fallback(self, mock_prompt_manager, memory_service, sample_chunk):
        """Test fallback when prompt template fails."""
        mock_prompt_manager.get_prompt.side_effect = Exception("Template error")
        
        messages = memory_service._build_fact_extraction_prompt(sample_chunk, [])
        
        assert len(messages) == 2
        assert "Extract semantic facts" in messages[1]["content"]
        assert sample_chunk['content'] in messages[1]["content"]


class TestLLMProviderIntegration:
    """Test LLM provider initialization and usage."""
    
    @patch.dict('os.environ', {'XAI_API_KEY': 'test-key'})
    @patch('src.memfuse_core.services.simplified_memory_service.OpenAIProvider')
    async def test_get_llm_provider_with_xai_key(self, mock_openai_provider, memory_service):
        """Test getting OpenAI provider with XAI API key."""
        mock_provider_instance = AsyncMock()
        mock_openai_provider.return_value = mock_provider_instance
        
        provider = await memory_service._get_llm_provider()
        
        assert provider is mock_provider_instance
        mock_openai_provider.assert_called_once()
        call_args = mock_openai_provider.call_args[0][0]
        assert call_args["api_key"] == "test-key"
        assert call_args["base_url"] == "https://api.x.ai/v1"
    
    @patch.dict('os.environ', {}, clear=True)
    @patch('src.memfuse_core.services.simplified_memory_service.MockProvider')
    async def test_get_llm_provider_fallback_to_mock(self, mock_provider, memory_service):
        """Test fallback to mock provider when no API key available."""
        mock_provider_instance = AsyncMock()
        mock_provider.return_value = mock_provider_instance
        
        provider = await memory_service._get_llm_provider()
        
        assert provider is mock_provider_instance
        mock_provider.assert_called_once()
    
    def test_get_preferred_extraction_model_openai(self, memory_service):
        """Test model selection with OpenAI API key."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            model = memory_service._get_preferred_extraction_model()
            assert model == "gpt-4o-2024-08-06"
    
    def test_get_preferred_extraction_model_xai_fallback(self, memory_service):
        """Test model selection fallback to XAI."""
        with patch.dict('os.environ', {}, clear=True):
            model = memory_service._get_preferred_extraction_model()
            assert model == "grok-3-mini"


class TestFactExtraction:
    """Test fact extraction methods."""
    
    def test_parse_structured_response(self, memory_service):
        """Test parsing structured JSON response."""
        json_content = '''
        {
            "facts": [
                {"content": "User is learning Python programming", "source_chunk_ids": ["123"]},
                {"content": "List comprehensions are a Python feature", "source_chunk_ids": ["123"]}
            ]
        }
        '''
        
        facts = memory_service._parse_fact_extraction_response(json_content)
        
        assert len(facts) == 2
        assert "User is learning Python programming" in facts
        assert "List comprehensions are a Python feature" in facts
    
    def test_parse_malformed_json_fallback(self, memory_service):
        """Test fallback parsing for malformed JSON."""
        content = '''
        Here are the extracted facts:
        - User asked about Python programming
        - Assistant explained list comprehensions
        - List comprehensions provide concise syntax
        '''
        
        facts = memory_service._parse_fact_extraction_response(content)
        
        assert len(facts) >= 1
        # Should extract meaningful lines
        assert any("Python programming" in fact for fact in facts)

    def test_parse_json_with_processing_notes_and_empty_facts(self, memory_service):
        """JSON with empty facts and processing_notes should yield no facts (avoid key leakage)."""
        content = '{"processing_notes": "no extractable facts", "facts": []}'
        facts = memory_service._parse_fact_extraction_response(content)
        assert facts == []
    
    def test_parse_empty_response(self, memory_service):
        """Test handling of empty response."""
        facts = memory_service._parse_fact_extraction_response("")
        assert facts == []
    
    @pytest.mark.asyncio
    async def test_extract_facts_with_retry_structured_success(self, memory_service):
        """Test successful structured fact extraction."""
        # Mock LLM provider with structured output support
        mock_provider = AsyncMock()
        mock_provider.generate_structured = AsyncMock()
        
        # Create mock structured response
        mock_extracted_facts = [
            ExtractedFact(
                content="User is learning Python programming",
                source_chunk_ids=["test-chunk-123"]
            ),
            ExtractedFact(
                content="Assistant explained list comprehensions",
                source_chunk_ids=["test-chunk-123"]
            )
        ]
        
        mock_response_data = FactExtractionResponse(facts=mock_extracted_facts)
        mock_response = LLMResponse(
            content='{"facts": [...]}',
            model="gpt-4o-2024-08-06",
            usage=LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            success=True,
            parsed_data=mock_response_data
        )
        
        mock_provider.generate_structured.return_value = mock_response
        
        from src.memfuse_core.llm.base import LLMRequest
        request = LLMRequest(
            messages=[{"role": "user", "content": "Extract facts"}],
            model="gpt-4o-2024-08-06"
        )
        
        facts = await memory_service._extract_facts_with_retry(mock_provider, request)
        
        assert len(facts) == 2
        assert "User is learning Python programming" in facts
        assert "Assistant explained list comprehensions" in facts
        mock_provider.generate_structured.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_facts_with_retry_json_fallback(self, memory_service):
        """Test JSON fallback when structured extraction fails."""
        # Mock LLM provider
        mock_provider = AsyncMock()
        mock_provider.generate_structured = AsyncMock(side_effect=Exception("Structured parsing failed"))
        mock_provider.generate = AsyncMock()
        
        # Mock JSON response
        json_content = '{"facts": [{"content": "User asked about Python"}, {"content": "Assistant provided help"}]}'
        mock_response = LLMResponse(
            content=json_content,
            model="grok-3-mini",
            usage=LLMUsage(),
            success=True
        )
        mock_provider.generate.return_value = mock_response
        
        from src.memfuse_core.llm.base import LLMRequest
        request = LLMRequest(
            messages=[{"role": "user", "content": "Extract facts"}],
            model="grok-3-mini"
        )
        
        facts = await memory_service._extract_facts_with_retry(mock_provider, request)
        
        assert len(facts) == 2
        assert "User asked about Python" in facts
        assert "Assistant provided help" in facts
        mock_provider.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_facts_with_retry_all_failures(self, memory_service):
        """Test handling when all retry attempts fail."""
        # Mock LLM provider that always fails
        mock_provider = AsyncMock()
        mock_provider.generate_structured = AsyncMock(side_effect=Exception("API Error"))
        mock_provider.generate = AsyncMock(side_effect=Exception("API Error"))
        
        from src.memfuse_core.llm.base import LLMRequest
        request = LLMRequest(
            messages=[{"role": "user", "content": "Extract facts"}],
            model="gpt-4o"
        )
        
        facts = await memory_service._extract_facts_with_retry(mock_provider, request, max_retries=2)
        
        assert facts == []


class TestIntegrationWithM2Workflow:
    """Test integration with existing M2 processing workflow."""
    
    @pytest.mark.asyncio
    async def test_extract_list_of_fact_content_from_chunk_full_flow(self, memory_service, sample_chunk):
        """Test the full fact extraction flow."""
        chunk_id = sample_chunk['chunk_id']
        
        # Mock the database methods
        memory_service._get_m1_chunk = AsyncMock(return_value=sample_chunk)
        memory_service._get_session_context_for_chunk = AsyncMock(return_value=[])
        
        # Mock LLM provider
        mock_provider = AsyncMock()
        mock_provider.generate_structured = AsyncMock()
        
        mock_extracted_facts = [
            ExtractedFact(
                content="User asked about Python programming",
                source_chunk_ids=[chunk_id]
            )
        ]
        
        mock_response_data = FactExtractionResponse(facts=mock_extracted_facts)
        mock_response = LLMResponse(
            content='{"facts": [...]}',
            model="gpt-4o",
            usage=LLMUsage(),
            success=True,
            parsed_data=mock_response_data
        )
        
        mock_provider.generate_structured.return_value = mock_response
        memory_service._get_llm_provider = AsyncMock(return_value=mock_provider)
        
        # Execute the method
        facts = await memory_service._extract_list_of_fact_content_from_chunk(chunk_id)
        
        # Verify results
        assert len(facts) == 1
        assert "User asked about Python programming" in facts
        
        # Verify method calls
        memory_service._get_m1_chunk.assert_called_once_with(chunk_id, None)
        memory_service._get_session_context_for_chunk.assert_called_once()
        mock_provider.generate_structured.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_list_of_fact_content_chunk_not_found(self, memory_service):
        """Test handling when target chunk is not found."""
        chunk_id = str(uuid.uuid4())
        
        # Mock chunk not found
        memory_service._get_m1_chunk = AsyncMock(return_value=None)
        
        facts = await memory_service._extract_list_of_fact_content_from_chunk(chunk_id)
        
        assert facts == []
        memory_service._get_m1_chunk.assert_called_once_with(chunk_id, None)
    
    @pytest.mark.asyncio 
    async def test_extract_list_of_fact_content_no_llm_provider(self, memory_service, sample_chunk):
        """Test handling when LLM provider is not available."""
        chunk_id = sample_chunk['chunk_id']
        
        memory_service._get_m1_chunk = AsyncMock(return_value=sample_chunk)
        memory_service._get_llm_provider = AsyncMock(return_value=None)
        
        facts = await memory_service._extract_list_of_fact_content_from_chunk(chunk_id)
        
        assert facts == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
