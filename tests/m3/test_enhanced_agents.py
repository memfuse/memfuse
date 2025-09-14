"""Test cases for enhanced M3 agents."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from src.memfuse_core.m3.agents.websearch import WebSearchAgent
from src.memfuse_core.m3.agents.database import DatabaseQueryAgent
from src.memfuse_core.m3.agents.shell import ShellCommandAgent


class TestWebSearchAgent:
    """Test WebSearchAgent functionality."""
    
    @pytest.fixture
    def agent(self):
        return WebSearchAgent(timeout=5.0)
    
    @pytest.mark.asyncio
    async def test_duckduckgo_search(self, agent):
        """Test DuckDuckGo search functionality."""
        # Mock the HTTP response
        mock_response_data = {
            "RelatedTopics": [
                {
                    "Text": "Test result 1",
                    "FirstURL": "https://example.com/1",
                    "Result": "Test snippet 1"
                }
            ],
            "AbstractText": "Test abstract",
            "AbstractURL": "https://example.com/abstract",
            "Heading": "Test heading"
        }
        
        with patch.object(agent, '_get_session') as mock_session:
            mock_resp = AsyncMock()
            mock_resp.json.return_value = mock_response_data
            mock_resp.raise_for_status.return_value = None
            
            mock_session_obj = AsyncMock()
            mock_session_obj.get.return_value.__aenter__.return_value = mock_resp
            mock_session.return_value = mock_session_obj
            
            result = await agent._duckduckgo_search("test query")
            
            assert result["engine"] == "duckduckgo"
            assert len(result["results"]) >= 1
            assert "Test result 1" in result["results"][0]["title"]
    
    @pytest.mark.asyncio
    async def test_arxiv_search(self, agent):
        """Test arXiv search functionality."""
        # Mock XML response
        mock_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/2301.00001v1</id>
                <title>Test Paper Title</title>
                <summary>Test paper abstract</summary>
                <published>2023-01-01T00:00:00Z</published>
            </entry>
        </feed>"""
        
        with patch.object(agent, '_get_session') as mock_session:
            mock_resp = AsyncMock()
            mock_resp.text.return_value = mock_xml
            mock_resp.raise_for_status.return_value = None
            
            mock_session_obj = AsyncMock()
            mock_session_obj.get.return_value.__aenter__.return_value = mock_resp
            mock_session.return_value = mock_session_obj
            
            result = await agent._arxiv_search("test query")
            
            assert result["engine"] == "arxiv"
            assert len(result["results"]) == 1
            assert "Test Paper Title" in result["results"][0]["title"]
    
    @pytest.mark.asyncio
    async def test_execute_multi_source(self, agent):
        """Test execution with multiple sources."""
        payload = {
            "query": "machine learning",
            "sources": ["duckduckgo", "arxiv"],
            "max_results": 5
        }
        
        # Mock both search methods
        with patch.object(agent, '_duckduckgo_search') as mock_ddg, \
             patch.object(agent, '_arxiv_search') as mock_arxiv:
            
            mock_ddg.return_value = {
                "engine": "duckduckgo", 
                "results": [{"title": "DuckDuckGo result", "url": "http://example.com", "snippet": "test"}],
                "total": 1
            }
            mock_arxiv.return_value = {
                "engine": "arxiv",
                "results": [{"title": "arXiv result", "url": "http://arxiv.org", "snippet": "test"}],
                "total": 1
            }
            
            result = await agent.execute("test_session", payload)
            
            assert "results" in result
            assert len(result["results"]) == 2
            assert result["total_found"] == 2
            assert "duckduckgo" in result["sources_used"]
            assert "arxiv" in result["sources_used"]


class TestDatabaseQueryAgent:
    """Test DatabaseQueryAgent functionality."""
    
    @pytest.fixture
    def agent(self):
        return DatabaseQueryAgent()
    
    def test_nl_to_sql_conversion(self, agent):
        """Test natural language to SQL conversion."""
        # Mock LLM response
        agent.llm.completion_json = Mock(return_value="SELECT * FROM m0_raw LIMIT 10;")
        
        sql = agent._nl_to_sql("Show me recent messages")
        assert sql == "SELECT * FROM m0_raw LIMIT 10;"
    
    def test_sql_validation(self, agent):
        """Test SQL safety validation."""
        # Valid queries
        assert agent._validate_sql("SELECT * FROM m0_raw;")
        assert agent._validate_sql("select id, content from m1_episodic where id = 1;")
        
        # Invalid queries
        assert not agent._validate_sql("DROP TABLE m0_raw;")
        assert not agent._validate_sql("INSERT INTO m0_raw VALUES (1, 'test');")
        assert not agent._validate_sql("UPDATE m0_raw SET content = 'test';")
        assert not agent._validate_sql("")
    
    @pytest.mark.asyncio
    async def test_execute_success(self, agent):
        """Test successful query execution."""
        payload = {"request": "Show me recent messages"}
        
        # Mock LLM and database
        agent.llm.completion_json = Mock(return_value="SELECT id, content FROM m0_raw LIMIT 5;")
        
        with patch('src.memfuse_core.services.database_service.DatabaseService.get_instance') as mock_db:
            mock_db_instance = AsyncMock()
            mock_connection = AsyncMock()
            mock_cursor = AsyncMock()
            
            mock_cursor.fetchall.return_value = [
                {"id": 1, "content": "Test message 1"},
                {"id": 2, "content": "Test message 2"}
            ]
            mock_cursor.description = [("id",), ("content",)]
            
            mock_connection.cursor.return_value.__aenter__.return_value = mock_cursor
            mock_db_instance.get_connection.return_value.__aenter__.return_value = mock_connection
            mock_db.return_value = mock_db_instance
            
            result = await agent.execute("test_session", payload)
            
            assert "sql" in result
            assert "rows" in result
            assert len(result["rows"]) == 2
            assert result["row_count"] == 2


class TestShellCommandAgent:
    """Test ShellCommandAgent functionality."""
    
    @pytest.fixture
    def agent(self):
        agent = ShellCommandAgent()
        agent.enabled = True  # Enable for testing
        return agent
    
    @pytest.mark.asyncio
    async def test_ripgrep_execution(self, agent):
        """Test ripgrep command execution."""
        payload = {
            "cmd": "rg",
            "pattern": "test_pattern",
            "path": "/tmp",
            "max": 50
        }
        
        with patch('shutil.which', return_value="/usr/bin/rg"), \
             patch('asyncio.create_subprocess_exec') as mock_subprocess:
            
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"test_file:1:test_pattern found", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            result = await agent.execute("test_session", payload)
            
            assert result["exit_code"] == 0
            assert "test_pattern found" in result["output"]
            assert result["pattern"] == "test_pattern"
    
    @pytest.mark.asyncio
    async def test_echo_execution(self, agent):
        """Test echo command execution."""
        payload = {
            "cmd": "echo",
            "text": "Hello, World!"
        }
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"Hello, World!\n", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            result = await agent.execute("test_session", payload)
            
            assert result["exit_code"] == 0
            assert "Hello, World!" in result["output"]
    
    @pytest.mark.asyncio
    async def test_disabled_agent(self):
        """Test agent behavior when disabled."""
        agent = ShellCommandAgent()
        agent.enabled = False
        
        payload = {"cmd": "echo", "text": "test"}
        result = await agent.execute("test_session", payload)
        
        assert "error" in result
        assert "disabled" in result["error"]
    
    @pytest.mark.asyncio
    async def test_disallowed_command(self, agent):
        """Test behavior with disallowed commands."""
        payload = {"cmd": "rm", "args": ["-rf", "/"]}
        
        result = await agent.execute("test_session", payload)
        
        assert "error" in result
        assert "not allowed" in result["error"]