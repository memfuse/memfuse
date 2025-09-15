"""Tests for M3 integration with the gateway pipeline."""

import pytest
import uuid
from unittest.mock import Mock, AsyncMock, patch

from src.memfuse_core.gateway.api_gateway import MemoryApiGateway
from src.memfuse_core.gateway.m3_processor import M3Processor
from src.memfuse_core.interfaces.gateway_interface import RequestContext, OperationType


@pytest.fixture
def mock_services():
    """Mock services for gateway testing."""
    mock_buffer = Mock()
    mock_buffer.query = AsyncMock(return_value={
        "status": "success",
        "data": {"results": [], "total": 0}
    })
    
    mock_db = Mock()
    mock_db.get_user.return_value = {"name": "test_user"}
    mock_db.get_agent.return_value = {"name": "test_agent"}
    mock_db.get_session.return_value = {"name": "test_session"}
    
    return mock_buffer, mock_db


@pytest.fixture
def gateway(mock_services):
    """Create gateway instance with mocked services."""
    mock_buffer, mock_db = mock_services
    return MemoryApiGateway(buffer_service=mock_buffer, db_service=mock_db)


class TestM3GatewayIntegration:
    """Test M3 integration with the gateway."""

    @pytest.mark.asyncio
    async def test_m3_trigger_detection(self, gateway):
        """Test that M3 triggers are properly detected."""
        # Request with task_eos metadata should trigger M3
        request_data = {
            "user_id": "test_user",
            "query": "Complete the research task",
            "metadata": {
                "task": "research_task",
                "task_eos": True
            }
        }
        
        # Mock M3 orchestrator
        with patch.object(gateway.m3_processor, 'process_m3_request') as mock_m3:
            mock_m3.return_value = {
                "status": "success",
                "code": 200,
                "data": {"m3_result": "Task completed"},
                "message": "M3 workflow completed",
                "errors": None
            }
            
            response = await gateway.process_request(request_data, OperationType.QUERY)
            
            # Verify M3 was triggered
            mock_m3.assert_called_once()
            assert response["status"] == "success"
            assert "m3_result" in response["data"]

    @pytest.mark.asyncio
    async def test_normal_query_without_m3(self, gateway):
        """Test normal query processing without M3 trigger."""
        request_data = {
            "user_id": "test_user",
            "query": "Regular query",
            "metadata": {}
        }
        
        with patch.object(gateway.m3_processor, 'process_m3_request') as mock_m3:
            response = await gateway.process_request(request_data, OperationType.QUERY)
            
            # Verify M3 was NOT triggered
            mock_m3.assert_not_called()
            assert response["status"] == "success"

    @pytest.mark.asyncio
    async def test_m3_workflow_name_trigger(self, gateway):
        """Test M3 trigger via workflow_name metadata."""
        request_data = {
            "user_id": "test_user",
            "query": "Execute workflow",
            "metadata": {
                "workflow_name": "data_analysis_workflow"
            }
        }
        
        with patch.object(gateway.m3_processor, 'process_m3_request') as mock_m3:
            mock_m3.return_value = {
                "status": "success",
                "code": 200,
                "data": {"m3_result": "Workflow executed"},
                "message": "M3 workflow completed",
                "errors": None
            }
            
            response = await gateway.process_request(request_data, OperationType.QUERY)
            
            # Verify M3 was triggered
            mock_m3.assert_called_once()
            assert response["status"] == "success"

    @pytest.mark.asyncio
    async def test_m3_response_enrichment(self, gateway):
        """Test M3 response enrichment for regular queries."""
        request_data = {
            "user_id": "test_user",
            "query": "Find documents",
            "metadata": {"task": "document_search"}
        }
        
        # Mock buffer service to return results with M3 metadata
        gateway.buffer_service.query.return_value = {
            "status": "success",
            "data": {
                "results": [
                    {
                        "id": "doc1",
                        "content": "Document content",
                        "metadata": {"workflow_id": "wf123"}
                    }
                ],
                "total": 1
            }
        }
        
        response = await gateway.process_request(request_data, OperationType.QUERY)
        
        # Verify response was enriched with M3 metadata
        assert response["status"] == "success"
        results = response["data"]["results"]
        assert len(results) == 1
        assert results[0]["metadata"]["m3_generated"] is True
        assert results[0]["metadata"]["task_context"] == "document_search"

    @pytest.mark.asyncio
    async def test_m3_error_handling(self, gateway):
        """Test M3 error handling in gateway."""
        request_data = {
            "user_id": "test_user",
            "query": "Trigger error",
            "metadata": {
                "task": "error_task",
                "task_eos": True
            }
        }
        
        # Mock M3 processor to raise exception
        with patch.object(gateway.m3_processor, 'process_m3_request') as mock_m3:
            mock_m3.side_effect = Exception("M3 processing failed")
            
            response = await gateway.process_request(request_data, OperationType.QUERY)
            
            # Should return error response but not crash
            assert response["status"] == "error"
            assert "M3 processing failed" in response["message"]

    def test_m3_metadata_extraction(self, gateway):
        """Test M3 metadata extraction."""
        request_data = {
            "query": "Test query",
            "metadata": {
                "task": "test_task",
                "workflow_name": "test_workflow",
                "task_eos": True,
                "reuse_threshold": 0.8
            }
        }
        
        m3_metadata = gateway.m3_metadata_extractor.extract_m3_metadata(request_data)
        
        assert m3_metadata["task"] == "test_task"
        assert m3_metadata["workflow_name"] == "test_workflow"
        assert m3_metadata["task_eos"] is True
        assert m3_metadata["reuse_threshold"] == 0.8
        
        assert gateway.m3_metadata_extractor.should_enable_m3_features(m3_metadata) is True

    def test_m3_trigger_conditions(self, gateway):
        """Test various M3 trigger conditions."""
        context = RequestContext(user_id="test_user")
        
        # Test task_eos trigger
        request1 = {"metadata": {"task_eos": True}}
        assert gateway.m3_processor.should_trigger_m3(request1, context) is True
        
        # Test explicit m3_trigger
        request2 = {"metadata": {"m3_trigger": True}}
        assert gateway.m3_processor.should_trigger_m3(request2, context) is True
        
        # Test workflow_name trigger
        request3 = {"metadata": {"workflow_name": "test_workflow"}}
        assert gateway.m3_processor.should_trigger_m3(request3, context) is True
        
        # Test no trigger
        request4 = {"metadata": {}}
        assert gateway.m3_processor.should_trigger_m3(request4, context) is False

    def test_m3_parameter_extraction(self, gateway):
        """Test M3 parameter extraction."""
        request_data = {
            "query": "Test query",
            "metadata": {
                "task": "test_task",
                "workflow_name": "test_workflow"
            },
            "messages": [{"role": "user", "content": "Hello"}]
        }
        
        context = RequestContext(
            user_id="user123",
            session_id="session456",
            agent_id="agent789"
        )
        
        params = gateway.m3_processor.extract_m3_params(request_data, context)
        
        assert params["task_name"] == "test_task"
        assert params["workflow_name"] == "test_workflow"
        assert params["user_goal"] == "Test query"
        assert params["session_id"] == "session456"
        assert params["user_id"] == "user123"
        assert params["agent_id"] == "agent789"
        assert len(params["messages"]) == 1


class TestM3Processor:
    """Test M3Processor directly."""

    @pytest.mark.asyncio
    async def test_m3_request_processing(self):
        """Test M3 request processing."""
        # Mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator.handle_request = AsyncMock(return_value="Workflow completed")
        mock_orchestrator.last_workflow_id = "wf123"
        mock_orchestrator.last_reused = True
        mock_orchestrator.last_plan_steps = []
        
        processor = M3Processor(orchestrator=mock_orchestrator)
        
        request_data = {
            "query": "Test workflow",
            "metadata": {"task": "test_task"}
        }
        
        context = RequestContext(
            user_id="user123",
            session_id="session456"
        )
        
        response = await processor.process_m3_request(request_data, context)
        
        assert response["status"] == "success"
        assert response["data"]["m3_result"] == "Workflow completed"
        assert response["data"]["workflow_id"] == "wf123"
        assert response["data"]["workflow_reused"] is True
        
        # Verify orchestrator was called correctly
        mock_orchestrator.handle_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_m3_request_error_handling(self):
        """Test M3 request error handling."""
        # Mock orchestrator to raise exception
        mock_orchestrator = Mock()
        mock_orchestrator.handle_request = AsyncMock(side_effect=Exception("Workflow failed"))
        
        processor = M3Processor(orchestrator=mock_orchestrator)
        
        request_data = {"query": "Test workflow"}
        context = RequestContext(user_id="user123")
        
        response = await processor.process_m3_request(request_data, context)
        
        assert response["status"] == "error"
        assert "Workflow failed" in response["message"]
        assert response["data"]["m3_result"] is None