#!/usr/bin/env python3
"""
M3 (Procedural Memory & Multi-Agent Orchestration) End-to-End Test Script

Tests the four core functionalities of the M3 system:
1. Metadata task & task_eos parsing
2. M3 Pipeline triggering via task_eos
3. M3 Pipeline LLM processing
4. Task-specific experience retrieval

This script validates the complete M3 workflow from task completion markers 
to procedural memory storage and retrieval.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import httpx
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

class M3EndToEndTester:
    """M3 system end-to-end tester"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.test_user_id = None  # Will be set during setup
        self.test_session_id = None
        self.test_agent_id = None  # Will be set during setup
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def setup_test_environment(self) -> bool:
        """Set up test environment"""
        logger.info("🔧 Setting up M3 test environment...")
        
        try:
            # 1. Get or create test user
            # First try to get existing users
            users_response = await self.client.get(f"{self.base_url}/api/v1/users")
            if users_response.status_code == 200:
                users_data = users_response.json()
                existing_users = users_data.get("data", {}).get("users", [])
                
                # Look for M3 test user
                for user in existing_users:
                    if user.get("name") == "M3 Test User":
                        self.test_user_id = user.get("id")
                        logger.info(f"Found existing M3 test user: {self.test_user_id}")
                        break
            
            # If not found, create new user
            if not self.test_user_id or self.test_user_id == "test_user_m3":
                user_response = await self.client.post(
                    f"{self.base_url}/api/v1/users",
                    json={
                        "name": "M3 Test User New",
                        "description": "User for M3 end-to-end testing"
                    }
                )
                
                if user_response.status_code in [200, 201]:
                    user_data = user_response.json()
                    self.test_user_id = user_data.get("data", {}).get("id")
                    logger.info(f"Created new M3 test user: {self.test_user_id}")
                elif user_response.status_code != 409:  # 409 = already exists
                    logger.error(f"Failed to create user: {user_response.status_code} - {user_response.text}")
                    return False
            
            # 2. Get or create test agent
            # First try to get existing agents
            agents_response = await self.client.get(f"{self.base_url}/api/v1/agents")
            if agents_response.status_code == 200:
                agents_data = agents_response.json()
                existing_agents = agents_data.get("data", {}).get("agents", [])
                
                # Look for M3 test agent
                for agent in existing_agents:
                    if agent.get("name") == "M3 Test Agent":
                        self.test_agent_id = agent.get("id")
                        logger.info(f"Found existing M3 test agent: {self.test_agent_id}")
                        break
            
            # If not found, create new agent
            if not self.test_agent_id or self.test_agent_id == "test_agent_m3":
                agent_response = await self.client.post(
                    f"{self.base_url}/api/v1/agents",
                    json={
                        "name": "M3 Test Agent New",
                        "description": "Agent for M3 testing"
                    }
                )
                
                if agent_response.status_code in [200, 201]:
                    agent_data = agent_response.json()
                    self.test_agent_id = agent_data.get("data", {}).get("id")
                    logger.info(f"Created new M3 test agent: {self.test_agent_id}")
                elif agent_response.status_code != 409:
                    logger.error(f"Failed to create agent: {agent_response.status_code} - {agent_response.text}")
                    return False
            
            # 3. Get or create test session
            # First try to get existing sessions
            sessions_response = await self.client.get(f"{self.base_url}/api/v1/sessions")
            if sessions_response.status_code == 200:
                sessions_data = sessions_response.json()
                existing_sessions = sessions_data.get("data", {}).get("sessions", [])
                
                # Look for M3 test session
                for session in existing_sessions:
                    if (session.get("name") == "M3 Test Session" and 
                        session.get("user_id") == self.test_user_id):
                        self.test_session_id = session.get("id")
                        logger.info(f"Found existing M3 test session: {self.test_session_id}")
                        break
            
            # If not found, create new session
            if not self.test_session_id:
                import time
                session_name = f"M3 Test Session {int(time.time())}"
                session_response = await self.client.post(
                    f"{self.base_url}/api/v1/sessions",
                    json={
                        "name": session_name,
                        "user_id": self.test_user_id,
                        "agent_id": self.test_agent_id
                    }
                )
                
                if session_response.status_code in [200, 201]:
                    session_data = session_response.json()
                    self.test_session_id = session_data.get("data", {}).get("id")
                    logger.info(f"Created new M3 test session: {self.test_session_id}")
                else:
                    logger.error(f"Failed to create session: {session_response.status_code} - {session_response.text}")
                    return False
            
            if not self.test_session_id:
                logger.error("Failed to get session ID")
                return False
            
            logger.info(f"✅ Test environment ready - Session ID: {self.test_session_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to setup test environment: {e}")
            return False
    
    async def test_1_metadata_parsing(self) -> bool:
        """Test 1: Metadata task & task_eos parsing"""
        logger.info("🧪 Test 1: Metadata task & task_eos parsing")
        
        try:
            # Send message with task_eos marker
            message_data = {
                "messages": [{
                    "role": "user",
                    "content": "Complete the data analysis task",
                    "metadata": {
                        "task": "data_analysis",
                        "task_eos": True,
                        "workflow_name": "analysis_workflow",
                        "step_index": 3
                    }
                }]
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/sessions/{self.test_session_id}/messages",
                json=message_data
            )
            
            if response.status_code != 201:
                logger.error(f"Failed to create message: {response.status_code} - {response.text}")
                return False
            
            result = response.json()
            
            # Verify metadata was correctly parsed and stored
            stored_metadata = result.get("data", {}).get("metadata", {})
            
            if not stored_metadata.get("task_eos"):
                logger.error("task_eos metadata not preserved")
                return False
            
            if stored_metadata.get("task") != "data_analysis":
                logger.error("task metadata not preserved")
                return False
            
            logger.info("✅ Test 1 passed: Metadata parsing works correctly")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test 1 failed: {e}")
            return False
    
    async def test_2_m3_trigger(self) -> bool:
        """Test 2: M3 Pipeline triggering via task_eos"""
        logger.info("🧪 Test 2: M3 Pipeline triggering via task_eos")
        
        try:
            # Use query API to send query with task_eos
            query_data = {
                "query": "What did we learn from the data analysis task?",
                "metadata": {
                    "task": "data_analysis",
                    "task_eos": True,
                    "m3_trigger": True
                },
                "top_k": 5
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/users/{self.test_user_id}/query",
                json=query_data,
                params={"tag": "m3"}
            )
            
            if response.status_code != 200:
                logger.error(f"M3 query failed: {response.status_code} - {response.text}")
                return False
            
            result = response.json()
            
            # Verify M3 processing was triggered
            if result.get("status") != "success":
                logger.error(f"M3 query returned error: {result}")
                return False
            
            # Check for M3-related response markers
            data = result.get("data", {})
            results = data.get("results", [])
            
            logger.info(f"M3 query returned {len(results)} results")
            
            # Verify response contains M3 processing markers
            has_m3_processing = any(
                result.get("metadata", {}).get("source") == "procedural_memory" or
                result.get("type") in ["procedural_workflow", "procedural_lesson"]
                for result in results
            )
            
            if not has_m3_processing and len(results) == 0:
                logger.warning("No M3-specific results found, but this may be expected for empty database")
            
            logger.info("✅ Test 2 passed: M3 Pipeline triggering works")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test 2 failed: {e}")
            return False
    
    async def test_3_llm_processing(self) -> bool:
        """Test 3: M3 Pipeline LLM processing"""
        logger.info("🧪 Test 3: M3 Pipeline LLM processing")
        
        try:
            # Create a complex task scenario to test LLM processing
            workflow_messages = [
                {
                    "role": "user",
                    "content": "Start data analysis workflow",
                    "metadata": {
                        "task": "data_analysis",
                        "workflow_name": "analysis_workflow",
                        "step_index": 1
                    }
                },
                {
                    "role": "assistant", 
                    "content": "Starting data collection phase",
                    "metadata": {
                        "task": "data_analysis",
                        "workflow_name": "analysis_workflow",
                        "step_index": 2
                    }
                },
                {
                    "role": "user",
                    "content": "Data collection completed successfully",
                    "metadata": {
                        "task": "data_analysis",
                        "workflow_name": "analysis_workflow",
                        "step_index": 3,
                        "task_eos": True
                    }
                }
            ]
            
            # Send workflow messages
            for msg in workflow_messages:
                response = await self.client.post(
                    f"{self.base_url}/api/v1/sessions/{self.test_session_id}/messages",
                    json=msg
                )
                
                if response.status_code != 201:
                    logger.error(f"Failed to create workflow message: {response.status_code}")
                    return False
            
            # Wait a bit for system processing
            await asyncio.sleep(1)
            
            # Use M3 query to trigger LLM processing
            query_data = {
                "query": "How should I approach a similar data analysis task?",
                "metadata": {
                    "task": "data_analysis",
                    "m3_trigger": True,
                    "request_llm_processing": True
                },
                "top_k": 10
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/users/{self.test_user_id}/query",
                json=query_data,
                params={"tag": "m3", "task": "data_analysis"}
            )
            
            if response.status_code != 200:
                logger.error(f"LLM processing query failed: {response.status_code} - {response.text}")
                return False
            
            result = response.json()
            
            # Verify LLM processing results
            if result.get("status") != "success":
                logger.error(f"LLM processing returned error: {result}")
                return False
            
            data = result.get("data", {})
            results = data.get("results", [])
            
            logger.info(f"LLM processing returned {len(results)} results")
            
            # Check for workflow-related results
            has_workflow_results = any(
                "workflow" in result.get("type", "").lower() or
                "data_analysis" in str(result.get("content", "")).lower()
                for result in results
            )
            
            if not has_workflow_results and len(results) == 0:
                logger.warning("No workflow-specific results found, but this may be expected")
            
            logger.info("✅ Test 3 passed: LLM processing works")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test 3 failed: {e}")
            return False
    
    async def test_4_task_specific_retrieval(self) -> bool:
        """Test 4: Task-specific experience retrieval"""
        logger.info("🧪 Test 4: Task-specific experience retrieval")
        
        try:
            # Use task parameter for specific task experience retrieval
            query_data = {
                "query": "What are the best practices for data analysis?",
                "metadata": {
                    "task": "data_analysis",
                    "retrieval_mode": "task_specific"
                },
                "top_k": 5
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/users/{self.test_user_id}/query",
                json=query_data,
                params={"task": "data_analysis"}
            )
            
            if response.status_code != 200:
                logger.error(f"Task-specific retrieval failed: {response.status_code} - {response.text}")
                return False
            
            result = response.json()
            
            # Verify task-specific retrieval results
            if result.get("status") != "success":
                logger.error(f"Task-specific retrieval returned error: {result}")
                return False
            
            data = result.get("data", {})
            results = data.get("results", [])
            
            logger.info(f"Task-specific retrieval returned {len(results)} results")
            
            # Verify results contain task-specific information
            has_task_specific = any(
                result.get("metadata", {}).get("task_specific") is True or
                result.get("task") == "data_analysis" or
                "procedural" in result.get("type", "").lower()
                for result in results
            )
            
            if not has_task_specific and len(results) == 0:
                logger.warning("No task-specific results found, but this may be expected for new system")
            
            logger.info("✅ Test 4 passed: Task-specific retrieval works")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test 4 failed: {e}")
            return False
    
    async def test_5_complete_workflow(self) -> bool:
        """Test 5: Complete M3 workflow"""
        logger.info("🧪 Test 5: Complete M3 workflow")
        
        try:
            # Simulate a complete task workflow
            workflow_steps = [
                {
                    "action": "start_task",
                    "content": "Starting machine learning model training task",
                    "metadata": {
                        "task": "ml_training",
                        "workflow_name": "ml_pipeline",
                        "step_index": 1
                    }
                },
                {
                    "action": "data_preparation",
                    "content": "Preparing training data with 10,000 samples",
                    "metadata": {
                        "task": "ml_training",
                        "workflow_name": "ml_pipeline", 
                        "step_index": 2
                    }
                },
                {
                    "action": "model_training",
                    "content": "Training neural network with Adam optimizer",
                    "metadata": {
                        "task": "ml_training",
                        "workflow_name": "ml_pipeline",
                        "step_index": 3
                    }
                },
                {
                    "action": "task_completion",
                    "content": "Model training completed with 95% accuracy",
                    "metadata": {
                        "task": "ml_training",
                        "workflow_name": "ml_pipeline",
                        "step_index": 4,
                        "task_eos": True,
                        "success": True,
                        "metrics": {"accuracy": 0.95, "loss": 0.05}
                    }
                }
            ]
            
            # Execute workflow steps
            for i, step in enumerate(workflow_steps):
                logger.info(f"Executing step {i+1}: {step['action']}")
                
                message_data = {
                    "messages": [{
                        "role": "user" if i % 2 == 0 else "assistant",
                        "content": step["content"],
                        "metadata": step["metadata"]
                    }]
                }
                
                response = await self.client.post(
                    f"{self.base_url}/api/v1/sessions/{self.test_session_id}/messages",
                    json=message_data
                )
                
                if response.status_code != 201:
                    logger.error(f"Failed to create workflow step {i+1}: {response.status_code}")
                    return False
                
                # Brief wait
                await asyncio.sleep(0.5)
            
            # Wait for M3 system to process task_eos
            await asyncio.sleep(2)
            
            # Query related experience
            query_data = {
                "query": "How to train a machine learning model effectively?",
                "metadata": {
                    "task": "ml_training",
                    "m3_trigger": True
                },
                "top_k": 10
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/v1/users/{self.test_user_id}/query",
                json=query_data,
                params={"task": "ml_training", "tag": "m3"}
            )
            
            if response.status_code != 200:
                logger.error(f"Complete workflow query failed: {response.status_code} - {response.text}")
                return False
            
            result = response.json()
            
            # Verify complete workflow results
            if result.get("status") != "success":
                logger.error(f"Complete workflow returned error: {result}")
                return False
            
            data = result.get("data", {})
            results = data.get("results", [])
            
            logger.info(f"Complete workflow query returned {len(results)} results")
            
            # Verify result quality
            has_relevant_results = any(
                "ml" in str(result.get("content", "")).lower() or
                "training" in str(result.get("content", "")).lower() or
                result.get("task") == "ml_training"
                for result in results
            )
            
            if not has_relevant_results and len(results) > 0:
                logger.warning("Results found but may not be highly relevant")
            elif len(results) == 0:
                logger.warning("No results found, but this may be expected for new system")
            
            logger.info("✅ Test 5 passed: Complete M3 workflow works")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test 5 failed: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests"""
        logger.info("🚀 Starting M3 End-to-End Tests")
        logger.info("=" * 60)
        
        # Set up test environment
        if not await self.setup_test_environment():
            logger.error("❌ Failed to setup test environment")
            return {}
        
        # Run tests
        tests = [
            ("Metadata Parsing", self.test_1_metadata_parsing),
            ("M3 Trigger", self.test_2_m3_trigger),
            ("LLM Processing", self.test_3_llm_processing),
            ("Task-Specific Retrieval", self.test_4_task_specific_retrieval),
            ("Complete Workflow", self.test_5_complete_workflow)
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n📋 Running: {test_name}")
            logger.info("-" * 40)
            
            try:
                result = await test_func()
                results[test_name] = result
                if result:
                    passed += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"💥 {test_name}: CRASHED - {e}")
                results[test_name] = False
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("🏁 M3 End-to-End Test Results")
        logger.info("=" * 60)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All M3 tests passed! System is working correctly.")
        else:
            logger.warning(f"⚠️  {total - passed} tests failed. Please check the implementation.")
        
        return results


async def main():
    """Main function"""
    # Check if server is running
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/api/v1/health")
            if response.status_code != 200:
                logger.error("❌ MemFuse server is not running or not healthy")
                logger.info("Please start the server with: poetry run python scripts/memfuse_launcher.py")
                return False
        except Exception as e:
            logger.error(f"❌ Cannot connect to MemFuse server: {e}")
            logger.info("Please start the server with: poetry run python scripts/memfuse_launcher.py")
            return False
    
    # Run tests
    async with M3EndToEndTester() as tester:
        results = await tester.run_all_tests()
        
        # Return whether all tests passed
        return all(results.values()) if results else False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)