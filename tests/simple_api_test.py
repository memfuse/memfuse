#!/usr/bin/env python3
"""
Simple API Test Script

Quick test to trigger the connection leak issue with minimal setup.

Usage:
    # Start MemFuse server first
    poetry run memfuse-core
    
    # In another terminal, run this test
    poetry run python tests/simple_api_test.py
"""

import requests
import time
import os
from loguru import logger

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("MEMFUSE_API_KEY", "test-api-key")

def test_api_calls():
    """Make multiple API calls to trigger connection leak."""
    headers = {"X-API-Key": API_KEY}
    
    # Create a test user
    logger.info("Creating test user...")
    user_response = requests.post(
        f"{API_BASE_URL}/api/v1/users",
        json={"name": f"test_user_{int(time.time())}"},
        headers=headers
    )
    
    if user_response.status_code != 201:
        logger.error(f"Failed to create user: {user_response.status_code} - {user_response.text}")
        return
    
    user_data = user_response.json()
    user_id = user_data["data"]["user"]["id"]
    logger.info(f"Created user with ID: {user_id}")
    
    # Make multiple query requests
    logger.info("Making multiple query requests...")
    for i in range(20):
        try:
            query_response = requests.post(
                f"{API_BASE_URL}/api/v1/users/{user_id}/query",
                json={
                    "query": f"test query {i}",
                    "top_k": 5,
                    "include_messages": True,
                    "include_knowledge": True
                },
                headers=headers,
                timeout=30
            )
            
            if query_response.status_code == 200:
                logger.info(f"Query {i+1}/20 successful")
            else:
                logger.error(f"Query {i+1}/20 failed: {query_response.status_code} - {query_response.text}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Query {i+1}/20 failed with exception: {e}")
        
        # Small delay between requests
        time.sleep(0.5)
    
    logger.info("Test completed. Check PostgreSQL connections with:")
    logger.info("poetry run python tests/connection_monitor.py")


if __name__ == "__main__":
    test_api_calls()
