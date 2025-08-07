#!/usr/bin/env python3
"""
Complete test for pgai auto embedding functionality.

This script tests:
1. Database cleanup and fresh start
2. Data insertion with immediate trigger
3. Embedding generation verification
4. Query functionality with embeddings
"""

import asyncio
import aiohttp
import json
import time
import psycopg
from typing import Dict, Any, List


class AutoEmbeddingTester:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "memfuse",
            "user": "postgres",
            "password": "postgres"
        }
        
    async def test_complete_flow(self):
        """Test complete auto embedding flow."""
        print("🧪 Starting complete auto embedding test...")
        
        # Step 1: Clean database
        await self._clean_database()
        
        # Step 2: Test server health
        if not await self._test_server_health():
            return False
            
        # Step 3: Insert test data
        test_data = await self._insert_test_data()
        if not test_data:
            return False
            
        # Step 4: Wait for auto embedding
        await self._wait_for_embeddings(test_data)
        
        # Step 5: Verify embeddings in database
        if not await self._verify_embeddings_in_db():
            return False
            
        # Step 6: Test query functionality
        if not await self._test_query_functionality():
            return False
            
        print("🎉 All auto embedding tests passed!")
        return True
        
    async def _clean_database(self):
        """Clean database for fresh test."""
        print("🧹 Cleaning database...")
        
        try:
            conn = await psycopg.AsyncConnection.connect(
                host=self.db_config["host"],
                port=self.db_config["port"],
                dbname=self.db_config["database"],
                user=self.db_config["user"],
                password=self.db_config["password"]
            )
            
            async with conn.cursor() as cur:
                # Clean m0_raw table
                await cur.execute("DELETE FROM m0_raw")
                await cur.execute("DELETE FROM m1_semantic")
                await cur.execute("DELETE FROM m2_relational")
                
            await conn.commit()
            await conn.close()
            
            print("✅ Database cleaned successfully")
            
        except Exception as e:
            print(f"❌ Database cleanup failed: {e}")
            
    async def _test_server_health(self) -> bool:
        """Test if server is responding."""
        print("🏥 Testing server health...")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Try different health endpoints
                endpoints = ["/", "/health", "/api/v1/health"]
                
                for endpoint in endpoints:
                    try:
                        async with session.get(f"{self.base_url}{endpoint}") as resp:
                            if resp.status in [200, 404]:  # 404 is OK, means server is running
                                print(f"✅ Server is responding (status: {resp.status})")
                                return True
                    except Exception:
                        continue
                        
                print("❌ Server is not responding")
                return False
                
        except Exception as e:
            print(f"❌ Server health check failed: {e}")
            return False
            
    async def _insert_test_data(self) -> List[Dict[str, Any]]:
        """Insert test data and return inserted records."""
        print("📝 Inserting test data...")
        
        test_messages = [
            {
                "user_id": "test_user_1",
                "session_id": "test_session_1",
                "messages": [
                    {"role": "user", "content": "What is machine learning?"},
                    {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."}
                ]
            },
            {
                "user_id": "test_user_2", 
                "session_id": "test_session_2",
                "messages": [
                    {"role": "user", "content": "Explain neural networks"},
                    {"role": "assistant", "content": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information."}
                ]
            }
        ]
        
        inserted_records = []
        
        try:
            async with aiohttp.ClientSession() as session:
                for test_data in test_messages:
                    async with session.post(
                        f"{self.base_url}/api/v1/memory/write",
                        json=test_data,
                        headers={"Content-Type": "application/json"}
                    ) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            inserted_records.append({
                                "user_id": test_data["user_id"],
                                "session_id": test_data["session_id"],
                                "content": test_data["messages"]
                            })
                            print(f"✅ Inserted data for {test_data['user_id']}")
                        else:
                            print(f"❌ Failed to insert data for {test_data['user_id']}: {resp.status}")
                            text = await resp.text()
                            print(f"Response: {text}")
                            
        except Exception as e:
            print(f"❌ Data insertion failed: {e}")
            
        return inserted_records
        
    async def _wait_for_embeddings(self, test_data: List[Dict[str, Any]]):
        """Wait for auto embeddings to be generated."""
        print("⏳ Waiting for auto embeddings to be generated...")
        
        # Wait a bit for immediate triggers to process
        await asyncio.sleep(3)
        
        print("✅ Waited for embedding processing")
        
    async def _verify_embeddings_in_db(self) -> bool:
        """Verify embeddings were generated in database."""
        print("🔍 Verifying embeddings in database...")
        
        try:
            conn = await psycopg.AsyncConnection.connect(
                host=self.db_config["host"],
                port=self.db_config["port"],
                dbname=self.db_config["database"],
                user=self.db_config["user"],
                password=self.db_config["password"]
            )
            
            async with conn.cursor() as cur:
                # Check m0_raw table for embeddings
                await cur.execute("""
                    SELECT id, content, embedding IS NOT NULL as has_embedding, needs_embedding
                    FROM m0_raw 
                    ORDER BY created_at DESC
                    LIMIT 10
                """)
                
                rows = await cur.fetchall()
                
                if not rows:
                    print("❌ No records found in m0_raw table")
                    await conn.close()
                    return False
                    
                embedded_count = 0
                total_count = len(rows)
                
                for row in rows:
                    record_id, content, has_embedding, needs_embedding = row
                    print(f"Record {record_id}: has_embedding={has_embedding}, needs_embedding={needs_embedding}")
                    print(f"  Content: {content[:50]}...")
                    
                    if has_embedding:
                        embedded_count += 1
                        
                await conn.close()
                
                print(f"📊 Embedding status: {embedded_count}/{total_count} records have embeddings")
                
                if embedded_count > 0:
                    print("✅ Auto embedding is working!")
                    return True
                else:
                    print("❌ No embeddings found - auto embedding may not be working")
                    return False
                    
        except Exception as e:
            print(f"❌ Database verification failed: {e}")
            return False
            
    async def _test_query_functionality(self) -> bool:
        """Test query functionality with embeddings."""
        print("🔎 Testing query functionality...")
        
        try:
            query_data = {
                "user_id": "test_user_1",
                "query": "machine learning artificial intelligence",
                "top_k": 5
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/v1/memory/query",
                    json=query_data,
                    headers={"Content-Type": "application/json"}
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        results = result.get("results", [])
                        
                        print(f"✅ Query successful: found {len(results)} results")
                        
                        for i, result in enumerate(results[:3]):
                            score = result.get("score", 0)
                            content = result.get("content", "")[:50]
                            print(f"  Result {i+1}: score={score:.3f}, content='{content}...'")
                            
                        return len(results) > 0
                    else:
                        print(f"❌ Query failed: {resp.status}")
                        text = await resp.text()
                        print(f"Response: {text}")
                        return False
                        
        except Exception as e:
            print(f"❌ Query test failed: {e}")
            return False


async def main():
    """Main test function."""
    tester = AutoEmbeddingTester()
    success = await tester.test_complete_flow()
    
    if success:
        print("\n🎉 All tests passed! Auto embedding is working correctly.")
    else:
        print("\n❌ Some tests failed. Check the logs above for details.")
        
    return success


if __name__ == "__main__":
    asyncio.run(main())
