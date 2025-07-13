#!/usr/bin/env python3
"""
MemFuse End-to-End Testing Script

This script performs comprehensive end-to-end testing of the MemFuse pgai environment
including database connectivity, vector operations, and basic MemFuse functionality.
"""

import sys
import time
import json
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
import traceback

# Configuration
POSTGRES_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'memfuse',
    'user': 'postgres',
    'password': 'postgres'
}

MEMFUSE_API_URL = 'http://localhost:8000'

def print_status(message, status="INFO"):
    """Print colored status messages"""
    colors = {
        "INFO": "\033[0;34m[INFO]\033[0m",
        "SUCCESS": "\033[0;32m[SUCCESS]\033[0m", 
        "WARNING": "\033[1;33m[WARNING]\033[0m",
        "ERROR": "\033[0;31m[ERROR]\033[0m"
    }
    print(f"{colors.get(status, '[INFO]')} {message}")

def test_postgres_connection():
    """Test PostgreSQL connection and basic operations"""
    print_status("Testing PostgreSQL connection...")
    
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Test basic connection
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print_status(f"PostgreSQL version: {version['version'][:50]}...", "SUCCESS")
        
        # Test pgvector extension
        cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        vector_ext = cursor.fetchone()
        if vector_ext:
            print_status("pgvector extension is available", "SUCCESS")
        else:
            print_status("pgvector extension not found", "WARNING")
        
        # Test basic table operations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_vectors (
                id SERIAL PRIMARY KEY,
                content TEXT,
                embedding vector(384)
            );
        """)
        
        # Insert test data
        test_embedding = [0.1] * 384  # Simple test vector
        cursor.execute("""
            INSERT INTO test_vectors (content, embedding) 
            VALUES (%s, %s) 
            RETURNING id;
        """, ("Test content", test_embedding))
        
        test_id = cursor.fetchone()['id']
        print_status(f"Created test record with ID: {test_id}", "SUCCESS")
        
        # Test vector similarity search
        cursor.execute("""
            SELECT id, content, embedding <-> %s as distance 
            FROM test_vectors 
            ORDER BY embedding <-> %s 
            LIMIT 1;
        """, (test_embedding, test_embedding))
        
        result = cursor.fetchone()
        print_status(f"Vector similarity search successful, distance: {result['distance']}", "SUCCESS")
        
        # Cleanup
        cursor.execute("DROP TABLE test_vectors;")
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print_status(f"PostgreSQL test failed: {str(e)}", "ERROR")
        return False

def test_memfuse_api():
    """Test MemFuse API endpoints"""
    print_status("Testing MemFuse API...")
    
    # Wait for API to be ready
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{MEMFUSE_API_URL}/api/v1/health", timeout=5)
            if response.status_code == 200:
                print_status("MemFuse API is healthy", "SUCCESS")
                break
        except requests.exceptions.RequestException:
            if attempt < max_attempts - 1:
                print_status(f"Waiting for MemFuse API... (attempt {attempt + 1}/{max_attempts})")
                time.sleep(2)
            else:
                print_status("MemFuse API health check failed", "ERROR")
                return False
    
    try:
        # Test memory storage
        test_memory = {
            "content": f"This is a test memory created at {time.time()}",
            "metadata": {
                "test": True,
                "timestamp": time.time()
            }
        }
        
        response = requests.post(
            f"{MEMFUSE_API_URL}/api/v1/memory",
            json=test_memory,
            timeout=30
        )
        
        if response.status_code == 200:
            memory_data = response.json()
            memory_id = memory_data.get('id')
            print_status(f"Memory stored successfully with ID: {memory_id}", "SUCCESS")
            
            # Wait a moment for potential embedding generation
            time.sleep(3)
            
            # Test memory retrieval
            query_data = {
                "query": "test memory",
                "limit": 5
            }
            
            response = requests.post(
                f"{MEMFUSE_API_URL}/api/v1/memory/query",
                json=query_data,
                timeout=30
            )
            
            if response.status_code == 200:
                query_results = response.json()
                results = query_results.get('results', [])
                print_status(f"Memory query successful, found {len(results)} results", "SUCCESS")
                
                if results:
                    for i, result in enumerate(results[:3]):
                        print_status(f"  Result {i+1}: {result.get('content', '')[:50]}...")
                
                return True
            else:
                print_status(f"Memory query failed: {response.status_code} - {response.text}", "ERROR")
                return False
        else:
            print_status(f"Memory storage failed: {response.status_code} - {response.text}", "ERROR")
            return False
            
    except Exception as e:
        print_status(f"MemFuse API test failed: {str(e)}", "ERROR")
        traceback.print_exc()
        return False

def test_database_integration():
    """Test direct database integration"""
    print_status("Testing database integration...")
    
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check if MemFuse tables exist
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE 'm%_episodic';
        """)
        
        tables = cursor.fetchall()
        if tables:
            table_name = tables[0]['table_name']
            print_status(f"Found MemFuse table: {table_name}", "SUCCESS")
            
            # Check table structure
            cursor.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}';
            """)
            
            columns = cursor.fetchall()
            print_status(f"Table has {len(columns)} columns", "SUCCESS")
            
            # Check for data
            cursor.execute(f"SELECT COUNT(*) as count FROM {table_name};")
            count = cursor.fetchone()['count']
            print_status(f"Table contains {count} records", "SUCCESS")
            
            if count > 0:
                # Check for embeddings
                cursor.execute(f"""
                    SELECT COUNT(*) as count 
                    FROM {table_name} 
                    WHERE embedding IS NOT NULL;
                """)
                embedding_count = cursor.fetchone()['count']
                print_status(f"Records with embeddings: {embedding_count}", "SUCCESS")
        else:
            print_status("No MemFuse tables found (this may be normal for first run)", "WARNING")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print_status(f"Database integration test failed: {str(e)}", "ERROR")
        return False

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print_status("🚀 Starting MemFuse Comprehensive End-to-End Tests")
    print_status("=" * 60)
    
    tests = [
        ("PostgreSQL Connection & Vector Operations", test_postgres_connection),
        ("MemFuse API Functionality", test_memfuse_api),
        ("Database Integration", test_database_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print_status(f"\n📋 Running: {test_name}")
        print_status("-" * 40)
        
        try:
            result = test_func()
            results[test_name] = result
            
            if result:
                print_status(f"✅ {test_name}: PASSED", "SUCCESS")
            else:
                print_status(f"❌ {test_name}: FAILED", "ERROR")
                
        except Exception as e:
            print_status(f"❌ {test_name}: FAILED with exception: {str(e)}", "ERROR")
            results[test_name] = False
            traceback.print_exc()
    
    # Summary
    print_status("\n" + "=" * 60)
    print_status("🎯 Test Results Summary")
    print_status("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print_status(f"{status}: {test_name}")
    
    print_status(f"\n📊 Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print_status("🎉 All tests passed! MemFuse pgai environment is working correctly!", "SUCCESS")
        return True
    else:
        print_status(f"⚠️  {total - passed} test(s) failed. Please check the logs above.", "WARNING")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
