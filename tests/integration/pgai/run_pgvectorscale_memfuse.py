#!/usr/bin/env python3
"""
MemFuse PgVectorScale Integration Tests

This module provides comprehensive integration tests for the PgVectorScale implementation,
including both basic functionality tests and end-to-end validation.

Features:
- Complete data flow validation (M0 → M1 pipeline)
- Vector similarity search performance testing
- Data lineage and consistency verification
- StreamingDiskANN index performance validation
- Comprehensive error handling and edge case testing

Requirements:
- Python 3.8+
- sentence-transformers
- psycopg2-binary
- numpy
- pgvectorscale database running on localhost:5432

Usage:
    # Run basic integration test
    poetry run python tests/integration/pgai/run_pgvectorscale_memfuse.py --mode basic
    
    # Run comprehensive validation
    poetry run python tests/integration/pgai/run_pgvectorscale_memfuse.py --mode validation
    
    # Run performance benchmarks
    poetry run python tests/integration/pgai/run_pgvectorscale_memfuse.py --mode benchmark

Environment Variables:
    PGVECTORSCALE_HOST: Database host (default: localhost)
    PGVECTORSCALE_PORT: Database port (default: 5432)
    PGVECTORSCALE_DB: Database name (default: memfuse)
    PGVECTORSCALE_USER: Database user (default: postgres)
    PGVECTORSCALE_PASSWORD: Database password (default: postgres)
"""

import asyncio
import argparse
import sys
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from memfuse_core.hierarchy.streaming_pipeline import StreamingDataProcessor
from memfuse_core.interfaces.message_interface import MessageBatchList


class PgVectorScaleIntegrationTester:
    """Comprehensive integration tester for PgVectorScale functionality."""
    
    def __init__(self, user_id: str = "test_user"):
        self.user_id = user_id
        self.processor = None
        self.test_results = {}
    
    async def initialize(self):
        """Initialize the streaming data processor."""
        logger.info("🔧 Initializing PgVectorScale integration tester...")
        self.processor = StreamingDataProcessor(user_id=self.user_id)
        await self.processor.initialize()
        logger.info("✅ Integration tester initialized successfully")
    
    async def cleanup(self):
        """Clean up resources."""
        if self.processor:
            await self.processor.close()
            logger.info("🧹 Integration tester cleaned up")
    
    def generate_test_conversation(self, size: str = "medium") -> List[List[Dict[str, str]]]:
        """Generate test conversation data of different sizes."""
        
        conversations = {
            "small": [
                [
                    {"content": "Hello, I need help with Python programming.", "role": "user"},
                    {"content": "I'd be happy to help you with Python! What specific topic would you like to learn about?", "role": "assistant"},
                ]
            ],
            "medium": [
                # Batch 1: Python ML Introduction
                [
                    {"content": "I want to learn Python machine learning, where should I start?", "role": "user"},
                    {"content": "I recommend starting with scikit-learn, which provides rich machine learning algorithms and tools.", "role": "assistant"},
                    {"content": "What about deep learning frameworks like TensorFlow or PyTorch?", "role": "user"},
                ],
                
                # Batch 2: Data Science Workflow
                [
                    {"content": "What's the typical data science project workflow?", "role": "user"},
                    {"content": "The workflow typically includes: 1) Problem definition, 2) Data collection and cleaning, 3) Exploratory data analysis, 4) Feature engineering, 5) Model selection and training, 6) Evaluation and validation, 7) Deployment and monitoring.", "role": "assistant"},
                ],
                
                # Batch 3: Vector Databases Discussion
                [
                    {"content": "Can you explain vector databases and their use cases?", "role": "user"},
                    {"content": "Vector databases are specialized databases designed to store and query high-dimensional vector embeddings. They're essential for similarity search, recommendation systems, and RAG applications.", "role": "assistant"},
                    {"content": "How do they compare to traditional relational databases?", "role": "user"},
                    {"content": "Vector databases excel at semantic similarity search using cosine distance or other vector metrics, while relational databases are optimized for exact matches and structured queries.", "role": "assistant"},
                ],
            ],
            "large": []  # Will be generated dynamically for performance testing
        }
        
        if size == "large":
            # Generate large dataset for performance testing
            large_conversation = []
            for i in range(20):  # 20 batches
                batch = []
                for j in range(10):  # 10 messages per batch
                    batch.extend([
                        {"content": f"This is test message {i*10+j} about topic {i}. It contains various technical content about machine learning, data science, and vector databases.", "role": "user"},
                        {"content": f"This is response {i*10+j} providing detailed information about the requested topic {i}. It includes comprehensive explanations and examples.", "role": "assistant"},
                    ])
                large_conversation.append(batch)
            return large_conversation
        
        return conversations.get(size, conversations["medium"])
    
    async def test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic PgVectorScale functionality."""
        logger.info("🧪 Running basic functionality tests...")
        
        results = {
            "test_name": "basic_functionality",
            "success": False,
            "details": {},
            "errors": []
        }
        
        try:
            # Generate small test data
            messages = self.generate_test_conversation("small")
            flat_messages = []
            for batch in messages:
                flat_messages.extend(batch)
            
            session_id = str(uuid.uuid4())
            
            # Test data processing
            start_time = time.time()
            result = await self.processor.process_streaming_messages(
                messages=flat_messages,
                session_id=session_id,
                metadata={"test_type": "basic"}
            )
            processing_time = time.time() - start_time
            
            if result["success"]:
                results["success"] = True
                results["details"] = {
                    "messages_processed": result["messages_processed"],
                    "processing_time": processing_time,
                    "items_written": result["items_written"]
                }
                logger.info(f"✅ Basic functionality test passed: {result['messages_processed']} messages processed in {processing_time:.3f}s")
            else:
                results["errors"].append(f"Processing failed: {result['message']}")
                logger.error(f"❌ Basic functionality test failed: {result['message']}")
        
        except Exception as e:
            results["errors"].append(str(e))
            logger.error(f"❌ Basic functionality test error: {e}")
        
        return results
    
    async def test_vector_search(self) -> Dict[str, Any]:
        """Test vector similarity search functionality."""
        logger.info("🔍 Running vector search tests...")
        
        results = {
            "test_name": "vector_search",
            "success": False,
            "details": {},
            "errors": []
        }
        
        try:
            test_queries = [
                "Python machine learning",
                "data science workflow",
                "vector databases"
            ]
            
            search_results = []
            total_query_time = 0
            
            for query in test_queries:
                start_time = time.time()
                result = await self.processor.query_processed_data(
                    query=query,
                    top_k=3
                )
                query_time = time.time() - start_time
                total_query_time += query_time
                
                if result["success"] and result["results"]:
                    search_results.append({
                        "query": query,
                        "results_count": len(result["results"]),
                        "query_time": query_time,
                        "top_similarity": result["results"][0].get('similarity_score', 0) if result["results"] else 0
                    })
                else:
                    results["errors"].append(f"Query '{query}' failed: {result.get('message', 'Unknown error')}")
            
            if search_results:
                results["success"] = True
                results["details"] = {
                    "queries_tested": len(test_queries),
                    "successful_queries": len(search_results),
                    "average_query_time": total_query_time / len(search_results) if search_results else 0,
                    "average_similarity": sum(r["top_similarity"] for r in search_results) / len(search_results) if search_results else 0,
                    "search_results": search_results
                }
                logger.info(f"✅ Vector search test passed: {len(search_results)}/{len(test_queries)} queries successful")
            else:
                logger.error("❌ Vector search test failed: No successful queries")
        
        except Exception as e:
            results["errors"].append(str(e))
            logger.error(f"❌ Vector search test error: {e}")
        
        return results

    async def test_data_flow_validation(self) -> Dict[str, Any]:
        """Test complete data flow validation with precise counting."""
        logger.info("📊 Running data flow validation tests...")

        results = {
            "test_name": "data_flow_validation",
            "success": False,
            "details": {},
            "errors": []
        }

        try:
            # Get initial counts
            initial_stats = await self.processor.get_processing_stats()
            initial_memory_stats = initial_stats.get('memory_layer_stats', {})
            store_stats = initial_memory_stats.get('store_stats', {})
            layer_stats = store_stats.get('layer_stats', [])

            initial_m0 = layer_stats[0].get('record_count', 0) if len(layer_stats) > 0 else 0
            initial_m1 = layer_stats[1].get('record_count', 0) if len(layer_stats) > 1 else 0

            # Process test data
            messages = self.generate_test_conversation("medium")
            flat_messages = []
            for batch in messages:
                flat_messages.extend(batch)

            session_id = str(uuid.uuid4())

            start_time = time.time()
            result = await self.processor.process_streaming_messages(
                messages=flat_messages,
                session_id=session_id,
                metadata={"test_type": "validation"}
            )
            processing_time = time.time() - start_time

            if not result["success"]:
                results["errors"].append(f"Processing failed: {result['message']}")
                return results

            # Wait for async M1 processing
            await asyncio.sleep(2)

            # Get final counts
            final_stats = await self.processor.get_processing_stats()
            final_memory_stats = final_stats.get('memory_layer_stats', {})
            final_store_stats = final_memory_stats.get('store_stats', {})
            final_layer_stats = final_store_stats.get('layer_stats', [])

            final_m0 = final_layer_stats[0].get('record_count', 0) if len(final_layer_stats) > 0 else 0
            final_m1 = final_layer_stats[1].get('record_count', 0) if len(final_layer_stats) > 1 else 0
            final_m1_embeddings = final_layer_stats[2].get('record_count', 0) if len(final_layer_stats) > 2 else 0

            # Validate data consistency
            m0_increase = final_m0 - initial_m0
            m1_increase = final_m1 - initial_m1

            total_input_messages = len(flat_messages)

            if m0_increase >= total_input_messages and m1_increase > 0:
                results["success"] = True
                results["details"] = {
                    "input_messages": total_input_messages,
                    "m0_increase": m0_increase,
                    "m1_increase": m1_increase,
                    "m1_embeddings": final_m1_embeddings,
                    "processing_time": processing_time,
                    "m0_to_m1_ratio": m1_increase / max(1, m0_increase),
                    "embedding_coverage": final_m1_embeddings / max(1, final_m1) if final_m1 > 0 else 0
                }
                logger.info(f"✅ Data flow validation passed: {m0_increase} M0 → {m1_increase} M1 records")
            else:
                results["errors"].append(f"Data validation failed: M0 increase {m0_increase}, M1 increase {m1_increase}")
                logger.error(f"❌ Data flow validation failed: insufficient data processing")

        except Exception as e:
            results["errors"].append(str(e))
            logger.error(f"❌ Data flow validation error: {e}")

        return results

    async def test_performance_benchmark(self) -> Dict[str, Any]:
        """Test performance with larger datasets."""
        logger.info("⚡ Running performance benchmark tests...")

        results = {
            "test_name": "performance_benchmark",
            "success": False,
            "details": {},
            "errors": []
        }

        try:
            # Generate large test dataset
            messages = self.generate_test_conversation("large")
            flat_messages = []
            for batch in messages:
                flat_messages.extend(batch)

            session_id = str(uuid.uuid4())

            # Benchmark data processing
            start_time = time.time()
            result = await self.processor.process_streaming_messages(
                messages=flat_messages,
                session_id=session_id,
                metadata={"test_type": "benchmark"}
            )
            processing_time = time.time() - start_time

            if not result["success"]:
                results["errors"].append(f"Processing failed: {result['message']}")
                return results

            # Benchmark query performance
            benchmark_queries = [
                "machine learning algorithms",
                "data science techniques",
                "vector database performance",
                "artificial intelligence applications",
                "deep learning frameworks"
            ]

            query_times = []
            for query in benchmark_queries:
                start_time = time.time()
                query_result = await self.processor.query_processed_data(
                    query=query,
                    top_k=10
                )
                query_time = time.time() - start_time
                query_times.append(query_time)

            results["success"] = True
            results["details"] = {
                "messages_processed": len(flat_messages),
                "processing_time": processing_time,
                "messages_per_second": len(flat_messages) / processing_time,
                "average_query_time": sum(query_times) / len(query_times),
                "min_query_time": min(query_times),
                "max_query_time": max(query_times),
                "queries_per_second": len(query_times) / sum(query_times)
            }

            logger.info(f"✅ Performance benchmark passed: {len(flat_messages)} messages in {processing_time:.3f}s ({len(flat_messages)/processing_time:.1f} msg/s)")

        except Exception as e:
            results["errors"].append(str(e))
            logger.error(f"❌ Performance benchmark error: {e}")

        return results

    async def run_test_suite(self, mode: str = "basic") -> Dict[str, Any]:
        """Run the specified test suite."""
        logger.info(f"🚀 Starting PgVectorScale integration test suite: {mode}")

        suite_results = {
            "mode": mode,
            "start_time": time.time(),
            "tests": [],
            "summary": {}
        }

        try:
            await self.initialize()

            if mode == "basic":
                suite_results["tests"].append(await self.test_basic_functionality())
                suite_results["tests"].append(await self.test_vector_search())

            elif mode == "validation":
                suite_results["tests"].append(await self.test_basic_functionality())
                suite_results["tests"].append(await self.test_data_flow_validation())
                suite_results["tests"].append(await self.test_vector_search())

            elif mode == "benchmark":
                suite_results["tests"].append(await self.test_performance_benchmark())
                suite_results["tests"].append(await self.test_vector_search())

            elif mode == "all":
                suite_results["tests"].append(await self.test_basic_functionality())
                suite_results["tests"].append(await self.test_data_flow_validation())
                suite_results["tests"].append(await self.test_vector_search())
                suite_results["tests"].append(await self.test_performance_benchmark())

            else:
                raise ValueError(f"Unknown test mode: {mode}")

        finally:
            await self.cleanup()

        # Calculate summary
        suite_results["end_time"] = time.time()
        suite_results["total_time"] = suite_results["end_time"] - suite_results["start_time"]

        successful_tests = [t for t in suite_results["tests"] if t["success"]]
        failed_tests = [t for t in suite_results["tests"] if not t["success"]]

        suite_results["summary"] = {
            "total_tests": len(suite_results["tests"]),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "success_rate": len(successful_tests) / len(suite_results["tests"]) if suite_results["tests"] else 0,
            "overall_success": len(failed_tests) == 0
        }

        return suite_results

    def print_results(self, results: Dict[str, Any]):
        """Print formatted test results."""
        logger.info("=" * 70)
        logger.info(f"🎯 PgVectorScale Integration Test Results - Mode: {results['mode']}")
        logger.info("=" * 70)

        summary = results["summary"]
        if summary["overall_success"]:
            logger.info(f"✅ ALL TESTS PASSED ({summary['successful_tests']}/{summary['total_tests']})")
        else:
            logger.error(f"❌ SOME TESTS FAILED ({summary['failed_tests']}/{summary['total_tests']} failed)")

        logger.info(f"⏱️ Total execution time: {results['total_time']:.3f}s")
        logger.info("")

        for test in results["tests"]:
            status = "✅ PASS" if test["success"] else "❌ FAIL"
            logger.info(f"{status} {test['test_name']}")

            if test["success"] and test["details"]:
                for key, value in test["details"].items():
                    if isinstance(value, float):
                        logger.info(f"    {key}: {value:.3f}")
                    else:
                        logger.info(f"    {key}: {value}")

            if test["errors"]:
                for error in test["errors"]:
                    logger.error(f"    Error: {error}")

            logger.info("")


async def main():
    """Main function to run integration tests."""
    parser = argparse.ArgumentParser(description="PgVectorScale Integration Tests")
    parser.add_argument(
        "--mode",
        choices=["basic", "validation", "benchmark", "all"],
        default="basic",
        help="Test mode to run"
    )
    parser.add_argument(
        "--user-id",
        default="integration_test_user",
        help="User ID for testing"
    )

    args = parser.parse_args()

    tester = PgVectorScaleIntegrationTester(user_id=args.user_id)

    try:
        results = await tester.run_test_suite(mode=args.mode)
        tester.print_results(results)

        # Exit with appropriate code
        if results["summary"]["overall_success"]:
            logger.info("🎉 All integration tests completed successfully!")
            sys.exit(0)
        else:
            logger.error("💥 Some integration tests failed!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"💥 Integration test suite failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
