#!/usr/bin/env python3
"""
Connection Pool Bottleneck Analysis

This script analyzes the root causes of connection pool issues and provides
specific recommendations for fixing parallel processing problems.
"""

import asyncio
import time
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from loguru import logger

# Add src path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from memfuse_core.services.global_connection_manager import get_global_connection_manager
from memfuse_core.services.database_service import DatabaseService
from memfuse_core.services.memory_service import MemoryService
from memfuse_core.store.pgai_store import PgaiStore
from memfuse_core.utils.config import config_manager


@dataclass
class ConnectionAnalysis:
    """Analysis results for connection usage patterns."""
    scenario: str
    initial_connections: int
    peak_connections: int
    final_connections: int
    connection_growth: int
    pool_stats: Dict[str, Any]
    timing_analysis: Dict[str, float]
    error_patterns: List[str]
    recommendations: List[str]


class ConnectionBottleneckAnalyzer:
    """Analyzer for connection pool bottlenecks."""
    
    def __init__(self):
        self.connection_manager = get_global_connection_manager()
        self.analyses: List[ConnectionAnalysis] = []
        
    async def _get_connection_count(self) -> int:
        """Get current PostgreSQL connection count."""
        try:
            db = await DatabaseService.get_instance()
            result = await db.backend.execute(
                "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()"
            )
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.warning(f"Could not get connection count: {e}")
            return 0
            
    async def _get_detailed_connections(self) -> List[Dict[str, Any]]:
        """Get detailed connection information."""
        try:
            db = await DatabaseService.get_instance()
            result = await db.backend.execute("""
                SELECT
                    pid,
                    usename,
                    application_name,
                    client_addr,
                    state,
                    state_change,
                    query_start,
                    backend_start,
                    query
                FROM pg_stat_activity
                WHERE datname = current_database()
                ORDER BY backend_start DESC
            """)

            connections = []
            for row in result:
                connections.append({
                    "pid": row['pid'],
                    "user": row['usename'],
                    "app": row['application_name'],
                    "client": str(row['client_addr']) if row['client_addr'] else None,
                    "state": row['state'],
                    "state_change": row['state_change'],
                    "query_start": row['query_start'],
                    "backend_start": row['backend_start'],
                    "query": row['query']
                })
            return connections
        except Exception as e:
            logger.warning(f"Could not get detailed connections: {e}")
            return []
            
    async def analyze_singleton_pattern_issues(self) -> ConnectionAnalysis:
        """Analyze issues with singleton pattern implementation."""
        logger.info("Analyzing singleton pattern issues...")
        
        initial_count = await self._get_connection_count()
        start_time = time.time()
        errors = []
        
        try:
            # Test multiple DatabaseService calls
            db_instances = []
            for i in range(5):
                db = await DatabaseService.get_instance()
                db_instances.append(db)
                
            # Check if they're the same instance
            all_same = all(db is db_instances[0] for db in db_instances[1:])
            if not all_same:
                errors.append("DatabaseService singleton pattern broken - multiple instances created")
                
            # Test connection reuse
            connection_counts = []
            for i in range(3):
                count = await self._get_connection_count()
                connection_counts.append(count)
                await asyncio.sleep(0.1)
                
            if len(set(connection_counts)) > 1:
                errors.append(f"Connection count varies during singleton usage: {connection_counts}")
                
        except Exception as e:
            errors.append(f"Singleton test failed: {str(e)}")
            
        peak_count = await self._get_connection_count()
        
        # Reset and check cleanup
        DatabaseService.reset_instance_sync()
        await asyncio.sleep(1)
        final_count = await self._get_connection_count()
        
        timing = {"total_time": time.time() - start_time}
        
        recommendations = []
        if errors:
            recommendations.append("Fix DatabaseService singleton implementation")
        if final_count > initial_count:
            recommendations.append("Improve connection cleanup in DatabaseService.reset_instance()")
            
        return ConnectionAnalysis(
            scenario="Singleton Pattern Analysis",
            initial_connections=initial_count,
            peak_connections=peak_count,
            final_connections=final_count,
            connection_growth=final_count - initial_count,
            pool_stats=self.connection_manager.get_pool_statistics(),
            timing_analysis=timing,
            error_patterns=errors,
            recommendations=recommendations
        )
        
    async def analyze_pgai_store_concurrency(self) -> ConnectionAnalysis:
        """Analyze PgaiStore concurrent access patterns."""
        logger.info("Analyzing PgaiStore concurrency issues...")
        
        initial_count = await self._get_connection_count()
        start_time = time.time()
        errors = []
        
        stores = []
        try:
            # Create multiple stores concurrently
            async def create_store(store_id: int):
                try:
                    config = config_manager.get_config()
                    store = PgaiStore(
                        table_name=f"concurrent_test_{store_id}",
                        config=config
                    )
                    await store.initialize()
                    return store
                except Exception as e:
                    errors.append(f"Store {store_id} creation failed: {str(e)}")
                    return None
                    
            # Create stores concurrently
            tasks = [create_store(i) for i in range(5)]
            store_results = await asyncio.gather(*tasks, return_exceptions=True)
            stores = [s for s in store_results if s is not None]
            
            peak_count = await self._get_connection_count()
            
            # Test concurrent operations
            async def test_store_operations(store):
                try:
                    if store:
                        # Add test data
                        await store.add({"content": "test", "metadata": {"test": True}})
                        # Query data
                        await store.query("test", limit=1)
                        return True
                except Exception as e:
                    errors.append(f"Store operation failed: {str(e)}")
                    return False
                    
            # Run operations concurrently
            operation_tasks = [test_store_operations(store) for store in stores]
            await asyncio.gather(*operation_tasks, return_exceptions=True)
            
        except Exception as e:
            errors.append(f"Concurrency test failed: {str(e)}")
            
        # Cleanup stores
        for store in stores:
            if store:
                try:
                    await store.close()
                except Exception as e:
                    errors.append(f"Store cleanup failed: {str(e)}")
                    
        await asyncio.sleep(1)
        final_count = await self._get_connection_count()
        
        timing = {"total_time": time.time() - start_time}
        
        recommendations = []
        if len(errors) > 2:
            recommendations.append("Improve error handling in PgaiStore initialization")
        if final_count > initial_count + 2:
            recommendations.append("Fix connection leaks in PgaiStore.close()")
        if peak_count > initial_count + 10:
            recommendations.append("Optimize connection pool size for concurrent PgaiStore usage")
            
        return ConnectionAnalysis(
            scenario="PgaiStore Concurrency Analysis",
            initial_connections=initial_count,
            peak_connections=peak_count,
            final_connections=final_count,
            connection_growth=final_count - initial_count,
            pool_stats=self.connection_manager.get_pool_statistics(),
            timing_analysis=timing,
            error_patterns=errors,
            recommendations=recommendations
        )
        
    async def analyze_parallel_processing_bottlenecks(self) -> ConnectionAnalysis:
        """Analyze M0/M1/M2 parallel processing connection usage."""
        logger.info("Analyzing parallel processing bottlenecks...")
        
        initial_count = await self._get_connection_count()
        start_time = time.time()
        errors = []
        
        try:
            # Test with parallel processing enabled
            config = config_manager.get_config()
            if "memory" not in config:
                config["memory"] = {}
            if "memory_service" not in config["memory"]:
                config["memory"]["memory_service"] = {}
            config["memory"]["memory_service"]["parallel_enabled"] = True
            
            memory_service = MemoryService(
                user="parallel_test_user",
                agent="parallel_test_agent",
                cfg=config
            )
            
            init_start = time.time()
            await memory_service.initialize()
            init_time = time.time() - init_start

            post_init_count = await self._get_connection_count()

            # Create test session to avoid foreign key constraint errors
            db = await DatabaseService.get_instance()
            user_id = await db.get_or_create_user_by_name("parallel_test_user")
            agent_id = await db.get_or_create_agent_by_name("parallel_test_agent")
            test_session_id = await db.create_session_with_name(user_id, agent_id, "test_session_parallel")

            # Process test data
            process_start = time.time()
            messages = [
                [
                    {
                        "role": "user",
                        "content": f"Parallel test message {i}",
                        "session_id": test_session_id,
                        "metadata": {"test_id": i, "session_id": test_session_id}
                    }
                ]
                for i in range(3)
            ]
            
            result = await memory_service.add_batch(messages)
            process_time = time.time() - process_start
            
            peak_count = await self._get_connection_count()
            
            if result.get("status") != "success":
                errors.append(f"Parallel processing failed: {result.get('message', 'Unknown error')}")
                
            # Cleanup
            await memory_service.shutdown()
            
        except Exception as e:
            errors.append(f"Parallel processing test failed: {str(e)}")
            init_time = 0
            process_time = 0
            post_init_count = initial_count
            peak_count = initial_count
            
        await asyncio.sleep(2)
        final_count = await self._get_connection_count()
        
        timing = {
            "total_time": time.time() - start_time,
            "init_time": init_time,
            "process_time": process_time
        }
        
        recommendations = []
        if init_time > 10:
            recommendations.append("Optimize MemoryService initialization - taking too long")
        if peak_count > initial_count + 15:
            recommendations.append("Parallel processing uses too many connections - optimize layer coordination")
        if final_count > initial_count + 3:
            recommendations.append("Fix connection leaks in parallel processing cleanup")
        if errors:
            recommendations.append("Fix parallel processing errors before enabling in production")
            
        return ConnectionAnalysis(
            scenario="Parallel Processing Analysis",
            initial_connections=initial_count,
            peak_connections=peak_count,
            final_connections=final_count,
            connection_growth=final_count - initial_count,
            pool_stats=self.connection_manager.get_pool_statistics(),
            timing_analysis=timing,
            error_patterns=errors,
            recommendations=recommendations
        )
        
    def print_analysis_report(self):
        """Print comprehensive analysis report."""
        print("\n" + "="*80)
        print("CONNECTION POOL BOTTLENECK ANALYSIS REPORT")
        print("="*80)
        
        for analysis in self.analyses:
            print(f"\n📊 {analysis.scenario}")
            print(f"   Initial connections: {analysis.initial_connections}")
            print(f"   Peak connections: {analysis.peak_connections}")
            print(f"   Final connections: {analysis.final_connections}")
            print(f"   Connection growth: {analysis.connection_growth}")
            
            if analysis.timing_analysis:
                print(f"   Timing: {analysis.timing_analysis}")
                
            if analysis.error_patterns:
                print(f"   ❌ Errors ({len(analysis.error_patterns)}):")
                for error in analysis.error_patterns[:3]:
                    print(f"      - {error}")
                if len(analysis.error_patterns) > 3:
                    print(f"      ... and {len(analysis.error_patterns) - 3} more")
                    
            if analysis.recommendations:
                print(f"   💡 Recommendations:")
                for rec in analysis.recommendations:
                    print(f"      - {rec}")
                    
        # Overall recommendations
        print(f"\n" + "="*80)
        print("OVERALL RECOMMENDATIONS")
        print("="*80)
        
        all_errors = sum(len(a.error_patterns) for a in self.analyses)
        total_growth = sum(a.connection_growth for a in self.analyses)
        
        if all_errors > 5:
            print("❌ High error rate detected - focus on error handling improvements")
        if total_growth > 10:
            print("❌ Significant connection leaks detected - review cleanup procedures")
            
        # Specific recommendations for parallel processing
        parallel_analysis = next((a for a in self.analyses if "Parallel" in a.scenario), None)
        if parallel_analysis:
            if parallel_analysis.error_patterns:
                print("🚫 RECOMMENDATION: Keep parallel_enabled=false until issues are resolved")
                print("   Reasons:")
                for error in parallel_analysis.error_patterns:
                    print(f"   - {error}")
            else:
                print("✅ RECOMMENDATION: parallel_enabled can be safely enabled")
                print("   Parallel processing appears stable")


async def main():
    """Run comprehensive connection bottleneck analysis."""
    print("🔍 Starting Connection Pool Bottleneck Analysis")
    
    analyzer = ConnectionBottleneckAnalyzer()
    
    try:
        # Run all analyses
        analysis1 = await analyzer.analyze_singleton_pattern_issues()
        analyzer.analyses.append(analysis1)
        
        analysis2 = await analyzer.analyze_pgai_store_concurrency()
        analyzer.analyses.append(analysis2)
        
        analysis3 = await analyzer.analyze_parallel_processing_bottlenecks()
        analyzer.analyses.append(analysis3)
        
        # Print comprehensive report
        analyzer.print_analysis_report()
        
        # Save detailed results
        results_file = Path(__file__).parent / "connection_analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(a) for a in analyzer.analyses], f, indent=2, default=str)
        print(f"\n📄 Detailed results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return False
        
    return True


if __name__ == "__main__":
    asyncio.run(main())
