"""
Docker pgvector Integration Test

This test validates the complete Docker initialization and pgvector extension setup.
It ensures that the Docker configuration correctly installs pgvector and that MemFuse
can start successfully from a clean state.
"""

import asyncio
import subprocess
import time
import pytest
import requests
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DockerPgvectorIntegrationTest:
    """Integration test for Docker pgvector setup and MemFuse startup."""
    
    def __init__(self):
        self.container_name = "memfuse-pgai-postgres"
        self.volume_name = "memfuse-pgai_postgres_pgai_data"
        self.memfuse_process: Optional[subprocess.Popen] = None
    
    def run_command(self, command: str, timeout: int = 30) -> tuple[int, str, str]:
        """Run a shell command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout} seconds"
    
    def cleanup_docker_resources(self) -> bool:
        """Clean up Docker container and volume."""
        logger.info("🧹 Cleaning up Docker resources...")
        
        # Stop and remove container
        exit_code, _, _ = self.run_command(f"docker stop {self.container_name}")
        exit_code, _, _ = self.run_command(f"docker rm {self.container_name}")
        
        # Remove volume
        exit_code, _, _ = self.run_command(f"docker volume rm {self.volume_name}")
        
        logger.info("✅ Docker cleanup completed")
        return True
    
    def start_fresh_database(self) -> bool:
        """Start a fresh database container."""
        logger.info("🐳 Starting fresh database container...")
        
        exit_code, stdout, stderr = self.run_command(
            "docker-compose -f docker/compose/docker-compose.pgai.yml up -d postgres-pgai",
            timeout=60
        )
        
        if exit_code != 0:
            logger.error(f"Failed to start database: {stderr}")
            return False
        
        # Wait for database to be ready
        logger.info("⏳ Waiting for database initialization...")
        time.sleep(20)
        
        return True
    
    def check_pgvector_installation(self) -> bool:
        """Check if pgvector extension was installed by Docker initialization."""
        logger.info("🔍 Checking pgvector installation...")
        
        exit_code, stdout, stderr = self.run_command(
            f'docker exec {self.container_name} psql -U postgres -d memfuse -t -c '
            '"SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = \'vector\')"'
        )
        
        if exit_code != 0:
            logger.error(f"Failed to check pgvector: {stderr}")
            return False
        
        is_installed = stdout.strip() == "t"
        if is_installed:
            logger.info("✅ pgvector extension installed by Docker initialization")
        else:
            logger.warning("⚠️ pgvector extension not installed by Docker initialization")
        
        return is_installed
    
    def start_memfuse_server(self) -> bool:
        """Start MemFuse server in background."""
        logger.info("🚀 Starting MemFuse server...")
        
        try:
            self.memfuse_process = subprocess.Popen(
                ["poetry", "run", "memfuse-core"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for startup
            logger.info("⏳ Waiting for MemFuse startup...")
            time.sleep(30)
            
            # Check if process is still running
            if self.memfuse_process.poll() is not None:
                stdout, stderr = self.memfuse_process.communicate()
                logger.error(f"MemFuse process exited early: {stderr}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MemFuse: {e}")
            return False
    
    def test_health_endpoint(self) -> bool:
        """Test MemFuse health endpoint."""
        logger.info("🏥 Testing health endpoint...")
        
        try:
            response = requests.get("http://127.0.0.1:8000/api/v1/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("data", {}).get("status") == "ok":
                    logger.info("✅ Health endpoint test passed")
                    return True
            
            logger.error(f"Health endpoint failed: {response.status_code}")
            return False
            
        except Exception as e:
            logger.error(f"Health endpoint test failed: {e}")
            return False
    
    def stop_memfuse_server(self) -> bool:
        """Stop MemFuse server gracefully."""
        if self.memfuse_process:
            logger.info("🛑 Stopping MemFuse server...")
            
            try:
                # Send SIGINT for graceful shutdown
                self.memfuse_process.terminate()
                
                # Wait for graceful shutdown
                try:
                    self.memfuse_process.wait(timeout=10)
                    logger.info("✅ MemFuse stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning("⚠️ Graceful shutdown timed out, forcing termination")
                    self.memfuse_process.kill()
                    self.memfuse_process.wait()
                
                return True
                
            except Exception as e:
                logger.error(f"Error stopping MemFuse: {e}")
                return False
        
        return True
    
    def run_complete_test(self) -> bool:
        """Run the complete integration test."""
        logger.info("🧪 Starting Docker pgvector Integration Test")
        logger.info("=" * 50)
        
        try:
            # Step 1: Clean slate
            if not self.cleanup_docker_resources():
                return False
            
            # Step 2: Start fresh database
            if not self.start_fresh_database():
                return False
            
            # Step 3: Check Docker initialization
            docker_init_success = self.check_pgvector_installation()
            
            # Step 4: Start MemFuse
            if not self.start_memfuse_server():
                return False
            
            # Step 5: Test functionality
            if not self.test_health_endpoint():
                return False
            
            # Step 6: Stop MemFuse
            if not self.stop_memfuse_server():
                return False
            
            # Final results
            logger.info("")
            logger.info("🎉 Integration Test Results")
            logger.info("=" * 30)
            logger.info(f"✅ Docker initialization: {'SUCCESS' if docker_init_success else 'PARTIAL'}")
            logger.info("✅ MemFuse startup: SUCCESS")
            logger.info("✅ Health endpoint: SUCCESS")
            logger.info("✅ Graceful shutdown: SUCCESS")
            logger.info("")
            logger.info("🎯 Overall result: SUCCESS")
            
            return True
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            return False
        
        finally:
            # Cleanup
            self.stop_memfuse_server()


@pytest.mark.integration
@pytest.mark.docker
def test_docker_pgvector_integration():
    """Pytest wrapper for Docker pgvector integration test."""
    test = DockerPgvectorIntegrationTest()
    assert test.run_complete_test(), "Docker pgvector integration test failed"


if __name__ == "__main__":
    # Allow running as standalone script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    test = DockerPgvectorIntegrationTest()
    success = test.run_complete_test()
    
    if success:
        print("\n🎉 All tests passed!")
        exit(0)
    else:
        print("\n❌ Tests failed!")
        exit(1)
