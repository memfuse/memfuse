"""
Test Coverage Assessment for MemFuse Project.

This script evaluates the current test coverage and identifies
missing test scenarios based on the updated architecture.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Any
from dataclasses import dataclass


@dataclass
class TestCoverageReport:
    """Test coverage report structure."""
    component: str
    existing_tests: List[str]
    missing_tests: List[str]
    coverage_percentage: float
    priority: str  # HIGH, MEDIUM, LOW


class TestCoverageAssessment:
    """Assess test coverage for MemFuse components."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tests_dir = project_root / "tests"
        self.src_dir = project_root / "src" / "memfuse_core"

    def assess_coverage(self) -> Dict[str, TestCoverageReport]:
        """Assess test coverage for all components."""
        reports = {}

        # Core components to assess
        components = {
            "memory_layers": self._assess_memory_layers(),
            "interfaces": self._assess_interfaces(),
            "services": self._assess_services(),
            "pgai_integration": self._assess_pgai_integration(),
            "configuration": self._assess_configuration(),
            "parallel_processing": self._assess_parallel_processing(),
            "error_handling": self._assess_error_handling(),
            "performance": self._assess_performance()
        }

        return components

    def _assess_memory_layers(self) -> TestCoverageReport:
        """Assess memory layers test coverage."""
        existing_tests = [
            "tests/unit/hierarchy/test_memory_layer_naming.py",
            "tests/integration/hierarchy/test_parallel_memory_architecture.py",
            "tests/unit/test_memory_service_layer.py"
        ]

        missing_tests = [
            "tests/unit/hierarchy/test_m0_raw_data_layer.py",
            "tests/unit/hierarchy/test_m1_episodic_layer.py", 
            "tests/unit/hierarchy/test_m2_semantic_layer.py",
            "tests/integration/hierarchy/test_layer_coordination.py",
            "tests/integration/hierarchy/test_cross_layer_data_flow.py",
            "tests/performance/hierarchy/test_layer_performance.py"
        ]

        return TestCoverageReport(
            component="Memory Layers (M0/M1/M2)",
            existing_tests=existing_tests,
            missing_tests=missing_tests,
            coverage_percentage=33.3,  # 3 existing / 9 total
            priority="HIGH"
        )

    def _assess_interfaces(self) -> TestCoverageReport:
        """Assess interfaces test coverage."""
        existing_tests = [
            "tests/unit/interfaces/test_memory_layer_interface.py"
        ]

        missing_tests = [
            "tests/unit/interfaces/test_service_interface.py",
            "tests/unit/interfaces/test_store_interface.py",
            "tests/unit/interfaces/test_buffer_interface.py",
            "tests/integration/interfaces/test_interface_compatibility.py",
            "tests/contract/test_interface_contracts.py"
        ]

        return TestCoverageReport(
            component="Interfaces",
            existing_tests=existing_tests,
            missing_tests=missing_tests,
            coverage_percentage=16.7,  # 1 existing / 6 total
            priority="HIGH"
        )

    def _assess_services(self) -> TestCoverageReport:
        """Assess services test coverage."""
        existing_tests = [
            "tests/unit/test_memory_service_layer.py",
            "tests/integration/services/test_memory_service_integration.py",
            "tests/unit/services/test_buffer_service.py"
        ]

        missing_tests = [
            "tests/unit/services/test_memory_service_complete.py",
            "tests/integration/services/test_service_coordination.py",
            "tests/integration/services/test_service_error_handling.py",
            "tests/performance/services/test_service_performance.py"
        ]

        return TestCoverageReport(
            component="Services",
            existing_tests=existing_tests,
            missing_tests=missing_tests,
            coverage_percentage=42.9,  # 3 existing / 7 total
            priority="MEDIUM"
        )

    def _assess_pgai_integration(self) -> TestCoverageReport:
        """Assess PgAI integration test coverage."""
        existing_tests = [
            "tests/integration/pgai/test_pgai_integration_status.py",
            "tests/e2e/test_pgai_e2e.py",
            "tests/integration/test_multi_layer_pgai_integration.py",
            "tests/store/test_event_driven_pgai_store.py",
            "tests/performance/test_pgai_batch_embedding.py"
        ]

        missing_tests = [
            "tests/unit/pgai/test_immediate_trigger_components.py",
            "tests/unit/pgai/test_multi_layer_store.py",
            "tests/integration/pgai/test_pgai_error_recovery.py",
            "tests/performance/pgai/test_embedding_performance.py"
        ]

        return TestCoverageReport(
            component="PgAI Integration",
            existing_tests=existing_tests,
            missing_tests=missing_tests,
            coverage_percentage=55.6,  # 5 existing / 9 total
            priority="MEDIUM"
        )

    def _assess_configuration(self) -> TestCoverageReport:
        """Assess configuration test coverage."""
        existing_tests = [
            "tests/unit/config/test_configuration_optimization.py",
            "tests/unit/test_config_loading.py",
            "tests/unit/test_unified_configuration.py"
        ]

        missing_tests = [
            "tests/unit/config/test_config_validation.py",
            "tests/integration/config/test_config_inheritance.py",
            "tests/integration/config/test_environment_configs.py"
        ]

        return TestCoverageReport(
            component="Configuration",
            existing_tests=existing_tests,
            missing_tests=missing_tests,
            coverage_percentage=50.0,  # 3 existing / 6 total
            priority="LOW"
        )

    def _assess_parallel_processing(self) -> TestCoverageReport:
        """Assess parallel processing test coverage."""
        existing_tests = [
            "tests/integration/hierarchy/test_parallel_memory_architecture.py"
        ]

        missing_tests = [
            "tests/unit/hierarchy/test_parallel_manager.py",
            "tests/integration/hierarchy/test_parallel_coordination.py",
            "tests/performance/hierarchy/test_parallel_performance.py",
            "tests/integration/hierarchy/test_parallel_error_handling.py",
            "tests/integration/hierarchy/test_parallel_fallback.py"
        ]

        return TestCoverageReport(
            component="Parallel Processing",
            existing_tests=existing_tests,
            missing_tests=missing_tests,
            coverage_percentage=16.7,  # 1 existing / 6 total
            priority="HIGH"
        )

    def _assess_error_handling(self) -> TestCoverageReport:
        """Assess error handling test coverage."""
        existing_tests = [
            "tests/integration/services/test_service_error_handling.py"
        ]

        missing_tests = [
            "tests/unit/error_handling/test_layer_error_recovery.py",
            "tests/unit/error_handling/test_service_error_propagation.py",
            "tests/integration/error_handling/test_system_error_recovery.py",
            "tests/integration/error_handling/test_graceful_degradation.py"
        ]

        return TestCoverageReport(
            component="Error Handling",
            existing_tests=existing_tests,
            missing_tests=missing_tests,
            coverage_percentage=20.0,  # 1 existing / 5 total
            priority="HIGH"
        )

    def _assess_performance(self) -> TestCoverageReport:
        """Assess performance test coverage."""
        existing_tests = [
            "tests/performance/test_query_method_performance.py",
            "tests/performance/test_pgai_batch_embedding.py",
            "tests/performance/test_flush_trigger.py"
        ]

        missing_tests = [
            "tests/performance/test_memory_layer_performance.py",
            "tests/performance/test_parallel_processing_performance.py",
            "tests/performance/test_service_layer_performance.py",
            "tests/performance/test_end_to_end_performance.py",
            "tests/performance/test_scalability.py"
        ]

        return TestCoverageReport(
            component="Performance",
            existing_tests=existing_tests,
            missing_tests=missing_tests,
            coverage_percentage=37.5,  # 3 existing / 8 total
            priority="MEDIUM"
        )

    def generate_report(self) -> str:
        """Generate a comprehensive test coverage report."""
        reports = self.assess_coverage()
        
        report_lines = [
            "# MemFuse Test Coverage Assessment Report",
            "",
            "## Executive Summary",
            "",
            "This report evaluates the current test coverage for MemFuse components",
            "after the recent architecture updates and naming convention changes.",
            "",
            "## Component Coverage Analysis",
            ""
        ]

        # Sort by priority and coverage percentage
        sorted_components = sorted(
            reports.items(),
            key=lambda x: (
                {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[x[1].priority],
                -x[1].coverage_percentage
            )
        )

        for component_name, report in sorted_components:
            report_lines.extend([
                f"### {report.component}",
                f"- **Coverage**: {report.coverage_percentage:.1f}%",
                f"- **Priority**: {report.priority}",
                f"- **Existing Tests**: {len(report.existing_tests)}",
                f"- **Missing Tests**: {len(report.missing_tests)}",
                "",
                "**Existing Tests:**"
            ])
            
            for test in report.existing_tests:
                report_lines.append(f"- ✅ {test}")
            
            report_lines.extend([
                "",
                "**Missing Tests:**"
            ])
            
            for test in report.missing_tests:
                report_lines.append(f"- ❌ {test}")
            
            report_lines.extend(["", "---", ""])

        # Add recommendations
        report_lines.extend([
            "## Recommendations",
            "",
            "### High Priority (Immediate Action Required)",
            "",
            "1. **Memory Layers Testing** - Critical for new M0/M1/M2 architecture",
            "2. **Interface Testing** - Essential for Service Layer decoupling",
            "3. **Parallel Processing Testing** - Core to the new architecture",
            "4. **Error Handling Testing** - Critical for production reliability",
            "",
            "### Medium Priority (Next Sprint)",
            "",
            "1. **PgAI Integration Testing** - Improve embedding system reliability",
            "2. **Services Testing** - Complete service layer coverage",
            "3. **Performance Testing** - Ensure scalability requirements",
            "",
            "### Low Priority (Future Iterations)",
            "",
            "1. **Configuration Testing** - Improve configuration management",
            "",
            "## Implementation Plan",
            "",
            "1. **Week 1**: Implement high-priority missing tests",
            "2. **Week 2**: Add medium-priority tests",
            "3. **Week 3**: Performance and integration tests",
            "4. **Week 4**: Low-priority and edge case tests"
        ])

        return "\n".join(report_lines)

    def identify_critical_gaps(self) -> List[str]:
        """Identify the most critical testing gaps."""
        reports = self.assess_coverage()
        
        critical_gaps = []
        
        for component_name, report in reports.items():
            if report.priority == "HIGH" and report.coverage_percentage < 50:
                critical_gaps.extend([
                    f"CRITICAL: {report.component} has only {report.coverage_percentage:.1f}% coverage",
                    f"Missing: {', '.join(report.missing_tests[:3])}..."
                ])
        
        return critical_gaps


def main():
    """Main function to run coverage assessment."""
    project_root = Path(__file__).parent.parent
    assessment = TestCoverageAssessment(project_root)
    
    # Generate full report
    report = assessment.generate_report()
    print(report)
    
    # Identify critical gaps
    print("\n" + "="*50)
    print("CRITICAL GAPS SUMMARY")
    print("="*50)
    
    critical_gaps = assessment.identify_critical_gaps()
    for gap in critical_gaps:
        print(f"⚠️  {gap}")


if __name__ == "__main__":
    main()
