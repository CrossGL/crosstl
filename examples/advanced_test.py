#!/usr/bin/env python3
"""
CrossGL Advanced Feature Test Suite

This script demonstrates the advanced IR capabilities by testing complex shaders
across all supported backends including graphics APIs, GPU computing platforms,
and modern programming languages.

Categories tested:
- Graphics Shaders (vertex, fragment, compute)
- GPU Computing (CUDA/HIP kernels)
- Advanced Language Features (generics, pattern matching)
- Cross-Platform Compatibility
"""

import crosstl
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class TestResult:
    def __init__(self, success: bool, time_taken: float, error: Optional[str] = None):
        self.success = success
        self.time_taken = time_taken
        self.error = error


class AdvancedTester:
    def __init__(self):
        self.results: Dict[str, Dict[str, TestResult]] = {}
        self.supported_backends = [
            'metal', 'directx', 'opengl', 'vulkan', 
            'rust', 'mojo', 'cuda', 'hip', 'slang'
        ]
        
        # Example categories with their files
        self.example_categories = {
            'graphics': [
                'SimpleShader.cgl',
                'PerlinNoise.cgl', 
                'ComplexShader.cgl'
            ],
            'compute': [
                'ParticleSimulation.cgl'
            ],
            'gpu_computing': [
                'MatrixMultiplication.cgl'
            ],
            'advanced': [
                'ArrayTest.cgl',
                'GenericPatternMatching.cgl'
            ],
            'cross_platform': [
                'UniversalPBRShader.cgl'
            ]
        }
        
        # Backend-specific features to test
        self.backend_features = {
            'metal': ['graphics', 'compute', 'cross_platform'],
            'directx': ['graphics', 'compute', 'cross_platform'],
            'opengl': ['graphics', 'compute', 'cross_platform'],
            'vulkan': ['graphics', 'compute', 'cross_platform'],
            'rust': ['advanced', 'cross_platform'],
            'mojo': ['advanced', 'gpu_computing'],
            'cuda': ['gpu_computing', 'compute'],
            'hip': ['gpu_computing', 'compute'],
            'slang': ['graphics', 'cross_platform']
        }

    def setup_output_directories(self):
        """Create organized output directories for each category and backend."""
        for category in self.example_categories.keys():
            for backend in self.supported_backends:
                output_dir = Path(f"output/{category}/{backend}")
                output_dir.mkdir(parents=True, exist_ok=True)

    def get_backend_extension(self, backend: str) -> str:
        """Get the appropriate file extension for each backend."""
        extensions = {
            'metal': '.metal',
            'directx': '.hlsl', 
            'opengl': '.glsl',
            'vulkan': '.spirv',
            'rust': '.rs',
            'mojo': '.mojo',
            'cuda': '.cu',
            'hip': '.hip',
            'slang': '.slang'
        }
        return extensions.get(backend, '.txt')

    def test_translation(self, category: str, example: str, backend: str) -> TestResult:
        """Test translation of a specific example to a specific backend."""
        input_file = Path(category) / example
        
        # Check if this backend supports this category
        if category not in self.backend_features.get(backend, []):
            return TestResult(True, 0.0)  # Skip unsupported combinations
        
        if not input_file.exists():
            return TestResult(False, 0.0, f"Input file {input_file} not found")
        
        extension = self.get_backend_extension(backend)
        example_name = Path(example).stem
        output_file = f"output/{category}/{backend}/{example_name}{extension}"
        
        start_time = time.time()
        
        try:
            result = crosstl.translate(
                str(input_file),
                backend=backend,
                save_shader=output_file
            )
            
            elapsed = time.time() - start_time
            
            # Verify output file was created and has content
            if not Path(output_file).exists():
                return TestResult(False, elapsed, "Output file was not created")
            
            if Path(output_file).stat().st_size == 0:
                return TestResult(False, elapsed, "Output file is empty")
            
            return TestResult(True, elapsed)
            
        except Exception as e:
            elapsed = time.time() - start_time
            return TestResult(False, elapsed, str(e))

    def test_category(self, category: str) -> Dict[str, Dict[str, TestResult]]:
        """Test all examples in a category across all appropriate backends."""
        print(f"\n{'='*60}")
        print(f"Testing {category.upper()} Examples")
        print(f"{'='*60}")
        
        category_results = {}
        
        for example in self.example_categories[category]:
            print(f"\nTesting {example}...")
            example_results = {}
            
            for backend in self.supported_backends:
                if category in self.backend_features.get(backend, []):
                    print(f"  → {backend.ljust(10)}", end="")
                    result = self.test_translation(category, example, backend)
                    
                    if result.success:
                        print(f"✅ ({result.time_taken:.2f}s)")
                    else:
                        print(f"❌ {result.error}")
                    
                    example_results[backend] = result
                else:
                    print(f"  → {backend.ljust(10)}⏭️  (not supported)")
            
            category_results[example] = example_results
        
        return category_results

    def run_comprehensive_test(self):
        """Run the complete test suite."""
        print("🚀 CrossGL Advanced Feature Test Suite")
        print("Testing comprehensive IR capabilities across all backends")
        
        self.setup_output_directories()
        
        total_tests = 0
        successful_tests = 0
        
        for category in self.example_categories.keys():
            category_results = self.test_category(category)
            self.results[category] = category_results
            
            # Count results
            for example_results in category_results.values():
                for result in example_results.values():
                    total_tests += 1
                    if result.success:
                        successful_tests += 1
        
        self.print_summary(total_tests, successful_tests)
        return successful_tests == total_tests

    def print_summary(self, total_tests: int, successful_tests: int):
        """Print a comprehensive summary of all test results."""
        print(f"\n{'='*80}")
        print("📊 COMPREHENSIVE TEST SUMMARY")
        print(f"{'='*80}")
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        print(f"Overall Success Rate: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
        
        # Per-category breakdown
        print(f"\n📂 Results by Category:")
        for category, category_results in self.results.items():
            cat_total = 0
            cat_success = 0
            
            for example_results in category_results.values():
                for result in example_results.values():
                    cat_total += 1
                    if result.success:
                        cat_success += 1
            
            cat_rate = (cat_success / cat_total * 100) if cat_total > 0 else 0
            status = "✅" if cat_rate == 100 else "⚠️" if cat_rate >= 80 else "❌"
            print(f"  {status} {category.ljust(15)}: {cat_success}/{cat_total} ({cat_rate:.1f}%)")
        
        # Per-backend breakdown
        print(f"\n🔧 Results by Backend:")
        backend_stats = {}
        
        for category_results in self.results.values():
            for example_results in category_results.values():
                for backend, result in example_results.items():
                    if backend not in backend_stats:
                        backend_stats[backend] = {'total': 0, 'success': 0}
                    
                    backend_stats[backend]['total'] += 1
                    if result.success:
                        backend_stats[backend]['success'] += 1
        
        for backend in sorted(backend_stats.keys()):
            stats = backend_stats[backend]
            rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            status = "✅" if rate == 100 else "⚠️" if rate >= 80 else "❌"
            print(f"  {status} {backend.ljust(10)}: {stats['success']}/{stats['total']} ({rate:.1f}%)")
        
        # Performance analysis
        print(f"\n⚡ Performance Analysis:")
        total_time = 0
        times_by_backend = {}
        
        for category_results in self.results.values():
            for example_results in category_results.values():
                for backend, result in example_results.items():
                    total_time += result.time_taken
                    if backend not in times_by_backend:
                        times_by_backend[backend] = []
                    times_by_backend[backend].append(result.time_taken)
        
        print(f"  Total translation time: {total_time:.2f}s")
        
        for backend in sorted(times_by_backend.keys()):
            times = times_by_backend[backend]
            avg_time = sum(times) / len(times) if times else 0
            print(f"  {backend.ljust(10)}: avg {avg_time:.3f}s")
        
        # Feature coverage
        print(f"\n🎯 Feature Coverage:")
        features_tested = {
            'Graphics Pipelines': len(self.example_categories['graphics']),
            'Compute Shaders': len(self.example_categories['compute']),
            'GPU Computing': len(self.example_categories['gpu_computing']),
            'Advanced Language Features': len(self.example_categories['advanced']),
            'Cross-Platform Features': len(self.example_categories['cross_platform'])
        }
        
        for feature, count in features_tested.items():
            print(f"  ✓ {feature}: {count} examples")
        
        # Error analysis
        print(f"\n🔍 Error Analysis:")
        error_types = {}
        
        for category_results in self.results.values():
            for example_results in category_results.values():
                for result in example_results.values():
                    if not result.success and result.error:
                        error_type = result.error.split(':')[0] if ':' in result.error else result.error
                        error_types[error_type] = error_types.get(error_type, 0) + 1
        
        if error_types:
            for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  • {error}: {count} occurrences")
        else:
            print("  🎉 No errors detected!")

    def test_advanced_features(self):
        """Test specific advanced IR features."""
        print(f"\n{'='*60}")
        print("🎯 ADVANCED IR FEATURE TESTS")
        print(f"{'='*60}")
        
        features_to_test = [
            ("Generic Types", "advanced/GenericPatternMatching.cgl", ['rust', 'mojo']),
            ("Pattern Matching", "advanced/GenericPatternMatching.cgl", ['rust']),
            ("Compute Kernels", "compute/ParticleSimulation.cgl", ['metal', 'directx', 'opengl', 'vulkan']),
            ("GPU Computing", "gpu_computing/MatrixMultiplication.cgl", ['cuda', 'hip']),
            ("Cross-Platform PBR", "cross_platform/UniversalPBRShader.cgl", ['metal', 'directx', 'opengl', 'vulkan']),
            ("Complex Shaders", "graphics/ComplexShader.cgl", ['metal', 'directx', 'opengl', 'rust', 'mojo'])
        ]
        
        for feature_name, example_path, backends in features_to_test:
            print(f"\nTesting {feature_name}:")
            
            for backend in backends:
                try:
                    extension = self.get_backend_extension(backend)
                    output_path = f"output/features/{feature_name.replace(' ', '_').lower()}_{backend}{extension}"
                    
                    # Ensure output directory exists
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    
                    start_time = time.time()
                    result = crosstl.translate(example_path, backend=backend, save_shader=output_path)
                    elapsed = time.time() - start_time
                    
                    print(f"  ✅ {backend.ljust(10)} ({elapsed:.2f}s)")
                    
                except Exception as e:
                    print(f"  ❌ {backend.ljust(10)} - {str(e)[:50]}...")


def main():
    """Main test execution."""
    print("🌟 Initializing CrossGL Advanced Feature Test Suite...")
    
    # Change to examples directory
    if not os.path.exists("graphics") and os.path.exists("examples"):
        os.chdir("examples")
    
    # Create tester instance
    tester = AdvancedTester()
    
    try:
        # Run comprehensive tests
        success = tester.run_comprehensive_test()
        
        # Test advanced features specifically
        tester.test_advanced_features()
        
        print(f"\n{'='*80}")
        if success:
            print("🎉 ALL TESTS PASSED! The new IR is production-ready!")
        else:
            print("⚠️  Some tests failed. Check the summary above for details.")
        print(f"{'='*80}")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Test suite interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\n\n💥 Fatal error in test suite: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 