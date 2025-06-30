#!/usr/bin/env python3
"""
CrossGL Example Conversion Test
Comprehensive test to verify translation functionality across all backends and example categories.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crosstl


def main():
    """Test comprehensive translation functionality across all example categories."""
    
    # Define the organized example structure
    examples_by_category = {
        "graphics": [
            "SimpleShader.cgl",
            "PerlinNoise.cgl", 
            "ComplexShader.cgl"
        ],
        "advanced": [
            "ArrayTest.cgl",
            "GenericPatternMatching.cgl"
        ],
        "compute": [
            "ParticleSimulation.cgl"
        ],
        "cross_platform": [
            "UniversalPBRShader.cgl"
        ],
        "gpu_computing": [
            "MatrixMultiplication.cgl"
        ]
    }

    # Define backend mappings with appropriate extensions
    backends = {
        "metal": ".metal",
        "directx": ".hlsl", 
        "opengl": ".glsl",
        "vulkan": ".spirv",
        "rust": ".rs",
        "mojo": ".mojo",
        "cuda": ".cu",
        "hip": ".hip",
        "slang": ".slang",
    }

    # Backend compatibility matrix - some examples work better with certain backends
    backend_compatibility = {
        "graphics": ["metal", "directx", "opengl", "vulkan", "rust", "mojo", "cuda", "hip", "slang"],
        "advanced": ["metal", "directx", "opengl", "vulkan", "rust", "mojo", "cuda", "hip", "slang"],
        "compute": ["metal", "directx", "opengl", "vulkan", "cuda", "hip"], 
        "cross_platform": ["metal", "directx", "opengl", "vulkan", "rust"],
        "gpu_computing": ["cuda", "hip", "mojo", "rust"]
    }

    print("🚀 CrossGL Comprehensive Translation Test")
    print("=" * 60)

    # Ensure output directories exist
    for backend in backends:
        os.makedirs(f"output/{backend}", exist_ok=True)

    total_tests = 0
    successful_tests = 0
    failed_tests = []

    # Test each category
    for category, examples in examples_by_category.items():
        print(f"\n📁 Testing {category.upper()} examples:")
        print("-" * 40)
        
        for example in examples:
            example_path = f"{category}/{example}"
            example_name = Path(example).stem
            
            if not Path(example_path).exists():
                print(f"⚠️  Skipping {example} (not found)")
                continue

            print(f"\n🔄 Translating {example_name}:")
            
            # Get compatible backends for this category
            compatible_backends = backend_compatibility.get(category, list(backends.keys()))
            
            for backend in compatible_backends:
                if backend not in backends:
                    continue
                    
                total_tests += 1
                try:
                    # Create organized output structure: output/backend/category/
                    backend_output_dir = f"output/{backend}/{category}"
                    os.makedirs(backend_output_dir, exist_ok=True)
                    
                    output_file = f"{backend_output_dir}/{example_name}{backends[backend]}"
                    
                    # Perform translation
                    result = crosstl.translate(example_path, backend=backend, save_shader=output_file)
                    
                    # Verify the output file was created and has content
                    if Path(output_file).exists() and Path(output_file).stat().st_size > 100:
                        print(f"  ✅ {backend:8} → {output_file}")
                        successful_tests += 1
                    else:
                        print(f"  ⚠️  {backend:8} → Output too small or missing")
                        failed_tests.append((example_name, backend, "Output file too small"))
                        
                except Exception as e:
                    print(f"  ❌ {backend:8} → Error: {str(e)[:50]}...")
                    failed_tests.append((example_name, backend, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TRANSLATION TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")

    if failed_tests:
        print(f"\n❌ Failed translations:")
        for example, backend, error in failed_tests:
            print(f"  • {example} → {backend}: {error[:60]}...")
    
    # Test cross-backend consistency
    print(f"\n🔍 Testing cross-backend consistency...")
    test_cross_backend_consistency()
    
    print(f"\n✨ Translation test complete!")
    print(f"📁 Check output/ directory for organized results by backend and category.")

def test_cross_backend_consistency():
    """Test that the same shader produces valid output across multiple backends."""
    test_shader = "graphics/SimpleShader.cgl"
    if not Path(test_shader).exists():
        print("⚠️  SimpleShader.cgl not found for consistency test")
        return
        
    consistency_backends = ["metal", "directx", "opengl", "vulkan"]
    outputs = {}
    
    for backend in consistency_backends:
        try:
            result = crosstl.translate(test_shader, backend=backend)
            outputs[backend] = len(result) if result else 0
        except Exception:
            outputs[backend] = 0
    
    print(f"  Output sizes: {outputs}")
    non_zero_outputs = [k for k, v in outputs.items() if v > 100]
    print(f"  ✅ {len(non_zero_outputs)}/{len(consistency_backends)} backends produced substantial output")

if __name__ == "__main__":
    main()
