#!/usr/bin/env python3
"""
CrossGL Example Conversion Test
Simple test to verify basic translation functionality across backends.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crosstl
from pathlib import Path


def main():
    """Test basic translation functionality."""
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    # Test examples with supported backends
    examples = [
        "graphics/SimpleShader.cgl",
        "graphics/PerlinNoise.cgl",
        "graphics/ComplexShader.cgl",
        "advanced/ArrayTest.cgl",
    ]

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

    print("Testing CrossGL translations...")

    for example in examples:
        if not Path(example).exists():
            print(f"⚠️  Skipping {example} (not found)")
            continue

        example_name = Path(example).stem
        print(f"\nTranslating {example_name}:")

        for backend, ext in backends.items():
            try:
                output_file = f"output/{example_name}{ext}"
                crosstl.translate(example, backend=backend, save_shader=output_file)
                print(f"  ✅ {backend}")
            except Exception as e:
                print(f"  ❌ {backend}: {str(e)[:60]}...")

    print(f"\nTranslation test complete. Check output/ directory for results.")


if __name__ == "__main__":
    main()
