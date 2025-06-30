import crosstl
import sys
import os


def main():
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    # Test with the new organized structure
    try:
        # Test graphics shaders
        print("Testing graphics shaders...")
        
        # Test simple shader
        try:
            crosstl.translate(
                "graphics/SimpleShader.cgl",
                backend="metal",
                save_shader="output/SimpleShader.metal",
            )
            crosstl.translate(
                "graphics/SimpleShader.cgl",
                backend="opengl",
                save_shader="output/SimpleShader.glsl",
            )
            crosstl.translate(
                "graphics/SimpleShader.cgl",
                backend="directx",
                save_shader="output/SimpleShader.hlsl",
            )
            crosstl.translate(
                "graphics/SimpleShader.cgl",
                backend="rust",
                save_shader="output/SimpleShader.rs",
            )
            crosstl.translate(
                "graphics/SimpleShader.cgl",
                backend="cuda",
                save_shader="output/SimpleShader.cu",
            )
            crosstl.translate(
                "graphics/SimpleShader.cgl",
                backend="hip",
                save_shader="output/SimpleShader.hip",
            )
            print("✅ Simple shader translation successful!")
        except Exception as e:
            print(f"❌ Simple shader translation failed: {e}")

        # Test Perlin noise shader
        try:
            crosstl.translate(
                "graphics/PerlinNoise.cgl",
                backend="metal",
                save_shader="output/PerlinNoise.metal",
            )
            crosstl.translate(
                "graphics/PerlinNoise.cgl",
                backend="opengl",
                save_shader="output/PerlinNoise.glsl",
            )
            crosstl.translate(
                "graphics/PerlinNoise.cgl",
                backend="directx",
                save_shader="output/PerlinNoise.hlsl",
            )
            crosstl.translate(
                "graphics/PerlinNoise.cgl",
                backend="rust",
                save_shader="output/PerlinNoise.rs",
            )
            crosstl.translate(
                "graphics/PerlinNoise.cgl",
                backend="cuda",
                save_shader="output/PerlinNoise.cu",
            )
            crosstl.translate(
                "graphics/PerlinNoise.cgl",
                backend="hip",
                save_shader="output/PerlinNoise.hip",
            )
            print("✅ Perlin noise shader translation successful!")
        except Exception as e:
            print(f"❌ Perlin noise shader translation failed: {e}")

        # Test array test (advanced features)
        try:
            crosstl.translate(
                "advanced/ArrayTest.cgl", 
                backend="metal", 
                save_shader="output/ArrayTest.metal"
            )
            crosstl.translate(
                "advanced/ArrayTest.cgl", 
                backend="opengl", 
                save_shader="output/ArrayTest.glsl"
            )
            crosstl.translate(
                "advanced/ArrayTest.cgl", 
                backend="directx", 
                save_shader="output/ArrayTest.hlsl"
            )
            crosstl.translate(
                "advanced/ArrayTest.cgl", 
                backend="vulkan", 
                save_shader="output/ArrayTest.spirv"
            )
            crosstl.translate(
                "advanced/ArrayTest.cgl", 
                backend="rust", 
                save_shader="output/ArrayTest.rs"
            )
            crosstl.translate(
                "advanced/ArrayTest.cgl", 
                backend="cuda", 
                save_shader="output/ArrayTest.cu"
            )
            crosstl.translate(
                "advanced/ArrayTest.cgl", 
                backend="hip", 
                save_shader="output/ArrayTest.hip"
            )
            print("✅ Array test shader translation successful!")
        except Exception as e:
            print(f"❌ Array test shader translation failed: {e}")

        # Try the complex shader
        try:
            crosstl.translate(
                "graphics/ComplexShader.cgl",
                backend="metal",
                save_shader="output/ComplexShader.metal",
            )
            crosstl.translate(
                "graphics/ComplexShader.cgl",
                backend="opengl",
                save_shader="output/ComplexShader.glsl",
            )
            crosstl.translate(
                "graphics/ComplexShader.cgl",
                backend="directx",
                save_shader="output/ComplexShader.hlsl",
            )
            crosstl.translate(
                "graphics/ComplexShader.cgl",
                backend="rust",
                save_shader="output/ComplexShader.rs",
            )
            crosstl.translate(
                "graphics/ComplexShader.cgl",
                backend="cuda",
                save_shader="output/ComplexShader.cu",
            )
            crosstl.translate(
                "graphics/ComplexShader.cgl",
                backend="hip",
                save_shader="output/ComplexShader.hip",
            )
            print("✅ Complex shader translation successful!")
        except Exception as e:
            print(f"❌ Complex shader translation failed: {e}")

        print("\n🎉 Basic shader tests completed! Run 'python advanced_test.py' for comprehensive testing.")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
