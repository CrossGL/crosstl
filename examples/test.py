import crosstl
import sys
import os


def main():
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    try:
        # Try to translate the complex shader, but don't fail if it doesn't work
        try:
            metal_code = crosstl.translate(
                "ComplexShader.cgl",
                backend="metal",
                save_shader="output/ComplexShader.metal",
            )

            glsl_code = crosstl.translate(
                "ComplexShader.cgl",
                backend="opengl",
                save_shader="output/ComplexShader.glsl",
            )

            hlsl_code = crosstl.translate(
                "ComplexShader.cgl",
                backend="directx",
                save_shader="output/ComplexShader.hlsl",
            )

            rust_code = crosstl.translate(
                "ComplexShader.cgl",
                backend="rust",
                save_shader="output/ComplexShader.rs",
            )

            cuda_code = crosstl.translate(
                "ComplexShader.cgl",
                backend="cuda",
                save_shader="output/ComplexShader.cu",
            )

            hip_code = crosstl.translate(
                "ComplexShader.cgl",
                backend="hip",
                save_shader="output/ComplexShader.hip",
            )

            print("Complex shader translation successful!")
        except Exception as e:
            print(f"Warning: Complex shader translation failed: {e}")

        # Try the Perlin noise shader
        try:
            metal_code = crosstl.translate(
                "PerlinNoise.cgl",
                backend="metal",
                save_shader="output/PerlinNoise.metal",
            )

            glsl_code = crosstl.translate(
                "PerlinNoise.cgl",
                backend="opengl",
                save_shader="output/PerlinNoise.glsl",
            )

            hlsl_code = crosstl.translate(
                "PerlinNoise.cgl",
                backend="directx",
                save_shader="output/PerlinNoise.hlsl",
            )

            rust_code = crosstl.translate(
                "PerlinNoise.cgl",
                backend="rust",
                save_shader="output/PerlinNoise.rs",
            )

            cuda_code = crosstl.translate(
                "PerlinNoise.cgl",
                backend="cuda",
                save_shader="output/PerlinNoise.cu",
            )

            hip_code = crosstl.translate(
                "PerlinNoise.cgl",
                backend="hip",
                save_shader="output/PerlinNoise.hip",
            )

            print("Perlin noise shader translation successful!")
        except Exception as e:
            print(f"Warning: Perlin noise shader translation failed: {e}")

        # Try the array test shader
        try:
            metal_code = crosstl.translate(
                "ArrayTest.cgl", backend="metal", save_shader="output/ArrayTest.metal"
            )

            glsl_code = crosstl.translate(
                "ArrayTest.cgl", backend="opengl", save_shader="output/ArrayTest.glsl"
            )

            hlsl_code = crosstl.translate(
                "ArrayTest.cgl", backend="directx", save_shader="output/ArrayTest.hlsl"
            )

            spirv_code = crosstl.translate(
                "ArrayTest.cgl", backend="vulkan", save_shader="output/ArrayTest.spirv"
            )

            rust_code = crosstl.translate(
                "ArrayTest.cgl", backend="rust", save_shader="output/ArrayTest.rs"
            )

            cuda_code = crosstl.translate(
                "ArrayTest.cgl", backend="cuda", save_shader="output/ArrayTest.cu"
            )

            hip_code = crosstl.translate(
                "ArrayTest.cgl", backend="hip", save_shader="output/ArrayTest.hip"
            )

            print("Array test shader translation successful!")
        except Exception as e:
            print(f"Warning: Array test shader translation failed: {e}")

        # Use the simple shader as a fallback - this should always work
        metal_code = crosstl.translate(
            "SimpleShader.cgl", backend="metal", save_shader="output/SimpleShader.metal"
        )

        glsl_code = crosstl.translate(
            "SimpleShader.cgl", backend="opengl", save_shader="output/SimpleShader.glsl"
        )

        hlsl_code = crosstl.translate(
            "SimpleShader.cgl",
            backend="directx",
            save_shader="output/SimpleShader.hlsl",
        )

        rust_code = crosstl.translate(
            "SimpleShader.cgl",
            backend="rust",
            save_shader="output/SimpleShader.rs",
        )

        cuda_code = crosstl.translate(
            "SimpleShader.cgl",
            backend="cuda",
            save_shader="output/SimpleShader.cu",
        )

        hip_code = crosstl.translate(
            "SimpleShader.cgl",
            backend="hip",
            save_shader="output/SimpleShader.hip",
        )

        print("Simple shader translation successful!")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
