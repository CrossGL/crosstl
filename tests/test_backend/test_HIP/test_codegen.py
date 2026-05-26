"""Test HIP Code Generation"""

import pytest
from crosstl import translate
from crosstl.backend.HIP.HipLexer import HipLexer
from crosstl.backend.HIP.HipParser import HipParser
from crosstl.backend.HIP.HipCrossGLCodeGen import HipToCrossGLConverter


class TestHipCodeGen:
    def test_hip_flat_builtin_alias_conversion(self):
        """Test HIP flat builtin aliases convert to CrossGL builtins."""
        code = """
        __global__ void kernel(float* out) {
            out[hipThreadIdx_x] = hipBlockDim_y + warpSize;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "out[gl_LocalInvocationID.x] = (gl_WorkGroupSize.y + 32);" in result
        assert "hipThreadIdx_x" not in result
        assert "hipBlockDim_y" not in result
        assert "warpSize" not in result

    def test_basic_kernel_conversion(self):
        """Test basic kernel to CrossGL conversion"""
        code = """
        #include <hip/hip_runtime.h>
        #define HIP_SCALE 2
        __constant__ int kLimit = 64;
        __managed__ float managedValue;
        __launch_bounds__(256, 2) __global__ void simple_kernel() {
            __shared__ float shared_data[256];
            int idx = HIP_SCALE;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// HIP to CrossGL conversion" in result
        assert "// HIP runtime functionality built-in" in result
        assert "// define HIP_SCALE 2" in result
        assert "@group(0) @binding(0) var<uniform> kLimit: i32 = 64;" in result
        assert "// HIP managed memory: managedValue" in result
        assert "var managedValue: f32;" in result
        assert "// Kernel: simple_kernel" in result
        assert "// HIP launch bounds: (256, 2)" in result
        assert "var<workgroup> shared_data: array<f32, 256>;" in result
        assert "PreprocessorNode" not in result
        assert "__constant__" not in result
        assert "__managed__" not in result
        assert "__shared__" not in result

    def test_device_function_conversion(self):
        """Test device function conversion"""
        code = """
        __device__ float add(float a, float b) {
            return a + b;
        }
        __forceinline__ __device__ float fast_add(float a, float b) {
            return a + b;
        }
        __noinline__ __device__ float slow_sub(float a, float b) {
            return a - b;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// HIP to CrossGL conversion" in result
        assert "// Function: add" in result
        assert "// Function: fast_add" in result
        assert "// Function: slow_sub" in result
        assert "f32 fast_add(f32 a, f32 b)" in result
        assert "f32 slow_sub(f32 a, f32 b)" in result

    def test_device_function_body_emitted_when_kernel_calls_it(self):
        """Test device helpers are emitted when kernels call them."""
        code = """
        __device__ float add(float a, float b) {
            return a + b;
        }

        __global__ void kernel(float* out) {
            out[0] = add(1.0f, 2.0f);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "f32 add(f32 a, f32 b) {" in result
        assert "return (a + b);" in result
        assert "out[0] = add(1.0f, 2.0f);" in result

    def test_multiple_kernels_conversion(self):
        """Test multiple kernels conversion"""
        code = """
        __global__ void kernel1() {
            int x = 1;
        }
        
        __global__ void kernel2() {
            int x = 2;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// Kernel: kernel1" in result
        assert "// Kernel: kernel2" in result

    def test_type_conversion(self):
        """Test HIP type to CrossGL type conversion"""
        codegen = HipToCrossGLConverter()

        # Test basic types
        assert codegen.convert_hip_type_to_crossgl("int") == "i32"
        assert codegen.convert_hip_type_to_crossgl("float") == "f32"
        assert codegen.convert_hip_type_to_crossgl("double") == "f64"
        assert codegen.convert_hip_type_to_crossgl("bool") == "bool"
        assert codegen.convert_hip_type_to_crossgl("void") == "void"
        assert codegen.convert_hip_type_to_crossgl("float *") == "ptr<f32>"
        assert codegen.convert_hip_type_to_crossgl("void * *") == "ptr<ptr<void>>"
        assert codegen.convert_hip_type_to_crossgl("void * []") == "array<ptr<void>>"
        assert codegen.convert_hip_type_to_crossgl("hipArray_t") == "ptr<void>"
        assert codegen.convert_hip_type_to_crossgl("hipArray_t *") == "ptr<ptr<void>>"
        assert codegen.convert_hip_type_to_crossgl("hipTextureObject_t") == "sampler"
        assert (
            codegen.convert_hip_type_to_crossgl("const hipTextureObject_t *")
            == "ptr<sampler>"
        )
        assert codegen.convert_hip_type_to_crossgl("hipSurfaceObject_t") == "image2D"
        assert (
            codegen.convert_hip_type_to_crossgl("hipSurfaceObject_t []")
            == "array<image2D>"
        )
        assert codegen.convert_hip_type_to_crossgl("texture<float4, 2>") == "sampler2D"
        assert (
            codegen.convert_hip_type_to_crossgl("texture<float4, hipTextureType3D>")
            == "sampler3D"
        )
        assert codegen.convert_hip_type_to_crossgl("surface<void, 2>") == "image2D"
        assert (
            codegen.convert_hip_type_to_crossgl("surface<void, hipSurfaceType3D>")
            == "image3D"
        )
        assert (
            codegen.convert_hip_type_to_crossgl(
                "surface<void, hipSurfaceType1DLayered>"
            )
            == "image1DArray"
        )
        assert (
            codegen.convert_hip_type_to_crossgl(
                "surface<void, hipSurfaceTypeCubemapLayered>"
            )
            == "imageCubeArray"
        )

    def test_function_conversion(self):
        """Test HIP function to CrossGL function conversion"""
        codegen = HipToCrossGLConverter()

        # Test math functions
        assert codegen.convert_hip_builtin_function("sqrtf") == "sqrt"
        assert codegen.convert_hip_builtin_function("sinf") == "sin"
        assert codegen.convert_hip_builtin_function("cosf") == "cos"
        assert codegen.convert_hip_builtin_function("fmodf") == "mod"
        assert codegen.convert_hip_builtin_function("fmod") == "mod"
        assert codegen.convert_hip_builtin_function("atan2f") == "atan2"
        assert codegen.convert_hip_builtin_function("asinf") == "asin"
        assert codegen.convert_hip_builtin_function("acosf") == "acos"
        assert codegen.convert_hip_builtin_function("atanf") == "atan"
        assert codegen.convert_hip_builtin_function("rsqrtf") == "inversesqrt"
        assert codegen.convert_hip_builtin_function("rsqrt") == "inversesqrt"
        assert codegen.convert_hip_builtin_function("roundf") == "round"
        assert codegen.convert_hip_builtin_function("truncf") == "trunc"
        assert codegen.convert_hip_builtin_function("exp2f") == "exp2"
        assert codegen.convert_hip_builtin_function("log2f") == "log2"
        assert (
            codegen.convert_hip_builtin_function("__syncthreads") == "workgroupBarrier"
        )

    def test_fmod_builtins_convert_to_crossgl_mod(self):
        """Test HIP fmod functions convert back to CrossGL mod."""
        code = """
        __global__ void wrap(float* out, float x) {
            out[0] = fmodf(x, 1.0f);
            out[1] = fmod(x, 2.0);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "out[0] = mod(x, 1.0f);" in result
        assert "out[1] = mod(x, 2.0);" in result
        assert "fmod" not in result

    def test_atan2_builtins_convert_to_crossgl_atan2(self):
        """Test HIP atan2f converts back to CrossGL atan2."""
        code = """
        __global__ void angle(float* out, float y, float x) {
            out[0] = atan2f(y, x);
            out[1] = atan2(y, x);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "out[0] = atan2(y, x);" in result
        assert "out[1] = atan2(y, x);" in result
        assert "atan2f" not in result

    def test_atomic_builtins_parse_and_convert_to_crossgl(self):
        """Test HIP atomic builtins parse and convert back to CrossGL names."""
        code = """
        __global__ void atomics(int* out, int* expected) {
            atomicAdd(out, 1);
            hipAtomicSub(out, 1);
            atomicMin(out, 0);
            hipAtomicMax(out, 7);
            hipAtomicExch(out, 2);
            atomicCAS(expected, 0, 3);
            atomicAnd(out, 1);
            hipAtomicOr(out, 2);
            atomicXor(out, 3);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "atomicAdd(out, 1);" in result
        assert "atomicSub(out, 1);" in result
        assert "atomicMin(out, 0);" in result
        assert "atomicMax(out, 7);" in result
        assert "atomicExchange(out, 2);" in result
        assert "atomicCompareExchange(expected, 0, 3);" in result
        assert "atomicAnd(out, 1);" in result
        assert "atomicOr(out, 2);" in result
        assert "atomicXor(out, 3);" in result
        assert "hipAtomicExch" not in result
        assert "hipAtomic" not in result
        assert "atomicCAS(" not in result

    def test_user_defined_atomic_name_call_does_not_convert_to_builtin(self):
        """Test user-defined HIP atomic names shadow builtin conversion."""
        code = """
        int hipAtomicExch(int value) {
            return value + 1;
        }

        __global__ void kernel(int* out) {
            int value = hipAtomicExch(7);
            atomicAdd(out, 1);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "i32 hipAtomicExch(i32 value)" in result
        assert "var value: i32 = hipAtomicExch(7);" in result
        assert "atomicAdd(out, 1);" in result
        assert "atomicExchange(7)" not in result

    def test_user_defined_atomic_name_declared_later_does_not_convert_to_builtin(
        self,
    ):
        """Test later user-defined HIP atomic names shadow builtin conversion."""
        code = """
        __global__ void kernel(int* out) {
            int value = hipAtomicExch(7);
            atomicAdd(out, 1);
        }

        int hipAtomicExch(int value) {
            return value + 1;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "i32 hipAtomicExch(i32 value)" in result
        assert "var value: i32 = hipAtomicExch(7);" in result
        assert "atomicAdd(out, 1);" in result
        assert "atomicExchange(7)" not in result

    def test_lerp_builtin_converts_to_crossgl_mix(self):
        """Test HIP lerp converts back to CrossGL mix."""
        code = """
        __global__ void blend(float* out, float a, float b, float t) {
            out[0] = lerp(a, b, t);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "out[0] = mix(a, b, t);" in result
        assert "out[0] = lerp(a, b, t);" not in result

    def test_lerp_with_bool_selector_converts_to_crossgl_mix(self):
        """Test HIP lerp conversion preserves selector-shaped mix calls."""
        code = """
        __global__ void blend(float* out, float a, float b, bool choose_b) {
            out[0] = lerp(a, b, choose_b);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "out[0] = mix(a, b, choose_b);" in result
        assert "out[0] = lerp(a, b, choose_b);" not in result

    def test_scalar_bool_ternary_selector_is_preserved(self):
        """Test HIP scalar bool ternaries round-trip as CrossGL ternaries."""
        code = """
        __global__ void choose(float* out, bool choose_b, float a, float b) {
            out[0] = choose_b ? b : a;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "out[0] = (choose_b ? b : a);" in result

    def test_user_defined_lerp_call_does_not_convert_to_mix(self):
        """Test user-defined HIP functions shadow builtin call conversion."""
        code = """
        float lerp(float x) {
            return x;
        }

        float tex2D(float x) {
            return x;
        }

        __global__ void kernel(float* out, float x) {
            out[0] = lerp(x);
            out[1] = tex2D(x);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "f32 lerp(f32 x) {" in result
        assert "f32 tex2D(f32 x) {" in result
        assert "out[0] = lerp(x);" in result
        assert "out[1] = tex2D(x);" in result
        assert "out[0] = mix(x);" not in result
        assert "out[1] = texture(x);" not in result

    def test_hyperbolic_builtins_convert_to_crossgl(self):
        """Test HIP hyperbolic functions convert back to CrossGL names."""
        code = """
        __global__ void hyper(float* out, float x) {
            out[0] = sinhf(x);
            out[1] = coshf(x);
            out[2] = tanhf(x);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "out[0] = sinh(x);" in result
        assert "out[1] = cosh(x);" in result
        assert "out[2] = tanh(x);" in result
        assert "sinhf" not in result
        assert "coshf" not in result
        assert "tanhf" not in result

    def test_inverse_hyperbolic_builtins_convert_to_crossgl(self):
        """Test HIP inverse hyperbolic functions convert back to CrossGL names."""
        code = """
        __global__ void invhyper(float* out, float x) {
            out[0] = asinhf(x);
            out[1] = acoshf(x);
            out[2] = atanhf(x);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "out[0] = asinh(x);" in result
        assert "out[1] = acosh(x);" in result
        assert "out[2] = atanh(x);" in result
        assert "asinhf" not in result
        assert "acoshf" not in result
        assert "atanhf" not in result

    def test_inverse_trig_builtins_convert_to_crossgl(self):
        """Test HIP inverse trig functions convert back to CrossGL names."""
        code = """
        __global__ void inverse(float* out, float x) {
            out[0] = asinf(x);
            out[1] = acosf(x);
            out[2] = atanf(x);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "out[0] = asin(x);" in result
        assert "out[1] = acos(x);" in result
        assert "out[2] = atan(x);" in result
        assert "asinf" not in result
        assert "acosf" not in result
        assert "atanf" not in result

    def test_extended_math_builtins_convert_to_crossgl(self):
        """Test HIP extended math functions convert back to CrossGL names."""
        code = """
        __global__ void math(float* out, double* precise, float x, double d) {
            out[0] = rsqrtf(x);
            out[1] = roundf(x);
            out[2] = truncf(x);
            out[3] = exp2f(x);
            out[4] = log2f(x);
            precise[0] = rsqrt(d);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "out[0] = inversesqrt(x);" in result
        assert "out[1] = round(x);" in result
        assert "out[2] = trunc(x);" in result
        assert "out[3] = exp2(x);" in result
        assert "out[4] = log2(x);" in result
        assert "precise[0] = inversesqrt(d);" in result
        assert "rsqrtf" not in result
        assert "rsqrt(" not in result
        assert "roundf" not in result
        assert "truncf" not in result
        assert "exp2f" not in result
        assert "log2f" not in result

    def test_struct_conversion(self):
        """Test struct conversion"""
        code = """
        struct Point {
            float x;
            float y;
        };
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// HIP to CrossGL conversion" in result
        assert "struct Point" in result

    def test_empty_program(self):
        """Test empty program conversion"""
        code = ""
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// HIP to CrossGL conversion" in result

    def test_vector_type_conversion(self):
        """Test HIP vector type conversion"""
        codegen = HipToCrossGLConverter()

        # Test vector type mappings
        assert codegen.convert_hip_type_to_crossgl("float2") == "vec2<f32>"
        assert codegen.convert_hip_type_to_crossgl("float3") == "vec3<f32>"
        assert codegen.convert_hip_type_to_crossgl("float4") == "vec4<f32>"
        assert codegen.convert_hip_type_to_crossgl("int2") == "vec2<i32>"
        assert codegen.convert_hip_type_to_crossgl("int3") == "vec3<i32>"
        assert codegen.convert_hip_type_to_crossgl("int4") == "vec4<i32>"

    def test_constructor_style_vector_declaration_conversion(self):
        """Test HIP constructor-style vector declarations convert"""
        code = """
        hipTextureObject_t globalTex;
        hipSurfaceObject_t globalSurf;

        void launch(hipTextureObject_t paramTex, hipSurfaceObject_t paramVolume) {
            dim3 grid(16, 8, 1);
            dim3 block(32);
            float3 v(1.0f, 2.0f, 3.0f);
            double2 d = make_double2(1.0, 2.0);
            uint4 ids = make_uint4(1u, 2u, 3u, 4u);
            uchar2 bytes(1, 2);
            hipTextureObject_t tex;
            hipTextureObject_t textures[2];
            hipTextureObject_t ambiguous;
            hipSurfaceObject_t surf;
            hipSurfaceObject_t lineSurf;
            texture<float4, 2> legacyTex;
            texture<float4, hipTextureTypeCubemap> cubeTex;
            texture<float4, hipTextureTypeCubemapLayered> cubeLayerTex;
            surface<void, hipSurfaceType1D> lineSurface;
            surface<void, hipSurfaceType1DLayered> lineLayerSurface;
            surface<void, 2> legacySurf;
            surface<void, hipSurfaceType3D> volumeSurface;
            surface<void, hipSurfaceTypeCubemap> cubeSurface;
            surface<void, hipSurfaceTypeCubemapLayered> cubeLayerSurface;
            float2 uv = make_float2(0.25f, 0.75f);
            float3 uvw = make_float3(0.25f, 0.75f, 0.5f);
            int2 pixel = make_int2(1, 2);
            float4 sampled = tex2D<float4>(tex, uv.x, uv.y);
            float4 arraySample = tex2D<float4>(textures[1], uv);
            float4 paramSample = tex2D<float4>(paramTex, uv);
            float4 globalSample = tex2D<float4>(globalTex, uv);
            float4 ambiguous2D = tex2D<float4>(ambiguous, uv);
            float4 ambiguous3D = tex3D<float4>(
                ambiguous,
                uvw.x,
                uvw.y,
                uvw.z
            );
            float4 sampledCoord = tex2D<float4>(tex, uv);
            float4 sampledLod = tex2DLod<float4>(tex, uv.x, uv.y, 1.0f);
            float4 sampledGrad = tex2DGrad<float4>(tex, uv, uv, uv);
            float4 legacySample = tex2D<float4>(legacyTex, uv);
            float4 cubeSample = texCubemap<float4>(
                cubeTex,
                uvw.x,
                uvw.y,
                uvw.z
            );
            float4 cubeLayerSample = texCubemapLayered<float4>(
                cubeLayerTex,
                uvw.x,
                uvw.y,
                uvw.z,
                pixel.x
            );
            float4 loaded = surf2Dread<float4>(
                surf,
                pixel.x * sizeof(float4),
                pixel.y
            );
            float4 legacyLoaded = surf2Dread<float4>(
                legacySurf,
                pixel.x * sizeof(float4),
                pixel.y
            );
            float4 paramLoaded = surf3Dread<float4>(
                paramVolume,
                pixel.x * sizeof(float4),
                pixel.y,
                0
            );
            float4 volumeLoaded = surf3Dread<float4>(
                volumeSurface,
                pixel.x * sizeof(float4),
                pixel.y,
                0
            );
            float4 cubeLoaded = surfCubemapread<float4>(
                cubeSurface,
                pixel.x * sizeof(float4),
                pixel.y,
                0
            );
            float4 cubeLayerLoaded = surfCubemapLayeredread<float4>(
                cubeLayerSurface,
                pixel.x * sizeof(float4),
                pixel.y,
                0
            );
            float4 lineLoaded;
            float4 lineLayerLoaded;
            float4 loadedByPointer;
            surf1Dread<float4>(
                &lineLoaded,
                lineSurface,
                pixel.x * sizeof(float4)
            );
            surf1Dread<float4>(&lineLoaded, lineSurf, pixel.x * sizeof(float4));
            surf1DLayeredread<float4>(
                &lineLayerLoaded,
                lineLayerSurface,
                pixel.x * sizeof(float4),
                pixel.y
            );
            surf2Dread<float4>(
                &loadedByPointer,
                surf,
                pixel.x * sizeof(float4),
                pixel.y
            );
            surf2Dwrite(loaded, surf, pixel.x * sizeof(float4), pixel.y);
            surf2Dwrite(loaded, globalSurf, pixel.x * sizeof(float4), pixel.y);
            surf1Dwrite(lineLoaded, lineSurface, pixel.x * sizeof(float4));
            surf1DLayeredwrite(
                lineLayerLoaded,
                lineLayerSurface,
                pixel.x * sizeof(float4),
                pixel.y
            );
            surf2Dwrite(
                legacyLoaded,
                legacySurf,
                pixel.x * sizeof(float4),
                pixel.y
            );
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var globalTex: sampler2D;" in result
        assert "var globalSurf: image2D;" in result
        assert "void launch(sampler2D paramTex, image3D paramVolume)" in result
        assert "var grid: vec3<u32> = vec3<u32>(16, 8, 1);" in result
        assert "var block: vec3<u32> = vec3<u32>(32);" in result
        assert "var v: vec3<f32> = vec3<f32>(1.0f, 2.0f, 3.0f);" in result
        assert "var d: vec2<f64> = vec2<f64>(1.0, 2.0);" in result
        assert "var ids: vec4<u32> = vec4<u32>(1u, 2u, 3u, 4u);" in result
        assert "var bytes: vec2<u8> = vec2<u8>(1, 2);" in result
        assert "var tex: sampler2D;" in result
        assert "var textures: array<sampler2D, 2>;" in result
        assert "var ambiguous: sampler;" in result
        assert "var surf: image2D;" in result
        assert "var lineSurf: image1D;" in result
        assert "var legacyTex: sampler2D;" in result
        assert "var lineSurface: image1D;" in result
        assert "var lineLayerSurface: image1DArray;" in result
        assert "var legacySurf: image2D;" in result
        assert "var cubeTex: samplerCube;" in result
        assert "var cubeLayerTex: samplerCubeArray;" in result
        assert "var volumeSurface: image3D;" in result
        assert "var cubeSurface: imageCube;" in result
        assert "var cubeLayerSurface: imageCubeArray;" in result
        assert "var sampled: vec4<f32> = texture(tex, vec2<f32>(uv.x, uv.y));" in result
        assert "var arraySample: vec4<f32> = texture(textures[1], uv);" in result
        assert "var paramSample: vec4<f32> = texture(paramTex, uv);" in result
        assert "var globalSample: vec4<f32> = texture(globalTex, uv);" in result
        assert "var ambiguous2D: vec4<f32> = texture(ambiguous, uv);" in result
        assert (
            "var ambiguous3D: vec4<f32> = texture("
            "ambiguous, vec3<f32>(uvw.x, uvw.y, uvw.z));" in result
        )
        assert "var sampledCoord: vec4<f32> = texture(tex, uv);" in result
        assert (
            "var sampledLod: vec4<f32> = textureLod("
            "tex, vec2<f32>(uv.x, uv.y), 1.0f);" in result
        )
        assert "var sampledGrad: vec4<f32> = textureGrad(tex, uv, uv, uv);" in result
        assert "var legacySample: vec4<f32> = texture(legacyTex, uv);" in result
        assert (
            "var cubeSample: vec4<f32> = texture("
            "cubeTex, vec3<f32>(uvw.x, uvw.y, uvw.z));" in result
        )
        assert (
            "var cubeLayerSample: vec4<f32> = texture("
            "cubeLayerTex, vec4<f32>(uvw.x, uvw.y, uvw.z, pixel.x));" in result
        )
        assert (
            "var loaded: vec4<f32> = imageLoad(surf, vec2<i32>(pixel.x, pixel.y));"
            in result
        )
        assert (
            "var paramLoaded: vec4<f32> = imageLoad("
            "paramVolume, vec3<i32>(pixel.x, pixel.y, 0));" in result
        )
        assert (
            "var volumeLoaded: vec4<f32> = imageLoad("
            "volumeSurface, vec3<i32>(pixel.x, pixel.y, 0));" in result
        )
        assert (
            "var cubeLoaded: vec4<f32> = imageLoad("
            "cubeSurface, vec3<i32>(pixel.x, pixel.y, 0));" in result
        )
        assert (
            "var cubeLayerLoaded: vec4<f32> = imageLoad("
            "cubeLayerSurface, vec3<i32>(pixel.x, pixel.y, 0));" in result
        )
        assert "lineLoaded = imageLoad(lineSurface, pixel.x);" in result
        assert "lineLoaded = imageLoad(lineSurf, pixel.x);" in result
        assert (
            "lineLayerLoaded = imageLoad("
            "lineLayerSurface, vec2<i32>(pixel.x, pixel.y));" in result
        )
        assert (
            "loadedByPointer = imageLoad(surf, vec2<i32>(pixel.x, pixel.y));" in result
        )
        assert (
            "var legacyLoaded: vec4<f32> = imageLoad("
            "legacySurf, vec2<i32>(pixel.x, pixel.y));" in result
        )
        assert "imageStore(surf, vec2<i32>(pixel.x, pixel.y), loaded);" in result
        assert "imageStore(globalSurf, vec2<i32>(pixel.x, pixel.y), loaded);" in result
        assert "imageStore(lineSurface, pixel.x, lineLoaded);" in result
        assert (
            "imageStore(lineLayerSurface, vec2<i32>(pixel.x, pixel.y), "
            "lineLayerLoaded);" in result
        )
        assert (
            "imageStore(legacySurf, vec2<i32>(pixel.x, pixel.y), legacyLoaded);"
            in result
        )
        assert "tex2D<" not in result
        assert "surf2D" not in result
        assert "surf1D" not in result

    def test_kernel_launch_conversion(self):
        """Test HIP kernel launch configuration conversion"""
        code = """
        void host(float* data, int stream) {
            dim3 grid(16);
            dim3 block(32);
            kernel<<<grid, block, 128, stream>>>(data, 1);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// Kernel launch: kernel<<<grid, block, 128, stream>>>()" in result
        assert "// Arguments: data, 1" in result

    def test_templated_kernel_launch_conversion(self):
        """Test HIP template-id kernel launch conversion"""
        code = """
        template <typename T>
        __global__ void scale(T* data, T factor) {
            data[threadIdx.x] *= factor;
        }

        void host(float* data) {
            scale<float><<<dim3(1), dim3(32)>>>(data, 2.0f);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// Kernel launch: scale<float><<<vec3<u32>(1), vec3<u32>(32)>>>()"
            in result
        )
        assert "// Arguments: data, 2.0f" in result

    def test_computed_kernel_launch_config_conversion(self):
        """Test HIP computed kernel launch configuration conversion"""
        code = """
        void host(float* data, int n, int stream) {
            int blockSize = 128;
            kernel<<<(n + blockSize - 1) / blockSize,
                     blockSize,
                     sizeof(float) * blockSize,
                     stream>>>(data, n);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// Kernel launch: kernel<<<(((n + blockSize) - 1) / blockSize), "
            "blockSize, (sizeof(float) * blockSize), stream>>>()"
        ) in result
        assert "// Arguments: data, n" in result

    def test_hip_launch_kernel_ggl_conversion(self):
        """Test hipLaunchKernelGGL converts through kernel launch metadata"""
        code = """
        void host(float* data, int n) {
            dim3 grid(16);
            dim3 block(32);
            void* packedArgs[] = { &data, &n };
            hipLaunchKernelGGL(kernel, grid, block, 0, 0, packedArgs);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var packedArgs: array<ptr<void>> = {(&data), (&n)};" in result
        assert "// Kernel launch: kernel<<<grid, block, 0, 0>>>()" in result
        assert "// Arguments: data, n" in result
        assert "hipLaunchKernelGGL" not in result

    def test_hip_launch_kernel_api_conversion(self):
        """Test hipLaunchKernel converts through kernel launch metadata"""
        code = """
        void host(float* data, int n, int stream) {
            dim3 grid(16);
            dim3 block(32);
            void** packedArgs = (void*[]){ &data, &n };
            hipLaunchKernel((const void*)kernel, grid, block, packedArgs, 0, stream);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// Kernel launch: kernel<<<grid, block, 0, stream>>>()" in result
        assert "// Arguments: data, n" in result
        assert "hipLaunchKernel(" not in result

    def test_user_defined_hip_launch_kernel_call_is_not_kernel_launch(self):
        """Test user-defined hipLaunchKernel shadows launch lowering."""
        code = """
        void hipLaunchKernel(float* out, int grid, int block, void* args, int shared, int stream) {
            return;
        }

        void host(float* out, int grid, int block, void* args) {
            hipLaunchKernel(out, grid, block, args, 0, 0);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "void hipLaunchKernel(ptr<f32> out, i32 grid, i32 block, "
            "ptr<void> args, i32 shared, i32 stream)"
        ) in result
        assert "hipLaunchKernel(out, grid, block, args, 0, 0);" in result
        assert "// Kernel launch: out<<<grid, block, 0, 0>>>()" not in result

    def test_user_defined_hip_launch_kernel_ggl_call_is_not_kernel_launch(self):
        """Test user-defined hipLaunchKernelGGL shadows launch lowering."""
        code = """
        void hipLaunchKernelGGL(float* out, int grid, int block, int shared, int stream) {
            return;
        }

        void host(float* out, int grid, int block) {
            hipLaunchKernelGGL(out, grid, block, 0, 0);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "void hipLaunchKernelGGL(ptr<f32> out, i32 grid, i32 block, "
            "i32 shared, i32 stream)"
        ) in result
        assert "hipLaunchKernelGGL(out, grid, block, 0, 0);" in result
        assert "// Kernel launch: out<<<grid, block, 0, 0>>>()" not in result

    def test_user_defined_hip_launch_kernel_ggl_declared_later_is_not_kernel_launch(
        self,
    ):
        """Test later user-defined hipLaunchKernelGGL shadows launch lowering."""
        code = """
        void host(float* out, int grid, int block) {
            hipLaunchKernelGGL(out, grid, block, 0, 0);
        }

        void hipLaunchKernelGGL(float* out, int grid, int block, int shared, int stream) {
            return;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "void hipLaunchKernelGGL(ptr<f32> out, i32 grid, i32 block, "
            "i32 shared, i32 stream)"
        ) in result
        assert "hipLaunchKernelGGL(out, grid, block, 0, 0);" in result
        assert "// Kernel launch: out<<<grid, block, 0, 0>>>()" not in result

    def test_templated_hip_launch_kernel_ggl_conversion(self):
        """Test hipLaunchKernelGGL converts a template-id kernel argument"""
        code = """
        template <typename T>
        __global__ void scale(T* data, T factor) {
            data[threadIdx.x] *= factor;
        }

        void host(float* data) {
            hipLaunchKernelGGL(
                scale<float>, dim3(1), dim3(32), 0, 0, data, 2.0f
            );
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// Kernel launch: scale<float><<<vec3<u32>(1), vec3<u32>(32), 0, 0>>>()"
            in result
        )
        assert "// Arguments: data, 2.0f" in result
        assert "hipLaunchKernelGGL" not in result

    def test_hip_launch_kernel_casted_packed_args_conversion(self):
        """Test casted packed args still expand in hipLaunchKernelGGL"""
        code = """
        void host(float* data, int n, int stream) {
            dim3 grid(16);
            dim3 block(32);
            void** packedArgs = { &data, &n };
            hipLaunchKernelGGL(kernel, grid, block, 0, stream, (void**)packedArgs);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var packedArgs: ptr<ptr<void>> = {(&data), (&n)};" in result
        assert "// Kernel launch: kernel<<<grid, block, 0, stream>>>()" in result
        assert "// Arguments: data, n" in result
        assert "ptr<ptr<void>>(packedArgs)" not in result

    def test_hip_launch_kernel_compound_literal_args_conversion(self):
        """Test compound literal packed args expand in hipLaunchKernelGGL"""
        code = """
        void host(float* data, int n, int stream) {
            dim3 grid(16);
            dim3 block(32);
            hipLaunchKernelGGL(kernel, grid, block, 0, stream,
                               (void*[]){ &data, &n });
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// Kernel launch: kernel<<<grid, block, 0, stream>>>()" in result
        assert "// Arguments: data, n" in result
        assert "array<ptr<void>>({(&data), (&n)})" not in result

    def test_compound_literal_packed_args_declaration_conversion(self):
        """Test local compound literal packed args can be reused"""
        code = """
        void host(float* data, int n, int stream) {
            dim3 grid(16);
            dim3 block(32);
            void** packedArgs = (void*[]){ &data, &n };
            hipLaunchKernelGGL(kernel, grid, block, 0, stream, packedArgs);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "var packedArgs: ptr<ptr<void>> = array<ptr<void>>({(&data), (&n)});"
            in result
        )
        assert "// Arguments: data, n" in result

    def test_initializer_arrays_do_not_expand_as_launch_args(self):
        """Test only void pointer packed args expand in kernel launches"""
        code = """
        void host() {
            dim3 grid(16);
            dim3 block(32);
            float values[2] = {1.0f, 2.0f};
            kernel<<<grid, block>>>(values);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var values: array<f32, 2> = {1.0f, 2.0f};" in result
        assert "// Arguments: values" in result
        assert "// Arguments: 1.0f, 2.0f" not in result

    def test_hip_runtime_memory_api_conversion(self):
        """Test HIP runtime memory API calls emit metadata comments"""
        code = """
        void host(float* h, int n) {
            float* d;
            float* d2;
            size_t pitch;
            size_t freeMem;
            size_t totalMem;
            size_t ptrSize;
            float symbol;
            hipStream_t stream;
            unsigned int flags;
            int pointerAttrValue;
            hipPointerAttribute_t pointerAttrs;
            void* resourceDesc;
            void* textureDesc;
            void* viewDesc;
            void* copyParams;
            hipArray_t array;
            hipChannelFormatDesc desc;
            hipPitchedPtr pitched;
            hipExtent extent;
            hipMemPool_t pool;
            hipMemPoolProps poolProps;
            hipMemAccessDesc accessDesc;
            hipMemLocation location;
            hipTextureObject_t texObj;
            hipSurfaceObject_t surfObj;
            hipMalloc((void**)&d, n * sizeof(float));
            hipMallocAsync((void**)&d, n * sizeof(float), stream);
            hipMemPoolCreate(&pool, &poolProps);
            hipDeviceGetDefaultMemPool(&pool, 0);
            hipDeviceSetMemPool(0, pool);
            hipMallocFromPoolAsync((void**)&d2, n * sizeof(float), pool, stream);
            hipMemPoolTrimTo(pool, n * sizeof(float));
            hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &ptrSize);
            hipMemPoolGetAttribute(pool, hipMemPoolAttrReleaseThreshold, &ptrSize);
            hipMemPoolSetAccess(pool, &accessDesc, 1);
            hipMemPoolGetAccess(&flags, pool, &location);
            hipHostMalloc((void**)&h, n * sizeof(float), hipHostMallocMapped);
            hipHostAlloc((void**)&h, n * sizeof(float), hipHostMallocDefault);
            hipHostRegister(h, n * sizeof(float), hipHostRegisterMapped);
            hipHostGetDevicePointer((void**)&d, h, 0);
            hipHostGetFlags(&flags, h);
            hipMallocPitch((void**)&d2, &pitch, n * sizeof(float), 4);
            hipMallocArray(&array, &desc, n, 4, hipArrayDefault);
            hipMalloc3D(&pitched, extent);
            hipMalloc3DArray(&array, &desc, extent, hipArrayDefault);
            hipMemcpy(d, h, n * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy2D(
                d2, pitch, h, n * sizeof(float), n * sizeof(float), 4,
                hipMemcpyHostToDevice
            );
            hipMemcpy2DAsync(
                d2, pitch, h, n * sizeof(float), n * sizeof(float), 4,
                hipMemcpyHostToDevice, 0
            );
            hipMemcpyToSymbol(
                symbol, h, n * sizeof(float), 0, hipMemcpyHostToDevice
            );
            hipMemcpyFromSymbol(
                h, symbol, n * sizeof(float), 0, hipMemcpyDeviceToHost
            );
            hipMemcpy3D(&copyParams);
            hipMemcpy3DAsync(&copyParams, 0);
            hipMemGetInfo(&freeMem, &totalMem);
            hipPointerGetAttributes(&pointerAttrs, d);
            hipPointerGetAttribute(&pointerAttrValue, hipPointerAttributeMemoryType, d);
            hipPointerSetAttribute(&pointerAttrValue, hipPointerAttributeMemoryType, d);
            hipMemPtrGetInfo(d, &ptrSize);
            hipCreateTextureObject(
                &texObj, resourceDesc, textureDesc, viewDesc
            );
            hipCreateSurfaceObject(&surfObj, resourceDesc);
            hipDestroyTextureObject(texObj);
            hipDestroySurfaceObject(surfObj);
            hipMemset(d, 0, n * sizeof(float));
            hipMemset2D(d2, pitch, 0, n * sizeof(float), 4);
            hipMemset2DAsync(d2, pitch, 1, n * sizeof(float), 4, 0);
            hipMemset3D(pitched, 0, extent);
            hipMemset3DAsync(pitched, 1, extent, 0);
            hipDeviceSynchronize();
            hipHostUnregister(h);
            hipFree(d);
            hipFreeAsync(d2, stream);
            hipHostFree(h);
            hipFreeHost(h);
            hipFreeArray(array);
            hipArrayDestroy(array);
            hipMemPoolDestroy(pool);
            hipError_t err = hipMalloc((void**)&d, n * sizeof(float));
            err = hipMallocAsync((void**)&d, n * sizeof(float), stream);
            err = hipMemPoolCreate(&pool, &poolProps);
            err = hipFreeAsync(d2, stream);
            err = hipHostRegister(h, n * sizeof(float), hipHostRegisterMapped);
            err = hipMallocPitch((void**)&d2, &pitch, n * sizeof(float), 4);
            err = hipMallocArray(&array, &desc, n, 4, hipArrayDefault);
            err = hipMemcpy3D(&copyParams);
            err = hipMemset3D(pitched, 0, extent);
            err = hipPointerGetAttributes(&pointerAttrs, d);
            err = hipCreateTextureObject(
                &texObj, resourceDesc, textureDesc, viewDesc
            );
            err = hipCreateSurfaceObject(&surfObj, resourceDesc);
            err = hipDestroyTextureObject(texObj);
            err = hipDestroySurfaceObject(surfObj);
            if (err != hipSuccess) { return; }
            err = hipDeviceSynchronize();
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            result.count("// HIP memory allocate: d, bytes: (n * sizeof(float))") == 2
        )
        assert (
            result.count(
                "// HIP async memory allocate: d, bytes: (n * sizeof(float)), "
                "stream: stream"
            )
            == 2
        )
        assert (
            result.count(
                "// HIP memory pool create: output: pool, properties: (&poolProps)"
            )
            == 2
        )
        assert "// HIP get default memory pool: output: pool, device: 0" in result
        assert "// HIP set device memory pool: device: 0, pool: pool" in result
        assert (
            "// HIP async memory allocate from pool: d2, bytes: (n * sizeof(float)), "
            "pool: pool, stream: stream"
        ) in result
        assert (
            "// HIP memory pool trim: pool: pool, minimum bytes: (n * sizeof(float))"
            in result
        )
        assert (
            "// HIP memory pool set attribute: pool: pool, "
            "attribute: hipMemPoolAttrReleaseThreshold, value: ptrSize"
        ) in result
        assert (
            "// HIP memory pool get attribute: pool: pool, "
            "attribute: hipMemPoolAttrReleaseThreshold, output: ptrSize"
        ) in result
        assert (
            "// HIP memory pool set access: pool: pool, descriptors: (&accessDesc), "
            "count: 1"
        ) in result
        assert (
            "// HIP memory pool get access: output: flags, pool: pool, "
            "location: (&location)"
        ) in result
        assert (
            "// HIP host memory allocate: h, bytes: (n * sizeof(float)), "
            "flags: hipHostMallocMapped"
        ) in result
        assert (
            "// HIP host memory allocate: h, bytes: (n * sizeof(float)), "
            "flags: hipHostMallocDefault"
        ) in result
        assert (
            result.count(
                "// HIP host memory register: h, bytes: (n * sizeof(float)), "
                "flags: hipHostRegisterMapped"
            )
            == 2
        )
        assert "// HIP host device pointer: output: d, host: h, flags: 0" in result
        assert "// HIP host memory flags: output: flags, host: h" in result
        assert (
            result.count(
                "// HIP pitched memory allocate: d2, pitch: pitch, "
                "width: (n * sizeof(float)), height: 4"
            )
            == 2
        )
        assert (
            result.count(
                "// HIP array allocate: array, desc: desc, width: n, "
                "height: 4, flags: hipArrayDefault"
            )
            == 2
        )
        assert "// HIP 3D memory allocate: pitched, extent: extent" in result
        assert (
            "// HIP 3D array allocate: array, desc: desc, extent: extent, "
            "flags: hipArrayDefault"
        ) in result
        assert (
            "// HIP memory copy: h -> d, bytes: (n * sizeof(float)), "
            "kind: hipMemcpyHostToDevice"
        ) in result
        assert (
            "// HIP 2D memory copy: h -> d2, dst pitch: pitch, "
            "src pitch: (n * sizeof(float)), width: (n * sizeof(float)), "
            "height: 4, kind: hipMemcpyHostToDevice"
        ) in result
        assert (
            "// HIP 2D memory copy: h -> d2, dst pitch: pitch, "
            "src pitch: (n * sizeof(float)), width: (n * sizeof(float)), "
            "height: 4, kind: hipMemcpyHostToDevice, stream: 0"
        ) in result
        assert (
            "// HIP symbol copy to: symbol, source: h, bytes: (n * sizeof(float)), "
            "offset: 0, kind: hipMemcpyHostToDevice"
        ) in result
        assert (
            "// HIP symbol copy from: symbol, destination: h, "
            "bytes: (n * sizeof(float)), offset: 0, kind: hipMemcpyDeviceToHost"
        ) in result
        assert result.count("// HIP 3D memory copy: params: copyParams") == 3
        assert "// HIP 3D memory copy: params: copyParams, stream: 0" in result
        assert (
            "// HIP memory info: free output: freeMem, total output: totalMem" in result
        )
        assert (
            result.count("// HIP pointer attributes: output: pointerAttrs, pointer: d")
            == 2
        )
        assert (
            "// HIP pointer attribute: output: pointerAttrValue, "
            "attribute: hipPointerAttributeMemoryType, pointer: d"
        ) in result
        assert (
            "// HIP pointer set attribute: value: pointerAttrValue, "
            "attribute: hipPointerAttributeMemoryType, pointer: d"
        ) in result
        assert "// HIP memory pointer info: pointer: d, size output: ptrSize" in result
        assert (
            result.count(
                "// HIP texture object create: texObj, resource: resourceDesc, "
                "texture desc: textureDesc, resource view: viewDesc"
            )
            == 2
        )
        assert (
            result.count(
                "// HIP surface object create: surfObj, resource: resourceDesc"
            )
            == 2
        )
        assert result.count("// HIP texture object destroy: texObj") == 2
        assert result.count("// HIP surface object destroy: surfObj") == 2
        assert "// HIP memory set: d, value: 0, bytes: (n * sizeof(float))" in result
        assert (
            "// HIP 2D memory set: d2, pitch: pitch, value: 0, "
            "width: (n * sizeof(float)), height: 4"
        ) in result
        assert (
            "// HIP 2D memory set: d2, pitch: pitch, value: 1, "
            "width: (n * sizeof(float)), height: 4, stream: 0"
        ) in result
        assert (
            result.count("// HIP 3D memory set: pitched, value: 0, extent: extent") == 2
        )
        assert (
            "// HIP 3D memory set: pitched, value: 1, extent: extent, stream: 0"
            in result
        )
        assert "// HIP device synchronize" in result
        assert result.count("// HIP memory free: h") == 2
        assert "// HIP host memory unregister: h" in result
        assert "// HIP memory free: d" in result
        assert result.count("// HIP async memory free: d2, stream: stream") == 2
        assert result.count("// HIP array free: array") == 2
        assert "// HIP memory pool destroy: pool" in result
        assert "workgroupBarrier();" not in result
        assert "var err: hipError_t = hipSuccess;" in result
        assert "if ((err != hipSuccess))" in result
        assert "err = hipSuccess;" in result
        assert "hipMalloc(ptr<ptr<void>>((&d)), (n * sizeof(float)))" not in result
        assert "hipMallocAsync(" not in result
        assert "hipMallocFromPoolAsync(" not in result
        assert "hipFreeAsync(" not in result
        assert "hipMemPoolCreate(" not in result
        assert "hipMemPoolDestroy(" not in result
        assert "hipMemPoolTrimTo(" not in result
        assert "hipMemPoolSetAttribute(" not in result
        assert "hipMemPoolGetAttribute(" not in result
        assert "hipMemPoolSetAccess(" not in result
        assert "hipMemPoolGetAccess(" not in result
        assert "hipDeviceGetDefaultMemPool(" not in result
        assert "hipDeviceSetMemPool(" not in result
        assert "hipHostMalloc(" not in result
        assert "hipHostAlloc(" not in result
        assert "hipHostRegister(" not in result
        assert "hipHostGetDevicePointer(" not in result
        assert "hipHostGetFlags(" not in result
        assert "hipMallocPitch(" not in result
        assert "hipMallocArray(" not in result
        assert "hipMalloc3D(" not in result
        assert "hipMalloc3DArray(" not in result
        assert "hipMemcpy2D(" not in result
        assert "hipMemcpy2DAsync(" not in result
        assert "hipMemcpy3D(" not in result
        assert "hipMemcpy3DAsync(" not in result
        assert "hipMemcpyToSymbol(" not in result
        assert "hipMemcpyFromSymbol(" not in result
        assert "hipMemGetInfo(" not in result
        assert "hipPointerGetAttributes(" not in result
        assert "hipPointerGetAttribute(" not in result
        assert "hipPointerSetAttribute(" not in result
        assert "hipMemPtrGetInfo(" not in result
        assert "hipMemset2D(" not in result
        assert "hipMemset2DAsync(" not in result
        assert "hipMemset3D(" not in result
        assert "hipMemset3DAsync(" not in result
        assert "hipCreateTextureObject(" not in result
        assert "hipCreateSurfaceObject(" not in result
        assert "hipDestroyTextureObject(" not in result
        assert "hipDestroySurfaceObject(" not in result
        assert "hipHostUnregister(" not in result
        assert "hipHostFree(" not in result
        assert "hipFreeHost(" not in result
        assert "hipFreeArray(" not in result
        assert "hipArrayDestroy(" not in result
        assert "err = hipDeviceSynchronize();" not in result

    def test_hip_runtime_memset_async_conversion(self):
        """Test hipMemsetAsync emits metadata comments and status success"""
        code = """
        void host(float* d, int n, hipStream_t stream) {
            hipMemsetAsync(d, 0, n * sizeof(float), stream);
            hipError_t err = hipMemsetAsync(d, 1, n * sizeof(float), stream);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// HIP memory set: d, value: 0, bytes: (n * sizeof(float)), "
            "stream: stream"
        ) in result
        assert (
            "// HIP memory set: d, value: 1, bytes: (n * sizeof(float)), "
            "stream: stream"
        ) in result
        assert "var err: hipError_t = hipSuccess;" in result
        assert "hipMemsetAsync(" not in result

    def test_hip_runtime_last_error_query_conversion(self):
        """Test HIP runtime error query calls emit metadata comments"""
        code = """
        void host() {
            hipGetLastError();
            hipError_t err = hipGetLastError();
            err = hipPeekAtLastError();
            const char* message = hipGetErrorString(err);
            const char* name = hipGetErrorName(err);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert result.count("// HIP get last error") == 2
        assert "// HIP peek at last error" in result
        assert "var err: hipError_t = hipSuccess;" in result
        assert "err = hipSuccess;" in result
        assert 'var message: ptr<i8> = /* HIP error string: err */ "";' in result
        assert 'var name: ptr<i8> = /* HIP error name: err */ "";' in result
        assert "hipGetLastError(" not in result
        assert "hipPeekAtLastError(" not in result
        assert "hipGetErrorString(" not in result
        assert "hipGetErrorName(" not in result

    def test_user_defined_hip_runtime_call_does_not_emit_runtime_comment(self):
        """Test user-defined HIP runtime names shadow runtime call comments."""
        code = """
        void hipMemcpy(float* dst, float* src, int bytes, int kind) {
            return;
        }

        void host(float* d, float* h, int n) {
            hipMemcpy(d, h, n, 7);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "void hipMemcpy(ptr<f32> dst, ptr<f32> src, i32 bytes, i32 kind)" in result
        )
        assert "hipMemcpy(d, h, n, 7);" in result
        assert "// HIP memory copy: h -> d, bytes: n, kind: 7" not in result

    def test_hip_runtime_event_api_conversion(self):
        """Test HIP stream and event API calls emit metadata comments"""
        code = """
        void bench() {
            hipStream_t stream;
            hipEvent_t start;
            hipEvent_t stop;
            float ms;
            int device = 0;
            int count = 0;
            int gridSize = 0;
            int blockSize = 0;
            int activeBlocks = 0;
            int attr = 0;
            int major = 0;
            int minor = 0;
            int leastPriority = 0;
            int greatestPriority = 0;
            int priority = 0;
            unsigned int flags = 0;
            size_t total = 0;
            size_t numDeps = 0;
            unsigned long long captureId = 0;
            char name[64];
            hipStreamCaptureStatus captureStatus;
            hipGraph_t graph;
            hipGraphNode_t* deps;
            hipDeviceProp_t props;
            void* kernel;
            void* callback;
            void* hostFn;
            void* userData;
            hipGetDevice(&device);
            hipGetDeviceCount(&count);
            hipSetDevice(device);
            hipGetDeviceProperties(&props, device);
            hipDeviceGetAttribute(
                &attr, hipDeviceAttributeMaxThreadsPerBlock, device
            );
            hipDeviceGetName(name, 64, device);
            hipDeviceTotalMem(&total, device);
            hipDeviceComputeCapability(&major, &minor, device);
            hipChooseDevice(&device, &props);
            hipGetDeviceFlags(&flags);
            hipSetDeviceFlags(hipDeviceScheduleAuto);
            hipOccupancyMaxPotentialBlockSize(
                &gridSize, &blockSize, kernel, 0, 0
            );
            hipOccupancyMaxPotentialBlockSizeVariableSMem(
                &gridSize, &blockSize, kernel, 0, 256
            );
            hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(
                &gridSize, &blockSize, kernel, 0, 256, 1
            );
            hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &activeBlocks, kernel, blockSize, 0
            );
            hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                &activeBlocks, kernel, blockSize, 0, 1
            );
            hipStreamCreate(&stream);
            hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
            hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
            hipStreamGetFlags(stream, &flags);
            hipStreamGetPriority(stream, &priority);
            hipStreamAddCallback(stream, callback, userData, 0);
            hipLaunchHostFunc(stream, hostFn, userData);
            hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
            hipStreamIsCapturing(stream, &captureStatus);
            hipStreamGetCaptureInfo(stream, &captureStatus, &captureId);
            hipStreamGetCaptureInfo_v2(
                stream, &captureStatus, &captureId, &graph, &deps, &numDeps
            );
            hipStreamUpdateCaptureDependencies(stream, deps, numDeps, 0);
            hipStreamEndCapture(stream, &graph);
            hipEventCreate(&start);
            hipEventCreateWithFlags(&stop, hipEventDisableTiming);
            hipEventRecord(start, stream);
            hipStreamWaitEvent(stream, start, 0);
            hipEventSynchronize(stop);
            hipEventElapsedTime(&ms, start, stop);
            hipEventQuery(stop);
            hipEventDestroy(start);
            hipStreamDestroy(stream);
            hipError_t err = hipEventRecord(stop, stream);
            err = hipGetDeviceCount(&count);
            err = hipLaunchHostFunc(stream, hostFn, userData);
            err = hipDeviceGetAttribute(
                &attr, hipDeviceAttributeMaxThreadsPerBlock, device
            );
            err = hipOccupancyMaxPotentialBlockSize(
                &gridSize, &blockSize, kernel, 0, 0
            );
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// HIP stream create: stream" in result
        assert "// HIP stream create: stream, flags: hipStreamNonBlocking" in result
        assert (
            "// HIP get stream priority range: least output: leastPriority, "
            "greatest output: greatestPriority"
        ) in result
        assert "// HIP get stream flags: stream: stream, output: flags" in result
        assert "// HIP get stream priority: stream: stream, output: priority" in result
        assert (
            "// HIP stream add callback: stream: stream, callback: callback, "
            "user data: userData, flags: 0"
        ) in result
        assert (
            result.count(
                "// HIP launch host function: stream: stream, function: hostFn, "
                "user data: userData"
            )
            == 2
        )
        assert (
            "// HIP stream begin capture: stream: stream, mode: "
            "hipStreamCaptureModeGlobal"
        ) in result
        assert (
            "// HIP stream is capturing: stream: stream, output: captureStatus"
            in result
        )
        assert (
            "// HIP stream capture info: stream: stream, "
            "status output: captureStatus, id output: captureId"
        ) in result
        assert (
            "// HIP stream capture info: stream: stream, "
            "status output: captureStatus, id output: captureId, "
            "graph output: graph, dependencies output: deps, "
            "dependency count output: numDeps"
        ) in result
        assert (
            "// HIP stream update capture dependencies: stream: stream, "
            "dependencies: deps, count: numDeps, flags: 0"
        ) in result
        assert (
            "// HIP stream end capture: stream: stream, graph output: graph" in result
        )
        assert "// HIP event create: start" in result
        assert "// HIP event create: stop, flags: hipEventDisableTiming" in result
        assert "// HIP event record: start, stream: stream" in result
        assert "// HIP stream wait event: stream waits for start, flags: 0" in result
        assert "// HIP event synchronize: stop" in result
        assert "// HIP event elapsed time: start -> stop, output: ms" in result
        assert "// HIP event query: stop" in result
        assert "// HIP event destroy: start" in result
        assert "// HIP stream destroy: stream" in result
        assert "// HIP event record: stop, stream: stream" in result
        assert "// HIP get current device: output: device" in result
        assert result.count("// HIP get device count: output: count") == 2
        assert "// HIP set device: device" in result
        assert "// HIP get device properties: props, device: device" in result
        assert (
            result.count(
                "// HIP get device attribute: output: attr, "
                "attribute: hipDeviceAttributeMaxThreadsPerBlock, device: device"
            )
            == 2
        )
        assert (
            "// HIP get device name: output: name, length: 64, device: device" in result
        )
        assert "// HIP get device total memory: output: total, device: device" in result
        assert (
            "// HIP get device compute capability: major output: major, "
            "minor output: minor, device: device"
        ) in result
        assert "// HIP choose device: output: device, properties: (&props)" in result
        assert "// HIP get device flags: output: flags" in result
        assert "// HIP set device flags: hipDeviceScheduleAuto" in result
        assert (
            result.count(
                "// HIP occupancy max potential block size: grid output: gridSize, "
                "block output: blockSize, kernel: kernel, "
                "dynamic shared memory: 0, block size limit: 0"
            )
            == 2
        )
        assert (
            "// HIP occupancy max potential block size: grid output: gridSize, "
            "block output: blockSize, kernel: kernel, dynamic shared memory: 0, "
            "block size limit: 256"
        ) in result
        assert (
            "// HIP occupancy max potential block size: grid output: gridSize, "
            "block output: blockSize, kernel: kernel, dynamic shared memory: 0, "
            "block size limit: 256, flags: 1"
        ) in result
        assert (
            "// HIP occupancy active blocks per multiprocessor: "
            "output: activeBlocks, kernel: kernel, block size: blockSize, "
            "dynamic shared memory: 0"
        ) in result
        assert (
            "// HIP occupancy active blocks per multiprocessor: "
            "output: activeBlocks, kernel: kernel, block size: blockSize, "
            "dynamic shared memory: 0, flags: 1"
        ) in result
        assert "var err: hipError_t = hipSuccess;" in result
        assert "err = hipSuccess;" in result
        assert "var err: hipError_t = hipEventRecord(stop, stream);" not in result
        assert "hipStreamCreateWithFlags(" not in result
        assert "hipDeviceGetStreamPriorityRange(" not in result
        assert "hipStreamGetFlags(" not in result
        assert "hipStreamGetPriority(" not in result
        assert "hipStreamAddCallback(" not in result
        assert "hipLaunchHostFunc(" not in result
        assert "hipStreamBeginCapture(" not in result
        assert "hipStreamEndCapture(" not in result
        assert "hipStreamIsCapturing(" not in result
        assert "hipStreamGetCaptureInfo(" not in result
        assert "hipStreamGetCaptureInfo_v2(" not in result
        assert "hipStreamUpdateCaptureDependencies(" not in result
        assert "hipGetDevice(" not in result
        assert "hipGetDeviceCount(" not in result
        assert "hipSetDevice(" not in result
        assert "hipGetDeviceProperties(" not in result
        assert "hipDeviceGetAttribute(" not in result
        assert "hipDeviceGetName(" not in result
        assert "hipDeviceTotalMem(" not in result
        assert "hipDeviceComputeCapability(" not in result
        assert "hipChooseDevice(" not in result
        assert "hipGetDeviceFlags(" not in result
        assert "hipSetDeviceFlags(" not in result
        assert "hipOccupancyMaxPotentialBlockSize(" not in result
        assert "hipOccupancyMaxPotentialBlockSizeVariableSMem(" not in result
        assert "hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(" not in result
        assert "hipOccupancyMaxActiveBlocksPerMultiprocessor(" not in result
        assert "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(" not in result

    def test_hip_runtime_stream_create_with_flags_status_conversion(self):
        """Test hipStreamCreateWithFlags status expressions lower to success"""
        code = """
        void bench() {
            hipStream_t stream;
            hipError_t err = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// HIP stream create: stream, flags: hipStreamNonBlocking" in result
        assert "var err: hipError_t = hipSuccess;" in result
        assert "hipStreamCreateWithFlags(" not in result

    def test_hip_runtime_module_api_conversion(self):
        """Test HIP module and function APIs emit metadata comments."""
        code = """
        void load_module(hipStream_t stream) {
            hipModule_t module;
            hipFunction_t function;
            hipFunction_t symbolFunction;
            hipFuncAttributes attrs;
            int attrValue;
            void* globalPtr;
            size_t globalBytes;
            void** params;
            void** extra;
            void* launchParams;
            void* image;
            hipJitOption options;
            void* optionValues;
            hipModuleLoad(&module, "kernel.hsaco");
            hipModuleLoadData(&module, image);
            hipModuleLoadDataEx(&module, image, 1, &options, &optionValues);
            hipModuleGetFunction(&function, module, "kernel");
            hipGetFuncBySymbol(&symbolFunction, function);
            hipFuncGetAttribute(
                &attrValue, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function
            );
            hipFuncGetAttributes(&attrs, function);
            hipFuncSetAttribute(
                function, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 1024
            );
            hipFuncSetCacheConfig(function, hipFuncCachePreferL1);
            hipFuncSetSharedMemConfig(function, hipSharedMemBankSizeEightByte);
            hipModuleGetGlobal(&globalPtr, &globalBytes, module, "symbol");
            hipModuleLaunchKernel(
                function, 8, 1, 1, 64, 1, 1, 0, stream, params, extra
            );
            hipLaunchCooperativeKernel(function, 8, 64, params, 0, stream);
            hipLaunchCooperativeKernelMultiDevice(launchParams, 1, 0);
            hipModuleLaunchCooperativeKernel(
                function, 8, 1, 1, 64, 1, 1, 0, stream, params
            );
            hipModuleLaunchCooperativeKernelMultiDevice(launchParams, 1, 0);
            hipModuleUnload(module);
            hipError_t err = hipModuleGetFunction(&function, module, "kernel");
            err = hipModuleLaunchKernel(
                function, 8, 1, 1, 64, 1, 1, 0, stream, params, extra
            );
            err = hipLaunchCooperativeKernel(function, 8, 64, params, 0, stream);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert '// HIP module load: output: module, file: "kernel.hsaco"' in result
        assert "// HIP module load: output: module, image: image" in result
        assert (
            "// HIP module load data ex: output: module, image: image, options: 1, "
            "option keys: (&options), option values: (&optionValues)"
        ) in result
        assert (
            result.count(
                "// HIP module get function: output: function, module: module, "
                'name: "kernel"'
            )
            == 2
        )
        assert (
            "// HIP get function by symbol: output: symbolFunction, symbol: function"
            in result
        )
        assert (
            "// HIP function get attribute: output: attrValue, "
            "attribute: HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function: function"
        ) in result
        assert (
            "// HIP function get attributes: output: attrs, function: function"
            in result
        )
        assert (
            "// HIP function set attribute: function: function, "
            "attribute: HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, value: 1024"
        ) in result
        assert (
            "// HIP function set cache config: function: function, "
            "config: hipFuncCachePreferL1"
        ) in result
        assert (
            "// HIP function set shared memory config: function: function, "
            "config: hipSharedMemBankSizeEightByte"
        ) in result
        assert (
            "// HIP module get global: pointer output: globalPtr, "
            'size output: globalBytes, module: module, name: "symbol"'
        ) in result
        assert (
            result.count(
                "// HIP module launch kernel: function: function, "
                "grid: (8, 1, 1), block: (64, 1, 1), shared memory: 0, "
                "stream: stream, params: params, extra: extra"
            )
            == 2
        )
        assert (
            result.count(
                "// HIP cooperative kernel launch: function: function, grid: 8, "
                "block: 64, params: params, shared memory: 0, stream: stream"
            )
            == 2
        )
        assert (
            "// HIP cooperative multi-device launch: params: launchParams, "
            "devices: 1, flags: 0"
        ) in result
        assert (
            "// HIP module cooperative kernel launch: function: function, "
            "grid: (8, 1, 1), block: (64, 1, 1), shared memory: 0, "
            "stream: stream, params: params"
        ) in result
        assert (
            "// HIP module cooperative multi-device launch: params: launchParams, "
            "devices: 1, flags: 0"
        ) in result
        assert "// HIP module unload: module" in result
        assert "var err: hipError_t = hipSuccess;" in result
        assert "err = hipSuccess;" in result
        assert "hipModuleLoad(" not in result
        assert "hipModuleLoadData(" not in result
        assert "hipModuleLoadDataEx(" not in result
        assert "hipModuleGetFunction(" not in result
        assert "hipGetFuncBySymbol(" not in result
        assert "hipFuncGetAttribute(" not in result
        assert "hipFuncGetAttributes(" not in result
        assert "hipFuncSetAttribute(" not in result
        assert "hipFuncSetCacheConfig(" not in result
        assert "hipFuncSetSharedMemConfig(" not in result
        assert "hipModuleGetGlobal(" not in result
        assert "hipModuleLaunchKernel(" not in result
        assert "hipLaunchCooperativeKernel(" not in result
        assert "hipLaunchCooperativeKernelMultiDevice(" not in result
        assert "hipModuleLaunchCooperativeKernel(" not in result
        assert "hipModuleLaunchCooperativeKernelMultiDevice(" not in result
        assert "hipModuleUnload(" not in result

    def test_hip_runtime_stream_create_with_priority_status_conversion(self):
        """Test hipStreamCreateWithPriority emits priority metadata"""
        code = """
        void bench() {
            hipStream_t stream;
            hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, 3);
            hipError_t err = hipStreamCreateWithPriority(
                &stream, hipStreamNonBlocking, 3
            );
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            result.count(
                "// HIP stream create: stream, flags: hipStreamNonBlocking, "
                "priority: 3"
            )
            == 2
        )
        assert "var err: hipError_t = hipSuccess;" in result
        assert "hipStreamCreateWithPriority(" not in result

    def test_std_chrono_benchmark_expression_conversion(self):
        """Test std::chrono benchmark expressions convert"""
        code = """
        void bench() {
            auto start = std::chrono::high_resolution_clock::now();
            kernel<<<1, 32>>>();
            auto stop = std::chrono::high_resolution_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                stop - start
            ).count();
            bool ordered = 1 < 2;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var start: auto = std::chrono::high_resolution_clock::now();" in result
        assert "// Kernel launch: kernel<<<1, 32>>>()" in result
        assert "var stop: auto = std::chrono::high_resolution_clock::now();" in result
        assert (
            "var us: auto = std::chrono::duration_cast<std::chrono::microseconds>"
            "((stop - start)).count();"
        ) in result
        assert "var ordered: bool = (1 < 2);" in result

    def test_std_vector_host_buffer_conversion(self):
        """Test std::vector host buffers convert in memory copy metadata"""
        code = """
        void host(float* d, int n) {
            std::vector<float> h(n);
            hipMemcpy(d, h.data(), h.size() * sizeof(float),
                      hipMemcpyHostToDevice);
            bool ordered = h.size() < n;
            std::chrono::high_resolution_clock::now();
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var h: std::vector<float> = std::vector<float>(n);" in result
        assert (
            "// HIP memory copy: h.data() -> d, bytes: "
            "(h.size() * sizeof(float)), kind: hipMemcpyHostToDevice"
        ) in result
        assert "var ordered: bool = (h.size() < n);" in result
        assert "std::chrono::high_resolution_clock::now();" in result

    def test_std_array_host_buffer_conversion(self):
        """Test std::array host buffers convert in memory copy metadata"""
        code = """
        void host(float* d) {
            std::array<float, 4> h{1.0f, 2.0f, 3.0f, 4.0f};
            std::array<float, 4> zeros{};
            hipMemcpy(d, h.data(), h.size() * sizeof(float),
                      hipMemcpyHostToDevice);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var h: std::array<float, 4> = {1.0f, 2.0f, 3.0f, 4.0f};" in result
        assert "var zeros: std::array<float, 4> = {};" in result
        assert (
            "// HIP memory copy: h.data() -> d, bytes: "
            "(h.size() * sizeof(float)), kind: hipMemcpyHostToDevice"
        ) in result

    def test_host_index_fill_scalar_constructor_conversion(self):
        """Test host-side scalar type constructors convert to CrossGL types"""
        code = """
        void host(int n) {
            std::vector<float> h(n);
            for (int i = 0; i < n; ++i) {
                h[i] = float(i);
            }
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "for (var i: i32 = 0; (i < n); (++i)) {" in result
        assert "h[i] = f32(i);" in result

    def test_reference_host_helper_parameters_conversion(self):
        """Test host helper reference parameters lower to value types"""
        code = """
        void prepare(std::vector<float>& h) {
            std::fill(h.begin(), h.end(), 1.0f);
        }
        size_t count(const std::vector<float>& h) {
            return h.size();
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "void prepare(std::vector<float> h) {" in result
        assert "std::fill(h.begin(), h.end(), 1.0f);" in result
        assert "u32 count(std::vector<float> h) {" in result
        assert "return h.size();" in result

    def test_multi_declarator_host_setup_conversion(self):
        """Test comma-separated host setup declarations convert separately"""
        code = """
        void host(int n) {
            std::vector<float> a(n), b(n), c(n);
            float *d_a, *d_b;
            int x = 1, y = 2, z;
            float weights[2] = {1.0f, 2.0f}, bias = 3.0f;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var a: std::vector<float> = std::vector<float>(n);" in result
        assert "var b: std::vector<float> = std::vector<float>(n);" in result
        assert "var c: std::vector<float> = std::vector<float>(n);" in result
        assert "var d_a: ptr<f32>;" in result
        assert "var d_b: ptr<f32>;" in result
        assert "var x: i32 = 1;" in result
        assert "var y: i32 = 2;" in result
        assert "var z: i32;" in result
        assert "var weights: array<f32, 2> = {1.0f, 2.0f};" in result
        assert "var bias: f32 = 3.0f;" in result

    def test_multi_declarator_for_initializer_conversion(self):
        """Test comma-separated for initializers lower into scoped declarations"""
        code = """
        void host(float* a, float* b, int n) {
            for (int i = 0, j = n; i < j; ++i) {
                j--;
            }
            for (float *pa = a, *pb = b; pa != pb; ++pa) {
            }
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var i: i32 = 0;" in result
        assert "var j: i32 = n;" in result
        assert "for (; (i < j); (++i)) {" in result
        assert "(j--);" in result
        assert "var pa: ptr<f32> = a;" in result
        assert "var pb: ptr<f32> = b;" in result
        assert "for (; (pa != pb); (++pa)) {" in result
        assert "for (var i: i32 = 0, var j:" not in result

    def test_multi_expression_for_update_conversion(self):
        """Test comma-separated for updates lower into one header"""
        code = """
        void host(int n) {
            for (int i = 0, j = n; i < j; ++i, --j) {
                sink(i);
            }
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "for (; (i < j); (++i), (--j)) {" in result
        assert "sink(i);" in result

    def test_auto_pointer_reference_local_declarations_conversion(self):
        """Test auto pointer and reference declarations convert"""
        code = """
        void host(std::vector<float>& h, float* data) {
            int value = 2;
            int scale = 3;
            value * scale;
            auto& x = h[0];
            auto* p = data;
            const auto* cp = data;
            auto *q = data, *r = data;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "(value * scale);" in result
        assert "var x: auto = h[0];" in result
        assert "var p: ptr<auto> = data;" in result
        assert "var cp: ptr<auto> = data;" in result
        assert "var q: ptr<auto> = data;" in result
        assert "var r: ptr<auto> = data;" in result

    def test_device_lambda_expression_conversion(self):
        """Test HIP device lambdas convert to CrossGL pseudo-lambda calls."""
        code = """
        void host() {
            auto folded = fold(values, 0,
                [&] __device__ (int acc, int x) { return (acc + x); });
            auto mapped = map(colors,
                [] __device__ (float3 color) -> float3 {
                    prepare(color);
                    return color;
                });
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "var folded: auto = fold(values, 0, "
            "lambda(i32 acc, i32 x, (acc + x)));" in result
        )
        assert (
            "var mapped: auto = map(colors, "
            "lambda(vec3<f32> color, { prepare(color); return color; }));" in result
        )

    def test_restrict_pointer_qualifier_conversion(self):
        """Test __restrict__ pointer qualifiers are stripped during conversion"""
        code = """
        __global__ void kernel(const float* __restrict__ input,
                               float __restrict__* output) {
            output[threadIdx.x] = input[threadIdx.x];
        }
        void host(float* data) {
            float* __restrict__ p = data;
            const float* __restrict__ cp = data;
            float *__restrict__ a = data, *__restrict__ b = data;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "input: array<f32>" in result
        assert "output: array<f32>" in result
        assert "var p: ptr<f32> = data;" in result
        assert "var cp: ptr<f32> = data;" in result
        assert "var a: ptr<f32> = data;" in result
        assert "var b: ptr<f32> = data;" in result
        assert "__restrict__" not in result

    def test_rvalue_reference_declarations_conversion(self):
        """Test rvalue references are stripped during conversion"""
        code = """
        void consume(float&& value, const float&& other) {
            sink(value);
        }
        void host(std::vector<float>& h, float value, float other) {
            auto&& ref = value;
            const auto&& cref = other;
            float&& local = value;
            bool ok = true && false;
            for (auto&& x : h) {
                sink(x);
            }
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "void consume(f32 value, f32 other)" in result
        assert "var ref: auto = value;" in result
        assert "var cref: auto = other;" in result
        assert "var local: f32 = value;" in result
        assert "var ok: bool = (true && false);" in result
        assert "for x in h {" in result
        assert "&&" not in result.replace("(true && false)", "")

    def test_cpp_named_casts_conversion(self):
        """Test C++ named casts lower through CastNode conversion"""
        code = """
        void host(const float* input, float* data, int i, int n) {
            float x = static_cast<float>(i);
            float* p = const_cast<float*>(input);
            hipMalloc(reinterpret_cast<void**>(&data), n * sizeof(float));
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var x: f32 = f32(i);" in result
        assert "var p: ptr<f32> = ptr<f32>(input);" in result
        assert "// HIP memory allocate: data, bytes: (n * sizeof(float))" in result
        assert "static_cast" not in result
        assert "const_cast" not in result
        assert "reinterpret_cast" not in result

    def test_new_delete_host_allocation_conversion(self):
        """Test C++ new/delete host allocation conversion"""
        code = """
        void host(int n) {
            float* h = new float[n];
            int* q = new int;
            float* p = new float(1.0f);
            h[0] = 1.0f;
            delete[] h;
            delete q;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var h: ptr<f32> = new_array<f32>(n);" in result
        assert "var q: ptr<i32> = new<i32>();" in result
        assert "var p: ptr<f32> = new<f32>(1.0f);" in result
        assert "h[0] = 1.0f;" in result
        assert "// delete array: h" in result
        assert "// delete: q" in result
        assert "array<delete>" not in result

    def test_unique_ptr_host_allocation_conversion(self):
        """Test common std::unique_ptr host allocation conversion"""
        code = """
        void host(int n) {
            std::unique_ptr<float[]> h = std::make_unique<float[]>(n);
            std::unique_ptr<int> q = std::make_unique<int>();
            std::unique_ptr<float[]> owned(new float[n]);
            std::unique_ptr<const float[]> constants =
                std::make_unique<const float[]>(n);
            std::unique_ptr<unsigned int[]> ids =
                std::make_unique<unsigned int[]>(n);
            float* raw = h.get();
            consume(h.get(), q.get());
            consume(constants.get(), ids.get());
            Holder other;
            int value = other.get();
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var h: ptr<f32> = new_array<f32>(n);" in result
        assert "var q: ptr<i32> = new<i32>();" in result
        assert "var owned: ptr<f32> = new_array<f32>(n);" in result
        assert "var constants: ptr<f32> = new_array<f32>(n);" in result
        assert "var ids: ptr<u32> = new_array<u32>(n);" in result
        assert "var raw: ptr<f32> = h;" in result
        assert "consume(h, q);" in result
        assert "consume(constants, ids);" in result
        assert "var value: i32 = other.get();" in result
        assert "std::unique_ptr" not in result
        assert "std::make_unique" not in result

    def test_non_std_unique_ptr_helpers_do_not_lower_to_host_allocation(self):
        """Test non-std unique_ptr helpers remain ordinary qualified calls."""
        code = """
        void host(int n) {
            auto p = my::make_unique<float[]>(n);
            my::unique_ptr<float[]> q =
                my::unique_ptr<float[]>(new float[n]);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var p: auto = my::make_unique<float[]>(n);" in result
        assert (
            "var q: my::unique_ptr<float[]> = "
            "my::unique_ptr<float[]>(new_array<f32>(n));"
        ) in result
        assert "var p: auto = new_array<f32>(n);" not in result
        assert "var q: ptr<f32>" not in result

    def test_qualified_template_argument_spacing_conversion(self):
        """Test template arguments keep spaces during conversion"""
        code = """
        void host() {
            std::array<unsigned int, 4> ids{};
            std::vector<const unsigned int> flags;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var ids: std::array<unsigned int, 4> = {};" in result
        assert "var flags: std::vector<const unsigned int>;" in result
        assert "unsignedint" not in result
        assert "constunsigned" not in result

    def test_nested_template_argument_conversion(self):
        """Test nested template and pointer-qualified unique_ptr conversion"""
        code = """
        void host(int n) {
            std::vector<std::array<unsigned int, 4>> table;
            std::vector<std::vector<float>> rows;
            std::unique_ptr<const float*> pointer =
                std::make_unique<const float*>(nullptr);
            std::unique_ptr<float[], HostDeleter> owned =
                std::unique_ptr<float[], HostDeleter>(new float[n]);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var table: std::vector<std::array<unsigned int, 4>>;" in result
        assert "var rows: std::vector<std::vector<float>>;" in result
        assert "var pointer: ptr<ptr<f32>> = new<ptr<f32>>(nullptr);" in result
        assert "var owned: ptr<f32> = new_array<f32>(n);" in result
        assert "std::unique_ptr" not in result
        assert "unsignedint" not in result

    def test_type_alias_conversion(self):
        """Test typedef and using aliases survive host conversion"""
        code = """
        using HostBuffer = std::unique_ptr<float[]>;
        typedef std::vector<std::array<unsigned int, 4>> Table;
        using namespace std;

        void host(int n) {
            using LocalBuffer = std::unique_ptr<float[], HostDeleter>;
            HostBuffer h = std::make_unique<float[]>(n);
            HostBuffer* hp = &h;
            LocalBuffer owned(new float[n]);
            Table table;
            consume(h.get(), owned.get());
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "typedef ptr<f32> HostBuffer;" in result
        assert "typedef std::vector<std::array<unsigned int, 4>> Table;" in result
        assert "typedef ptr<f32> LocalBuffer;" in result
        assert "var h: HostBuffer = new_array<f32>(n);" in result
        assert "var hp: ptr<HostBuffer> = (&h);" in result
        assert "var owned: LocalBuffer = new_array<f32>(n);" in result
        assert "var table: Table;" in result
        assert "consume(h, owned);" in result
        assert "using namespace" not in result

    def test_typedef_multi_declarator_alias_conversion(self):
        """Test multi-declarator typedef aliases with pointers and arrays"""
        code = """
        typedef float Real, *RealPtr;
        typedef float Tile[16];
        typedef std::unique_ptr<float[]> Buffer, *BufferPtr;

        void host(int n, Real x) {
            Real y = x;
            RealPtr p;
            Tile tile;
            Buffer h = std::make_unique<float[]>(n);
            BufferPtr hp = &h;
            consume(h.get());
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "typedef f32 Real;" in result
        assert "typedef ptr<f32> RealPtr;" in result
        assert "typedef array<f32, 16> Tile;" in result
        assert "typedef ptr<f32> Buffer;" in result
        assert "typedef ptr<ptr<f32>> BufferPtr;" in result
        assert "void host(i32 n, Real x)" in result
        assert "var y: Real = x;" in result
        assert "var p: RealPtr;" in result
        assert "var tile: Tile;" in result
        assert "var h: Buffer = new_array<f32>(n);" in result
        assert "var hp: BufferPtr = (&h);" in result
        assert "consume(h);" in result
        assert "array<std::unique_ptr" not in result

    def test_type_alias_c_style_cast_conversion(self):
        """Test C-style casts to typedef aliases convert without stray statements"""
        code = """
        typedef unsigned int LaneMask;

        LaneMask helper(float x) {
            return (LaneMask)x;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "typedef u32 LaneMask;" in result
        assert "return LaneMask(x);" in result
        assert "return LaneMask;" not in result
        assert "x;" not in result

    def test_range_based_for_loop_conversion(self):
        """Test C++ range-based for loops lower to CrossGL for-in loops"""
        code = """
        void host(std::vector<float>& h) {
            for (auto& x : h) {
                x = 1.0f;
            }
            float sum = 0.0f;
            for (const auto& x : h) {
                sum += x;
            }
            for (float y : h) {
                sink(y);
            }
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert result.count("for x in h {") == 2
        assert "x = 1.0f;" in result
        assert "var sum: f32 = 0.0f;" in result
        assert "sum += x;" in result
        assert "for y in h {" in result
        assert "sink(y);" in result

    def test_multiline_initializer_and_packed_launch_conversion(self):
        """Test multiline brace initializers and packed launch args convert"""
        code = """
        void host(float* data, int n) {
            dim3 grid(16);
            dim3 block(32);
            void* packedArgs[] = {
                &data,
                &n,
            };
            float matrix[2][2] = {
                {1.0f, 2.0f},
                {3.0f, 4.0f},
            };
            hipLaunchKernelGGL(kernel, grid, block, 0, 0, packedArgs);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var packedArgs: array<ptr<void>> = {(&data), (&n)};" in result
        assert (
            "var matrix: array<array<f32, 2>, 2> = " "{{1.0f, 2.0f}, {3.0f, 4.0f}};"
        ) in result
        assert "// Kernel launch: kernel<<<grid, block, 0, 0>>>()" in result
        assert "// Arguments: data, n" in result

    def test_fixed_array_initializer_conversion(self):
        """Test fixed arrays and brace initializer conversion"""
        code = """
        float weights[4] = {1.0f, 2.0f, 3.0f, 4.0f};

        struct Filter {
            float taps[3];
            float matrix[2][2];
        };

        __global__ void kernel(float input[4]) {
            float local[2] = {1.0f, 2.0f};
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var weights: array<f32, 4> = {1.0f, 2.0f, 3.0f, 4.0f};" in result
        assert "array<f32, 3> taps;" in result
        assert "array<array<f32, 2>, 2> matrix;" in result
        assert "array<f32, 4> input" in result
        assert "var local: array<f32, 2> = {1.0f, 2.0f};" in result

    def test_designated_initializer_conversion(self):
        """Test C99 designated initializer conversion"""
        code = """
        struct Pair {
            float x;
            float y;
        };

        void host() {
            int values[4] = { [2] = 7, [0] = 1 };
            Pair point = { .y = 2.0f, .x = 1.0f };
            Pair points[2] = {
                [0].y = 2.0f,
                [1].x = 3.0f,
            };
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var values: array<i32, 4> = {[2] = 7, [0] = 1};" in result
        assert "var point: Pair = {.y = 2.0f, .x = 1.0f};" in result
        assert (
            "var points: array<Pair, 2> = " "{[0].y = 2.0f, [1].x = 3.0f};"
        ) in result

    def test_kernel_pointer_parameters_lower_to_storage_arrays(self):
        """Test pointer kernel params lower to storage arrays of element type"""
        code = """
        __global__ void kernel(float* data, const int* indices, float value) {
            data[indices[0]] = value;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "data: array<f32>" in result
        assert "indices: array<i32>" in result
        assert "array<ptr" not in result

    def test_kernel_pointer_parameters_roundtrip_to_rust(self, tmp_path):
        code = """
        __global__ void kernel(float* data, const int* indices, float value) {
            data[indices[0]] = value;
        }
        """
        source_path = tmp_path / "kernel.hip"
        source_path.write_text(code)

        result = translate(str(source_path), backend="rust", format_output=False)

        assert "pub fn kernel(data: Vec<f32>, indices: Vec<i32>, value: f32)" in result
        assert "data[indices[0] as usize] = value;" in result

    def test_qualified_declaration_conversion(self):
        """Test const, unsigned, and static declaration conversion"""
        code = """
        static float cached = 1.0f;
        unsigned int mask = 3u;
        signed int signedMask = -1;
        long long wide = 2ll;
        unsigned long long uwide = 3ull;

        __global__ void kernel(unsigned int* out, const float scale, long long x) {
            const int local = 1;
            unsigned int idx = 2u;
            unsigned long long y = 1ull;
            long long z = (long long)x;
            static float tmp = 0.0f;
            out[0] = idx;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var cached: f32 = 1.0f;" in result
        assert "var mask: u32 = 3u;" in result
        assert "var signedMask: i32 = (-1);" in result
        assert "var wide: i64 = 2ll;" in result
        assert "var uwide: u64 = 3ull;" in result
        assert "out: array<u32>" in result
        assert "f32 scale" in result
        assert "i64 x" in result
        assert "var local: i32 = 1;" in result
        assert "var idx: u32 = 2u;" in result
        assert "var y: u64 = 1ull;" in result
        assert "var z: i64 = i64(x);" in result
        assert "var tmp: f32 = 0.0f;" in result

    def test_qualified_and_pointer_return_function_conversion(self):
        """Test qualified scalar and pointer return conversion"""
        code = """
        unsigned int lane_mask() { return 3u; }
        float* get_data(float* data) { return data; }
        const float* get_const_data(const float* data) { return data; }
        static inline unsigned int helper(unsigned int x) { return x; }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "u32 lane_mask()" in result
        assert "ptr<f32> get_data(ptr<f32> data)" in result
        assert "ptr<f32> get_const_data(ptr<f32> data)" in result
        assert "u32 helper(u32 x)" in result

    def test_expression_statements_and_for_header_conversion(self):
        """Test expression statements and for headers are preserved"""
        code = """
        void helper(float* data, int n) {
            int i = 0;
            i += 1;
            sink(i);
            for (int j = 0; j < n; j++) {
                sink(j);
            }
            __syncthreads();
            __threadfence();
            __threadfence_block();
            __threadfence_system();
            return;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "i += 1;" in result
        assert "sink(i);" in result
        assert "for (var j: i32 = 0; (j < n); (j++))" in result
        assert "sink(j);" in result
        assert "workgroupBarrier();" in result
        assert result.count("memoryBarrier();") == 3
        assert "__threadfence" not in result
        assert "None" not in result

    def test_c_style_for_structured_assignment_updates_conversion(self):
        """Test array and member assignment targets survive for updates"""
        code = """
        void helper(float* values, int n) {
            int value = 1;
            for (int i = 0; i < n; values[i] += value) {
                values[i] = value;
            }
            for (int j = 0; j < n; object.field = value) {
                sink(j);
            }
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "for (var i: i32 = 0; (i < n); values[i] += value) {" in result
        assert "values[i] = value;" in result
        assert "for (var j: i32 = 0; (j < n); object.field = value) {" in result
        assert "sink(j);" in result

    def test_assignment_expression_conversion(self):
        """Test nested HIP assignment expressions convert without stray output"""
        code = """
        void f() {
            int a = 0;
            int b = 0;
            int c = 0;
            a = b = c;
            int d = (a = b);
            sink(a = 1);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "a = b = c;" in result
        assert "var d: i32 = a = b;" in result
        assert "sink(a = 1);" in result
        assert "None" not in result

    def test_local_pointer_declarations_and_unary_pointer_conversion(self):
        """Test local pointer declarations, address-of, and dereference"""
        code = """
        void helper(float* data, unsigned int* ids) {
            float* p = data;
            const float* cp = data;
            unsigned int* ip = ids;
            float x = 1.0f;
            float* q = &x;
            *p = *q;
            ip[0] = 1u;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var p: ptr<f32> = data;" in result
        assert "var cp: ptr<f32> = data;" in result
        assert "var ip: ptr<u32> = ids;" in result
        assert "var q: ptr<f32> = (&x);" in result
        assert "(*p) = (*q);" in result
        assert "ip[0] = 1u;" in result
        assert "BinaryOpNode" not in result
        assert "UnaryOpNode" not in result

    def test_pointer_member_access_operator_conversion(self):
        """Test pointer and value member access preserve their operators"""
        code = """
        struct Item { int value; };
        void helper(Item* p, Item v) {
            int a = p->value;
            int b = v.value;
            p->value = b;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var a: i32 = p->value;" in result
        assert "var b: i32 = v.value;" in result
        assert "p->value = b;" in result

    def test_bitwise_logical_and_shift_expression_conversion(self):
        """Test bitwise, shift, logical, and compound shift conversion"""
        code = """
        unsigned int helper(unsigned int a, unsigned int b) {
            unsigned int x = (a & b) | (a ^ b);
            unsigned int y = x << 2;
            unsigned int z = y >> 1;
            if (a > 0 && b > 0 || a == b) {
                z <<= 1;
                z >>= 1;
            }
            return z;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var x: u32 = ((a & b) | (a ^ b));" in result
        assert "var y: u32 = (x << 2);" in result
        assert "var z: u32 = (y >> 1);" in result
        assert "if ((((a > 0) && (b > 0)) || (a == b))) {" in result
        assert "z <<= 1;" in result
        assert "z >>= 1;" in result
        assert "BinaryOpNode" not in result

    def test_numeric_literal_conversion(self):
        """Test integer mask and float suffix literals are preserved"""
        code = """
        unsigned int helper() {
            unsigned int mask = 0xffu;
            unsigned int bits = 0b1010u;
            unsigned int oct = 0777u;
            float x = 1e-3f;
            float y = .5f;
            return mask | bits | oct;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var mask: u32 = 0xffu;" in result
        assert "var bits: u32 = 0b1010u;" in result
        assert "var oct: u32 = 0777u;" in result
        assert "var x: f32 = 1e-3f;" in result
        assert "var y: f32 = .5f;" in result
        assert "return ((mask | bits) | oct);" in result

    def test_boolean_null_and_character_literal_conversion(self):
        """Test bool, null, nullptr, and character literals are preserved"""
        code = r"""
        bool helper(int* ptr) {
            bool yes = true;
            bool no = false;
            char c = 'x';
            char escaped = '\n';
            int* p = nullptr;
            int* q = NULL;
            return yes && !no;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var yes: bool = true;" in result
        assert "var no: bool = false;" in result
        assert "var c: i8 = 'x';" in result
        assert "var escaped: i8 = '\\n';" in result
        assert "var p: ptr<i32> = nullptr;" in result
        assert "var q: ptr<i32> = NULL;" in result
        assert "return (yes && (!no));" in result

    def test_control_flow_and_cast_expression_conversion(self):
        """Test while, do-while, switch, ternary, and casts are emitted"""
        code = """
        int helper(float x, int n) {
            int i = 0;
            while (i < n) {
                i += 1;
            }
            do {
                i += 1;
            } while (i < n);
            switch (i) {
                case 1:
                    i += 2;
                    break;
                default:
                    i += 3;
                    break;
            }
            return i > 0 ? (int)x : n;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "while ((i < n)) {" in result
        assert "do {" in result
        assert "} while ((i < n));" in result
        assert "switch (i) {" in result
        assert "case 1:" in result
        assert "default:" in result
        assert "return ((i > 0) ? i32(x) : n);" in result
        assert "WhileNode" not in result
        assert "DoWhileNode" not in result
        assert "SwitchNode" not in result
        assert "TernaryOpNode" not in result
        assert "CastNode" not in result

    def test_empty_default_switch_codegen_emits_default_label(self):
        """Test empty default labels are emitted instead of dropped."""
        code = """
        void f(int value) {
            switch (value) {
                case 0:
                    break;
                default:
            }
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "switch (value) {" in result
        assert "case 0:" in result
        assert "default:" in result
        assert result.index("case 0:") < result.index("default:")

    def test_switch_codegen_preserves_default_before_later_case_order(self):
        """Test default labels are not reordered behind later case labels."""
        code = """
        void f(int value) {
            switch (value) {
                default:
                    value += 1;
                case 1:
                    value += 2;
            }
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "switch (value) {" in result
        assert "default:" in result
        assert "case 1:" in result
        assert result.index("default:") < result.index("case 1:")
