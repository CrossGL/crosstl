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

    def test_threadfence_converts_to_crossgl_memory_barrier(self):
        """Test HIP thread fence variants convert back to CrossGL memoryBarrier."""
        code = """
        __global__ void fence(float* out) {
            __threadfence();
            __threadfence_block();
            __threadfence_system();
            __syncthreads();
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert result.count("memoryBarrier();") == 3
        assert "workgroupBarrier();" in result
        assert "__threadfence" not in result

    def test_syncwarp_mask_emits_explicit_diagnostic(self):
        """Test HIP warp synchronization emits an explicit CrossGL diagnostic."""
        code = """
        __global__ void sync(unsigned int mask) {
            __syncwarp(mask);
            __syncwarp();
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert "// __syncwarp(mask) not directly supported in CrossGL" in result
        assert "// __syncwarp() not directly supported in CrossGL" in result
        assert "Warp sync not directly supported in CrossGL" not in result
        assert "None" not in result

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
                0,
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
            "cubeLayerSurface, vec4<i32>(pixel.x, pixel.y, 0, 0));" in result
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

    def test_hip_sparse_texture_fetch_helpers_emit_diagnostics(self):
        """Test sparse HIP texture fetches import as explicit diagnostics."""
        code = """
        void sparseTextureOps(
            hipTextureObject_t lineTex,
            hipTextureObject_t tex,
            hipTextureObject_t volumeTex,
            hipTextureObject_t layerTex,
            bool* resident,
            float x,
            float2 uv,
            float3 uvw,
            int layer,
            float lod
        ) {
            float4 sparseFetched = tex1Dfetch<float4>(lineTex, 3, resident);
            float4 sparseLine = tex1D<float4>(lineTex, x, resident);
            float4 sparseSample = tex2D<float4>(tex, uv.x, uv.y, resident);
            float4 sparseLod = tex2DLod<float4>(tex, uv.x, uv.y, lod, resident);
            float4 sparseGrad = tex2DGrad<float4>(
                tex,
                uv.x,
                uv.y,
                uv,
                uv,
                resident
            );
            float4 sparseVolume = tex3D<float4>(
                volumeTex,
                uvw.x,
                uvw.y,
                uvw.z,
                resident
            );
            float4 sparseLayer = tex2DLayered<float4>(
                layerTex,
                uv.x,
                uv.y,
                layer,
                resident
            );
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert "sampler1D lineTex" in result
        assert "sampler2D tex" in result
        assert "sampler3D volumeTex" in result
        assert "sampler2DArray layerTex" in result
        for name, helper in (
            ("sparseFetched", "tex1Dfetch"),
            ("sparseLine", "tex1D"),
            ("sparseSample", "tex2D"),
            ("sparseLod", "tex2DLod"),
            ("sparseGrad", "tex2DGrad"),
            ("sparseVolume", "tex3D"),
            ("sparseLayer", "tex2DLayered"),
        ):
            assert (
                f"var {name}: vec4<f32> = "
                f"(/* hip texture.{helper} sparse residency not directly "
                "supported in CrossGL */ vec4<f32>(0.0, 0.0, 0.0, 0.0));"
            ) in result
            assert f"{helper}<float4>(" not in result

    def test_hip_sparse_texture_layered_cube_helpers_emit_diagnostics(self):
        """Test sparse HIP layered/cube texture forms emit diagnostics."""
        code = """
        void sparseTextureFamilyOps(
            hipTextureObject_t lineLayerTex,
            hipTextureObject_t volumeTex,
            hipTextureObject_t cubeTex,
            hipTextureObject_t cubeLayerTex,
            bool* resident,
            float x,
            float dx,
            float dy,
            float3 uvw,
            float3 dPdx,
            float3 dPdy,
            int layer,
            float lod
        ) {
            float4 sparse1DLayer = tex1DLayered<float4>(
                lineLayerTex,
                x,
                layer,
                resident
            );
            float4 sparse1DLayerLod = tex1DLayeredLod<float4>(
                lineLayerTex,
                x,
                layer,
                lod,
                resident
            );
            float4 sparse1DLayerGrad = tex1DLayeredGrad<float4>(
                lineLayerTex,
                x,
                layer,
                dx,
                dy,
                resident
            );
            float4 sparse3DLod = tex3DLod<float4>(
                volumeTex,
                uvw.x,
                uvw.y,
                uvw.z,
                lod,
                resident
            );
            float4 sparse3DGrad = tex3DGrad<float4>(
                volumeTex,
                uvw.x,
                uvw.y,
                uvw.z,
                dPdx,
                dPdy,
                resident
            );
            float4 sparseCube = texCubemap<float4>(
                cubeTex,
                uvw.x,
                uvw.y,
                uvw.z,
                resident
            );
            float4 sparseCubeLod = texCubemapLod<float4>(
                cubeTex,
                uvw.x,
                uvw.y,
                uvw.z,
                lod,
                resident
            );
            float4 sparseCubeGrad = texCubemapGrad<float4>(
                cubeTex,
                uvw.x,
                uvw.y,
                uvw.z,
                dPdx,
                dPdy,
                resident
            );
            float4 sparseCubeLayer = texCubemapLayered<float4>(
                cubeLayerTex,
                uvw.x,
                uvw.y,
                uvw.z,
                layer,
                resident
            );
            float4 sparseCubeLayerLod = texCubemapLayeredLod<float4>(
                cubeLayerTex,
                uvw.x,
                uvw.y,
                uvw.z,
                layer,
                lod,
                resident
            );
            float4 sparseCubeLayerGrad = texCubemapLayeredGrad<float4>(
                cubeLayerTex,
                uvw.x,
                uvw.y,
                uvw.z,
                layer,
                dPdx,
                dPdy,
                resident
            );
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert "sampler1DArray lineLayerTex" in result
        assert "sampler3D volumeTex" in result
        assert "samplerCube cubeTex" in result
        assert "samplerCubeArray cubeLayerTex" in result
        for name, helper in (
            ("sparse1DLayer", "tex1DLayered"),
            ("sparse1DLayerLod", "tex1DLayeredLod"),
            ("sparse1DLayerGrad", "tex1DLayeredGrad"),
            ("sparse3DLod", "tex3DLod"),
            ("sparse3DGrad", "tex3DGrad"),
            ("sparseCube", "texCubemap"),
            ("sparseCubeLod", "texCubemapLod"),
            ("sparseCubeGrad", "texCubemapGrad"),
            ("sparseCubeLayer", "texCubemapLayered"),
            ("sparseCubeLayerLod", "texCubemapLayeredLod"),
            ("sparseCubeLayerGrad", "texCubemapLayeredGrad"),
        ):
            assert (
                f"var {name}: vec4<f32> = "
                f"(/* hip texture.{helper} sparse residency not directly "
                "supported in CrossGL */ vec4<f32>(0.0, 0.0, 0.0, 0.0));"
            ) in result
            assert f"{helper}<float4>(" not in result

    def test_hip_texture_scalar_coordinates_emit_rank_diagnostics(self):
        """Test malformed scalar texture coordinates emit explicit diagnostics."""
        code = """
        void textureCoordinateRankOps(
            hipTextureObject_t lineLayerTex,
            hipTextureObject_t layerTex,
            hipTextureObject_t cubeTex,
            hipTextureObject_t cubeLayerTex,
            float x,
            float2 uv,
            float3 uvw,
            float4 uvwl,
            float lod,
            float3 dPdx,
            float3 dPdy,
            float4 d4Pdx,
            float4 d4Pdy
        ) {
            float4 badLineLayer = tex1DLayered<float4>(lineLayerTex, x);
            float4 badLayerGrad = tex2DLayeredGrad<float4>(
                layerTex,
                uv,
                dPdx,
                dPdy
            );
            float4 badCube = texCubemap<float4>(cubeTex, x);
            float4 badCubeLayerLod = texCubemapLayeredLod<float4>(
                cubeLayerTex,
                uvw,
                lod
            );
            float4 goodLineLayer = tex1DLayered<float4>(lineLayerTex, uv);
            float4 goodLayerGrad = tex2DLayeredGrad<float4>(
                layerTex,
                uvw,
                dPdx,
                dPdy
            );
            float4 goodCube = texCubemap<float4>(cubeTex, uvw);
            float4 goodCubeLayerLod = texCubemapLayeredLod<float4>(
                cubeLayerTex,
                uvwl,
                lod
            );
            float4 goodCubeLayerGrad = texCubemapLayeredGrad<float4>(
                cubeLayerTex,
                uvwl,
                d4Pdx,
                d4Pdy
            );
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert "sampler1DArray lineLayerTex" in result
        assert "sampler2DArray layerTex" in result
        assert "samplerCube cubeTex" in result
        assert "samplerCubeArray cubeLayerTex" in result
        for name, helper in (
            ("badLineLayer", "tex1DLayered"),
            ("badLayerGrad", "tex2DLayeredGrad"),
            ("badCube", "texCubemap"),
            ("badCubeLayerLod", "texCubemapLayeredLod"),
        ):
            assert (
                f"var {name}: vec4<f32> = "
                f"(/* hip texture.{helper} coordinate rank mismatch not directly "
                "supported in CrossGL */ vec4<f32>(0.0, 0.0, 0.0, 0.0));"
            ) in result

        assert "texture(lineLayerTex, x)" not in result
        assert "textureGrad(layerTex, uv, dPdx, dPdy)" not in result
        assert "texture(cubeTex, x)" not in result
        assert "textureLod(cubeLayerTex, uvw, lod)" not in result
        assert "var goodLineLayer: vec4<f32> = texture(lineLayerTex, uv);" in result
        assert (
            "var goodLayerGrad: vec4<f32> = textureGrad("
            "layerTex, uvw, dPdx, dPdy);" in result
        )
        assert "var goodCube: vec4<f32> = texture(cubeTex, uvw);" in result
        assert (
            "var goodCubeLayerLod: vec4<f32> = textureLod("
            "cubeLayerTex, uvwl, lod);" in result
        )
        assert (
            "var goodCubeLayerGrad: vec4<f32> = textureGrad("
            "cubeLayerTex, uvwl, d4Pdx, d4Pdy);" in result
        )

    def test_hip_surface_malformed_coordinates_emit_rank_diagnostics(self):
        """Test malformed surface coordinates emit explicit diagnostics."""
        code = """
        void surfaceCoordinateRankOps(
            hipSurfaceObject_t lineLayerSurface,
            hipSurfaceObject_t layerSurface,
            hipSurfaceObject_t cubeSurface,
            hipSurfaceObject_t cubeLayerSurface,
            int x,
            int y,
            int layer,
            int face,
            int2 pixel,
            int3 xyz,
            float4 value
        ) {
            float4 badLineLayer = surf1DLayeredread<float4>(
                lineLayerSurface,
                x * sizeof(float4)
            );
            float4 badCubeVector = surfCubemapread<float4>(
                cubeSurface,
                pixel,
                face
            );
            float4 badCubeLayerVector = surfCubemapLayeredread<float4>(
                cubeLayerSurface,
                xyz,
                face,
                layer,
                0
            );
            float4 badLayerByPointer;
            surf2DLayeredread<float4>(
                &badLayerByPointer,
                layerSurface,
                pixel,
                layer
            );
            surf2DLayeredwrite(value, layerSurface, pixel, layer);
            surfCubemapLayeredwrite(
                value,
                cubeLayerSurface,
                xyz,
                face,
                layer,
                0
            );
            float4 goodLineLayer = surf1DLayeredread<float4>(
                lineLayerSurface,
                x * sizeof(float4),
                layer
            );
            float4 goodCube = surfCubemapread<float4>(
                cubeSurface,
                x * sizeof(float4),
                y,
                face
            );
            surf2DLayeredwrite(
                value,
                layerSurface,
                x * sizeof(float4),
                y,
                layer
            );
            surfCubemapLayeredwrite(
                value,
                cubeLayerSurface,
                x * sizeof(float4),
                y,
                face,
                layer
            );
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert "image1DArray lineLayerSurface" in result
        assert "image2DArray layerSurface" in result
        assert "imageCube cubeSurface" in result
        assert "imageCubeArray cubeLayerSurface" in result
        for name, helper in (
            ("badLineLayer", "surf1DLayeredread"),
            ("badCubeVector", "surfCubemapread"),
            ("badCubeLayerVector", "surfCubemapLayeredread"),
        ):
            assert (
                f"var {name}: vec4<f32> = "
                f"(/* hip surface.{helper} coordinate shape mismatch not directly "
                "supported in CrossGL */ vec4<f32>(0.0, 0.0, 0.0, 0.0));"
            ) in result

        assert (
            "badLayerByPointer = "
            "(/* hip surface.surf2DLayeredread coordinate shape mismatch "
            "not directly supported in CrossGL */ "
            "vec4<f32>(0.0, 0.0, 0.0, 0.0));"
        ) in result
        for helper in ("surf2DLayeredwrite", "surfCubemapLayeredwrite"):
            assert (
                f"(/* hip surface.{helper} coordinate shape mismatch not directly "
                "supported in CrossGL */ 0);"
            ) in result

        assert "imageLoad(cubeSurface, vec3<i32>(pixel, face" not in result
        assert (
            "imageLoad(cubeLayerSurface, vec4<i32>(xyz, face, layer, 0))" not in result
        )
        assert "imageStore(layerSurface, vec3<i32>(pixel, layer), value)" not in result
        assert (
            "imageStore(cubeLayerSurface, vec4<i32>(xyz, face, layer, 0), value)"
            not in result
        )
        assert (
            "var goodLineLayer: vec4<f32> = imageLoad("
            "lineLayerSurface, vec2<i32>(x, layer));"
        ) in result
        assert (
            "var goodCube: vec4<f32> = imageLoad("
            "cubeSurface, vec3<i32>(x, y, face));"
        ) in result
        assert "imageStore(layerSurface, vec3<i32>(x, y, layer), value);" in result
        assert (
            "imageStore(cubeLayerSurface, vec4<i32>(x, y, face, layer), value);"
            in result
        )

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

    def test_hip_runtime_launch_support_api_conversion(self):
        """Test lower-level HIP launch support APIs emit metadata comments."""
        code = """
        void host(void* kernel, hipFunction_t function, hipStream_t stream, void** args, const hipLaunchConfig_t* config, const HIP_LAUNCH_CONFIG* driverConfig) {
            dim3 grid(16);
            dim3 block(32);
            dim3 outGrid;
            dim3 outBlock;
            size_t sharedMem;
            hipStream_t outStream;
            hipEvent_t startEvent;
            hipEvent_t stopEvent;
            hipLaunchParams* launchParams;
            int value = 7;
            size_t offset = 0;

            hipConfigureCall(grid, block, 0, stream);
            __hipPushCallConfiguration(grid, block, 0, stream);
            __hipPopCallConfiguration(&outGrid, &outBlock, &sharedMem, &outStream);
            hipSetupArgument(&value, sizeof(value), offset);
            hipLaunchByPtr(kernel);
            hipLaunchKernelExC(config, kernel, args);
            hipDrvLaunchKernelEx(driverConfig, function, args, NULL);
            hipExtLaunchKernel(
                (const void*)kernel, grid, block, args, 0, stream, startEvent,
                stopEvent, hipExtAnyOrderLaunch
            );
            hipExtLaunchKernelGGL(
                kernel, grid, block, 0, stream, startEvent, stopEvent,
                hipExtAnyOrderLaunch, kernel, args
            );
            hipExtLaunchMultiKernelMultiDevice(launchParams, 2, hipExtAnyOrderLaunch);
            hipExtModuleLaunchKernel(
                function, 512, 1, 1, 64, 1, 1, 0, stream, args, NULL, startEvent,
                stopEvent, hipExtAnyOrderLaunch
            );

            hipError_t err = hipConfigureCall(grid, block, 0, stream);
            err = __hipPushCallConfiguration(grid, block, 0, stream);
            err = __hipPopCallConfiguration(
                &outGrid, &outBlock, &sharedMem, &outStream
            );
            err = hipSetupArgument(&value, sizeof(value), offset);
            err = hipLaunchByPtr(kernel);
            err = hipLaunchKernelExC(config, kernel, args);
            err = hipDrvLaunchKernelEx(driverConfig, function, args, NULL);
            err = hipExtLaunchKernel(
                (const void*)kernel, grid, block, args, 0, stream, startEvent,
                stopEvent, hipExtAnyOrderLaunch
            );
            err = hipExtLaunchMultiKernelMultiDevice(
                launchParams, 2, hipExtAnyOrderLaunch
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
                "// HIP configure call: grid: grid, block: block, "
                "shared memory: 0, stream: stream"
            )
            == 2
        )
        assert (
            result.count(
                "// HIP push call configuration: grid: grid, block: block, "
                "shared memory: 0, stream: stream"
            )
            == 2
        )
        assert (
            result.count(
                "// HIP pop call configuration: grid output: outGrid, "
                "block output: outBlock, shared memory output: sharedMem, "
                "stream output: outStream"
            )
            == 2
        )
        assert (
            result.count(
                "// HIP setup kernel argument: value: (&value), "
                "bytes: sizeof(value), offset: offset"
            )
            == 2
        )
        assert result.count("// HIP launch by pointer: function: kernel") == 2
        assert (
            result.count(
                "// HIP launch kernel ex: config: config, function: kernel, "
                "args: args"
            )
            == 2
        )
        assert (
            result.count(
                "// HIP driver launch kernel ex: config: driverConfig, "
                "function: function, params: args, extra: NULL"
            )
            == 2
        )
        assert (
            result.count(
                "// HIP extended kernel launch: function: kernel, grid: grid, "
                "block: block, args: args, shared memory: 0, stream: stream, "
                "start event: startEvent, stop event: stopEvent, "
                "flags: hipExtAnyOrderLaunch"
            )
            == 2
        )
        assert (
            "// HIP extended kernel launch GGL: function: kernel, grid: grid, "
            "block: block, shared memory: 0, stream: stream, "
            "start event: startEvent, stop event: stopEvent, "
            "flags: hipExtAnyOrderLaunch, args: kernel, args"
        ) in result
        assert (
            result.count(
                "// HIP extended multi-kernel multi-device launch: "
                "params: launchParams, devices: 2, flags: hipExtAnyOrderLaunch"
            )
            == 2
        )
        assert (
            "// HIP extended module launch kernel: function: function, "
            "global work size: (512, 1, 1), local work size: (64, 1, 1), "
            "shared memory: 0, stream: stream, params: args, extra: NULL, "
            "start event: startEvent, stop event: stopEvent, "
            "flags: hipExtAnyOrderLaunch"
        ) in result
        assert "var err: hipError_t = hipSuccess;" in result
        assert "err = hipSuccess;" in result

        for function_name in [
            "hipConfigureCall",
            "__hipPushCallConfiguration",
            "__hipPopCallConfiguration",
            "hipSetupArgument",
            "hipLaunchByPtr",
            "hipLaunchKernelExC",
            "hipDrvLaunchKernelEx",
            "hipExtLaunchKernel",
            "hipExtLaunchKernelGGL",
            "hipExtLaunchMultiKernelMultiDevice",
            "hipExtModuleLaunchKernel",
        ]:
            assert f"{function_name}(" not in result

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
            size_t granularity;
            float symbol;
            hipStream_t stream;
            unsigned int flags;
            unsigned long long accessFlags;
            int pointerAttrValue;
            void* rangeAttributeData;
            size_t rangeAttributeSizes;
            hipMemRangeAttribute rangeAttribute;
            hipPointerAttribute_t pointerAttrs;
            hipDeviceptr_t devicePtr;
            hipDeviceptr_t devicePtr2;
            hipDeviceptr_t basePtr;
            void* driverHost;
            void* virtualAddress;
            void* preferredAddress;
            void* shareableHandle;
            void* mappedPointer;
            void* resourceDesc;
            void* textureDesc;
            void* viewDesc;
            void* copyParams;
            void** copyDsts;
            void** copySrcs;
            size_t* copySizes;
            size_t* copyAttrIndices;
            size_t failIndex;
            unsigned int glBuffer;
            unsigned int glImage;
            unsigned int glTarget;
            hipArray_t array;
            hipArray_t levelArray;
            HIP_ARRAY_DESCRIPTOR arrayDesc;
            HIP_ARRAY3D_DESCRIPTOR array3DDesc;
            hipChannelFormatDesc desc;
            hipPitchedPtr pitched;
            hipExtent extent;
            hip_Memcpy2D copy2DParams;
            HIP_MEMCPY3D driverCopyParams;
            hipMemcpy3DPeerParms peerCopyParams;
            hipMemcpyAttributes copyAttrs;
            hipMemcpy3DBatchOp batch3DOp;
            hipMemPool_t pool;
            hipMemPool_t importedPool;
            hipMemPoolProps poolProps;
            hipMemPoolPtrExportData poolExportData;
            hipMemAccessDesc accessDesc;
            hipMemLocation location;
            hipMemGenericAllocationHandle_t allocationHandle;
            hipMemGenericAllocationHandle_t importedHandle;
            hipMemAllocationProp allocationProp;
            hipMemAllocationHandleType handleType;
            hipExternalMemory_t externalMemory;
            hipExternalMemoryHandleDesc memoryHandleDesc;
            hipExternalMemoryBufferDesc bufferDesc;
            hipExternalMemoryMipmappedArrayDesc mipmapDesc;
            hipMipmappedArray_t mipmappedArray;
            hipExternalSemaphore_t externalSemaphore;
            hipExternalSemaphoreHandleDesc semaphoreHandleDesc;
            hipExternalSemaphoreSignalParams signalParams;
            hipExternalSemaphoreWaitParams waitParams;
            hipGraphicsResource_t graphicsResource;
            hipGraphicsResource_t imageResource;
            hipTextureObject_t texObj;
            hipSurfaceObject_t surfObj;
            hipResourceDesc outResourceDesc;
            hipTextureDesc outTextureDesc;
            hipResourceViewDesc outViewDesc;
            hipPointer_attribute pointerAttribute;
            hipIpcMemHandle_t ipcMemHandle;
            hipIpcEventHandle_t ipcEventHandle;
            hipEvent_t ipcEvent;
            hipProfilerStart();
            hipMalloc((void**)&d, n * sizeof(float));
            hipExtMallocWithFlags((void**)&d, n * sizeof(float), hipDeviceMallocDefault);
            hipMallocAsync((void**)&d, n * sizeof(float), stream);
            hipMemPoolCreate(&pool, &poolProps);
            hipDeviceGetDefaultMemPool(&pool, 0);
            hipDeviceSetMemPool(0, pool);
            hipDeviceGetMemPool(&pool, 0);
            hipMallocFromPoolAsync((void**)&d2, n * sizeof(float), pool, stream);
            hipMemPoolTrimTo(pool, n * sizeof(float));
            hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &ptrSize);
            hipMemPoolGetAttribute(pool, hipMemPoolAttrReleaseThreshold, &ptrSize);
            hipMemPoolSetAccess(pool, &accessDesc, 1);
            hipMemPoolGetAccess(&flags, pool, &location);
            hipMemPoolExportToShareableHandle(
                shareableHandle, pool, handleType, 0
            );
            hipMemPoolImportFromShareableHandle(
                &importedPool, shareableHandle, handleType, 0
            );
            hipMemPoolExportPointer(&poolExportData, d);
            hipMemPoolImportPointer(&mappedPointer, importedPool, &poolExportData);
            hipMemPrefetchAsync(d, n * sizeof(float), 0, stream);
            hipMemPrefetchAsync_v2(d, n * sizeof(float), location, 0, stream);
            hipMemAdvise(d, n * sizeof(float), hipMemAdviseSetReadMostly, 0);
            hipMemAdvise_v2(
                d, n * sizeof(float), hipMemAdviseSetPreferredLocation, location
            );
            hipMemRangeGetAttribute(
                &accessFlags, sizeof(accessFlags), hipMemRangeAttributeAccessedBy,
                d, n * sizeof(float)
            );
            hipMemRangeGetAttributes(
                &rangeAttributeData, &rangeAttributeSizes, &rangeAttribute, 1,
                d, n * sizeof(float)
            );
            hipStreamAttachMemAsync(stream, d, n * sizeof(float), hipMemAttachSingle);
            hipMemAlloc(&devicePtr, n * sizeof(float));
            hipMemAllocPitch(&devicePtr2, &pitch, n * sizeof(float), 4, 4);
            hipMemAllocHost(&driverHost, n * sizeof(float));
            hipMemHostAlloc(&driverHost, n * sizeof(float), hipHostMallocDefault);
            hipMemHostGetDevicePointer(&devicePtr, driverHost, 0);
            hipMemGetAddressRange(&basePtr, &ptrSize, devicePtr);
            hipMemcpyHtoD(devicePtr, h, n * sizeof(float));
            hipMemcpyHtoDAsync(devicePtr, h, n * sizeof(float), stream);
            hipMemcpyDtoH(h, devicePtr, n * sizeof(float));
            hipMemcpyDtoHAsync(h, devicePtr, n * sizeof(float), stream);
            hipMemcpyDtoD(devicePtr2, devicePtr, n * sizeof(float));
            hipMemcpyDtoDAsync(devicePtr2, devicePtr, n * sizeof(float), stream);
            hipMemcpyAtoH(h, array, 0, n * sizeof(float));
            hipMemcpyAtoHAsync(h, array, 0, n * sizeof(float), stream);
            hipMemcpyHtoA(array, 4, h, n * sizeof(float));
            hipMemcpyHtoAAsync(array, 4, h, n * sizeof(float), stream);
            hipMemcpyAtoD(devicePtr, array, 8, n * sizeof(float));
            hipMemcpyDtoA(array, 12, devicePtr, n * sizeof(float));
            hipMemcpyAtoA(array, 16, array, 20, n * sizeof(float));
            hipMemsetD8(devicePtr, 0, n);
            hipMemsetD8Async(devicePtr, 1, n, stream);
            hipMemsetD16(devicePtr, 0, n);
            hipMemsetD16Async(devicePtr, 1, n, stream);
            hipMemsetD32(devicePtr, 0, n);
            hipMemsetD32Async(devicePtr, 1, n, stream);
            hipIpcGetMemHandle(&ipcMemHandle, d);
            hipIpcOpenMemHandle(
                &mappedPointer, ipcMemHandle, hipIpcMemLazyEnablePeerAccess
            );
            hipIpcCloseMemHandle(mappedPointer);
            hipIpcGetEventHandle(&ipcEventHandle, ipcEvent);
            hipIpcOpenEventHandle(&ipcEvent, ipcEventHandle);
            hipMemGetAllocationGranularity(
                &granularity, &allocationProp, hipMemAllocationGranularityMinimum
            );
            hipMemCreate(&allocationHandle, n * sizeof(float), &allocationProp, 0);
            hipMemAddressReserve(
                &virtualAddress, n * sizeof(float), granularity, preferredAddress, 0
            );
            hipMemMap(virtualAddress, n * sizeof(float), 0, allocationHandle, 0);
            hipMemSetAccess(virtualAddress, n * sizeof(float), &accessDesc, 1);
            hipMemGetAccess(&accessFlags, &location, virtualAddress);
            hipMemGetAllocationPropertiesFromHandle(
                &allocationProp, allocationHandle
            );
            hipMemRetainAllocationHandle(&importedHandle, virtualAddress);
            hipMemExportToShareableHandle(
                &shareableHandle, allocationHandle, handleType, 0
            );
            hipMemImportFromShareableHandle(
                &importedHandle, shareableHandle, handleType
            );
            hipImportExternalMemory(&externalMemory, &memoryHandleDesc);
            hipExternalMemoryGetMappedBuffer(
                &mappedPointer, externalMemory, &bufferDesc
            );
            hipExternalMemoryGetMappedMipmappedArray(
                &mipmappedArray, externalMemory, &mipmapDesc
            );
            hipImportExternalSemaphore(&externalSemaphore, &semaphoreHandleDesc);
            hipSignalExternalSemaphoresAsync(
                &externalSemaphore, &signalParams, 1, stream
            );
            hipWaitExternalSemaphoresAsync(
                &externalSemaphore, &waitParams, 1, stream
            );
            hipGraphicsGLRegisterBuffer(
                &graphicsResource, glBuffer, hipGraphicsRegisterFlagsWriteDiscard
            );
            hipGraphicsGLRegisterImage(
                &imageResource, glImage, glTarget, hipGraphicsRegisterFlagsSurfaceLoadStore
            );
            hipGraphicsMapResources(1, &graphicsResource, stream);
            hipGraphicsResourceGetMappedPointer(
                &mappedPointer, &ptrSize, graphicsResource
            );
            hipGraphicsSubResourceGetMappedArray(&array, imageResource, 0, 0);
            hipGraphicsUnmapResources(1, &graphicsResource, stream);
            hipHostMalloc((void**)&h, n * sizeof(float), hipHostMallocMapped);
            hipHostAlloc((void**)&h, n * sizeof(float), hipHostMallocDefault);
            hipHostRegister(h, n * sizeof(float), hipHostRegisterMapped);
            hipHostGetDevicePointer((void**)&d, h, 0);
            hipHostGetFlags(&flags, h);
            hipMallocPitch((void**)&d2, &pitch, n * sizeof(float), 4);
            hipMallocArray(&array, &desc, n, 4, hipArrayDefault);
            hipMalloc3D(&pitched, extent);
            hipMalloc3DArray(&array, &desc, extent, hipArrayDefault);
            hipArrayCreate(&array, &arrayDesc);
            hipArray3DCreate(&array, &array3DDesc);
            hipArrayGetDescriptor(&arrayDesc, array);
            hipArray3DGetDescriptor(&array3DDesc, array);
            hipArrayGetInfo(&desc, &extent, &flags, array);
            hipMallocMipmappedArray(
                &mipmappedArray, &desc, extent, 4, hipArrayDefault
            );
            hipMipmappedArrayCreate(&mipmappedArray, &array3DDesc, 4);
            hipGetMipmappedArrayLevel(&levelArray, mipmappedArray, 1);
            hipMipmappedArrayGetLevel(&levelArray, mipmappedArray, 2);
            hipMemcpy(d, h, n * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpyWithStream(
                d, h, n * sizeof(float), hipMemcpyHostToDevice, stream
            );
            hipMemcpyPeer(d2, 1, d, 0, n * sizeof(float));
            hipMemcpyPeerAsync(d2, 1, d, 0, n * sizeof(float), stream);
            hipMemcpy2D(
                d2, pitch, h, n * sizeof(float), n * sizeof(float), 4,
                hipMemcpyHostToDevice
            );
            hipMemcpy2DAsync(
                d2, pitch, h, n * sizeof(float), n * sizeof(float), 4,
                hipMemcpyHostToDevice, 0
            );
            hipMemcpyToArray(
                array, 0, 0, h, n * sizeof(float), hipMemcpyHostToDevice
            );
            hipMemcpyToArrayAsync(
                array, 0, 0, h, n * sizeof(float), hipMemcpyHostToDevice, stream
            );
            hipMemcpyFromArray(
                h, array, 0, 0, n * sizeof(float), hipMemcpyDeviceToHost
            );
            hipMemcpyFromArrayAsync(
                h, array, 0, 0, n * sizeof(float), hipMemcpyDeviceToHost, stream
            );
            hipMemcpy2DToArray(
                array, 0, 0, h, pitch, n * sizeof(float), 4, hipMemcpyHostToDevice
            );
            hipMemcpy2DToArrayAsync(
                array, 0, 0, h, pitch, n * sizeof(float), 4,
                hipMemcpyHostToDevice, stream
            );
            hipMemcpy2DFromArray(
                h, pitch, array, 0, 0, n * sizeof(float), 4, hipMemcpyDeviceToHost
            );
            hipMemcpy2DFromArrayAsync(
                h, pitch, array, 0, 0, n * sizeof(float), 4,
                hipMemcpyDeviceToHost, stream
            );
            hipMemcpyArrayToArray(
                array, 0, 0, array, 4, 0, n * sizeof(float),
                hipMemcpyDeviceToDevice
            );
            hipMemcpy2DArrayToArray(
                array, 0, 0, array, 4, 0, n * sizeof(float), 4,
                hipMemcpyDeviceToDevice
            );
            hipMemcpyToSymbol(
                symbol, h, n * sizeof(float), 0, hipMemcpyHostToDevice
            );
            hipMemcpyToSymbolAsync(
                symbol, h, n * sizeof(float), 0, hipMemcpyHostToDevice, stream
            );
            hipMemcpyFromSymbol(
                h, symbol, n * sizeof(float), 0, hipMemcpyDeviceToHost
            );
            hipMemcpyFromSymbolAsync(
                h, symbol, n * sizeof(float), 0, hipMemcpyDeviceToHost, stream
            );
            hipGetSymbolAddress((void**)&d, symbol);
            hipGetSymbolSize(&ptrSize, symbol);
            hipMemcpyParam2D(&copy2DParams);
            hipMemcpyParam2DAsync(&copy2DParams, stream);
            hipMemcpy3D(&copyParams);
            hipMemcpy3DAsync(&copyParams, 0);
            hipDrvMemcpy3D(&driverCopyParams);
            hipDrvMemcpy3DAsync(&driverCopyParams, stream);
            hipMemcpyBatchAsync(
                copyDsts, copySrcs, copySizes, 1, &copyAttrs, copyAttrIndices, 1,
                &failIndex, stream
            );
            hipMemcpy3DBatchAsync(1, &batch3DOp, &failIndex, 0ull, stream);
            hipMemcpy3DPeer(&peerCopyParams);
            hipMemcpy3DPeerAsync(&peerCopyParams, stream);
            hipMemGetInfo(&freeMem, &totalMem);
            hipPointerGetAttributes(&pointerAttrs, d);
            hipDrvPointerGetAttributes(
                1, &pointerAttribute, &rangeAttributeData, devicePtr
            );
            hipPointerGetAttribute(&pointerAttrValue, hipPointerAttributeMemoryType, d);
            hipPointerSetAttribute(&pointerAttrValue, hipPointerAttributeMemoryType, d);
            hipMemPtrGetInfo(d, &ptrSize);
            hipGetChannelDesc(&desc, array);
            hipCreateTextureObject(
                &texObj, resourceDesc, textureDesc, viewDesc
            );
            hipTexObjectCreate(&texObj, resourceDesc, textureDesc, viewDesc);
            hipGetTextureObjectResourceDesc(&outResourceDesc, texObj);
            hipGetTextureObjectTextureDesc(&outTextureDesc, texObj);
            hipGetTextureObjectResourceViewDesc(&outViewDesc, texObj);
            hipTexObjectGetResourceDesc(&outResourceDesc, texObj);
            hipTexObjectGetTextureDesc(&outTextureDesc, texObj);
            hipTexObjectGetResourceViewDesc(&outViewDesc, texObj);
            hipCreateSurfaceObject(&surfObj, resourceDesc);
            hipDestroyTextureObject(texObj);
            hipTexObjectDestroy(texObj);
            hipDestroySurfaceObject(surfObj);
            hipGraphicsUnregisterResource(imageResource);
            hipGraphicsUnregisterResource(graphicsResource);
            hipDestroyExternalSemaphore(externalSemaphore);
            hipFreeMipmappedArray(mipmappedArray);
            hipMipmappedArrayDestroy(mipmappedArray);
            hipDestroyExternalMemory(externalMemory);
            hipMemset(d, 0, n * sizeof(float));
            hipMemset2D(d2, pitch, 0, n * sizeof(float), 4);
            hipMemset2DAsync(d2, pitch, 1, n * sizeof(float), 4, 0);
            hipMemset3D(pitched, 0, extent);
            hipMemset3DAsync(pitched, 1, extent, 0);
            hipMemsetD2D8(devicePtr, pitch, 0, n, 4);
            hipMemsetD2D8Async(devicePtr, pitch, 1, n, 4, stream);
            hipMemsetD2D16(devicePtr, pitch, 0, n, 4);
            hipMemsetD2D16Async(devicePtr, pitch, 1, n, 4, stream);
            hipMemsetD2D32(devicePtr, pitch, 0, n, 4);
            hipMemsetD2D32Async(devicePtr, pitch, 1, n, 4, stream);
            hipDeviceSynchronize();
            hipHostUnregister(h);
            hipFree(d);
            hipFreeAsync(d2, stream);
            hipHostFree(h);
            hipFreeHost(h);
            hipFreeArray(array);
            hipArrayDestroy(array);
            hipMemFreeHost(driverHost);
            hipMemFree(devicePtr2);
            hipMemFree(devicePtr);
            hipMemUnmap(virtualAddress, n * sizeof(float));
            hipMemRelease(importedHandle);
            hipMemRelease(allocationHandle);
            hipMemAddressFree(virtualAddress, n * sizeof(float));
            hipProfilerStop();
            hipMemPoolDestroy(pool);
            hipError_t err = hipMalloc((void**)&d, n * sizeof(float));
            err = hipMallocAsync((void**)&d, n * sizeof(float), stream);
            err = hipMemPoolCreate(&pool, &poolProps);
            err = hipDeviceGetMemPool(&pool, 0);
            err = hipMemPoolExportToShareableHandle(
                shareableHandle, pool, handleType, 0
            );
            err = hipMemPoolImportPointer(
                &mappedPointer, importedPool, &poolExportData
            );
            err = hipMemPrefetchAsync(d, n * sizeof(float), 0, stream);
            err = hipMemAdvise(d, n * sizeof(float), hipMemAdviseSetReadMostly, 0);
            err = hipMemRangeGetAttribute(
                &accessFlags, sizeof(accessFlags), hipMemRangeAttributeAccessedBy,
                d, n * sizeof(float)
            );
            err = hipStreamAttachMemAsync(
                stream, d, n * sizeof(float), hipMemAttachSingle
            );
            err = hipProfilerStart();
            err = hipMemAlloc(&devicePtr, n * sizeof(float));
            err = hipMemcpyHtoD(devicePtr, h, n * sizeof(float));
            err = hipMemsetD32(devicePtr, 0, n);
            err = hipIpcOpenMemHandle(
                &mappedPointer, ipcMemHandle, hipIpcMemLazyEnablePeerAccess
            );
            err = hipMemCreate(&allocationHandle, n * sizeof(float), &allocationProp, 0);
            err = hipMemMap(virtualAddress, n * sizeof(float), 0, allocationHandle, 0);
            err = hipMemSetAccess(virtualAddress, n * sizeof(float), &accessDesc, 1);
            err = hipMemAddressFree(virtualAddress, n * sizeof(float));
            err = hipImportExternalMemory(&externalMemory, &memoryHandleDesc);
            err = hipGraphicsMapResources(1, &graphicsResource, stream);
            err = hipFreeAsync(d2, stream);
            err = hipHostRegister(h, n * sizeof(float), hipHostRegisterMapped);
            err = hipMallocPitch((void**)&d2, &pitch, n * sizeof(float), 4);
            err = hipMallocArray(&array, &desc, n, 4, hipArrayDefault);
            err = hipMemcpyPeer(d2, 1, d, 0, n * sizeof(float));
            err = hipMemcpyToArray(
                array, 0, 0, h, n * sizeof(float), hipMemcpyHostToDevice
            );
            err = hipMemcpy2DFromArray(
                h, pitch, array, 0, 0, n * sizeof(float), 4,
                hipMemcpyDeviceToHost
            );
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
        result_lines = [line.strip() for line in result.splitlines()]

        assert (
            result.count("// HIP memory allocate: d, bytes: (n * sizeof(float))") == 2
        )
        assert (
            "// HIP extended memory allocate: d, bytes: (n * sizeof(float)), "
            "flags: hipDeviceMallocDefault"
        ) in result
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
            result.count("// HIP get device memory pool: output: pool, device: 0") == 2
        )
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
            result.count(
                "// HIP memory pool export to shareable handle: "
                "output: shareableHandle, pool: pool, handle type: handleType, flags: 0"
            )
            == 2
        )
        assert (
            "// HIP memory pool import from shareable handle: output: importedPool, "
            "handle: shareableHandle, handle type: handleType, flags: 0"
        ) in result
        assert (
            "// HIP memory pool export pointer: output: poolExportData, pointer: d"
            in result
        )
        assert (
            result.count(
                "// HIP memory pool import pointer: output: mappedPointer, "
                "pool: importedPool, export data: (&poolExportData)"
            )
            == 2
        )
        assert (
            result.count(
                "// HIP memory prefetch: pointer: d, bytes: (n * sizeof(float)), "
                "device: 0, stream: stream"
            )
            == 2
        )
        assert (
            "// HIP memory prefetch v2: pointer: d, bytes: (n * sizeof(float)), "
            "location: location, flags: 0, stream: stream"
        ) in result
        assert (
            result.count(
                "// HIP memory advise: pointer: d, bytes: (n * sizeof(float)), "
                "advice: hipMemAdviseSetReadMostly, device: 0"
            )
            == 2
        )
        assert (
            "// HIP memory advise v2: pointer: d, bytes: (n * sizeof(float)), "
            "advice: hipMemAdviseSetPreferredLocation, location: location"
        ) in result
        assert (
            result.count(
                "// HIP memory range get attribute: output: accessFlags, "
                "output bytes: sizeof(accessFlags), "
                "attribute: hipMemRangeAttributeAccessedBy, pointer: d, "
                "range bytes: (n * sizeof(float))"
            )
            == 2
        )
        assert (
            "// HIP memory range get attributes: outputs: (&rangeAttributeData), "
            "output sizes: (&rangeAttributeSizes), attributes: (&rangeAttribute), "
            "attribute count: 1, pointer: d, range bytes: (n * sizeof(float))"
        ) in result
        assert (
            result.count(
                "// HIP stream attach memory: stream: stream, pointer: d, "
                "bytes: (n * sizeof(float)), flags: hipMemAttachSingle"
            )
            == 2
        )
        assert (
            result.count(
                "// HIP driver memory allocate: output: devicePtr, "
                "bytes: (n * sizeof(float))"
            )
            == 2
        )
        assert (
            "// HIP driver pitched memory allocate: output: devicePtr2, "
            "pitch output: pitch, width: (n * sizeof(float)), height: 4, "
            "element bytes: 4"
        ) in result
        assert (
            "// HIP driver host memory allocate: output: driverHost, "
            "bytes: (n * sizeof(float))"
        ) in result
        assert (
            "// HIP driver host memory allocate: output: driverHost, "
            "bytes: (n * sizeof(float)), flags: hipHostMallocDefault"
        ) in result
        assert (
            "// HIP driver host device pointer: output: devicePtr, "
            "host: driverHost, flags: 0"
        ) in result
        assert (
            "// HIP driver memory address range: base output: basePtr, "
            "size output: ptrSize, pointer: devicePtr"
        ) in result
        assert (
            result_lines.count(
                "// HIP driver memory copy host to device: source: h, "
                "destination: devicePtr, bytes: (n * sizeof(float))"
            )
            == 2
        )
        assert (
            "// HIP driver memory copy host to device: source: h, "
            "destination: devicePtr, bytes: (n * sizeof(float)), stream: stream"
        ) in result
        assert (
            "// HIP driver memory copy device to host: source: devicePtr, "
            "destination: h, bytes: (n * sizeof(float))"
        ) in result
        assert (
            "// HIP driver memory copy device to host: source: devicePtr, "
            "destination: h, bytes: (n * sizeof(float)), stream: stream"
        ) in result
        assert (
            "// HIP driver memory copy device to device: source: devicePtr, "
            "destination: devicePtr2, bytes: (n * sizeof(float))"
        ) in result
        assert (
            "// HIP driver memory copy device to device: source: devicePtr, "
            "destination: devicePtr2, bytes: (n * sizeof(float)), stream: stream"
        ) in result
        assert (
            "// HIP driver memory copy array to host: source array: array, "
            "source offset: 0, destination host: h, bytes: (n * sizeof(float))"
        ) in result
        assert (
            "// HIP driver memory copy array to host: source array: array, "
            "source offset: 0, destination host: h, bytes: (n * sizeof(float)), "
            "stream: stream"
        ) in result
        assert (
            "// HIP driver memory copy host to array: source host: h, "
            "destination array: array, destination offset: 4, "
            "bytes: (n * sizeof(float))"
        ) in result
        assert (
            "// HIP driver memory copy host to array: source host: h, "
            "destination array: array, destination offset: 4, "
            "bytes: (n * sizeof(float)), stream: stream"
        ) in result
        assert (
            "// HIP driver memory copy array to device: source array: array, "
            "source offset: 8, destination device: devicePtr, "
            "bytes: (n * sizeof(float))"
        ) in result
        assert (
            "// HIP driver memory copy device to array: source device: devicePtr, "
            "destination array: array, destination offset: 12, "
            "bytes: (n * sizeof(float))"
        ) in result
        assert (
            "// HIP driver memory copy array to array: source array: array, "
            "source offset: 20, destination array: array, "
            "destination offset: 16, bytes: (n * sizeof(float))"
        ) in result
        assert (
            "// HIP driver memory set 8-bit: pointer: devicePtr, value: 0, count: n"
            in result
        )
        assert (
            "// HIP driver memory set 8-bit: pointer: devicePtr, value: 1, "
            "count: n, stream: stream"
        ) in result
        assert (
            "// HIP driver memory set 16-bit: pointer: devicePtr, value: 0, count: n"
            in result
        )
        assert (
            "// HIP driver memory set 16-bit: pointer: devicePtr, value: 1, "
            "count: n, stream: stream"
        ) in result
        assert (
            result.count(
                "// HIP driver memory set 32-bit: pointer: devicePtr, "
                "value: 0, count: n"
            )
            == 2
        )
        assert (
            "// HIP driver memory set 32-bit: pointer: devicePtr, value: 1, "
            "count: n, stream: stream"
        ) in result
        assert (
            "// HIP IPC get memory handle: output: ipcMemHandle, pointer: d" in result
        )
        assert (
            result.count(
                "// HIP IPC open memory handle: output: mappedPointer, "
                "handle: ipcMemHandle, flags: hipIpcMemLazyEnablePeerAccess"
            )
            == 2
        )
        assert "// HIP IPC close memory handle: pointer: mappedPointer" in result
        assert (
            "// HIP IPC get event handle: output: ipcEventHandle, event: ipcEvent"
            in result
        )
        assert (
            "// HIP IPC open event handle: output: ipcEvent, handle: ipcEventHandle"
            in result
        )
        assert (
            "// HIP virtual memory allocation granularity: output: granularity, "
            "properties: (&allocationProp), option: hipMemAllocationGranularityMinimum"
        ) in result
        assert (
            result.count(
                "// HIP virtual memory create allocation: output: allocationHandle, "
                "bytes: (n * sizeof(float)), properties: (&allocationProp), flags: 0"
            )
            == 2
        )
        assert (
            "// HIP virtual memory reserve address: output: virtualAddress, "
            "bytes: (n * sizeof(float)), alignment: granularity, "
            "address: preferredAddress, flags: 0"
        ) in result
        assert (
            result.count(
                "// HIP virtual memory map: pointer: virtualAddress, "
                "bytes: (n * sizeof(float)), offset: 0, handle: allocationHandle, "
                "flags: 0"
            )
            == 2
        )
        assert (
            result.count(
                "// HIP virtual memory set access: pointer: virtualAddress, "
                "bytes: (n * sizeof(float)), descriptors: (&accessDesc), count: 1"
            )
            == 2
        )
        assert (
            "// HIP virtual memory get access: output: accessFlags, "
            "location: (&location), pointer: virtualAddress"
        ) in result
        assert (
            "// HIP virtual memory allocation properties: output: allocationProp, "
            "handle: allocationHandle"
        ) in result
        assert (
            "// HIP virtual memory retain allocation handle: output: importedHandle, "
            "address: virtualAddress"
        ) in result
        assert (
            "// HIP virtual memory export shareable handle: output: shareableHandle, "
            "handle: allocationHandle, handle type: handleType, flags: 0"
        ) in result
        assert (
            "// HIP virtual memory import shareable handle: output: importedHandle, "
            "shareable handle: shareableHandle, handle type: handleType"
        ) in result
        assert result.count("// HIP profiler start") == 2
        assert "// HIP profiler stop" in result
        assert (
            result.count(
                "// HIP import external memory: output: externalMemory, "
                "descriptor: (&memoryHandleDesc)"
            )
            == 2
        )
        assert (
            "// HIP external memory mapped buffer: output: mappedPointer, "
            "memory: externalMemory, descriptor: (&bufferDesc)"
        ) in result
        assert (
            "// HIP external memory mapped mipmapped array: output: mipmappedArray, "
            "memory: externalMemory, descriptor: (&mipmapDesc)"
        ) in result
        assert (
            "// HIP import external semaphore: output: externalSemaphore, "
            "descriptor: (&semaphoreHandleDesc)"
        ) in result
        assert (
            "// HIP signal external semaphores: semaphores: (&externalSemaphore), "
            "params: (&signalParams), count: 1, stream: stream"
        ) in result
        assert (
            "// HIP wait external semaphores: semaphores: (&externalSemaphore), "
            "params: (&waitParams), count: 1, stream: stream"
        ) in result
        assert (
            "// HIP OpenGL register buffer: output: graphicsResource, "
            "buffer: glBuffer, flags: hipGraphicsRegisterFlagsWriteDiscard"
        ) in result
        assert (
            "// HIP OpenGL register image: output: imageResource, image: glImage, "
            "target: glTarget, flags: hipGraphicsRegisterFlagsSurfaceLoadStore"
        ) in result
        assert (
            result.count(
                "// HIP graphics map resources: count: 1, "
                "resources: (&graphicsResource), stream: stream"
            )
            == 2
        )
        assert (
            "// HIP graphics mapped pointer: pointer output: mappedPointer, "
            "size output: ptrSize, resource: graphicsResource"
        ) in result
        assert (
            "// HIP graphics mapped subresource array: output: array, "
            "resource: imageResource, array index: 0, mip level: 0"
        ) in result
        assert (
            "// HIP graphics unmap resources: count: 1, "
            "resources: (&graphicsResource), stream: stream"
        ) in result
        assert "// HIP graphics unregister resource: imageResource" in result
        assert "// HIP graphics unregister resource: graphicsResource" in result
        assert "// HIP destroy external semaphore: externalSemaphore" in result
        assert "// HIP free mipmapped array: mipmappedArray" in result
        assert "// HIP destroy external memory: externalMemory" in result
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
        assert "// HIP array create: output: array, descriptor: arrayDesc" in result
        assert (
            "// HIP 3D array create: output: array, descriptor: array3DDesc" in result
        )
        assert "// HIP array get descriptor: output: arrayDesc, array: array" in result
        assert (
            "// HIP array get 3D descriptor: output: array3DDesc, array: array"
            in result
        )
        assert (
            "// HIP array get info: desc output: desc, extent output: extent, "
            "flags output: flags, array: array"
        ) in result
        assert (
            "// HIP mipmapped array allocate: output: mipmappedArray, desc: desc, "
            "extent: extent, levels: 4, flags: hipArrayDefault"
        ) in result
        assert (
            "// HIP mipmapped array create: output: mipmappedArray, "
            "descriptor: array3DDesc, levels: 4"
        ) in result
        assert (
            "// HIP mipmapped array get level: output: levelArray, "
            "mipmapped array: mipmappedArray, level: 1"
        ) in result
        assert (
            "// HIP mipmapped array get level: output: levelArray, "
            "mipmapped array: mipmappedArray, level: 2"
        ) in result
        assert (
            "// HIP memory copy: h -> d, bytes: (n * sizeof(float)), "
            "kind: hipMemcpyHostToDevice"
        ) in result
        assert (
            "// HIP memory copy: h -> d, bytes: (n * sizeof(float)), "
            "kind: hipMemcpyHostToDevice, stream: stream"
        ) in result
        assert (
            result_lines.count(
                "// HIP peer memory copy: source: d, source device: 0, "
                "destination: d2, destination device: 1, bytes: (n * sizeof(float))"
            )
            == 2
        )
        assert (
            "// HIP peer memory copy: source: d, source device: 0, destination: d2, "
            "destination device: 1, bytes: (n * sizeof(float)), stream: stream"
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
            result_lines.count(
                "// HIP memory copy to array: source: h, destination array: array, "
                "w offset: 0, h offset: 0, bytes: (n * sizeof(float)), "
                "kind: hipMemcpyHostToDevice"
            )
            == 2
        )
        assert (
            "// HIP memory copy to array: source: h, destination array: array, "
            "w offset: 0, h offset: 0, bytes: (n * sizeof(float)), "
            "kind: hipMemcpyHostToDevice, stream: stream"
        ) in result
        assert (
            "// HIP memory copy from array: source array: array, w offset: 0, "
            "h offset: 0, destination: h, bytes: (n * sizeof(float)), "
            "kind: hipMemcpyDeviceToHost"
        ) in result
        assert (
            "// HIP memory copy from array: source array: array, w offset: 0, "
            "h offset: 0, destination: h, bytes: (n * sizeof(float)), "
            "kind: hipMemcpyDeviceToHost, stream: stream"
        ) in result
        assert (
            "// HIP 2D memory copy to array: source: h, source pitch: pitch, "
            "destination array: array, w offset: 0, h offset: 0, "
            "width: (n * sizeof(float)), height: 4, kind: hipMemcpyHostToDevice"
        ) in result
        assert (
            "// HIP 2D memory copy to array: source: h, source pitch: pitch, "
            "destination array: array, w offset: 0, h offset: 0, "
            "width: (n * sizeof(float)), height: 4, kind: hipMemcpyHostToDevice, "
            "stream: stream"
        ) in result
        assert (
            result_lines.count(
                "// HIP 2D memory copy from array: source array: array, "
                "w offset: 0, h offset: 0, destination: h, "
                "destination pitch: pitch, width: (n * sizeof(float)), height: 4, "
                "kind: hipMemcpyDeviceToHost"
            )
            == 2
        )
        assert (
            "// HIP 2D memory copy from array: source array: array, "
            "w offset: 0, h offset: 0, destination: h, destination pitch: pitch, "
            "width: (n * sizeof(float)), height: 4, kind: hipMemcpyDeviceToHost, "
            "stream: stream"
        ) in result
        assert (
            "// HIP memory copy array to array: source array: array, "
            "source w offset: 4, source h offset: 0, destination array: array, "
            "destination w offset: 0, destination h offset: 0, "
            "bytes: (n * sizeof(float)), kind: hipMemcpyDeviceToDevice"
        ) in result
        assert (
            "// HIP 2D memory copy array to array: source array: array, "
            "source w offset: 4, source h offset: 0, destination array: array, "
            "destination w offset: 0, destination h offset: 0, "
            "width: (n * sizeof(float)), height: 4, kind: hipMemcpyDeviceToDevice"
        ) in result
        assert (
            "// HIP symbol copy to: symbol, source: h, bytes: (n * sizeof(float)), "
            "offset: 0, kind: hipMemcpyHostToDevice"
        ) in result
        assert (
            "// HIP symbol copy to: symbol, source: h, bytes: (n * sizeof(float)), "
            "offset: 0, kind: hipMemcpyHostToDevice, stream: stream"
        ) in result
        assert (
            "// HIP symbol copy from: symbol, destination: h, "
            "bytes: (n * sizeof(float)), offset: 0, kind: hipMemcpyDeviceToHost"
        ) in result
        assert (
            "// HIP symbol copy from: symbol, destination: h, "
            "bytes: (n * sizeof(float)), offset: 0, kind: hipMemcpyDeviceToHost, "
            "stream: stream"
        ) in result
        assert "// HIP get symbol address: output: d, symbol: symbol" in result
        assert "// HIP get symbol size: output: ptrSize, symbol: symbol" in result
        assert "// HIP 2D parameterized memory copy: params: copy2DParams" in result
        assert (
            "// HIP 2D parameterized memory copy: params: copy2DParams, "
            "stream: stream"
        ) in result
        assert result.count("// HIP 3D memory copy: params: copyParams") == 3
        assert "// HIP 3D memory copy: params: copyParams, stream: 0" in result
        assert "// HIP driver 3D memory copy: params: driverCopyParams" in result
        assert (
            "// HIP driver 3D memory copy: params: driverCopyParams, stream: stream"
            in result
        )
        assert (
            "// HIP batched memory copy: destinations: copyDsts, sources: copySrcs, "
            "sizes: copySizes, count: 1, attributes: (&copyAttrs), "
            "attribute indices: copyAttrIndices, attribute count: 1, "
            "fail index output: failIndex, stream: stream"
        ) in result
        assert (
            "// HIP batched 3D memory copy: count: 1, operations: (&batch3DOp), "
            "fail index output: failIndex, flags: 0ull, stream: stream"
        ) in result
        assert "// HIP 3D peer memory copy: params: peerCopyParams" in result
        assert (
            "// HIP 3D peer memory copy: params: peerCopyParams, stream: stream"
            in result
        )
        assert (
            "// HIP memory info: free output: freeMem, total output: totalMem" in result
        )
        assert (
            result.count("// HIP pointer attributes: output: pointerAttrs, pointer: d")
            == 2
        )
        assert (
            "// HIP driver pointer attributes: count: 1, "
            "attributes: (&pointerAttribute), data: (&rangeAttributeData), "
            "pointer: devicePtr"
        ) in result
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
            == 3
        )
        assert "// HIP get channel desc: output: desc, array: array" in result
        assert (
            result.count(
                "// HIP texture object get resource desc: output: outResourceDesc, "
                "texture: texObj"
            )
            == 2
        )
        assert (
            result.count(
                "// HIP texture object get texture desc: output: outTextureDesc, "
                "texture: texObj"
            )
            == 2
        )
        assert (
            result.count(
                "// HIP texture object get resource view desc: output: outViewDesc, "
                "texture: texObj"
            )
            == 2
        )
        assert (
            result.count(
                "// HIP surface object create: surfObj, resource: resourceDesc"
            )
            == 2
        )
        assert result.count("// HIP texture object destroy: texObj") == 3
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
        assert (
            "// HIP driver 2D memory set 8-bit: pointer: devicePtr, pitch: pitch, "
            "value: 0, width: n, height: 4"
        ) in result
        assert (
            "// HIP driver 2D memory set 8-bit: pointer: devicePtr, pitch: pitch, "
            "value: 1, width: n, height: 4, stream: stream"
        ) in result
        assert (
            "// HIP driver 2D memory set 16-bit: pointer: devicePtr, pitch: pitch, "
            "value: 0, width: n, height: 4"
        ) in result
        assert (
            "// HIP driver 2D memory set 16-bit: pointer: devicePtr, pitch: pitch, "
            "value: 1, width: n, height: 4, stream: stream"
        ) in result
        assert (
            "// HIP driver 2D memory set 32-bit: pointer: devicePtr, pitch: pitch, "
            "value: 0, width: n, height: 4"
        ) in result
        assert (
            "// HIP driver 2D memory set 32-bit: pointer: devicePtr, pitch: pitch, "
            "value: 1, width: n, height: 4, stream: stream"
        ) in result
        assert "// HIP device synchronize" in result
        assert result.count("// HIP memory free: h") == 2
        assert "// HIP host memory unregister: h" in result
        assert "// HIP memory free: d" in result
        assert result.count("// HIP async memory free: d2, stream: stream") == 2
        assert result.count("// HIP array free: array") == 2
        assert "// HIP driver host memory free: driverHost" in result
        assert "// HIP driver memory free: devicePtr2" in result
        assert "// HIP driver memory free: devicePtr" in result
        assert (
            "// HIP virtual memory unmap: pointer: virtualAddress, "
            "bytes: (n * sizeof(float))"
        ) in result
        assert "// HIP virtual memory release allocation: importedHandle" in result
        assert "// HIP virtual memory release allocation: allocationHandle" in result
        assert (
            result.count(
                "// HIP virtual memory free address: pointer: virtualAddress, "
                "bytes: (n * sizeof(float))"
            )
            == 2
        )
        assert "// HIP memory pool destroy: pool" in result
        assert "workgroupBarrier();" not in result
        assert "var err: hipError_t = hipSuccess;" in result
        assert "if ((err != hipSuccess))" in result
        assert "err = hipSuccess;" in result
        assert "hipMalloc(ptr<ptr<void>>((&d)), (n * sizeof(float)))" not in result
        assert "hipExtMallocWithFlags(" not in result
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
        assert "hipMemPoolExportToShareableHandle(" not in result
        assert "hipMemPoolImportFromShareableHandle(" not in result
        assert "hipMemPoolExportPointer(" not in result
        assert "hipMemPoolImportPointer(" not in result
        assert "hipDeviceGetDefaultMemPool(" not in result
        assert "hipDeviceSetMemPool(" not in result
        assert "hipDeviceGetMemPool(" not in result
        assert "hipMemPrefetchAsync(" not in result
        assert "hipMemPrefetchAsync_v2(" not in result
        assert "hipMemAdvise(" not in result
        assert "hipMemAdvise_v2(" not in result
        assert "hipMemRangeGetAttribute(" not in result
        assert "hipMemRangeGetAttributes(" not in result
        assert "hipStreamAttachMemAsync(" not in result
        assert "hipMemAlloc(" not in result
        assert "hipMemAllocPitch(" not in result
        assert "hipMemFree(" not in result
        assert "hipMemAllocHost(" not in result
        assert "hipMemHostAlloc(" not in result
        assert "hipMemFreeHost(" not in result
        assert "hipMemHostGetDevicePointer(" not in result
        assert "hipMemGetAddressRange(" not in result
        assert "hipMemcpyHtoD(" not in result
        assert "hipMemcpyHtoDAsync(" not in result
        assert "hipMemcpyDtoH(" not in result
        assert "hipMemcpyDtoHAsync(" not in result
        assert "hipMemcpyDtoD(" not in result
        assert "hipMemcpyDtoDAsync(" not in result
        assert "hipMemsetD8(" not in result
        assert "hipMemsetD8Async(" not in result
        assert "hipMemsetD16(" not in result
        assert "hipMemsetD16Async(" not in result
        assert "hipMemsetD32(" not in result
        assert "hipMemsetD32Async(" not in result
        assert "hipIpcGetMemHandle(" not in result
        assert "hipIpcOpenMemHandle(" not in result
        assert "hipIpcCloseMemHandle(" not in result
        assert "hipIpcGetEventHandle(" not in result
        assert "hipIpcOpenEventHandle(" not in result
        assert "hipMemGetAllocationGranularity(" not in result
        assert "hipMemCreate(" not in result
        assert "hipMemRelease(" not in result
        assert "hipMemAddressReserve(" not in result
        assert "hipMemAddressFree(" not in result
        assert "hipMemMap(" not in result
        assert "hipMemUnmap(" not in result
        assert "hipMemSetAccess(" not in result
        assert "hipMemGetAccess(" not in result
        assert "hipMemGetAllocationPropertiesFromHandle(" not in result
        assert "hipMemRetainAllocationHandle(" not in result
        assert "hipMemExportToShareableHandle(" not in result
        assert "hipMemImportFromShareableHandle(" not in result
        assert "hipProfilerStart(" not in result
        assert "hipProfilerStop(" not in result
        assert "hipImportExternalMemory(" not in result
        assert "hipDestroyExternalMemory(" not in result
        assert "hipExternalMemoryGetMappedBuffer(" not in result
        assert "hipExternalMemoryGetMappedMipmappedArray(" not in result
        assert "hipFreeMipmappedArray(" not in result
        assert "hipImportExternalSemaphore(" not in result
        assert "hipDestroyExternalSemaphore(" not in result
        assert "hipSignalExternalSemaphoresAsync(" not in result
        assert "hipWaitExternalSemaphoresAsync(" not in result
        assert "hipGraphicsGLRegisterBuffer(" not in result
        assert "hipGraphicsGLRegisterImage(" not in result
        assert "hipGraphicsMapResources(" not in result
        assert "hipGraphicsUnmapResources(" not in result
        assert "hipGraphicsResourceGetMappedPointer(" not in result
        assert "hipGraphicsSubResourceGetMappedArray(" not in result
        assert "hipGraphicsUnregisterResource(" not in result
        assert "hipHostMalloc(" not in result
        assert "hipHostAlloc(" not in result
        assert "hipHostRegister(" not in result
        assert "hipHostGetDevicePointer(" not in result
        assert "hipHostGetFlags(" not in result
        assert "hipMallocPitch(" not in result
        assert "hipMallocArray(" not in result
        assert "hipMalloc3D(" not in result
        assert "hipMalloc3DArray(" not in result
        assert "hipArrayCreate(" not in result
        assert "hipArray3DCreate(" not in result
        assert "hipArrayGetDescriptor(" not in result
        assert "hipArray3DGetDescriptor(" not in result
        assert "hipArrayGetInfo(" not in result
        assert "hipMallocMipmappedArray(" not in result
        assert "hipMipmappedArrayCreate(" not in result
        assert "hipGetMipmappedArrayLevel(" not in result
        assert "hipMipmappedArrayGetLevel(" not in result
        assert "hipMipmappedArrayDestroy(" not in result
        assert "hipMemcpyWithStream(" not in result
        assert "hipMemcpyPeer(" not in result
        assert "hipMemcpyPeerAsync(" not in result
        assert "hipMemcpy2D(" not in result
        assert "hipMemcpy2DAsync(" not in result
        assert "hipMemcpyToArray(" not in result
        assert "hipMemcpyToArrayAsync(" not in result
        assert "hipMemcpyFromArray(" not in result
        assert "hipMemcpyFromArrayAsync(" not in result
        assert "hipMemcpy2DToArray(" not in result
        assert "hipMemcpy2DToArrayAsync(" not in result
        assert "hipMemcpy2DFromArray(" not in result
        assert "hipMemcpy2DFromArrayAsync(" not in result
        assert "hipMemcpyArrayToArray(" not in result
        assert "hipMemcpy2DArrayToArray(" not in result
        assert "hipMemcpy3D(" not in result
        assert "hipMemcpy3DAsync(" not in result
        assert "hipDrvMemcpy3D(" not in result
        assert "hipDrvMemcpy3DAsync(" not in result
        assert "hipMemcpyParam2D(" not in result
        assert "hipMemcpyParam2DAsync(" not in result
        assert "hipMemcpyBatchAsync(" not in result
        assert "hipMemcpy3DBatchAsync(" not in result
        assert "hipMemcpy3DPeer(" not in result
        assert "hipMemcpy3DPeerAsync(" not in result
        assert "hipMemcpyAtoH(" not in result
        assert "hipMemcpyAtoHAsync(" not in result
        assert "hipMemcpyHtoA(" not in result
        assert "hipMemcpyHtoAAsync(" not in result
        assert "hipMemcpyAtoD(" not in result
        assert "hipMemcpyDtoA(" not in result
        assert "hipMemcpyAtoA(" not in result
        assert "hipMemcpyToSymbol(" not in result
        assert "hipMemcpyToSymbolAsync(" not in result
        assert "hipMemcpyFromSymbol(" not in result
        assert "hipMemcpyFromSymbolAsync(" not in result
        assert "hipGetSymbolAddress(" not in result
        assert "hipGetSymbolSize(" not in result
        assert "hipMemGetInfo(" not in result
        assert "hipPointerGetAttributes(" not in result
        assert "hipDrvPointerGetAttributes(" not in result
        assert "hipPointerGetAttribute(" not in result
        assert "hipPointerSetAttribute(" not in result
        assert "hipMemPtrGetInfo(" not in result
        assert "hipMemsetD2D8(" not in result
        assert "hipMemsetD2D8Async(" not in result
        assert "hipMemsetD2D16(" not in result
        assert "hipMemsetD2D16Async(" not in result
        assert "hipMemsetD2D32(" not in result
        assert "hipMemsetD2D32Async(" not in result
        assert "hipMemset2D(" not in result
        assert "hipMemset2DAsync(" not in result
        assert "hipMemset3D(" not in result
        assert "hipMemset3DAsync(" not in result
        assert "hipGetChannelDesc(" not in result
        assert "hipCreateTextureObject(" not in result
        assert "hipTexObjectCreate(" not in result
        assert "hipGetTextureObjectResourceDesc(" not in result
        assert "hipGetTextureObjectTextureDesc(" not in result
        assert "hipGetTextureObjectResourceViewDesc(" not in result
        assert "hipTexObjectGetResourceDesc(" not in result
        assert "hipTexObjectGetTextureDesc(" not in result
        assert "hipTexObjectGetResourceViewDesc(" not in result
        assert "hipCreateSurfaceObject(" not in result
        assert "hipDestroyTextureObject(" not in result
        assert "hipTexObjectDestroy(" not in result
        assert "hipDestroySurfaceObject(" not in result
        assert "hipHostUnregister(" not in result
        assert "hipHostFree(" not in result
        assert "hipFreeHost(" not in result
        assert "hipFreeArray(" not in result
        assert "hipArrayDestroy(" not in result
        assert "err = hipDeviceSynchronize();" not in result

    def test_hip_memory_pointer_occupancy_expression_contexts_emit_status(self):
        """Test memory, pointer, and occupancy expressions emit status metadata."""
        code = """
        hipError_t memoryExpressions(
            size_t bytes,
            hipStream_t stream,
            hipMemPool_t pool,
            bool retry
        ) {
            float* devicePtr;
            float* managedPtr;
            float* pooledPtr;
            float* hostPtr;
            float* pitchedPtr;
            size_t pitch;
            size_t freeMem;
            size_t totalMem;
            size_t pointerSize;
            void* virtualAddress;
            void* preferredAddress;
            hipDeviceptr_t driverPtr;
            hipDeviceptr_t driverPitchedPtr;
            hipMemGenericAllocationHandle_t allocationHandle;
            hipMemPoolProps poolProps;
            hipMemLocation location;
            hipMemAccessDesc accessDesc;
            hipPointerAttribute_t pointerAttrs;
            hipPointer_attribute pointerAttribute;
            void* attributeData;
            unsigned long long accessFlags;
            int pointerAttrValue;
            int gridSize;
            int blockSize;
            int activeBlocks;
            void* kernel;
            bool allocated =
                hipMalloc((void**)&devicePtr, bytes) == hipSuccess;
            bool managed =
                hipMallocManaged((void**)&managedPtr, bytes) == hipSuccess;
            bool extended =
                hipExtMallocWithFlags(
                    (void**)&devicePtr,
                    bytes,
                    hipDeviceMallocDefault
                ) == hipSuccess;
            bool asyncAllocated =
                hipMallocAsync((void**)&devicePtr, bytes, stream) == hipSuccess;
            bool pooled =
                hipMallocFromPoolAsync(
                    (void**)&pooledPtr,
                    bytes,
                    pool,
                    stream
                ) == hipSuccess;
            bool hostAllocated =
                hipHostMalloc(
                    (void**)&hostPtr,
                    bytes,
                    hipHostMallocMapped
                ) == hipSuccess;
            bool pitched =
                hipMallocPitch((void**)&pitchedPtr, &pitch, bytes, 4) == hipSuccess;
            bool poolCreated =
                hipMemPoolCreate(&pool, &poolProps) == hipSuccess;
            bool infoReady = hipMemGetInfo(&freeMem, &totalMem) == hipSuccess;
            bool pointerReady =
                hipPointerGetAttributes(&pointerAttrs, devicePtr) == hipSuccess;
            bool driverPointerReady =
                hipDrvPointerGetAttributes(
                    1,
                    &pointerAttribute,
                    &attributeData,
                    driverPtr
                ) == hipSuccess;
            bool pointerAttrReady =
                hipPointerGetAttribute(
                    &pointerAttrValue,
                    hipPointerAttributeMemoryType,
                    devicePtr
                ) == hipSuccess;
            bool pointerAttrSet =
                hipPointerSetAttribute(
                    &pointerAttrValue,
                    hipPointerAttributeMemoryType,
                    devicePtr
                ) == hipSuccess;
            bool pointerInfoReady =
                hipMemPtrGetInfo(devicePtr, &pointerSize) == hipSuccess;
            bool occupancyReady =
                hipOccupancyMaxPotentialBlockSize(
                    &gridSize,
                    &blockSize,
                    kernel,
                    0,
                    0
                ) == hipSuccess;
            bool occupancyVariableReady =
                hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(
                    &gridSize,
                    &blockSize,
                    kernel,
                    0,
                    256,
                    1
                ) == hipSuccess;
            bool activeReady =
                hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                    &activeBlocks,
                    kernel,
                    blockSize,
                    0,
                    1
                ) == hipSuccess;
            if (hipMemAlloc(&driverPtr, bytes) != hipSuccess) {
                return hipMemGetAddressRange(
                    &driverPitchedPtr,
                    &pointerSize,
                    driverPtr
                );
            }
            if (hipMemAllocPitch(&driverPitchedPtr, &pitch, bytes, 4, 4) == hipSuccess) {
                hipError_t selectedDriver =
                    retry ? hipMemFree(driverPitchedPtr) : hipMemAllocHost(&hostPtr, bytes);
                return selectedDriver;
            }
            if (hipMemAddressReserve(&virtualAddress, bytes, 0, preferredAddress, 0) == hipSuccess) {
                hipError_t selectedVirtual =
                    retry ? hipMemMap(virtualAddress, bytes, 0, allocationHandle, 0) : hipMemSetAccess(virtualAddress, bytes, &accessDesc, 1);
                return selectedVirtual;
            }
            if (hipMemGetAccess(&accessFlags, &location, virtualAddress) == hipSuccess) {
                return hipMemAddressFree(virtualAddress, bytes);
            }
            hipError_t selected =
                retry ? hipFree(devicePtr) : hipMemPoolDestroy(pool);
            return selected;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "var allocated: bool = ((/* HIP memory allocate: devicePtr, "
            "bytes: bytes */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var managed: bool = ((/* HIP memory allocate: managedPtr, "
            "bytes: bytes */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var extended: bool = ((/* HIP extended memory allocate: "
            "devicePtr, bytes: bytes, flags: hipDeviceMallocDefault */ hipSuccess) "
            "== hipSuccess);"
        ) in result
        assert (
            "var pooled: bool = ((/* HIP async memory allocate from pool: "
            "pooledPtr, bytes: bytes, pool: pool, stream: stream */ hipSuccess) "
            "== hipSuccess);"
        ) in result
        assert (
            "var hostAllocated: bool = ((/* HIP host memory allocate: hostPtr, "
            "bytes: bytes, flags: hipHostMallocMapped */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var infoReady: bool = ((/* HIP memory info: free output: freeMem, "
            "total output: totalMem */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var pointerReady: bool = ((/* HIP pointer attributes: output: "
            "pointerAttrs, pointer: devicePtr */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var driverPointerReady: bool = ((/* HIP driver pointer attributes: "
            "count: 1, attributes: (&pointerAttribute), data: (&attributeData), "
            "pointer: driverPtr */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var pointerAttrSet: bool = ((/* HIP pointer set attribute: value: "
            "pointerAttrValue, attribute: hipPointerAttributeMemoryType, "
            "pointer: devicePtr */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var occupancyReady: bool = ((/* HIP occupancy max potential block "
            "size: grid output: gridSize, block output: blockSize, kernel: "
            "kernel, dynamic shared memory: 0, block size limit: 0 */ hipSuccess) "
            "== hipSuccess);"
        ) in result
        assert (
            "var activeReady: bool = ((/* HIP occupancy active blocks per "
            "multiprocessor: output: activeBlocks, kernel: kernel, block size: "
            "blockSize, dynamic shared memory: 0, flags: 1 */ hipSuccess) "
            "== hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP driver memory allocate: output: driverPtr, "
            "bytes: bytes */ hipSuccess) != hipSuccess))"
        ) in result
        assert (
            "return (/* HIP driver memory address range: base output: "
            "driverPitchedPtr, size output: pointerSize, pointer: driverPtr */ "
            "hipSuccess);"
        ) in result
        assert (
            "var selectedDriver: hipError_t = (retry ? "
            "(/* HIP driver memory free: driverPitchedPtr */ hipSuccess) : "
            "(/* HIP driver host memory allocate: output: hostPtr, bytes: bytes */ "
            "hipSuccess));"
        ) in result
        assert (
            "var selectedVirtual: hipError_t = (retry ? "
            "(/* HIP virtual memory map: pointer: virtualAddress, bytes: bytes, "
            "offset: 0, handle: allocationHandle, flags: 0 */ hipSuccess) : "
            "(/* HIP virtual memory set access: pointer: virtualAddress, "
            "bytes: bytes, descriptors: (&accessDesc), count: 1 */ hipSuccess));"
        ) in result
        assert (
            "return (/* HIP virtual memory free address: pointer: "
            "virtualAddress, bytes: bytes */ hipSuccess);"
        ) in result
        assert (
            "var selected: hipError_t = (retry ? "
            "(/* HIP memory free: devicePtr */ hipSuccess) : "
            "(/* HIP memory pool destroy: pool */ hipSuccess));"
        ) in result
        for function_name in [
            "hipMalloc",
            "hipMallocManaged",
            "hipExtMallocWithFlags",
            "hipMallocAsync",
            "hipMallocFromPoolAsync",
            "hipHostMalloc",
            "hipMallocPitch",
            "hipMemPoolCreate",
            "hipMemGetInfo",
            "hipPointerGetAttributes",
            "hipDrvPointerGetAttributes",
            "hipPointerGetAttribute",
            "hipPointerSetAttribute",
            "hipMemPtrGetInfo",
            "hipOccupancyMaxPotentialBlockSize",
            "hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags",
            "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
            "hipMemAlloc",
            "hipMemGetAddressRange",
            "hipMemAllocPitch",
            "hipMemFree",
            "hipMemAllocHost",
            "hipMemAddressReserve",
            "hipMemMap",
            "hipMemSetAccess",
            "hipMemGetAccess",
            "hipMemAddressFree",
            "hipFree",
            "hipMemPoolDestroy",
        ]:
            assert f"{function_name}(" not in result

    def test_hip_array_mipmapped_expression_contexts_emit_status(self):
        """Test array and mipmapped-array expressions emit status metadata."""
        code = """
        hipError_t acceptStatus(hipError_t status) {
            return status;
        }

        hipError_t arrayMipmapExpressions(
            HIP_ARRAY_DESCRIPTOR* arrayDesc,
            HIP_ARRAY3D_DESCRIPTOR* array3DDesc,
            hipChannelFormatDesc* channelDesc,
            hipExtent extent,
            size_t width,
            bool release
        ) {
            hipArray_t array;
            hipArray_t array3D;
            hipArray_t levelArray;
            hipMipmappedArray_t mipmappedArray;
            bool allocated =
                hipMallocArray(
                    &array,
                    channelDesc,
                    width,
                    4,
                    hipArrayDefault
                ) == hipSuccess;
            bool allocated3D =
                hipMalloc3DArray(
                    &array3D,
                    channelDesc,
                    extent,
                    hipArrayLayered
                ) == hipSuccess;
            bool created = hipArrayCreate(&array, arrayDesc) == hipSuccess;
            if (hipArray3DCreate(&array3D, array3DDesc) != hipSuccess) {
                return hipArrayDestroy(array3D);
            }
            if (hipMallocMipmappedArray(
                &mipmappedArray,
                channelDesc,
                extent,
                4,
                hipArrayDefault
            ) == hipSuccess) {
                hipError_t selectedLevel =
                    release ? hipMipmappedArrayDestroy(mipmappedArray) : hipGetMipmappedArrayLevel(&levelArray, mipmappedArray, 1);
                return selectedLevel;
            }
            hipError_t selectedCreate =
                release ? hipFreeArray(array) : hipMipmappedArrayCreate(&mipmappedArray, array3DDesc, 3);
            return acceptStatus(
                release ? hipMipmappedArrayGetLevel(&levelArray, mipmappedArray, 2) : hipArrayDestroy(array)
            );
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "var allocated: bool = ((/* HIP array allocate: array, desc: "
            "channelDesc, width: width, height: 4, flags: hipArrayDefault */ "
            "hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var allocated3D: bool = ((/* HIP 3D array allocate: array3D, "
            "desc: channelDesc, extent: extent, flags: hipArrayLayered */ "
            "hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var created: bool = ((/* HIP array create: output: array, "
            "descriptor: arrayDesc */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP 3D array create: output: array3D, "
            "descriptor: array3DDesc */ hipSuccess) != hipSuccess))"
        ) in result
        assert "return (/* HIP array free: array3D */ hipSuccess);" in result
        assert (
            "if (((/* HIP mipmapped array allocate: output: mipmappedArray, "
            "desc: channelDesc, extent: extent, levels: 4, "
            "flags: hipArrayDefault */ hipSuccess) == hipSuccess))"
        ) in result
        assert (
            "var selectedLevel: hipError_t = (release ? (/* HIP free "
            "mipmapped array: mipmappedArray */ hipSuccess) : (/* HIP "
            "mipmapped array get level: output: levelArray, mipmapped array: "
            "mipmappedArray, level: 1 */ hipSuccess));"
        ) in result
        assert (
            "var selectedCreate: hipError_t = (release ? (/* HIP array free: "
            "array */ hipSuccess) : (/* HIP mipmapped array create: output: "
            "mipmappedArray, descriptor: array3DDesc, levels: 3 */ hipSuccess));"
        ) in result
        assert (
            "return acceptStatus((release ? (/* HIP mipmapped array get level: "
            "output: levelArray, mipmapped array: mipmappedArray, level: 2 */ "
            "hipSuccess) : (/* HIP array free: array */ hipSuccess)));"
        ) in result
        for function_name in [
            "hipMallocArray",
            "hipMalloc3DArray",
            "hipArrayCreate",
            "hipArray3DCreate",
            "hipArrayDestroy",
            "hipMallocMipmappedArray",
            "hipMipmappedArrayDestroy",
            "hipGetMipmappedArrayLevel",
            "hipFreeArray",
            "hipMipmappedArrayCreate",
            "hipMipmappedArrayGetLevel",
        ]:
            assert f"{function_name}(" not in result

    def test_hip_copy_memset_symbol_expression_contexts_emit_status(self):
        """Test HIP copy/memset/symbol expressions emit status metadata."""
        code = """
        hipError_t copyExpressions(
            float* dst,
            float* src,
            hipDeviceptr_t driverDst,
            hipDeviceptr_t driverSrc,
            hipStream_t stream,
            hipArray_t array,
            bool useAsync
        ) {
            size_t bytes;
            size_t pitch;
            size_t height;
            size_t count;
            size_t symbolBytes;
            float symbol;
            hipPitchedPtr pitched;
            hipExtent extent;
            hip_Memcpy2D copy2DParams;
            void* copyParams;
            HIP_MEMCPY3D driverCopyParams;
            hipMemcpy3DPeerParms peerCopyParams;
            void** copyDsts;
            void** copySrcs;
            size_t* copySizes;
            hipMemcpyAttributes copyAttrs;
            size_t* copyAttrIndices;
            size_t failIndex;
            hipMemcpy3DBatchOp batch3DOp;
            bool copied =
                hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice) == hipSuccess;
            bool copiedAsync =
                hipMemcpyAsync(
                    dst,
                    src,
                    bytes,
                    hipMemcpyDeviceToDevice,
                    stream
                ) == hipSuccess;
            bool copiedWithStream =
                hipMemcpyWithStream(
                    dst,
                    src,
                    bytes,
                    hipMemcpyDeviceToDevice,
                    stream
                ) == hipSuccess;
            bool copiedPeer =
                hipMemcpyPeer(dst, 1, src, 0, bytes) == hipSuccess;
            bool copied2D =
                hipMemcpy2D(
                    dst,
                    pitch,
                    src,
                    pitch,
                    bytes,
                    height,
                    hipMemcpyDeviceToDevice
                ) == hipSuccess;
            bool copiedToArray =
                hipMemcpyToArray(
                    array,
                    0,
                    0,
                    src,
                    bytes,
                    hipMemcpyDeviceToDevice
                ) == hipSuccess;
            bool copiedFromArray =
                hipMemcpyFromArray(
                    dst,
                    array,
                    0,
                    0,
                    bytes,
                    hipMemcpyDeviceToDevice
                ) == hipSuccess;
            bool copiedArray =
                hipMemcpyArrayToArray(
                    array,
                    0,
                    0,
                    array,
                    4,
                    0,
                    bytes,
                    hipMemcpyDeviceToDevice
                ) == hipSuccess;
            bool copiedParam2D =
                hipMemcpyParam2D(&copy2DParams) == hipSuccess;
            bool copied3D =
                hipMemcpy3D(&copyParams) == hipSuccess;
            bool copiedDriver3D =
                hipDrvMemcpy3D(&driverCopyParams) == hipSuccess;
            bool copiedBatch =
                hipMemcpyBatchAsync(
                    copyDsts,
                    copySrcs,
                    copySizes,
                    1,
                    &copyAttrs,
                    copyAttrIndices,
                    1,
                    &failIndex,
                    stream
                ) == hipSuccess;
            bool copiedBatch3D =
                hipMemcpy3DBatchAsync(
                    1,
                    &batch3DOp,
                    &failIndex,
                    0ull,
                    stream
                ) == hipSuccess;
            bool symbolAddressReady =
                hipGetSymbolAddress((void**)&dst, symbol) == hipSuccess;
            bool symbolSizeReady =
                hipGetSymbolSize(&symbolBytes, symbol) == hipSuccess;
            bool copiedToSymbol =
                hipMemcpyToSymbol(
                    symbol,
                    src,
                    bytes,
                    0,
                    hipMemcpyDeviceToDevice
                ) == hipSuccess;
            bool copiedFromSymbol =
                hipMemcpyFromSymbol(
                    dst,
                    symbol,
                    bytes,
                    0,
                    hipMemcpyDeviceToDevice
                ) == hipSuccess;
            bool memset1D =
                hipMemset(dst, 0, bytes) == hipSuccess;
            bool memset2D =
                hipMemset2D(dst, pitch, 0, bytes, height) == hipSuccess;
            bool memset3D =
                hipMemset3D(pitched, 0, extent) == hipSuccess;
            bool driverCopied =
                hipMemcpyHtoD(driverDst, src, bytes) == hipSuccess;
            bool driverArrayCopied =
                hipMemcpyAtoH(dst, array, 0, bytes) == hipSuccess;
            bool driverMemset =
                hipMemsetD32(driverDst, 0, count) == hipSuccess;
            bool driverMemset2D =
                hipMemsetD2D32(driverDst, pitch, 0, count, height) == hipSuccess;
            if (hipMemcpy2DFromArray(
                dst,
                pitch,
                array,
                0,
                0,
                bytes,
                height,
                hipMemcpyDeviceToDevice
            ) != hipSuccess) {
                return hipMemcpyDtoH(dst, driverSrc, bytes);
            }
            if (hipMemcpy3DPeer(&peerCopyParams) == hipSuccess) {
                return hipMemcpyAtoD(driverDst, array, 0, bytes);
            }
            hipError_t selected = useAsync ? hipMemcpy3DAsync(&copyParams, stream) : hipMemsetAsync(dst, 0, bytes, stream);
            hipError_t selectedDriver = useAsync ? hipMemcpyHtoAAsync(array, 0, src, bytes, stream) : hipMemsetD2D32Async(driverDst, pitch, 0, count, height, stream);
            return useAsync ? selected : selectedDriver;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "var copied: bool = ((/* HIP memory copy: src -> dst, bytes: bytes, "
            "kind: hipMemcpyDeviceToDevice */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var copiedAsync: bool = ((/* HIP memory copy: src -> dst, bytes: "
            "bytes, kind: hipMemcpyDeviceToDevice, stream: stream */ hipSuccess) "
            "== hipSuccess);"
        ) in result
        assert (
            "var copiedPeer: bool = ((/* HIP peer memory copy: source: src, "
            "source device: 0, destination: dst, destination device: 1, "
            "bytes: bytes */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var copied2D: bool = ((/* HIP 2D memory copy: src -> dst, dst pitch: "
            "pitch, src pitch: pitch, width: bytes, height: height, "
            "kind: hipMemcpyDeviceToDevice */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var copiedToArray: bool = ((/* HIP memory copy to array: source: "
            "src, destination array: array, w offset: 0, h offset: 0, "
            "bytes: bytes, kind: hipMemcpyDeviceToDevice */ hipSuccess) "
            "== hipSuccess);"
        ) in result
        assert (
            "var copiedParam2D: bool = ((/* HIP 2D parameterized memory copy: "
            "params: copy2DParams */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var copiedBatch: bool = ((/* HIP batched memory copy: destinations: "
            "copyDsts, sources: copySrcs, sizes: copySizes, count: 1, "
            "attributes: (&copyAttrs), attribute indices: copyAttrIndices, "
            "attribute count: 1, fail index output: failIndex, stream: stream */ "
            "hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var symbolAddressReady: bool = ((/* HIP get symbol address: "
            "output: dst, symbol: symbol */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var copiedToSymbol: bool = ((/* HIP symbol copy to: symbol, "
            "source: src, bytes: bytes, offset: 0, kind: hipMemcpyDeviceToDevice "
            "*/ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var memset2D: bool = ((/* HIP 2D memory set: dst, pitch: pitch, "
            "value: 0, width: bytes, height: height */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var driverCopied: bool = ((/* HIP driver memory copy host to device: "
            "source: src, destination: driverDst, bytes: bytes */ hipSuccess) "
            "== hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP 2D memory copy from array: source array: array, "
            "w offset: 0, h offset: 0, destination: dst, destination pitch: "
            "pitch, width: bytes, height: height, kind: hipMemcpyDeviceToDevice "
            "*/ hipSuccess) != hipSuccess))"
        ) in result
        assert (
            "return (/* HIP driver memory copy device to host: source: "
            "driverSrc, destination: dst, bytes: bytes */ hipSuccess);"
        ) in result
        assert (
            "return (/* HIP driver memory copy array to device: source array: "
            "array, source offset: 0, destination device: driverDst, "
            "bytes: bytes */ hipSuccess);"
        ) in result
        assert (
            "var selected: hipError_t = (useAsync ? "
            "(/* HIP 3D memory copy: params: copyParams, stream: stream */ "
            "hipSuccess) : (/* HIP memory set: dst, value: 0, bytes: bytes, "
            "stream: stream */ hipSuccess));"
        ) in result
        assert (
            "var selectedDriver: hipError_t = (useAsync ? "
            "(/* HIP driver memory copy host to array: source host: src, "
            "destination array: array, destination offset: 0, bytes: bytes, "
            "stream: stream */ hipSuccess) : (/* HIP driver 2D memory set "
            "32-bit: pointer: driverDst, pitch: pitch, value: 0, width: count, "
            "height: height, stream: stream */ hipSuccess));"
        ) in result
        for function_name in [
            "hipMemcpy",
            "hipMemcpyAsync",
            "hipMemcpyWithStream",
            "hipMemcpyPeer",
            "hipMemcpy2D",
            "hipMemcpyToArray",
            "hipMemcpyFromArray",
            "hipMemcpyArrayToArray",
            "hipMemcpyParam2D",
            "hipMemcpy3D",
            "hipDrvMemcpy3D",
            "hipMemcpyBatchAsync",
            "hipMemcpy3DBatchAsync",
            "hipGetSymbolAddress",
            "hipGetSymbolSize",
            "hipMemcpyToSymbol",
            "hipMemcpyFromSymbol",
            "hipMemset",
            "hipMemset2D",
            "hipMemset3D",
            "hipMemcpyHtoD",
            "hipMemcpyDtoH",
            "hipMemcpyAtoH",
            "hipMemcpyHtoAAsync",
            "hipMemcpyAtoD",
            "hipMemsetD32",
            "hipMemsetD2D32",
            "hipMemsetD2D32Async",
        ]:
            assert f"{function_name}(" not in result

    def test_hip_surface_object_descriptor_query_is_explicitly_unsupported(self):
        """Test CUDA-parity surface descriptor queries stay explicit for HIP."""
        code = """
        void querySurfaceObject(hipSurfaceObject_t surfaceObj) {
            hipResourceDesc resourceDesc;
            hipGetSurfaceObjectResourceDesc(&resourceDesc, surfaceObj);
            hipError_t err = hipGetSurfaceObjectResourceDesc(
                &resourceDesc,
                surfaceObj
            );
            err = hipGetSurfaceObjectResourceDesc(&resourceDesc, surfaceObj);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        unsupported_query = (
            "// HIP surface object resource descriptor query not supported "
            "by HIP runtime: surface: surfaceObj, output: resourceDesc"
        )
        assert result.count(unsupported_query) == 3
        assert "var err: hipError_t = hipErrorNotSupported;" in result
        assert "err = hipErrorNotSupported;" in result
        assert "hipGetSurfaceObjectResourceDesc(" not in result

    def test_hip_object_descriptor_query_expression_contexts_emit_status(self):
        """Test object descriptor queries in expressions stay explicit."""
        code = """
        hipError_t queryObjectExpressions(
            hipTextureObject_t texObj,
            hipSurfaceObject_t surfaceObj
        ) {
            hipResourceDesc resourceDesc;
            hipTextureDesc textureDesc;
            hipResourceViewDesc viewDesc;
            bool resourceOk =
                hipGetTextureObjectResourceDesc(&resourceDesc, texObj) == hipSuccess;
            bool textureOk =
                hipGetTextureObjectTextureDesc(&textureDesc, texObj) == hipSuccess;
            if (hipGetSurfaceObjectResourceDesc(&resourceDesc, surfaceObj) != hipSuccess) {
                return hipGetSurfaceObjectResourceDesc(&resourceDesc, surfaceObj);
            }
            if (hipTexObjectGetResourceViewDesc(&viewDesc, texObj) == hipSuccess) {
                return hipTexObjectGetResourceDesc(&resourceDesc, texObj);
            }
            return hipTexObjectGetTextureDesc(&textureDesc, texObj);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "var resourceOk: bool = ((/* HIP texture object get resource desc: "
            "output: resourceDesc, texture: texObj */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var textureOk: bool = ((/* HIP texture object get texture desc: "
            "output: textureDesc, texture: texObj */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP surface object resource descriptor query not supported "
            "by HIP runtime: surface: surfaceObj, output: resourceDesc */ "
            "hipErrorNotSupported) != hipSuccess))"
        ) in result
        assert (
            "return (/* HIP surface object resource descriptor query not supported "
            "by HIP runtime: surface: surfaceObj, output: resourceDesc */ "
            "hipErrorNotSupported);"
        ) in result
        assert (
            "if (((/* HIP texture object get resource view desc: output: viewDesc, "
            "texture: texObj */ hipSuccess) == hipSuccess))"
        ) in result
        assert (
            "return (/* HIP texture object get resource desc: output: resourceDesc, "
            "texture: texObj */ hipSuccess);"
        ) in result
        assert (
            "return (/* HIP texture object get texture desc: output: textureDesc, "
            "texture: texObj */ hipSuccess);"
        ) in result
        assert "hipGetTextureObjectResourceDesc(" not in result
        assert "hipGetTextureObjectTextureDesc(" not in result
        assert "hipGetSurfaceObjectResourceDesc(" not in result
        assert "hipTexObjectGetResourceViewDesc(" not in result
        assert "hipTexObjectGetResourceDesc(" not in result
        assert "hipTexObjectGetTextureDesc(" not in result

    def test_hip_object_descriptor_alias_expression_contexts_emit_status(self):
        """Test texture/surface descriptor aliases stay explicit in expressions."""
        code = """
        hipError_t acceptStatus(hipError_t status) {
            return status;
        }

        hipError_t queryObjectAliasExpressions(
            hipTextureObject_t texObj,
            hipSurfaceObject_t surfaceObj,
            bool useSurface
        ) {
            hipResourceDesc resourceDesc;
            hipTextureDesc textureDesc;
            hipResourceViewDesc viewDesc;
            bool viewReady =
                hipGetTextureObjectResourceViewDesc(&viewDesc, texObj) == hipSuccess;
            hipError_t selected = useSurface ? hipGetSurfaceObjectResourceDesc(&resourceDesc, surfaceObj) : hipTexObjectGetResourceDesc(&resourceDesc, texObj);
            hipError_t textureStatus =
                hipTexObjectGetTextureDesc(&textureDesc, texObj);
            if (hipGetTextureObjectResourceViewDesc(&viewDesc, texObj) != hipSuccess) {
                return hipGetSurfaceObjectResourceDesc(&resourceDesc, surfaceObj);
            }
            return acceptStatus(useSurface ? hipGetSurfaceObjectResourceDesc(&resourceDesc, surfaceObj) : hipGetTextureObjectTextureDesc(&textureDesc, texObj));
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "var viewReady: bool = ((/* HIP texture object get resource view "
            "desc: output: viewDesc, texture: texObj */ hipSuccess) == "
            "hipSuccess);"
        ) in result
        assert (
            "var selected: hipError_t = (useSurface ? (/* HIP surface object "
            "resource descriptor query not supported by HIP runtime: surface: "
            "surfaceObj, output: resourceDesc */ hipErrorNotSupported) : "
            "(/* HIP texture object get resource desc: output: resourceDesc, "
            "texture: texObj */ hipSuccess));"
        ) in result
        assert (
            "// HIP texture object get texture desc: output: textureDesc, "
            "texture: texObj"
        ) in result
        assert ("var textureStatus: hipError_t = hipSuccess;") in result
        assert (
            "if (((/* HIP texture object get resource view desc: output: "
            "viewDesc, texture: texObj */ hipSuccess) != hipSuccess))"
        ) in result
        assert (
            "return (/* HIP surface object resource descriptor query not "
            "supported by HIP runtime: surface: surfaceObj, output: "
            "resourceDesc */ hipErrorNotSupported);"
        ) in result
        assert (
            "return acceptStatus((useSurface ? (/* HIP surface object resource "
            "descriptor query not supported by HIP runtime: surface: "
            "surfaceObj, output: resourceDesc */ hipErrorNotSupported) : "
            "(/* HIP texture object get texture desc: output: textureDesc, "
            "texture: texObj */ hipSuccess)));"
        ) in result
        assert "hipGetTextureObjectResourceViewDesc(" not in result
        assert "hipTexObjectGetResourceDesc(" not in result
        assert "hipTexObjectGetTextureDesc(" not in result
        assert "hipGetTextureObjectTextureDesc(" not in result
        assert "hipGetSurfaceObjectResourceDesc(" not in result

    def test_hip_texture_object_descriptor_member_reads_emit_metadata_expressions(
        self,
    ):
        """Test texture object descriptor output members lower to metadata."""
        code = """
        void host(
            hipTextureObject_t texObj,
            int* ints,
            size_t* dims,
            float* floats
        ) {
            hipResourceDesc resourceDesc;
            hipResourceDesc aliasResourceDesc;
            hipTextureDesc textureDesc;
            hipTextureDesc aliasTextureDesc;
            hipResourceViewDesc viewDesc;
            hipResourceViewDesc aliasViewDesc;
            hipGetTextureObjectResourceDesc(&resourceDesc, texObj);
            ints[0] = resourceDesc.resType;
            hipGetTextureObjectTextureDesc(&textureDesc, texObj);
            ints[1] = textureDesc.filterMode;
            ints[2] = textureDesc.readMode;
            ints[3] = textureDesc.normalizedCoords;
            floats[0] = textureDesc.mipmapLevelBias;
            hipGetTextureObjectResourceViewDesc(&viewDesc, texObj);
            ints[4] = viewDesc.format;
            dims[0] = viewDesc.width;
            dims[1] = viewDesc.height;
            dims[2] = viewDesc.depth;
            dims[3] = viewDesc.firstMipmapLevel;
            dims[4] = viewDesc.lastMipmapLevel;
            dims[5] = viewDesc.firstLayer;
            dims[6] = viewDesc.lastLayer;
            textureDesc.filterMode = hipFilterModePoint;
            int manualFilter = textureDesc.filterMode;
            viewDesc.width = 64;
            size_t manualWidth = viewDesc.width;
            hipError_t resErr = hipTexObjectGetResourceDesc(&aliasResourceDesc, texObj);
            ints[5] = aliasResourceDesc.resType;
            hipError_t texErr = hipTexObjectGetTextureDesc(&aliasTextureDesc, texObj);
            ints[6] = aliasTextureDesc.maxAnisotropy;
            hipError_t viewErr = hipTexObjectGetResourceViewDesc(&aliasViewDesc, texObj);
            dims[7] = aliasViewDesc.lastLayer;
            aliasViewDesc.firstLayer++;
            size_t manualLayer = aliasViewDesc.firstLayer;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "// HIP texture object get resource desc: output: resourceDesc, "
            "texture: texObj"
        ) in result
        assert (
            "// HIP texture object get texture desc: output: textureDesc, "
            "texture: texObj"
        ) in result
        assert (
            "// HIP texture object get resource view desc: output: viewDesc, "
            "texture: texObj"
        ) in result
        assert (
            "ints[0] = (/* HIP device query: "
            "textureObject.resourceDesc.resType(texObj) */ 0);"
        ) in result
        for index, member in [
            (1, "filterMode"),
            (2, "readMode"),
            (3, "normalizedCoords"),
        ]:
            assert (
                f"ints[{index}] = (/* HIP device query: "
                f"textureObject.textureDesc.{member}(texObj) */ 0);"
            ) in result
        assert (
            "floats[0] = (/* HIP device query: "
            "textureObject.textureDesc.mipmapLevelBias(texObj) */ 0);"
        ) in result
        assert (
            "ints[4] = (/* HIP device query: "
            "textureObject.resourceViewDesc.format(texObj) */ 0);"
        ) in result
        for index, member in enumerate(
            [
                "width",
                "height",
                "depth",
                "firstMipmapLevel",
                "lastMipmapLevel",
                "firstLayer",
                "lastLayer",
            ]
        ):
            assert (
                f"dims[{index}] = (/* HIP device query: "
                f"textureObject.resourceViewDesc.{member}(texObj) */ 0);"
            ) in result
        assert "textureDesc.filterMode = hipFilterModePoint;" in result
        assert "var manualFilter: i32 = textureDesc.filterMode;" in result
        assert "viewDesc.width = 64;" in result
        assert "var manualWidth: u32 = viewDesc.width;" in result
        assert "var resErr: hipError_t = hipSuccess;" in result
        assert (
            "ints[5] = (/* HIP device query: "
            "textureObject.resourceDesc.resType(texObj) */ 0);"
        ) in result
        assert "var texErr: hipError_t = hipSuccess;" in result
        assert (
            "ints[6] = (/* HIP device query: "
            "textureObject.textureDesc.maxAnisotropy(texObj) */ 0);"
        ) in result
        assert "var viewErr: hipError_t = hipSuccess;" in result
        assert (
            "dims[7] = (/* HIP device query: "
            "textureObject.resourceViewDesc.lastLayer(texObj) */ 0);"
        ) in result
        assert "(aliasViewDesc.firstLayer++);" in result
        assert "var manualLayer: u32 = aliasViewDesc.firstLayer;" in result
        assert "ints[0] = resourceDesc.resType;" not in result
        assert "ints[1] = textureDesc.filterMode;" not in result
        assert "dims[0] = viewDesc.width;" not in result
        assert "ints[5] = aliasResourceDesc.resType;" not in result
        assert "ints[6] = aliasTextureDesc.maxAnisotropy;" not in result
        assert "dims[7] = aliasViewDesc.lastLayer;" not in result
        assert "var manualFilter: i32 = (/* HIP device query:" not in result
        assert "var manualWidth: u32 = (/* HIP device query:" not in result
        assert "var manualLayer: u32 = (/* HIP device query:" not in result

    def test_hip_texture_object_nested_descriptor_member_reads_emit_metadata_expressions(
        self,
    ):
        """Test nested texture object descriptor output members lower to metadata."""
        code = """
        void host(
            hipTextureObject_t texObj,
            int* ints,
            size_t* dims,
            void** ptrs,
            hipArray_t* arrays,
            hipMipmappedArray_t* mipmaps
        ) {
            hipResourceDesc resourceDesc;
            hipResourceDesc aliasResourceDesc;
            hipTextureDesc textureDesc;
            hipTextureDesc aliasTextureDesc;
            hipGetTextureObjectResourceDesc(&resourceDesc, texObj);
            dims[0] = resourceDesc.res.pitch2D.width;
            dims[1] = resourceDesc.res.pitch2D.height;
            dims[2] = resourceDesc.res.pitch2D.pitchInBytes;
            dims[3] = resourceDesc.res.linear.sizeInBytes;
            ints[0] = resourceDesc.res.pitch2D.desc.x;
            ints[1] = resourceDesc.res.linear.desc.f;
            ptrs[0] = resourceDesc.res.linear.devPtr;
            ptrs[1] = resourceDesc.res.pitch2D.devPtr;
            hipGetTextureObjectTextureDesc(&textureDesc, texObj);
            ints[2] = textureDesc.addressMode[0];
            ints[3] = textureDesc.addressMode[1];
            textureDesc.addressMode[0] = hipAddressModeClamp;
            int manualAddress = textureDesc.addressMode[0];
            resourceDesc.res.pitch2D.width = 32;
            size_t manualPitchWidth = resourceDesc.res.pitch2D.width;
            hipError_t resErr = hipTexObjectGetResourceDesc(
                &aliasResourceDesc,
                texObj
            );
            mipmaps[0] = aliasResourceDesc.res.mipmap.mipmap;
            arrays[0] = aliasResourceDesc.res.array.array;
            ints[5] = aliasResourceDesc.res.pitch2D.desc.y;
            hipError_t texErr = hipTexObjectGetTextureDesc(
                &aliasTextureDesc,
                texObj
            );
            ints[6] = aliasTextureDesc.addressMode[2];
            aliasTextureDesc.addressMode[1]++;
            int manualAliasAddress = aliasTextureDesc.addressMode[1];
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        for index, member in [
            (0, "res.pitch2D.width"),
            (1, "res.pitch2D.height"),
            (2, "res.pitch2D.pitchInBytes"),
            (3, "res.linear.sizeInBytes"),
        ]:
            assert (
                f"dims[{index}] = (/* HIP device query: "
                f"textureObject.resourceDesc.{member}(texObj) */ 0);"
            ) in result
        for index, member in [
            (0, "res.pitch2D.desc.x"),
            (1, "res.linear.desc.f"),
        ]:
            assert (
                f"ints[{index}] = (/* HIP device query: "
                f"textureObject.resourceDesc.{member}(texObj) */ 0);"
            ) in result
        for index, member in [(2, "addressMode[0]"), (3, "addressMode[1]")]:
            assert (
                f"ints[{index}] = (/* HIP device query: "
                f"textureObject.textureDesc.{member}(texObj) */ 0);"
            ) in result
        assert "textureDesc.addressMode[0] = hipAddressModeClamp;" in result
        assert "var manualAddress: i32 = textureDesc.addressMode[0];" in result
        assert "resourceDesc.res.pitch2D.width = 32;" in result
        assert "var manualPitchWidth: u32 = resourceDesc.res.pitch2D.width;" in result
        assert "ptrs[0] = resourceDesc.res.linear.devPtr;" in result
        assert "ptrs[1] = resourceDesc.res.pitch2D.devPtr;" in result
        assert "var resErr: hipError_t = hipSuccess;" in result
        assert "mipmaps[0] = aliasResourceDesc.res.mipmap.mipmap;" in result
        assert "arrays[0] = aliasResourceDesc.res.array.array;" in result
        assert (
            "ints[5] = (/* HIP device query: "
            "textureObject.resourceDesc.res.pitch2D.desc.y(texObj) */ 0);"
        ) in result
        assert "var texErr: hipError_t = hipSuccess;" in result
        assert (
            "ints[6] = (/* HIP device query: "
            "textureObject.textureDesc.addressMode[2](texObj) */ 0);"
        ) in result
        assert "(aliasTextureDesc.addressMode[1]++);" in result
        assert (
            "var manualAliasAddress: i32 = aliasTextureDesc.addressMode[1];" in result
        )
        assert "dims[0] = resourceDesc.res.pitch2D.width;" not in result
        assert "ints[2] = textureDesc.addressMode[0];" not in result
        assert "ints[6] = aliasTextureDesc.addressMode[2];" not in result
        assert "textureObject.resourceDesc.res.linear.devPtr(texObj)" not in result
        assert "textureObject.resourceDesc.res.pitch2D.devPtr(texObj)" not in result
        assert "textureObject.resourceDesc.res.mipmap.mipmap(texObj)" not in result
        assert "textureObject.resourceDesc.res.array.array(texObj)" not in result
        assert "var manualAddress: i32 = (/* HIP device query:" not in result
        assert "var manualPitchWidth: u32 = (/* HIP device query:" not in result
        assert "var manualAliasAddress: i32 = (/* HIP device query:" not in result

    def test_hip_texture_object_descriptor_unmapped_fields_remain_raw(self):
        """Test unsupported descriptor member paths are not pseudo-queried."""
        code = """
        void host(
            hipTextureObject_t texObj,
            int* ints,
            float* floats,
            int index
        ) {
            hipTextureDesc textureDesc;
            hipGetTextureObjectTextureDesc(&textureDesc, texObj);
            ints[0] = textureDesc.reserved[0];
            ints[1] = textureDesc.addressMode[index];
            floats[0] = textureDesc.borderColor[index];
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert "ints[0] = textureDesc.reserved[0];" in result
        assert "ints[1] = textureDesc.addressMode[index];" in result
        assert "floats[0] = textureDesc.borderColor[index];" in result
        assert "textureObject.textureDesc.reserved[0](texObj)" not in result
        assert "textureObject.textureDesc.addressMode[index](texObj)" not in result
        assert "textureObject.textureDesc.borderColor[index](texObj)" not in result

    def test_hip_texture_object_border_color_descriptor_reads_emit_metadata_expressions(
        self,
    ):
        """Test HIP texture descriptor flags and border-color outputs."""
        code = """
        void host(
            hipTextureObject_t texObj,
            unsigned int* flagsOut,
            float* floats
        ) {
            hipTextureDesc textureDesc;
            hipTextureDesc aliasTextureDesc;
            hipGetTextureObjectTextureDesc(&textureDesc, texObj);
            flagsOut[0] = textureDesc.flags;
            floats[0] = textureDesc.borderColor[0];
            floats[1] = textureDesc.borderColor[1];
            floats[2] = textureDesc.borderColor[2];
            textureDesc.borderColor[0] = 1.0;
            float manualBorder = textureDesc.borderColor[0];
            textureDesc.flags = 7;
            unsigned int manualFlags = textureDesc.flags;
            hipError_t err = hipTexObjectGetTextureDesc(&aliasTextureDesc, texObj);
            floats[3] = aliasTextureDesc.borderColor[3];
            aliasTextureDesc.borderColor[1]++;
            float manualAliasBorder = aliasTextureDesc.borderColor[1];
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "flagsOut[0] = (/* HIP device query: "
            "textureObject.textureDesc.flags(texObj) */ 0);"
        ) in result
        for index in range(3):
            assert (
                f"floats[{index}] = (/* HIP device query: "
                f"textureObject.textureDesc.borderColor[{index}](texObj) */ 0);"
            ) in result
        assert "textureDesc.borderColor[0] = 1.0;" in result
        assert "var manualBorder: f32 = textureDesc.borderColor[0];" in result
        assert "textureDesc.flags = 7;" in result
        assert "var manualFlags: u32 = textureDesc.flags;" in result
        assert "var err: hipError_t = hipSuccess;" in result
        assert (
            "floats[3] = (/* HIP device query: "
            "textureObject.textureDesc.borderColor[3](texObj) */ 0);"
        ) in result
        assert "(aliasTextureDesc.borderColor[1]++);" in result
        assert "var manualAliasBorder: f32 = aliasTextureDesc.borderColor[1];" in result
        assert "flagsOut[0] = textureDesc.flags;" not in result
        assert "floats[0] = textureDesc.borderColor[0];" not in result
        assert "floats[3] = aliasTextureDesc.borderColor[3];" not in result
        assert "var manualBorder: f32 = (/* HIP device query:" not in result
        assert "var manualFlags: u32 = (/* HIP device query:" not in result
        assert "var manualAliasBorder: f32 = (/* HIP device query:" not in result

    def test_hip_channel_descriptor_expression_contexts_emit_status(self):
        """Test channel and array descriptor queries in expressions stay explicit."""
        code = """
        hipError_t acceptStatus(hipError_t status) {
            return status;
        }

        hipError_t queryChannelDescriptorExpressions(
            hipArray_t array,
            bool useChannel
        ) {
            hipChannelFormatDesc channelDesc;
            HIP_ARRAY_DESCRIPTOR arrayDesc;
            HIP_ARRAY3D_DESCRIPTOR array3DDesc;
            hipExtent extent;
            unsigned int flags;
            bool channelReady =
                hipGetChannelDesc(&channelDesc, array) == hipSuccess;
            bool arrayReady =
                hipArrayGetDescriptor(&arrayDesc, array) == hipSuccess;
            if (hipArray3DGetDescriptor(&array3DDesc, array) != hipSuccess) {
                return hipArrayGetInfo(&channelDesc, &extent, &flags, array);
            }
            hipError_t selected = useChannel ? hipGetChannelDesc(&channelDesc, array) : hipArrayGetDescriptor(&arrayDesc, array);
            return acceptStatus(hipGetChannelDesc(&channelDesc, array));
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "var channelReady: bool = ((/* HIP get channel desc: output: "
            "channelDesc, array: array */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var arrayReady: bool = ((/* HIP array get descriptor: output: "
            "arrayDesc, array: array */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP array get 3D descriptor: output: array3DDesc, "
            "array: array */ hipSuccess) != hipSuccess))"
        ) in result
        assert (
            "return (/* HIP array get info: desc output: channelDesc, "
            "extent output: extent, flags output: flags, array: array */ "
            "hipSuccess);"
        ) in result
        assert (
            "var selected: hipError_t = (useChannel ? "
            "(/* HIP get channel desc: output: channelDesc, array: array */ "
            "hipSuccess) : (/* HIP array get descriptor: output: arrayDesc, "
            "array: array */ hipSuccess));"
        ) in result
        assert (
            "return acceptStatus((/* HIP get channel desc: output: "
            "channelDesc, array: array */ hipSuccess));"
        ) in result
        assert "hipGetChannelDesc(" not in result
        assert "hipArrayGetDescriptor(" not in result
        assert "hipArray3DGetDescriptor(" not in result
        assert "hipArrayGetInfo(" not in result

    def test_hip_channel_array_descriptor_member_reads_emit_metadata_expressions(
        self,
    ):
        """Test HIP descriptor member reads lower to explicit metadata."""
        code = """
        void host(hipArray_t array, int* ints, size_t* dims) {
            hipChannelFormatDesc channelDesc;
            HIP_ARRAY_DESCRIPTOR arrayDesc;
            HIP_ARRAY3D_DESCRIPTOR array3DDesc;
            hipGetChannelDesc(&channelDesc, array);
            ints[0] = channelDesc.x;
            ints[1] = channelDesc.y;
            ints[2] = channelDesc.z;
            ints[3] = channelDesc.w;
            ints[4] = channelDesc.f;
            hipArrayGetDescriptor(&arrayDesc, array);
            dims[0] = arrayDesc.Width;
            dims[1] = arrayDesc.Height;
            ints[5] = arrayDesc.Format;
            ints[6] = arrayDesc.NumChannels;
            hipError_t err = hipArray3DGetDescriptor(&array3DDesc, array);
            dims[2] = array3DDesc.Width;
            dims[3] = array3DDesc.Height;
            dims[4] = array3DDesc.Depth;
            ints[7] = array3DDesc.Format;
            ints[8] = array3DDesc.NumChannels;
            ints[9] = array3DDesc.Flags;
            channelDesc.x = 32;
            int manualChannelX = channelDesc.x;
            arrayDesc.Width = 99;
            size_t manualWidth = arrayDesc.Width;
            array3DDesc.Flags++;
            int manualFlags = array3DDesc.Flags;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert "// HIP get channel desc: output: channelDesc, array: array" in result
        for index, member in enumerate(["x", "y", "z", "w", "f"]):
            assert (
                f"ints[{index}] = (/* HIP device query: "
                f"array.channelDesc.{member}(array) */ 0);"
            ) in result
        for index, member in [(0, "Width"), (1, "Height")]:
            assert (
                f"dims[{index}] = (/* HIP device query: "
                f"array.descriptor.{member}(array) */ 0);"
            ) in result
        for index, member in [(5, "Format"), (6, "NumChannels")]:
            assert (
                f"ints[{index}] = (/* HIP device query: "
                f"array.descriptor.{member}(array) */ 0);"
            ) in result
        for index, member in [(2, "Width"), (3, "Height"), (4, "Depth")]:
            assert (
                f"dims[{index}] = (/* HIP device query: "
                f"array.descriptor3D.{member}(array) */ 0);"
            ) in result
        for index, member in [(7, "Format"), (8, "NumChannels"), (9, "Flags")]:
            assert (
                f"ints[{index}] = (/* HIP device query: "
                f"array.descriptor3D.{member}(array) */ 0);"
            ) in result
        assert "channelDesc.x = 32;" in result
        assert "var manualChannelX: i32 = channelDesc.x;" in result
        assert "arrayDesc.Width = 99;" in result
        assert "var manualWidth: u32 = arrayDesc.Width;" in result
        assert "(array3DDesc.Flags++);" in result
        assert "var manualFlags: i32 = array3DDesc.Flags;" in result
        assert "ints[0] = channelDesc.x;" not in result
        assert "dims[0] = arrayDesc.Width;" not in result
        assert "dims[2] = array3DDesc.Width;" not in result
        assert "var manualChannelX: i32 = (/* HIP device query:" not in result
        assert "var manualWidth: u32 = (/* HIP device query:" not in result
        assert "var manualFlags: i32 = (/* HIP device query:" not in result

    def test_hip_array_info_descriptor_extent_member_reads_emit_metadata_expressions(
        self,
    ):
        """Test hipArrayGetInfo output struct members lower to metadata."""
        code = """
        void host(hipArray_t array, int* ints, size_t* dims, unsigned int* flagsOut) {
            hipChannelFormatDesc desc;
            hipChannelFormatDesc statusDesc;
            hipExtent extent;
            hipExtent statusExtent;
            unsigned int flags = 0;
            unsigned int statusFlags = 0;
            hipArrayGetInfo(&desc, &extent, &flags, array);
            ints[0] = desc.x;
            ints[1] = desc.y;
            ints[2] = desc.z;
            ints[3] = desc.w;
            ints[4] = desc.f;
            dims[0] = extent.width;
            dims[1] = extent.height;
            dims[2] = extent.depth;
            flagsOut[0] = flags;
            desc.x = 16;
            int manualDescX = desc.x;
            extent.width = 64;
            size_t manualWidth = extent.width;
            hipError_t err = hipArrayGetInfo(&statusDesc, &statusExtent, &statusFlags, array);
            ints[5] = statusDesc.f;
            dims[3] = statusExtent.height;
            flagsOut[1] = statusFlags;
            statusExtent.depth++;
            size_t manualDepth = statusExtent.depth;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "// HIP array get info: desc output: desc, extent output: extent, "
            "flags output: flags, array: array"
        ) in result
        for index, member in enumerate(["x", "y", "z", "w", "f"]):
            assert (
                f"ints[{index}] = (/* HIP device query: "
                f"array.info.channelDesc.{member}(array) */ 0);"
            ) in result
        for index, member in enumerate(["width", "height", "depth"]):
            assert (
                f"dims[{index}] = (/* HIP device query: "
                f"array.info.extent.{member}(array) */ 0);"
            ) in result
        assert (
            "flagsOut[0] = (/* HIP device query: array.info.flags(array) */ 0);"
            in result
        )
        assert "desc.x = 16;" in result
        assert "var manualDescX: i32 = desc.x;" in result
        assert "extent.width = 64;" in result
        assert "var manualWidth: u32 = extent.width;" in result
        assert "var err: hipError_t = hipSuccess;" in result
        assert (
            "ints[5] = (/* HIP device query: " "array.info.channelDesc.f(array) */ 0);"
        ) in result
        assert (
            "dims[3] = (/* HIP device query: " "array.info.extent.height(array) */ 0);"
        ) in result
        assert (
            "flagsOut[1] = (/* HIP device query: array.info.flags(array) */ 0);"
            in result
        )
        assert "(statusExtent.depth++);" in result
        assert "var manualDepth: u32 = statusExtent.depth;" in result
        assert "ints[0] = desc.x;" not in result
        assert "dims[0] = extent.width;" not in result
        assert "flagsOut[0] = flags;" not in result
        assert "ints[5] = statusDesc.f;" not in result
        assert "dims[3] = statusExtent.height;" not in result
        assert "flagsOut[1] = statusFlags;" not in result
        assert "var manualDescX: i32 = (/* HIP device query:" not in result
        assert "var manualWidth: u32 = (/* HIP device query:" not in result
        assert "var manualDepth: u32 = (/* HIP device query:" not in result

    def test_hip_object_lifecycle_expression_contexts_emit_status(self):
        """Test object lifecycle calls in expressions stay explicit."""
        code = """
        hipError_t lifecycleObjectExpressions(
            hipResourceDesc* resourceDesc,
            hipTextureDesc* textureDesc,
            hipResourceViewDesc* viewDesc,
            bool retry
        ) {
            hipTextureObject_t texObj;
            hipSurfaceObject_t surfObj;
            bool created =
                hipCreateTextureObject(
                    &texObj,
                    resourceDesc,
                    textureDesc,
                    viewDesc
                ) == hipSuccess;
            bool aliasCreated =
                hipTexObjectCreate(
                    &texObj,
                    resourceDesc,
                    textureDesc,
                    viewDesc
                ) == hipSuccess;
            if (hipCreateSurfaceObject(&surfObj, resourceDesc) == hipSuccess) {
                return hipDestroySurfaceObject(surfObj);
            }
            if (hipTexObjectDestroy(texObj) != hipSuccess) {
                hipError_t selected = retry ? hipDestroyTextureObject(texObj) : hipCreateSurfaceObject(&surfObj, resourceDesc);
                return selected;
            }
            return hipDestroyTextureObject(texObj);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "var created: bool = ((/* HIP texture object create: texObj, "
            "resource: resourceDesc, texture desc: textureDesc, resource view: "
            "viewDesc */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var aliasCreated: bool = ((/* HIP texture object create: texObj, "
            "resource: resourceDesc, texture desc: textureDesc, resource view: "
            "viewDesc */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP surface object create: surfObj, resource: resourceDesc */ "
            "hipSuccess) == hipSuccess))"
        ) in result
        assert (
            "return (/* HIP surface object destroy: surfObj */ hipSuccess);" in result
        )
        assert (
            "if (((/* HIP texture object destroy: texObj */ hipSuccess) != "
            "hipSuccess))"
        ) in result
        assert (
            "var selected: hipError_t = (retry ? "
            "(/* HIP texture object destroy: texObj */ hipSuccess) : "
            "(/* HIP surface object create: surfObj, resource: resourceDesc */ "
            "hipSuccess));"
        ) in result
        assert "return selected;" in result
        assert "return (/* HIP texture object destroy: texObj */ hipSuccess);" in result
        assert "hipCreateTextureObject(" not in result
        assert "hipTexObjectCreate(" not in result
        assert "hipCreateSurfaceObject(" not in result
        assert "hipDestroySurfaceObject(" not in result
        assert "hipTexObjectDestroy(" not in result
        assert "hipDestroyTextureObject(" not in result

    def test_hip_stream_event_graph_expression_contexts_emit_status(self):
        """Test stream/event/graph calls in expressions stay explicit."""
        code = """
        hipError_t streamEventGraphExpressions(
            hipStream_t stream,
            hipGraph_t graph,
            hipGraphExec_t exec,
            hipEvent_t event,
            hipGraphNode_t* deps,
            size_t numDeps,
            bool replay
        ) {
            hipGraph_t captured;
            hipStreamCaptureStatus captureStatus;
            unsigned long long captureId = 0ull;
            bool capturing =
                hipStreamBeginCapture(
                    stream,
                    hipStreamCaptureModeGlobal
                ) == hipSuccess;
            bool infoReady =
                hipStreamGetCaptureInfo(
                    stream,
                    &captureStatus,
                    &captureId
                ) == hipSuccess;
            bool eventReady = hipEventQuery(event) == hipSuccess;
            if (hipStreamIsCapturing(stream, &captureStatus) != hipSuccess) {
                return hipStreamEndCapture(stream, &captured);
            }
            if (hipGraphLaunch(exec, stream) == hipSuccess) {
                hipError_t selected = replay ? hipGraphUpload(exec, stream) : hipStreamUpdateCaptureDependencies(stream, deps, numDeps, 0);
                return selected;
            }
            return hipEventRecord(event, stream);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "var capturing: bool = ((/* HIP stream begin capture: "
            "stream: stream, mode: hipStreamCaptureModeGlobal */ hipSuccess) "
            "== hipSuccess);"
        ) in result
        assert (
            "var infoReady: bool = ((/* HIP stream capture info: "
            "stream: stream, status output: captureStatus, id output: "
            "captureId */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var eventReady: bool = ((/* HIP event query: event */ hipSuccess) "
            "== hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP stream is capturing: stream: stream, output: "
            "captureStatus */ hipSuccess) != hipSuccess))"
        ) in result
        assert (
            "return (/* HIP stream end capture: stream: stream, graph output: "
            "captured */ hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP graph launch: exec: exec, stream: stream */ "
            "hipSuccess) == hipSuccess))"
        ) in result
        assert (
            "var selected: hipError_t = (replay ? "
            "(/* HIP graph upload: exec: exec, stream: stream */ hipSuccess) : "
            "(/* HIP stream update capture dependencies: stream: stream, "
            "dependencies: deps, count: numDeps, flags: 0 */ hipSuccess));"
        ) in result
        assert "return selected;" in result
        assert (
            "return (/* HIP event record: event, stream: stream */ hipSuccess);"
            in result
        )
        for function_name in [
            "hipStreamBeginCapture",
            "hipStreamGetCaptureInfo",
            "hipEventQuery",
            "hipStreamIsCapturing",
            "hipStreamEndCapture",
            "hipGraphLaunch",
            "hipGraphUpload",
            "hipStreamUpdateCaptureDependencies",
            "hipEventRecord",
        ]:
            assert f"{function_name}(" not in result

    def test_hip_graph_lifecycle_expression_contexts_emit_status(self):
        """Test core graph lifecycle calls in expressions stay explicit."""
        code = """
        hipError_t acceptStatus(hipError_t status) {
            return status;
        }

        hipError_t graphLifecycleExpressions(
            hipGraph_t source,
            hipGraphExec_t exec,
            hipGraphNode_t node,
            hipGraphNode_t* deps,
            size_t* count,
            bool cloneFirst
        ) {
            hipGraph_t graph;
            hipGraph_t clone;
            hipGraphNode_t cloneNode;
            hipGraphNode_t* nodes;
            hipGraphNode_t* roots;
            hipGraphNode_t* fromNodes;
            hipGraphNode_t* toNodes;
            hipGraphNodeType nodeType;
            hipGraphNode_t errorNode;
            char log[64];
            hipGraphExecUpdateResult updateResult;

            bool created = hipGraphCreate(&graph, 0) == hipSuccess;
            bool cloned = hipGraphClone(&clone, source) == hipSuccess;
            if (hipGraphGetNodes(graph, nodes, count) != hipSuccess) {
                return hipGraphDestroy(graph);
            }
            if (hipGraphGetRootNodes(graph, roots, count) == hipSuccess) {
                hipError_t selected = cloneFirst ? hipGraphNodeFindInClone(&cloneNode, node, clone) : hipGraphNodeGetType(node, &nodeType);
                return selected;
            }
            if (hipGraphGetEdges(graph, fromNodes, toNodes, count) != hipSuccess) {
                return hipGraphExecDestroy(exec);
            }
            hipError_t dependencyStatus = cloneFirst ? hipGraphNodeGetDependencies(node, deps, count) : hipGraphNodeGetDependentNodes(node, deps, count);
            if (hipGraphInstantiate(&exec, graph, &errorNode, log, 64) != hipSuccess) {
                return hipGraphInstantiateWithFlags(&exec, graph, 0);
            }
            if (hipGraphExecUpdate(exec, graph, &errorNode, &updateResult) == hipSuccess) {
                return dependencyStatus;
            }
            return acceptStatus(cloneFirst ? hipGraphDestroyNode(node) : hipGraphDestroy(clone));
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "var created: bool = ((/* HIP graph create: output: graph, "
            "flags: 0 */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var cloned: bool = ((/* HIP graph clone: output: clone, "
            "source: source */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP graph get nodes: graph: graph, nodes output: nodes, "
            "count output: count */ hipSuccess) != hipSuccess))"
        ) in result
        assert "return (/* HIP graph destroy: graph */ hipSuccess);" in result
        assert (
            "if (((/* HIP graph get root nodes: graph: graph, nodes output: "
            "roots, count output: count */ hipSuccess) == hipSuccess))"
        ) in result
        assert (
            "var selected: hipError_t = (cloneFirst ? (/* HIP graph node find "
            "in clone: output: cloneNode, original: node, clone graph: clone */ "
            "hipSuccess) : (/* HIP graph node get type: node: node, output: "
            "nodeType */ hipSuccess));"
        ) in result
        assert (
            "if (((/* HIP graph get edges: graph: graph, from output: fromNodes, "
            "to output: toNodes, count output: count */ hipSuccess) != "
            "hipSuccess))"
        ) in result
        assert "return (/* HIP graph exec destroy: exec */ hipSuccess);" in result
        assert (
            "var dependencyStatus: hipError_t = (cloneFirst ? (/* HIP graph "
            "node get dependencies: node: node, nodes output: deps, count "
            "output: count */ hipSuccess) : (/* HIP graph node get dependent "
            "nodes: node: node, nodes output: deps, count output: count */ "
            "hipSuccess));"
        ) in result
        assert (
            "if (((/* HIP graph instantiate: output: exec, graph: graph, "
            "error node output: errorNode, log buffer: log, log bytes: 64 */ "
            "hipSuccess) != hipSuccess))"
        ) in result
        assert (
            "return (/* HIP graph instantiate with flags: output: exec, "
            "graph: graph, flags: 0 */ hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP graph exec update: exec: exec, graph: graph, "
            "error node output: errorNode, result output: updateResult */ "
            "hipSuccess) == hipSuccess))"
        ) in result
        assert "return dependencyStatus;" in result
        assert (
            "return acceptStatus((cloneFirst ? (/* HIP graph destroy node: "
            "node */ hipSuccess) : (/* HIP graph destroy: clone */ hipSuccess)));"
        ) in result
        for function_name in [
            "hipGraphCreate",
            "hipGraphClone",
            "hipGraphGetNodes",
            "hipGraphDestroy",
            "hipGraphGetRootNodes",
            "hipGraphNodeFindInClone",
            "hipGraphNodeGetType",
            "hipGraphGetEdges",
            "hipGraphExecDestroy",
            "hipGraphNodeGetDependencies",
            "hipGraphNodeGetDependentNodes",
            "hipGraphInstantiate",
            "hipGraphInstantiateWithFlags",
            "hipGraphExecUpdate",
            "hipGraphDestroyNode",
        ]:
            assert f"{function_name}(" not in result

    def test_hip_user_object_expression_contexts_emit_status(self):
        """Test HIP user object calls in expressions emit status metadata."""
        code = """
        hipError_t acceptStatus(hipError_t status) {
            return status;
        }

        hipError_t userObjectExpressions(
            hipGraph_t graph,
            hipUserObject_t userObject,
            hipHostFn_t destructor,
            void* resource,
            bool retain
        ) {
            bool created =
                hipUserObjectCreate(
                    &userObject,
                    resource,
                    destructor,
                    1,
                    0
                ) == hipSuccess;
            if (hipUserObjectRetain(userObject, 2) != hipSuccess) {
                return hipUserObjectRelease(userObject, 1);
            }
            hipError_t selected =
                retain ? hipGraphRetainUserObject(graph, userObject, 1, 0) : hipGraphReleaseUserObject(graph, userObject, 1);
            return acceptStatus(
                retain ? hipUserObjectRetain(userObject, 1) : hipUserObjectRelease(userObject, 1)
            );
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "var created: bool = ((/* HIP user object create: output: "
            "userObject, resource: resource, destructor: destructor, initial "
            "refcount: 1, flags: 0 */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP user object retain: object: userObject, count: 2 */ "
            "hipSuccess) != hipSuccess))"
        ) in result
        assert (
            "return (/* HIP user object release: object: userObject, count: 1 */ "
            "hipSuccess);"
        ) in result
        assert (
            "var selected: hipError_t = (retain ? (/* HIP graph retain user "
            "object: graph: graph, object: userObject, count: 1, flags: 0 */ "
            "hipSuccess) : (/* HIP graph release user object: graph: graph, "
            "object: userObject, count: 1 */ hipSuccess));"
        ) in result
        assert (
            "return acceptStatus((retain ? (/* HIP user object retain: object: "
            "userObject, count: 1 */ hipSuccess) : (/* HIP user object release: "
            "object: userObject, count: 1 */ hipSuccess)));"
        ) in result
        for function_name in [
            "hipUserObjectCreate",
            "hipUserObjectRetain",
            "hipUserObjectRelease",
            "hipGraphRetainUserObject",
            "hipGraphReleaseUserObject",
        ]:
            assert f"{function_name}(" not in result

    def test_hip_graph_memory_memcpy_expression_contexts_emit_status(self):
        """Test HIP graph memory/memcpy calls in expressions emit status."""
        code = """
        hipError_t acceptStatus(hipError_t status) {
            return status;
        }

        hipError_t graphMemoryMemcpyExpressions(
            hipGraph_t graph,
            hipGraphExec_t exec,
            hipGraphNode_t* deps,
            size_t numDeps,
            void* dst,
            void* src,
            void* symbol,
            void* devicePtr,
            bool updateExec
        ) {
            hipGraphNode_t allocNode;
            hipGraphNode_t freeNode;
            hipGraphNode_t copyNode;
            hipGraphNode_t fromSymbolNode;
            hipGraphNode_t toSymbolNode;
            hipMemAllocNodeParams allocParams;
            size_t bytes;
            bool allocAdded =
                hipGraphAddMemAllocNode(
                    &allocNode,
                    graph,
                    deps,
                    numDeps,
                    &allocParams
                ) == hipSuccess;
            bool copyAdded =
                hipGraphAddMemcpyNode1D(
                    &copyNode,
                    graph,
                    deps,
                    numDeps,
                    dst,
                    src,
                    bytes,
                    hipMemcpyDeviceToDevice
                ) == hipSuccess;
            if (hipGraphMemAllocNodeGetParams(allocNode, &allocParams) != hipSuccess) {
                return hipGraphAddMemFreeNode(&freeNode, graph, deps, numDeps, devicePtr);
            }
            if (hipGraphMemFreeNodeGetParams(freeNode, &devicePtr) == hipSuccess) {
                hipError_t selectedCopy =
                    updateExec ? hipGraphExecMemcpyNodeSetParams1D(exec, copyNode, dst, src, bytes, hipMemcpyDeviceToDevice) : hipGraphMemcpyNodeSetParams1D(copyNode, dst, src, bytes, hipMemcpyDeviceToDevice);
                return selectedCopy;
            }
            if (hipGraphAddMemcpyNodeFromSymbol(
                &fromSymbolNode,
                graph,
                deps,
                numDeps,
                dst,
                symbol,
                bytes,
                4,
                hipMemcpyDeviceToDevice
            ) == hipSuccess) {
                hipError_t selectedSymbol =
                    updateExec ? hipGraphExecMemcpyNodeSetParamsFromSymbol(exec, fromSymbolNode, dst, symbol, bytes, 4, hipMemcpyDeviceToDevice) : hipGraphMemcpyNodeSetParamsFromSymbol(fromSymbolNode, dst, symbol, bytes, 4, hipMemcpyDeviceToDevice);
                return selectedSymbol;
            }
            return acceptStatus(
                updateExec ? hipGraphExecMemcpyNodeSetParamsToSymbol(exec, toSymbolNode, symbol, src, bytes, 8, hipMemcpyDeviceToDevice) : hipGraphAddMemcpyNodeToSymbol(&toSymbolNode, graph, deps, numDeps, symbol, src, bytes, 8, hipMemcpyDeviceToDevice)
            );
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "var allocAdded: bool = ((/* HIP graph add memory alloc node: "
            "output: allocNode, graph: graph, dependencies: deps, count: "
            "numDeps, params: (&allocParams) */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var copyAdded: bool = ((/* HIP graph add memcpy 1D node: output: "
            "copyNode, graph: graph, dependencies: deps, count: numDeps, "
            "destination: dst, source: src, bytes: bytes, kind: "
            "hipMemcpyDeviceToDevice */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP graph memory alloc node get params: node: allocNode, "
            "params output: allocParams */ hipSuccess) != hipSuccess))"
        ) in result
        assert (
            "return (/* HIP graph add memory free node: output: freeNode, "
            "graph: graph, dependencies: deps, count: numDeps, pointer: "
            "devicePtr */ hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP graph memory free node get params: node: freeNode, "
            "pointer output: devicePtr */ hipSuccess) == hipSuccess))"
        ) in result
        assert (
            "var selectedCopy: hipError_t = (updateExec ? (/* HIP graph exec "
            "memcpy 1D node set params: exec: exec, node: copyNode, "
            "destination: dst, source: src, bytes: bytes, kind: "
            "hipMemcpyDeviceToDevice */ hipSuccess) : (/* HIP graph memcpy 1D "
            "node set params: node: copyNode, destination: dst, source: src, "
            "bytes: bytes, kind: hipMemcpyDeviceToDevice */ hipSuccess));"
        ) in result
        assert (
            "if (((/* HIP graph add memcpy from symbol node: output: "
            "fromSymbolNode, graph: graph, dependencies: deps, count: numDeps, "
            "destination: dst, source: symbol, bytes: bytes, offset: 4, kind: "
            "hipMemcpyDeviceToDevice */ hipSuccess) == hipSuccess))"
        ) in result
        assert (
            "var selectedSymbol: hipError_t = (updateExec ? (/* HIP graph exec "
            "memcpy from symbol node set params: exec: exec, node: "
            "fromSymbolNode, destination: dst, source: symbol, bytes: bytes, "
            "offset: 4, kind: hipMemcpyDeviceToDevice */ hipSuccess) : (/* HIP "
            "graph memcpy from symbol node set params: node: fromSymbolNode, "
            "destination: dst, source: symbol, bytes: bytes, offset: 4, kind: "
            "hipMemcpyDeviceToDevice */ hipSuccess));"
        ) in result
        assert (
            "return acceptStatus((updateExec ? (/* HIP graph exec memcpy to "
            "symbol node set params: exec: exec, node: toSymbolNode, "
            "destination: symbol, source: src, bytes: bytes, offset: 8, kind: "
            "hipMemcpyDeviceToDevice */ hipSuccess) : (/* HIP graph add memcpy "
            "to symbol node: output: toSymbolNode, graph: graph, dependencies: "
            "deps, count: numDeps, destination: symbol, source: src, bytes: "
            "bytes, offset: 8, kind: hipMemcpyDeviceToDevice */ hipSuccess)));"
        ) in result
        for function_name in [
            "hipGraphAddMemAllocNode",
            "hipGraphMemAllocNodeGetParams",
            "hipGraphAddMemFreeNode",
            "hipGraphMemFreeNodeGetParams",
            "hipGraphAddMemcpyNode1D",
            "hipGraphMemcpyNodeSetParams1D",
            "hipGraphExecMemcpyNodeSetParams1D",
            "hipGraphAddMemcpyNodeFromSymbol",
            "hipGraphMemcpyNodeSetParamsFromSymbol",
            "hipGraphExecMemcpyNodeSetParamsFromSymbol",
            "hipGraphAddMemcpyNodeToSymbol",
            "hipGraphExecMemcpyNodeSetParamsToSymbol",
        ]:
            assert f"{function_name}(" not in result

    def test_hip_driver_graph_memcpy_node_emits_metadata_and_status(self):
        """Test HIP driver graph memcpy node calls emit metadata/status."""
        code = """
        hipError_t acceptStatus(hipError_t status) {
            return status;
        }

        hipError_t driverGraphMemcpyNode(
            hipGraph_t graph,
            hipGraphNode_t* deps,
            size_t numDeps,
            HIP_MEMCPY3D* params,
            hipCtx_t ctx,
            bool retry
        ) {
            hipGraphNode_t node;
            hipDrvGraphAddMemcpyNode(&node, graph, deps, numDeps, params, ctx);
            hipError_t err = hipDrvGraphAddMemcpyNode(&node, graph, deps, numDeps, params, ctx);
            if (hipDrvGraphAddMemcpyNode(&node, graph, deps, numDeps, params, ctx) != hipSuccess) {
                return hipDrvGraphAddMemcpyNode(&node, graph, deps, numDeps, params, ctx);
            }
            hipError_t selected = retry ? hipDrvGraphAddMemcpyNode(&node, graph, deps, numDeps, params, ctx) : hipSuccess;
            return acceptStatus(hipDrvGraphAddMemcpyNode(&node, graph, deps, numDeps, params, ctx));
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        metadata = (
            "HIP driver graph add memcpy node: output: node, graph: graph, "
            "dependencies: deps, count: numDeps, params: params, context: ctx"
        )
        assert result.count(f"// {metadata}") == 2
        assert "var err: hipError_t = hipSuccess;" in result
        assert f"if (((/* {metadata} */ hipSuccess) != hipSuccess))" in result
        assert f"return (/* {metadata} */ hipSuccess);" in result
        assert (
            f"var selected: hipError_t = (retry ? (/* {metadata} */ hipSuccess) "
            ": hipSuccess);"
        ) in result
        assert f"return acceptStatus((/* {metadata} */ hipSuccess));" in result
        assert "hipDrvGraphAddMemcpyNode(" not in result

    def test_hip_driver_graph_memory_node_apis_emit_metadata_and_status(self):
        """Test HIP driver graph memory node calls emit metadata/status."""
        code = """
        hipError_t acceptStatus(hipError_t status) {
            return status;
        }

        hipError_t driverGraphMemoryNodes(
            hipGraph_t graph,
            hipGraphExec_t exec,
            hipGraphNode_t* deps,
            size_t numDeps,
            HIP_MEMCPY3D* copyParams,
            hipMemsetParams* memsetParams,
            hipCtx_t ctx,
            hipDeviceptr_t ptr,
            bool updateExec
        ) {
            hipGraphNode_t copyNode;
            hipGraphNode_t memsetNode;
            hipGraphNode_t freeNode;
            hipDrvGraphAddMemsetNode(&memsetNode, graph, deps, numDeps, memsetParams, ctx);
            hipError_t err = hipDrvGraphAddMemFreeNode(&freeNode, graph, deps, numDeps, ptr);
            if (hipDrvGraphMemcpyNodeGetParams(copyNode, copyParams) != hipSuccess) {
                return hipDrvGraphMemcpyNodeSetParams(copyNode, copyParams);
            }
            hipError_t selected = updateExec ? hipDrvGraphExecMemcpyNodeSetParams(exec, copyNode, copyParams, ctx) : hipDrvGraphExecMemsetNodeSetParams(exec, memsetNode, memsetParams, ctx);
            return acceptStatus(hipDrvGraphAddMemsetNode(&memsetNode, graph, deps, numDeps, memsetParams, ctx));
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        add_memset = (
            "HIP driver graph add memset node: output: memsetNode, graph: graph, "
            "dependencies: deps, count: numDeps, params: memsetParams, context: ctx"
        )
        add_free = (
            "HIP driver graph add memory free node: output: freeNode, graph: graph, "
            "dependencies: deps, count: numDeps, pointer: ptr"
        )
        get_params = (
            "HIP driver graph memcpy node get params: node: copyNode, "
            "params output: copyParams"
        )
        set_params = (
            "HIP driver graph memcpy node set params: node: copyNode, "
            "params: copyParams"
        )
        exec_copy = (
            "HIP driver graph exec memcpy node set params: exec: exec, "
            "node: copyNode, params: copyParams, context: ctx"
        )
        exec_memset = (
            "HIP driver graph exec memset node set params: exec: exec, "
            "node: memsetNode, params: memsetParams, context: ctx"
        )

        assert f"// {add_memset}" in result
        assert f"// {add_free}" in result
        assert "var err: hipError_t = hipSuccess;" in result
        assert f"if (((/* {get_params} */ hipSuccess) != hipSuccess))" in result
        assert f"return (/* {set_params} */ hipSuccess);" in result
        assert (
            f"var selected: hipError_t = (updateExec ? (/* {exec_copy} */ "
            f"hipSuccess) : (/* {exec_memset} */ hipSuccess));"
        ) in result
        assert f"return acceptStatus((/* {add_memset} */ hipSuccess));" in result
        for function_name in [
            "hipDrvGraphAddMemsetNode",
            "hipDrvGraphAddMemFreeNode",
            "hipDrvGraphMemcpyNodeGetParams",
            "hipDrvGraphMemcpyNodeSetParams",
            "hipDrvGraphExecMemcpyNodeSetParams",
            "hipDrvGraphExecMemsetNodeSetParams",
        ]:
            assert f"{function_name}(" not in result

    def test_hip_graph_external_semaphore_expression_contexts_emit_status(self):
        """Test HIP graph external semaphore calls in expressions emit status."""
        code = """
        hipError_t acceptStatus(hipError_t status) {
            return status;
        }

        hipError_t graphSemaphoreExpressions(
            hipGraph_t graph,
            hipGraphExec_t exec,
            hipGraphNode_t* deps,
            size_t numDeps,
            bool signalFirst
        ) {
            hipGraphNode_t signalNode;
            hipGraphNode_t waitNode;
            hipExternalSemaphoreSignalNodeParams signalParams;
            hipExternalSemaphoreWaitNodeParams waitParams;
            bool signalAdded =
                hipGraphAddExternalSemaphoresSignalNode(
                    &signalNode,
                    graph,
                    deps,
                    numDeps,
                    &signalParams
                ) == hipSuccess;
            if (hipGraphExternalSemaphoresSignalNodeGetParams(signalNode, &signalParams) != hipSuccess) {
                return hipGraphExternalSemaphoresSignalNodeSetParams(signalNode, &signalParams);
            }
            hipError_t selected =
                signalFirst ? hipGraphExecExternalSemaphoresSignalNodeSetParams(exec, signalNode, &signalParams) : hipGraphAddExternalSemaphoresWaitNode(&waitNode, graph, deps, numDeps, &waitParams);
            if (hipGraphExternalSemaphoresWaitNodeGetParams(waitNode, &waitParams) == hipSuccess) {
                return selected;
            }
            return acceptStatus(
                signalFirst ? hipGraphExecExternalSemaphoresWaitNodeSetParams(exec, waitNode, &waitParams) : hipGraphExternalSemaphoresWaitNodeSetParams(waitNode, &waitParams)
            );
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "var signalAdded: bool = ((/* HIP graph add external semaphore "
            "signal node: output: signalNode, graph: graph, dependencies: "
            "deps, count: numDeps, params: (&signalParams) */ hipSuccess) == "
            "hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP graph external semaphore signal node get params: "
            "node: signalNode, params: (&signalParams) */ hipSuccess) != "
            "hipSuccess))"
        ) in result
        assert (
            "return (/* HIP graph external semaphore signal node set params: "
            "node: signalNode, params: (&signalParams) */ hipSuccess);"
        ) in result
        assert (
            "var selected: hipError_t = (signalFirst ? (/* HIP graph exec "
            "external semaphore signal node set params: exec: exec, node: "
            "signalNode, params: (&signalParams) */ hipSuccess) : (/* HIP "
            "graph add external semaphore wait node: output: waitNode, graph: "
            "graph, dependencies: deps, count: numDeps, params: (&waitParams) "
            "*/ hipSuccess));"
        ) in result
        assert (
            "if (((/* HIP graph external semaphore wait node get params: "
            "node: waitNode, params: (&waitParams) */ hipSuccess) == "
            "hipSuccess))"
        ) in result
        assert "return selected;" in result
        assert (
            "return acceptStatus((signalFirst ? (/* HIP graph exec external "
            "semaphore wait node set params: exec: exec, node: waitNode, "
            "params: (&waitParams) */ hipSuccess) : (/* HIP graph external "
            "semaphore wait node set params: node: waitNode, params: "
            "(&waitParams) */ hipSuccess)));"
        ) in result
        for function_name in [
            "hipGraphAddExternalSemaphoresSignalNode",
            "hipGraphExternalSemaphoresSignalNodeGetParams",
            "hipGraphExternalSemaphoresSignalNodeSetParams",
            "hipGraphExecExternalSemaphoresSignalNodeSetParams",
            "hipGraphAddExternalSemaphoresWaitNode",
            "hipGraphExternalSemaphoresWaitNodeGetParams",
            "hipGraphExternalSemaphoresWaitNodeSetParams",
            "hipGraphExecExternalSemaphoresWaitNodeSetParams",
        ]:
            assert f"{function_name}(" not in result

    def test_hip_extended_graph_expression_contexts_emit_status(self):
        """Test extended HIP graph calls in expressions emit status metadata."""
        code = """
        hipError_t acceptStatus(hipError_t status) {
            return status;
        }

        hipError_t extendedGraphExpressions(
            hipGraph_t graph,
            hipGraphExec_t exec,
            hipGraphNode_t genericNode,
            hipGraphNode_t kernelNode,
            hipGraphNode_t childNode,
            bool update
        ) {
            hipGraph_t embeddedGraph;
            hipGraphNodeParams nodeParams;
            hipGraphInstantiateParams instantiateParams;
            hipKernelNodeAttrValue attrValue;
            unsigned long long execFlags = 0ull;
            size_t bytes = 0;
            int device = 0;
            bool paramsSet =
                hipGraphNodeSetParams(genericNode, &nodeParams) == hipSuccess;
            if (hipGraphExecNodeSetParams(exec, genericNode, &nodeParams) != hipSuccess) {
                return hipGraphInstantiateWithParams(&exec, graph, &instantiateParams);
            }
            hipError_t selected =
                update ? hipGraphExecGetFlags(exec, &execFlags) : hipDeviceGetGraphMemAttribute(device, hipGraphMemAttrUsedMemCurrent, &bytes);
            if (hipDeviceSetGraphMemAttribute(device, hipGraphMemAttrReserveMemCurrent, &bytes) == hipSuccess) {
                return selected;
            }
            if (hipGraphChildGraphNodeGetGraph(childNode, &embeddedGraph) != hipSuccess) {
                return hipGraphKernelNodeSetAttribute(kernelNode, hipKernelNodeAttributeCooperative, &attrValue);
            }
            hipError_t trimOrPrint =
                update ? hipDeviceGraphMemTrim(device) : hipGraphDebugDotPrint(graph, "graph.dot", 0);
            return acceptStatus(
                update ? hipGraphKernelNodeGetAttribute(kernelNode, hipKernelNodeAttributeCooperative, &attrValue) : trimOrPrint
            );
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "var paramsSet: bool = ((/* HIP graph generic node set params: "
            "node: genericNode, params: (&nodeParams) */ hipSuccess) == "
            "hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP graph exec generic node set params: exec: exec, "
            "node: genericNode, params: (&nodeParams) */ hipSuccess) != "
            "hipSuccess))"
        ) in result
        assert (
            "return (/* HIP graph instantiate with params: output: exec, "
            "graph: graph, params: (&instantiateParams) */ hipSuccess);"
        ) in result
        assert (
            "var selected: hipError_t = (update ? (/* HIP graph exec get flags: "
            "exec: exec, output: execFlags */ hipSuccess) : (/* HIP device "
            "graph memory get attribute: device: device, attribute: "
            "hipGraphMemAttrUsedMemCurrent, output: bytes */ hipSuccess));"
        ) in result
        assert (
            "if (((/* HIP device graph memory set attribute: device: device, "
            "attribute: hipGraphMemAttrReserveMemCurrent, value: bytes */ "
            "hipSuccess) == hipSuccess))"
        ) in result
        assert "return selected;" in result
        assert (
            "if (((/* HIP graph child node get graph: node: childNode, output: "
            "embeddedGraph */ hipSuccess) != hipSuccess))"
        ) in result
        assert (
            "return (/* HIP graph kernel node set attribute: node: kernelNode, "
            "attribute: hipKernelNodeAttributeCooperative, value: attrValue */ "
            "hipSuccess);"
        ) in result
        assert (
            "var trimOrPrint: hipError_t = (update ? (/* HIP device graph "
            "memory trim: device: device */ hipSuccess) : (/* HIP graph debug "
            'dot print: graph: graph, path: "graph.dot", flags: 0 */ '
            "hipSuccess));"
        ) in result
        assert (
            "return acceptStatus((update ? (/* HIP graph kernel node get "
            "attribute: node: kernelNode, attribute: "
            "hipKernelNodeAttributeCooperative, output: attrValue */ "
            "hipSuccess) : trimOrPrint));"
        ) in result
        for function_name in [
            "hipGraphNodeSetParams",
            "hipGraphExecNodeSetParams",
            "hipGraphInstantiateWithParams",
            "hipGraphExecGetFlags",
            "hipDeviceGetGraphMemAttribute",
            "hipDeviceSetGraphMemAttribute",
            "hipGraphChildGraphNodeGetGraph",
            "hipGraphKernelNodeSetAttribute",
            "hipDeviceGraphMemTrim",
            "hipGraphDebugDotPrint",
            "hipGraphKernelNodeGetAttribute",
        ]:
            assert f"{function_name}(" not in result

    def test_hip_interop_ipc_launch_expression_contexts_emit_status(self):
        """Test interop, IPC, profiler, and launch support expressions."""
        code = """
        hipError_t interopIpcLaunchExpressions(
            hipStream_t stream,
            hipGraphicsResource_t* resources,
            hipGraphicsResource_t imageResource,
            hipIpcMemHandle_t ipcMemHandle,
            hipIpcEventHandle_t ipcEventHandle,
            hipLaunchConfig_t* config,
            HIP_LAUNCH_CONFIG* driverConfig,
            hipFunction_t function,
            void* kernel,
            void** args,
            hipLaunchParams* launchParams,
            bool retry
        ) {
            hipGraphicsResource_t resource;
            hipArray_t array;
            hipEvent_t event;
            hipEvent_t startEvent;
            hipEvent_t stopEvent;
            void* mappedPointer;
            size_t ptrSize;
            dim3 grid;
            dim3 block;
            dim3 outGrid;
            dim3 outBlock;
            size_t sharedMem;
            hipStream_t outStream;
            int value;
            unsigned int glBuffer;
            unsigned int glImage;
            unsigned int glTarget;
            hipIpcMemHandle_t exportedHandle;
            hipIpcEventHandle_t exportedEventHandle;
            bool registered =
                hipGraphicsGLRegisterBuffer(
                    &resource,
                    glBuffer,
                    hipGraphicsRegisterFlagsWriteDiscard
                ) == hipSuccess;
            bool registeredImage =
                hipGraphicsGLRegisterImage(
                    &imageResource,
                    glImage,
                    glTarget,
                    hipGraphicsRegisterFlagsSurfaceLoadStore
                ) == hipSuccess;
            bool mapped =
                hipGraphicsMapResources(1, resources, stream) == hipSuccess;
            bool unregistered =
                hipGraphicsUnregisterResource(resource) == hipSuccess;
            bool exported =
                hipIpcGetMemHandle(&exportedHandle, mappedPointer) == hipSuccess;
            bool opened =
                hipIpcOpenMemHandle(
                    &mappedPointer,
                    ipcMemHandle,
                    hipIpcMemLazyEnablePeerAccess
                ) == hipSuccess;
            bool closed = hipIpcCloseMemHandle(mappedPointer) == hipSuccess;
            bool eventExported =
                hipIpcGetEventHandle(&exportedEventHandle, event) == hipSuccess;
            bool configured =
                hipConfigureCall(grid, block, 0, stream) == hipSuccess;
            bool pushed =
                __hipPushCallConfiguration(grid, block, 0, stream) == hipSuccess;
            bool profiled = hipProfilerStart() == hipSuccess;
            bool stopped = hipProfilerStop() == hipSuccess;
            bool driverLaunched =
                hipDrvLaunchKernelEx(driverConfig, function, args, NULL) == hipSuccess;
            bool extendedLaunched =
                hipExtLaunchKernel(
                    (const void*)kernel,
                    grid,
                    block,
                    args,
                    0,
                    stream,
                    startEvent,
                    stopEvent,
                    hipExtAnyOrderLaunch
                ) == hipSuccess;
            bool gglLaunched =
                hipExtLaunchKernelGGL(
                    kernel,
                    grid,
                    block,
                    0,
                    stream,
                    startEvent,
                    stopEvent,
                    hipExtAnyOrderLaunch,
                    kernel,
                    args
                ) == hipSuccess;
            bool multiLaunched =
                hipExtLaunchMultiKernelMultiDevice(
                    launchParams,
                    2,
                    hipExtAnyOrderLaunch
                ) == hipSuccess;
            if (hipGraphicsResourceGetMappedPointer(&mappedPointer, &ptrSize, resource) != hipSuccess) {
                return hipIpcOpenEventHandle(&event, ipcEventHandle);
            }
            if (__hipPopCallConfiguration(&outGrid, &outBlock, &sharedMem, &outStream) == hipSuccess) {
                hipError_t selected = retry ? hipGraphicsUnmapResources(1, resources, stream) : hipSetupArgument(&value, sizeof(value), 0);
                return selected;
            }
            if (hipLaunchKernelExC(config, kernel, args) == hipSuccess) {
                return hipLaunchByPtr(kernel);
            }
            if (hipExtModuleLaunchKernel(function, 512, 1, 1, 64, 1, 1, 0, stream, args, NULL, startEvent, stopEvent, hipExtAnyOrderLaunch) == hipSuccess) {
                return hipHccModuleLaunchKernel(function, 512, 1, 1, 64, 1, 1, 0, stream, args, NULL, startEvent, stopEvent, hipExtAnyOrderLaunch);
            }
            if (hipLaunchCooperativeKernel(function, grid, block, args, 0, stream) == hipSuccess) {
                return hipLaunchCooperativeKernelMultiDevice(launchParams, 2, 0);
            }
            if (hipModuleLaunchCooperativeKernel(function, 8, 1, 1, 32, 1, 1, 0, stream, args) == hipSuccess) {
                return hipModuleLaunchCooperativeKernelMultiDevice(launchParams, 2, 0);
            }
            return hipGraphicsSubResourceGetMappedArray(
                &array,
                imageResource,
                0,
                0
            );
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "var registered: bool = ((/* HIP OpenGL register buffer: output: "
            "resource, buffer: glBuffer, flags: hipGraphicsRegisterFlagsWriteDiscard "
            "*/ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var registeredImage: bool = ((/* HIP OpenGL register image: "
            "output: imageResource, image: glImage, target: glTarget, "
            "flags: hipGraphicsRegisterFlagsSurfaceLoadStore */ hipSuccess) "
            "== hipSuccess);"
        ) in result
        assert (
            "var mapped: bool = ((/* HIP graphics map resources: count: 1, "
            "resources: resources, stream: stream */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var unregistered: bool = ((/* HIP graphics unregister resource: "
            "resource */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var exported: bool = ((/* HIP IPC get memory handle: output: "
            "exportedHandle, pointer: mappedPointer */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var opened: bool = ((/* HIP IPC open memory handle: output: "
            "mappedPointer, handle: ipcMemHandle, flags: "
            "hipIpcMemLazyEnablePeerAccess */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var closed: bool = ((/* HIP IPC close memory handle: pointer: "
            "mappedPointer */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var eventExported: bool = ((/* HIP IPC get event handle: output: "
            "exportedEventHandle, event: event */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var configured: bool = ((/* HIP configure call: grid: grid, block: "
            "block, shared memory: 0, stream: stream */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var pushed: bool = ((/* HIP push call configuration: grid: grid, "
            "block: block, shared memory: 0, stream: stream */ hipSuccess) "
            "== hipSuccess);"
        ) in result
        assert (
            "var profiled: bool = ((/* HIP profiler start */ hipSuccess) == "
            "hipSuccess);"
        ) in result
        assert (
            "var stopped: bool = ((/* HIP profiler stop */ hipSuccess) == "
            "hipSuccess);"
        ) in result
        assert (
            "var driverLaunched: bool = ((/* HIP driver launch kernel ex: "
            "config: driverConfig, function: function, params: args, extra: "
            "NULL */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var extendedLaunched: bool = ((/* HIP extended kernel launch: "
            "function: kernel, grid: grid, block: block, args: args, shared "
            "memory: 0, stream: stream, start event: startEvent, stop event: "
            "stopEvent, flags: hipExtAnyOrderLaunch */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var gglLaunched: bool = ((/* HIP extended kernel launch GGL: "
            "function: kernel, grid: grid, block: block, shared memory: 0, "
            "stream: stream, start event: startEvent, stop event: stopEvent, "
            "flags: hipExtAnyOrderLaunch, args: kernel, args */ hipSuccess) "
            "== hipSuccess);"
        ) in result
        assert (
            "var multiLaunched: bool = ((/* HIP extended multi-kernel "
            "multi-device launch: params: launchParams, devices: 2, flags: "
            "hipExtAnyOrderLaunch */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP graphics mapped pointer: pointer output: mappedPointer, "
            "size output: ptrSize, resource: resource */ hipSuccess) != hipSuccess))"
        ) in result
        assert (
            "return (/* HIP IPC open event handle: output: event, "
            "handle: ipcEventHandle */ hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP pop call configuration: grid output: outGrid, "
            "block output: outBlock, shared memory output: sharedMem, "
            "stream output: outStream */ hipSuccess) == hipSuccess))"
        ) in result
        assert (
            "var selected: hipError_t = (retry ? "
            "(/* HIP graphics unmap resources: count: 1, resources: resources, "
            "stream: stream */ hipSuccess) : "
            "(/* HIP setup kernel argument: value: (&value), bytes: sizeof(value), "
            "offset: 0 */ hipSuccess));"
        ) in result
        assert "return selected;" in result
        assert (
            "if (((/* HIP launch kernel ex: config: config, function: kernel, "
            "args: args */ hipSuccess) == hipSuccess))"
        ) in result
        assert (
            "return (/* HIP launch by pointer: function: kernel */ hipSuccess);"
            in result
        )
        assert (
            "if (((/* HIP extended module launch kernel: function: function, "
            "global work size: (512, 1, 1), local work size: (64, 1, 1), "
            "shared memory: 0, stream: stream, params: args, extra: NULL, "
            "start event: startEvent, stop event: stopEvent, flags: "
            "hipExtAnyOrderLaunch */ hipSuccess) == hipSuccess))"
        ) in result
        assert (
            "return (/* HIP HCC module launch kernel: function: function, "
            "global work size: (512, 1, 1), local work size: (64, 1, 1), "
            "shared memory: 0, stream: stream, params: args, extra: NULL, "
            "start event: startEvent, stop event: stopEvent, flags: "
            "hipExtAnyOrderLaunch */ hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP cooperative kernel launch: function: function, "
            "grid: grid, block: block, params: args, shared memory: 0, "
            "stream: stream */ hipSuccess) == hipSuccess))"
        ) in result
        assert (
            "return (/* HIP cooperative multi-device launch: params: "
            "launchParams, devices: 2, flags: 0 */ hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP module cooperative kernel launch: function: function, "
            "grid: (8, 1, 1), block: (32, 1, 1), shared memory: 0, "
            "stream: stream, params: args */ hipSuccess) == hipSuccess))"
        ) in result
        assert (
            "return (/* HIP module cooperative multi-device launch: params: "
            "launchParams, devices: 2, flags: 0 */ hipSuccess);"
        ) in result
        assert (
            "return (/* HIP graphics mapped subresource array: output: array, "
            "resource: imageResource, array index: 0, mip level: 0 */ hipSuccess);"
        ) in result
        for function_name in [
            "hipGraphicsGLRegisterBuffer",
            "hipGraphicsGLRegisterImage",
            "hipGraphicsMapResources",
            "hipGraphicsUnregisterResource",
            "hipIpcGetMemHandle",
            "hipIpcOpenMemHandle",
            "hipIpcCloseMemHandle",
            "hipIpcGetEventHandle",
            "hipConfigureCall",
            "__hipPushCallConfiguration",
            "hipProfilerStart",
            "hipProfilerStop",
            "hipDrvLaunchKernelEx",
            "hipExtLaunchKernel",
            "hipExtLaunchKernelGGL",
            "hipExtLaunchMultiKernelMultiDevice",
            "hipGraphicsResourceGetMappedPointer",
            "hipIpcOpenEventHandle",
            "__hipPopCallConfiguration",
            "hipGraphicsUnmapResources",
            "hipSetupArgument",
            "hipLaunchKernelExC",
            "hipLaunchByPtr",
            "hipExtModuleLaunchKernel",
            "hipHccModuleLaunchKernel",
            "hipLaunchCooperativeKernel",
            "hipLaunchCooperativeKernelMultiDevice",
            "hipModuleLaunchCooperativeKernel",
            "hipModuleLaunchCooperativeKernelMultiDevice",
            "hipGraphicsSubResourceGetMappedArray",
        ]:
            assert f"{function_name}(" not in result

    def test_hip_external_memory_semaphore_expression_contexts_emit_status(self):
        """Test external memory and semaphore expressions emit status metadata."""
        code = """
        hipError_t externalInteropExpressions(
            hipStream_t stream,
            hipExternalMemory_t memory,
            hipExternalSemaphore_t semaphore,
            hipExternalMemoryHandleDesc* memoryDesc,
            hipExternalMemoryBufferDesc* bufferDesc,
            hipExternalMemoryMipmappedArrayDesc* mipmapDesc,
            hipExternalSemaphoreHandleDesc* semaphoreDesc,
            hipExternalSemaphoreSignalParams* signalParams,
            hipExternalSemaphoreWaitParams* waitParams,
            bool release
        ) {
            hipExternalMemory_t importedMemory;
            hipExternalSemaphore_t importedSemaphore;
            void* mappedPointer;
            hipMipmappedArray_t mipmappedArray;
            bool imported =
                hipImportExternalMemory(&importedMemory, memoryDesc) == hipSuccess;
            bool mapped =
                hipExternalMemoryGetMappedBuffer(
                    &mappedPointer,
                    memory,
                    bufferDesc
                ) == hipSuccess;
            bool mipmapped =
                hipExternalMemoryGetMappedMipmappedArray(
                    &mipmappedArray,
                    memory,
                    mipmapDesc
                ) == hipSuccess;
            bool freed = hipFreeMipmappedArray(mipmappedArray) == hipSuccess;
            bool semaphoreImported =
                hipImportExternalSemaphore(
                    &importedSemaphore,
                    semaphoreDesc
                ) == hipSuccess;
            if (hipSignalExternalSemaphoresAsync(&semaphore, signalParams, 1, stream) != hipSuccess) {
                return hipWaitExternalSemaphoresAsync(
                    &semaphore,
                    waitParams,
                    1,
                    stream
                );
            }
            hipError_t selected = release ? hipDestroyExternalSemaphore(importedSemaphore) : hipDestroyExternalMemory(importedMemory);
            return selected;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "var imported: bool = ((/* HIP import external memory: output: "
            "importedMemory, descriptor: memoryDesc */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var mapped: bool = ((/* HIP external memory mapped buffer: output: "
            "mappedPointer, memory: memory, descriptor: bufferDesc */ hipSuccess) "
            "== hipSuccess);"
        ) in result
        assert (
            "var mipmapped: bool = ((/* HIP external memory mapped mipmapped "
            "array: output: mipmappedArray, memory: memory, descriptor: "
            "mipmapDesc */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var freed: bool = ((/* HIP free mipmapped array: mipmappedArray */ "
            "hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var semaphoreImported: bool = ((/* HIP import external semaphore: "
            "output: importedSemaphore, descriptor: semaphoreDesc */ hipSuccess) "
            "== hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP signal external semaphores: semaphores: (&semaphore), "
            "params: signalParams, count: 1, stream: stream */ hipSuccess) != "
            "hipSuccess))"
        ) in result
        assert (
            "return (/* HIP wait external semaphores: semaphores: (&semaphore), "
            "params: waitParams, count: 1, stream: stream */ hipSuccess);"
        ) in result
        assert (
            "var selected: hipError_t = (release ? "
            "(/* HIP destroy external semaphore: importedSemaphore */ hipSuccess) "
            ": (/* HIP destroy external memory: importedMemory */ hipSuccess));"
        ) in result
        assert "return selected;" in result
        for function_name in [
            "hipImportExternalMemory",
            "hipExternalMemoryGetMappedBuffer",
            "hipExternalMemoryGetMappedMipmappedArray",
            "hipFreeMipmappedArray",
            "hipImportExternalSemaphore",
            "hipSignalExternalSemaphoresAsync",
            "hipWaitExternalSemaphoresAsync",
            "hipDestroyExternalSemaphore",
            "hipDestroyExternalMemory",
        ]:
            assert f"{function_name}(" not in result

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

    def test_hip_runtime_last_error_expression_contexts_emit_status(self):
        """Test HIP last-error calls in expressions stay explicit."""
        code = """
        hipError_t acceptStatus(hipError_t status) {
            return status;
        }

        hipError_t queryLastErrorExpressions(bool retry) {
            bool lastOk = hipGetLastError() == hipSuccess;
            if (hipPeekAtLastError() != hipSuccess) {
                return hipGetLastError();
            }
            hipError_t selected = retry ? hipPeekAtLastError() : hipGetLastError();
            return acceptStatus(
                retry ? hipGetLastError() : hipPeekAtLastError()
            );
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "var lastOk: bool = ((/* HIP get last error */ hipSuccess) == "
            "hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP peek at last error */ hipSuccess) != hipSuccess))" in result
        )
        assert "return (/* HIP get last error */ hipSuccess);" in result
        assert (
            "var selected: hipError_t = (retry ? "
            "(/* HIP peek at last error */ hipSuccess) : "
            "(/* HIP get last error */ hipSuccess));"
        ) in result
        assert (
            "return acceptStatus((retry ? (/* HIP get last error */ "
            "hipSuccess) : (/* HIP peek at last error */ hipSuccess)));"
        ) in result
        assert "hipGetLastError(" not in result
        assert "hipPeekAtLastError(" not in result

    def test_hip_device_property_member_reads_emit_metadata_expressions(self):
        """Test hipDeviceProp_t member reads lower to explicit metadata."""
        code = """
        void host(int device, hipDeviceProp_t* propsPtr) {
            hipDeviceProp_t props;
            hipGetDeviceProperties(&props, device);
            int sms = props.multiProcessorCount;
            int warp = props.warpSize;
            size_t total = propsPtr->totalGlobalMem;
            props.multiProcessorCount = 7;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert "// HIP get device properties: props, device: device" in result
        assert (
            "var sms: i32 = (/* HIP device property: multiProcessorCount, "
            "device: device */ 0);"
        ) in result
        assert (
            "var warp: i32 = (/* HIP device property: warpSize, "
            "device: device */ 0);"
        ) in result
        assert (
            "var total: u32 = (/* HIP device property: totalGlobalMem */ 0);"
        ) in result
        assert "props.multiProcessorCount = 7;" in result
        assert "var sms: i32 = props.multiProcessorCount;" not in result
        assert "var warp: i32 = props.warpSize;" not in result
        assert "var total: u32 = propsPtr->totalGlobalMem;" not in result

    def test_hip_device_attribute_reads_emit_metadata_expressions(self):
        """Test hipDeviceGetAttribute output reads lower to explicit metadata."""
        code = """
        void host(int device, int* out) {
            int attr = 0;
            int other = 0;
            hipDeviceGetAttribute(
                &attr, hipDeviceAttributeMaxThreadsPerBlock, device
            );
            int maxThreads = attr;
            out[0] = attr;
            attr = 7;
            int manual = attr;
            hipDeviceGetAttribute(
                &other, hipDeviceAttributeWarpSize, device + 1
            );
            other++;
            int cleared = other;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "// HIP get device attribute: output: attr, "
            "attribute: hipDeviceAttributeMaxThreadsPerBlock, device: device"
        ) in result
        assert (
            "var maxThreads: i32 = (/* HIP device attribute: "
            "hipDeviceAttributeMaxThreadsPerBlock, device: device */ 0);"
        ) in result
        assert (
            "out[0] = (/* HIP device attribute: "
            "hipDeviceAttributeMaxThreadsPerBlock, device: device */ 0);"
        ) in result
        assert "attr = 7;" in result
        assert "var manual: i32 = attr;" in result
        assert (
            "// HIP get device attribute: output: other, "
            "attribute: hipDeviceAttributeWarpSize, device: (device + 1)"
        ) in result
        assert "(other++);" in result
        assert "var cleared: i32 = other;" in result
        assert "var maxThreads: i32 = attr;" not in result
        assert "out[0] = attr;" not in result
        assert "var manual: i32 = (/* HIP device attribute:" not in result
        assert "var cleared: i32 = (/* HIP device attribute:" not in result

    def test_hip_device_scalar_query_reads_emit_metadata_expressions(self):
        """Test scalar HIP device-query outputs lower to explicit metadata."""
        code = """
        void host(int device, size_t* out) {
            size_t total = 0;
            size_t statusTotal = 0;
            int major = 0;
            int minor = 0;
            hipDeviceTotalMem(&total, device);
            out[0] = total;
            total = 1024;
            size_t manualTotal = total;
            hipDeviceComputeCapability(&major, &minor, device + 1);
            int packed = major * 10 + minor;
            major = 9;
            int manualMajor = major;
            minor++;
            int manualMinor = minor;
            hipError_t err = hipDeviceTotalMem(&statusTotal, device + 2);
            size_t fromStatus = statusTotal;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert "// HIP get device total memory: output: total, device: device" in result
        assert (
            "out[0] = (/* HIP device query: totalMem, device: device */ 0);" in result
        )
        assert "total = 1024;" in result
        assert "var manualTotal: u32 = total;" in result
        assert (
            "// HIP get device compute capability: major output: major, "
            "minor output: minor, device: (device + 1)"
        ) in result
        assert (
            "var packed: i32 = (((/* HIP device query: "
            "computeCapability.major, device: (device + 1) */ 0) * 10) + "
            "(/* HIP device query: computeCapability.minor, "
            "device: (device + 1) */ 0));"
        ) in result
        assert "major = 9;" in result
        assert "var manualMajor: i32 = major;" in result
        assert "(minor++);" in result
        assert "var manualMinor: i32 = minor;" in result
        assert (
            "// HIP get device total memory: output: statusTotal, "
            "device: (device + 2)"
        ) in result
        assert "var err: hipError_t = hipSuccess;" in result
        assert (
            "var fromStatus: u32 = (/* HIP device query: totalMem, "
            "device: (device + 2) */ 0);"
        ) in result
        assert "out[0] = total;" not in result
        assert "var packed: i32 = ((major * 10) + minor);" not in result
        assert "var manualTotal: u32 = (/* HIP device query:" not in result
        assert "var manualMajor: i32 = (/* HIP device query:" not in result
        assert "var manualMinor: i32 = (/* HIP device query:" not in result
        assert "var fromStatus: u32 = statusTotal;" not in result

    def test_hip_device_current_and_count_reads_emit_metadata_expressions(self):
        """Test hipGetDevice and hipGetDeviceCount outputs lower to metadata."""
        code = """
        void host(int* out) {
            int current = -1;
            int count = 0;
            int statusCurrent = -1;
            int statusCount = 0;
            hipGetDevice(&current);
            out[0] = current;
            hipSetDevice(current);
            current = 3;
            int manualCurrent = current;
            hipGetDeviceCount(&count);
            int total = count;
            count++;
            int manualCount = count;
            hipError_t errDevice = hipGetDevice(&statusCurrent);
            int fromStatusCurrent = statusCurrent;
            hipError_t errCount = hipGetDeviceCount(&statusCount);
            int fromStatusCount = statusCount;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert "// HIP get current device: output: current" in result
        assert "out[0] = (/* HIP device query: currentDevice */ 0);" in result
        assert "// HIP set device: current" in result
        assert "current = 3;" in result
        assert "var manualCurrent: i32 = current;" in result
        assert "// HIP get device count: output: count" in result
        assert "var total: i32 = (/* HIP device query: deviceCount */ 0);" in result
        assert "(count++);" in result
        assert "var manualCount: i32 = count;" in result
        assert "// HIP get current device: output: statusCurrent" in result
        assert "var errDevice: hipError_t = hipSuccess;" in result
        assert (
            "var fromStatusCurrent: i32 = " "(/* HIP device query: currentDevice */ 0);"
        ) in result
        assert "// HIP get device count: output: statusCount" in result
        assert "var errCount: hipError_t = hipSuccess;" in result
        assert (
            "var fromStatusCount: i32 = " "(/* HIP device query: deviceCount */ 0);"
        ) in result
        assert "out[0] = current;" not in result
        assert "var total: i32 = count;" not in result
        assert "var manualCurrent: i32 = (/* HIP device query:" not in result
        assert "var manualCount: i32 = (/* HIP device query:" not in result
        assert "var fromStatusCurrent: i32 = statusCurrent;" not in result
        assert "var fromStatusCount: i32 = statusCount;" not in result

    def test_hip_device_selection_limit_and_flag_reads_emit_metadata_expressions(
        self,
    ):
        """Test selected-device, limit, and flag outputs lower to metadata."""
        code = """
        void host(hipDeviceProp_t* props, char* pciBusId, int* out, size_t* sizes) {
            int device = 0;
            size_t limitValue = 0;
            size_t statusLimit = 0;
            unsigned int flags = 0;
            hipChooseDevice(&device, props);
            out[0] = device;
            device = 4;
            int manualDevice = device;
            hipError_t errDevice = hipDeviceGetByPCIBusId(&device, pciBusId);
            out[1] = device;
            device++;
            int manualBusDevice = device;
            hipDeviceGetLimit(&limitValue, hipLimitMallocHeapSize);
            sizes[0] = limitValue;
            hipDeviceSetLimit(hipLimitMallocHeapSize, limitValue);
            limitValue = 4096;
            size_t manualLimit = limitValue;
            hipError_t errLimit =
                hipDeviceGetLimit(&statusLimit, hipLimitPrintfFifoSize);
            size_t statusLimitValue = statusLimit;
            hipGetDeviceFlags(&flags);
            unsigned int capturedFlags = flags;
            hipSetDeviceFlags(flags);
            flags++;
            unsigned int manualFlags = flags;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert "// HIP choose device: output: device, properties: props" in result
        assert "out[0] = (/* HIP device query: selectedDevice */ 0);" in result
        assert "device = 4;" in result
        assert "var manualDevice: i32 = device;" in result
        assert (
            "// HIP get device by PCI bus id: output: device, bus id: pciBusId"
            in result
        )
        assert "var errDevice: hipError_t = hipSuccess;" in result
        assert (
            "out[1] = (/* HIP device query: deviceByPCIBusId(pciBusId) */ 0);" in result
        )
        assert "(device++);" in result
        assert "var manualBusDevice: i32 = device;" in result
        assert (
            "// HIP get device limit: output: limitValue, "
            "limit: hipLimitMallocHeapSize"
        ) in result
        assert (
            "sizes[0] = (/* HIP device query: limit.hipLimitMallocHeapSize */ 0);"
            in result
        )
        assert (
            "// HIP set device limit: limit: hipLimitMallocHeapSize, "
            "value: limitValue"
        ) in result
        assert "limitValue = 4096;" in result
        assert "var manualLimit: u32 = limitValue;" in result
        assert (
            "// HIP get device limit: output: statusLimit, "
            "limit: hipLimitPrintfFifoSize"
        ) in result
        assert "var errLimit: hipError_t = hipSuccess;" in result
        assert (
            "var statusLimitValue: u32 = "
            "(/* HIP device query: limit.hipLimitPrintfFifoSize */ 0);"
        ) in result
        assert "// HIP get device flags: output: flags" in result
        assert (
            "var capturedFlags: u32 = (/* HIP device query: deviceFlags */ 0);"
            in result
        )
        assert "// HIP set device flags: flags" in result
        assert "(flags++);" in result
        assert "var manualFlags: u32 = flags;" in result
        assert "out[0] = device;" not in result
        assert "out[1] = device;" not in result
        assert "sizes[0] = limitValue;" not in result
        assert "var capturedFlags: u32 = flags;" not in result
        assert "var manualDevice: i32 = (/* HIP device query:" not in result
        assert "var manualBusDevice: i32 = (/* HIP device query:" not in result
        assert "var manualLimit: u32 = (/* HIP device query:" not in result
        assert "var manualFlags: u32 = (/* HIP device query:" not in result

    def test_hip_cache_and_shared_memory_config_reads_emit_metadata_expressions(
        self,
    ):
        """Test HIP cache/shared-memory config outputs lower to metadata."""
        code = """
        void host(int* out) {
            hipFuncCache_t deviceCache = hipFuncCachePreferNone;
            hipSharedMemConfig deviceShared = hipSharedMemBankSizeDefault;
            hipFuncCache_t contextCache = hipFuncCachePreferNone;
            hipSharedMemConfig contextShared = hipSharedMemBankSizeDefault;
            hipFuncCache_t statusCache = hipFuncCachePreferNone;
            hipSharedMemConfig statusShared = hipSharedMemBankSizeDefault;
            hipDeviceGetCacheConfig(&deviceCache);
            out[0] = deviceCache;
            hipDeviceSetCacheConfig(deviceCache);
            deviceCache = hipFuncCachePreferEqual;
            hipFuncCache_t manualDeviceCache = deviceCache;
            hipDeviceGetSharedMemConfig(&deviceShared);
            out[1] = deviceShared;
            hipDeviceSetSharedMemConfig(deviceShared);
            deviceShared = hipSharedMemBankSizeEightByte;
            hipSharedMemConfig manualDeviceShared = deviceShared;
            hipError_t errCache = hipCtxGetCacheConfig(&contextCache);
            out[2] = contextCache;
            hipCtxSetCacheConfig(contextCache);
            contextCache = hipFuncCachePreferL1;
            hipFuncCache_t manualContextCache = contextCache;
            hipError_t errShared = hipCtxGetSharedMemConfig(&contextShared);
            out[3] = contextShared;
            hipCtxSetSharedMemConfig(contextShared);
            contextShared = hipSharedMemBankSizeFourByte;
            hipSharedMemConfig manualContextShared = contextShared;
            hipError_t statusDeviceCache = hipDeviceGetCacheConfig(&statusCache);
            out[4] = statusCache;
            hipError_t statusDeviceShared =
                hipDeviceGetSharedMemConfig(&statusShared);
            out[5] = statusShared;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert "// HIP get device cache config: output: deviceCache" in result
        assert "out[0] = (/* HIP device query: cacheConfig */ 0);" in result
        assert "// HIP set device cache config: deviceCache" in result
        assert "deviceCache = hipFuncCachePreferEqual;" in result
        assert "var manualDeviceCache: hipFuncCache_t = deviceCache;" in result
        assert "// HIP get device shared memory config: output: deviceShared" in result
        assert "out[1] = (/* HIP device query: sharedMemConfig */ 0);" in result
        assert "// HIP set device shared memory config: deviceShared" in result
        assert "deviceShared = hipSharedMemBankSizeEightByte;" in result
        assert "var manualDeviceShared: hipSharedMemConfig = deviceShared;" in result
        assert "// HIP context get cache config: output: contextCache" in result
        assert "var errCache: hipError_t = hipSuccess;" in result
        assert "out[2] = (/* HIP device query: context.cacheConfig */ 0);" in result
        assert "// HIP context set cache config: contextCache" in result
        assert "contextCache = hipFuncCachePreferL1;" in result
        assert "var manualContextCache: hipFuncCache_t = contextCache;" in result
        assert (
            "// HIP context get shared memory config: output: contextShared" in result
        )
        assert "var errShared: hipError_t = hipSuccess;" in result
        assert "out[3] = (/* HIP device query: context.sharedMemConfig */ 0);" in result
        assert "// HIP context set shared memory config: contextShared" in result
        assert "contextShared = hipSharedMemBankSizeFourByte;" in result
        assert "var manualContextShared: hipSharedMemConfig = contextShared;" in result
        assert "var statusDeviceCache: hipError_t = hipSuccess;" in result
        assert "out[4] = (/* HIP device query: cacheConfig */ 0);" in result
        assert "var statusDeviceShared: hipError_t = hipSuccess;" in result
        assert "out[5] = (/* HIP device query: sharedMemConfig */ 0);" in result
        assert "out[0] = deviceCache;" not in result
        assert "out[1] = deviceShared;" not in result
        assert "out[2] = contextCache;" not in result
        assert "out[3] = contextShared;" not in result
        assert "out[4] = statusCache;" not in result
        assert "out[5] = statusShared;" not in result
        assert (
            "var manualDeviceCache: hipFuncCache_t = (/* HIP device query:"
            not in result
        )
        assert (
            "var manualDeviceShared: hipSharedMemConfig = (/* HIP device query:"
            not in result
        )
        assert (
            "var manualContextCache: hipFuncCache_t = (/* HIP device query:"
            not in result
        )
        assert (
            "var manualContextShared: hipSharedMemConfig = (/* HIP device query:"
            not in result
        )

    def test_hip_context_scalar_output_reads_emit_metadata_expressions(self):
        """Test HIP context scalar outputs lower to explicit metadata."""
        code = """
        void host(hipCtx_t ctx, int* out) {
            hipDevice_t device;
            unsigned int apiVersion = 0;
            unsigned int flags = 0;
            unsigned int statusFlags = 0;
            int active = 0;
            int statusActive = 0;
            hipCtxGetDevice(&device);
            out[0] = device;
            device = 2;
            hipDevice_t manualDevice = device;
            hipError_t errApi = hipCtxGetApiVersion(ctx, &apiVersion);
            out[1] = apiVersion;
            apiVersion = 13;
            unsigned int manualApiVersion = apiVersion;
            hipCtxGetFlags(&flags);
            out[2] = flags;
            hipDevicePrimaryCtxSetFlags(device, flags);
            flags = 7;
            unsigned int manualContextFlags = flags;
            hipDevicePrimaryCtxGetState(device, &flags, &active);
            out[3] = flags;
            out[4] = active;
            hipDevicePrimaryCtxSetFlags(device, flags);
            flags = 9;
            active = 1;
            unsigned int manualPrimaryFlags = flags;
            int manualPrimaryActive = active;
            hipError_t errState =
                hipDevicePrimaryCtxGetState(
                    device + 1,
                    &statusFlags,
                    &statusActive
                );
            out[5] = statusFlags;
            out[6] = statusActive;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert "// HIP context get device: output: device" in result
        assert "out[0] = (/* HIP device query: context.device */ 0);" in result
        assert "device = 2;" in result
        assert "var manualDevice: hipDevice_t = device;" in result
        assert (
            "// HIP context get API version: context: ctx, output: apiVersion" in result
        )
        assert "var errApi: hipError_t = hipSuccess;" in result
        assert "out[1] = (/* HIP device query: context.apiVersion(ctx) */ 0);" in result
        assert "apiVersion = 13;" in result
        assert "var manualApiVersion: u32 = apiVersion;" in result
        assert "// HIP context get flags: output: flags" in result
        assert "out[2] = (/* HIP device query: context.flags */ 0);" in result
        assert (
            "// HIP primary context set flags: device: device, flags: flags" in result
        )
        assert "flags = 7;" in result
        assert "var manualContextFlags: u32 = flags;" in result
        assert (
            "// HIP primary context get state: device: device, "
            "flags output: flags, active output: active"
        ) in result
        assert (
            "out[3] = (/* HIP device query: primaryContext.flags, "
            "device: device */ 0);"
        ) in result
        assert (
            "out[4] = (/* HIP device query: primaryContext.active, "
            "device: device */ 0);"
        ) in result
        assert "flags = 9;" in result
        assert "active = 1;" in result
        assert "var manualPrimaryFlags: u32 = flags;" in result
        assert "var manualPrimaryActive: i32 = active;" in result
        assert (
            "// HIP primary context get state: device: (device + 1), "
            "flags output: statusFlags, active output: statusActive"
        ) in result
        assert "var errState: hipError_t = hipSuccess;" in result
        assert (
            "out[5] = (/* HIP device query: primaryContext.flags, "
            "device: (device + 1) */ 0);"
        ) in result
        assert (
            "out[6] = (/* HIP device query: primaryContext.active, "
            "device: (device + 1) */ 0);"
        ) in result
        assert "out[0] = device;" not in result
        assert "out[1] = apiVersion;" not in result
        assert "out[2] = flags;" not in result
        assert "out[3] = flags;" not in result
        assert "out[4] = active;" not in result
        assert "out[5] = statusFlags;" not in result
        assert "out[6] = statusActive;" not in result
        assert "var manualDevice: hipDevice_t = (/* HIP device query:" not in result
        assert "var manualApiVersion: u32 = (/* HIP device query:" not in result
        assert "var manualContextFlags: u32 = (/* HIP device query:" not in result
        assert "var manualPrimaryFlags: u32 = (/* HIP device query:" not in result
        assert "var manualPrimaryActive: i32 = (/* HIP device query:" not in result

    def test_hip_driver_device_output_reads_emit_metadata_expressions(self):
        """Test HIP driver/device scalar outputs lower to explicit metadata."""
        code = """
        void host(int ordinal, hipDevice_t peerDevice, int* out) {
            hipDevice_t device;
            hipDevice_t statusDevice;
            int canAccess = 0;
            int statusCanAccess = 0;
            int p2pAttribute = 0;
            int statusP2PAttribute = 0;
            unsigned int linkType = 0;
            unsigned int hopCount = 0;
            unsigned int statusLinkType = 0;
            unsigned int statusHopCount = 0;
            hipDeviceGet(&device, ordinal);
            out[0] = device;
            hipDeviceCanAccessPeer(&canAccess, device, peerDevice);
            out[1] = canAccess;
            canAccess = 0;
            int manualCanAccess = canAccess;
            hipDeviceGetP2PAttribute(
                &p2pAttribute,
                hipDevP2PAttrPerformanceRank,
                device,
                peerDevice
            );
            out[2] = p2pAttribute;
            p2pAttribute = 7;
            int manualP2PAttribute = p2pAttribute;
            hipExtGetLinkTypeAndHopCount(
                device,
                peerDevice,
                &linkType,
                &hopCount
            );
            out[3] = linkType;
            out[4] = hopCount;
            linkType = 3;
            hopCount = 4;
            unsigned int manualLinkType = linkType;
            unsigned int manualHopCount = hopCount;
            hipError_t errDevice = hipDeviceGet(&statusDevice, ordinal + 1);
            out[5] = statusDevice;
            hipError_t errCanAccess =
                hipDeviceCanAccessPeer(&statusCanAccess, device, peerDevice);
            out[6] = statusCanAccess;
            hipError_t errP2P =
                hipDeviceGetP2PAttribute(
                    &statusP2PAttribute,
                    hipDevP2PAttrAccessSupported,
                    device,
                    peerDevice
                );
            out[7] = statusP2PAttribute;
            hipError_t errLink =
                hipExtGetLinkTypeAndHopCount(
                    device,
                    peerDevice,
                    &statusLinkType,
                    &statusHopCount
                );
            out[8] = statusLinkType;
            out[9] = statusHopCount;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert "// HIP get device handle: output: device, ordinal: ordinal" in result
        assert "out[0] = (/* HIP device query: deviceHandle(ordinal) */ 0);" in result
        assert (
            "// HIP device can access peer: output: canAccess, device: device, "
            "peer device: peerDevice"
        ) in result
        assert (
            "out[1] = (/* HIP device query: canAccessPeer(device, peerDevice) */ 0);"
            in result
        )
        assert "canAccess = 0;" in result
        assert "var manualCanAccess: i32 = canAccess;" in result
        assert (
            "// HIP get P2P attribute: output: p2pAttribute, "
            "attribute: hipDevP2PAttrPerformanceRank, source device: device, "
            "destination device: peerDevice"
        ) in result
        assert (
            "out[2] = (/* HIP device query: "
            "p2pAttribute.hipDevP2PAttrPerformanceRank(device, peerDevice) */ 0);"
            in result
        )
        assert "p2pAttribute = 7;" in result
        assert "var manualP2PAttribute: i32 = p2pAttribute;" in result
        assert (
            "// HIP get link type and hop count: device 1: device, "
            "device 2: peerDevice, link type output: linkType, "
            "hop count output: hopCount"
        ) in result
        assert (
            "out[3] = (/* HIP device query: linkType(device, peerDevice) */ 0);"
            in result
        )
        assert (
            "out[4] = (/* HIP device query: hopCount(device, peerDevice) */ 0);"
            in result
        )
        assert "linkType = 3;" in result
        assert "hopCount = 4;" in result
        assert "var manualLinkType: u32 = linkType;" in result
        assert "var manualHopCount: u32 = hopCount;" in result
        assert (
            "// HIP get device handle: output: statusDevice, ordinal: (ordinal + 1)"
            in result
        )
        assert "var errDevice: hipError_t = hipSuccess;" in result
        assert (
            "out[5] = (/* HIP device query: deviceHandle((ordinal + 1)) */ 0);"
            in result
        )
        assert "var errCanAccess: hipError_t = hipSuccess;" in result
        assert (
            "out[6] = (/* HIP device query: canAccessPeer(device, peerDevice) */ 0);"
            in result
        )
        assert "var errP2P: hipError_t = hipSuccess;" in result
        assert (
            "out[7] = (/* HIP device query: "
            "p2pAttribute.hipDevP2PAttrAccessSupported(device, peerDevice) */ 0);"
            in result
        )
        assert "var errLink: hipError_t = hipSuccess;" in result
        assert (
            "out[8] = (/* HIP device query: linkType(device, peerDevice) */ 0);"
            in result
        )
        assert (
            "out[9] = (/* HIP device query: hopCount(device, peerDevice) */ 0);"
            in result
        )
        assert "out[0] = device;" not in result
        assert "out[1] = canAccess;" not in result
        assert "out[2] = p2pAttribute;" not in result
        assert "out[3] = linkType;" not in result
        assert "out[4] = hopCount;" not in result
        assert "out[5] = statusDevice;" not in result
        assert "out[6] = statusCanAccess;" not in result
        assert "out[7] = statusP2PAttribute;" not in result
        assert "out[8] = statusLinkType;" not in result
        assert "out[9] = statusHopCount;" not in result
        assert "var manualCanAccess: i32 = (/* HIP device query:" not in result
        assert "var manualP2PAttribute: i32 = (/* HIP device query:" not in result
        assert "var manualLinkType: u32 = (/* HIP device query:" not in result
        assert "var manualHopCount: u32 = (/* HIP device query:" not in result

    def test_hip_runtime_version_output_reads_emit_metadata_expressions(self):
        """Test HIP runtime version outputs lower to explicit metadata."""
        code = """
        void host(int* out) {
            int driverVersion = 0;
            int runtimeVersion = 0;
            int statusDriverVersion = 0;
            int statusRuntimeVersion = 0;
            void* proc = 0;
            unsigned int procFlags = 0;
            hipDriverProcAddressQueryResult procStatus;
            hipDriverGetVersion(&driverVersion);
            hipRuntimeGetVersion(&runtimeVersion);
            out[0] = driverVersion;
            out[1] = runtimeVersion;
            hipGetProcAddress(
                "hipMalloc",
                &proc,
                runtimeVersion,
                procFlags,
                &procStatus
            );
            driverVersion = 1;
            runtimeVersion = 2;
            int manualDriverVersion = driverVersion;
            int manualRuntimeVersion = runtimeVersion;
            hipError_t errDriver = hipDriverGetVersion(&statusDriverVersion);
            hipError_t errRuntime = hipRuntimeGetVersion(&statusRuntimeVersion);
            out[2] = statusDriverVersion;
            out[3] = statusRuntimeVersion;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert "// HIP get driver version: output: driverVersion" in result
        assert "// HIP get runtime version: output: runtimeVersion" in result
        assert "out[0] = (/* HIP device query: driver.version */ 0);" in result
        assert "out[1] = (/* HIP device query: runtime.version */ 0);" in result
        assert (
            '// HIP get proc address: symbol: "hipMalloc", output: proc, '
            "version: runtimeVersion, flags: procFlags, "
            "status output: procStatus"
        ) in result
        assert "driverVersion = 1;" in result
        assert "runtimeVersion = 2;" in result
        assert "var manualDriverVersion: i32 = driverVersion;" in result
        assert "var manualRuntimeVersion: i32 = runtimeVersion;" in result
        assert "// HIP get driver version: output: statusDriverVersion" in result
        assert "// HIP get runtime version: output: statusRuntimeVersion" in result
        assert "var errDriver: hipError_t = hipSuccess;" in result
        assert "var errRuntime: hipError_t = hipSuccess;" in result
        assert "out[2] = (/* HIP device query: driver.version */ 0);" in result
        assert "out[3] = (/* HIP device query: runtime.version */ 0);" in result
        assert "out[0] = driverVersion;" not in result
        assert "out[1] = runtimeVersion;" not in result
        assert "out[2] = statusDriverVersion;" not in result
        assert "out[3] = statusRuntimeVersion;" not in result
        assert "var manualDriverVersion: i32 = (/* HIP device query:" not in result
        assert "var manualRuntimeVersion: i32 = (/* HIP device query:" not in result

    def test_hip_stream_scalar_output_reads_emit_metadata_expressions(self):
        """Test HIP stream scalar outputs lower to explicit metadata."""
        code = """
        void host(hipStream_t stream, int* out) {
            int leastPriority = 0;
            int greatestPriority = 0;
            int statusLeastPriority = 0;
            int statusGreatestPriority = 0;
            unsigned int flags = 0;
            unsigned int statusFlags = 0;
            int priority = 0;
            int statusPriority = 0;
            hipStreamCaptureStatus captureStatus;
            hipStreamCaptureStatus statusCaptureStatus;
            unsigned long long captureId = 0;
            unsigned long long statusCaptureId = 0;
            size_t numDeps = 0;
            size_t statusNumDeps = 0;
            hipGraph_t graph;
            hipGraphNode_t* deps;
            hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
            out[0] = leastPriority;
            out[1] = greatestPriority;
            leastPriority = 1;
            greatestPriority = 2;
            int manualLeastPriority = leastPriority;
            int manualGreatestPriority = greatestPriority;
            hipStreamGetFlags(stream, &flags);
            out[2] = flags;
            hipStreamUpdateCaptureDependencies(stream, deps, numDeps, flags);
            flags = 0;
            unsigned int manualFlags = flags;
            hipStreamGetPriority(stream, &priority);
            out[3] = priority;
            priority = 3;
            int manualPriority = priority;
            hipStreamIsCapturing(stream, &captureStatus);
            out[4] = captureStatus;
            hipStreamGetCaptureInfo(stream, &captureStatus, &captureId);
            out[5] = captureStatus;
            out[6] = captureId;
            hipStreamGetCaptureInfo_v2(
                stream,
                &captureStatus,
                &captureId,
                &graph,
                &deps,
                &numDeps
            );
            out[7] = captureStatus;
            out[8] = captureId;
            out[9] = numDeps;
            numDeps = 4;
            size_t manualNumDeps = numDeps;
            hipError_t errRange =
                hipDeviceGetStreamPriorityRange(
                    &statusLeastPriority,
                    &statusGreatestPriority
                );
            out[10] = statusLeastPriority;
            out[11] = statusGreatestPriority;
            hipError_t errFlags = hipStreamGetFlags(stream, &statusFlags);
            out[12] = statusFlags;
            hipError_t errPriority =
                hipStreamGetPriority(stream, &statusPriority);
            out[13] = statusPriority;
            hipError_t errCapture =
                hipStreamGetCaptureInfo_v2(
                    stream,
                    &statusCaptureStatus,
                    &statusCaptureId,
                    &graph,
                    &deps,
                    &statusNumDeps
                );
            out[14] = statusCaptureStatus;
            out[15] = statusCaptureId;
            out[16] = statusNumDeps;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "// HIP get stream priority range: least output: leastPriority, "
            "greatest output: greatestPriority"
        ) in result
        assert (
            "out[0] = (/* HIP device query: streamPriorityRange.least */ 0);" in result
        )
        assert (
            "out[1] = (/* HIP device query: streamPriorityRange.greatest */ 0);"
            in result
        )
        assert "leastPriority = 1;" in result
        assert "greatestPriority = 2;" in result
        assert "var manualLeastPriority: i32 = leastPriority;" in result
        assert "var manualGreatestPriority: i32 = greatestPriority;" in result
        assert "// HIP get stream flags: stream: stream, output: flags" in result
        assert "out[2] = (/* HIP device query: stream.flags(stream) */ 0);" in result
        assert (
            "// HIP stream update capture dependencies: stream: stream, "
            "dependencies: deps, count: numDeps, flags: flags"
        ) in result
        assert "flags = 0;" in result
        assert "var manualFlags: u32 = flags;" in result
        assert "// HIP get stream priority: stream: stream, output: priority" in result
        assert "out[3] = (/* HIP device query: stream.priority(stream) */ 0);" in result
        assert "priority = 3;" in result
        assert "var manualPriority: i32 = priority;" in result
        assert (
            "// HIP stream is capturing: stream: stream, output: captureStatus"
            in result
        )
        assert (
            "out[4] = (/* HIP device query: stream.captureStatus(stream) */ 0);"
            in result
        )
        assert (
            "// HIP stream capture info: stream: stream, "
            "status output: captureStatus, id output: captureId"
        ) in result
        assert (
            "out[5] = (/* HIP device query: stream.captureStatus(stream) */ 0);"
            in result
        )
        assert (
            "out[6] = (/* HIP device query: stream.captureId(stream) */ 0);" in result
        )
        assert (
            "// HIP stream capture info: stream: stream, "
            "status output: captureStatus, id output: captureId, "
            "graph output: graph, dependencies output: deps, "
            "dependency count output: numDeps"
        ) in result
        assert (
            "out[7] = (/* HIP device query: stream.captureStatus(stream) */ 0);"
            in result
        )
        assert (
            "out[8] = (/* HIP device query: stream.captureId(stream) */ 0);" in result
        )
        assert (
            "out[9] = (/* HIP device query: "
            "stream.captureDependencyCount(stream) */ 0);"
        ) in result
        assert "numDeps = 4;" in result
        assert "var manualNumDeps: u32 = numDeps;" in result
        assert "var errRange: hipError_t = hipSuccess;" in result
        assert (
            "out[10] = (/* HIP device query: streamPriorityRange.least */ 0);" in result
        )
        assert (
            "out[11] = (/* HIP device query: streamPriorityRange.greatest */ 0);"
            in result
        )
        assert "var errFlags: hipError_t = hipSuccess;" in result
        assert "out[12] = (/* HIP device query: stream.flags(stream) */ 0);" in result
        assert "var errPriority: hipError_t = hipSuccess;" in result
        assert (
            "out[13] = (/* HIP device query: stream.priority(stream) */ 0);" in result
        )
        assert "var errCapture: hipError_t = hipSuccess;" in result
        assert (
            "out[14] = (/* HIP device query: stream.captureStatus(stream) */ 0);"
            in result
        )
        assert (
            "out[15] = (/* HIP device query: stream.captureId(stream) */ 0);" in result
        )
        assert (
            "out[16] = (/* HIP device query: "
            "stream.captureDependencyCount(stream) */ 0);"
        ) in result
        assert "out[0] = leastPriority;" not in result
        assert "out[1] = greatestPriority;" not in result
        assert "out[2] = flags;" not in result
        assert "out[3] = priority;" not in result
        assert "out[4] = captureStatus;" not in result
        assert "out[6] = captureId;" not in result
        assert "out[9] = numDeps;" not in result
        assert "var manualLeastPriority: i32 = (/* HIP device query:" not in result
        assert "var manualGreatestPriority: i32 = (/* HIP device query:" not in result
        assert "var manualFlags: u32 = (/* HIP device query:" not in result
        assert "var manualPriority: i32 = (/* HIP device query:" not in result
        assert "var manualNumDeps: u32 = (/* HIP device query:" not in result

    def test_hip_occupancy_output_reads_emit_metadata_expressions(self):
        """Test HIP occupancy scalar outputs lower to explicit metadata."""
        code = """
        void host(void* kernel, void* dynamicSmem, int* out) {
            int gridSize = 0;
            int blockSize = 0;
            int variableGridSize = 0;
            int variableBlockSize = 0;
            int flaggedGridSize = 0;
            int flaggedBlockSize = 0;
            int activeBlocks = 0;
            int flaggedActiveBlocks = 0;
            int statusGridSize = 0;
            int statusBlockSize = 0;
            int statusActiveBlocks = 0;
            hipOccupancyMaxPotentialBlockSize(
                &gridSize,
                &blockSize,
                kernel,
                0,
                0
            );
            out[0] = gridSize;
            out[1] = blockSize;
            hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &activeBlocks,
                kernel,
                blockSize,
                0
            );
            out[2] = activeBlocks;
            gridSize = 1;
            blockSize = 2;
            activeBlocks = 3;
            int manualGridSize = gridSize;
            int manualBlockSize = blockSize;
            int manualActiveBlocks = activeBlocks;
            hipOccupancyMaxPotentialBlockSizeVariableSMem(
                &variableGridSize,
                &variableBlockSize,
                kernel,
                dynamicSmem,
                128
            );
            out[3] = variableGridSize;
            out[4] = variableBlockSize;
            hipOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(
                &flaggedGridSize,
                &flaggedBlockSize,
                kernel,
                dynamicSmem,
                256,
                1
            );
            out[5] = flaggedGridSize;
            out[6] = flaggedBlockSize;
            hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                &flaggedActiveBlocks,
                kernel,
                flaggedBlockSize,
                0,
                1
            );
            out[7] = flaggedActiveBlocks;
            hipError_t errPotential =
                hipOccupancyMaxPotentialBlockSize(
                    &statusGridSize,
                    &statusBlockSize,
                    kernel,
                    0,
                    0
                );
            out[8] = statusGridSize;
            out[9] = statusBlockSize;
            hipError_t errActive =
                hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                    &statusActiveBlocks,
                    kernel,
                    statusBlockSize,
                    0,
                    1
                );
            out[10] = statusActiveBlocks;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "// HIP occupancy max potential block size: grid output: gridSize, "
            "block output: blockSize, kernel: kernel, dynamic shared memory: 0, "
            "block size limit: 0"
        ) in result
        assert (
            "out[0] = (/* HIP device query: "
            "occupancy.maxPotentialBlockSize.grid(kernel, 0, 0) */ 0);"
        ) in result
        assert (
            "out[1] = (/* HIP device query: "
            "occupancy.maxPotentialBlockSize.block(kernel, 0, 0) */ 0);"
        ) in result
        assert (
            "// HIP occupancy active blocks per multiprocessor: "
            "output: activeBlocks, kernel: kernel, block size: blockSize, "
            "dynamic shared memory: 0"
        ) in result
        assert (
            "out[2] = (/* HIP device query: "
            "occupancy.maxActiveBlocksPerMultiprocessor(kernel, blockSize, 0) */ 0);"
        ) in result
        assert "gridSize = 1;" in result
        assert "blockSize = 2;" in result
        assert "activeBlocks = 3;" in result
        assert "var manualGridSize: i32 = gridSize;" in result
        assert "var manualBlockSize: i32 = blockSize;" in result
        assert "var manualActiveBlocks: i32 = activeBlocks;" in result
        assert (
            "// HIP occupancy max potential block size: "
            "grid output: variableGridSize, block output: variableBlockSize, "
            "kernel: kernel, dynamic shared memory: dynamicSmem, "
            "block size limit: 128"
        ) in result
        assert (
            "out[3] = (/* HIP device query: "
            "occupancy.maxPotentialBlockSizeVariableSMem.grid("
            "kernel, dynamicSmem, 128) */ 0);"
        ) in result
        assert (
            "out[4] = (/* HIP device query: "
            "occupancy.maxPotentialBlockSizeVariableSMem.block("
            "kernel, dynamicSmem, 128) */ 0);"
        ) in result
        assert (
            "// HIP occupancy max potential block size: "
            "grid output: flaggedGridSize, block output: flaggedBlockSize, "
            "kernel: kernel, dynamic shared memory: dynamicSmem, "
            "block size limit: 256, flags: 1"
        ) in result
        assert (
            "out[5] = (/* HIP device query: "
            "occupancy.maxPotentialBlockSizeVariableSMemWithFlags.grid("
            "kernel, dynamicSmem, 256, 1) */ 0);"
        ) in result
        assert (
            "out[6] = (/* HIP device query: "
            "occupancy.maxPotentialBlockSizeVariableSMemWithFlags.block("
            "kernel, dynamicSmem, 256, 1) */ 0);"
        ) in result
        assert (
            "// HIP occupancy active blocks per multiprocessor: "
            "output: flaggedActiveBlocks, kernel: kernel, "
            "block size: flaggedBlockSize, dynamic shared memory: 0, flags: 1"
        ) in result
        assert (
            "out[7] = (/* HIP device query: "
            "occupancy.maxActiveBlocksPerMultiprocessorWithFlags("
            "kernel, flaggedBlockSize, 0, 1) */ 0);"
        ) in result
        assert "var errPotential: hipError_t = hipSuccess;" in result
        assert (
            "out[8] = (/* HIP device query: "
            "occupancy.maxPotentialBlockSize.grid(kernel, 0, 0) */ 0);"
        ) in result
        assert (
            "out[9] = (/* HIP device query: "
            "occupancy.maxPotentialBlockSize.block(kernel, 0, 0) */ 0);"
        ) in result
        assert "var errActive: hipError_t = hipSuccess;" in result
        assert (
            "out[10] = (/* HIP device query: "
            "occupancy.maxActiveBlocksPerMultiprocessorWithFlags("
            "kernel, statusBlockSize, 0, 1) */ 0);"
        ) in result
        assert "out[0] = gridSize;" not in result
        assert "out[1] = blockSize;" not in result
        assert "out[2] = activeBlocks;" not in result
        assert "out[3] = variableGridSize;" not in result
        assert "out[4] = variableBlockSize;" not in result
        assert "out[5] = flaggedGridSize;" not in result
        assert "out[6] = flaggedBlockSize;" not in result
        assert "out[7] = flaggedActiveBlocks;" not in result
        assert "out[8] = statusGridSize;" not in result
        assert "out[9] = statusBlockSize;" not in result
        assert "out[10] = statusActiveBlocks;" not in result
        assert "var manualGridSize: i32 = (/* HIP device query:" not in result
        assert "var manualBlockSize: i32 = (/* HIP device query:" not in result
        assert "var manualActiveBlocks: i32 = (/* HIP device query:" not in result

    def test_hip_memory_event_output_reads_emit_metadata_expressions(self):
        """Test HIP memory and event outputs lower to explicit metadata."""
        code = """
        void host(
            hipEvent_t start,
            hipEvent_t stop,
            void* devicePtr,
            hipMemPool_t pool,
            float* times,
            size_t* sizes,
            int* attrs,
            unsigned long long* flags
        ) {
            float elapsedMs = 0.0f;
            float statusElapsedMs = 0.0f;
            size_t freeMem = 0;
            size_t totalMem = 0;
            void* basePtr;
            size_t rangeSize = 0;
            int pointerAttrValue = 0;
            int statusPointerAttrValue = 0;
            size_t pointerSize = 0;
            size_t statusPointerSize = 0;
            unsigned long long accessFlags = 0;
            unsigned long long poolAccessFlags = 0;
            unsigned int hostFlags = 0;
            size_t poolAttrValue = 0;
            size_t allocationGranularity = 0;
            hipMemLocation location;
            hipMemAllocationProp prop;
            hipEventElapsedTime(&elapsedMs, start, stop);
            times[0] = elapsedMs;
            elapsedMs = 1.0f;
            float manualElapsedMs = elapsedMs;
            hipMemGetInfo(&freeMem, &totalMem);
            sizes[0] = freeMem;
            sizes[1] = totalMem;
            hipMemGetAddressRange(&basePtr, &rangeSize, devicePtr);
            void* rangeBase = basePtr;
            sizes[2] = rangeSize;
            basePtr = devicePtr;
            rangeSize = 3;
            void* manualBase = basePtr;
            size_t manualRangeSize = rangeSize;
            hipPointerGetAttribute(
                &pointerAttrValue,
                hipPointerAttributeMemoryType,
                devicePtr
            );
            attrs[0] = pointerAttrValue;
            hipMemPtrGetInfo(devicePtr, &pointerSize);
            sizes[3] = pointerSize;
            hipMemGetAccess(&accessFlags, &location, devicePtr);
            flags[0] = accessFlags;
            hipMemPoolGetAttribute(pool, hipMemPoolAttrReleaseThreshold, &poolAttrValue);
            sizes[4] = poolAttrValue;
            hipMemPoolGetAccess(&poolAccessFlags, pool, &location);
            flags[1] = poolAccessFlags;
            hipHostGetFlags(&hostFlags, devicePtr);
            flags[2] = hostFlags;
            hipMemGetAllocationGranularity(
                &allocationGranularity,
                &prop,
                hipMemAllocationGranularityMinimum
            );
            sizes[5] = allocationGranularity;
            freeMem = 4;
            totalMem = 5;
            pointerAttrValue = 6;
            pointerSize = 7;
            accessFlags = 8;
            poolAttrValue = 9;
            poolAccessFlags = 10;
            hostFlags = 11;
            allocationGranularity = 12;
            size_t manualFreeMem = freeMem;
            size_t manualTotalMem = totalMem;
            int manualPointerAttrValue = pointerAttrValue;
            size_t manualPointerSize = pointerSize;
            unsigned long long manualAccessFlags = accessFlags;
            size_t manualPoolAttrValue = poolAttrValue;
            unsigned long long manualPoolAccessFlags = poolAccessFlags;
            unsigned int manualHostFlags = hostFlags;
            size_t manualAllocationGranularity = allocationGranularity;
            hipError_t errElapsed =
                hipEventElapsedTime(&statusElapsedMs, start, stop);
            times[1] = statusElapsedMs;
            hipError_t errPointerSize =
                hipMemPtrGetInfo(devicePtr, &statusPointerSize);
            sizes[6] = statusPointerSize;
            hipError_t errPointerAttr =
                hipPointerGetAttribute(
                    &statusPointerAttrValue,
                    hipPointerAttributeDevicePointer,
                    devicePtr
                );
            attrs[1] = statusPointerAttrValue;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert "// HIP event elapsed time: start -> stop, output: elapsedMs" in result
        assert (
            "times[0] = (/* HIP device query: " "event.elapsedTime(start, stop) */ 0);"
        ) in result
        assert "elapsedMs = 1.0f;" in result
        assert "var manualElapsedMs: f32 = elapsedMs;" in result
        assert (
            "// HIP memory info: free output: freeMem, total output: totalMem" in result
        )
        assert "sizes[0] = (/* HIP device query: memory.info.free */ 0);" in result
        assert "sizes[1] = (/* HIP device query: memory.info.total */ 0);" in result
        assert (
            "// HIP driver memory address range: base output: basePtr, "
            "size output: rangeSize, pointer: devicePtr"
        ) in result
        assert (
            "var rangeBase: ptr<void> = (/* HIP device query: "
            "memory.addressRange.base(devicePtr) */ 0);"
        ) in result
        assert (
            "sizes[2] = (/* HIP device query: "
            "memory.addressRange.size(devicePtr) */ 0);"
        ) in result
        assert "basePtr = devicePtr;" in result
        assert "rangeSize = 3;" in result
        assert "var manualBase: ptr<void> = basePtr;" in result
        assert "var manualRangeSize: u32 = rangeSize;" in result
        assert (
            "// HIP pointer attribute: output: pointerAttrValue, "
            "attribute: hipPointerAttributeMemoryType, pointer: devicePtr"
        ) in result
        assert (
            "attrs[0] = (/* HIP device query: "
            "pointer.attribute(hipPointerAttributeMemoryType, devicePtr) */ 0);"
        ) in result
        assert (
            "// HIP memory pointer info: pointer: devicePtr, "
            "size output: pointerSize"
        ) in result
        assert (
            "sizes[3] = (/* HIP device query: " "memoryPointer.size(devicePtr) */ 0);"
        ) in result
        assert (
            "// HIP virtual memory get access: output: accessFlags, "
            "location: (&location), pointer: devicePtr"
        ) in result
        assert (
            "flags[0] = (/* HIP device query: "
            "virtualMemory.accessFlags(location, devicePtr) */ 0);"
        ) in result
        assert (
            "// HIP memory pool get attribute: pool: pool, "
            "attribute: hipMemPoolAttrReleaseThreshold, output: poolAttrValue"
        ) in result
        assert (
            "sizes[4] = (/* HIP device query: "
            "memoryPool.attribute(pool, hipMemPoolAttrReleaseThreshold) */ 0);"
        ) in result
        assert (
            "// HIP memory pool get access: output: poolAccessFlags, "
            "pool: pool, location: (&location)"
        ) in result
        assert (
            "flags[1] = (/* HIP device query: "
            "memoryPool.accessFlags(pool, location) */ 0);"
        ) in result
        assert "// HIP host memory flags: output: hostFlags, host: devicePtr" in result
        assert (
            "flags[2] = (/* HIP device query: " "hostMemory.flags(devicePtr) */ 0);"
        ) in result
        assert (
            "// HIP virtual memory allocation granularity: "
            "output: allocationGranularity, properties: (&prop), "
            "option: hipMemAllocationGranularityMinimum"
        ) in result
        assert (
            "sizes[5] = (/* HIP device query: "
            "virtualMemory.allocationGranularity("
            "prop, hipMemAllocationGranularityMinimum) */ 0);"
        ) in result
        assert "var manualFreeMem: u32 = freeMem;" in result
        assert "var manualTotalMem: u32 = totalMem;" in result
        assert "var manualPointerAttrValue: i32 = pointerAttrValue;" in result
        assert "var manualPointerSize: u32 = pointerSize;" in result
        assert "var manualAccessFlags: u64 = accessFlags;" in result
        assert "var manualPoolAttrValue: u32 = poolAttrValue;" in result
        assert "var manualPoolAccessFlags: u64 = poolAccessFlags;" in result
        assert "var manualHostFlags: u32 = hostFlags;" in result
        assert "var manualAllocationGranularity: u32 = allocationGranularity;" in result
        assert "var errElapsed: hipError_t = hipSuccess;" in result
        assert (
            "times[1] = (/* HIP device query: " "event.elapsedTime(start, stop) */ 0);"
        ) in result
        assert "var errPointerSize: hipError_t = hipSuccess;" in result
        assert (
            "sizes[6] = (/* HIP device query: " "memoryPointer.size(devicePtr) */ 0);"
        ) in result
        assert "var errPointerAttr: hipError_t = hipSuccess;" in result
        assert (
            "attrs[1] = (/* HIP device query: "
            "pointer.attribute(hipPointerAttributeDevicePointer, devicePtr) */ 0);"
        ) in result
        assert "times[0] = elapsedMs;" not in result
        assert "sizes[0] = freeMem;" not in result
        assert "sizes[1] = totalMem;" not in result
        assert "sizes[2] = rangeSize;" not in result
        assert "attrs[0] = pointerAttrValue;" not in result
        assert "sizes[3] = pointerSize;" not in result
        assert "flags[0] = accessFlags;" not in result
        assert "sizes[4] = poolAttrValue;" not in result
        assert "flags[1] = poolAccessFlags;" not in result
        assert "flags[2] = hostFlags;" not in result
        assert "sizes[5] = allocationGranularity;" not in result
        assert "times[1] = statusElapsedMs;" not in result
        assert "sizes[6] = statusPointerSize;" not in result
        assert "attrs[1] = statusPointerAttrValue;" not in result
        assert "var manualElapsedMs: f32 = (/* HIP device query:" not in result
        assert "var manualFreeMem: u32 = (/* HIP device query:" not in result
        assert "var manualPointerSize: u32 = (/* HIP device query:" not in result
        assert "var manualAccessFlags: u64 = (/* HIP device query:" not in result

    def test_hip_symbol_range_function_output_reads_emit_metadata_expressions(self):
        """Test HIP symbol, range, function, and array scalar outputs."""
        code = """
        void host(
            void* symbol,
            void* devicePtr,
            hipFunction_t function,
            hipArray_t array,
            size_t* sizes,
            int* values,
            unsigned int* flagsOut
        ) {
            size_t symbolSize = 0;
            size_t statusSymbolSize = 0;
            int rangeValue = 0;
            int statusRangeValue = 0;
            int functionAttribute = 0;
            int statusFunctionAttribute = 0;
            unsigned int arrayFlags = 0;
            HIP_ARRAY_DESCRIPTOR descriptor;
            hipExtent extent;
            hipGetSymbolSize(&symbolSize, symbol);
            sizes[0] = symbolSize;
            symbolSize = 1;
            size_t manualSymbolSize = symbolSize;
            hipMemRangeGetAttribute(
                &rangeValue,
                sizeof(int),
                hipMemRangeAttributePreferredLocation,
                devicePtr,
                256
            );
            values[0] = rangeValue;
            rangeValue = 2;
            int manualRangeValue = rangeValue;
            hipFuncGetAttribute(
                &functionAttribute,
                hipFuncAttributeMaxThreadsPerBlock,
                function
            );
            values[1] = functionAttribute;
            functionAttribute = 3;
            int manualFunctionAttribute = functionAttribute;
            hipArrayGetInfo(&descriptor, &extent, &arrayFlags, array);
            flagsOut[0] = arrayFlags;
            arrayFlags = 4;
            unsigned int manualArrayFlags = arrayFlags;
            hipError_t errSymbol = hipGetSymbolSize(&statusSymbolSize, symbol);
            sizes[1] = statusSymbolSize;
            hipError_t errRange =
                hipMemRangeGetAttribute(
                    &statusRangeValue,
                    sizeof(int),
                    hipMemRangeAttributeLastPrefetchLocation,
                    devicePtr,
                    512
                );
            values[2] = statusRangeValue;
            hipError_t errFunction =
                hipFuncGetAttribute(
                    &statusFunctionAttribute,
                    hipFuncAttributeSharedSizeBytes,
                    function
                );
            values[3] = statusFunctionAttribute;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert "// HIP get symbol size: output: symbolSize, symbol: symbol" in result
        assert "sizes[0] = (/* HIP device query: symbol.size(symbol) */ 0);" in result
        assert "symbolSize = 1;" in result
        assert "var manualSymbolSize: u32 = symbolSize;" in result
        assert (
            "// HIP memory range get attribute: output: rangeValue, "
            "output bytes: sizeof(int), "
            "attribute: hipMemRangeAttributePreferredLocation, "
            "pointer: devicePtr, range bytes: 256"
        ) in result
        assert (
            "values[0] = (/* HIP device query: "
            "memory.rangeAttribute("
            "hipMemRangeAttributePreferredLocation, devicePtr, 256) */ 0);"
        ) in result
        assert "rangeValue = 2;" in result
        assert "var manualRangeValue: i32 = rangeValue;" in result
        assert (
            "// HIP function get attribute: output: functionAttribute, "
            "attribute: hipFuncAttributeMaxThreadsPerBlock, function: function"
        ) in result
        assert (
            "values[1] = (/* HIP device query: "
            "function.attribute(hipFuncAttributeMaxThreadsPerBlock, function) */ 0);"
        ) in result
        assert "functionAttribute = 3;" in result
        assert "var manualFunctionAttribute: i32 = functionAttribute;" in result
        assert (
            "// HIP array get info: desc output: descriptor, "
            "extent output: extent, flags output: arrayFlags, array: array"
        ) in result
        assert (
            "flagsOut[0] = (/* HIP device query: array.info.flags(array) */ 0);"
            in result
        )
        assert "arrayFlags = 4;" in result
        assert "var manualArrayFlags: u32 = arrayFlags;" in result
        assert "var errSymbol: hipError_t = hipSuccess;" in result
        assert "sizes[1] = (/* HIP device query: symbol.size(symbol) */ 0);" in result
        assert "var errRange: hipError_t = hipSuccess;" in result
        assert (
            "values[2] = (/* HIP device query: "
            "memory.rangeAttribute("
            "hipMemRangeAttributeLastPrefetchLocation, devicePtr, 512) */ 0);"
        ) in result
        assert "var errFunction: hipError_t = hipSuccess;" in result
        assert (
            "values[3] = (/* HIP device query: "
            "function.attribute(hipFuncAttributeSharedSizeBytes, function) */ 0);"
        ) in result
        assert "sizes[0] = symbolSize;" not in result
        assert "values[0] = rangeValue;" not in result
        assert "values[1] = functionAttribute;" not in result
        assert "flagsOut[0] = arrayFlags;" not in result
        assert "sizes[1] = statusSymbolSize;" not in result
        assert "values[2] = statusRangeValue;" not in result
        assert "values[3] = statusFunctionAttribute;" not in result
        assert "var manualSymbolSize: u32 = (/* HIP device query:" not in result
        assert "var manualRangeValue: i32 = (/* HIP device query:" not in result
        assert "var manualFunctionAttribute: i32 = (/* HIP device query:" not in result
        assert "var manualArrayFlags: u32 = (/* HIP device query:" not in result

    def test_hip_function_attribute_member_reads_emit_metadata_expressions(self):
        """Test hipFuncAttributes member reads lower to explicit metadata."""
        code = """
        void host(hipFunction_t function, int* out) {
            hipFuncAttributes attrs;
            hipFuncAttributes statusAttrs;
            hipFuncGetAttributes(&attrs, function);
            out[0] = attrs.maxThreadsPerBlock;
            out[1] = attrs.sharedSizeBytes;
            int regs = attrs.numRegs;
            out[2] = attrs.binaryVersion;
            out[3] = attrs.cacheModeCA;
            out[4] = attrs.constSizeBytes;
            out[5] = attrs.maxDynamicSharedSizeBytes;
            out[6] = attrs.preferredShmemCarveout;
            out[7] = attrs.ptxVersion;
            attrs.maxThreadsPerBlock = 128;
            int manualMax = attrs.maxThreadsPerBlock;
            hipError_t err = hipFuncGetAttributes(&statusAttrs, function);
            out[8] = statusAttrs.localSizeBytes;
            statusAttrs.sharedSizeBytes++;
            int manualShared = statusAttrs.sharedSizeBytes;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "// HIP function get attributes: output: attrs, function: function"
            in result
        )
        assert (
            "out[0] = (/* HIP device query: "
            "function.attributes.maxThreadsPerBlock(function) */ 0);"
        ) in result
        assert (
            "out[1] = (/* HIP device query: "
            "function.attributes.sharedSizeBytes(function) */ 0);"
        ) in result
        assert (
            "var regs: i32 = (/* HIP device query: "
            "function.attributes.numRegs(function) */ 0);"
        ) in result
        for index, member in [
            (2, "binaryVersion"),
            (3, "cacheModeCA"),
            (4, "constSizeBytes"),
            (5, "maxDynamicSharedSizeBytes"),
            (6, "preferredShmemCarveout"),
            (7, "ptxVersion"),
        ]:
            assert (
                f"out[{index}] = (/* HIP device query: "
                f"function.attributes.{member}(function) */ 0);"
            ) in result
        assert "attrs.maxThreadsPerBlock = 128;" in result
        assert "var manualMax: i32 = attrs.maxThreadsPerBlock;" in result
        assert "var err: hipError_t = hipSuccess;" in result
        assert (
            "out[8] = (/* HIP device query: "
            "function.attributes.localSizeBytes(function) */ 0);"
        ) in result
        assert "(statusAttrs.sharedSizeBytes++);" in result
        assert "var manualShared: i32 = statusAttrs.sharedSizeBytes;" in result
        assert "out[0] = attrs.maxThreadsPerBlock;" not in result
        assert "out[1] = attrs.sharedSizeBytes;" not in result
        assert "var regs: i32 = attrs.numRegs;" not in result
        for index, member in [
            (2, "binaryVersion"),
            (3, "cacheModeCA"),
            (4, "constSizeBytes"),
            (5, "maxDynamicSharedSizeBytes"),
            (6, "preferredShmemCarveout"),
            (7, "ptxVersion"),
        ]:
            assert f"out[{index}] = attrs.{member};" not in result
        assert "out[8] = statusAttrs.localSizeBytes;" not in result
        assert "var manualMax: i32 = (/* HIP device query:" not in result
        assert "var manualShared: i32 = (/* HIP device query:" not in result

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
            size_t limitValue = 0;
            size_t numDeps = 0;
            unsigned long long captureId = 0;
            int validDevices[2];
            char name[64];
            char pciBusId[32];
            hipUUID uuid;
            hipDriverProcAddressQueryResult queryStatus;
            hipStreamCaptureStatus captureStatus;
            hipGraph_t graph;
            hipGraphNode_t* deps;
            hipDeviceProp_t props;
            hipFuncCache_t cacheConfig;
            hipSharedMemConfig sharedConfig;
            void* kernel;
            void* callback;
            void* hostFn;
            void* userData;
            void* proc;
            unsigned long long procFlags = 0ull;
            hipGetDevice(&device);
            hipGetDeviceCount(&count);
            hipSetDevice(device);
            hipSetValidDevices(validDevices, 2);
            hipGetDeviceProperties(&props, device);
            hipDeviceGetAttribute(
                &attr, hipDeviceAttributeMaxThreadsPerBlock, device
            );
            hipDeviceGetName(name, 64, device);
            hipDeviceGetUuid(&uuid, device);
            hipDeviceTotalMem(&total, device);
            hipDeviceComputeCapability(&major, &minor, device);
            hipChooseDevice(&device, &props);
            hipDeviceGetPCIBusId(pciBusId, 32, device);
            hipDeviceGetByPCIBusId(&device, pciBusId);
            hipDeviceGetCacheConfig(&cacheConfig);
            hipDeviceSetCacheConfig(hipFuncCachePreferShared);
            hipDeviceGetSharedMemConfig(&sharedConfig);
            hipDeviceSetSharedMemConfig(hipSharedMemBankSizeFourByte);
            hipDeviceGetLimit(&limitValue, hipLimitMallocHeapSize);
            hipDeviceSetLimit(hipLimitMallocHeapSize, limitValue);
            hipDeviceReset();
            hipGetDeviceFlags(&flags);
            hipSetDeviceFlags(hipDeviceScheduleAuto);
            hipGetProcAddress(
                "hipMalloc", &proc, runtimeVersion, procFlags, &queryStatus
            );
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
            err = hipSetValidDevices(validDevices, 2);
            err = hipDeviceGetUuid(&uuid, device);
            err = hipDeviceGetPCIBusId(pciBusId, 32, device);
            err = hipDeviceSetLimit(hipLimitMallocHeapSize, limitValue);
            err = hipDeviceReset();
            err = hipGetProcAddress(
                "hipMalloc", &proc, runtimeVersion, procFlags, &queryStatus
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
        assert (
            result.count("// HIP set valid devices: devices: validDevices, count: 2")
            == 2
        )
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
        assert result.count("// HIP get device UUID: output: uuid, device: device") == 2
        assert "// HIP get device total memory: output: total, device: device" in result
        assert (
            "// HIP get device compute capability: major output: major, "
            "minor output: minor, device: device"
        ) in result
        assert "// HIP choose device: output: device, properties: (&props)" in result
        assert (
            result.count(
                "// HIP get device PCI bus id: output: pciBusId, "
                "length: 32, device: device"
            )
            == 2
        )
        assert (
            "// HIP get device by PCI bus id: output: device, bus id: pciBusId"
            in result
        )
        assert "// HIP get device cache config: output: cacheConfig" in result
        assert "// HIP set device cache config: hipFuncCachePreferShared" in result
        assert "// HIP get device shared memory config: output: sharedConfig" in result
        assert (
            "// HIP set device shared memory config: hipSharedMemBankSizeFourByte"
            in result
        )
        assert (
            "// HIP get device limit: output: limitValue, limit: hipLimitMallocHeapSize"
            in result
        )
        assert (
            result.count(
                "// HIP set device limit: limit: hipLimitMallocHeapSize, "
                "value: limitValue"
            )
            == 2
        )
        assert result.count("// HIP device reset") == 2
        assert "// HIP get device flags: output: flags" in result
        assert "// HIP set device flags: hipDeviceScheduleAuto" in result
        assert (
            result.count(
                '// HIP get proc address: symbol: "hipMalloc", output: proc, '
                "version: runtimeVersion, flags: procFlags, "
                "status output: queryStatus"
            )
            == 2
        )
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
        assert "hipSetValidDevices(" not in result
        assert "hipGetDeviceProperties(" not in result
        assert "hipDeviceGetAttribute(" not in result
        assert "hipDeviceGetName(" not in result
        assert "hipDeviceGetUuid(" not in result
        assert "hipDeviceTotalMem(" not in result
        assert "hipDeviceComputeCapability(" not in result
        assert "hipChooseDevice(" not in result
        assert "hipDeviceGetPCIBusId(" not in result
        assert "hipDeviceGetByPCIBusId(" not in result
        assert "hipDeviceGetCacheConfig(" not in result
        assert "hipDeviceSetCacheConfig(" not in result
        assert "hipDeviceGetSharedMemConfig(" not in result
        assert "hipDeviceSetSharedMemConfig(" not in result
        assert "hipDeviceGetLimit(" not in result
        assert "hipDeviceSetLimit(" not in result
        assert "hipDeviceReset(" not in result
        assert "hipGetDeviceFlags(" not in result
        assert "hipSetDeviceFlags(" not in result
        assert "hipGetProcAddress(" not in result
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

    def test_hip_stream_memory_operations_emit_metadata_and_status(self):
        """Test HIP stream memory operation APIs emit metadata comments."""
        code = """
        void stream_memory_ops(
            hipStream_t stream,
            unsigned int* ptr32,
            unsigned long long* ptr64,
            hipStreamBatchMemOpParams* params,
            bool retry
        ) {
            hipStreamWaitValue32(stream, ptr32, 7u, hipStreamWaitValueEq);
            hipStreamWriteValue32(stream, ptr32, 9u, 0);
            hipStreamWaitValue64(stream, ptr64, 11ull, hipStreamWaitValueGte);
            hipStreamWriteValue64(stream, ptr64, 13ull, 0);
            hipStreamBatchMemOp(stream, 2, params, 0);
            hipError_t err = hipStreamWaitValue32(stream, ptr32, 1u, 0);
            bool wrote = hipStreamWriteValue64(stream, ptr64, 2ull, 0) == hipSuccess;
            hipError_t selected = retry ? hipStreamBatchMemOp(stream, 2, params, 0) : hipStreamWriteValue32(stream, ptr32, 3u, 0);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// HIP stream wait value32: stream: stream, address: ptr32, "
            "value: 7u, flags: hipStreamWaitValueEq"
        ) in result
        assert (
            "// HIP stream write value32: stream: stream, address: ptr32, "
            "value: 9u, flags: 0"
        ) in result
        assert (
            "// HIP stream wait value64: stream: stream, address: ptr64, "
            "value: 11ull, flags: hipStreamWaitValueGte"
        ) in result
        assert (
            "// HIP stream write value64: stream: stream, address: ptr64, "
            "value: 13ull, flags: 0"
        ) in result
        assert (
            "// HIP stream batch memory op: stream: stream, count: 2, "
            "params: params, flags: 0"
        ) in result
        assert (
            "// HIP stream wait value32: stream: stream, address: ptr32, "
            "value: 1u, flags: 0"
        ) in result
        assert "var err: hipError_t = hipSuccess;" in result
        assert (
            "var wrote: bool = "
            "((/* HIP stream write value64: stream: stream, address: ptr64, "
            "value: 2ull, flags: 0 */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var selected: hipError_t = "
            "(retry ? (/* HIP stream batch memory op: stream: stream, count: 2, "
            "params: params, flags: 0 */ hipSuccess) : "
            "(/* HIP stream write value32: stream: stream, address: ptr32, "
            "value: 3u, flags: 0 */ hipSuccess));"
        ) in result
        assert "hipStreamWaitValue32(" not in result
        assert "hipStreamWaitValue64(" not in result
        assert "hipStreamWriteValue32(" not in result
        assert "hipStreamWriteValue64(" not in result
        assert "hipStreamBatchMemOp(" not in result

    def test_hip_runtime_module_api_conversion(self):
        """Test HIP module and function APIs emit metadata comments."""
        code = """
        void load_module(hipStream_t stream) {
            hipModule_t module;
            hipFunction_t function;
            hipFunction_t symbolFunction;
            hipFunction_t libraryFunction;
            hipFuncAttributes attrs;
            hipLibrary_t library;
            hipLibrary_t libraryFromKernel;
            hipKernel_t libraryKernel;
            hipKernel_t* libraryKernels;
            int attrValue;
            int device;
            unsigned int moduleFunctionCount;
            unsigned int libraryKernelCount;
            void* globalPtr;
            size_t globalBytes;
            size_t paramOffset;
            size_t paramSize;
            void** params;
            void** extra;
            void* launchParams;
            void* image;
            void* fatBinary;
            void* driverEntry;
            void* addressHandle;
            void* linkedImage;
            hipDeviceptr_t devicePtr;
            hipJitOption options;
            hipLibraryOption libraryOptions;
            void* optionValues;
            void* libraryOptionValues;
            hipEvent_t startEvent;
            hipEvent_t stopEvent;
            hipDriverEntryPointQueryResult driverStatus;
            textureReference* texRef;
            const char* kernelName;
            hipLinkState_t runtimeLinkState;
            hipJitInputType runtimeInputType;
            hiprtcProgram program;
            hiprtcLinkState linkState;
            hiprtcJIT_option jitOptions;
            hiprtcJITInputType inputType;
            const char* rtcSource;
            const char** rtcHeaders;
            const char** rtcIncludeNames;
            const char** rtcOptions;
            const char* loweredName;
            char* rtcLog;
            char* rtcCode;
            char* rtcBitcode;
            void* linkedBinary;
            size_t logSize;
            size_t codeSize;
            size_t bitcodeSize;
            int rtcMajor;
            int rtcMinor;
            hipModuleLoad(&module, "kernel.hsaco");
            hipModuleLoadData(&module, image);
            hipModuleLoadDataEx(&module, image, 1, &options, &optionValues);
            hipModuleLoadFatBinary(&module, fatBinary);
            hipModuleGetFunction(&function, module, "kernel");
            hipModuleGetFunctionCount(&moduleFunctionCount, module);
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
            hipModuleGetTexRef(&texRef, module, "tex");
            hipGetDriverEntryPoint("hipMalloc", &driverEntry, 0, &driverStatus);
            hipLibraryLoadFromFile(
                &library, "kernel.hipfb", &options, &optionValues, 1,
                &libraryOptions, &libraryOptionValues, 1
            );
            hipLibraryLoadData(
                &library, image, &options, &optionValues, 1, &libraryOptions,
                &libraryOptionValues, 1
            );
            hipLibraryGetKernel(&libraryKernel, library, "kernel");
            hipLibraryGetKernelCount(&libraryKernelCount, library);
            hipLibraryEnumerateKernels(libraryKernels, libraryKernelCount, library);
            hipKernelGetLibrary(&libraryFromKernel, libraryKernel);
            hipKernelGetName(&kernelName, libraryKernel);
            hipKernelGetParamInfo(libraryKernel, 0, &paramOffset, &paramSize);
            hipKernelGetFunction(&libraryFunction, libraryKernel);
            hipKernelGetAttribute(
                &attrValue, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, libraryKernel,
                device
            );
            hipKernelSetAttribute(
                HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 2048,
                libraryKernel, device
            );
            hipLinkCreate(1, &options, &optionValues, &runtimeLinkState);
            hipLinkAddFile(
                runtimeLinkState, runtimeInputType, "kernel.bc", 0, &options,
                &optionValues
            );
            hipLinkAddData(
                runtimeLinkState, runtimeInputType, rtcCode, codeSize, "kernel", 0,
                &options, &optionValues
            );
            hipLinkComplete(runtimeLinkState, &linkedImage, &globalBytes);
            hipLinkDestroy(runtimeLinkState);
            hipMemGetHandleForAddressRange(
                addressHandle, devicePtr, globalBytes, hipMemRangeHandleTypeDmaBuf, 0
            );
            hipLibraryUnload(library);
            hipModuleLaunchKernel(
                function, 8, 1, 1, 64, 1, 1, 0, stream, params, extra
            );
            hipExtModuleLaunchKernel(
                function, 512, 1, 1, 64, 1, 1, 0, stream, params, extra,
                startEvent, stopEvent
            );
            hipHccModuleLaunchKernel(
                function, 512, 1, 1, 64, 1, 1, 0, stream, params, extra,
                startEvent, stopEvent
            );
            hipLaunchCooperativeKernel(function, 8, 64, params, 0, stream);
            hipLaunchCooperativeKernelMultiDevice(launchParams, 1, 0);
            hipModuleLaunchCooperativeKernel(
                function, 8, 1, 1, 64, 1, 1, 0, stream, params
            );
            hipModuleLaunchCooperativeKernelMultiDevice(launchParams, 1, 0);
            hipModuleUnload(module);
            hiprtcVersion(&rtcMajor, &rtcMinor);
            hiprtcCreateProgram(
                &program, rtcSource, "kernel.hip", 0, rtcHeaders, rtcIncludeNames
            );
            hiprtcCompileProgram(program, 1, rtcOptions);
            hiprtcGetProgramLogSize(program, &logSize);
            hiprtcGetProgramLog(program, rtcLog);
            hiprtcGetCodeSize(program, &codeSize);
            hiprtcGetCode(program, rtcCode);
            hiprtcGetBitcodeSize(program, &bitcodeSize);
            hiprtcGetBitcode(program, rtcBitcode);
            hiprtcAddNameExpression(program, "kernel");
            hiprtcGetLoweredName(program, "kernel", &loweredName);
            hiprtcLinkCreate(1, &jitOptions, &optionValues, &linkState);
            hiprtcLinkAddFile(
                linkState, inputType, "kernel.bc", 0, &jitOptions, &optionValues
            );
            hiprtcLinkAddData(
                linkState, inputType, rtcCode, codeSize, "kernel", 0, &jitOptions,
                &optionValues
            );
            hiprtcLinkComplete(linkState, &linkedBinary, &globalBytes);
            hiprtcLinkDestroy(linkState);
            hiprtcDestroyProgram(&program);
            hipError_t err = hipModuleGetFunction(&function, module, "kernel");
            err = hipLibraryGetKernel(&libraryKernel, library, "kernel");
            err = hipModuleLaunchKernel(
                function, 8, 1, 1, 64, 1, 1, 0, stream, params, extra
            );
            err = hipExtModuleLaunchKernel(
                function, 512, 1, 1, 64, 1, 1, 0, stream, params, extra,
                startEvent, stopEvent
            );
            err = hipLaunchCooperativeKernel(function, 8, 64, params, 0, stream);
            hiprtcResult rtc = hiprtcCreateProgram(
                &program, rtcSource, "kernel.hip", 0, rtcHeaders, rtcIncludeNames
            );
            rtc = hiprtcCompileProgram(program, 1, rtcOptions);
            const char* rtcMessage = hiprtcGetErrorString(rtc);
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
            "// HIP module load fat binary: output: module, fat binary: fatBinary"
            in result
        )
        assert (
            result.count(
                "// HIP module get function: output: function, module: module, "
                'name: "kernel"'
            )
            == 2
        )
        assert (
            "// HIP module get function count: output: moduleFunctionCount, "
            "module: module"
        ) in result
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
            "// HIP module get texture reference: output: texRef, module: module, "
            'name: "tex"'
        ) in result
        assert (
            '// HIP get driver entry point: symbol: "hipMalloc", output: driverEntry, '
            "flags: 0, status output: driverStatus"
        ) in result
        assert (
            '// HIP library load: output: library, file: "kernel.hipfb", '
            "jit options: (&options), jit option values: (&optionValues), "
            "jit option count: 1, library options: (&libraryOptions), "
            "library option values: (&libraryOptionValues), library option count: 1"
        ) in result
        assert (
            "// HIP library load: output: library, code: image, "
            "jit options: (&options), jit option values: (&optionValues), "
            "jit option count: 1, library options: (&libraryOptions), "
            "library option values: (&libraryOptionValues), library option count: 1"
        ) in result
        assert (
            result.count(
                "// HIP library get kernel: output: libraryKernel, "
                'library: library, name: "kernel"'
            )
            == 2
        )
        assert (
            "// HIP library get kernel count: output: libraryKernelCount, "
            "library: library"
        ) in result
        assert (
            "// HIP library enumerate kernels: output: libraryKernels, "
            "max kernels: libraryKernelCount, library: library"
        ) in result
        assert (
            "// HIP kernel get library: output: libraryFromKernel, "
            "kernel: libraryKernel"
        ) in result
        assert (
            "// HIP kernel get name: output: kernelName, kernel: libraryKernel"
            in result
        )
        assert (
            "// HIP kernel get parameter info: kernel: libraryKernel, "
            "param index: 0, offset output: paramOffset, size output: paramSize"
        ) in result
        assert (
            "// HIP kernel get function: output: libraryFunction, "
            "kernel: libraryKernel"
        ) in result
        assert (
            "// HIP kernel get attribute: output: attrValue, "
            "attribute: HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, "
            "kernel: libraryKernel, device: device"
        ) in result
        assert (
            "// HIP kernel set attribute: "
            "attribute: HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, "
            "value: 2048, kernel: libraryKernel, device: device"
        ) in result
        assert (
            "// HIP link create: options: 1, option keys: (&options), "
            "option values: (&optionValues), state output: runtimeLinkState"
        ) in result
        assert (
            "// HIP link add file: state: runtimeLinkState, "
            'input type: runtimeInputType, path: "kernel.bc", options: 0, '
            "option keys: (&options), option values: (&optionValues)"
        ) in result
        assert (
            "// HIP link add data: state: runtimeLinkState, "
            "input type: runtimeInputType, image: rtcCode, bytes: codeSize, "
            'name: "kernel", options: 0, option keys: (&options), '
            "option values: (&optionValues)"
        ) in result
        assert (
            "// HIP link complete: state: runtimeLinkState, "
            "binary output: linkedImage, size output: globalBytes"
        ) in result
        assert "// HIP link destroy: state: runtimeLinkState" in result
        assert (
            "// HIP memory get handle for address range: output: addressHandle, "
            "device pointer: devicePtr, bytes: globalBytes, "
            "handle type: hipMemRangeHandleTypeDmaBuf, flags: 0"
        ) in result
        assert "// HIP library unload: library" in result
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
                "// HIP extended module launch kernel: function: function, "
                "global work size: (512, 1, 1), local work size: (64, 1, 1), "
                "shared memory: 0, stream: stream, params: params, extra: extra, "
                "start event: startEvent, stop event: stopEvent"
            )
            == 2
        )
        assert (
            "// HIP HCC module launch kernel: function: function, "
            "global work size: (512, 1, 1), local work size: (64, 1, 1), "
            "shared memory: 0, stream: stream, params: params, extra: extra, "
            "start event: startEvent, stop event: stopEvent"
        ) in result
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
        assert (
            "// HIPRTC version: major output: rtcMajor, minor output: rtcMinor"
            in result
        )
        assert (
            result.count(
                "// HIPRTC create program: output: program, source: rtcSource, "
                'name: "kernel.hip", headers: 0, header sources: rtcHeaders, '
                "include names: rtcIncludeNames"
            )
            == 2
        )
        assert (
            result.count(
                "// HIPRTC compile program: program: program, options: 1, "
                "option values: rtcOptions"
            )
            == 2
        )
        assert (
            "// HIPRTC get program log size: program: program, output: logSize"
            in result
        )
        assert "// HIPRTC get program log: program: program, output: rtcLog" in result
        assert "// HIPRTC get code size: program: program, output: codeSize" in result
        assert "// HIPRTC get code: program: program, output: rtcCode" in result
        assert (
            "// HIPRTC get bitcode size: program: program, output: bitcodeSize"
            in result
        )
        assert "// HIPRTC get bitcode: program: program, output: rtcBitcode" in result
        assert (
            '// HIPRTC add name expression: program: program, expression: "kernel"'
            in result
        )
        assert (
            '// HIPRTC get lowered name: program: program, expression: "kernel", '
            "output: loweredName"
        ) in result
        assert (
            "// HIPRTC link create: options: 1, option keys: (&jitOptions), "
            "option values: (&optionValues), state output: linkState"
        ) in result
        assert (
            "// HIPRTC link add file: state: linkState, input type: inputType, "
            'path: "kernel.bc", options: 0, option keys: (&jitOptions), '
            "option values: (&optionValues)"
        ) in result
        assert (
            "// HIPRTC link add data: state: linkState, input type: inputType, "
            'image: rtcCode, bytes: codeSize, name: "kernel", options: 0, '
            "option keys: (&jitOptions), option values: (&optionValues)"
        ) in result
        assert (
            "// HIPRTC link complete: state: linkState, binary output: linkedBinary, "
            "size output: globalBytes"
        ) in result
        assert "// HIPRTC link destroy: state: linkState" in result
        assert "// HIPRTC destroy program: output: program" in result
        assert "var err: hipError_t = hipSuccess;" in result
        assert "err = hipSuccess;" in result
        assert "var rtc: hiprtcResult = HIPRTC_SUCCESS;" in result
        assert "rtc = HIPRTC_SUCCESS;" in result
        assert 'var rtcMessage: ptr<i8> = /* HIPRTC error string: rtc */ "";' in result
        assert "hipModuleLoad(" not in result
        assert "hipModuleLoadData(" not in result
        assert "hipModuleLoadDataEx(" not in result
        assert "hipModuleLoadFatBinary(" not in result
        assert "hipModuleGetFunction(" not in result
        assert "hipModuleGetFunctionCount(" not in result
        assert "hipGetFuncBySymbol(" not in result
        assert "hipFuncGetAttribute(" not in result
        assert "hipFuncGetAttributes(" not in result
        assert "hipFuncSetAttribute(" not in result
        assert "hipFuncSetCacheConfig(" not in result
        assert "hipFuncSetSharedMemConfig(" not in result
        assert "hipModuleGetGlobal(" not in result
        assert "hipModuleGetTexRef(" not in result
        assert "hipGetDriverEntryPoint(" not in result
        assert "hipLibraryLoadFromFile(" not in result
        assert "hipLibraryLoadData(" not in result
        assert "hipLibraryGetKernel(" not in result
        assert "hipLibraryGetKernelCount(" not in result
        assert "hipLibraryEnumerateKernels(" not in result
        assert "hipKernelGetLibrary(" not in result
        assert "hipKernelGetName(" not in result
        assert "hipKernelGetParamInfo(" not in result
        assert "hipKernelGetFunction(" not in result
        assert "hipKernelGetAttribute(" not in result
        assert "hipKernelSetAttribute(" not in result
        assert "hipLinkCreate(" not in result
        assert "hipLinkAddFile(" not in result
        assert "hipLinkAddData(" not in result
        assert "hipLinkComplete(" not in result
        assert "hipLinkDestroy(" not in result
        assert "hipMemGetHandleForAddressRange(" not in result
        assert "hipLibraryUnload(" not in result
        assert "hipModuleLaunchKernel(" not in result
        assert "hipExtModuleLaunchKernel(" not in result
        assert "hipHccModuleLaunchKernel(" not in result
        assert "hipLaunchCooperativeKernel(" not in result
        assert "hipLaunchCooperativeKernelMultiDevice(" not in result
        assert "hipModuleLaunchCooperativeKernel(" not in result
        assert "hipModuleLaunchCooperativeKernelMultiDevice(" not in result
        assert "hipModuleUnload(" not in result
        assert "hiprtcVersion(" not in result
        assert "hiprtcCreateProgram(" not in result
        assert "hiprtcDestroyProgram(" not in result
        assert "hiprtcCompileProgram(" not in result
        assert "hiprtcGetProgramLogSize(" not in result
        assert "hiprtcGetProgramLog(" not in result
        assert "hiprtcGetCodeSize(" not in result
        assert "hiprtcGetCode(" not in result
        assert "hiprtcGetBitcodeSize(" not in result
        assert "hiprtcGetBitcode(" not in result
        assert "hiprtcAddNameExpression(" not in result
        assert "hiprtcGetLoweredName(" not in result
        assert "hiprtcGetErrorString(" not in result
        assert "hiprtcLinkCreate(" not in result
        assert "hiprtcLinkAddFile(" not in result
        assert "hiprtcLinkAddData(" not in result
        assert "hiprtcLinkComplete(" not in result
        assert "hiprtcLinkDestroy(" not in result

    def test_hip_module_library_link_rtc_expression_contexts_emit_status(self):
        """Test module, library, link, and HIPRTC expressions emit status metadata."""
        code = """
        hipError_t moduleLibraryLinkExpressions(hipStream_t stream, bool retry) {
            hipModule_t module;
            hipFunction_t function;
            hipFunction_t symbolFunction;
            hipFunction_t libraryFunction;
            hipFuncAttributes attrs;
            hipLibrary_t library;
            hipLibrary_t libraryFromKernel;
            hipKernel_t libraryKernel;
            hipKernel_t* libraryKernels;
            int attrValue;
            int device;
            unsigned int moduleFunctionCount;
            unsigned int libraryKernelCount;
            void* globalPtr;
            size_t globalBytes;
            size_t paramOffset;
            size_t paramSize;
            void** params;
            void** extra;
            void* image;
            void* fatBinary;
            void* driverEntry;
            void* addressHandle;
            void* linkedImage;
            hipDeviceptr_t devicePtr;
            hipJitOption options;
            hipLibraryOption libraryOptions;
            void* optionValues;
            void* libraryOptionValues;
            hipDriverEntryPointQueryResult driverStatus;
            textureReference* texRef;
            const char* kernelName;
            hipLinkState_t runtimeLinkState;
            hipJitInputType runtimeInputType;

            bool loadedFile = hipModuleLoad(&module, "kernel.hsaco") == hipSuccess;
            bool loadedData = hipModuleLoadData(&module, image) == hipSuccess;
            bool loadedDataEx =
                hipModuleLoadDataEx(
                    &module,
                    image,
                    1,
                    &options,
                    &optionValues
                ) == hipSuccess;
            bool loadedFat =
                hipModuleLoadFatBinary(&module, fatBinary) == hipSuccess;
            bool functionCountReady =
                hipModuleGetFunctionCount(&moduleFunctionCount, module) == hipSuccess;
            bool functionBySymbol =
                hipGetFuncBySymbol(&symbolFunction, function) == hipSuccess;
            bool functionAttribute =
                hipFuncGetAttribute(
                    &attrValue,
                    HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                    function
                ) == hipSuccess;
            bool functionAttributes =
                hipFuncGetAttributes(&attrs, function) == hipSuccess;
            bool functionAttributeSet =
                hipFuncSetAttribute(
                    function,
                    HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    1024
                ) == hipSuccess;
            bool functionCacheSet =
                hipFuncSetCacheConfig(function, hipFuncCachePreferL1) == hipSuccess;
            bool functionSharedSet =
                hipFuncSetSharedMemConfig(
                    function,
                    hipSharedMemBankSizeEightByte
                ) == hipSuccess;
            bool globalReady =
                hipModuleGetGlobal(
                    &globalPtr,
                    &globalBytes,
                    module,
                    "symbol"
                ) == hipSuccess;
            bool texReady = hipModuleGetTexRef(&texRef, module, "tex") == hipSuccess;
            bool entryReady =
                hipGetDriverEntryPoint(
                    "hipMalloc",
                    &driverEntry,
                    0,
                    &driverStatus
                ) == hipSuccess;
            bool libraryDataReady =
                hipLibraryLoadData(
                    &library,
                    image,
                    &options,
                    &optionValues,
                    1,
                    &libraryOptions,
                    &libraryOptionValues,
                    1
                ) == hipSuccess;
            bool kernelCountReady =
                hipLibraryGetKernelCount(&libraryKernelCount, library) == hipSuccess;
            bool kernelsEnumerated =
                hipLibraryEnumerateKernels(
                    libraryKernels,
                    libraryKernelCount,
                    library
                ) == hipSuccess;
            bool kernelLibraryReady =
                hipKernelGetLibrary(&libraryFromKernel, libraryKernel) == hipSuccess;
            bool kernelNameReady =
                hipKernelGetName(&kernelName, libraryKernel) == hipSuccess;
            bool kernelParamReady =
                hipKernelGetParamInfo(
                    libraryKernel,
                    0,
                    &paramOffset,
                    &paramSize
                ) == hipSuccess;
            bool kernelFunctionAttribute =
                hipKernelGetAttribute(
                    &attrValue,
                    HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                    libraryKernel,
                    device
                ) == hipSuccess;
            bool kernelAttributeSet =
                hipKernelSetAttribute(
                    HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    2048,
                    libraryKernel,
                    device
                ) == hipSuccess;
            bool linked =
                hipLinkComplete(
                    runtimeLinkState,
                    &linkedImage,
                    &globalBytes
                ) == hipSuccess;
            bool launched =
                hipModuleLaunchKernel(
                    function,
                    8,
                    1,
                    1,
                    64,
                    1,
                    1,
                    0,
                    stream,
                    params,
                    extra
                ) == hipSuccess;

            if (hipModuleGetFunction(&function, module, "kernel") != hipSuccess) {
                return hipModuleUnload(module);
            }
            if (hipLibraryLoadFromFile(&library, "kernel.hipfb", &options, &optionValues, 1, &libraryOptions, &libraryOptionValues, 1) == hipSuccess) {
                hipError_t selected = retry ? hipLibraryGetKernel(&libraryKernel, library, "kernel") : hipLibraryUnload(library);
                return selected;
            }
            if (hipLinkCreate(1, &options, &optionValues, &runtimeLinkState) == hipSuccess) {
                hipError_t selectedLink = retry ? hipLinkAddFile(runtimeLinkState, runtimeInputType, "kernel.bc", 0, &options, &optionValues) : hipLinkAddData(runtimeLinkState, runtimeInputType, image, globalBytes, "kernel", 0, &options, &optionValues);
                return selectedLink;
            }
            if (hipMemGetHandleForAddressRange(addressHandle, devicePtr, globalBytes, hipMemRangeHandleTypeDmaBuf, 0) == hipSuccess) {
                return hipLinkDestroy(runtimeLinkState);
            }
            return hipKernelGetFunction(&libraryFunction, libraryKernel);
        }

        hiprtcResult rtcExpressions(bool preferLog) {
            hiprtcProgram program;
            hiprtcLinkState linkState;
            hiprtcJIT_option jitOptions;
            hiprtcJITInputType inputType;
            const char* rtcSource;
            const char** rtcHeaders;
            const char** rtcIncludeNames;
            const char** rtcOptions;
            const char* loweredName;
            char* rtcLog;
            char* rtcCode;
            char* rtcBitcode;
            void* optionValues;
            void* linkedBinary;
            size_t globalBytes;
            size_t logSize;
            size_t codeSize;
            size_t bitcodeSize;
            int rtcMajor;
            int rtcMinor;

            bool versionReady = hiprtcVersion(&rtcMajor, &rtcMinor) == HIPRTC_SUCCESS;
            bool programReady =
                hiprtcCreateProgram(
                    &program,
                    rtcSource,
                    "kernel.hip",
                    0,
                    rtcHeaders,
                    rtcIncludeNames
                ) == HIPRTC_SUCCESS;
            bool logSized =
                hiprtcGetProgramLogSize(program, &logSize) == HIPRTC_SUCCESS;
            bool codeSized = hiprtcGetCodeSize(program, &codeSize) == HIPRTC_SUCCESS;
            bool bitcodeSized =
                hiprtcGetBitcodeSize(program, &bitcodeSize) == HIPRTC_SUCCESS;
            bool named =
                hiprtcAddNameExpression(program, "kernel") == HIPRTC_SUCCESS;

            if (hiprtcCompileProgram(program, 1, rtcOptions) != HIPRTC_SUCCESS) {
                return hiprtcDestroyProgram(&program);
            }
            if (hiprtcGetLoweredName(program, "kernel", &loweredName) == HIPRTC_SUCCESS) {
                hiprtcResult selectedRtc = preferLog ? hiprtcGetProgramLog(program, rtcLog) : hiprtcGetCode(program, rtcCode);
                return selectedRtc;
            }
            if (hiprtcLinkCreate(1, &jitOptions, &optionValues, &linkState) == HIPRTC_SUCCESS) {
                hiprtcResult selectedLink = preferLog ? hiprtcLinkAddFile(linkState, inputType, "kernel.bc", 0, &jitOptions, &optionValues) : hiprtcLinkAddData(linkState, inputType, rtcCode, codeSize, "kernel", 0, &jitOptions, &optionValues);
                return selectedLink;
            }
            if (hiprtcLinkComplete(linkState, &linkedBinary, &globalBytes) == HIPRTC_SUCCESS) {
                return hiprtcLinkDestroy(linkState);
            }
            return hiprtcGetBitcode(program, rtcBitcode);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "var loadedFile: bool = ((/* HIP module load: output: module, "
            'file: "kernel.hsaco" */ hipSuccess) == hipSuccess);'
        ) in result
        assert (
            "var functionBySymbol: bool = ((/* HIP get function by symbol: "
            "output: symbolFunction, symbol: function */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var entryReady: bool = ((/* HIP get driver entry point: symbol: "
            '"hipMalloc", output: driverEntry, flags: 0, status output: '
            "driverStatus */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var kernelNameReady: bool = ((/* HIP kernel get name: output: "
            "kernelName, kernel: libraryKernel */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var launched: bool = ((/* HIP module launch kernel: "
            "function: function, grid: (8, 1, 1), block: (64, 1, 1), "
            "shared memory: 0, stream: stream, params: params, extra: extra */ "
            "hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP module get function: output: function, module: module, "
            'name: "kernel" */ hipSuccess) != hipSuccess))'
        ) in result
        assert "return (/* HIP module unload: module */ hipSuccess);" in result
        assert (
            "var selected: hipError_t = (retry ? "
            "(/* HIP library get kernel: output: libraryKernel, library: library, "
            'name: "kernel" */ hipSuccess) : '
            "(/* HIP library unload: library */ hipSuccess));"
        ) in result
        assert (
            "var selectedLink: hipError_t = (retry ? "
            "(/* HIP link add file: state: runtimeLinkState, input type: "
            'runtimeInputType, path: "kernel.bc", options: 0, option keys: '
            "(&options), option values: (&optionValues) */ hipSuccess) : "
            "(/* HIP link add data: state: runtimeLinkState, input type: "
            'runtimeInputType, image: image, bytes: globalBytes, name: "kernel", '
            "options: 0, option keys: (&options), option values: "
            "(&optionValues) */ hipSuccess));"
        ) in result
        assert (
            "return (/* HIP kernel get function: output: libraryFunction, "
            "kernel: libraryKernel */ hipSuccess);"
        ) in result

        assert (
            "var versionReady: bool = ((/* HIPRTC version: major output: "
            "rtcMajor, minor output: rtcMinor */ HIPRTC_SUCCESS) == HIPRTC_SUCCESS);"
        ) in result
        assert (
            "var named: bool = ((/* HIPRTC add name expression: program: "
            'program, expression: "kernel" */ HIPRTC_SUCCESS) == HIPRTC_SUCCESS);'
        ) in result
        assert (
            "if (((/* HIPRTC compile program: program: program, options: 1, "
            "option values: rtcOptions */ HIPRTC_SUCCESS) != HIPRTC_SUCCESS))"
        ) in result
        assert (
            "return (/* HIPRTC destroy program: output: program */ " "HIPRTC_SUCCESS);"
        ) in result
        assert (
            "var selectedRtc: hiprtcResult = (preferLog ? "
            "(/* HIPRTC get program log: program: program, output: rtcLog */ "
            "HIPRTC_SUCCESS) : (/* HIPRTC get code: program: program, output: "
            "rtcCode */ HIPRTC_SUCCESS));"
        ) in result
        assert (
            "var selectedLink: hiprtcResult = (preferLog ? "
            "(/* HIPRTC link add file: state: linkState, input type: inputType, "
            'path: "kernel.bc", options: 0, option keys: (&jitOptions), option '
            "values: (&optionValues) */ HIPRTC_SUCCESS) : "
            "(/* HIPRTC link add data: state: linkState, input type: inputType, "
            'image: rtcCode, bytes: codeSize, name: "kernel", options: 0, '
            "option keys: (&jitOptions), option values: (&optionValues) */ "
            "HIPRTC_SUCCESS));"
        ) in result
        assert (
            "return (/* HIPRTC get bitcode: program: program, output: "
            "rtcBitcode */ HIPRTC_SUCCESS);"
        ) in result

        for function_name in [
            "hipModuleLoad",
            "hipModuleLoadData",
            "hipModuleLoadDataEx",
            "hipModuleLoadFatBinary",
            "hipModuleGetFunction",
            "hipModuleGetFunctionCount",
            "hipGetFuncBySymbol",
            "hipFuncGetAttribute",
            "hipFuncGetAttributes",
            "hipFuncSetAttribute",
            "hipFuncSetCacheConfig",
            "hipFuncSetSharedMemConfig",
            "hipModuleGetGlobal",
            "hipModuleGetTexRef",
            "hipGetDriverEntryPoint",
            "hipLibraryLoadFromFile",
            "hipLibraryLoadData",
            "hipLibraryGetKernel",
            "hipLibraryGetKernelCount",
            "hipLibraryEnumerateKernels",
            "hipKernelGetLibrary",
            "hipKernelGetName",
            "hipKernelGetParamInfo",
            "hipKernelGetFunction",
            "hipKernelGetAttribute",
            "hipKernelSetAttribute",
            "hipLinkCreate",
            "hipLinkAddFile",
            "hipLinkAddData",
            "hipLinkComplete",
            "hipLinkDestroy",
            "hipMemGetHandleForAddressRange",
            "hipLibraryUnload",
            "hipModuleLaunchKernel",
            "hipModuleUnload",
            "hiprtcVersion",
            "hiprtcCreateProgram",
            "hiprtcDestroyProgram",
            "hiprtcCompileProgram",
            "hiprtcGetProgramLogSize",
            "hiprtcGetProgramLog",
            "hiprtcGetCodeSize",
            "hiprtcGetCode",
            "hiprtcGetBitcodeSize",
            "hiprtcGetBitcode",
            "hiprtcAddNameExpression",
            "hiprtcGetLoweredName",
            "hiprtcLinkCreate",
            "hiprtcLinkAddFile",
            "hiprtcLinkAddData",
            "hiprtcLinkComplete",
            "hiprtcLinkDestroy",
        ]:
            assert f"{function_name}(" not in result

    def test_hip_module_library_handle_outputs_clear_stale_metadata(self):
        """Test raw module/library handle outputs clear prior query metadata."""
        code = """
        void queryModuleHandles(
            hipModule_t module,
            const void* symbol,
            hipLibrary_t library,
            hipKernel_t kernel,
            hipDeviceptr_t devicePtr,
            void** ptrs
        ) {
            void* rawHandle = 0;
            size_t rangeSize = 0;
            hipMemGetAddressRange(&rawHandle, &rangeSize, devicePtr);
            hipModuleGetFunction(
                (void**)&rawHandle,
                module,
                "module_kernel"
            );
            ptrs[0] = rawHandle;
            hipMemGetAddressRange(&rawHandle, &rangeSize, devicePtr);
            hipGetFuncBySymbol((void**)&rawHandle, symbol);
            ptrs[1] = rawHandle;
            hipMemGetAddressRange(&rawHandle, &rangeSize, devicePtr);
            hipLibraryGetKernel(
                (void**)&rawHandle,
                library,
                "library_kernel"
            );
            ptrs[2] = rawHandle;
            hipMemGetAddressRange(&rawHandle, &rangeSize, devicePtr);
            hipKernelGetFunction((void**)&rawHandle, kernel);
            ptrs[3] = rawHandle;
            hipMemGetAddressRange(&rawHandle, &rangeSize, devicePtr);
            hipKernelGetLibrary((void**)&rawHandle, kernel);
            ptrs[4] = rawHandle;
            hipError_t err =
                hipModuleGetFunction(
                    (void**)&rawHandle,
                    module,
                    "status_kernel"
                );
            ptrs[5] = rawHandle;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "// HIP module get function: output: rawHandle, module: module, "
            'name: "module_kernel"'
        ) in result
        assert (
            "// HIP get function by symbol: output: rawHandle, symbol: symbol" in result
        )
        assert (
            "// HIP library get kernel: output: rawHandle, library: library, "
            'name: "library_kernel"'
        ) in result
        assert "// HIP kernel get function: output: rawHandle, kernel: kernel" in result
        assert "// HIP kernel get library: output: rawHandle, kernel: kernel" in result
        assert "var err: hipError_t = hipSuccess;" in result
        for index in range(6):
            assert f"ptrs[{index}] = rawHandle;" in result
        assert "ptrs[0] = (/* HIP device query:" not in result
        assert "memory.addressRange.base(devicePtr)" not in result

    def test_hip_module_library_kernel_scalar_outputs_replace_stale_metadata(self):
        """Test scalar module/library/kernel outputs replace prior query metadata."""
        code = """
        void queryModuleKernelScalars(
            hipModule_t module,
            hipLibrary_t library,
            hipKernel_t kernel,
            hipDeviceptr_t devicePtr,
            unsigned int* counts,
            int* values,
            size_t* sizes,
            const char** names
        ) {
            unsigned int moduleCount = 0;
            unsigned int kernelCount = 0;
            int attributeValue = 0;
            size_t paramOffset = 0;
            size_t paramSize = 0;
            const char* kernelName = 0;

            hipPointerGetAttribute(
                &moduleCount,
                hipPointerAttributeMemoryType,
                devicePtr
            );
            hipModuleGetFunctionCount(&moduleCount, module);
            counts[0] = moduleCount;

            hipPointerGetAttribute(
                &kernelCount,
                hipPointerAttributeMemoryType,
                devicePtr
            );
            hipLibraryGetKernelCount(&kernelCount, library);
            counts[1] = kernelCount;

            hipPointerGetAttribute(
                &attributeValue,
                hipPointerAttributeMemoryType,
                devicePtr
            );
            hipKernelGetAttribute(
                &attributeValue,
                HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                kernel,
                0
            );
            values[0] = attributeValue;

            hipMemGetAddressRange((void**)&kernelName, &paramSize, devicePtr);
            hipKernelGetName(&kernelName, kernel);
            names[0] = kernelName;

            hipGetSymbolSize(&paramOffset, kernelName);
            hipGetSymbolSize(&paramSize, kernelName);
            hipKernelGetParamInfo(kernel, 0, &paramOffset, &paramSize);
            sizes[0] = paramOffset;
            sizes[1] = paramSize;

            if (hipModuleGetFunctionCount(&moduleCount, module) == hipSuccess) {
                counts[2] = moduleCount;
            }
            if (hipKernelGetParamInfo(kernel, 1, &paramOffset, &paramSize) == hipSuccess) {
                sizes[2] = paramOffset;
                sizes[3] = paramSize;
            }
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "counts[0] = (/* HIP device query: module.functionCount(module) */ 0);"
            in result
        )
        assert (
            "counts[1] = (/* HIP device query: library.kernelCount(library) */ 0);"
            in result
        )
        assert (
            "values[0] = (/* HIP device query: "
            "kernel.attribute(HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, "
            "kernel, 0) */ 0);"
        ) in result
        assert "names[0] = kernelName;" in result
        assert (
            "sizes[0] = (/* HIP device query: " "kernel.param.offset(kernel, 0) */ 0);"
        ) in result
        assert (
            "sizes[1] = (/* HIP device query: " "kernel.param.size(kernel, 0) */ 0);"
        ) in result
        assert (
            "counts[2] = (/* HIP device query: module.functionCount(module) */ 0);"
            in result
        )
        assert (
            "sizes[2] = (/* HIP device query: " "kernel.param.offset(kernel, 1) */ 0);"
        ) in result
        assert (
            "sizes[3] = (/* HIP device query: " "kernel.param.size(kernel, 1) */ 0);"
        ) in result
        assert "counts[0] = (/* HIP device query: pointer.attribute" not in result
        assert "values[0] = (/* HIP device query: pointer.attribute" not in result
        assert "names[0] = (/* HIP device query:" not in result
        assert "symbol.size(kernelName)" not in result
        assert "memory.addressRange.base(devicePtr)" not in result

    def test_hip_module_global_size_output_metadata_replaces_symbol_size(self):
        """Test module global size outputs replace stale symbol-size metadata."""
        code = """
        void queryModuleGlobals(
            hipModule_t module,
            const void* symbol,
            size_t* sizes,
            void** ptrs,
            textureReference** refs
        ) {
            hipDeviceptr_t globalPtr;
            size_t globalBytes = 0;
            textureReference* moduleRef;
            textureReference* symbolRef;
            hipGetSymbolSize(&globalBytes, symbol);
            hipModuleGetGlobal(&globalPtr, &globalBytes, module, "device_symbol");
            ptrs[0] = globalPtr;
            sizes[0] = globalBytes;
            hipModuleGetGlobal(&globalPtr, &globalBytes, module, "other_symbol");
            sizes[1] = globalBytes;
            hipError_t err =
                hipModuleGetGlobal(
                    &globalPtr,
                    &globalBytes,
                    module,
                    "status_symbol"
                );
            sizes[2] = globalBytes;
            hipModuleGetTexRef(&moduleRef, module, "tex");
            refs[0] = moduleRef;
            hipGetTextureReference(&symbolRef, symbol);
            refs[1] = symbolRef;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "// HIP module get global: pointer output: globalPtr, "
            'size output: globalBytes, module: module, name: "device_symbol"'
        ) in result
        assert "ptrs[0] = globalPtr;" in result
        assert (
            "sizes[0] = (/* HIP device query: "
            'module.global.size(module, "device_symbol") */ 0);'
        ) in result
        assert (
            "sizes[1] = (/* HIP device query: "
            'module.global.size(module, "other_symbol") */ 0);'
        ) in result
        assert "var err: hipError_t = hipSuccess;" in result
        assert (
            "sizes[2] = (/* HIP device query: "
            'module.global.size(module, "status_symbol") */ 0);'
        ) in result
        assert (
            "// HIP module get texture reference: output: moduleRef, "
            'module: module, name: "tex"'
        ) in result
        assert "refs[0] = moduleRef;" in result
        assert (
            "// HIP get texture reference: output: symbolRef, symbol: symbol" in result
        )
        assert "refs[1] = symbolRef;" in result
        assert (
            "sizes[0] = (/* HIP device query: symbol.size(symbol) */ 0);" not in result
        )
        assert "textureReference.moduleRef" not in result
        assert "textureReference.symbolRef" not in result

    def test_hip_texture_reference_expression_contexts_emit_status(self):
        """Test deprecated texture-reference helpers in expressions stay explicit."""
        code = """
        hipError_t acceptStatus(hipError_t status) {
            return status;
        }

        hipError_t textureReferenceExpressions(
            textureReference* texRef,
            hipArray_t array,
            hipMipmappedArray_t mipArray,
            void* devicePtr,
            hipChannelFormatDesc* desc,
            void* symbol,
            bool useMip
        ) {
            size_t offset;
            textureReference* symbolRef;
            bool boundLinear =
                hipBindTexture(&offset, texRef, devicePtr, desc, 256) == hipSuccess;
            bool bound2D =
                hipBindTexture2D(
                    &offset,
                    texRef,
                    devicePtr,
                    desc,
                    16,
                    8,
                    64
                ) == hipSuccess;
            bool gotReference =
                hipGetTextureReference(&symbolRef, symbol) == hipSuccess;
            bool aligned =
                hipGetTextureAlignmentOffset(&offset, texRef) == hipSuccess;
            if (hipBindTextureToArray(texRef, array, desc) != hipSuccess) {
                return hipUnbindTexture(texRef);
            }
            hipError_t selected = useMip ? hipBindTextureToMipmappedArray(texRef, mipArray, desc) : hipBindTexture(&offset, texRef, devicePtr, desc, 512);
            return acceptStatus(hipGetTextureAlignmentOffset(&offset, texRef));
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "var boundLinear: bool = ((/* HIP texture reference bind: "
            "offset output: offset, texture: texRef, pointer: devicePtr, "
            "desc: desc, bytes: 256 */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var bound2D: bool = ((/* HIP texture reference bind 2D: "
            "offset output: offset, texture: texRef, pointer: devicePtr, "
            "desc: desc, width: 16, height: 8, pitch: 64 */ hipSuccess) "
            "== hipSuccess);"
        ) in result
        assert (
            "var gotReference: bool = ((/* HIP get texture reference: "
            "output: symbolRef, symbol: symbol */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var aligned: bool = ((/* HIP texture alignment offset query: "
            "output: offset, texture: texRef */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP texture reference bind array: texture: texRef, "
            "array: array, desc: desc */ hipSuccess) != hipSuccess))"
        ) in result
        assert (
            "return (/* HIP texture reference unbind: texRef */ hipSuccess);" in result
        )
        assert (
            "var selected: hipError_t = (useMip ? "
            "(/* HIP texture reference bind mipmapped array: texture: texRef, "
            "mipmapped array: mipArray, desc: desc */ hipSuccess) : "
            "(/* HIP texture reference bind: offset output: offset, texture: "
            "texRef, pointer: devicePtr, desc: desc, bytes: 512 */ hipSuccess));"
        ) in result
        assert (
            "return acceptStatus((/* HIP texture alignment offset query: "
            "output: offset, texture: texRef */ hipSuccess));"
        ) in result

        for function_name in [
            "hipBindTexture",
            "hipBindTexture2D",
            "hipBindTextureToArray",
            "hipBindTextureToMipmappedArray",
            "hipGetTextureReference",
            "hipGetTextureAlignmentOffset",
            "hipUnbindTexture",
        ]:
            assert f"{function_name}(" not in result

    def test_hip_texture_reference_get_set_expression_contexts_emit_status(self):
        """Test deprecated texture-reference getter/setters stay explicit."""
        code = """
        hipError_t acceptStatus(hipError_t status) {
            return status;
        }

        hipError_t textureReferenceGetSetExpressions(
            textureReference* texRef,
            hipArray_t array,
            hipMipmappedArray_t mipArray,
            hipDeviceptr_t devicePtr,
            HIP_ARRAY_DESCRIPTOR* arrayDesc,
            float* borderColor,
            bool useMip
        ) {
            size_t byteOffset;
            hipDeviceptr_t address;
            hipTextureAddressMode addressMode;
            hipTextureFilterMode filterMode;
            unsigned int flags;
            hipArray_Format format;
            int channels;
            int maxAniso;
            float bias;
            float minClamp;
            float maxClamp;
            hipArray_t gotArray;
            hipMipmappedArray_t gotMipArray;
            bool gotAddress =
                hipTexRefGetAddress(&address, texRef) == hipSuccess;
            bool gotMode =
                hipTexRefGetAddressMode(&addressMode, texRef, 1) == hipSuccess;
            bool gotArrayStatus =
                hipTexRefGetArray(&gotArray, texRef) == hipSuccess;
            bool gotBorder =
                hipTexRefGetBorderColor(borderColor, texRef) == hipSuccess;
            bool gotFilter =
                hipTexRefGetFilterMode(&filterMode, texRef) == hipSuccess;
            bool gotFlags =
                hipTexRefGetFlags(&flags, texRef) == hipSuccess;
            if (hipTexRefGetFormat(&format, &channels, texRef) != hipSuccess) {
                return hipTexRefSetAddressMode(texRef, 0, addressMode);
            }
            hipError_t addressStatus =
                hipTexRefSetAddress(&byteOffset, texRef, devicePtr, 1024);
            bool address2D =
                hipTexRefSetAddress2D(
                    texRef,
                    arrayDesc,
                    devicePtr,
                    128
                ) == hipSuccess;
            bool setArray =
                hipTexRefSetArray(texRef, array, flags) == hipSuccess;
            bool setBorder =
                hipTexRefSetBorderColor(texRef, borderColor) == hipSuccess;
            bool setFilter =
                hipTexRefSetFilterMode(texRef, filterMode) == hipSuccess;
            bool setFlags =
                hipTexRefSetFlags(texRef, flags) == hipSuccess;
            bool setFormat =
                hipTexRefSetFormat(texRef, format, channels) == hipSuccess;
            bool setAniso =
                hipTexRefSetMaxAnisotropy(texRef, maxAniso) == hipSuccess;
            hipError_t mipFilter = useMip ? hipTexRefSetMipmapFilterMode(texRef, filterMode) : hipTexRefGetMipmapFilterMode(&filterMode, texRef);
            hipError_t mipBias = useMip ? hipTexRefSetMipmapLevelBias(texRef, bias) : hipTexRefGetMipmapLevelBias(&bias, texRef);
            hipError_t mipClamp = useMip ? hipTexRefSetMipmapLevelClamp(texRef, minClamp, maxClamp) : hipTexRefGetMipmapLevelClamp(&minClamp, &maxClamp, texRef);
            hipError_t mipArrayStatus = useMip ? hipTexRefSetMipmappedArray(texRef, mipArray, flags) : hipTexRefGetMipMappedArray(&gotMipArray, texRef);
            return acceptStatus(useMip ? hipTexRefGetMaxAnisotropy(&maxAniso, texRef) : hipTexRefSetAddress(&byteOffset, texRef, devicePtr, 2048));
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        expected_snippets = [
            (
                "var gotAddress: bool = ((/* HIP texture reference get address: "
                "output: address, texture: texRef */ hipSuccess) == hipSuccess);"
            ),
            (
                "var gotMode: bool = ((/* HIP texture reference get address mode: "
                "output: addressMode, texture: texRef, dim: 1 */ hipSuccess) "
                "== hipSuccess);"
            ),
            (
                "var gotArrayStatus: bool = ((/* HIP texture reference get array: "
                "output: gotArray, texture: texRef */ hipSuccess) == hipSuccess);"
            ),
            (
                "var gotBorder: bool = ((/* HIP texture reference get border color: "
                "output: borderColor, texture: texRef */ hipSuccess) == hipSuccess);"
            ),
            (
                "var gotFilter: bool = ((/* HIP texture reference get filter mode: "
                "output: filterMode, texture: texRef */ hipSuccess) == hipSuccess);"
            ),
            (
                "var gotFlags: bool = ((/* HIP texture reference get flags: "
                "output: flags, texture: texRef */ hipSuccess) == hipSuccess);"
            ),
            (
                "if (((/* HIP texture reference get format: format output: format, "
                "channels output: channels, texture: texRef */ hipSuccess) "
                "!= hipSuccess))"
            ),
            (
                "return (/* HIP texture reference set address mode: "
                "texture: texRef, dim: 0, mode: addressMode */ hipSuccess);"
            ),
            (
                "// HIP texture reference set address: offset output: byteOffset, "
                "texture: texRef, pointer: devicePtr, bytes: 1024"
            ),
            "var addressStatus: hipError_t = hipSuccess;",
            (
                "var address2D: bool = ((/* HIP texture reference set address 2D: "
                "texture: texRef, desc: arrayDesc, pointer: devicePtr, pitch: 128 */ "
                "hipSuccess) == hipSuccess);"
            ),
            (
                "var setArray: bool = ((/* HIP texture reference set array: "
                "texture: texRef, array: array, flags: flags */ hipSuccess) "
                "== hipSuccess);"
            ),
            (
                "var setBorder: bool = ((/* HIP texture reference set border color: "
                "texture: texRef, color: borderColor */ hipSuccess) == hipSuccess);"
            ),
            (
                "var setFilter: bool = ((/* HIP texture reference set filter mode: "
                "texture: texRef, mode: filterMode */ hipSuccess) == hipSuccess);"
            ),
            (
                "var setFlags: bool = ((/* HIP texture reference set flags: "
                "texture: texRef, flags: flags */ hipSuccess) == hipSuccess);"
            ),
            (
                "var setFormat: bool = ((/* HIP texture reference set format: "
                "texture: texRef, format: format, components: channels */ hipSuccess) "
                "== hipSuccess);"
            ),
            (
                "var setAniso: bool = ((/* HIP texture reference set max anisotropy: "
                "texture: texRef, value: maxAniso */ hipSuccess) == hipSuccess);"
            ),
            (
                "var mipFilter: hipError_t = (useMip ? "
                "(/* HIP texture reference set mipmap filter mode: texture: texRef, "
                "mode: filterMode */ hipSuccess) : "
                "(/* HIP texture reference get mipmap filter mode: "
                "output: filterMode, texture: texRef */ hipSuccess));"
            ),
            (
                "var mipBias: hipError_t = (useMip ? "
                "(/* HIP texture reference set mipmap level bias: texture: texRef, "
                "bias: bias */ hipSuccess) : "
                "(/* HIP texture reference get mipmap level bias: output: bias, "
                "texture: texRef */ hipSuccess));"
            ),
            (
                "var mipClamp: hipError_t = (useMip ? "
                "(/* HIP texture reference set mipmap level clamp: texture: texRef, "
                "min: minClamp, max: maxClamp */ hipSuccess) : "
                "(/* HIP texture reference get mipmap level clamp: min output: "
                "minClamp, max output: maxClamp, texture: texRef */ hipSuccess));"
            ),
            (
                "var mipArrayStatus: hipError_t = (useMip ? "
                "(/* HIP texture reference set mipmapped array: texture: texRef, "
                "mipmapped array: mipArray, flags: flags */ hipSuccess) : "
                "(/* HIP texture reference get mipmapped array: output: "
                "gotMipArray, texture: texRef */ hipSuccess));"
            ),
            (
                "return acceptStatus((useMip ? "
                "(/* HIP texture reference get max anisotropy: output: maxAniso, "
                "texture: texRef */ hipSuccess) : "
                "(/* HIP texture reference set address: offset output: byteOffset, "
                "texture: texRef, pointer: devicePtr, bytes: 2048 */ hipSuccess)));"
            ),
        ]
        for snippet in expected_snippets:
            assert snippet in result

        for function_name in [
            "hipTexRefGetAddress",
            "hipTexRefGetAddressMode",
            "hipTexRefGetArray",
            "hipTexRefGetBorderColor",
            "hipTexRefGetFilterMode",
            "hipTexRefGetFlags",
            "hipTexRefGetFormat",
            "hipTexRefGetMaxAnisotropy",
            "hipTexRefGetMipmapFilterMode",
            "hipTexRefGetMipmapLevelBias",
            "hipTexRefGetMipmapLevelClamp",
            "hipTexRefGetMipMappedArray",
            "hipTexRefSetAddress",
            "hipTexRefSetAddress2D",
            "hipTexRefSetAddressMode",
            "hipTexRefSetArray",
            "hipTexRefSetBorderColor",
            "hipTexRefSetFilterMode",
            "hipTexRefSetFlags",
            "hipTexRefSetFormat",
            "hipTexRefSetMaxAnisotropy",
            "hipTexRefSetMipmapFilterMode",
            "hipTexRefSetMipmapLevelBias",
            "hipTexRefSetMipmapLevelClamp",
            "hipTexRefSetMipmappedArray",
        ]:
            assert f"{function_name}(" not in result

    def test_hip_texture_reference_output_reads_emit_metadata_expressions(self):
        """Test texture-reference scalar getter outputs lower to metadata."""
        code = """
        void queryTextureReference(
            textureReference* texRef,
            int* ints,
            unsigned int* flagsOut,
            float* floats,
            size_t* sizes
        ) {
            size_t alignment = 0;
            size_t statusAlignment = 0;
            hipTextureAddressMode addressMode;
            hipTextureFilterMode filterMode;
            hipTextureFilterMode mipFilterMode;
            unsigned int flags = 0;
            unsigned int statusFlags = 0;
            int maxAniso = 0;
            float mipBias = 0.0f;
            hipArray_Format format;
            int channels = 0;
            float minClamp = 0.0f;
            float maxClamp = 0.0f;
            float statusMinClamp = 0.0f;
            float statusMaxClamp = 0.0f;
            hipGetTextureAlignmentOffset(&alignment, texRef);
            sizes[0] = alignment;
            alignment = 4;
            size_t manualAlignment = alignment;
            hipTexRefGetAddressMode(&addressMode, texRef, 1);
            ints[0] = addressMode;
            addressMode = hipAddressModeClamp;
            hipTextureAddressMode manualAddressMode = addressMode;
            hipTexRefGetFilterMode(&filterMode, texRef);
            ints[1] = filterMode;
            hipTexRefGetFlags(&flags, texRef);
            flagsOut[0] = flags;
            flags = 7;
            unsigned int manualFlags = flags;
            hipTexRefGetMaxAnisotropy(&maxAniso, texRef);
            ints[2] = maxAniso;
            hipTexRefGetMipmapFilterMode(&mipFilterMode, texRef);
            ints[3] = mipFilterMode;
            hipTexRefGetMipmapLevelBias(&mipBias, texRef);
            floats[0] = mipBias;
            hipTexRefGetFormat(&format, &channels, texRef);
            ints[4] = format;
            ints[5] = channels;
            hipTexRefGetMipmapLevelClamp(&minClamp, &maxClamp, texRef);
            floats[1] = minClamp;
            floats[2] = maxClamp;
            hipError_t errAlignment =
                hipGetTextureAlignmentOffset(&statusAlignment, texRef);
            sizes[1] = statusAlignment;
            hipError_t errFlags = hipTexRefGetFlags(&statusFlags, texRef);
            flagsOut[1] = statusFlags;
            hipError_t errClamp =
                hipTexRefGetMipmapLevelClamp(
                    &statusMinClamp,
                    &statusMaxClamp,
                    texRef
                );
            floats[3] = statusMinClamp;
            floats[4] = statusMaxClamp;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "sizes[0] = "
            "(/* HIP device query: textureReference.alignmentOffset(texRef) */ 0);"
            in result
        )
        assert "alignment = 4;" in result
        assert "var manualAlignment: u32 = alignment;" in result
        assert (
            "ints[0] = (/* HIP device query: "
            "textureReference.addressMode(texRef, 1) */ 0);"
        ) in result
        assert "var manualAddressMode: hipTextureAddressMode = addressMode;" in result
        assert (
            "ints[1] = (/* HIP device query: "
            "textureReference.filterMode(texRef) */ 0);"
        ) in result
        assert (
            "flagsOut[0] = (/* HIP device query: "
            "textureReference.flags(texRef) */ 0);"
        ) in result
        assert "flags = 7;" in result
        assert "var manualFlags: u32 = flags;" in result
        assert (
            "ints[2] = (/* HIP device query: "
            "textureReference.maxAnisotropy(texRef) */ 0);"
        ) in result
        assert (
            "ints[3] = (/* HIP device query: "
            "textureReference.mipmapFilterMode(texRef) */ 0);"
        ) in result
        assert (
            "floats[0] = (/* HIP device query: "
            "textureReference.mipmapLevelBias(texRef) */ 0);"
        ) in result
        assert (
            "ints[4] = (/* HIP device query: " "textureReference.format(texRef) */ 0);"
        ) in result
        assert (
            "ints[5] = (/* HIP device query: "
            "textureReference.channelCount(texRef) */ 0);"
        ) in result
        assert (
            "floats[1] = (/* HIP device query: "
            "textureReference.mipmapLevelClamp.min(texRef) */ 0);"
        ) in result
        assert (
            "floats[2] = (/* HIP device query: "
            "textureReference.mipmapLevelClamp.max(texRef) */ 0);"
        ) in result
        assert "var errAlignment: hipError_t = hipSuccess;" in result
        assert (
            "sizes[1] = "
            "(/* HIP device query: textureReference.alignmentOffset(texRef) */ 0);"
            in result
        )
        assert "var errFlags: hipError_t = hipSuccess;" in result
        assert (
            "flagsOut[1] = (/* HIP device query: "
            "textureReference.flags(texRef) */ 0);"
        ) in result
        assert "var errClamp: hipError_t = hipSuccess;" in result
        assert (
            "floats[3] = (/* HIP device query: "
            "textureReference.mipmapLevelClamp.min(texRef) */ 0);"
        ) in result
        assert (
            "floats[4] = (/* HIP device query: "
            "textureReference.mipmapLevelClamp.max(texRef) */ 0);"
        ) in result
        assert "sizes[0] = alignment;" not in result
        assert "ints[0] = addressMode;" not in result
        assert "ints[1] = filterMode;" not in result
        assert "flagsOut[0] = flags;" not in result
        assert "ints[2] = maxAniso;" not in result
        assert "ints[3] = mipFilterMode;" not in result
        assert "floats[0] = mipBias;" not in result
        assert "ints[4] = format;" not in result
        assert "ints[5] = channels;" not in result
        assert "floats[1] = minClamp;" not in result
        assert "floats[2] = maxClamp;" not in result
        assert "sizes[1] = statusAlignment;" not in result
        assert "flagsOut[1] = statusFlags;" not in result
        assert "floats[3] = statusMinClamp;" not in result
        assert "floats[4] = statusMaxClamp;" not in result
        assert "var manualAlignment: u32 = (/* HIP device query:" not in result
        assert (
            "var manualAddressMode: hipTextureAddressMode = (/* HIP device query:"
            not in result
        )
        assert "var manualFlags: u32 = (/* HIP device query:" not in result

    def test_hip_texture_reference_pointer_outputs_and_border_color_metadata(self):
        """Test raw texture-reference outputs clear stale metadata."""
        code = """
        void queryTextureReferencePointerOutputs(
            textureReference* texRef,
            float* borderColor,
            size_t* sizes,
            void** ptrs,
            hipArray_t* arrays,
            hipMipmappedArray_t* mipmaps,
            float* floats
        ) {
            size_t alignment = 0;
            hipDeviceptr_t address;
            hipArray_t array;
            hipMipmappedArray_t mipmap;
            int index = 1;
            hipGetTextureAlignmentOffset(&alignment, texRef);
            hipTexRefGetAddress(&alignment, texRef);
            sizes[0] = alignment;
            hipGetTextureAlignmentOffset(&alignment, texRef);
            sizes[1] = alignment;
            hipTexRefGetAddress(&address, texRef);
            ptrs[0] = address;
            hipTexRefGetArray(&array, texRef);
            arrays[0] = array;
            hipTexRefGetMipMappedArray(&mipmap, texRef);
            mipmaps[0] = mipmap;
            hipTexRefGetBorderColor(borderColor, texRef);
            floats[0] = borderColor[0];
            floats[1] = borderColor[3];
            floats[2] = borderColor[index];
            borderColor[0] = 1.0f;
            floats[3] = borderColor[0];
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "// HIP texture reference get address: output: alignment, texture: texRef"
            in result
        )
        assert "sizes[0] = alignment;" in result
        assert (
            "sizes[0] = "
            "(/* HIP device query: textureReference.alignmentOffset(texRef) */ 0);"
            not in result
        )
        assert (
            "sizes[1] = "
            "(/* HIP device query: textureReference.alignmentOffset(texRef) */ 0);"
            in result
        )
        assert "ptrs[0] = address;" in result
        assert "arrays[0] = array;" in result
        assert "mipmaps[0] = mipmap;" in result
        assert (
            "floats[0] = (/* HIP device query: "
            "textureReference.borderColor[0](texRef) */ 0);"
        ) in result
        assert (
            "floats[1] = (/* HIP device query: "
            "textureReference.borderColor[3](texRef) */ 0);"
        ) in result
        assert "floats[2] = borderColor[index];" in result
        assert "borderColor[0] = 1.0f;" in result
        assert "floats[3] = borderColor[0];" in result
        assert "textureReference.address(texRef)" not in result
        assert "textureReference.array(texRef)" not in result
        assert "textureReference.mipmappedArray(texRef)" not in result

    def test_hip_runtime_callback_activity_expression_conversion(self):
        """Test HIP callback/activity helper expressions lower to stable metadata."""
        code = """
        void inspect(hipFunction_t function, hipStream_t stream, void* hostFunction) {
            const char* apiName = hipApiName(1);
            const char* kernelName = hipKernelNameRef(function);
            const char* pointerKernelName =
                hipKernelNameRefByPtr(hostFunction, stream);
            int streamDevice = hipGetStreamDeviceId(stream);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert 'var apiName: ptr<i8> = /* HIP API name: 1 */ "";' in result
        assert (
            'var kernelName: ptr<i8> = /* HIP kernel name for function: function */ "";'
            in result
        )
        assert (
            "var pointerKernelName: ptr<i8> = /* HIP kernel name for host function: "
            'hostFunction, stream: stream */ "";'
        ) in result
        assert "var streamDevice: i32 = /* HIP stream device id: stream */ 0;" in result

        for function_name in [
            "hipApiName",
            "hipKernelNameRef",
            "hipKernelNameRefByPtr",
            "hipGetStreamDeviceId",
        ]:
            assert f"{function_name}(" not in result

    def test_hip_runtime_callback_activity_nested_expressions_convert(self):
        """Test HIP callback/activity helpers convert in nested expressions."""
        code = """
        const char* chooseName(const char* a, const char* b, bool useA) {
            return useA ? a : b;
        }

        int addOne(int value) {
            return value + 1;
        }

        void inspectNested(
            hipFunction_t function,
            hipStream_t stream,
            void* hostFunction,
            bool preferApi,
            int apiId,
            int expectedDevice
        ) {
            const char* selected =
                preferApi ? hipApiName(apiId) : hipKernelNameRef(function);
            const char* forwarded =
                chooseName(
                    hipKernelNameRef(function),
                    hipKernelNameRefByPtr(hostFunction, stream),
                    preferApi
                );
            bool streamMatches = hipGetStreamDeviceId(stream) == expectedDevice;
            int streamPlusOne = addOne(hipGetStreamDeviceId(stream));
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "var selected: ptr<i8> = (preferApi ? /* HIP API name: apiId */ "
            '"" : /* HIP kernel name for function: function */ "");'
        ) in result
        assert (
            "var forwarded: ptr<i8> = chooseName(/* HIP kernel name for "
            'function: function */ "", /* HIP kernel name for host function: '
            'hostFunction, stream: stream */ "", preferApi);'
        ) in result
        assert (
            "var streamMatches: bool = (/* HIP stream device id: stream */ "
            "0 == expectedDevice);"
        ) in result
        assert (
            "var streamPlusOne: i32 = addOne(/* HIP stream device id: stream */ 0);"
            in result
        )

        for function_name in [
            "hipApiName",
            "hipKernelNameRef",
            "hipKernelNameRefByPtr",
            "hipGetStreamDeviceId",
        ]:
            assert f"{function_name}(" not in result

    def test_hip_runtime_driver_context_api_conversion(self):
        """Test HIP initialization, context, and peer APIs emit metadata comments."""
        code = """
        void configure_context(int ordinal) {
            hipDevice_t device;
            hipDevice_t peerDevice;
            hipCtx_t ctx;
            hipCtx_t current;
            int driverVersion = 0;
            int runtimeVersion = 0;
            int canAccess = 0;
            int p2pAttribute = 0;
            int active = 0;
            unsigned int flags = 0;
            unsigned int apiVersion = 0;
            unsigned int linkType = 0;
            unsigned int hopCount = 0;
            hipFuncCache_t cacheConfig;
            hipSharedMemConfig sharedConfig;

            hipInit(0);
            hipDriverGetVersion(&driverVersion);
            hipRuntimeGetVersion(&runtimeVersion);
            hipDeviceGet(&device, ordinal);
            hipDeviceGet(&peerDevice, 1);
            hipDeviceCanAccessPeer(&canAccess, device, peerDevice);
            hipDeviceGetP2PAttribute(
                &p2pAttribute, hipDevP2PAttrPerformanceRank, device, peerDevice
            );
            hipDeviceEnablePeerAccess(peerDevice, 0);
            hipDeviceDisablePeerAccess(peerDevice);
            hipExtGetLinkTypeAndHopCount(device, peerDevice, &linkType, &hopCount);
            hipCtxCreate(&ctx, 0, device);
            hipCtxPushCurrent(ctx);
            hipCtxGetCurrent(&current);
            hipCtxSetCurrent(current);
            hipCtxGetDevice(&device);
            hipCtxGetApiVersion(ctx, &apiVersion);
            hipCtxGetCacheConfig(&cacheConfig);
            hipCtxSetCacheConfig(hipFuncCachePreferShared);
            hipCtxGetSharedMemConfig(&sharedConfig);
            hipCtxSetSharedMemConfig(hipSharedMemBankSizeEightByte);
            hipCtxGetFlags(&flags);
            hipCtxSynchronize();
            hipCtxPopCurrent(&current);
            hipCtxDestroy(ctx);
            hipDevicePrimaryCtxRetain(&ctx, device);
            hipDevicePrimaryCtxSetFlags(device, flags);
            hipDevicePrimaryCtxGetState(device, &flags, &active);
            hipDevicePrimaryCtxRelease(device);
            hipDevicePrimaryCtxReset(device);
            hipError_t err = hipDeviceGet(&device, ordinal);
            err = hipDeviceGetP2PAttribute(
                &p2pAttribute, hipDevP2PAttrPerformanceRank, device, peerDevice
            );
            err = hipExtGetLinkTypeAndHopCount(
                device, peerDevice, &linkType, &hopCount
            );
            err = hipCtxSynchronize();
            err = hipDevicePrimaryCtxGetState(device, &flags, &active);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// HIP initialize runtime: flags: 0" in result
        assert "// HIP get driver version: output: driverVersion" in result
        assert "// HIP get runtime version: output: runtimeVersion" in result
        assert (
            result.count("// HIP get device handle: output: device, ordinal: ordinal")
            == 2
        )
        assert "// HIP get device handle: output: peerDevice, ordinal: 1" in result
        assert (
            "// HIP device can access peer: output: canAccess, device: device, "
            "peer device: peerDevice"
        ) in result
        assert (
            result.count(
                "// HIP get P2P attribute: output: p2pAttribute, "
                "attribute: hipDevP2PAttrPerformanceRank, source device: device, "
                "destination device: peerDevice"
            )
            == 2
        )
        assert "// HIP enable peer access: peer device: peerDevice, flags: 0" in result
        assert "// HIP disable peer access: peer device: peerDevice" in result
        assert (
            result.count(
                "// HIP get link type and hop count: device 1: device, "
                "device 2: peerDevice, link type output: linkType, "
                "hop count output: hopCount"
            )
            == 2
        )
        assert "// HIP context create: output: ctx, flags: 0, device: device" in result
        assert "// HIP context push current: ctx" in result
        assert "// HIP context get current: output: current" in result
        assert "// HIP context set current: current" in result
        assert "// HIP context get device: output: device" in result
        assert (
            "// HIP context get API version: context: ctx, output: apiVersion" in result
        )
        assert "// HIP context get cache config: output: cacheConfig" in result
        assert "// HIP context set cache config: hipFuncCachePreferShared" in result
        assert "// HIP context get shared memory config: output: sharedConfig" in result
        assert (
            "// HIP context set shared memory config: hipSharedMemBankSizeEightByte"
            in result
        )
        assert "// HIP context get flags: output: flags" in result
        assert result.count("// HIP context synchronize") == 2
        assert "// HIP context pop current: output: current" in result
        assert "// HIP context destroy: ctx" in result
        assert "// HIP primary context retain: output: ctx, device: device" in result
        assert (
            "// HIP primary context set flags: device: device, flags: flags" in result
        )
        assert (
            result.count(
                "// HIP primary context get state: device: device, "
                "flags output: flags, active output: active"
            )
            == 2
        )
        assert "// HIP primary context release: device: device" in result
        assert "// HIP primary context reset: device: device" in result
        assert "var err: hipError_t = hipSuccess;" in result
        assert "err = hipSuccess;" in result

        for function_name in [
            "hipInit",
            "hipDriverGetVersion",
            "hipRuntimeGetVersion",
            "hipDeviceGet",
            "hipDeviceCanAccessPeer",
            "hipDeviceGetP2PAttribute",
            "hipDeviceEnablePeerAccess",
            "hipDeviceDisablePeerAccess",
            "hipExtGetLinkTypeAndHopCount",
            "hipCtxCreate",
            "hipCtxDestroy",
            "hipCtxPopCurrent",
            "hipCtxPushCurrent",
            "hipCtxSetCurrent",
            "hipCtxGetCurrent",
            "hipCtxGetDevice",
            "hipCtxGetApiVersion",
            "hipCtxGetCacheConfig",
            "hipCtxSetCacheConfig",
            "hipCtxGetSharedMemConfig",
            "hipCtxSetSharedMemConfig",
            "hipCtxGetFlags",
            "hipCtxSynchronize",
            "hipDevicePrimaryCtxRetain",
            "hipDevicePrimaryCtxRelease",
            "hipDevicePrimaryCtxReset",
            "hipDevicePrimaryCtxSetFlags",
            "hipDevicePrimaryCtxGetState",
        ]:
            assert f"{function_name}(" not in result

    def test_hip_driver_device_context_expression_contexts_emit_status(self):
        """Test driver, device, and context calls in expressions stay explicit."""
        code = """
        hipError_t driverDeviceContextExpressions(int ordinal, bool reuse) {
            hipDevice_t device;
            hipDevice_t peerDevice;
            hipCtx_t ctx;
            hipCtx_t current;
            int driverVersion = 0;
            int runtimeVersion = 0;
            int currentDevice = 0;
            int deviceCount = 0;
            int validDevices[2];
            int canAccess = 0;
            int p2pAttribute = 0;
            int attr = 0;
            int major = 0;
            int minor = 0;
            int active = 0;
            unsigned int apiVersion = 0;
            unsigned int flags = 0;
            unsigned int linkType = 0;
            unsigned int hopCount = 0;
            unsigned long long procFlags = 0ull;
            size_t total = 0;
            size_t limitValue = 0;
            char deviceName[64];
            char pciBusId[32];
            void* proc;
            hipUUID uuid;
            hipDeviceProp_t props;
            hipFuncCache_t cacheConfig;
            hipFuncCache_t deviceCacheConfig;
            hipSharedMemConfig sharedConfig;
            hipSharedMemConfig deviceSharedConfig;
            hipDriverProcAddressQueryResult queryStatus;

            bool initialized = hipInit(0) == hipSuccess;
            bool driverVersioned = hipDriverGetVersion(&driverVersion) == hipSuccess;
            bool runtimeVersioned = hipRuntimeGetVersion(&runtimeVersion) == hipSuccess;
            bool currentReady = hipGetDevice(&currentDevice) == hipSuccess;
            bool countReady = hipGetDeviceCount(&deviceCount) == hipSuccess;
            bool deviceSelected = hipSetDevice(ordinal) == hipSuccess;
            bool validSet = hipSetValidDevices(validDevices, 2) == hipSuccess;
            bool gotDevice = hipDeviceGet(&device, ordinal) == hipSuccess;
            bool gotPeer = hipDeviceGet(&peerDevice, 1) == hipSuccess;
            bool propsReady = hipGetDeviceProperties(&props, device) == hipSuccess;
            bool attrReady =
                hipDeviceGetAttribute(
                    &attr,
                    hipDeviceAttributeMaxThreadsPerBlock,
                    device
                ) == hipSuccess;
            bool nameReady = hipDeviceGetName(deviceName, 64, device) == hipSuccess;
            bool uuidReady = hipDeviceGetUuid(&uuid, device) == hipSuccess;
            bool totalReady = hipDeviceTotalMem(&total, device) == hipSuccess;
            bool capabilityReady =
                hipDeviceComputeCapability(&major, &minor, device) == hipSuccess;
            bool chosen = hipChooseDevice(&device, &props) == hipSuccess;
            bool busReady =
                hipDeviceGetPCIBusId(pciBusId, 32, device) == hipSuccess;
            bool busLookup =
                hipDeviceGetByPCIBusId(&device, pciBusId) == hipSuccess;
            bool deviceCacheReady =
                hipDeviceGetCacheConfig(&deviceCacheConfig) == hipSuccess;
            bool deviceCacheSet =
                hipDeviceSetCacheConfig(hipFuncCachePreferShared) == hipSuccess;
            bool deviceSharedReady =
                hipDeviceGetSharedMemConfig(&deviceSharedConfig) == hipSuccess;
            bool deviceSharedSet =
                hipDeviceSetSharedMemConfig(hipSharedMemBankSizeFourByte) == hipSuccess;
            bool limitReady =
                hipDeviceGetLimit(&limitValue, hipLimitMallocHeapSize) == hipSuccess;
            bool limitSet =
                hipDeviceSetLimit(hipLimitMallocHeapSize, limitValue) == hipSuccess;
            bool deviceReset = hipDeviceReset() == hipSuccess;
            bool deviceFlagsReady = hipGetDeviceFlags(&flags) == hipSuccess;
            bool deviceFlagsSet =
                hipSetDeviceFlags(hipDeviceScheduleAuto) == hipSuccess;
            bool procReady =
                hipGetProcAddress(
                    "hipMalloc",
                    &proc,
                    runtimeVersion,
                    procFlags,
                    &queryStatus
                ) == hipSuccess;
            bool peerCheck =
                hipDeviceCanAccessPeer(&canAccess, device, peerDevice) == hipSuccess;
            bool p2pReady =
                hipDeviceGetP2PAttribute(
                    &p2pAttribute,
                    hipDevP2PAttrPerformanceRank,
                    device,
                    peerDevice
                ) == hipSuccess;
            bool peerEnabled =
                hipDeviceEnablePeerAccess(peerDevice, 0) == hipSuccess;
            bool peerDisabled =
                hipDeviceDisablePeerAccess(peerDevice) == hipSuccess;
            bool linkReady =
                hipExtGetLinkTypeAndHopCount(
                    device,
                    peerDevice,
                    &linkType,
                    &hopCount
                ) == hipSuccess;
            bool pushed = hipCtxPushCurrent(ctx) == hipSuccess;
            bool gotCurrent = hipCtxGetCurrent(&current) == hipSuccess;
            bool gotContextDevice = hipCtxGetDevice(&device) == hipSuccess;
            bool gotCache = hipCtxGetCacheConfig(&cacheConfig) == hipSuccess;
            bool setCache =
                hipCtxSetCacheConfig(hipFuncCachePreferShared) == hipSuccess;
            bool gotShared =
                hipCtxGetSharedMemConfig(&sharedConfig) == hipSuccess;
            bool setShared =
                hipCtxSetSharedMemConfig(hipSharedMemBankSizeEightByte) == hipSuccess;
            bool gotFlags = hipCtxGetFlags(&flags) == hipSuccess;
            bool synchronized = hipCtxSynchronize() == hipSuccess;
            bool popped = hipCtxPopCurrent(&current) == hipSuccess;
            bool primaryFlags =
                hipDevicePrimaryCtxSetFlags(device, flags) == hipSuccess;
            bool reset = hipDevicePrimaryCtxReset(device) == hipSuccess;
            if (hipCtxCreate(&ctx, 0, device) != hipSuccess) {
                return hipCtxDestroy(ctx);
            }
            if (hipCtxGetApiVersion(ctx, &apiVersion) == hipSuccess) {
                hipError_t selected = reuse ? hipCtxSetCurrent(current) : hipDevicePrimaryCtxRetain(&current, device);
                return selected;
            }
            if (hipDevicePrimaryCtxGetState(device, &flags, &active) == hipSuccess) {
                return hipDevicePrimaryCtxRelease(device);
            }
            return hipDevicePrimaryCtxReset(device);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        result = HipToCrossGLConverter().generate(ast)

        assert (
            "var initialized: bool = ((/* HIP initialize runtime: flags: 0 */ "
            "hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var currentReady: bool = ((/* HIP get current device: output: "
            "currentDevice */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var gotDevice: bool = ((/* HIP get device handle: output: device, "
            "ordinal: ordinal */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var propsReady: bool = ((/* HIP get device properties: props, "
            "device: device */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var deviceCacheReady: bool = ((/* HIP get device cache config: "
            "output: deviceCacheConfig */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var procReady: bool = ((/* HIP get proc address: symbol: "
            '"hipMalloc", output: proc, version: runtimeVersion, flags: '
            "procFlags, status output: queryStatus */ hipSuccess) == "
            "hipSuccess);"
        ) in result
        assert (
            "var peerCheck: bool = ((/* HIP device can access peer: output: "
            "canAccess, device: device, peer device: peerDevice */ hipSuccess) "
            "== hipSuccess);"
        ) in result
        assert (
            "var linkReady: bool = ((/* HIP get link type and hop count: device "
            "1: device, device 2: peerDevice, link type output: linkType, "
            "hop count output: hopCount */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var gotCache: bool = ((/* HIP context get cache config: output: "
            "cacheConfig */ hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "var synchronized: bool = ((/* HIP context synchronize */ "
            "hipSuccess) == hipSuccess);"
        ) in result
        assert (
            "if (((/* HIP context create: output: ctx, flags: 0, device: device "
            "*/ hipSuccess) != hipSuccess))"
        ) in result
        assert "return (/* HIP context destroy: ctx */ hipSuccess);" in result
        assert (
            "var selected: hipError_t = (reuse ? "
            "(/* HIP context set current: current */ hipSuccess) : "
            "(/* HIP primary context retain: output: current, device: device */ "
            "hipSuccess));"
        ) in result
        assert (
            "if (((/* HIP primary context get state: device: device, flags "
            "output: flags, active output: active */ hipSuccess) == hipSuccess))"
        ) in result
        assert (
            "return (/* HIP primary context release: device: device */ " "hipSuccess);"
        ) in result
        for function_name in [
            "hipInit",
            "hipDriverGetVersion",
            "hipRuntimeGetVersion",
            "hipGetDevice",
            "hipGetDeviceCount",
            "hipSetDevice",
            "hipSetValidDevices",
            "hipGetDeviceProperties",
            "hipDeviceGet",
            "hipDeviceCanAccessPeer",
            "hipDeviceGetP2PAttribute",
            "hipDeviceEnablePeerAccess",
            "hipDeviceDisablePeerAccess",
            "hipExtGetLinkTypeAndHopCount",
            "hipDeviceGetAttribute",
            "hipDeviceGetName",
            "hipDeviceGetUuid",
            "hipDeviceTotalMem",
            "hipDeviceComputeCapability",
            "hipChooseDevice",
            "hipDeviceGetPCIBusId",
            "hipDeviceGetByPCIBusId",
            "hipDeviceGetCacheConfig",
            "hipDeviceSetCacheConfig",
            "hipDeviceGetSharedMemConfig",
            "hipDeviceSetSharedMemConfig",
            "hipDeviceGetLimit",
            "hipDeviceSetLimit",
            "hipDeviceReset",
            "hipGetDeviceFlags",
            "hipSetDeviceFlags",
            "hipGetProcAddress",
            "hipCtxCreate",
            "hipCtxDestroy",
            "hipCtxPopCurrent",
            "hipCtxPushCurrent",
            "hipCtxSetCurrent",
            "hipCtxGetCurrent",
            "hipCtxGetDevice",
            "hipCtxGetApiVersion",
            "hipCtxGetCacheConfig",
            "hipCtxSetCacheConfig",
            "hipCtxGetSharedMemConfig",
            "hipCtxSetSharedMemConfig",
            "hipCtxGetFlags",
            "hipCtxSynchronize",
            "hipDevicePrimaryCtxRetain",
            "hipDevicePrimaryCtxRelease",
            "hipDevicePrimaryCtxReset",
            "hipDevicePrimaryCtxSetFlags",
            "hipDevicePrimaryCtxGetState",
        ]:
            assert f"{function_name}(" not in result

    def test_hip_runtime_graph_api_conversion(self):
        """Test HIP graph APIs emit metadata comments."""
        code = """
        void build_graph(hipStream_t stream, hipEvent_t event) {
            hipGraph_t graph;
            hipGraph_t clone;
            hipGraph_t child;
            hipGraphExec_t exec;
            hipGraphNode_t emptyNode;
            hipGraphNode_t hostNode;
            hipGraphNode_t kernelNode;
            hipGraphNode_t memcpyNode;
            hipGraphNode_t memsetNode;
            hipGraphNode_t childNode;
            hipGraphNode_t recordNode;
            hipGraphNode_t waitNode;
            hipGraphNode_t cloneNode;
            hipGraphNode_t errorNode;
            hipGraphNode_t* deps;
            hipGraphNode_t* nodes;
            hipGraphNode_t* fromNodes;
            hipGraphNode_t* toNodes;
            hipKernelNodeParams kernelParams;
            hipMemcpy3DParms copyParams;
            hipMemsetParams memsetParams;
            hipHostNodeParams hostParams;
            hipGraphExecUpdateResult updateResult;
            hipGraphNodeType nodeType;
            size_t numDeps = 0;
            size_t numNodes = 0;
            size_t numEdges = 0;
            unsigned int flags = 0;
            char log[128];

            hipGraphCreate(&graph, 0);
            hipGraphCreate(&child, 0);
            hipGraphClone(&clone, graph);
            hipGraphAddEmptyNode(&emptyNode, graph, deps, numDeps);
            hipGraphAddHostNode(&hostNode, graph, deps, numDeps, &hostParams);
            hipGraphAddKernelNode(&kernelNode, graph, deps, numDeps, &kernelParams);
            hipGraphAddMemcpyNode(&memcpyNode, graph, deps, numDeps, &copyParams);
            hipGraphAddMemsetNode(&memsetNode, graph, deps, numDeps, &memsetParams);
            hipGraphAddChildGraphNode(&childNode, graph, deps, numDeps, child);
            hipGraphAddEventRecordNode(&recordNode, graph, deps, numDeps, event);
            hipGraphAddEventWaitNode(&waitNode, graph, deps, numDeps, event);
            hipGraphAddDependencies(graph, fromNodes, toNodes, numDeps);
            hipGraphRemoveDependencies(graph, fromNodes, toNodes, numDeps);
            hipGraphGetNodes(graph, nodes, &numNodes);
            hipGraphGetRootNodes(graph, nodes, &numNodes);
            hipGraphGetEdges(graph, fromNodes, toNodes, &numEdges);
            hipGraphNodeGetDependencies(kernelNode, deps, &numDeps);
            hipGraphNodeGetDependentNodes(kernelNode, deps, &numDeps);
            hipGraphNodeFindInClone(&cloneNode, kernelNode, clone);
            hipGraphNodeGetType(kernelNode, &nodeType);
            hipGraphKernelNodeGetParams(kernelNode, &kernelParams);
            hipGraphKernelNodeSetParams(kernelNode, &kernelParams);
            hipGraphMemcpyNodeGetParams(memcpyNode, &copyParams);
            hipGraphMemcpyNodeSetParams(memcpyNode, &copyParams);
            hipGraphMemsetNodeGetParams(memsetNode, &memsetParams);
            hipGraphMemsetNodeSetParams(memsetNode, &memsetParams);
            hipGraphHostNodeGetParams(hostNode, &hostParams);
            hipGraphHostNodeSetParams(hostNode, &hostParams);
            hipGraphEventRecordNodeGetEvent(recordNode, &event);
            hipGraphEventRecordNodeSetEvent(recordNode, event);
            hipGraphEventWaitNodeGetEvent(waitNode, &event);
            hipGraphEventWaitNodeSetEvent(waitNode, event);
            hipGraphInstantiate(&exec, graph, &errorNode, log, 128);
            hipGraphInstantiateWithFlags(&exec, graph, flags);
            hipGraphLaunch(exec, stream);
            hipGraphExecUpdate(exec, graph, &errorNode, &updateResult);
            hipGraphExecKernelNodeSetParams(exec, kernelNode, &kernelParams);
            hipGraphExecMemcpyNodeSetParams(exec, memcpyNode, &copyParams);
            hipGraphExecMemsetNodeSetParams(exec, memsetNode, &memsetParams);
            hipGraphExecHostNodeSetParams(exec, hostNode, &hostParams);
            hipGraphExecChildGraphNodeSetParams(exec, childNode, child);
            hipGraphExecEventRecordNodeSetEvent(exec, recordNode, event);
            hipGraphExecEventWaitNodeSetEvent(exec, waitNode, event);
            hipError_t err = hipGraphLaunch(exec, stream);
            err = hipGraphExecDestroy(exec);
            hipGraphDestroyNode(emptyNode);
            hipGraphExecDestroy(exec);
            hipGraphDestroy(clone);
            hipGraphDestroy(child);
            hipGraphDestroy(graph);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// HIP graph create: output: graph, flags: 0" in result
        assert "// HIP graph create: output: child, flags: 0" in result
        assert "// HIP graph clone: output: clone, source: graph" in result
        assert (
            "// HIP graph add empty node: output: emptyNode, graph: graph, "
            "dependencies: deps, count: numDeps"
        ) in result
        assert (
            "// HIP graph add host node: output: hostNode, graph: graph, "
            "dependencies: deps, count: numDeps, params: (&hostParams)"
        ) in result
        assert (
            "// HIP graph add kernel node: output: kernelNode, graph: graph, "
            "dependencies: deps, count: numDeps, params: (&kernelParams)"
        ) in result
        assert (
            "// HIP graph add memcpy node: output: memcpyNode, graph: graph, "
            "dependencies: deps, count: numDeps, params: (&copyParams)"
        ) in result
        assert (
            "// HIP graph add memset node: output: memsetNode, graph: graph, "
            "dependencies: deps, count: numDeps, params: (&memsetParams)"
        ) in result
        assert (
            "// HIP graph add child graph node: output: childNode, graph: graph, "
            "dependencies: deps, count: numDeps, child graph: child"
        ) in result
        assert (
            "// HIP graph add event record node: output: recordNode, graph: graph, "
            "dependencies: deps, count: numDeps, event: event"
        ) in result
        assert (
            "// HIP graph add event wait node: output: waitNode, graph: graph, "
            "dependencies: deps, count: numDeps, event: event"
        ) in result
        assert (
            "// HIP graph add dependencies: graph: graph, from: fromNodes, "
            "to: toNodes, count: numDeps"
        ) in result
        assert (
            "// HIP graph remove dependencies: graph: graph, from: fromNodes, "
            "to: toNodes, count: numDeps"
        ) in result
        assert (
            "// HIP graph get nodes: graph: graph, nodes output: nodes, "
            "count output: numNodes"
        ) in result
        assert (
            "// HIP graph get root nodes: graph: graph, nodes output: nodes, "
            "count output: numNodes"
        ) in result
        assert (
            "// HIP graph get edges: graph: graph, from output: fromNodes, "
            "to output: toNodes, count output: numEdges"
        ) in result
        assert (
            "// HIP graph node get dependencies: node: kernelNode, "
            "nodes output: deps, count output: numDeps"
        ) in result
        assert (
            "// HIP graph node get dependent nodes: node: kernelNode, "
            "nodes output: deps, count output: numDeps"
        ) in result
        assert (
            "// HIP graph node find in clone: output: cloneNode, "
            "original: kernelNode, clone graph: clone"
        ) in result
        assert (
            "// HIP graph node get type: node: kernelNode, output: nodeType" in result
        )
        assert (
            "// HIP graph kernel node get params: node: kernelNode, "
            "params: (&kernelParams)"
        ) in result
        assert (
            "// HIP graph kernel node set params: node: kernelNode, "
            "params: (&kernelParams)"
        ) in result
        assert (
            "// HIP graph memcpy node get params: node: memcpyNode, "
            "params: (&copyParams)"
        ) in result
        assert (
            "// HIP graph memcpy node set params: node: memcpyNode, "
            "params: (&copyParams)"
        ) in result
        assert (
            "// HIP graph memset node get params: node: memsetNode, "
            "params: (&memsetParams)"
        ) in result
        assert (
            "// HIP graph memset node set params: node: memsetNode, "
            "params: (&memsetParams)"
        ) in result
        assert (
            "// HIP graph host node get params: node: hostNode, params: (&hostParams)"
            in result
        )
        assert (
            "// HIP graph host node set params: node: hostNode, params: (&hostParams)"
            in result
        )
        assert (
            "// HIP graph event record node get event: node: recordNode, output: event"
            in result
        )
        assert (
            "// HIP graph event record node set event: node: recordNode, event: event"
            in result
        )
        assert (
            "// HIP graph event wait node get event: node: waitNode, output: event"
            in result
        )
        assert (
            "// HIP graph event wait node set event: node: waitNode, event: event"
            in result
        )
        assert (
            "// HIP graph instantiate: output: exec, graph: graph, "
            "error node output: errorNode, log buffer: log, log bytes: 128"
        ) in result
        assert (
            "// HIP graph instantiate with flags: output: exec, graph: graph, "
            "flags: flags"
        ) in result
        assert result.count("// HIP graph launch: exec: exec, stream: stream") == 2
        assert (
            "// HIP graph exec update: exec: exec, graph: graph, "
            "error node output: errorNode, result output: updateResult"
        ) in result
        assert (
            "// HIP graph exec set kernel node params: exec: exec, "
            "node: kernelNode, params: (&kernelParams)"
        ) in result
        assert (
            "// HIP graph exec set memcpy node params: exec: exec, "
            "node: memcpyNode, params: (&copyParams)"
        ) in result
        assert (
            "// HIP graph exec set memset node params: exec: exec, "
            "node: memsetNode, params: (&memsetParams)"
        ) in result
        assert (
            "// HIP graph exec set host node params: exec: exec, "
            "node: hostNode, params: (&hostParams)"
        ) in result
        assert (
            "// HIP graph exec set child graph node params: exec: exec, "
            "node: childNode, params: child"
        ) in result
        assert (
            "// HIP graph exec event record node set event: exec: exec, "
            "node: recordNode, event: event"
        ) in result
        assert (
            "// HIP graph exec event wait node set event: exec: exec, "
            "node: waitNode, event: event"
        ) in result
        assert result.count("// HIP graph exec destroy: exec") == 2
        assert "// HIP graph destroy node: emptyNode" in result
        assert "// HIP graph destroy: clone" in result
        assert "// HIP graph destroy: child" in result
        assert "// HIP graph destroy: graph" in result
        assert "var err: hipError_t = hipSuccess;" in result
        assert "err = hipSuccess;" in result

        for function_name in [
            "hipGraphCreate",
            "hipGraphDestroy",
            "hipGraphClone",
            "hipGraphAddEmptyNode",
            "hipGraphAddHostNode",
            "hipGraphAddKernelNode",
            "hipGraphAddMemcpyNode",
            "hipGraphAddMemsetNode",
            "hipGraphAddChildGraphNode",
            "hipGraphAddEventRecordNode",
            "hipGraphAddEventWaitNode",
            "hipGraphAddDependencies",
            "hipGraphRemoveDependencies",
            "hipGraphGetNodes",
            "hipGraphGetRootNodes",
            "hipGraphGetEdges",
            "hipGraphNodeGetDependencies",
            "hipGraphNodeGetDependentNodes",
            "hipGraphNodeFindInClone",
            "hipGraphNodeGetType",
            "hipGraphDestroyNode",
            "hipGraphInstantiate",
            "hipGraphInstantiateWithFlags",
            "hipGraphLaunch",
            "hipGraphExecUpdate",
            "hipGraphExecDestroy",
            "hipGraphKernelNodeGetParams",
            "hipGraphKernelNodeSetParams",
            "hipGraphMemcpyNodeGetParams",
            "hipGraphMemcpyNodeSetParams",
            "hipGraphMemsetNodeGetParams",
            "hipGraphMemsetNodeSetParams",
            "hipGraphHostNodeGetParams",
            "hipGraphHostNodeSetParams",
            "hipGraphEventRecordNodeGetEvent",
            "hipGraphEventRecordNodeSetEvent",
            "hipGraphEventWaitNodeGetEvent",
            "hipGraphEventWaitNodeSetEvent",
            "hipGraphExecKernelNodeSetParams",
            "hipGraphExecMemcpyNodeSetParams",
            "hipGraphExecMemsetNodeSetParams",
            "hipGraphExecHostNodeSetParams",
            "hipGraphExecChildGraphNodeSetParams",
            "hipGraphExecEventRecordNodeSetEvent",
            "hipGraphExecEventWaitNodeSetEvent",
        ]:
            assert f"{function_name}(" not in result

    def test_hip_runtime_extended_graph_api_conversion(self):
        """Test extended HIP graph APIs emit metadata comments."""
        code = """
        void tune_graph(hipStream_t stream) {
            hipGraph_t graph;
            hipGraph_t child;
            hipGraph_t embeddedGraph;
            hipGraphExec_t exec;
            hipGraphNode_t genericNode;
            hipGraphNode_t kernelNode;
            hipGraphNode_t copy1DNode;
            hipGraphNode_t fromSymbolNode;
            hipGraphNode_t toSymbolNode;
            hipGraphNode_t allocNode;
            hipGraphNode_t freeNode;
            hipGraphNode_t childNode;
            hipGraphNode_t signalNode;
            hipGraphNode_t waitNode;
            hipGraphNode_t* deps;
            hipGraphNodeParams nodeParams;
            hipGraphInstantiateParams instantiateParams;
            hipMemAllocNodeParams allocParams;
            hipExternalSemaphoreSignalNodeParams signalParams;
            hipExternalSemaphoreWaitNodeParams waitParams;
            hipKernelNodeAttrValue attrValue;
            hipStreamCaptureMode mode;
            hipUserObject_t userObject;
            hipHostFn_t destructor;
            void* depData;
            void* resource;
            void* devicePtr;
            void* dst;
            void* src;
            void* symbol;
            int device = 0;
            int enabled = 0;
            unsigned long long execFlags = 0;
            size_t numDeps = 0;
            size_t bytes = 64;

            hipStreamBeginCaptureToGraph(
                stream, graph, deps, depData, numDeps, hipStreamCaptureModeGlobal
            );
            hipThreadExchangeStreamCaptureMode(&mode);
            hipGraphAddNode(&genericNode, graph, deps, numDeps, &nodeParams);
            hipGraphNodeSetParams(genericNode, &nodeParams);
            hipGraphExecNodeSetParams(exec, genericNode, &nodeParams);
            hipGraphInstantiateWithParams(&exec, graph, &instantiateParams);
            hipGraphUpload(exec, stream);
            hipGraphExecGetFlags(exec, &execFlags);
            hipGraphAddMemAllocNode(&allocNode, graph, deps, numDeps, &allocParams);
            hipGraphMemAllocNodeGetParams(allocNode, &allocParams);
            hipGraphAddMemFreeNode(&freeNode, graph, deps, numDeps, devicePtr);
            hipGraphMemFreeNodeGetParams(freeNode, &devicePtr);
            hipGraphAddMemcpyNode1D(
                &copy1DNode, graph, deps, numDeps, dst, src, bytes,
                hipMemcpyDeviceToDevice
            );
            hipGraphMemcpyNodeSetParams1D(
                copy1DNode, dst, src, bytes, hipMemcpyDeviceToDevice
            );
            hipGraphExecMemcpyNodeSetParams1D(
                exec, copy1DNode, dst, src, bytes, hipMemcpyDeviceToDevice
            );
            hipGraphAddMemcpyNodeFromSymbol(
                &fromSymbolNode, graph, deps, numDeps, dst, symbol, bytes, 4,
                hipMemcpyDeviceToDevice
            );
            hipGraphMemcpyNodeSetParamsFromSymbol(
                fromSymbolNode, dst, symbol, bytes, 4, hipMemcpyDeviceToDevice
            );
            hipGraphExecMemcpyNodeSetParamsFromSymbol(
                exec, fromSymbolNode, dst, symbol, bytes, 4, hipMemcpyDeviceToDevice
            );
            hipGraphAddMemcpyNodeToSymbol(
                &toSymbolNode, graph, deps, numDeps, symbol, src, bytes, 8,
                hipMemcpyDeviceToDevice
            );
            hipGraphMemcpyNodeSetParamsToSymbol(
                toSymbolNode, symbol, src, bytes, 8, hipMemcpyDeviceToDevice
            );
            hipGraphExecMemcpyNodeSetParamsToSymbol(
                exec, toSymbolNode, symbol, src, bytes, 8, hipMemcpyDeviceToDevice
            );
            hipGraphChildGraphNodeGetGraph(childNode, &embeddedGraph);
            hipGraphKernelNodeCopyAttributes(kernelNode, genericNode);
            hipGraphKernelNodeSetAttribute(
                kernelNode, hipKernelNodeAttributeCooperative, &attrValue
            );
            hipGraphKernelNodeGetAttribute(
                kernelNode, hipKernelNodeAttributeCooperative, &attrValue
            );
            hipGraphNodeSetEnabled(exec, genericNode, 1);
            hipGraphNodeGetEnabled(exec, genericNode, &enabled);
            hipGraphAddExternalSemaphoresSignalNode(
                &signalNode, graph, deps, numDeps, &signalParams
            );
            hipGraphExternalSemaphoresSignalNodeGetParams(signalNode, &signalParams);
            hipGraphExternalSemaphoresSignalNodeSetParams(signalNode, &signalParams);
            hipGraphExecExternalSemaphoresSignalNodeSetParams(
                exec, signalNode, &signalParams
            );
            hipGraphAddExternalSemaphoresWaitNode(
                &waitNode, graph, deps, numDeps, &waitParams
            );
            hipGraphExternalSemaphoresWaitNodeGetParams(waitNode, &waitParams);
            hipGraphExternalSemaphoresWaitNodeSetParams(waitNode, &waitParams);
            hipGraphExecExternalSemaphoresWaitNodeSetParams(
                exec, waitNode, &waitParams
            );
            hipDeviceGetGraphMemAttribute(
                device, hipGraphMemAttrUsedMemCurrent, &bytes
            );
            hipDeviceSetGraphMemAttribute(
                device, hipGraphMemAttrReserveMemCurrent, &bytes
            );
            hipDeviceGraphMemTrim(device);
            hipGraphDebugDotPrint(graph, "graph.dot", 0);
            hipUserObjectCreate(&userObject, resource, destructor, 1, 0);
            hipUserObjectRetain(userObject, 1);
            hipGraphRetainUserObject(graph, userObject, 1, 0);
            hipGraphReleaseUserObject(graph, userObject, 1);
            hipUserObjectRelease(userObject, 1);
            hipError_t err = hipGraphUpload(exec, stream);
            err = hipGraphExecGetFlags(exec, &execFlags);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        codegen = HipToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// HIP stream begin capture to graph: stream: stream, graph: graph, "
            "dependencies: deps, dependency data: depData, count: numDeps, "
            "mode: hipStreamCaptureModeGlobal"
        ) in result
        assert "// HIP exchange stream capture mode: output: mode" in result
        assert (
            "// HIP graph add generic node: output: genericNode, graph: graph, "
            "dependencies: deps, count: numDeps, params: (&nodeParams)"
        ) in result
        assert (
            "// HIP graph generic node set params: node: genericNode, "
            "params: (&nodeParams)"
        ) in result
        assert (
            "// HIP graph exec generic node set params: exec: exec, "
            "node: genericNode, params: (&nodeParams)"
        ) in result
        assert (
            "// HIP graph instantiate with params: output: exec, graph: graph, "
            "params: (&instantiateParams)"
        ) in result
        assert result.count("// HIP graph upload: exec: exec, stream: stream") == 2
        assert (
            result.count("// HIP graph exec get flags: exec: exec, output: execFlags")
            == 2
        )
        assert (
            "// HIP graph add memory alloc node: output: allocNode, graph: graph, "
            "dependencies: deps, count: numDeps, params: (&allocParams)"
        ) in result
        assert (
            "// HIP graph memory alloc node get params: node: allocNode, "
            "params output: allocParams"
        ) in result
        assert (
            "// HIP graph add memory free node: output: freeNode, graph: graph, "
            "dependencies: deps, count: numDeps, pointer: devicePtr"
        ) in result
        assert (
            "// HIP graph memory free node get params: node: freeNode, "
            "pointer output: devicePtr"
        ) in result
        assert (
            "// HIP graph add memcpy 1D node: output: copy1DNode, graph: graph, "
            "dependencies: deps, count: numDeps, destination: dst, source: src, "
            "bytes: bytes, kind: hipMemcpyDeviceToDevice"
        ) in result
        assert (
            "// HIP graph memcpy 1D node set params: node: copy1DNode, "
            "destination: dst, source: src, bytes: bytes, kind: hipMemcpyDeviceToDevice"
        ) in result
        assert (
            "// HIP graph exec memcpy 1D node set params: exec: exec, "
            "node: copy1DNode, destination: dst, source: src, bytes: bytes, "
            "kind: hipMemcpyDeviceToDevice"
        ) in result
        assert (
            "// HIP graph add memcpy from symbol node: output: fromSymbolNode, "
            "graph: graph, dependencies: deps, count: numDeps, destination: dst, "
            "source: symbol, bytes: bytes, offset: 4, kind: hipMemcpyDeviceToDevice"
        ) in result
        assert (
            "// HIP graph memcpy from symbol node set params: node: fromSymbolNode, "
            "destination: dst, source: symbol, bytes: bytes, offset: 4, "
            "kind: hipMemcpyDeviceToDevice"
        ) in result
        assert (
            "// HIP graph exec memcpy from symbol node set params: exec: exec, "
            "node: fromSymbolNode, destination: dst, source: symbol, bytes: bytes, "
            "offset: 4, kind: hipMemcpyDeviceToDevice"
        ) in result
        assert (
            "// HIP graph add memcpy to symbol node: output: toSymbolNode, "
            "graph: graph, dependencies: deps, count: numDeps, destination: symbol, "
            "source: src, bytes: bytes, offset: 8, kind: hipMemcpyDeviceToDevice"
        ) in result
        assert (
            "// HIP graph memcpy to symbol node set params: node: toSymbolNode, "
            "destination: symbol, source: src, bytes: bytes, offset: 8, "
            "kind: hipMemcpyDeviceToDevice"
        ) in result
        assert (
            "// HIP graph exec memcpy to symbol node set params: exec: exec, "
            "node: toSymbolNode, destination: symbol, source: src, bytes: bytes, "
            "offset: 8, kind: hipMemcpyDeviceToDevice"
        ) in result
        assert (
            "// HIP graph child node get graph: node: childNode, output: embeddedGraph"
            in result
        )
        assert (
            "// HIP graph kernel node copy attributes: source: kernelNode, "
            "destination: genericNode"
        ) in result
        assert (
            "// HIP graph kernel node set attribute: node: kernelNode, "
            "attribute: hipKernelNodeAttributeCooperative, value: attrValue"
        ) in result
        assert (
            "// HIP graph kernel node get attribute: node: kernelNode, "
            "attribute: hipKernelNodeAttributeCooperative, output: attrValue"
        ) in result
        assert (
            "// HIP graph node set enabled: exec: exec, node: genericNode, value: 1"
            in result
        )
        assert (
            "// HIP graph node get enabled: exec: exec, node: genericNode, "
            "output: enabled"
        ) in result
        assert (
            "// HIP graph add external semaphore signal node: output: signalNode, "
            "graph: graph, dependencies: deps, count: numDeps, params: (&signalParams)"
        ) in result
        assert (
            "// HIP graph external semaphore signal node get params: node: signalNode, "
            "params: (&signalParams)"
        ) in result
        assert (
            "// HIP graph external semaphore signal node set params: node: signalNode, "
            "params: (&signalParams)"
        ) in result
        assert (
            "// HIP graph exec external semaphore signal node set params: exec: exec, "
            "node: signalNode, params: (&signalParams)"
        ) in result
        assert (
            "// HIP graph add external semaphore wait node: output: waitNode, "
            "graph: graph, dependencies: deps, count: numDeps, params: (&waitParams)"
        ) in result
        assert (
            "// HIP graph external semaphore wait node get params: node: waitNode, "
            "params: (&waitParams)"
        ) in result
        assert (
            "// HIP graph external semaphore wait node set params: node: waitNode, "
            "params: (&waitParams)"
        ) in result
        assert (
            "// HIP graph exec external semaphore wait node set params: exec: exec, "
            "node: waitNode, params: (&waitParams)"
        ) in result
        assert (
            "// HIP device graph memory get attribute: device: device, "
            "attribute: hipGraphMemAttrUsedMemCurrent, output: bytes"
        ) in result
        assert (
            "// HIP device graph memory set attribute: device: device, "
            "attribute: hipGraphMemAttrReserveMemCurrent, value: bytes"
        ) in result
        assert "// HIP device graph memory trim: device: device" in result
        assert (
            '// HIP graph debug dot print: graph: graph, path: "graph.dot", flags: 0'
            in result
        )
        assert (
            "// HIP user object create: output: userObject, resource: resource, "
            "destructor: destructor, initial refcount: 1, flags: 0"
        ) in result
        assert "// HIP user object retain: object: userObject, count: 1" in result
        assert (
            "// HIP graph retain user object: graph: graph, object: userObject, "
            "count: 1, flags: 0"
        ) in result
        assert (
            "// HIP graph release user object: graph: graph, object: userObject, "
            "count: 1"
        ) in result
        assert "// HIP user object release: object: userObject, count: 1" in result
        assert "var err: hipError_t = hipSuccess;" in result
        assert "err = hipSuccess;" in result

        for function_name in [
            "hipStreamBeginCaptureToGraph",
            "hipThreadExchangeStreamCaptureMode",
            "hipGraphAddNode",
            "hipGraphNodeSetParams",
            "hipGraphExecNodeSetParams",
            "hipGraphInstantiateWithParams",
            "hipGraphUpload",
            "hipGraphExecGetFlags",
            "hipGraphAddMemAllocNode",
            "hipGraphMemAllocNodeGetParams",
            "hipGraphAddMemFreeNode",
            "hipGraphMemFreeNodeGetParams",
            "hipGraphAddMemcpyNode1D",
            "hipGraphMemcpyNodeSetParams1D",
            "hipGraphExecMemcpyNodeSetParams1D",
            "hipGraphAddMemcpyNodeFromSymbol",
            "hipGraphMemcpyNodeSetParamsFromSymbol",
            "hipGraphExecMemcpyNodeSetParamsFromSymbol",
            "hipGraphAddMemcpyNodeToSymbol",
            "hipGraphMemcpyNodeSetParamsToSymbol",
            "hipGraphExecMemcpyNodeSetParamsToSymbol",
            "hipGraphChildGraphNodeGetGraph",
            "hipGraphKernelNodeCopyAttributes",
            "hipGraphKernelNodeSetAttribute",
            "hipGraphKernelNodeGetAttribute",
            "hipGraphNodeSetEnabled",
            "hipGraphNodeGetEnabled",
            "hipGraphAddExternalSemaphoresSignalNode",
            "hipGraphExternalSemaphoresSignalNodeGetParams",
            "hipGraphExternalSemaphoresSignalNodeSetParams",
            "hipGraphExecExternalSemaphoresSignalNodeSetParams",
            "hipGraphAddExternalSemaphoresWaitNode",
            "hipGraphExternalSemaphoresWaitNodeGetParams",
            "hipGraphExternalSemaphoresWaitNodeSetParams",
            "hipGraphExecExternalSemaphoresWaitNodeSetParams",
            "hipDeviceGetGraphMemAttribute",
            "hipDeviceSetGraphMemAttribute",
            "hipDeviceGraphMemTrim",
            "hipGraphDebugDotPrint",
            "hipUserObjectCreate",
            "hipUserObjectRetain",
            "hipGraphRetainUserObject",
            "hipGraphReleaseUserObject",
            "hipUserObjectRelease",
        ]:
            assert f"{function_name}(" not in result

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

        assert (
            "pub fn kernel(mut data: Vec<f32>, indices: Vec<i32>, value: f32)" in result
        )
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
