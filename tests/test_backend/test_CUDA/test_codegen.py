"""Test CUDA Code Generation"""

import pytest
from crosstl import translate
from crosstl.backend.CUDA.CudaLexer import CudaLexer
from crosstl.backend.CUDA.CudaParser import CudaParser
from crosstl.backend.CUDA.CudaCrossGLCodeGen import CudaToCrossGLConverter


class TestCudaCodeGen:
    def test_basic_kernel_conversion(self):
        """Test basic kernel to CrossGL conversion"""
        code = """
        __global__ void simple_kernel(float* data) {
            int idx = threadIdx.x;
            data[idx] = idx;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// CUDA to CrossGL conversion" in result
        assert "// Kernel: simple_kernel" in result

    def test_device_function_conversion(self):
        """Test device function conversion"""
        code = """
        __device__ float add(float a, float b) {
            return a + b;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// CUDA to CrossGL conversion" in result
        assert "// Function: add" in result

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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "f32 add(f32 a, f32 b) {" in result
        assert "return (a + b);" in result
        assert "out[0] = add(1.0f, 2.0f);" in result

    def test_multiple_kernels_conversion(self):
        """Test multiple kernels conversion"""
        code = """
        __global__ void kernel1(float* data) {
            data[threadIdx.x] = 1.0f;
        }
        
        __global__ void kernel2(int* data) {
            data[threadIdx.x] = 2;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// Kernel: kernel1" in result
        assert "// Kernel: kernel2" in result

    def test_type_conversion(self):
        """Test CUDA type to CrossGL type conversion"""
        codegen = CudaToCrossGLConverter()

        # Test basic types
        assert codegen.convert_cuda_type_to_crossgl("int") == "i32"
        assert codegen.convert_cuda_type_to_crossgl("float") == "f32"
        assert codegen.convert_cuda_type_to_crossgl("double") == "f64"
        assert codegen.convert_cuda_type_to_crossgl("bool") == "bool"
        assert codegen.convert_cuda_type_to_crossgl("void") == "void"
        assert codegen.convert_cuda_type_to_crossgl("long long") == "i64"
        assert codegen.convert_cuda_type_to_crossgl("unsigned long long") == "u64"
        assert codegen.convert_cuda_type_to_crossgl("float *") == "ptr<f32>"
        assert (
            codegen.convert_cuda_type_to_crossgl("unsigned long long *") == "ptr<u64>"
        )
        assert codegen.convert_cuda_type_to_crossgl("void * *") == "ptr<ptr<void>>"
        assert codegen.convert_cuda_type_to_crossgl("void * []") == "array<ptr<void>>"

    def test_constructor_style_vector_declaration_conversion(self):
        """Test CUDA constructor-style vector declarations convert"""
        code = """
        void launch() {
            dim3 grid(16, 8, 1);
            dim3 block(32);
            float3 v(1.0f, 2.0f, 3.0f);
            double2 d = make_double2(1.0, 2.0);
            uint4 ids = make_uint4(1u, 2u, 3u, 4u);
            uchar2 bytes(1, 2);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var grid: vec3<u32> = vec3<u32>(16, 8, 1);" in result
        assert "var block: vec3<u32> = vec3<u32>(32);" in result
        assert "var v: vec3<f32> = vec3<f32>(1.0f, 2.0f, 3.0f);" in result
        assert "var d: vec2<f64> = vec2<f64>(1.0, 2.0);" in result
        assert "var ids: vec4<u32> = vec4<u32>(1u, 2u, 3u, 4u);" in result
        assert "var bytes: vec2<u8> = vec2<u8>(1, 2);" in result

    def test_fmod_builtins_convert_to_crossgl_mod(self):
        """Test CUDA fmod functions convert back to CrossGL mod."""
        code = """
        __global__ void wrap(float* out, float x) {
            out[0] = fmodf(x, 1.0f);
            out[1] = fmod(x, 2.0);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "out[0] = mod(x, 1.0f);" in result
        assert "out[1] = mod(x, 2.0);" in result
        assert "fmod" not in result

    def test_atan2_builtins_convert_to_crossgl_atan2(self):
        """Test CUDA atan2f converts back to CrossGL atan2."""
        code = """
        __global__ void angle(float* out, float y, float x) {
            out[0] = atan2f(y, x);
            out[1] = atan2(y, x);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "out[0] = atan2(y, x);" in result
        assert "out[1] = atan2(y, x);" in result
        assert "atan2f" not in result

    def test_lerp_builtin_converts_to_crossgl_mix(self):
        """Test CUDA lerp converts back to CrossGL mix."""
        code = """
        __global__ void blend(float* out, float a, float b, float t) {
            out[0] = lerp(a, b, t);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "out[0] = mix(a, b, t);" in result
        assert "out[0] = lerp(a, b, t);" not in result

    def test_lerp_with_bool_selector_converts_to_crossgl_mix(self):
        """Test CUDA lerp conversion preserves selector-shaped mix calls."""
        code = """
        __global__ void blend(float* out, float a, float b, bool choose_b) {
            out[0] = lerp(a, b, choose_b);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "out[0] = mix(a, b, choose_b);" in result
        assert "out[0] = lerp(a, b, choose_b);" not in result

    def test_scalar_bool_ternary_selector_is_preserved(self):
        """Test CUDA scalar bool ternaries round-trip as CrossGL ternaries."""
        code = """
        __global__ void choose(float* out, bool choose_b, float a, float b) {
            out[0] = choose_b ? b : a;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "out[0] = (choose_b ? b : a);" in result

    def test_user_defined_lerp_call_does_not_convert_to_mix(self):
        """Test user-defined CUDA functions shadow builtin call conversion."""
        code = """
        float lerp(float x) {
            return x;
        }

        float tex2D(float x) {
            return x + 1.0f;
        }

        __global__ void kernel(float* out, float x) {
            out[0] = lerp(x);
            out[1] = tex2D(x);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "f32 lerp(f32 x) {" in result
        assert "f32 tex2D(f32 x) {" in result
        assert "out[0] = lerp(x);" in result
        assert "out[1] = tex2D(x);" in result
        assert "out[0] = mix(x);" not in result
        assert "out[1] = texture(x);" not in result

    def test_user_defined_cuda_atomic_name_call_does_not_convert_to_builtin(self):
        """Test user-defined CUDA atomic names shadow builtin conversion."""
        code = """
        int atomicExch(int value) {
            return value + 1;
        }

        __global__ void kernel(int* out, int* expected) {
            int value = atomicExch(7);
            atomicCAS(expected, 0, 3);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "i32 atomicExch(i32 value)" in result
        assert "var value: i32 = atomicExch(7);" in result
        assert "atomicCompareExchange(expected, 0, 3);" in result
        assert "atomicExchange(7)" not in result

    def test_user_defined_cuda_atomic_name_declared_later_does_not_convert_to_builtin(
        self,
    ):
        """Test later user-defined CUDA atomic names shadow builtin conversion."""
        code = """
        __global__ void kernel(int* out, int* expected) {
            int value = atomicExch(7);
            atomicCAS(expected, 0, 3);
        }

        int atomicExch(int value) {
            return value + 1;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "i32 atomicExch(i32 value)" in result
        assert "var value: i32 = atomicExch(7);" in result
        assert "atomicCompareExchange(expected, 0, 3);" in result
        assert "atomicExchange(7)" not in result

    def test_address_taken_pointer_array_and_shared_atomics_lower_to_lvalues(self):
        """Test CUDA pointer atomics convert address-taken targets to CrossGL lvalues."""
        code = """
        __global__ void kernel(int* values, int* expected, int* desired, int index) {
            __shared__ int sharedCounts[32];
            atomicAdd(&values[index], 1);
            int old = atomicCAS(&values[index], expected[index], desired[index]);
            atomicMax(&sharedCounts[threadIdx.x], old);
            atomicExch(&values[index + 1], old);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var<workgroup> sharedCounts: array<i32, 32>;" in result
        assert "atomicAdd(values[index], 1);" in result
        assert (
            "var old: i32 = atomicCompareExchange("
            "values[index], expected[index], desired[index]);"
        ) in result
        assert "atomicMax(sharedCounts[gl_LocalInvocationID.x], old);" in result
        assert "atomicExchange(values[(index + 1)], old);" in result
        assert "(&values[index])" not in result
        assert "(&sharedCounts[gl_LocalInvocationID.x])" not in result

    def test_bitwise_atomics_lower_to_crossgl_lvalue_targets(self):
        """Test CUDA bitwise atomics convert to CrossGL atomic calls."""
        code = """
        __global__ void kernel(unsigned int* values, unsigned int mask, int index) {
            __shared__ unsigned int sharedMasks[32];
            unsigned int oldAnd = atomicAnd(&values[index], mask);
            unsigned int oldOr = atomicOr(&sharedMasks[threadIdx.x], oldAnd);
            atomicXor(&values[index + 1], oldOr);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var<workgroup> sharedMasks: array<u32, 32>;" in result
        assert "var oldAnd: u32 = atomicAnd(values[index], mask);" in result
        expected_old_or = (
            "var oldOr: u32 = atomicOr(sharedMasks[gl_LocalInvocationID.x], oldAnd);"
        )
        assert expected_old_or in result
        assert "atomicXor(values[(index + 1)], oldOr);" in result
        assert "(&values[index])" not in result
        assert "(&sharedMasks[gl_LocalInvocationID.x])" not in result

    def test_bounded_wrap_atomics_preserve_cuda_semantics(self):
        """Test CUDA atomicInc/atomicDec preserve bounded wrap intrinsics."""
        code = """
        __global__ void kernel(unsigned int* values, unsigned int limit, int index) {
            __shared__ unsigned int sharedCounters[32];
            unsigned int oldInc = atomicInc(&values[index], limit);
            unsigned int oldDec = atomicDec(&sharedCounters[threadIdx.x], limit);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var<workgroup> sharedCounters: array<u32, 32>;" in result
        assert "var oldInc: u32 = atomicInc(values[index], limit);" in result
        expected_old_dec = (
            "var oldDec: u32 = "
            "atomicDec(sharedCounters[gl_LocalInvocationID.x], limit);"
        )
        assert expected_old_dec in result
        assert "atomicAdd(values[index], limit)" not in result
        assert "atomicSub(sharedCounters[gl_LocalInvocationID.x], limit)" not in result
        assert "(&values[index])" not in result
        assert "(&sharedCounters[gl_LocalInvocationID.x])" not in result

    def test_user_defined_cuda_runtime_call_does_not_emit_runtime_comment(self):
        """Test user-defined CUDA runtime names shadow runtime call comments."""
        code = """
        void cudaFree(float* p) {
            p[0] = 1.0f;
        }

        __global__ void kernel(float* out) {
            cudaFree(out);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "void cudaFree(ptr<f32> p)" in result
        assert "cudaFree(out);" in result
        assert "// CUDA memory free: out" not in result

    def test_threadfence_converts_to_crossgl_memory_barrier(self):
        """Test CUDA thread fence variants convert back to CrossGL memoryBarrier."""
        code = """
        __global__ void fence(float* out) {
            __threadfence();
            __threadfence_block();
            __threadfence_system();
            __syncthreads();
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert result.count("memoryBarrier();") == 3
        assert "workgroupBarrier();" in result
        assert "__threadfence" not in result

    def test_syncwarp_mask_emits_explicit_diagnostic(self):
        """Test CUDA warp synchronization emits an explicit CrossGL diagnostic."""
        code = """
        __global__ void sync(unsigned int mask) {
            __syncwarp(mask);
            __syncwarp();
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// __syncwarp(mask) not directly supported in CrossGL" in result
        assert "// __syncwarp() not directly supported in CrossGL" in result
        assert "None" not in result

    def test_inverse_trig_builtins_convert_to_crossgl(self):
        """Test CUDA inverse trig functions convert back to CrossGL names."""
        code = """
        __global__ void inverse(float* out, float x) {
            out[0] = asinf(x);
            out[1] = acosf(x);
            out[2] = atanf(x);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "out[0] = asin(x);" in result
        assert "out[1] = acos(x);" in result
        assert "out[2] = atan(x);" in result
        assert "asinf" not in result
        assert "acosf" not in result
        assert "atanf" not in result

    def test_hyperbolic_builtins_convert_to_crossgl(self):
        """Test CUDA hyperbolic functions convert back to CrossGL names."""
        code = """
        __global__ void hyper(float* out, float x) {
            out[0] = sinhf(x);
            out[1] = coshf(x);
            out[2] = tanhf(x);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "out[0] = sinh(x);" in result
        assert "out[1] = cosh(x);" in result
        assert "out[2] = tanh(x);" in result
        assert "sinhf" not in result
        assert "coshf" not in result
        assert "tanhf" not in result

    def test_extended_math_builtins_convert_to_crossgl(self):
        """Test CUDA extended math functions convert back to CrossGL names."""
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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

    def test_kernel_launch_conversion(self):
        """Test CUDA kernel launch configuration conversion"""
        code = """
        void host(float* data, int stream) {
            dim3 grid(16);
            dim3 block(32);
            kernel<<<grid, block, 128, stream>>>(data, 1);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// Kernel launch: kernel<<<grid, block, 128, stream>>>()" in result
        assert "// Arguments: data, 1" in result

    def test_templated_kernel_launch_conversion(self):
        """Test CUDA template-id kernel launch conversion"""
        code = """
        template <typename T>
        __global__ void scale(T* data, T factor) {
            data[threadIdx.x] *= factor;
        }

        void host(float* data) {
            scale<float><<<1, 32>>>(data, 2.0f);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// Kernel launch: scale<float><<<1, 32>>>()" in result
        assert "// Arguments: data, 2.0f" in result

    def test_computed_kernel_launch_config_conversion(self):
        """Test CUDA computed kernel launch configuration conversion"""
        code = """
        void host(float* data, int n, int stream) {
            int blockSize = 128;
            kernel<<<(n + blockSize - 1) / blockSize,
                     blockSize,
                     sizeof(float) * blockSize,
                     stream>>>(data, n);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// Kernel launch: kernel<<<(((n + blockSize) - 1) / blockSize), "
            "blockSize, (sizeof(float) * blockSize), stream>>>()"
        ) in result
        assert "// Arguments: data, n" in result

    def test_cuda_launch_kernel_api_conversion(self):
        """Test cudaLaunchKernel converts through kernel launch metadata"""
        code = """
        void host(float* data, int n, int stream) {
            dim3 grid(16);
            dim3 block(32);
            void* packedArgs[] = { &data, &n };
            cudaLaunchKernel((void*)kernel, grid, block, packedArgs, 0, stream);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var packedArgs: array<ptr<void>> = {(&data), (&n)};" in result
        assert "// Kernel launch: kernel<<<grid, block, 0, stream>>>()" in result
        assert "// Arguments: data, n" in result
        assert "cudaLaunchKernel" not in result

    def test_user_defined_cuda_launch_kernel_call_is_not_kernel_launch(self):
        """Test user-defined cudaLaunchKernel calls are not launch metadata"""
        code = """
        void cudaLaunchKernel(float* out, int grid, int block, void* args, int shared, int stream) {
            return;
        }

        void host(float* out, int grid, int block, void* args) {
            cudaLaunchKernel(out, grid, block, args, 0, 0);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// Function: cudaLaunchKernel" in result
        assert "void cudaLaunchKernel(" in result
        assert "cudaLaunchKernel(out, grid, block, args, 0, 0);" in result
        assert "// Kernel launch: out<<<grid, block, 0, 0>>>()" not in result

    def test_user_defined_cuda_launch_kernel_declared_later_is_not_kernel_launch(self):
        """Test later user-defined cudaLaunchKernel shadows launch metadata."""
        code = """
        void host(float* out, int grid, int block, void* args) {
            cudaLaunchKernel(out, grid, block, args, 0, 0);
        }

        void cudaLaunchKernel(float* out, int grid, int block, void* args, int shared, int stream) {
            return;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// Function: cudaLaunchKernel" in result
        assert "void cudaLaunchKernel(" in result
        assert "cudaLaunchKernel(out, grid, block, args, 0, 0);" in result
        assert "// Kernel launch: out<<<grid, block, 0, 0>>>()" not in result

    def test_cuda_launch_kernel_casted_packed_args_conversion(self):
        """Test casted packed args still expand in cudaLaunchKernel"""
        code = """
        void host(float* data, int n, int stream) {
            dim3 grid(16);
            dim3 block(32);
            void** packedArgs = { &data, &n };
            cudaLaunchKernel((void*)kernel, grid, block, (void**)packedArgs, 0, stream);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var packedArgs: ptr<ptr<void>> = {(&data), (&n)};" in result
        assert "// Kernel launch: kernel<<<grid, block, 0, stream>>>()" in result
        assert "// Arguments: data, n" in result
        assert "ptr<ptr<void>>(packedArgs)" not in result

    def test_cuda_launch_kernel_compound_literal_args_conversion(self):
        """Test compound literal packed args expand in cudaLaunchKernel"""
        code = """
        void host(float* data, int n, int stream) {
            dim3 grid(16);
            dim3 block(32);
            cudaLaunchKernel((void*)kernel, grid, block,
                             (void*[]){ &data, &n }, 0, stream);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
            cudaLaunchKernel((void*)kernel, grid, block, packedArgs, 0, stream);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var values: array<f32, 2> = {1.0f, 2.0f};" in result
        assert "// Arguments: values" in result
        assert "// Arguments: 1.0f, 2.0f" not in result

    def test_cuda_runtime_memory_api_conversion(self):
        """Test CUDA runtime memory API calls emit metadata comments"""
        code = """
        void host(float* h, int n) {
            float* d;
            cudaMalloc((void**)&d, n * sizeof(float));
            cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemset(d, 0, n * sizeof(float));
            cudaDeviceSynchronize();
            cudaFree(d);
            cudaError_t err = cudaMalloc((void**)&d, n * sizeof(float));
            if (err != cudaSuccess) { return; }
            err = cudaDeviceSynchronize();
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            result.count("// CUDA memory allocate: d, bytes: (n * sizeof(float))") == 2
        )
        assert (
            "// CUDA memory copy: h -> d, bytes: (n * sizeof(float)), "
            "kind: cudaMemcpyHostToDevice"
        ) in result
        assert "// CUDA memory set: d, value: 0, bytes: (n * sizeof(float))" in result
        assert result.count("// CUDA device synchronize") == 2
        assert "// CUDA memory free: d" in result
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "if ((err != cudaSuccess))" in result
        assert "err = cudaSuccess;" in result
        assert "cudaMalloc(ptr<ptr<void>>((&d)), (n * sizeof(float)))" not in result
        assert "err = cudaDeviceSynchronize();" not in result

    def test_cuda_runtime_memset_async_conversion(self):
        """Test cudaMemsetAsync emits metadata comments and status success"""
        code = """
        void host(float* d, int n, cudaStream_t stream) {
            cudaMemsetAsync(d, 0, n * sizeof(float), stream);
            cudaError_t err = cudaMemsetAsync(d, 1, n * sizeof(float), stream);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA memory set: d, value: 0, bytes: (n * sizeof(float)), "
            "stream: stream"
        ) in result
        assert (
            "// CUDA memory set: d, value: 1, bytes: (n * sizeof(float)), "
            "stream: stream"
        ) in result
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "cudaMemsetAsync(" not in result

    def test_cuda_runtime_last_error_query_conversion(self):
        """Test CUDA runtime error query calls emit metadata comments"""
        code = """
        void host() {
            cudaGetLastError();
            cudaError_t err = cudaGetLastError();
            err = cudaPeekAtLastError();
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert result.count("// CUDA get last error") == 2
        assert "// CUDA peek at last error" in result
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "err = cudaSuccess;" in result
        assert "cudaGetLastError(" not in result
        assert "cudaPeekAtLastError(" not in result

    def test_cuda_runtime_event_api_conversion(self):
        """Test CUDA stream and event API calls emit metadata comments"""
        code = """
        void bench() {
            cudaStream_t stream;
            cudaEvent_t start;
            cudaEvent_t stop;
            float ms;
            cudaStreamCreate(&stream);
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            cudaEventCreate(&start);
            cudaEventCreateWithFlags(&stop, cudaEventDisableTiming);
            cudaEventRecord(start, stream);
            cudaStreamWaitEvent(stream, start, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms, start, stop);
            cudaEventQuery(stop);
            cudaEventDestroy(start);
            cudaStreamDestroy(stream);
            cudaError_t err = cudaEventRecord(stop, stream);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// CUDA stream create: stream" in result
        assert "// CUDA stream create: stream, flags: cudaStreamNonBlocking" in result
        assert "// CUDA event create: start" in result
        assert "// CUDA event create: stop, flags: cudaEventDisableTiming" in result
        assert "// CUDA event record: start, stream: stream" in result
        assert "// CUDA event record: stop, stream: stream" in result
        assert "// CUDA stream wait event: stream waits for start, flags: 0" in result
        assert "// CUDA event synchronize: stop" in result
        assert "// CUDA event elapsed time: start -> stop, output: ms" in result
        assert "// CUDA event query: stop" in result
        assert "// CUDA event destroy: start" in result
        assert "// CUDA stream destroy: stream" in result
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "cudaEventRecord(stop, stream)" not in result
        assert "cudaStreamCreateWithFlags(" not in result

    def test_cuda_runtime_stream_create_with_priority_status_conversion(self):
        """Test cudaStreamCreateWithPriority emits priority metadata"""
        code = """
        void bench() {
            cudaStream_t stream;
            cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, 3);
            cudaError_t err = cudaStreamCreateWithPriority(
                &stream, cudaStreamNonBlocking, 3
            );
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            result.count(
                "// CUDA stream create: stream, flags: cudaStreamNonBlocking, "
                "priority: 3"
            )
            == 2
        )
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "cudaStreamCreateWithPriority(" not in result

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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
            cudaMemcpy(d, h.data(), h.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
            bool ordered = h.size() < n;
            std::chrono::high_resolution_clock::now();
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var h: std::vector<float> = std::vector<float>(n);" in result
        assert (
            "// CUDA memory copy: h.data() -> d, bytes: "
            "(h.size() * sizeof(float)), kind: cudaMemcpyHostToDevice"
        ) in result
        assert "var ordered: bool = (h.size() < n);" in result
        assert "std::chrono::high_resolution_clock::now();" in result

    def test_std_array_host_buffer_conversion(self):
        """Test std::array host buffers convert in memory copy metadata"""
        code = """
        void host(float* d) {
            std::array<float, 4> h{1.0f, 2.0f, 3.0f, 4.0f};
            std::array<float, 4> zeros{};
            cudaMemcpy(d, h.data(), h.size() * sizeof(float),
                       cudaMemcpyHostToDevice);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var h: std::array<float, 4> = {1.0f, 2.0f, 3.0f, 4.0f};" in result
        assert "var zeros: std::array<float, 4> = {};" in result
        assert (
            "// CUDA memory copy: h.data() -> d, bytes: "
            "(h.size() * sizeof(float)), kind: cudaMemcpyHostToDevice"
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "for (; (i < j); (++i), (--j)) {" in result
        assert "sink(i);" in result

    def test_grid_stride_for_update_compound_assignment_conversion(self):
        """Test CUDA grid-stride for-loop updates lower into CrossGL"""
        code = """
        __global__ void kernel(float* data, int n) {
            for (int i = blockIdx.x * blockDim.x + threadIdx.x;
                 i < n;
                 i += blockDim.x * gridDim.x) {
                data[i] = 0.0f;
            }
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "for (var i: i32 = ((gl_WorkGroupID.x * gl_WorkGroupSize.x) + "
            "gl_LocalInvocationID.x); (i < n); i += "
            "(gl_WorkGroupSize.x * gl_NumWorkGroups.x)) {"
        ) in result
        assert "data[i] = 0.0f;" in result

    def test_assignment_expression_conversion(self):
        """Test nested CUDA assignment expressions convert without stray output"""
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "a = b = c;" in result
        assert "var d: i32 = a = b;" in result
        assert "sink(a = 1);" in result
        assert "None" not in result

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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "(value * scale);" in result
        assert "var x: auto = h[0];" in result
        assert "var p: ptr<auto> = data;" in result
        assert "var cp: ptr<auto> = data;" in result
        assert "var q: ptr<auto> = data;" in result
        assert "var r: ptr<auto> = data;" in result

    def test_device_lambda_expression_conversion(self):
        """Test CUDA device lambdas convert to CrossGL pseudo-lambda calls."""
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
            cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(float));
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var x: f32 = f32(i);" in result
        assert "var p: ptr<f32> = ptr<f32>(input);" in result
        assert "// CUDA memory allocate: data, bytes: (n * sizeof(float))" in result
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
        struct ResourceHolder {
            cudaTextureObject_t sharedName;
        };

        cudaTextureObject_t sharedName;
        cudaTextureObject_t globalTex;
        cudaSurfaceObject_t globalSurface;
        cudaSurfaceObject_t globalCubeLayerSurface;
        cudaTextureObject_t globalTextures[2];
        cudaTextureObject_t mixedGlobal;

        void host() {
            std::array<unsigned int, 4> ids{};
            std::vector<const unsigned int> flags;
            texture<float4, 2> tex2d;
            texture<float4, cudaTextureType2DLayered> layeredTex;
            texture<float4, cudaTextureTypeCubemap> cubeTex;
            surface<void, 1> lineSurface;
            surface<void, cudaSurfaceType1DLayered> lineLayerSurface;
            surface<void, 2> surface2d;
            surface<void, cudaSurfaceType3D> volumeSurface;
            surface<void, cudaSurfaceTypeCubemap> cubeSurface;
            surface<void, cudaSurfaceTypeCubemapLayered> cubeLayerSurface;
            cudaArray_t arrayRef;
            cudaArray* rawArray;
        }

        void resourceOps(
            texture<float4, 1> tex1d,
            texture<float4, 2> tex2d,
            texture<float4, cudaTextureType3D> tex3d,
            texture<float4, cudaTextureType2DLayered> layeredTex,
            texture<float4, cudaTextureTypeCubemap> cubeTex,
            texture<float4, cudaTextureTypeCubemapLayered> cubeLayerTex,
            surface<void, 1> lineSurface,
            surface<void, cudaSurfaceType1DLayered> lineLayerSurface,
            surface<void, 2> surface2d,
            surface<void, cudaSurfaceType3D> volumeSurface,
            surface<void, cudaSurfaceTypeCubemap> cubeSurface,
            surface<void, cudaSurfaceTypeCubemapLayered> cubeLayerSurface,
            cudaTextureObject_t objectTex,
            cudaSurfaceObject_t objectSurface,
            cudaTextureObject_t objectTextures[2],
            cudaSurfaceObject_t objectSurfaces[2],
            cudaTextureObject_t ambiguousTex,
            int2 pixel,
            int3 voxel,
            float2 uv,
            float3 uvw
        ) {
            float4 line = tex1D<float4>(tex1d, uv.x);
            float4 sampled = tex2D<float4>(tex2d, uv.x, uv.y);
            float4 sampledCoord = tex2D<float4>(tex2d, uv);
            float4 sampledLod = tex2DLod<float4>(tex2d, uv.x, uv.y, 1.0f);
            float4 sampledGrad = tex2DGrad<float4>(tex2d, uv, uv, uv);
            float4 layer = tex2DLayered<float4>(
                layeredTex,
                uv.x,
                uv.y,
                pixel.x
            );
            float4 volume = tex3D<float4>(tex3d, uvw.x, uvw.y, uvw.z);
            float4 cube = texCubemap<float4>(cubeTex, uvw.x, uvw.y, uvw.z);
            float4 cubeLayer = texCubemapLayered<float4>(
                cubeLayerTex,
                uvw.x,
                uvw.y,
                uvw.z,
                pixel.x
            );
            float4 lineRead = surf1Dread<float4>(
                lineSurface,
                pixel.x * sizeof(float4)
            );
            surf1Dwrite(lineRead, lineSurface, pixel.x * sizeof(float4));
            float4 lineLayerRead = surf1DLayeredread<float4>(
                lineLayerSurface,
                pixel.x * sizeof(float4),
                pixel.y
            );
            surf1DLayeredwrite(
                lineLayerRead,
                lineLayerSurface,
                pixel.x * sizeof(float4),
                pixel.y
            );
            float4 read = surf2Dread<float4>(
                surface2d,
                pixel.x * sizeof(float4),
                pixel.y
            );
            surf2Dwrite(read, surface2d, pixel.x * sizeof(float4), pixel.y);
            float4 voxelValue = surf3Dread<float4>(
                volumeSurface,
                voxel.x * sizeof(float4),
                voxel.y,
                voxel.z
            );
            surf3Dwrite(
                voxelValue,
                volumeSurface,
                voxel.x * sizeof(float4),
                voxel.y,
                voxel.z
            );
            float4 cubeRead = surfCubemapread<float4>(
                cubeSurface,
                pixel.x * sizeof(float4),
                pixel.y,
                voxel.z
            );
            surfCubemapwrite(
                cubeRead,
                cubeSurface,
                pixel.x * sizeof(float4),
                pixel.y,
                voxel.z
            );
            float4 cubeLayerRead = surfCubemapLayeredread<float4>(
                cubeLayerSurface,
                pixel.x * sizeof(float4),
                pixel.y,
                voxel.z
            );
            surfCubemapLayeredwrite(
                cubeLayerRead,
                cubeLayerSurface,
                pixel.x * sizeof(float4),
                pixel.y,
                voxel.z
            );
            float4 objectSample = tex2D<float4>(objectTex, uv.x, uv.y);
            float4 objectSampleArray = tex2D<float4>(objectTextures[1], uv);
            float4 objectRead = surf2Dread<float4>(
                objectSurface,
                pixel.x * sizeof(float4),
                pixel.y
            );
            surf2Dwrite(
                objectRead,
                objectSurfaces[1],
                pixel.x * sizeof(float4),
                pixel.y
            );
            float4 ambiguousSample = tex2D<float4>(ambiguousTex, uv.x, uv.y);
            float4 ambiguousVolume = tex3D<float4>(
                ambiguousTex,
                uvw.x,
                uvw.y,
                uvw.z
            );
        }

        void localResourceObjects(int2 pixel, float2 uv) {
            cudaTextureObject_t localTex;
            cudaSurfaceObject_t localSurface;
            float4 sampled = tex2D<float4>(localTex, uv);
            float4 loaded = surf2Dread<float4>(
                localSurface,
                pixel.x * sizeof(float4),
                pixel.y
            );
        }

        void sample2DObject(cudaTextureObject_t sharedName, float2 uv) {
            float4 sampled = tex2D<float4>(sharedName, uv);
        }

        void sample3DObject(cudaTextureObject_t sharedName, float3 uvw) {
            float4 sampled = tex3D<float4>(
                sharedName,
                uvw.x,
                uvw.y,
                uvw.z
            );
        }

        void sampleGlobals(int2 pixel, float2 uv, float3 uvw) {
            float4 sampled = tex2D<float4>(globalTex, uv);
            float4 arraySample = tex2D<float4>(globalTextures[1], uv);
            float4 loaded = surf2Dread<float4>(
                globalSurface,
                pixel.x * sizeof(float4),
                pixel.y
            );
            float4 cubeLayerLoaded = surfCubemapLayeredread<float4>(
                globalCubeLayerSurface,
                pixel.x * sizeof(float4),
                pixel.y,
                voxel.z
            );
            float4 mixedSample = tex2D<float4>(mixedGlobal, uv);
            float4 mixedVolume = tex3D<float4>(
                mixedGlobal,
                uvw.x,
                uvw.y,
                uvw.z
            );
        }

        void shadowGlobal(cudaTextureObject_t globalTex, float3 uvw) {
            float4 sampled = tex3D<float4>(
                globalTex,
                uvw.x,
                uvw.y,
                uvw.z
            );
        }

        void shadowUnused(cudaSurfaceObject_t globalSurface) {
            return;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var ids: std::array<unsigned int, 4> = {};" in result
        assert "var flags: std::vector<const unsigned int>;" in result
        assert "cudaTextureObject_t sharedName;" in result
        assert "var sharedName: cudaTextureObject_t;" in result
        assert "var globalTex: sampler2D;" in result
        assert "var globalSurface: image2D;" in result
        assert "var globalCubeLayerSurface: imageCubeArray;" in result
        assert "var globalTextures: array<sampler2D, 2>;" in result
        assert "var mixedGlobal: cudaTextureObject_t;" in result
        assert "var tex2d: sampler2D;" in result
        assert "var layeredTex: sampler2DArray;" in result
        assert "var cubeTex: samplerCube;" in result
        assert "var lineSurface: image1D;" in result
        assert "var lineLayerSurface: image1DArray;" in result
        assert "var surface2d: image2D;" in result
        assert "var volumeSurface: image3D;" in result
        assert "var cubeSurface: imageCube;" in result
        assert "var cubeLayerSurface: imageCubeArray;" in result
        assert "var arrayRef: cudaArray_t;" in result
        assert "var rawArray: ptr<cudaArray>;" in result
        assert "void resourceOps(" in result
        assert "sampler1D tex1d" in result
        assert "sampler2D tex2d" in result
        assert "sampler3D tex3d" in result
        assert "sampler2DArray layeredTex" in result
        assert "samplerCube cubeTex" in result
        assert "samplerCubeArray cubeLayerTex" in result
        assert "image1D lineSurface" in result
        assert "image1DArray lineLayerSurface" in result
        assert "image2D surface2d" in result
        assert "image3D volumeSurface" in result
        assert "imageCube cubeSurface" in result
        assert "imageCubeArray cubeLayerSurface" in result
        assert "sampler2D objectTex" in result
        assert "image2D objectSurface" in result
        assert "array<sampler2D, 2> objectTextures" in result
        assert "array<image2D, 2> objectSurfaces" in result
        assert "cudaTextureObject_t ambiguousTex" in result
        assert "var line: vec4<f32> = texture(tex1d, uv.x);" in result
        assert (
            "var sampled: vec4<f32> = texture(tex2d, vec2<f32>(uv.x, uv.y));" in result
        )
        assert "var sampledCoord: vec4<f32> = texture(tex2d, uv);" in result
        assert (
            "var sampledLod: vec4<f32> = textureLod("
            "tex2d, vec2<f32>(uv.x, uv.y), 1.0f);" in result
        )
        assert "var sampledGrad: vec4<f32> = textureGrad(tex2d, uv, uv, uv);" in result
        assert (
            "var layer: vec4<f32> = texture("
            "layeredTex, vec3<f32>(uv.x, uv.y, pixel.x));" in result
        )
        assert (
            "var volume: vec4<f32> = texture(tex3d, "
            "vec3<f32>(uvw.x, uvw.y, uvw.z));" in result
        )
        assert (
            "var cube: vec4<f32> = texture(cubeTex, "
            "vec3<f32>(uvw.x, uvw.y, uvw.z));" in result
        )
        assert (
            "var cubeLayer: vec4<f32> = texture("
            "cubeLayerTex, vec4<f32>(uvw.x, uvw.y, uvw.z, pixel.x));" in result
        )
        assert "var lineRead: vec4<f32> = imageLoad(lineSurface, pixel.x);" in result
        assert "imageStore(lineSurface, pixel.x, lineRead);" in result
        assert (
            "var lineLayerRead: vec4<f32> = imageLoad("
            "lineLayerSurface, vec2<i32>(pixel.x, pixel.y));" in result
        )
        assert (
            "imageStore(lineLayerSurface, vec2<i32>(pixel.x, pixel.y), "
            "lineLayerRead);" in result
        )
        assert (
            "var read: vec4<f32> = imageLoad("
            "surface2d, vec2<i32>(pixel.x, pixel.y));" in result
        )
        assert "imageStore(surface2d, vec2<i32>(pixel.x, pixel.y), read);" in result
        assert (
            "var voxelValue: vec4<f32> = imageLoad("
            "volumeSurface, vec3<i32>(voxel.x, voxel.y, voxel.z));" in result
        )
        assert (
            "imageStore(volumeSurface, vec3<i32>(voxel.x, voxel.y, voxel.z), "
            "voxelValue);" in result
        )
        assert (
            "var cubeRead: vec4<f32> = imageLoad("
            "cubeSurface, vec3<i32>(pixel.x, pixel.y, voxel.z));" in result
        )
        assert (
            "imageStore(cubeSurface, vec3<i32>(pixel.x, pixel.y, voxel.z), "
            "cubeRead);" in result
        )
        assert (
            "var cubeLayerRead: vec4<f32> = imageLoad("
            "cubeLayerSurface, vec3<i32>(pixel.x, pixel.y, voxel.z));" in result
        )
        assert (
            "imageStore(cubeLayerSurface, vec3<i32>(pixel.x, pixel.y, voxel.z), "
            "cubeLayerRead);" in result
        )
        assert (
            "var objectSample: vec4<f32> = texture("
            "objectTex, vec2<f32>(uv.x, uv.y));" in result
        )
        assert (
            "var objectSampleArray: vec4<f32> = texture(objectTextures[1], uv);"
            in result
        )
        assert (
            "var objectRead: vec4<f32> = imageLoad("
            "objectSurface, vec2<i32>(pixel.x, pixel.y));" in result
        )
        assert (
            "imageStore(objectSurfaces[1], vec2<i32>(pixel.x, pixel.y), "
            "objectRead);" in result
        )
        assert (
            "var ambiguousVolume: vec4<f32> = texture(ambiguousTex, "
            "vec3<f32>(uvw.x, uvw.y, uvw.z));" in result
        )
        assert "void localResourceObjects(vec2<i32> pixel, vec2<f32> uv) {" in result
        assert "var localTex: sampler2D;" in result
        assert "var localSurface: image2D;" in result
        assert "void sample2DObject(sampler2D sharedName, vec2<f32> uv) {" in result
        assert "void sample3DObject(sampler3D sharedName, vec3<f32> uvw) {" in result
        assert "var arraySample: vec4<f32> = texture(globalTextures[1], uv);" in result
        assert (
            "var loaded: vec4<f32> = imageLoad("
            "globalSurface, vec2<i32>(pixel.x, pixel.y));" in result
        )
        assert (
            "var cubeLayerLoaded: vec4<f32> = imageLoad("
            "globalCubeLayerSurface, vec3<i32>(pixel.x, pixel.y, voxel.z));" in result
        )
        assert "void shadowGlobal(sampler3D globalTex, vec3<f32> uvw) {" in result
        assert "void shadowUnused(cudaSurfaceObject_t globalSurface) {" in result
        assert "unsignedint" not in result
        assert "constunsigned" not in result

    def test_resource_object_pointer_and_legacy_texture_template_conversion(self):
        """Test CUDA resource object pointers and legacy texture refs convert."""
        code = """
        texture<float4, cudaTextureType2D, cudaReadModeElementType> texRef;
        surface<void, cudaSurfaceType2D> surfaceRef;

        void resourcePointerOps(
            cudaTextureObject_t* textureObjects,
            cudaSurfaceObject_t* surfaceObjects,
            texture<float4, cudaTextureType2D, cudaReadModeElementType> legacyTex,
            surface<void, cudaSurfaceType2D> legacySurface,
            int index,
            int2 pixel,
            float2 uv
        ) {
            float4 fromPointer = tex2D<float4>(
                textureObjects[index],
                uv.x,
                uv.y
            );
            float4 fromLegacy = tex2D<float4>(legacyTex, uv);
            float4 fromGlobal = tex2D<float4>(texRef, uv);
            float4 loadedPointer = surf2Dread<float4>(
                surfaceObjects[index],
                pixel.x * sizeof(float4),
                pixel.y
            );
            surf2Dwrite(
                loadedPointer,
                surfaceObjects[index],
                pixel.x * sizeof(float4),
                pixel.y
            );
            float4 loadedLegacy = surf2Dread<float4>(
                legacySurface,
                pixel.x * sizeof(float4),
                pixel.y
            );
            surf2Dwrite(
                loadedLegacy,
                legacySurface,
                pixel.x * sizeof(float4),
                pixel.y
            );
            float4 loadedGlobal = surf2Dread<float4>(
                surfaceRef,
                pixel.x * sizeof(float4),
                pixel.y
            );
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        result = CudaToCrossGLConverter().generate(ast)

        assert "var texRef: sampler2D;" in result
        assert "var surfaceRef: image2D;" in result
        assert (
            "void resourcePointerOps("
            "ptr<sampler2D> textureObjects, ptr<image2D> surfaceObjects, "
            "sampler2D legacyTex, image2D legacySurface, i32 index"
        ) in result
        assert (
            "var fromPointer: vec4<f32> = texture("
            "textureObjects[index], vec2<f32>(uv.x, uv.y));" in result
        )
        assert "var fromLegacy: vec4<f32> = texture(legacyTex, uv);" in result
        assert "var fromGlobal: vec4<f32> = texture(texRef, uv);" in result
        assert (
            "var loadedPointer: vec4<f32> = imageLoad("
            "surfaceObjects[index], vec2<i32>(pixel.x, pixel.y));" in result
        )
        assert (
            "imageStore(surfaceObjects[index], vec2<i32>(pixel.x, pixel.y), "
            "loadedPointer);" in result
        )
        assert (
            "var loadedLegacy: vec4<f32> = imageLoad("
            "legacySurface, vec2<i32>(pixel.x, pixel.y));" in result
        )
        assert (
            "imageStore(legacySurface, vec2<i32>(pixel.x, pixel.y), "
            "loadedLegacy);" in result
        )
        assert (
            "var loadedGlobal: vec4<f32> = imageLoad("
            "surfaceRef, vec2<i32>(pixel.x, pixel.y));" in result
        )
        assert "cudaReadModeElementType" not in result
        assert "cudaTextureObject_t" not in result
        assert "cudaSurfaceObject_t" not in result
        assert "tex2D" not in result
        assert "surf2Dread" not in result
        assert "surf2Dwrite" not in result

    def test_qualified_resource_object_pointer_array_conversion(self):
        """Test qualified CUDA resource object arrays retain inferred shapes."""
        code = """
        const cudaTextureObject_t globalConstTextures[2];
        cudaSurfaceObject_t *__restrict__ globalSurfaceRows[2];

        void qualifiedResourceHandles(
            const cudaTextureObject_t* __restrict__ textureObjects,
            cudaSurfaceObject_t *__restrict__ surfaceObjects,
            const cudaTextureObject_t textureArray[2],
            cudaSurfaceObject_t *__restrict__ surfaceRows[2],
            int row,
            int slot,
            int2 pixel,
            float2 uv
        ) {
            float4 pointerSample = tex2D<float4>(
                textureObjects[slot],
                uv.x,
                uv.y
            );
            float4 arraySample = tex2D<float4>(textureArray[slot], uv);
            float4 globalSample = tex2D<float4>(globalConstTextures[slot], uv);
            float4 pointerRead = surf2Dread<float4>(
                surfaceObjects[slot],
                pixel.x * sizeof(float4),
                pixel.y
            );
            surf2Dwrite(
                pointerRead,
                surfaceObjects[slot],
                pixel.x * sizeof(float4),
                pixel.y
            );
            float4 rowRead = surf2Dread<float4>(
                surfaceRows[row][slot],
                pixel.x * sizeof(float4),
                pixel.y
            );
            surf2Dwrite(
                rowRead,
                surfaceRows[row][slot],
                pixel.x * sizeof(float4),
                pixel.y
            );
            float4 globalRowRead = surf2Dread<float4>(
                globalSurfaceRows[row][slot],
                pixel.x * sizeof(float4),
                pixel.y
            );
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        result = CudaToCrossGLConverter().generate(ast)

        assert "var globalConstTextures: array<sampler2D, 2>;" in result
        assert "var globalSurfaceRows: array<ptr<image2D>, 2>;" in result
        assert (
            "void qualifiedResourceHandles("
            "ptr<sampler2D> textureObjects, ptr<image2D> surfaceObjects, "
            "array<sampler2D, 2> textureArray, "
            "array<ptr<image2D>, 2> surfaceRows, i32 row, i32 slot"
        ) in result
        assert (
            "var pointerSample: vec4<f32> = texture("
            "textureObjects[slot], vec2<f32>(uv.x, uv.y));" in result
        )
        assert "var arraySample: vec4<f32> = texture(textureArray[slot], uv);" in result
        assert (
            "var globalSample: vec4<f32> = texture(globalConstTextures[slot], uv);"
            in result
        )
        assert (
            "var pointerRead: vec4<f32> = imageLoad("
            "surfaceObjects[slot], vec2<i32>(pixel.x, pixel.y));" in result
        )
        assert (
            "imageStore(surfaceObjects[slot], vec2<i32>(pixel.x, pixel.y), "
            "pointerRead);" in result
        )
        assert (
            "var rowRead: vec4<f32> = imageLoad("
            "surfaceRows[row][slot], vec2<i32>(pixel.x, pixel.y));" in result
        )
        assert (
            "imageStore(surfaceRows[row][slot], vec2<i32>(pixel.x, pixel.y), "
            "rowRead);" in result
        )
        assert (
            "var globalRowRead: vec4<f32> = imageLoad("
            "globalSurfaceRows[row][slot], vec2<i32>(pixel.x, pixel.y));" in result
        )
        assert "__restrict__" not in result
        assert "cudaTextureObject_t" not in result
        assert "cudaSurfaceObject_t" not in result
        assert "tex2D" not in result
        assert "surf2Dread" not in result
        assert "surf2Dwrite" not in result

    def test_struct_resource_object_members_infer_crossgl_resource_types(self):
        """Test CUDA resource calls infer struct member resource types."""
        code = """
        struct ResourcePair {
            cudaTextureObject_t tex;
            cudaSurfaceObject_t surf;
            cudaTextureObject_t ambiguous;
        };

        ResourcePair globalPairs[2];

        void usePair(
            ResourcePair pair,
            ResourcePair pairs[2],
            int index,
            int2 pixel,
            float2 uv,
            float3 uvw
        ) {
            float4 sampled = tex2D<float4>(pair.tex, uv.x, uv.y);
            float4 arraySample = tex2D<float4>(pairs[index].tex, uv);
            float4 loaded = surf2Dread<float4>(
                pair.surf,
                pixel.x * sizeof(float4),
                pixel.y
            );
            surf2Dwrite(loaded, pair.surf, pixel.x * sizeof(float4), pixel.y);
            float4 globalLoaded = surf2Dread<float4>(
                globalPairs[index].surf,
                pixel.x * sizeof(float4),
                pixel.y
            );
            float4 ambiguous2d = tex2D<float4>(pair.ambiguous, uv);
            float4 ambiguous3d = tex3D<float4>(
                pair.ambiguous,
                uvw.x,
                uvw.y,
                uvw.z
            );
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        result = CudaToCrossGLConverter().generate(ast)

        assert "struct ResourcePair {" in result
        assert "sampler2D tex;" in result
        assert "image2D surf;" in result
        assert "cudaTextureObject_t ambiguous;" in result
        assert "var globalPairs: array<ResourcePair, 2>;" in result
        assert (
            "void usePair(ResourcePair pair, array<ResourcePair, 2> pairs, "
            "i32 index, vec2<i32> pixel, vec2<f32> uv, vec3<f32> uvw)" in result
        )
        assert (
            "var sampled: vec4<f32> = texture(pair.tex, "
            "vec2<f32>(uv.x, uv.y));" in result
        )
        assert "var arraySample: vec4<f32> = texture(pairs[index].tex, uv);" in result
        assert (
            "var loaded: vec4<f32> = imageLoad("
            "pair.surf, vec2<i32>(pixel.x, pixel.y));" in result
        )
        assert "imageStore(pair.surf, vec2<i32>(pixel.x, pixel.y), loaded);" in result
        assert (
            "var globalLoaded: vec4<f32> = imageLoad("
            "globalPairs[index].surf, vec2<i32>(pixel.x, pixel.y));" in result
        )
        assert "tex2D" not in result
        assert "surf2Dread" not in result
        assert "surf2Dwrite" not in result

    def test_nested_struct_resource_object_members_infer_crossgl_resource_types(self):
        """Test nested struct member chains infer CUDA resource handle types."""
        code = """
        struct ResourcePair {
            cudaTextureObject_t tex;
            cudaSurfaceObject_t surf;
        };

        struct ResourceBox {
            ResourcePair pair;
            ResourcePair* pairPtr;
            ResourcePair pairs[2];
        };

        void useBox(
            ResourceBox box,
            ResourceBox* boxPtr,
            ResourceBox boxes[2],
            int index,
            int2 pixel,
            float2 uv
        ) {
            float4 directSample = tex2D<float4>(box.pair.tex, uv);
            float4 pointerSample = tex2D<float4>(boxPtr->pair.tex, uv);
            float4 pointerMemberSample = tex2D<float4>(box.pairPtr->tex, uv);
            float4 nestedArraySample = tex2D<float4>(boxes[index].pair.tex, uv);
            float4 arrayLoaded = surf2Dread<float4>(
                box.pairs[index].surf,
                pixel.x * sizeof(float4),
                pixel.y
            );
            surf2Dwrite(
                arrayLoaded,
                boxPtr->pair.surf,
                pixel.x * sizeof(float4),
                pixel.y
            );
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        result = CudaToCrossGLConverter().generate(ast)

        assert "struct ResourcePair {" in result
        assert "sampler2D tex;" in result
        assert "image2D surf;" in result
        assert "ResourcePair pair;" in result
        assert "ptr<ResourcePair> pairPtr;" in result
        assert "array<ResourcePair, 2> pairs;" in result
        assert "var directSample: vec4<f32> = texture(box.pair.tex, uv);" in result
        assert "var pointerSample: vec4<f32> = texture(boxPtr->pair.tex, uv);" in result
        assert (
            "var pointerMemberSample: vec4<f32> = texture(box.pairPtr->tex, uv);"
            in result
        )
        assert (
            "var nestedArraySample: vec4<f32> = texture("
            "boxes[index].pair.tex, uv);" in result
        )
        assert (
            "var arrayLoaded: vec4<f32> = imageLoad("
            "box.pairs[index].surf, vec2<i32>(pixel.x, pixel.y));" in result
        )
        assert (
            "imageStore(boxPtr->pair.surf, "
            "vec2<i32>(pixel.x, pixel.y), arrayLoaded);" in result
        )
        assert "cudaTextureObject_t" not in result
        assert "cudaSurfaceObject_t" not in result
        assert "tex2D" not in result
        assert "surf2Dread" not in result
        assert "surf2Dwrite" not in result

    def test_cast_and_dereference_resource_object_members_infer_crossgl_types(self):
        """Test casted and dereferenced struct resource handles infer types."""
        code = """
        struct ResourcePair {
            cudaTextureObject_t tex;
            cudaSurfaceObject_t surf;
        };

        void usePair(void* raw, ResourcePair* pairPtr, int2 pixel, float2 uv) {
            float4 sampled = tex2D<float4>(((ResourcePair*)raw)->tex, uv);
            float4 loaded = surf2Dread<float4>(
                (*pairPtr).surf,
                pixel.x * sizeof(float4),
                pixel.y
            );
            surf2Dwrite(
                loaded,
                ((ResourcePair*)raw)->surf,
                pixel.x * sizeof(float4),
                pixel.y
            );
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        result = CudaToCrossGLConverter().generate(ast)

        assert "struct ResourcePair {" in result
        assert "sampler2D tex;" in result
        assert "image2D surf;" in result
        assert (
            "void usePair(ptr<void> raw, ptr<ResourcePair> pairPtr, "
            "vec2<i32> pixel, vec2<f32> uv)" in result
        )
        assert (
            "var sampled: vec4<f32> = texture(ptr<ResourcePair>(raw)->tex, uv);"
            in result
        )
        assert (
            "var loaded: vec4<f32> = imageLoad("
            "(*pairPtr).surf, vec2<i32>(pixel.x, pixel.y));" in result
        )
        assert (
            "imageStore(ptr<ResourcePair>(raw)->surf, "
            "vec2<i32>(pixel.x, pixel.y), loaded);" in result
        )
        assert "cudaTextureObject_t" not in result
        assert "cudaSurfaceObject_t" not in result
        assert "tex2D" not in result
        assert "surf2Dread" not in result
        assert "surf2Dwrite" not in result

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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert result.count("for x in h {") == 2
        assert "x = 1.0f;" in result
        assert "var sum: f32 = 0.0f;" in result
        assert "sum += x;" in result
        assert "for y in h {" in result
        assert "sink(y);" in result

    def test_empty_program(self):
        """Test empty program conversion"""
        code = ""
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// CUDA to CrossGL conversion" in result

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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
        source_path = tmp_path / "kernel.cu"
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

        __global__ void kernel(unsigned int* out, const float scale) {
            const int local = 1;
            unsigned int idx = 2u;
            static float tmp = 0.0f;
            out[0] = idx;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var cached: f32 = 1.0f;" in result
        assert "var mask: u32 = 3u;" in result
        assert "out: array<u32>" in result
        assert "f32 scale" in result
        assert "var local: i32 = 1;" in result
        assert "var idx: u32 = 2u;" in result
        assert "var tmp: f32 = 0.0f;" in result

    def test_qualified_and_pointer_return_function_conversion(self):
        """Test qualified scalar and pointer return conversion"""
        code = """
        unsigned int lane_mask() { return 3u; }
        float* get_data(float* data) { return data; }
        const float* get_const_data(const float* data) { return data; }
        static inline unsigned int helper(unsigned int x) { return x; }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
            return;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "i += 1;" in result
        assert "sink(i);" in result
        assert "for (var j: i32 = 0; (j < n); (j++))" in result
        assert "sink(j);" in result
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var mask: u32 = 0xffu;" in result
        assert "var bits: u32 = 0b1010u;" in result
        assert "var oct: u32 = 0777u;" in result
        assert "var x: f32 = 1e-3f;" in result
        assert "var y: f32 = .5f;" in result
        assert "return ((mask | bits) | oct);" in result

    def test_long_long_type_conversion(self):
        """Test scalar long long CUDA types convert to 64-bit CrossGL types"""
        code = """
        __global__ void kernel(unsigned long long* out, long long x) {
            unsigned long long y = 1ull;
            long long z = (long long)x;
            out[0] = y + z;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "@group(0) @binding(0) var<storage, read_write> out: array<u64>" in result
        )
        assert "i64 x" in result
        assert "var y: u64 = 1ull;" in result
        assert "var z: i64 = i64(x);" in result
        assert "out[0] = (y + z);" in result

    def test_control_flow_and_cast_expression_conversion(self):
        """Test do-while, switch, ternary, and casts are emitted"""
        code = """
        int helper(float x, int n) {
            int i = 0;
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "do {" in result
        assert "} while ((i < n));" in result
        assert "switch (i) {" in result
        assert "case 1:" in result
        assert "default:" in result
        assert "return ((i > 0) ? i32(x) : n);" in result
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "switch (value) {" in result
        assert "default:" in result
        assert "case 1:" in result
        assert result.index("default:") < result.index("case 1:")
