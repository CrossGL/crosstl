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

    def test_user_defined_lerp_call_does_not_convert_to_mix(self):
        """Test user-defined CUDA functions shadow builtin call conversion."""
        code = """
        float lerp(float x) {
            return x;
        }

        __global__ void kernel(float* out, float x) {
            out[0] = lerp(x);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "f32 lerp(f32 x) {" in result
        assert "out[0] = lerp(x);" in result
        assert "out[0] = mix(x);" not in result

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
        """Test CUDA thread fence converts back to CrossGL memoryBarrier."""
        code = """
        __global__ void fence(float* out) {
            __threadfence();
            __syncthreads();
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "memoryBarrier();" in result
        assert "workgroupBarrier();" in result
        assert "__threadfence" not in result

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
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// CUDA memory allocate: d, bytes: (n * sizeof(float))" in result
        assert (
            "// CUDA memory copy: h -> d, bytes: (n * sizeof(float)), "
            "kind: cudaMemcpyHostToDevice"
        ) in result
        assert "// CUDA memory set: d, value: 0, bytes: (n * sizeof(float))" in result
        assert "// CUDA device synchronize" in result
        assert "// CUDA memory free: d" in result
        assert (
            "var err: cudaError_t = "
            "cudaMalloc(ptr<ptr<void>>((&d)), (n * sizeof(float)));"
        ) in result

    def test_cuda_runtime_event_api_conversion(self):
        """Test CUDA stream and event API calls emit metadata comments"""
        code = """
        void bench() {
            cudaStream_t stream;
            cudaEvent_t start;
            cudaEvent_t stop;
            float ms;
            cudaStreamCreate(&stream);
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
        assert "// CUDA event create: start" in result
        assert "// CUDA event create: stop, flags: cudaEventDisableTiming" in result
        assert "// CUDA event record: start, stream: stream" in result
        assert "// CUDA stream wait event: stream waits for start, flags: 0" in result
        assert "// CUDA event synchronize: stop" in result
        assert "// CUDA event elapsed time: start -> stop, output: ms" in result
        assert "// CUDA event query: stop" in result
        assert "// CUDA event destroy: start" in result
        assert "// CUDA stream destroy: stream" in result
        assert "var err: cudaError_t = cudaEventRecord(stop, stream);" in result

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
        void host() {
            std::array<unsigned int, 4> ids{};
            std::vector<const unsigned int> flags;
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

        assert "pub fn kernel(data: Vec<f32>, indices: Vec<i32>, value: f32)" in result
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
