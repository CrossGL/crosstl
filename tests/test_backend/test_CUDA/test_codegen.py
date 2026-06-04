from crosstl import translate
from crosstl.backend.CUDA.CudaCrossGLCodeGen import CudaToCrossGLConverter
from crosstl.backend.CUDA.CudaLexer import CudaLexer
from crosstl.backend.CUDA.CudaParser import CudaParser


class TestCudaCodeGen:
    def test_basic_kernel_conversion(self):
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

    def test_launch_bounds_kernel_attribute_conversion(self):
        code = """
        __launch_bounds__(128) __global__ void bounded(float* data) {
            data[threadIdx.x] = 0.0f;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// CUDA launch bounds: (128)" in result
        assert "// Kernel: bounded" in result

    def test_public_cuda_samples_tile_global_kernel_conversion(self):
        code = """
        __tile_global__ void matmul_naive(float* C) {
            C[0] = 0.0f;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// Kernel: matmul_naive" in result
        assert "// Function: matmul_naive" not in result
        assert "@compute" in result

    def test_cuda_function_attributes_and_inline_asm_conversion(self):
        code = r"""
        __cluster_dims__(2, 1, 1) __block_size__(128) __global__ void clustered(
            unsigned int* out,
            unsigned int in) {
            unsigned int lane = 0;
            asm volatile(
                "add.u32 %0, %1, 1;"
                : [result] "=r"(lane)
                : [source] "r"(in)
                : "memory"
            );
            out[threadIdx.x] = lane;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// CUDA cluster dims: (2, 1, 1)" in result
        assert "// CUDA block size: (128)" in result
        assert '// CUDA inline PTX volatile: "add.u32 %0, %1, 1;"' in result
        assert '// CUDA inline PTX outputs: [result] "=r"(lane)' in result
        assert '// CUDA inline PTX inputs: [source] "r"(in_)' in result
        assert '// CUDA inline PTX clobbers: "memory"' in result
        assert "CudaAsmNode" not in result

    def test_typedef_struct_with_alignment_conversion(self):
        code = """
        typedef struct __align__(8) {
            unsigned int x;
            unsigned int y;
        } Pair;

        __global__ void kernel(Pair* pairs) {
            pairs[threadIdx.x].x = 1;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "struct Pair {" in result
        assert "u32 x;" in result
        assert "u32 y;" in result
        assert "Pair" in result
        assert "__align__" not in result

    def test_grid_constant_kernel_parameter_conversion(self):
        code = """
        __global__ void kernelLargeParam(__grid_constant__ const int scale,
                                         int* result) {
            result[threadIdx.x] = scale;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// CUDA grid constant parameter: scale" in result
        assert "i32 scale" in result
        assert "__grid_constant__" not in result

    def test_c_linkage_block_kernel_conversion(self):
        code = """
        extern "C" {
        __global__ void kernel(float* out) {
            out[threadIdx.x] = 1.0f;
        }
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// Kernel: kernel" in result
        assert "fn kernel(" in result
        assert "out[gl_LocalInvocationID.x] = 1.0f;" in result
        assert "extern" not in result
        assert '"C"' not in result

    def test_device_function_conversion(self):
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

    def test_public_cuda_samples_device_global_variables_conversion(self):
        code = """
        __device__ int g_uids = 0;
        __device__ double grid_dot_result = 0.0;
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var g_uids: i32 = 0;" in result
        assert "var grid_dot_result: f64 = 0.0;" in result

    def test_fixed_width_pointer_declaration_list_conversion(self):
        code = """
        __global__ void bit_extract_kernel(
            uint32_t* d_output,
            const uint32_t* d_input,
            size_t size) {
            uint32_t *d_local, *d_shadow;
            d_output[0] = ((d_input[0] & 0xf00) >> 8);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "@group(0) @binding(0) var<storage, read_write> d_output: array<u32>"
            in result
        )
        assert (
            "@group(0) @binding(1) var<storage, read_write> d_input: array<u32>"
            in result
        )
        assert "var d_local: ptr<u32>;" in result
        assert "var d_shadow: ptr<u32>;" in result
        assert "d_output[0] = ((d_input[0] & 0xf00) >> 8);" in result

    def test_enum_class_declaration_conversion(self):
        code = """
        enum class MemoryMode : unsigned int
        {
            PAGED,
            PINNED
        };

        void run_copy(const MemoryMode memory_mode) {
            if (memory_mode == MemoryMode::PAGED) {
                return;
            }
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "enum MemoryMode : u32 {" in result
        assert "PAGED," in result
        assert "PINNED," in result
        assert "run_copy(MemoryMode memory_mode)" in result
        assert "MemoryMode::PAGED" in result
        assert "EnumNode" not in result

    def test_device_function_body_emitted_when_kernel_calls_it(self):
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

    def test_bodyless_prototypes_are_not_emitted(self):
        code = """
        void declared(int value);

        int main(void) {
            return 0;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// Function: declared" not in result
        assert "declared(" not in result
        assert "i32 main() {" in result

    def test_multiple_kernels_conversion(self):
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
        codegen = CudaToCrossGLConverter()

        # Test basic types
        assert codegen.convert_cuda_type_to_crossgl("int") == "i32"
        assert codegen.convert_cuda_type_to_crossgl("uint") == "u32"
        assert codegen.convert_cuda_type_to_crossgl("float") == "f32"
        assert codegen.convert_cuda_type_to_crossgl("double") == "f64"
        assert codegen.convert_cuda_type_to_crossgl("bool") == "bool"
        assert codegen.convert_cuda_type_to_crossgl("void") == "void"
        assert codegen.convert_cuda_type_to_crossgl("long long") == "i64"
        assert codegen.convert_cuda_type_to_crossgl("unsigned long long") == "u64"
        assert codegen.convert_cuda_type_to_crossgl("half") == "f16"
        assert codegen.convert_cuda_type_to_crossgl("__half") == "f16"
        assert codegen.convert_cuda_type_to_crossgl("half2") == "vec2<f16>"
        assert codegen.convert_cuda_type_to_crossgl("__half2") == "vec2<f16>"
        assert codegen.convert_cuda_type_to_crossgl("float *") == "ptr<f32>"
        assert (
            codegen.convert_cuda_type_to_crossgl("unsigned long long *") == "ptr<u64>"
        )
        assert codegen.convert_cuda_type_to_crossgl("half2 *") == "ptr<vec2<f16>>"
        assert codegen.convert_cuda_type_to_crossgl("void * *") == "ptr<ptr<void>>"
        assert codegen.convert_cuda_type_to_crossgl("void * []") == "array<ptr<void>>"

    def test_public_cuda_scan_uint_shared_barrier_loop_conversion(self):
        """Covers cuda-samples scan.cu uint shared-memory barrier loops."""
        code = """
        #define THREADBLOCK_SIZE 256
        inline __device__ uint scan1Inclusive(
            uint idata,
            volatile uint *s_Data,
            uint size) {
            uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
            for (uint offset = 1; offset < size; offset <<= 1) {
                __syncthreads();
                uint t = s_Data[pos] + s_Data[pos - offset];
                __syncthreads();
                s_Data[pos] = t;
            }
            return s_Data[pos];
        }

        __global__ void scanExclusiveShared(
            uint4 *d_Dst,
            uint4 *d_Src,
            uint size) {
            __shared__ uint s_Data[2 * THREADBLOCK_SIZE];
            uint pos = blockIdx.x * blockDim.x + threadIdx.x;
            uint4 idata4 = d_Src[pos];
            uint oval = scan1Inclusive(idata4.w, s_Data, size / 4);
            idata4.x += oval;
            d_Dst[pos] = idata4;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        result = CudaToCrossGLConverter().generate(ast)

        assert "u32 scan1Inclusive(u32 idata, ptr<u32> s_Data, u32 size)" in result
        assert "var<workgroup> s_Data: array<u32, (2 * 256)>;" in result
        expected_pos = (
            "var pos: u32 = "
            "((gl_WorkGroupID.x * gl_WorkGroupSize.x) + gl_LocalInvocationID.x);"
        )
        assert expected_pos in result
        assert "for (var offset: u32 = 1; (offset < size); offset <<= 1)" in result
        assert result.count("workgroupBarrier();") == 2
        assert "uint scan1Inclusive" not in result
        assert "ptr<uint>" not in result
        assert "array<uint" not in result

    def test_cuda_fp16_half2_types_and_intrinsics_convert_to_crossgl(self):
        code = """
        __device__ half2 fp16_ops(half2 a, half2 b, float x) {
            half2 scalar = __float2half2_rn(x);
            half2 prod = __hmul2(a, b);
            half2 sum = __hadd2(prod, scalar);
            half2 fused = __hfma2(a, b, sum);
            float low = __low2float(fused);
            return fused;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        result = CudaToCrossGLConverter().generate(ast)

        assert "vec2<f16> fp16_ops(vec2<f16> a, vec2<f16> b, f32 x)" in result
        assert "var scalar: vec2<f16> = vec2<f16>(x, x);" in result
        assert "var prod: vec2<f16> = (a * b);" in result
        assert "var sum: vec2<f16> = (prod + scalar);" in result
        assert "var fused: vec2<f16> = fma(a, b, sum);" in result
        assert "var low: f32 = f32(fused.x);" in result
        for raw_name in (
            "half2",
            "__float2half2_rn",
            "__hmul2",
            "__hadd2",
            "__hfma2",
            "__low2float",
        ):
            assert raw_name not in result

    def test_cuda_fp16_pointer_array_declarations_convert_to_crossgl(self):
        code = """
        void host() {
            half2 *vec[2];
            half2 *const devVec[2];
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        result = CudaToCrossGLConverter().generate(ast)

        assert "var vec: array<ptr<vec2<f16>>, 2>;" in result
        assert "var devVec: array<ptr<vec2<f16>>, 2>;" in result
        assert "(half2 * vec[2]);" not in result

    def test_constructor_style_vector_declaration_conversion(self):
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

    def test_dynamic_shared_memory_codegen_marks_launch_sized_storage(self):
        code = """
        __global__ void kernel(float* out, const float* in) {
            extern __shared__ float shared[];
            shared[threadIdx.x] = in[threadIdx.x];
            __syncthreads();
            out[threadIdx.x] = shared[threadIdx.x];
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA dynamic shared memory: shared uses launch-time shared memory size"
            in result
        )
        assert "var<workgroup> shared: array<f32>;" in result
        assert "workgroupBarrier();" in result

    def test_bitwise_atomics_lower_to_crossgl_lvalue_targets(self):
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

    def test_warp_vote_and_shuffle_intrinsics_do_not_leak_raw_cuda_calls(self):
        code = """
        __global__ void warp(unsigned int mask, int pred, int value, int lane, unsigned int* out) {
            unsigned int active = __activemask();
            unsigned int bits = __ballot_sync(0xffffffff, pred);
            int any_set = __any_sync(0xffffffff, pred);
            int y = __shfl_sync(0xffffffff, value, lane);
            unsigned int custom = __ballot_sync(mask, pred);
            int unsupported = __shfl_down_sync(0xffffffff, value, 1);
            out[lane] = active + bits + any_set + y + custom + unsupported;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        result = CudaToCrossGLConverter().generate(ast)

        assert "WaveActiveBallot(true).x" in result
        assert "WaveActiveBallot((pred != 0)).x" in result
        assert "(WaveActiveAnyTrue((pred != 0)) ? 1 : 0)" in result
        assert "WaveReadLaneAt(value, lane)" in result
        assert (
            "/* cuda warp intrinsic __ballot_sync(mask, pred) not directly "
            "supported in CrossGL */ 0"
        ) in result
        assert (
            "/* cuda warp intrinsic __shfl_down_sync(0xffffffff, value, 1) "
            "not directly supported in CrossGL */ 0"
        ) in result
        assert "__activemask()" not in result
        assert "__ballot_sync(0xffffffff, pred)" not in result
        assert "__shfl_sync(0xffffffff, value, lane)" not in result

    def test_cooperative_groups_thread_block_sync_converts(self):
        code = """
        #include <cooperative_groups.h>
        namespace cg = cooperative_groups;

        __global__ void sync(float* out) {
            cg::thread_block block = cg::this_thread_block();
            block.sync();
            auto direct = cooperative_groups::this_thread_block();
            direct.sync();
            cooperative_groups::this_thread_block().sync();
            cg::sync(block);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert result.count("workgroupBarrier();") == 4
        assert (
            "// cooperative_groups thread_block block maps to the current workgroup"
            in result
        )
        assert (
            "// cooperative_groups thread_block direct maps to the current workgroup"
            in result
        )
        assert "cg::this_thread_block" not in result
        assert "cooperative_groups::this_thread_block" not in result

    def test_cooperative_groups_thread_block_rank_and_size_converts(self):
        code = """
        __global__ void ranks(unsigned int* out) {
            auto block = cooperative_groups::this_thread_block();
            unsigned int rank = block.thread_rank();
            unsigned int size = block.size();
            unsigned int direct =
                cooperative_groups::this_thread_block().thread_rank();
            out[rank] = size + direct;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var rank: u32 = gl_LocalInvocationIndex;" in result
        assert (
            "var size: u32 = "
            "((gl_WorkGroupSize.x * gl_WorkGroupSize.y) * gl_WorkGroupSize.z);"
            in result
        )
        assert "var direct: u32 = gl_LocalInvocationIndex;" in result
        assert "thread_rank" not in result
        assert "block.size" not in result
        assert "cooperative_groups::this_thread_block" not in result

    def test_cooperative_groups_tiled_partition_rank_and_size_converts(self):
        code = """
        namespace cg = cooperative_groups;

        __global__ void tile_ranks(unsigned int* out) {
            auto block = cg::this_thread_block();
            auto tile = cg::tiled_partition<32>(block);
            unsigned int lane = tile.thread_rank();
            unsigned int width = tile.size();
            out[lane] = width;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// cooperative_groups thread_block_tile<32> tile maps to a "
            "tiled partition of the current workgroup" in result
        )
        assert "var lane: u32 = (gl_LocalInvocationIndex % 32);" in result
        assert "var width: u32 = 32;" in result
        assert "tiled_partition" not in result
        assert "thread_rank" not in result
        assert "tile.size" not in result

    def test_cooperative_groups_sync_factory_and_member_aliases_convert(self):
        code = """
        namespace cg = cooperative_groups;

        __global__ void metadata(unsigned int* out) {
            auto block = cg::this_thread_block();
            cg::sync(cg::this_thread_block());
            unsigned int count = block.num_threads();
            dim3 local = block.thread_index();
            dim3 dims = block.dim_threads();
            auto tile = cg::tiled_partition<16>(block);
            unsigned int tile_count = tile.num_threads();
            out[count] = local.x + dims.x + tile_count;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "workgroupBarrier();" in result
        assert (
            "var count: u32 = "
            "((gl_WorkGroupSize.x * gl_WorkGroupSize.y) * gl_WorkGroupSize.z);"
            in result
        )
        assert "var local: vec3<u32> = gl_LocalInvocationID;" in result
        assert "var dims: vec3<u32> = gl_WorkGroupSize;" in result
        assert "var tile_count: u32 = 16;" in result
        assert "cg::sync" not in result
        assert "cg::this_thread_block" not in result
        assert "num_threads" not in result
        assert "thread_index" not in result
        assert "dim_threads" not in result

    def test_unsupported_cooperative_group_rank_is_expression_safe(self):
        code = """
        __global__ void unsupported(unsigned int* out) {
            auto group = cooperative_groups::coalesced_threads();
            unsigned int rank = group.thread_rank();
            unsigned int width = group.size();
            unsigned int count = group.num_threads();
            unsigned int direct =
                cooperative_groups::coalesced_threads().thread_rank();
            out[0] = rank + width + count + direct;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "var rank: u32 = (/* cooperative_groups "
            "coalesced_group.thread_rank not directly supported in CrossGL */ 0);"
            in result
        )
        assert (
            "var width: u32 = (/* cooperative_groups "
            "coalesced_group.size not directly supported in CrossGL */ 0);" in result
        )
        assert (
            "var count: u32 = (/* cooperative_groups "
            "coalesced_group.num_threads not directly supported in CrossGL */ 0);"
            in result
        )
        assert (
            "var direct: u32 = (/* cooperative_groups "
            "coalesced_group.thread_rank not directly supported in CrossGL */ 0);"
            in result
        )
        assert "cooperative_groups::coalesced_threads" not in result
        assert "None" not in result

    def test_unsupported_cooperative_groups_emit_diagnostics(self):
        code = """
        __global__ void sync_grid(float* out) {
            auto grid = cooperative_groups::this_grid();
            grid.sync();
            cooperative_groups::coalesced_threads().sync();
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// cooperative_groups grid_group for grid not directly supported in CrossGL"
            in result
        )
        assert (
            "// cooperative_groups grid_group.sync not directly supported in CrossGL"
            in result
        )
        assert (
            "// cooperative_groups coalesced_group.sync not directly supported in CrossGL"
            in result
        )
        assert "cooperative_groups::this_grid" not in result
        assert "cooperative_groups::coalesced_threads" not in result

    def test_cooperative_groups_async_copy_wait_emit_diagnostics(self):
        code = """
        namespace cg = cooperative_groups;

        __global__ void async_copy(int* shared, const int* global) {
            auto block = cg::this_thread_block();
            cg::memcpy_async(block, shared, global, sizeof(int) * block.size());
            cg::memcpy_async(cg::this_thread_block(), shared, global, 16);
            cg::wait(block);
            cooperative_groups::wait(cg::this_thread_block());
            cooperative_groups::wait_prior<1>(block);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert result.count("cooperative_groups thread_block.memcpy_async") == 2
        assert "cooperative_groups thread_block.wait not directly supported" in result
        assert (
            "cooperative_groups thread_block.wait_prior not directly supported"
            in result
        )
        assert "cg::memcpy_async" not in result
        assert "cooperative_groups::wait" not in result
        assert "wait_prior<1>" not in result
        assert "None" not in result

    def test_cuda_barrier_pipeline_async_copy_emit_diagnostics(self):
        code = """
        __global__ void async_copy(int* shared, const int* global, int* out) {
            __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> state;
            cuda::barrier<cuda::thread_scope_block> blockBarrier;
            init(&blockBarrier, blockDim.x);
            cuda::memcpy_async(
                shared,
                global,
                sizeof(int) * blockDim.x,
                blockBarrier
            );
            auto token = blockBarrier.arrive();
            bool ready = blockBarrier.try_wait(token);
            blockBarrier.wait(token);
            blockBarrier.arrive_and_wait();
            auto pipe = cuda::make_pipeline();
            pipe.producer_acquire();
            cuda::memcpy_async(shared, global, 16, pipe);
            pipe.producer_commit();
            pipe.consumer_wait();
            pipe.consumer_release();
            if (ready) {
                out[0] = 1;
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
            "// cuda pipeline_shared_state state not directly supported in CrossGL"
            in result
        )
        assert (
            "// cuda barrier blockBarrier not directly supported in CrossGL" in result
        )
        assert "// cuda barrier.init not directly supported in CrossGL" in result
        assert (
            "// cuda barrier.memcpy_async not directly supported in CrossGL" in result
        )
        assert (
            "var token: auto = "
            "(/* cuda barrier.arrive not directly supported in CrossGL */ 0);" in result
        )
        assert (
            "var ready: bool = "
            "(/* cuda barrier.try_wait not directly supported in CrossGL */ false);"
            in result
        )
        assert "// cuda barrier.wait not directly supported in CrossGL" in result
        assert (
            "// cuda barrier.arrive_and_wait not directly supported in CrossGL"
            in result
        )
        assert "// cuda pipeline pipe not directly supported in CrossGL" in result
        assert (
            "// cuda pipeline.producer_acquire not directly supported in CrossGL"
            in result
        )
        assert (
            "// cuda pipeline.memcpy_async not directly supported in CrossGL" in result
        )
        assert (
            "// cuda pipeline.producer_commit not directly supported in CrossGL"
            in result
        )
        assert (
            "// cuda pipeline.consumer_wait not directly supported in CrossGL" in result
        )
        assert (
            "// cuda pipeline.consumer_release not directly supported in CrossGL"
            in result
        )
        assert "cuda::memcpy_async" not in result
        assert "cuda::make_pipeline" not in result
        assert "blockBarrier.arrive" not in result
        assert "pipe.producer" not in result
        assert "pipe.consumer" not in result
        assert "None" not in result

    def test_cuda_pipeline_primitive_intrinsics_emit_diagnostics(self):
        code = """
        __global__ void primitive_async_copy(
            int* shared,
            const int* global,
            __mbarrier_t* barrier
        ) {
            __pipeline_memcpy_async(shared, global, 16);
            __pipeline_memcpy_async(shared + 16, global + 16, 16, 0);
            __pipeline_commit();
            __pipeline_wait_prior(1);
            __pipeline_arrive_on(barrier);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert result.count("// cuda pipeline.memcpy_async not directly supported") == 2
        assert "// cuda pipeline.commit not directly supported in CrossGL" in result
        assert (
            "// cuda pipeline.wait_prior not directly supported in CrossGL: 1" in result
        )
        assert (
            "// cuda pipeline.arrive_on not directly supported in CrossGL: barrier"
            in result
        )
        assert "__pipeline_memcpy_async" not in result
        assert "__pipeline_commit" not in result
        assert "__pipeline_wait_prior" not in result
        assert "__pipeline_arrive_on" not in result
        assert "None" not in result

    def test_inverse_trig_builtins_convert_to_crossgl(self):
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

    def test_cuda_launch_cooperative_kernel_api_conversion(self):
        code = """
        void host(void** params) {
            dim3 grid(16);
            dim3 block(32);
            cudaLaunchCooperativeKernel((void*)k, grid, block, params, 0, NULL);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// Kernel launch: k<<<grid, block, 0, NULL>>>()" in result
        assert "// Arguments: params" in result
        assert "cudaLaunchCooperativeKernel" not in result

    def test_cuda_launch_cooperative_kernel_error_wrapper_from_nvidia_sample(self):
        code = """
        void host(void** kernelArgs, dim3 dimGrid, dim3 dimBlock) {
            checkCudaErrors(cudaLaunchCooperativeKernel(
                (void*)normVecByDotProductAWBarrier,
                dimGrid,
                dimBlock,
                kernelArgs,
                0,
                0));
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// Kernel launch: normVecByDotProductAWBarrier<<<dimGrid, dimBlock, 0, 0>>>()"
            in result
        )
        assert "// Arguments: kernelArgs" in result
        assert "checkCudaErrors" not in result
        assert "cudaLaunchCooperativeKernel" not in result

    def test_user_defined_cuda_launch_kernel_call_is_not_kernel_launch(self):
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

    def test_user_defined_cuda_launch_cooperative_kernel_call_is_not_kernel_launch(
        self,
    ):
        code = """
        void cudaLaunchCooperativeKernel(float* out, int grid, int block, void* args, int shared, int stream) {
            return;
        }

        void host(float* out, int grid, int block, void* args) {
            cudaLaunchCooperativeKernel(out, grid, block, args, 0, 0);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// Function: cudaLaunchCooperativeKernel" in result
        assert "void cudaLaunchCooperativeKernel(" in result
        assert "cudaLaunchCooperativeKernel(out, grid, block, args, 0, 0);" in result
        assert "// Kernel launch: out<<<grid, block, 0, 0>>>()" not in result

    def test_user_defined_cuda_launch_kernel_declared_later_is_not_kernel_launch(self):
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

    def test_cuda_runtime_error_wrapper_unwraps_runtime_calls(self):
        code = """
        void host(float* h, size_t mem_size_A, cudaStream_t stream) {
            float* d_A;
            checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
            checkCudaErrors(cudaMemcpyAsync(
                d_A, h, mem_size_A, cudaMemcpyHostToDevice, stream
            ));
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// CUDA memory allocate: d_A, bytes: mem_size_A" in result
        assert (
            "// CUDA memory copy: h -> d_A, bytes: mem_size_A, "
            "kind: cudaMemcpyHostToDevice, stream: stream"
        ) in result
        assert "checkCudaErrors(" not in result
        assert "cudaMalloc(" not in result
        assert "cudaMemcpyAsync(" not in result

    def test_cuda_runtime_memset_async_conversion(self):
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

    def test_cuda_texture_object_descriptor_query_conversion(self):
        code = """
        void queryTextureObject(cudaTextureObject_t objectTex) {
            cudaResourceDesc resourceDesc;
            cudaTextureDesc textureDesc;
            cudaResourceViewDesc viewDesc;
            cudaGetTextureObjectResourceDesc(&resourceDesc, objectTex);
            cudaError_t err = cudaGetTextureObjectTextureDesc(
                &textureDesc,
                objectTex
            );
            err = cudaGetTextureObjectResourceViewDesc(&viewDesc, objectTex);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA texture object resource descriptor query: "
            "objectTex, output: resourceDesc"
        ) in result
        assert (
            "// CUDA texture object texture descriptor query: "
            "objectTex, output: textureDesc"
        ) in result
        assert (
            "// CUDA texture object resource view descriptor query: "
            "objectTex, output: viewDesc"
        ) in result
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "err = cudaSuccess;" in result
        assert "cudaGetTextureObjectResourceDesc(" not in result
        assert "cudaGetTextureObjectTextureDesc(" not in result
        assert "cudaGetTextureObjectResourceViewDesc(" not in result

    def test_cuda_surface_object_descriptor_query_conversion(self):
        code = """
        void querySurfaceObject(cudaSurfaceObject_t surfaceObj) {
            cudaResourceDesc resourceDesc;
            cudaGetSurfaceObjectResourceDesc(&resourceDesc, surfaceObj);
            cudaError_t err = cudaGetSurfaceObjectResourceDesc(
                &resourceDesc,
                surfaceObj
            );
            err = cudaGetSurfaceObjectResourceDesc(&resourceDesc, surfaceObj);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA surface object resource descriptor query: "
            "surfaceObj, output: resourceDesc"
        ) in result
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "err = cudaSuccess;" in result
        assert "cudaGetSurfaceObjectResourceDesc(" not in result

    def test_cuda_external_memory_runtime_conversion(self):
        code = """
        void importExternalMemory(cudaExternalMemoryHandleDesc handleDesc) {
            cudaExternalMemory_t memory;
            cudaExternalMemoryBufferDesc bufferDesc;
            cudaExternalMemoryMipmappedArrayDesc mipDesc;
            cudaMipmappedArray_t mipmapped;
            void* ptr;
            cudaImportExternalMemory(&memory, &handleDesc);
            cudaError_t err = cudaExternalMemoryGetMappedBuffer(
                &ptr,
                memory,
                &bufferDesc
            );
            err = cudaExternalMemoryGetMappedMipmappedArray(
                &mipmapped,
                memory,
                &mipDesc
            );
            err = cudaDestroyExternalMemory(memory);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA external memory import: output: memory, handle: (&handleDesc)"
            in result
        )
        assert (
            "// CUDA external memory mapped buffer: memory, "
            "desc: (&bufferDesc), output: ptr"
        ) in result
        assert (
            "// CUDA external memory mapped mipmapped array: memory, "
            "desc: (&mipDesc), output: mipmapped"
        ) in result
        assert "// CUDA external memory destroy: memory" in result
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "err = cudaSuccess;" in result
        assert "cudaImportExternalMemory(" not in result
        assert "cudaExternalMemoryGetMappedBuffer(" not in result
        assert "cudaExternalMemoryGetMappedMipmappedArray(" not in result
        assert "cudaDestroyExternalMemory(" not in result

    def test_cuda_external_semaphore_runtime_conversion(self):
        code = """
        void syncExternalSemaphore(
            cudaExternalSemaphoreHandleDesc handleDesc,
            cudaStream_t stream
        ) {
            cudaExternalSemaphore_t semaphore;
            cudaExternalSemaphoreSignalParams signalParams;
            cudaExternalSemaphoreWaitParams waitParams;
            cudaImportExternalSemaphore(&semaphore, &handleDesc);
            cudaError_t err = cudaSignalExternalSemaphoresAsync(
                &semaphore,
                &signalParams,
                1,
                stream
            );
            err = cudaWaitExternalSemaphoresAsync(
                &semaphore,
                &waitParams,
                1,
                stream
            );
            err = cudaDestroyExternalSemaphore(semaphore);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA external semaphore import: output: semaphore, "
            "handle: (&handleDesc)"
        ) in result
        assert (
            "// CUDA external semaphore signal: semaphores: (&semaphore), "
            "params: (&signalParams), count: 1, stream: stream"
        ) in result
        assert (
            "// CUDA external semaphore wait: semaphores: (&semaphore), "
            "params: (&waitParams), count: 1, stream: stream"
        ) in result
        assert "// CUDA external semaphore destroy: semaphore" in result
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "err = cudaSuccess;" in result
        assert "cudaImportExternalSemaphore(" not in result
        assert "cudaSignalExternalSemaphoresAsync(" not in result
        assert "cudaWaitExternalSemaphoresAsync(" not in result
        assert "cudaDestroyExternalSemaphore(" not in result

    def test_cuda_graph_lifecycle_runtime_conversion(self):
        code = """
        void runGraph(cudaStream_t stream, unsigned long long flags) {
            cudaGraph_t graph;
            cudaGraph_t clone;
            cudaGraph_t captured;
            cudaGraphExec_t exec;
            cudaGraphNode_t node;
            cudaGraphNode_t errorNode;
            void* log;

            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            cudaStreamEndCapture(stream, &captured);
            cudaGraphCreate(&graph, 0);
            cudaGraphClone(&clone, graph);
            cudaError_t err = cudaGraphInstantiate(
                &exec,
                clone,
                &errorNode,
                log,
                128
            );
            err = cudaGraphInstantiateWithFlags(&exec, graph, flags);
            err = cudaGraphLaunch(exec, stream);
            err = cudaGraphExecDestroy(exec);
            cudaGraphDestroyNode(node);
            cudaGraphDestroy(clone);
            err = cudaGraphDestroy(graph);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA stream begin capture: stream: stream, "
            "mode: cudaStreamCaptureModeGlobal"
        ) in result
        assert (
            "// CUDA stream end capture: stream: stream, graph output: captured"
            in result
        )
        assert "// CUDA graph create: output: graph, flags: 0" in result
        assert "// CUDA graph clone: output: clone, source: graph" in result
        assert (
            "// CUDA graph instantiate: output: exec, graph: clone, "
            "error node output: errorNode, log buffer: log, log bytes: 128"
        ) in result
        assert (
            "// CUDA graph instantiate with flags: output: exec, graph: graph, "
            "flags: flags"
        ) in result
        assert "// CUDA graph launch: exec: exec, stream: stream" in result
        assert "// CUDA graph exec destroy: exec" in result
        assert "// CUDA graph destroy node: node" in result
        assert "// CUDA graph destroy: clone" in result
        assert "// CUDA graph destroy: graph" in result
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "err = cudaSuccess;" in result
        for function_name in [
            "cudaStreamBeginCapture",
            "cudaStreamEndCapture",
            "cudaGraphCreate",
            "cudaGraphClone",
            "cudaGraphInstantiate",
            "cudaGraphInstantiateWithFlags",
            "cudaGraphLaunch",
            "cudaGraphExecDestroy",
            "cudaGraphDestroyNode",
            "cudaGraphDestroy",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_graph_lifecycle_runtime_conversion(self):
        code = """
        bool checkStatus(CUresult status) {
            return status == CUDA_SUCCESS;
        }

        CUresult runDriverGraph(
            CUstream stream,
            unsigned long long flags,
            bool ok
        ) {
            CUgraph graph;
            CUgraph clone;
            CUgraphExec exec;
            CUgraphNode node;
            void* log;

            CUresult status = cuGraphCreate(&graph, 0);
            status = cuGraphClone(&clone, graph);
            status = cuGraphInstantiate(&exec, graph, &node, log, 128);
            status = cuGraphInstantiate(&exec, clone, flags);
            status = cuGraphInstantiateWithFlags(&exec, clone, flags);
            status = cuGraphUpload(exec, stream);
            cuGraphLaunch(exec, stream);
            status = cuGraphExecDestroy(exec);
            cuGraphDestroyNode(node);
            status = cuGraphDestroy(clone);

            if (cuGraphInstantiateWithFlags(&exec, graph, flags) != CUDA_SUCCESS) {
                return status;
            }

            if (cuGraphLaunch(exec, stream) != CUDA_SUCCESS) {
                return status;
            }

            bool instantiated = checkStatus(
                cuGraphInstantiateWithFlags(&exec, graph, flags)
            );
            bool launched = checkStatus(cuGraphUpload(exec, stream));
            return launched && instantiated ? cuGraphDestroy(graph) : status;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// CUDA driver graph create: output: graph, flags: 0" in result
        assert "// CUDA driver graph clone: output: clone, source: graph" in result
        assert (
            "// CUDA driver graph instantiate: output: exec, graph: graph, "
            "error node output: node, log buffer: log, log bytes: 128"
        ) in result
        assert (
            "// CUDA driver graph instantiate: output: exec, graph: clone, "
            "flags: flags"
        ) in result
        assert (
            "// CUDA driver graph instantiate with flags: output: exec, "
            "graph: clone, flags: flags"
        ) in result
        assert "// CUDA driver graph upload: exec: exec, stream: stream" in result
        assert "// CUDA driver graph launch: exec: exec, stream: stream" in result
        assert "// CUDA driver graph exec destroy: exec" in result
        assert "// CUDA driver graph destroy node: node" in result
        assert "// CUDA driver graph destroy: clone" in result
        assert (
            "if (((/* CUDA driver graph instantiate with flags: "
            "output: exec, graph: graph, flags: flags */ CUDA_SUCCESS) "
            "!= CUDA_SUCCESS))"
        ) in result
        assert (
            "if (((/* CUDA driver graph launch: exec: exec, stream: stream */ "
            "CUDA_SUCCESS) != CUDA_SUCCESS))"
        ) in result
        assert (
            "checkStatus((/* CUDA driver graph instantiate with flags: "
            "output: exec, graph: graph, flags: flags */ CUDA_SUCCESS))"
        ) in result
        assert (
            "checkStatus((/* CUDA driver graph upload: exec: exec, stream: stream */ "
            "CUDA_SUCCESS))"
        ) in result
        assert (
            "((launched && instantiated) ? "
            "(/* CUDA driver graph destroy: graph */ CUDA_SUCCESS) : status)"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuGraphCreate",
            "cuGraphClone",
            "cuGraphInstantiate",
            "cuGraphInstantiateWithFlags",
            "cuGraphUpload",
            "cuGraphLaunch",
            "cuGraphExecDestroy",
            "cuGraphDestroyNode",
            "cuGraphDestroy",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_graph_instantiate_update_params_runtime_conversion(self):
        code = """
        void refreshGraph(cudaGraph_t graph, cudaGraphExec_t exec) {
            cudaGraphExec_t newExec;
            cudaGraphInstantiateParams instantiateParams;
            cudaGraphExecUpdateResultInfo resultInfo;

            cudaError_t err = cudaGraphInstantiateWithParams(
                &newExec,
                graph,
                &instantiateParams
            );
            err = cudaGraphExecUpdate(exec, graph, &resultInfo);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA graph instantiate with params: output: newExec, graph: graph, "
            "params: (&instantiateParams)"
        ) in result
        assert (
            "// CUDA graph exec update: exec: exec, graph: graph, "
            "result info: (&resultInfo)"
        ) in result
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "err = cudaSuccess;" in result
        for function_name in [
            "cudaGraphInstantiateWithParams",
            "cudaGraphExecUpdate",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_device_graph_launch_runtime_conversion(self):
        code = """
        __global__ void launchDeviceGraphs(cudaGraphExec_t child) {
            cudaGraphExec_t current = cudaGetCurrentGraphExec();
            cudaGraphExec_t tail;

            cudaError_t err = cudaGraphLaunch(
                child,
                cudaStreamGraphFireAndForget
            );
            tail = cudaGetCurrentGraphExec();
            err = cudaGraphLaunch(tail, cudaStreamGraphTailLaunch);
            err = cudaGraphLaunch(
                cudaGetCurrentGraphExec(),
                cudaStreamGraphFireAndForgetAsSibling
            );
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// CUDA device graph get current exec" in result
        assert "var current: cudaGraphExec_t = 0;" in result
        assert "tail = 0;" in result
        assert (
            "// CUDA graph device launch: exec: child, mode: fire-and-forget" in result
        )
        assert "// CUDA graph device launch: exec: tail, mode: tail" in result
        assert (
            "// CUDA graph device launch: "
            "exec: (/* CUDA device graph get current exec */ 0), "
            "mode: fire-and-forget sibling"
        ) in result
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "err = cudaSuccess;" in result
        assert "cudaGetCurrentGraphExec(" not in result
        assert "cudaGraphLaunch(" not in result

    def test_cuda_device_graph_kernel_node_update_runtime_conversion(self):
        code = """
        __device__ void updateDeviceGraphKernelNode(
            cudaGraphDeviceNode_t node,
            cudaGraphKernelNodeUpdate* updates,
            int value
        ) {
            cudaError_t err = cudaGraphKernelNodeSetEnabled(node, true);
            err = cudaGraphKernelNodeSetGridDim(node, dim3(2, 3, 4));
            err = cudaGraphKernelNodeSetParam(node, 16, &value, sizeof(value));
            err = cudaGraphKernelNodeUpdatesApply(updates, 2);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA device graph kernel node set enabled: " "node: node, enabled: true"
        ) in result
        assert (
            "// CUDA device graph kernel node set grid dim: "
            "node: node, grid dim: vec3<u32>(2, 3, 4)"
        ) in result
        assert (
            "// CUDA device graph kernel node set param: "
            "node: node, offset: 16, value: (&value), bytes: sizeof(value)"
        ) in result
        assert (
            "// CUDA device graph kernel node updates apply: "
            "updates: updates, count: 2"
        ) in result
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "err = cudaSuccess;" in result
        for function_name in [
            "cudaGraphKernelNodeSetEnabled",
            "cudaGraphKernelNodeSetGridDim",
            "cudaGraphKernelNodeSetParam",
            "cudaGraphKernelNodeUpdatesApply",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_graph_conditional_runtime_conversion(self):
        code = """
        __device__ void setConditional(
            cudaGraphConditionalHandle handle,
            unsigned int value
        ) {
            cudaGraphSetConditional(handle, value);
        }

        void buildConditionalGraph(
            cudaGraph_t graph,
            cudaGraphNode_t* deps,
            cudaGraphEdgeData* edgeData,
            size_t count
        ) {
            cudaGraphConditionalHandle handle;
            cudaGraphConditionalHandle handleWithContext;
            cudaExecutionContext_t context;
            cudaGraphNode_t conditionalNode;
            cudaGraphNode_t legacyConditionalNode;
            cudaGraphNodeParams params;

            cudaError_t err = cudaGraphConditionalHandleCreate(
                &handle,
                graph,
                1u,
                cudaGraphCondAssignDefault
            );
            err = cudaGraphConditionalHandleCreate_v2(
                &handleWithContext,
                graph,
                context,
                0u,
                0
            );
            params.type = cudaGraphNodeTypeConditional;
            params.conditional.handle = handle;
            params.conditional.type = cudaGraphCondTypeIf;
            params.conditional.size = 2;
            err = cudaGraphAddNode(
                &conditionalNode,
                graph,
                deps,
                edgeData,
                count,
                &params
            );
            err = cudaGraphAddNode(
                &legacyConditionalNode,
                graph,
                deps,
                count,
                &params
            );
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// CUDA graph set conditional: handle: handle, value: value" in result
        assert (
            "// CUDA graph conditional handle create: output: handle, "
            "graph: graph, default launch value: 1u, "
            "flags: cudaGraphCondAssignDefault"
        ) in result
        assert (
            "// CUDA graph conditional handle create v2: output: handleWithContext, "
            "graph: graph, context: context, default launch value: 0u, flags: 0"
        ) in result
        assert (
            "// CUDA graph add generic node: output: conditionalNode, graph: graph, "
            "dependencies: deps, edge data: edgeData, dependency count: count, "
            "params: (&params)"
        ) in result
        assert (
            "// CUDA graph add generic node: output: legacyConditionalNode, "
            "graph: graph, dependencies: deps, dependency count: count, "
            "params: (&params)"
        ) in result
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "err = cudaSuccess;" in result
        for function_name in [
            "cudaGraphSetConditional",
            "cudaGraphConditionalHandleCreate",
            "cudaGraphConditionalHandleCreate_v2",
            "cudaGraphAddNode",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_graph_node_update_runtime_conversion(self):
        code = """
        void updateGraph(
            cudaGraph_t graph,
            cudaGraphExec_t exec,
            cudaGraphNode_t kernelNode,
            cudaGraphNode_t memcpyNode,
            cudaGraphNode_t memsetNode,
            cudaGraphNode_t hostNode
        ) {
            cudaKernelNodeParams kernelParams;
            cudaMemcpy3DParms copyParams;
            cudaMemsetParams memsetParams;
            cudaHostNodeParams hostParams;
            cudaGraphNode_t errorNode;
            cudaGraphExecUpdateResult updateResult;

            cudaGraphKernelNodeGetParams(kernelNode, &kernelParams);
            cudaGraphKernelNodeSetParams(kernelNode, &kernelParams);
            cudaGraphMemcpyNodeGetParams(memcpyNode, &copyParams);
            cudaGraphMemcpyNodeSetParams(memcpyNode, &copyParams);
            cudaGraphMemsetNodeGetParams(memsetNode, &memsetParams);
            cudaGraphMemsetNodeSetParams(memsetNode, &memsetParams);
            cudaGraphHostNodeGetParams(hostNode, &hostParams);
            cudaGraphHostNodeSetParams(hostNode, &hostParams);
            cudaError_t err = cudaGraphExecUpdate(
                exec,
                graph,
                &errorNode,
                &updateResult
            );
            err = cudaGraphExecKernelNodeSetParams(
                exec,
                kernelNode,
                &kernelParams
            );
            err = cudaGraphExecMemcpyNodeSetParams(
                exec,
                memcpyNode,
                &copyParams
            );
            err = cudaGraphExecMemsetNodeSetParams(
                exec,
                memsetNode,
                &memsetParams
            );
            err = cudaGraphExecHostNodeSetParams(exec, hostNode, &hostParams);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA graph kernel node get params: node: kernelNode, "
            "params: (&kernelParams)"
        ) in result
        assert (
            "// CUDA graph kernel node set params: node: kernelNode, "
            "params: (&kernelParams)"
        ) in result
        assert (
            "// CUDA graph memcpy node get params: node: memcpyNode, "
            "params: (&copyParams)"
        ) in result
        assert (
            "// CUDA graph memcpy node set params: node: memcpyNode, "
            "params: (&copyParams)"
        ) in result
        assert (
            "// CUDA graph memset node get params: node: memsetNode, "
            "params: (&memsetParams)"
        ) in result
        assert (
            "// CUDA graph memset node set params: node: memsetNode, "
            "params: (&memsetParams)"
        ) in result
        assert (
            "// CUDA graph host node get params: node: hostNode, "
            "params: (&hostParams)"
        ) in result
        assert (
            "// CUDA graph host node set params: node: hostNode, "
            "params: (&hostParams)"
        ) in result
        assert (
            "// CUDA graph exec update: exec: exec, graph: graph, "
            "error node output: errorNode, result output: updateResult"
        ) in result
        assert (
            "// CUDA graph exec set kernel node params: exec: exec, "
            "node: kernelNode, params: (&kernelParams)"
        ) in result
        assert (
            "// CUDA graph exec set memcpy node params: exec: exec, "
            "node: memcpyNode, params: (&copyParams)"
        ) in result
        assert (
            "// CUDA graph exec set memset node params: exec: exec, "
            "node: memsetNode, params: (&memsetParams)"
        ) in result
        assert (
            "// CUDA graph exec set host node params: exec: exec, "
            "node: hostNode, params: (&hostParams)"
        ) in result
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "err = cudaSuccess;" in result
        for function_name in [
            "cudaGraphKernelNodeGetParams",
            "cudaGraphKernelNodeSetParams",
            "cudaGraphMemcpyNodeGetParams",
            "cudaGraphMemcpyNodeSetParams",
            "cudaGraphMemsetNodeGetParams",
            "cudaGraphMemsetNodeSetParams",
            "cudaGraphHostNodeGetParams",
            "cudaGraphHostNodeSetParams",
            "cudaGraphExecUpdate",
            "cudaGraphExecKernelNodeSetParams",
            "cudaGraphExecMemcpyNodeSetParams",
            "cudaGraphExecMemsetNodeSetParams",
            "cudaGraphExecHostNodeSetParams",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_graph_kernel_node_attribute_runtime_conversion(self):
        code = """
        void updateKernelAttributes(
            cudaGraphNode_t kernelNode,
            cudaGraphNode_t targetNode
        ) {
            cudaKernelNodeAttrValue attrValue;
            cudaError_t err = cudaGraphKernelNodeGetAttribute(
                kernelNode,
                cudaKernelNodeAttributeAccessPolicyWindow,
                &attrValue
            );
            err = cudaGraphKernelNodeSetAttribute(
                kernelNode,
                cudaKernelNodeAttributeAccessPolicyWindow,
                &attrValue
            );
            err = cudaGraphKernelNodeCopyAttributes(kernelNode, targetNode);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA graph kernel node get attribute: node: kernelNode, "
            "attribute: cudaKernelNodeAttributeAccessPolicyWindow, "
            "output: attrValue"
        ) in result
        assert (
            "// CUDA graph kernel node set attribute: node: kernelNode, "
            "attribute: cudaKernelNodeAttributeAccessPolicyWindow, "
            "value: (&attrValue)"
        ) in result
        assert (
            "// CUDA graph kernel node copy attributes: source: kernelNode, "
            "destination: targetNode"
        ) in result
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "err = cudaSuccess;" in result
        for function_name in [
            "cudaGraphKernelNodeGetAttribute",
            "cudaGraphKernelNodeSetAttribute",
            "cudaGraphKernelNodeCopyAttributes",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_graph_exec_memcpy_convenience_update_runtime_conversion(self):
        code = """
        void updateExecMemcpy(
            cudaGraphExec_t exec,
            cudaGraphNode_t memcpyNode,
            void* dst,
            void* src,
            size_t bytes,
            size_t offset
        ) {
            int symbolData;
            cudaError_t err = cudaGraphExecMemcpyNodeSetParams1D(
                exec,
                memcpyNode,
                dst,
                src,
                bytes,
                cudaMemcpyDeviceToDevice
            );
            err = cudaGraphExecMemcpyNodeSetParamsFromSymbol(
                exec,
                memcpyNode,
                dst,
                symbolData,
                bytes,
                offset,
                cudaMemcpyDeviceToDevice
            );
            err = cudaGraphExecMemcpyNodeSetParamsToSymbol(
                exec,
                memcpyNode,
                symbolData,
                src,
                bytes,
                offset,
                cudaMemcpyDeviceToDevice
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
            "// CUDA graph exec set memcpy 1D node params: exec: exec, "
            "node: memcpyNode, dst: dst, src: src, byte count: bytes, "
            "kind: cudaMemcpyDeviceToDevice"
        ) in result
        assert (
            "// CUDA graph exec set memcpy-from-symbol node params: exec: exec, "
            "node: memcpyNode, dst: dst, symbol: symbolData, byte count: bytes, "
            "offset: offset, kind: cudaMemcpyDeviceToDevice"
        ) in result
        assert (
            "// CUDA graph exec set memcpy-to-symbol node params: exec: exec, "
            "node: memcpyNode, symbol: symbolData, src: src, byte count: bytes, "
            "offset: offset, kind: cudaMemcpyDeviceToDevice"
        ) in result
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "err = cudaSuccess;" in result
        for function_name in [
            "cudaGraphExecMemcpyNodeSetParams1D",
            "cudaGraphExecMemcpyNodeSetParamsFromSymbol",
            "cudaGraphExecMemcpyNodeSetParamsToSymbol",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_graph_memcpy_convenience_node_update_runtime_conversion(self):
        code = """
        void updateMemcpyNode(
            cudaGraphNode_t memcpyNode,
            void* dst,
            void* src,
            size_t bytes,
            size_t offset
        ) {
            int symbolData;
            cudaError_t err = cudaGraphMemcpyNodeSetParams1D(
                memcpyNode,
                dst,
                src,
                bytes,
                cudaMemcpyDeviceToDevice
            );
            err = cudaGraphMemcpyNodeSetParamsFromSymbol(
                memcpyNode,
                dst,
                symbolData,
                bytes,
                offset,
                cudaMemcpyDeviceToDevice
            );
            err = cudaGraphMemcpyNodeSetParamsToSymbol(
                memcpyNode,
                symbolData,
                src,
                bytes,
                offset,
                cudaMemcpyDeviceToDevice
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
            "// CUDA graph set memcpy 1D node params: node: memcpyNode, "
            "dst: dst, src: src, byte count: bytes, "
            "kind: cudaMemcpyDeviceToDevice"
        ) in result
        assert (
            "// CUDA graph set memcpy-from-symbol node params: "
            "node: memcpyNode, dst: dst, symbol: symbolData, byte count: bytes, "
            "offset: offset, kind: cudaMemcpyDeviceToDevice"
        ) in result
        assert (
            "// CUDA graph set memcpy-to-symbol node params: "
            "node: memcpyNode, symbol: symbolData, src: src, byte count: bytes, "
            "offset: offset, kind: cudaMemcpyDeviceToDevice"
        ) in result
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "err = cudaSuccess;" in result
        for function_name in [
            "cudaGraphMemcpyNodeSetParams1D",
            "cudaGraphMemcpyNodeSetParamsFromSymbol",
            "cudaGraphMemcpyNodeSetParamsToSymbol",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_graph_batch_mem_op_node_runtime_conversion(self):
        code = """
        void updateBatchMemOpNode(
            CUgraph graph,
            CUgraphExec exec,
            CUgraphNode* deps,
            size_t count
        ) {
            CUgraphNode batchNode;
            CUDA_BATCH_MEM_OP_NODE_PARAMS params;

            CUresult status = cuGraphAddBatchMemOpNode(
                &batchNode,
                graph,
                deps,
                count,
                &params
            );
            status = cuGraphBatchMemOpNodeGetParams(batchNode, &params);
            status = cuGraphBatchMemOpNodeSetParams(batchNode, &params);
            status = cuGraphExecBatchMemOpNodeSetParams(exec, batchNode, &params);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver graph add batch memory operation node: "
            "output: batchNode, graph: graph, dependencies: deps, "
            "dependency count: count, params: (&params)"
        ) in result
        assert (
            "// CUDA driver graph batch memory operation node get params: "
            "node: batchNode, params: (&params)"
        ) in result
        assert (
            "// CUDA driver graph batch memory operation node set params: "
            "node: batchNode, params: (&params)"
        ) in result
        assert (
            "// CUDA driver graph exec set batch memory operation node params: "
            "exec: exec, node: batchNode, params: (&params)"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuGraphAddBatchMemOpNode",
            "cuGraphBatchMemOpNodeGetParams",
            "cuGraphBatchMemOpNodeSetParams",
            "cuGraphExecBatchMemOpNodeSetParams",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_stream_memory_operation_runtime_conversion(self):
        code = """
        void scheduleStreamMemOps(
            CUstream stream,
            CUdeviceptr addr32,
            CUdeviceptr addr64,
            CUstreamBatchMemOpParams* ops,
            unsigned int flags
        ) {
            cuuint32_t value32 = 7;
            cuuint64_t value64 = 9;

            CUresult status = cuStreamWaitValue32(
                stream,
                addr32,
                value32,
                flags
            );
            status = cuStreamWaitValue64(stream, addr64, value64, flags);
            status = cuStreamWriteValue32(
                stream,
                addr32,
                value32,
                CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER
            );
            status = cuStreamWriteValue64(stream, addr64, value64, 0);
            status = cuStreamBatchMemOp(stream, 2, ops, 0);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver stream wait 32-bit value: stream: stream, "
            "address: addr32, value: value32, flags: flags"
        ) in result
        assert (
            "// CUDA driver stream wait 64-bit value: stream: stream, "
            "address: addr64, value: value64, flags: flags"
        ) in result
        assert (
            "// CUDA driver stream write 32-bit value: stream: stream, "
            "address: addr32, value: value32, "
            "flags: CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER"
        ) in result
        assert (
            "// CUDA driver stream write 64-bit value: stream: stream, "
            "address: addr64, value: value64, flags: 0"
        ) in result
        assert (
            "// CUDA driver stream batch memory operation: stream: stream, "
            "count: 2, params: ops, flags: 0"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuStreamWaitValue32",
            "cuStreamWaitValue64",
            "cuStreamWriteValue32",
            "cuStreamWriteValue64",
            "cuStreamBatchMemOp",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_stream_attribute_query_runtime_conversion(self):
        code = """
        void inspectStreamAttributes(
            CUstream stream,
            CUstream source,
            CUstreamAttrValue* attrValue,
            CUgraphNode** dependencies,
            size_t* dependencyCount,
            unsigned int flags
        ) {
            CUcontext context;
            CUgraph graph;
            CUstreamCaptureStatus captureStatus;
            cuuint64_t captureId;
            int priority;
            unsigned int streamFlags;

            CUresult status = cuStreamGetAttribute(
                stream,
                CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                attrValue
            );
            status = cuStreamSetAttribute(
                stream,
                CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY,
                attrValue
            );
            status = cuStreamCopyAttributes(stream, source);
            status = cuStreamGetCtx(stream, &context);
            status = cuStreamGetFlags(stream, &streamFlags);
            status = cuStreamGetPriority(stream, &priority);
            status = cuStreamIsCapturing(stream, &captureStatus);
            status = cuStreamGetCaptureInfo_v2(
                stream,
                &captureStatus,
                &captureId,
                &graph,
                dependencies,
                dependencyCount
            );
            status = cuStreamUpdateCaptureDependencies(
                stream,
                dependencies,
                *dependencyCount,
                flags
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
            "// CUDA driver stream get attribute: stream: stream, "
            "attribute: CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, "
            "output: attrValue"
        ) in result
        assert (
            "// CUDA driver stream set attribute: stream: stream, "
            "attribute: CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY, "
            "value: attrValue"
        ) in result
        assert (
            "// CUDA driver stream copy attributes: destination: stream, "
            "source: source"
        ) in result
        assert (
            "// CUDA driver stream context query: stream: stream, output: context"
            in result
        )
        assert (
            "// CUDA driver stream flags query: stream: stream, output: streamFlags"
            in result
        )
        assert (
            "// CUDA driver stream priority query: stream: stream, output: priority"
            in result
        )
        assert (
            "// CUDA driver stream capture status query: stream: stream, "
            "output: captureStatus"
        ) in result
        assert (
            "// CUDA driver stream capture info query: stream: stream, "
            "status output: captureStatus, id output: captureId, "
            "graph output: graph, dependencies output: dependencies, "
            "dependency count output: dependencyCount"
        ) in result
        assert (
            "// CUDA driver stream update capture dependencies: stream: stream, "
            "dependencies: dependencies, dependency count: (*dependencyCount), "
            "flags: flags"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuStreamGetAttribute",
            "cuStreamSetAttribute",
            "cuStreamCopyAttributes",
            "cuStreamGetCtx",
            "cuStreamGetFlags",
            "cuStreamGetPriority",
            "cuStreamIsCapturing",
            "cuStreamGetCaptureInfo_v2",
            "cuStreamUpdateCaptureDependencies",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_stream_lifecycle_synchronization_runtime_conversion(self):
        code = """
        CUresult controlStream(
            CUstream stream,
            CUevent event,
            CUhostFn callback,
            void* userData,
            CUdeviceptr devicePtr,
            size_t length,
            int priority
        ) {
            CUstream created;
            CUstream priorityStream;
            CUgraph graph;

            CUresult status = cuStreamCreate(
                &created,
                CU_STREAM_NON_BLOCKING
            );
            status = cuStreamCreateWithPriority(
                &priorityStream,
                CU_STREAM_NON_BLOCKING,
                priority
            );
            status = cuStreamQuery(stream);
            status = cuStreamSynchronize(stream);
            status = cuStreamWaitEvent(stream, event, 0);
            status = cuLaunchHostFunc(stream, callback, userData);
            status = cuStreamAttachMemAsync(
                stream,
                devicePtr,
                length,
                CU_MEM_ATTACH_GLOBAL
            );
            cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL);
            status = cuStreamEndCapture(stream, &graph);
            status = cuStreamDestroy(created);

            return cuStreamSynchronize(priorityStream);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver stream create: output: created, "
            "flags: CU_STREAM_NON_BLOCKING"
        ) in result
        assert (
            "// CUDA driver stream create with priority: output: priorityStream, "
            "flags: CU_STREAM_NON_BLOCKING, priority: priority"
        ) in result
        assert "// CUDA driver stream query: stream: stream" in result
        assert "// CUDA driver stream synchronize: stream: stream" in result
        assert (
            "// CUDA driver stream wait event: stream: stream, event: event, flags: 0"
            in result
        )
        assert (
            "// CUDA driver stream launch host function: stream: stream, "
            "callback: callback, user data: userData"
        ) in result
        assert (
            "// CUDA driver stream attach memory: stream: stream, "
            "pointer: devicePtr, bytes: length, flags: CU_MEM_ATTACH_GLOBAL"
        ) in result
        assert (
            "// CUDA driver stream begin capture: stream: stream, "
            "mode: CU_STREAM_CAPTURE_MODE_GLOBAL"
        ) in result
        assert (
            "// CUDA driver stream end capture: stream: stream, graph output: graph"
            in result
        )
        assert "// CUDA driver stream destroy: stream: created" in result
        assert "return CUDA_SUCCESS;" in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuStreamCreate",
            "cuStreamCreateWithPriority",
            "cuStreamQuery",
            "cuStreamSynchronize",
            "cuStreamWaitEvent",
            "cuLaunchHostFunc",
            "cuStreamAttachMemAsync",
            "cuStreamBeginCapture",
            "cuStreamEndCapture",
            "cuStreamDestroy",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_profiler_stream_callback_runtime_conversion(self):
        code = """
        CUresult profileStream(
            CUstream stream,
            CUstreamCallback callback,
            void* userData,
            const char* configFile,
            const char* outputFile,
            CUoutput_mode outputMode,
            unsigned int flags,
            bool ok
        ) {
            CUresult status = cuProfilerInitialize(
                configFile,
                outputFile,
                outputMode
            );
            status = cuProfilerStart();
            status = cuStreamAddCallback(stream, callback, userData, 0);
            cuProfilerStop();

            if (cuStreamAddCallback(stream, callback, userData, flags)
                != CUDA_SUCCESS) {
                return status;
            }

            checkStatus(cuProfilerStart());
            return ok ? cuProfilerStop() : status;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver profiler initialize: config: configFile, "
            "output: outputFile, mode: outputMode"
        ) in result
        assert "// CUDA driver profiler start" in result
        assert "// CUDA driver profiler stop" in result
        assert (
            "// CUDA driver stream add callback: stream: stream, "
            "callback: callback, user data: userData, flags: 0"
        ) in result
        assert (
            "/* CUDA driver stream add callback: stream: stream, "
            "callback: callback, user data: userData, flags: flags */ "
            "CUDA_SUCCESS) != CUDA_SUCCESS"
        ) in result
        assert "checkStatus((/* CUDA driver profiler start */ CUDA_SUCCESS));" in result
        assert (
            "return (ok ? (/* CUDA driver profiler stop */ CUDA_SUCCESS) : status);"
            in result
        )
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuProfilerInitialize",
            "cuProfilerStart",
            "cuProfilerStop",
            "cuStreamAddCallback",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_event_lifecycle_query_runtime_conversion(self):
        code = """
        CUresult controlEvents(CUstream stream) {
            CUevent start;
            CUevent stop;
            float elapsed;

            CUresult status = cuEventCreate(&start, CU_EVENT_DEFAULT);
            status = cuEventCreateWithFlags(&stop, CU_EVENT_BLOCKING_SYNC);
            status = cuEventRecord(start, stream);
            status = cuEventRecordWithFlags(
                stop,
                stream,
                CU_EVENT_RECORD_DEFAULT
            );
            status = cuEventQuery(start);
            status = cuEventSynchronize(stop);
            status = cuEventElapsedTime(&elapsed, start, stop);
            cuEventDestroy(start);

            return cuEventDestroy(stop);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver event create: output: start, flags: CU_EVENT_DEFAULT"
            in result
        )
        assert (
            "// CUDA driver event create: output: stop, "
            "flags: CU_EVENT_BLOCKING_SYNC"
        ) in result
        assert "// CUDA driver event record: event: start, stream: stream" in result
        assert (
            "// CUDA driver event record with flags: event: stop, stream: stream, "
            "flags: CU_EVENT_RECORD_DEFAULT"
        ) in result
        assert "// CUDA driver event query: event: start" in result
        assert "// CUDA driver event synchronize: event: stop" in result
        assert (
            "// CUDA driver event elapsed time: start -> stop, output: elapsed"
            in result
        )
        assert "// CUDA driver event destroy: event: start" in result
        assert "// CUDA driver event destroy: event: stop" in result
        assert "return CUDA_SUCCESS;" in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuEventCreate",
            "cuEventCreateWithFlags",
            "cuEventRecord",
            "cuEventRecordWithFlags",
            "cuEventQuery",
            "cuEventSynchronize",
            "cuEventElapsedTime",
            "cuEventDestroy",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_memory_allocation_copy_runtime_conversion(self):
        code = """
        CUresult driverMemory(
            CUstream stream,
            CUdeviceptr source,
            void* host,
            size_t bytes,
            size_t pitch,
            size_t width,
            size_t height,
            size_t elements
        ) {
            CUdeviceptr device;
            CUdeviceptr managed;
            void* pinned;
            CUDA_MEMCPY2D copy2d;
            CUDA_MEMCPY3D copy3d;

            CUresult status = cuMemAlloc(&device, bytes);
            status = cuMemAllocManaged(&managed, bytes, CU_MEM_ATTACH_GLOBAL);
            status = cuMemAllocHost(&pinned, bytes);
            status = cuMemHostAlloc(&pinned, bytes, CU_MEMHOSTALLOC_PORTABLE);
            status = cuMemcpy(device, source, bytes);
            status = cuMemcpyAsync(device, source, bytes, stream);
            status = cuMemcpyHtoD(managed, host, bytes);
            status = cuMemcpyDtoH(host, device, bytes);
            status = cuMemcpyDtoD(device, managed, bytes);
            status = cuMemcpyHtoDAsync(device, host, bytes, stream);
            status = cuMemcpyDtoHAsync(host, managed, bytes, stream);
            status = cuMemcpyDtoDAsync(managed, source, bytes, stream);
            status = cuMemcpy2D(&copy2d);
            status = cuMemcpy2DAsync(&copy2d, stream);
            status = cuMemcpy3D(&copy3d);
            status = cuMemcpy3DAsync(&copy3d, stream);
            status = cuMemsetD8(device, 0, bytes);
            status = cuMemsetD32(managed, 0, elements);
            status = cuMemsetD8Async(device, 1, bytes, stream);
            status = cuMemsetD32Async(managed, 2, elements, stream);
            status = cuMemsetD2D8(device, pitch, 0, width, height);
            status = cuMemsetD2D32(managed, pitch, 3, width, height);
            status = cuMemsetD2D8Async(device, pitch, 4, width, height, stream);
            status = cuMemsetD2D32Async(managed, pitch, 5, width, height, stream);
            cuMemFreeHost(pinned);
            status = cuMemFree(managed);

            return cuMemFree(device);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// CUDA driver memory allocate: output: device, bytes: bytes" in result
        assert (
            "// CUDA driver memory allocate managed: output: managed, "
            "bytes: bytes, flags: CU_MEM_ATTACH_GLOBAL"
        ) in result
        assert (
            "// CUDA driver host memory allocate: output: pinned, bytes: bytes"
            in result
        )
        assert (
            "// CUDA driver host memory allocate: output: pinned, bytes: bytes, "
            "flags: CU_MEMHOSTALLOC_PORTABLE"
        ) in result
        assert "// CUDA driver memory copy: source -> device, bytes: bytes" in result
        assert (
            "// CUDA driver memory copy: source -> device, bytes: bytes, "
            "stream: stream"
        ) in result
        assert (
            "// CUDA driver memory copy HtoD: host -> managed, bytes: bytes" in result
        )
        assert "// CUDA driver memory copy DtoH: device -> host, bytes: bytes" in result
        assert (
            "// CUDA driver memory copy DtoD: managed -> device, bytes: bytes" in result
        )
        assert (
            "// CUDA driver memory copy HtoD: host -> device, bytes: bytes, "
            "stream: stream"
        ) in result
        assert (
            "// CUDA driver memory copy DtoH: managed -> host, bytes: bytes, "
            "stream: stream"
        ) in result
        assert (
            "// CUDA driver memory copy DtoD: source -> managed, bytes: bytes, "
            "stream: stream"
        ) in result
        assert "// CUDA driver memory copy 2D: params: copy2d" in result
        assert "// CUDA driver memory copy 2D: params: copy2d, stream: stream" in result
        assert "// CUDA driver memory copy 3D: params: copy3d" in result
        assert "// CUDA driver memory copy 3D: params: copy3d, stream: stream" in result
        assert "// CUDA driver memory set D8: device, value: 0, count: bytes" in result
        assert (
            "// CUDA driver memory set D32: managed, value: 0, count: elements"
            in result
        )
        assert (
            "// CUDA driver memory set D8: device, value: 1, count: bytes, "
            "stream: stream"
        ) in result
        assert (
            "// CUDA driver memory set D32: managed, value: 2, count: elements, "
            "stream: stream"
        ) in result
        assert (
            "// CUDA driver memory set 2D D8: device, pitch: pitch, value: 0, "
            "width: width, height: height"
        ) in result
        assert (
            "// CUDA driver memory set 2D D32: managed, pitch: pitch, value: 3, "
            "width: width, height: height"
        ) in result
        assert (
            "// CUDA driver memory set 2D D8: device, pitch: pitch, value: 4, "
            "width: width, height: height, stream: stream"
        ) in result
        assert (
            "// CUDA driver memory set 2D D32: managed, pitch: pitch, value: 5, "
            "width: width, height: height, stream: stream"
        ) in result
        assert "// CUDA driver host memory free: pinned" in result
        assert "// CUDA driver memory free: managed" in result
        assert "// CUDA driver memory free: device" in result
        assert "return CUDA_SUCCESS;" in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuMemAlloc",
            "cuMemAllocManaged",
            "cuMemAllocHost",
            "cuMemHostAlloc",
            "cuMemcpy",
            "cuMemcpyAsync",
            "cuMemcpyHtoD",
            "cuMemcpyDtoH",
            "cuMemcpyDtoD",
            "cuMemcpyHtoDAsync",
            "cuMemcpyDtoHAsync",
            "cuMemcpyDtoDAsync",
            "cuMemcpy2D",
            "cuMemcpy2DAsync",
            "cuMemcpy3D",
            "cuMemcpy3DAsync",
            "cuMemsetD8",
            "cuMemsetD32",
            "cuMemsetD8Async",
            "cuMemsetD32Async",
            "cuMemsetD2D8",
            "cuMemsetD2D32",
            "cuMemsetD2D8Async",
            "cuMemsetD2D32Async",
            "cuMemFreeHost",
            "cuMemFree",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_uva_memory_range_runtime_conversion(self):
        code = """
        bool checkStatus(CUresult status) {
            return status == CUDA_SUCCESS;
        }

        CUresult inspectMemory(
            CUdeviceptr pointer,
            CUdeviceptr alternate,
            CUdevice device,
            size_t bytes
        ) {
            void* base;
            size_t rangeBytes;
            size_t freeBytes;
            size_t totalBytes;
            int memoryType;
            CUpointer_attribute attrs[2];
            void* values[2];

            CUresult status = cuPointerGetAttribute(
                &memoryType,
                CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                pointer
            );
            status = cuPointerGetAttributes(2, attrs, values, pointer);
            status = cuMemGetAddressRange(&base, &rangeBytes, pointer);
            status = cuMemGetInfo(&freeBytes, &totalBytes);
            status = cuMemGetInfo_v2(&freeBytes, &totalBytes);
            status = cuMemAdvise(
                pointer,
                bytes,
                CU_MEM_ADVISE_SET_READ_MOSTLY,
                device
            );
            cuMemAdvise(
                alternate,
                bytes,
                CU_MEM_ADVISE_UNSET_READ_MOSTLY,
                device
            );

            if (cuMemGetAddressRange(&base, &rangeBytes, alternate)
                != CUDA_SUCCESS) {
                return CUDA_ERROR_UNKNOWN;
            }

            bool ok = checkStatus(cuMemGetInfo(&freeBytes, &totalBytes));
            return ok ? cuMemAdvise(
                pointer,
                rangeBytes,
                CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                device
            ) : status;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver pointer get attribute: output: memoryType, "
            "attribute: CU_POINTER_ATTRIBUTE_MEMORY_TYPE, pointer: pointer"
        ) in result
        assert (
            "// CUDA driver pointer get attributes: count: 2, attributes: attrs, "
            "data: values, pointer: pointer"
        ) in result
        assert (
            "// CUDA driver memory get address range: base output: base, "
            "size output: rangeBytes, pointer: pointer"
        ) in result
        assert (
            "// CUDA driver memory get info: free output: freeBytes, "
            "total output: totalBytes"
        ) in result
        assert (
            "// CUDA driver memory get info v2: free output: freeBytes, "
            "total output: totalBytes"
        ) in result
        assert (
            "// CUDA driver memory advise: pointer: pointer, bytes: bytes, "
            "advice: CU_MEM_ADVISE_SET_READ_MOSTLY, device: device"
        ) in result
        assert (
            "// CUDA driver memory advise: pointer: alternate, bytes: bytes, "
            "advice: CU_MEM_ADVISE_UNSET_READ_MOSTLY, device: device"
        ) in result
        assert (
            "if (((/* CUDA driver memory get address range: base output: base, "
            "size output: rangeBytes, pointer: alternate */ CUDA_SUCCESS) "
            "!= CUDA_SUCCESS))"
        ) in result
        assert (
            "checkStatus((/* CUDA driver memory get info: free output: freeBytes, "
            "total output: totalBytes */ CUDA_SUCCESS))"
        ) in result
        assert (
            "(ok ? (/* CUDA driver memory advise: pointer: pointer, "
            "bytes: rangeBytes, advice: CU_MEM_ADVISE_SET_PREFERRED_LOCATION, "
            "device: device */ CUDA_SUCCESS) : status)"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuPointerGetAttribute",
            "cuPointerGetAttributes",
            "cuMemGetAddressRange",
            "cuMemGetInfo",
            "cuMemGetInfo_v2",
            "cuMemAdvise",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_memory_pool_async_runtime_conversion(self):
        code = """
        bool checkStatus(CUresult status) {
            return status == CUDA_SUCCESS;
        }

        CUresult managePools(
            CUdevice device,
            CUstream stream,
            CUmemPoolProps* props,
            CUmemAccessDesc* accessDesc,
            CUmemLocation* location,
            size_t bytes,
            size_t trimBytes
        ) {
            CUdeviceptr pointer;
            CUmemoryPool pool;
            CUmemoryPool defaultPool;
            CUmemoryPool currentPool;
            cuuint64_t threshold;
            CUmemAccess_flags flags;

            CUresult status = cuMemPoolCreate(&pool, props);
            status = cuDeviceGetDefaultMemPool(&defaultPool, device);
            status = cuDeviceGetMemPool(&currentPool, device);
            status = cuDeviceSetMemPool(device, pool);
            status = cuMemPoolSetAttribute(
                pool,
                CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &threshold
            );
            status = cuMemPoolGetAttribute(
                pool,
                CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &threshold
            );
            status = cuMemPoolSetAccess(pool, accessDesc, 1);
            status = cuMemPoolGetAccess(&flags, pool, location);
            status = cuMemAllocAsync(&pointer, bytes, stream);
            status = cuMemFreeAsync(pointer, stream);
            status = cuMemPoolTrimTo(pool, trimBytes);
            cuMemPoolDestroy(currentPool);

            if (cuMemAllocAsync(&pointer, bytes, stream) != CUDA_SUCCESS) {
                return CUDA_ERROR_UNKNOWN;
            }

            bool ok = checkStatus(cuMemPoolGetAccess(&flags, pool, location));
            return ok ? cuMemPoolDestroy(pool) : status;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver memory pool create: output: pool, props: props"
        ) in result
        assert (
            "// CUDA driver device get default memory pool: output: defaultPool, "
            "device: device"
        ) in result
        assert (
            "// CUDA driver device get memory pool: output: currentPool, "
            "device: device"
        ) in result
        assert (
            "// CUDA driver device set memory pool: device: device, pool: pool"
        ) in result
        assert (
            "// CUDA driver memory pool set attribute: pool: pool, "
            "attribute: CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, value: threshold"
        ) in result
        assert (
            "// CUDA driver memory pool get attribute: pool: pool, "
            "attribute: CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, output: threshold"
        ) in result
        assert (
            "// CUDA driver memory pool set access: pool: pool, "
            "descriptors: accessDesc, count: 1"
        ) in result
        assert (
            "// CUDA driver memory pool get access: flags output: flags, "
            "pool: pool, location: location"
        ) in result
        assert (
            "// CUDA driver memory allocate async: output: pointer, "
            "bytes: bytes, stream: stream"
        ) in result
        assert (
            "// CUDA driver memory free async: pointer: pointer, stream: stream"
        ) in result
        assert (
            "// CUDA driver memory pool trim: pool: pool, bytes: trimBytes"
        ) in result
        assert "// CUDA driver memory pool destroy: pool: currentPool" in result
        assert (
            "if (((/* CUDA driver memory allocate async: output: pointer, "
            "bytes: bytes, stream: stream */ CUDA_SUCCESS) != CUDA_SUCCESS))"
        ) in result
        assert (
            "checkStatus((/* CUDA driver memory pool get access: "
            "flags output: flags, pool: pool, location: location */ CUDA_SUCCESS))"
        ) in result
        assert (
            "(ok ? (/* CUDA driver memory pool destroy: pool: pool */ CUDA_SUCCESS) "
            ": status)"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuMemPoolCreate",
            "cuDeviceGetDefaultMemPool",
            "cuDeviceGetMemPool",
            "cuDeviceSetMemPool",
            "cuMemPoolSetAttribute",
            "cuMemPoolGetAttribute",
            "cuMemPoolSetAccess",
            "cuMemPoolGetAccess",
            "cuMemAllocAsync",
            "cuMemFreeAsync",
            "cuMemPoolTrimTo",
            "cuMemPoolDestroy",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_memory_pool_ipc_runtime_conversion(self):
        code = """
        bool checkStatus(CUresult status) {
            return status == CUDA_SUCCESS;
        }

        CUresult sharePools(
            CUmemoryPool pool,
            CUdeviceptr pointer,
            void* handle,
            CUmemAllocationHandleType handleType,
            unsigned long long flags
        ) {
            CUmemoryPool importedPool;
            CUdeviceptr importedPointer;
            CUmemPoolPtrExportData exportData;

            CUresult status = cuMemPoolExportToShareableHandle(
                handle,
                pool,
                handleType,
                flags
            );
            status = cuMemPoolImportFromShareableHandle(
                &importedPool,
                handle,
                handleType,
                flags
            );
            status = cuMemPoolExportPointer(&exportData, pointer);
            status = cuMemPoolImportPointer(
                &importedPointer,
                importedPool,
                &exportData
            );
            cuMemPoolExportPointer(&exportData, importedPointer);

            if (cuMemPoolImportPointer(
                &importedPointer,
                pool,
                &exportData
            ) != CUDA_SUCCESS) {
                return CUDA_ERROR_UNKNOWN;
            }

            bool ok = checkStatus(
                cuMemPoolExportToShareableHandle(
                    handle,
                    importedPool,
                    handleType,
                    flags
                )
            );
            return ok ? cuMemPoolImportFromShareableHandle(
                &importedPool,
                handle,
                handleType,
                flags
            ) : status;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver memory pool export shareable handle: "
            "handle output: handle, pool: pool, handle type: handleType, "
            "flags: flags"
        ) in result
        assert (
            "// CUDA driver memory pool import shareable handle: "
            "output: importedPool, handle: handle, handle type: handleType, "
            "flags: flags"
        ) in result
        assert (
            "// CUDA driver memory pool export pointer: "
            "share data output: exportData, pointer: pointer"
        ) in result
        assert (
            "// CUDA driver memory pool import pointer: "
            "pointer output: importedPointer, pool: importedPool, "
            "share data: exportData"
        ) in result
        assert (
            "// CUDA driver memory pool export pointer: "
            "share data output: exportData, pointer: importedPointer"
        ) in result
        assert (
            "if (((/* CUDA driver memory pool import pointer: "
            "pointer output: importedPointer, pool: pool, "
            "share data: exportData */ CUDA_SUCCESS) != CUDA_SUCCESS))"
        ) in result
        assert (
            "checkStatus((/* CUDA driver memory pool export shareable handle: "
            "handle output: handle, pool: importedPool, "
            "handle type: handleType, flags: flags */ CUDA_SUCCESS))"
        ) in result
        assert (
            "(ok ? (/* CUDA driver memory pool import shareable handle: "
            "output: importedPool, handle: handle, "
            "handle type: handleType, flags: flags */ CUDA_SUCCESS) : status)"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuMemPoolExportToShareableHandle",
            "cuMemPoolImportFromShareableHandle",
            "cuMemPoolExportPointer",
            "cuMemPoolImportPointer",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_virtual_memory_runtime_conversion(self):
        code = """
        bool checkStatus(CUresult status) {
            return status == CUDA_SUCCESS;
        }

        CUresult manageVirtualMemory(
            CUmemAllocationProp* props,
            CUmemAccessDesc* accessDesc,
            CUmemLocation* location,
            void* osHandle,
            size_t bytes,
            size_t alignment,
            CUmemAllocationHandleType handleType,
            unsigned long long flags
        ) {
            CUdeviceptr address;
            CUmemGenericAllocationHandle allocation;
            CUmemGenericAllocationHandle importedAllocation;
            unsigned long long accessFlags;

            CUresult status = cuMemAddressReserve(
                &address,
                bytes,
                alignment,
                0,
                flags
            );
            status = cuMemCreate(&allocation, bytes, props, flags);
            status = cuMemMap(address, bytes, 0, allocation, flags);
            status = cuMemSetAccess(address, bytes, accessDesc, 1);
            status = cuMemGetAccess(&accessFlags, location, address);
            status = cuMemExportToShareableHandle(
                osHandle,
                allocation,
                handleType,
                flags
            );
            status = cuMemImportFromShareableHandle(
                &importedAllocation,
                osHandle,
                handleType
            );
            status = cuMemRetainAllocationHandle(&allocation, osHandle);
            cuMemUnmap(address, bytes);
            cuMemRelease(importedAllocation);

            if (cuMemMap(address, bytes, 0, allocation, flags) != CUDA_SUCCESS) {
                return CUDA_ERROR_UNKNOWN;
            }

            bool ok = checkStatus(cuMemGetAccess(&accessFlags, location, address));
            status = cuMemAddressFree(address, bytes);
            return ok ? cuMemRelease(allocation) : status;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver virtual memory reserve: output: address, "
            "bytes: bytes, alignment: alignment, address: 0, flags: flags"
        ) in result
        assert (
            "// CUDA driver virtual memory create allocation: "
            "output: allocation, bytes: bytes, props: props, flags: flags"
        ) in result
        assert (
            "// CUDA driver virtual memory map: address: address, "
            "bytes: bytes, offset: 0, allocation: allocation, flags: flags"
        ) in result
        assert (
            "// CUDA driver virtual memory set access: address: address, "
            "bytes: bytes, descriptors: accessDesc, count: 1"
        ) in result
        assert (
            "// CUDA driver virtual memory get access: flags output: accessFlags, "
            "location: location, address: address"
        ) in result
        assert (
            "// CUDA driver virtual memory export shareable handle: "
            "handle output: osHandle, allocation: allocation, "
            "handle type: handleType, flags: flags"
        ) in result
        assert (
            "// CUDA driver virtual memory import shareable handle: "
            "output: importedAllocation, handle: osHandle, handle type: handleType"
        ) in result
        assert (
            "// CUDA driver virtual memory retain allocation handle: "
            "output: allocation, address: osHandle"
        ) in result
        assert (
            "// CUDA driver virtual memory unmap: address: address, bytes: bytes"
        ) in result
        assert (
            "// CUDA driver virtual memory release allocation: allocation: "
            "importedAllocation"
        ) in result
        assert (
            "if (((/* CUDA driver virtual memory map: address: address, "
            "bytes: bytes, offset: 0, allocation: allocation, flags: flags */ "
            "CUDA_SUCCESS) != CUDA_SUCCESS))"
        ) in result
        assert (
            "checkStatus((/* CUDA driver virtual memory get access: "
            "flags output: accessFlags, location: location, address: address */ "
            "CUDA_SUCCESS))"
        ) in result
        assert (
            "(ok ? (/* CUDA driver virtual memory release allocation: "
            "allocation: allocation */ CUDA_SUCCESS) : status)"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuMemAddressReserve",
            "cuMemAddressFree",
            "cuMemCreate",
            "cuMemRelease",
            "cuMemMap",
            "cuMemUnmap",
            "cuMemSetAccess",
            "cuMemGetAccess",
            "cuMemRetainAllocationHandle",
            "cuMemExportToShareableHandle",
            "cuMemImportFromShareableHandle",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_external_memory_semaphore_runtime_conversion(self):
        code = """
        bool checkStatus(CUresult status) {
            return status == CUDA_SUCCESS;
        }

        CUresult syncExternalHandles(
            CUDA_EXTERNAL_MEMORY_HANDLE_DESC* memoryHandleDesc,
            CUDA_EXTERNAL_MEMORY_BUFFER_DESC* bufferDesc,
            CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipDesc,
            CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* semaphoreHandleDesc,
            CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* signalParams,
            CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* waitParams,
            CUstream stream
        ) {
            CUexternalMemory memory;
            CUexternalSemaphore semaphore;
            CUmipmappedArray mipmapped;
            void* ptr;

            CUresult status = cuImportExternalMemory(&memory, memoryHandleDesc);
            status = cuExternalMemoryGetMappedBuffer(&ptr, memory, bufferDesc);
            status = cuExternalMemoryGetMappedMipmappedArray(
                &mipmapped,
                memory,
                mipDesc
            );
            status = cuImportExternalSemaphore(
                &semaphore,
                semaphoreHandleDesc
            );
            status = cuSignalExternalSemaphoresAsync(
                &semaphore,
                signalParams,
                1,
                stream
            );
            cuWaitExternalSemaphoresAsync(&semaphore, waitParams, 1, stream);

            if (
                cuSignalExternalSemaphoresAsync(
                    &semaphore,
                    signalParams,
                    1,
                    stream
                ) != CUDA_SUCCESS
            ) {
                return CUDA_ERROR_UNKNOWN;
            }

            bool ok = checkStatus(
                cuWaitExternalSemaphoresAsync(&semaphore, waitParams, 1, stream)
            );
            cuDestroyExternalSemaphore(semaphore);
            status = cuDestroyExternalMemory(memory);
            return ok ? cuDestroyExternalSemaphore(semaphore) : status;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver external memory import: output: memory, "
            "handle: memoryHandleDesc"
        ) in result
        assert (
            "// CUDA driver external memory mapped buffer: memory, "
            "desc: bufferDesc, output: ptr"
        ) in result
        assert (
            "// CUDA driver external memory mapped mipmapped array: memory, "
            "desc: mipDesc, output: mipmapped"
        ) in result
        assert (
            "// CUDA driver external semaphore import: output: semaphore, "
            "handle: semaphoreHandleDesc"
        ) in result
        assert (
            "// CUDA driver external semaphore signal: semaphores: (&semaphore), "
            "params: signalParams, count: 1, stream: stream"
        ) in result
        assert (
            "// CUDA driver external semaphore wait: semaphores: (&semaphore), "
            "params: waitParams, count: 1, stream: stream"
        ) in result
        assert "// CUDA driver external memory destroy: memory" in result
        assert "// CUDA driver external semaphore destroy: semaphore" in result
        assert (
            "if (((/* CUDA driver external semaphore signal: "
            "semaphores: (&semaphore), params: signalParams, count: 1, "
            "stream: stream */ CUDA_SUCCESS) != CUDA_SUCCESS))"
        ) in result
        assert (
            "checkStatus((/* CUDA driver external semaphore wait: "
            "semaphores: (&semaphore), params: waitParams, count: 1, "
            "stream: stream */ CUDA_SUCCESS))"
        ) in result
        assert (
            "(ok ? (/* CUDA driver external semaphore destroy: semaphore */ "
            "CUDA_SUCCESS) : status)"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuImportExternalMemory",
            "cuExternalMemoryGetMappedBuffer",
            "cuExternalMemoryGetMappedMipmappedArray",
            "cuDestroyExternalMemory",
            "cuImportExternalSemaphore",
            "cuSignalExternalSemaphoresAsync",
            "cuWaitExternalSemaphoresAsync",
            "cuDestroyExternalSemaphore",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_array_mipmapped_array_runtime_conversion(self):
        code = """
        bool checkStatus(CUresult status) {
            return status == CUDA_SUCCESS;
        }

        CUresult manageArrays(
            CUDA_ARRAY_DESCRIPTOR* desc,
            CUDA_ARRAY3D_DESCRIPTOR* desc3d,
            CUDA_ARRAY_MEMORY_REQUIREMENTS* memoryRequirements,
            CUDA_ARRAY_SPARSE_PROPERTIES* sparseProperties,
            unsigned int levels,
            unsigned int levelIndex,
            unsigned int planeIndex,
            CUdevice device
        ) {
            CUarray array;
            CUarray array3d;
            CUarray levelArray;
            CUarray planeArray;
            CUmipmappedArray mipmapped;
            CUDA_ARRAY_DESCRIPTOR outputDesc;
            CUDA_ARRAY3D_DESCRIPTOR outputDesc3d;

            CUresult status = cuArrayCreate(&array, desc);
            status = cuArrayCreate_v2(&array, desc);
            status = cuArray3DCreate(&array3d, desc3d);
            status = cuArray3DCreate_v2(&array3d, desc3d);
            status = cuMipmappedArrayCreate(&mipmapped, desc3d, levels);
            status = cuMipmappedArrayGetLevel(
                &levelArray,
                mipmapped,
                levelIndex
            );
            status = cuArrayGetDescriptor(&outputDesc, array);
            status = cuArrayGetDescriptor_v2(&outputDesc, array);
            status = cuArray3DGetDescriptor(&outputDesc3d, array3d);
            status = cuArray3DGetDescriptor_v2(&outputDesc3d, array3d);
            status = cuArrayGetMemoryRequirements(
                memoryRequirements,
                array,
                device
            );
            status = cuMipmappedArrayGetMemoryRequirements(
                memoryRequirements,
                mipmapped,
                device
            );
            status = cuArrayGetPlane(&planeArray, array, planeIndex);
            status = cuArrayGetSparseProperties(sparseProperties, array);
            status = cuMipmappedArrayGetSparseProperties(
                sparseProperties,
                mipmapped
            );
            cuArrayDestroy(planeArray);

            if (cuArrayDestroy(array) != CUDA_SUCCESS) {
                return CUDA_ERROR_UNKNOWN;
            }

            bool ok = checkStatus(cuMipmappedArrayDestroy(mipmapped));
            return ok ? cuArrayDestroy(array3d) : status;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// CUDA driver array create: output: array, desc: desc" in result
        assert (
            "// CUDA driver array 3D create: output: array3d, desc: desc3d"
        ) in result
        assert (
            "// CUDA driver mipmapped array create: output: mipmapped, "
            "desc: desc3d, levels: levels"
        ) in result
        assert (
            "// CUDA driver mipmapped array get level: output: levelArray, "
            "mipmapped array: mipmapped, level: levelIndex"
        ) in result
        assert (
            "// CUDA driver array get descriptor: output: outputDesc, array: array"
        ) in result
        assert (
            "// CUDA driver array 3D get descriptor: output: outputDesc3d, "
            "array: array3d"
        ) in result
        assert (
            "// CUDA driver array get memory requirements: "
            "output: memoryRequirements, array: array, device: device"
        ) in result
        assert (
            "// CUDA driver mipmapped array get memory requirements: "
            "output: memoryRequirements, mipmapped array: mipmapped, device: device"
        ) in result
        assert (
            "// CUDA driver array get plane: output: planeArray, array: array, "
            "plane: planeIndex"
        ) in result
        assert (
            "// CUDA driver array get sparse properties: "
            "output: sparseProperties, array: array"
        ) in result
        assert (
            "// CUDA driver mipmapped array get sparse properties: "
            "output: sparseProperties, mipmapped array: mipmapped"
        ) in result
        assert "// CUDA driver array destroy: array: planeArray" in result
        assert (
            "if (((/* CUDA driver array destroy: array: array */ CUDA_SUCCESS) "
            "!= CUDA_SUCCESS))"
        ) in result
        assert (
            "checkStatus((/* CUDA driver mipmapped array destroy: "
            "mipmapped array: mipmapped */ CUDA_SUCCESS))"
        ) in result
        assert (
            "(ok ? (/* CUDA driver array destroy: array: array3d */ "
            "CUDA_SUCCESS) : status)"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuArrayCreate",
            "cuArrayCreate_v2",
            "cuArray3DCreate",
            "cuArray3DCreate_v2",
            "cuArrayDestroy",
            "cuMipmappedArrayCreate",
            "cuMipmappedArrayGetLevel",
            "cuMipmappedArrayDestroy",
            "cuArrayGetDescriptor",
            "cuArrayGetDescriptor_v2",
            "cuArray3DGetDescriptor",
            "cuArray3DGetDescriptor_v2",
            "cuArrayGetMemoryRequirements",
            "cuMipmappedArrayGetMemoryRequirements",
            "cuArrayGetPlane",
            "cuArrayGetSparseProperties",
            "cuMipmappedArrayGetSparseProperties",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_texture_surface_object_runtime_conversion(self):
        code = """
        bool checkStatus(CUresult status) {
            return status == CUDA_SUCCESS;
        }

        CUresult manageTextureSurfaceObjects(
            CUDA_RESOURCE_DESC* resourceDesc,
            CUDA_TEXTURE_DESC* textureDesc,
            CUDA_RESOURCE_VIEW_DESC* viewDesc,
            CUtexObject textureObject,
            CUsurfObject surfaceObject
        ) {
            CUtexObject createdTexture;
            CUsurfObject createdSurface;
            CUDA_RESOURCE_DESC outputResourceDesc;
            CUDA_TEXTURE_DESC outputTextureDesc;
            CUDA_RESOURCE_VIEW_DESC outputViewDesc;

            CUresult status = cuTexObjectCreate(
                &createdTexture,
                resourceDesc,
                textureDesc,
                viewDesc
            );
            status = cuTexObjectGetResourceDesc(
                &outputResourceDesc,
                createdTexture
            );
            status = cuTexObjectGetTextureDesc(
                &outputTextureDesc,
                createdTexture
            );
            status = cuTexObjectGetResourceViewDesc(
                &outputViewDesc,
                createdTexture
            );
            status = cuSurfObjectCreate(&createdSurface, resourceDesc);
            status = cuSurfObjectGetResourceDesc(
                &outputResourceDesc,
                createdSurface
            );
            cuTexObjectDestroy(textureObject);

            if (cuTexObjectDestroy(createdTexture) != CUDA_SUCCESS) {
                return CUDA_ERROR_UNKNOWN;
            }

            bool ok = checkStatus(cuSurfObjectDestroy(createdSurface));
            return ok ? cuSurfObjectDestroy(surfaceObject) : status;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver texture object create: output: createdTexture, "
            "resource desc: resourceDesc, texture desc: textureDesc, "
            "resource view desc: viewDesc"
        ) in result
        assert (
            "// CUDA driver texture object get resource desc: "
            "output: outputResourceDesc, texture object: createdTexture"
        ) in result
        assert (
            "// CUDA driver texture object get texture desc: "
            "output: outputTextureDesc, texture object: createdTexture"
        ) in result
        assert (
            "// CUDA driver texture object get resource view desc: "
            "output: outputViewDesc, texture object: createdTexture"
        ) in result
        assert (
            "// CUDA driver surface object create: output: createdSurface, "
            "resource desc: resourceDesc"
        ) in result
        assert (
            "// CUDA driver surface object get resource desc: "
            "output: outputResourceDesc, surface object: createdSurface"
        ) in result
        assert (
            "// CUDA driver texture object destroy: texture object: textureObject"
        ) in result
        assert (
            "if (((/* CUDA driver texture object destroy: "
            "texture object: createdTexture */ CUDA_SUCCESS) != CUDA_SUCCESS))"
        ) in result
        assert (
            "checkStatus((/* CUDA driver surface object destroy: "
            "surface object: createdSurface */ CUDA_SUCCESS))"
        ) in result
        assert (
            "(ok ? (/* CUDA driver surface object destroy: "
            "surface object: surfaceObject */ CUDA_SUCCESS) : status)"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuTexObjectCreate",
            "cuTexObjectDestroy",
            "cuTexObjectGetResourceDesc",
            "cuTexObjectGetTextureDesc",
            "cuTexObjectGetResourceViewDesc",
            "cuSurfObjectCreate",
            "cuSurfObjectDestroy",
            "cuSurfObjectGetResourceDesc",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_texture_reference_runtime_conversion(self):
        code = """
        bool checkStatus(CUresult status) {
            return status == CUDA_SUCCESS;
        }

        CUresult manageTextureReference(
            CUtexref textureRef,
            CUarray array,
            CUmipmappedArray mipmapped,
            CUdeviceptr pointer,
            CUDA_ARRAY_DESCRIPTOR* desc,
            float* borderColor,
            size_t bytes,
            size_t pitch,
            unsigned int flags,
            int dimension,
            CUarray_format format,
            int components,
            CUaddress_mode addressMode,
            CUfilter_mode filterMode,
            float bias,
            float minLevel,
            float maxLevel,
            unsigned int anisotropy
        ) {
            size_t byteOffset;
            CUdeviceptr outputPointer;
            CUarray outputArray;
            CUmipmappedArray outputMipmapped;
            CUaddress_mode outputAddressMode;
            CUfilter_mode outputFilterMode;
            CUarray_format outputFormat;
            int outputChannels;
            float outputBias;
            float outputMinLevel;
            float outputMaxLevel;
            int outputAnisotropy;
            unsigned int outputFlags;

            CUresult status = cuTexRefSetArray(textureRef, array, flags);
            status = cuTexRefSetMipmappedArray(textureRef, mipmapped, flags);
            status = cuTexRefSetAddress(
                &byteOffset,
                textureRef,
                pointer,
                bytes
            );
            status = cuTexRefSetAddress_v2(
                &byteOffset,
                textureRef,
                pointer,
                bytes
            );
            status = cuTexRefSetAddress2D(textureRef, desc, pointer, pitch);
            status = cuTexRefSetAddress2D_v2(textureRef, desc, pointer, pitch);
            status = cuTexRefSetFormat(textureRef, format, components);
            status = cuTexRefSetAddressMode(textureRef, dimension, addressMode);
            status = cuTexRefSetFilterMode(textureRef, filterMode);
            status = cuTexRefSetMipmapFilterMode(textureRef, filterMode);
            status = cuTexRefSetMipmapLevelBias(textureRef, bias);
            status = cuTexRefSetMipmapLevelClamp(
                textureRef,
                minLevel,
                maxLevel
            );
            status = cuTexRefSetMaxAnisotropy(textureRef, anisotropy);
            status = cuTexRefSetBorderColor(textureRef, borderColor);
            status = cuTexRefSetFlags(textureRef, flags);
            status = cuTexRefGetAddress(&outputPointer, textureRef);
            status = cuTexRefGetAddress_v2(&outputPointer, textureRef);
            status = cuTexRefGetArray(&outputArray, textureRef);
            status = cuTexRefGetMipmappedArray(&outputMipmapped, textureRef);
            status = cuTexRefGetAddressMode(
                &outputAddressMode,
                textureRef,
                dimension
            );
            status = cuTexRefGetFilterMode(&outputFilterMode, textureRef);
            status = cuTexRefGetFormat(
                &outputFormat,
                &outputChannels,
                textureRef
            );
            status = cuTexRefGetMipmapFilterMode(
                &outputFilterMode,
                textureRef
            );
            status = cuTexRefGetMipmapLevelBias(&outputBias, textureRef);
            status = cuTexRefGetMipmapLevelClamp(
                &outputMinLevel,
                &outputMaxLevel,
                textureRef
            );
            status = cuTexRefGetMaxAnisotropy(&outputAnisotropy, textureRef);
            status = cuTexRefGetBorderColor(borderColor, textureRef);
            cuTexRefGetFlags(&outputFlags, textureRef);

            if (cuTexRefSetFlags(textureRef, flags) != CUDA_SUCCESS) {
                return CUDA_ERROR_UNKNOWN;
            }

            bool ok = checkStatus(cuTexRefGetFlags(&outputFlags, textureRef));
            return ok ? cuTexRefSetArray(textureRef, array, flags) : status;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver texture reference set array: "
            "texture ref: textureRef, array: array, flags: flags"
        ) in result
        assert (
            "// CUDA driver texture reference set mipmapped array: "
            "texture ref: textureRef, mipmapped array: mipmapped, flags: flags"
        ) in result
        assert (
            "// CUDA driver texture reference set address: "
            "byte offset output: byteOffset, texture ref: textureRef, "
            "pointer: pointer, bytes: bytes"
        ) in result
        assert (
            "// CUDA driver texture reference set address 2D: "
            "texture ref: textureRef, desc: desc, pointer: pointer, pitch: pitch"
        ) in result
        assert (
            "// CUDA driver texture reference set format: "
            "texture ref: textureRef, format: format, components: components"
        ) in result
        assert (
            "// CUDA driver texture reference set address mode: "
            "texture ref: textureRef, dimension: dimension, mode: addressMode"
        ) in result
        assert (
            "// CUDA driver texture reference set filter mode: "
            "texture ref: textureRef, mode: filterMode"
        ) in result
        assert (
            "// CUDA driver texture reference set mipmap filter mode: "
            "texture ref: textureRef, mode: filterMode"
        ) in result
        assert (
            "// CUDA driver texture reference set mipmap level bias: "
            "texture ref: textureRef, bias: bias"
        ) in result
        assert (
            "// CUDA driver texture reference set mipmap level clamp: "
            "texture ref: textureRef, min level: minLevel, max level: maxLevel"
        ) in result
        assert (
            "// CUDA driver texture reference set max anisotropy: "
            "texture ref: textureRef, anisotropy: anisotropy"
        ) in result
        assert (
            "// CUDA driver texture reference set border color: "
            "texture ref: textureRef, color: borderColor"
        ) in result
        assert (
            "// CUDA driver texture reference set flags: "
            "texture ref: textureRef, flags: flags"
        ) in result
        assert (
            "// CUDA driver texture reference get address: "
            "output: outputPointer, texture ref: textureRef"
        ) in result
        assert (
            "// CUDA driver texture reference get array: "
            "output: outputArray, texture ref: textureRef"
        ) in result
        assert (
            "// CUDA driver texture reference get mipmapped array: "
            "output: outputMipmapped, texture ref: textureRef"
        ) in result
        assert (
            "// CUDA driver texture reference get address mode: "
            "output: outputAddressMode, texture ref: textureRef, "
            "dimension: dimension"
        ) in result
        assert (
            "// CUDA driver texture reference get filter mode: "
            "output: outputFilterMode, texture ref: textureRef"
        ) in result
        assert (
            "// CUDA driver texture reference get format: "
            "format output: outputFormat, channel output: outputChannels, "
            "texture ref: textureRef"
        ) in result
        assert (
            "// CUDA driver texture reference get mipmap filter mode: "
            "output: outputFilterMode, texture ref: textureRef"
        ) in result
        assert (
            "// CUDA driver texture reference get mipmap level bias: "
            "output: outputBias, texture ref: textureRef"
        ) in result
        assert (
            "// CUDA driver texture reference get mipmap level clamp: "
            "min output: outputMinLevel, max output: outputMaxLevel, "
            "texture ref: textureRef"
        ) in result
        assert (
            "// CUDA driver texture reference get max anisotropy: "
            "output: outputAnisotropy, texture ref: textureRef"
        ) in result
        assert (
            "// CUDA driver texture reference get border color: "
            "output: borderColor, texture ref: textureRef"
        ) in result
        assert (
            "// CUDA driver texture reference get flags: "
            "output: outputFlags, texture ref: textureRef"
        ) in result
        assert (
            "if (((/* CUDA driver texture reference set flags: "
            "texture ref: textureRef, flags: flags */ CUDA_SUCCESS) "
            "!= CUDA_SUCCESS))"
        ) in result
        assert (
            "checkStatus((/* CUDA driver texture reference get flags: "
            "output: outputFlags, texture ref: textureRef */ CUDA_SUCCESS))"
        ) in result
        assert (
            "(ok ? (/* CUDA driver texture reference set array: "
            "texture ref: textureRef, array: array, flags: flags */ "
            "CUDA_SUCCESS) : status)"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuTexRefSetArray",
            "cuTexRefSetMipmappedArray",
            "cuTexRefSetAddress",
            "cuTexRefSetAddress_v2",
            "cuTexRefSetAddress2D",
            "cuTexRefSetAddress2D_v2",
            "cuTexRefSetFormat",
            "cuTexRefSetAddressMode",
            "cuTexRefSetFilterMode",
            "cuTexRefSetMipmapFilterMode",
            "cuTexRefSetMipmapLevelBias",
            "cuTexRefSetMipmapLevelClamp",
            "cuTexRefSetMaxAnisotropy",
            "cuTexRefSetBorderColor",
            "cuTexRefSetFlags",
            "cuTexRefGetAddress",
            "cuTexRefGetAddress_v2",
            "cuTexRefGetArray",
            "cuTexRefGetMipmappedArray",
            "cuTexRefGetAddressMode",
            "cuTexRefGetFilterMode",
            "cuTexRefGetFormat",
            "cuTexRefGetMipmapFilterMode",
            "cuTexRefGetMipmapLevelBias",
            "cuTexRefGetMipmapLevelClamp",
            "cuTexRefGetMaxAnisotropy",
            "cuTexRefGetBorderColor",
            "cuTexRefGetFlags",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_module_surface_reference_runtime_conversion(self):
        code = """
        bool checkStatus(CUresult status) {
            return status == CUDA_SUCCESS;
        }

        CUresult manageModuleReferences(
            CUmodule module,
            const char* textureName,
            const char* surfaceName,
            CUarray array,
            unsigned int flags
        ) {
            CUtexref textureRef;
            CUsurfref surfaceRef;
            CUarray outputArray;

            CUresult status = cuModuleGetTexRef(
                &textureRef,
                module,
                textureName
            );
            status = cuModuleGetSurfRef(&surfaceRef, module, surfaceName);
            status = cuSurfRefSetArray(surfaceRef, array, flags);
            status = cuSurfRefGetArray(&outputArray, surfaceRef);
            cuSurfRefSetArray(surfaceRef, array, flags);

            if (cuSurfRefGetArray(&outputArray, surfaceRef) != CUDA_SUCCESS) {
                return CUDA_ERROR_UNKNOWN;
            }

            bool ok = checkStatus(cuSurfRefSetArray(surfaceRef, array, flags));
            return ok ? cuModuleGetTexRef(&textureRef, module, textureName)
                      : status;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver module get texture reference: "
            "output: textureRef, module: module, name: textureName"
        ) in result
        assert (
            "// CUDA driver module get surface reference: "
            "output: surfaceRef, module: module, name: surfaceName"
        ) in result
        assert (
            "// CUDA driver surface reference set array: "
            "surface ref: surfaceRef, array: array, flags: flags"
        ) in result
        assert (
            "// CUDA driver surface reference get array: "
            "output: outputArray, surface ref: surfaceRef"
        ) in result
        assert (
            "if (((/* CUDA driver surface reference get array: "
            "output: outputArray, surface ref: surfaceRef */ CUDA_SUCCESS) "
            "!= CUDA_SUCCESS))"
        ) in result
        assert (
            "checkStatus((/* CUDA driver surface reference set array: "
            "surface ref: surfaceRef, array: array, flags: flags */ "
            "CUDA_SUCCESS))"
        ) in result
        assert (
            "(ok ? (/* CUDA driver module get texture reference: "
            "output: textureRef, module: module, name: textureName */ "
            "CUDA_SUCCESS) : status)"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuModuleGetTexRef",
            "cuModuleGetSurfRef",
            "cuSurfRefSetArray",
            "cuSurfRefGetArray",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_graphics_resource_runtime_conversion(self):
        code = """
        bool checkStatus(CUresult status) {
            return status == CUDA_SUCCESS;
        }

        CUresult manageGraphicsResources(
            unsigned int glBuffer,
            unsigned int glImage,
            unsigned int glTarget,
            void* d3dResource,
            CUgraphicsResource* resources,
            unsigned int flags,
            CUstream stream
        ) {
            CUgraphicsResource bufferResource;
            CUgraphicsResource imageResource;
            CUgraphicsResource d3dInteropResource;
            CUgraphicsResource d3d10InteropResource;
            CUgraphicsResource d3d9InteropResource;
            CUdeviceptr mappedPointer;
            size_t mappedBytes;
            CUarray mappedArray;
            CUmipmappedArray mappedMipmappedArray;

            CUresult status = cuGraphicsGLRegisterBuffer(
                &bufferResource,
                glBuffer,
                flags
            );
            status = cuGraphicsGLRegisterImage(
                &imageResource,
                glImage,
                glTarget,
                flags
            );
            status = cuGraphicsD3D11RegisterResource(
                &d3dInteropResource,
                d3dResource,
                flags
            );
            status = cuGraphicsD3D10RegisterResource(
                &d3d10InteropResource,
                d3dResource,
                flags
            );
            status = cuGraphicsD3D9RegisterResource(
                &d3d9InteropResource,
                d3dResource,
                flags
            );
            status = cuGraphicsResourceSetMapFlags(bufferResource, flags);
            status = cuGraphicsResourceSetMapFlags_v2(imageResource, flags);
            status = cuGraphicsMapResources(2, resources, stream);
            status = cuGraphicsResourceGetMappedPointer(
                &mappedPointer,
                &mappedBytes,
                bufferResource
            );
            status = cuGraphicsResourceGetMappedPointer_v2(
                &mappedPointer,
                &mappedBytes,
                imageResource
            );
            status = cuGraphicsSubResourceGetMappedArray(
                &mappedArray,
                imageResource,
                0,
                1
            );
            status = cuGraphicsResourceGetMappedMipmappedArray(
                &mappedMipmappedArray,
                imageResource
            );
            status = cuGraphicsUnmapResources(2, resources, stream);
            status = cuGraphicsUnregisterResource(bufferResource);
            cuGraphicsUnregisterResource(imageResource);

            if (cuGraphicsMapResources(2, resources, stream) != CUDA_SUCCESS) {
                return CUDA_ERROR_UNKNOWN;
            }

            bool ok = checkStatus(
                cuGraphicsResourceSetMapFlags(bufferResource, flags)
            );
            return ok ? cuGraphicsUnmapResources(2, resources, stream)
                      : status;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver graphics GL register buffer: "
            "output: bufferResource, buffer: glBuffer, flags: flags"
        ) in result
        assert (
            "// CUDA driver graphics GL register image: "
            "output: imageResource, image: glImage, target: glTarget, "
            "flags: flags"
        ) in result
        assert (
            "// CUDA driver graphics D3D11 register resource: "
            "output: d3dInteropResource, resource: d3dResource, flags: flags"
        ) in result
        assert (
            "// CUDA driver graphics D3D10 register resource: "
            "output: d3d10InteropResource, resource: d3dResource, flags: flags"
        ) in result
        assert (
            "// CUDA driver graphics D3D9 register resource: "
            "output: d3d9InteropResource, resource: d3dResource, flags: flags"
        ) in result
        assert (
            "// CUDA driver graphics resource set map flags: "
            "resource: bufferResource, flags: flags"
        ) in result
        assert (
            "// CUDA driver graphics resource set map flags: "
            "resource: imageResource, flags: flags"
        ) in result
        assert (
            "// CUDA driver graphics map resources: "
            "count: 2, resources: resources, stream: stream"
        ) in result
        assert (
            "// CUDA driver graphics mapped pointer: "
            "pointer output: mappedPointer, size output: mappedBytes, "
            "resource: bufferResource"
        ) in result
        assert (
            "// CUDA driver graphics mapped pointer: "
            "pointer output: mappedPointer, size output: mappedBytes, "
            "resource: imageResource"
        ) in result
        assert (
            "// CUDA driver graphics subresource mapped array: "
            "output: mappedArray, resource: imageResource, array index: 0, "
            "mip level: 1"
        ) in result
        assert (
            "// CUDA driver graphics mapped mipmapped array: "
            "output: mappedMipmappedArray, resource: imageResource"
        ) in result
        assert (
            "// CUDA driver graphics unmap resources: "
            "count: 2, resources: resources, stream: stream"
        ) in result
        assert (
            "// CUDA driver graphics unregister resource: " "resource: bufferResource"
        ) in result
        assert (
            "if (((/* CUDA driver graphics map resources: "
            "count: 2, resources: resources, stream: stream */ "
            "CUDA_SUCCESS) != CUDA_SUCCESS))"
        ) in result
        assert (
            "checkStatus((/* CUDA driver graphics resource set map flags: "
            "resource: bufferResource, flags: flags */ CUDA_SUCCESS))"
        ) in result
        assert (
            "(ok ? (/* CUDA driver graphics unmap resources: "
            "count: 2, resources: resources, stream: stream */ CUDA_SUCCESS) "
            ": status)"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuGraphicsGLRegisterBuffer",
            "cuGraphicsGLRegisterImage",
            "cuGraphicsD3D11RegisterResource",
            "cuGraphicsD3D10RegisterResource",
            "cuGraphicsD3D9RegisterResource",
            "cuGraphicsResourceSetMapFlags",
            "cuGraphicsResourceSetMapFlags_v2",
            "cuGraphicsMapResources",
            "cuGraphicsResourceGetMappedPointer",
            "cuGraphicsResourceGetMappedPointer_v2",
            "cuGraphicsSubResourceGetMappedArray",
            "cuGraphicsResourceGetMappedMipmappedArray",
            "cuGraphicsUnmapResources",
            "cuGraphicsUnregisterResource",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_legacy_opengl_interop_runtime_conversion(self):
        code = """
        bool checkStatus(CUresult status) {
            return status == CUDA_SUCCESS;
        }

        CUresult manageLegacyOpenGLInterop(
            CUdevice device,
            CUdevice* devices,
            unsigned int maxDevices,
            unsigned int deviceList,
            unsigned int bufferObject,
            unsigned int flags,
            CUstream stream
        ) {
            CUcontext context;
            unsigned int deviceCount;
            CUdeviceptr mappedPointer;
            size_t mappedBytes;

            CUresult status = cuGLInit();
            status = cuGLCtxCreate(&context, flags, device);
            status = cuGLCtxCreate_v2(&context, flags, device);
            status = cuGLGetDevices(
                &deviceCount,
                devices,
                maxDevices,
                deviceList
            );
            status = cuGLRegisterBufferObject(bufferObject);
            status = cuGLSetBufferObjectMapFlags(bufferObject, flags);
            status = cuGLMapBufferObject(
                &mappedPointer,
                &mappedBytes,
                bufferObject
            );
            status = cuGLMapBufferObject_v2(
                &mappedPointer,
                &mappedBytes,
                bufferObject
            );
            status = cuGLMapBufferObjectAsync(
                &mappedPointer,
                &mappedBytes,
                bufferObject,
                stream
            );
            status = cuGLMapBufferObjectAsync_v2(
                &mappedPointer,
                &mappedBytes,
                bufferObject,
                stream
            );
            status = cuGLUnmapBufferObject(bufferObject);
            status = cuGLUnmapBufferObjectAsync(bufferObject, stream);
            status = cuGLUnregisterBufferObject(bufferObject);
            cuGLUnregisterBufferObject(bufferObject);

            if (cuGLMapBufferObject(&mappedPointer, &mappedBytes, bufferObject)
                != CUDA_SUCCESS) {
                return CUDA_ERROR_UNKNOWN;
            }

            bool ok = checkStatus(cuGLSetBufferObjectMapFlags(
                bufferObject,
                flags
            ));
            return ok ? cuGLUnmapBufferObjectAsync(bufferObject, stream)
                      : status;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// CUDA driver OpenGL initialize" in result
        assert (
            "// CUDA driver OpenGL context create: "
            "output: context, flags: flags, device: device"
        ) in result
        assert (
            "// CUDA driver OpenGL get devices: count output: deviceCount, "
            "devices: devices, max devices: maxDevices, device list: deviceList"
        ) in result
        assert (
            "// CUDA driver OpenGL register buffer object: "
            "buffer object: bufferObject"
        ) in result
        assert (
            "// CUDA driver OpenGL set buffer object map flags: "
            "buffer object: bufferObject, flags: flags"
        ) in result
        assert (
            "// CUDA driver OpenGL map buffer object: "
            "pointer output: mappedPointer, size output: mappedBytes, "
            "buffer object: bufferObject"
        ) in result
        assert (
            "// CUDA driver OpenGL map buffer object async: "
            "pointer output: mappedPointer, size output: mappedBytes, "
            "buffer object: bufferObject, stream: stream"
        ) in result
        assert (
            "// CUDA driver OpenGL unmap buffer object: " "buffer object: bufferObject"
        ) in result
        assert (
            "// CUDA driver OpenGL unmap buffer object async: "
            "buffer object: bufferObject, stream: stream"
        ) in result
        assert (
            "// CUDA driver OpenGL unregister buffer object: "
            "buffer object: bufferObject"
        ) in result
        assert (
            "if (((/* CUDA driver OpenGL map buffer object: "
            "pointer output: mappedPointer, size output: mappedBytes, "
            "buffer object: bufferObject */ CUDA_SUCCESS) != CUDA_SUCCESS))"
        ) in result
        assert (
            "checkStatus((/* CUDA driver OpenGL set buffer object map flags: "
            "buffer object: bufferObject, flags: flags */ CUDA_SUCCESS))"
        ) in result
        assert (
            "(ok ? (/* CUDA driver OpenGL unmap buffer object async: "
            "buffer object: bufferObject, stream: stream */ CUDA_SUCCESS) "
            ": status)"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuGLInit",
            "cuGLCtxCreate",
            "cuGLCtxCreate_v2",
            "cuGLGetDevices",
            "cuGLRegisterBufferObject",
            "cuGLSetBufferObjectMapFlags",
            "cuGLMapBufferObject",
            "cuGLMapBufferObject_v2",
            "cuGLMapBufferObjectAsync",
            "cuGLMapBufferObjectAsync_v2",
            "cuGLUnmapBufferObject",
            "cuGLUnmapBufferObjectAsync",
            "cuGLUnregisterBufferObject",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_module_launch_runtime_conversion(self):
        code = """
        CUresult driverLaunch(
            const char* path,
            void* image,
            CUstream stream,
            void** kernelParams,
            void** extra,
            unsigned int optionCount,
            CUjit_option* options,
            void** optionValues,
            unsigned int gridX,
            unsigned int gridY,
            unsigned int gridZ,
            unsigned int blockX,
            unsigned int blockY,
            unsigned int blockZ,
            unsigned int sharedBytes,
            int blockSize
        ) {
            CUmodule module;
            CUfunction function;
            CUdeviceptr global;
            size_t globalBytes;
            int activeBlocks;
            int minGrid;
            int suggestedBlock;
            CUDA_LAUNCH_PARAMS launchParams;

            CUresult status = cuModuleLoad(&module, path);
            status = cuModuleLoadData(&module, image);
            status = cuModuleLoadDataEx(
                &module,
                image,
                optionCount,
                options,
                optionValues
            );
            status = cuModuleGetFunction(&function, module, "kernel");
            status = cuModuleGetGlobal(&global, &globalBytes, module, "symbol");
            status = cuLaunchKernel(
                function,
                gridX,
                gridY,
                gridZ,
                blockX,
                blockY,
                blockZ,
                sharedBytes,
                stream,
                kernelParams,
                extra
            );
            status = cuLaunchCooperativeKernel(
                function,
                gridX,
                gridY,
                gridZ,
                blockX,
                blockY,
                blockZ,
                sharedBytes,
                stream,
                kernelParams
            );
            status = cuLaunchCooperativeKernelMultiDevice(&launchParams, 1, 0);
            status = cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &activeBlocks,
                function,
                blockSize,
                sharedBytes
            );
            status = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                &activeBlocks,
                function,
                blockSize,
                sharedBytes,
                CU_OCCUPANCY_DEFAULT
            );
            status = cuOccupancyMaxPotentialBlockSize(
                &minGrid,
                &suggestedBlock,
                function,
                0,
                sharedBytes,
                blockSize
            );
            status = cuOccupancyMaxPotentialBlockSizeWithFlags(
                &minGrid,
                &suggestedBlock,
                function,
                0,
                sharedBytes,
                blockSize,
                CU_OCCUPANCY_DEFAULT
            );
            cuModuleUnload(module);

            return cuModuleUnload(module);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// CUDA driver module load: output: module, path: path" in result
        assert "// CUDA driver module load data: output: module, image: image" in result
        assert (
            "// CUDA driver module load data with options: output: module, "
            "image: image, option count: optionCount, options: options, "
            "option values: optionValues"
        ) in result
        assert (
            "// CUDA driver module get function: output: function, "
            'module: module, name: "kernel"'
        ) in result
        assert (
            "// CUDA driver module get global: output: global, "
            'bytes output: globalBytes, module: module, name: "symbol"'
        ) in result
        assert (
            "// CUDA driver launch kernel: function: function, "
            "grid: gridX x gridY x gridZ, block: blockX x blockY x blockZ, "
            "shared memory: sharedBytes, stream: stream, params: kernelParams, "
            "extra: extra"
        ) in result
        assert (
            "// CUDA driver launch cooperative kernel: function: function, "
            "grid: gridX x gridY x gridZ, block: blockX x blockY x blockZ, "
            "shared memory: sharedBytes, stream: stream, params: kernelParams"
        ) in result
        assert (
            "// CUDA driver launch cooperative kernel multi-device: "
            "params: launchParams, count: 1, flags: 0"
        ) in result
        assert (
            "// CUDA driver occupancy active blocks: output: activeBlocks, "
            "function: function, block size: blockSize, "
            "dynamic shared memory: sharedBytes"
        ) in result
        assert (
            "// CUDA driver occupancy active blocks: output: activeBlocks, "
            "function: function, block size: blockSize, "
            "dynamic shared memory: sharedBytes, flags: CU_OCCUPANCY_DEFAULT"
        ) in result
        assert (
            "// CUDA driver occupancy potential block size: min grid output: minGrid, "
            "block size output: suggestedBlock, function: function, "
            "dynamic shared memory callback: 0, dynamic shared memory: sharedBytes, "
            "block size limit: blockSize, flags: 0"
        ) in result
        assert (
            "// CUDA driver occupancy potential block size: min grid output: minGrid, "
            "block size output: suggestedBlock, function: function, "
            "dynamic shared memory callback: 0, dynamic shared memory: sharedBytes, "
            "block size limit: blockSize, flags: CU_OCCUPANCY_DEFAULT"
        ) in result
        assert "// CUDA driver module unload: module" in result
        assert "return CUDA_SUCCESS;" in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuModuleLoad",
            "cuModuleLoadData",
            "cuModuleLoadDataEx",
            "cuModuleGetFunction",
            "cuModuleGetGlobal",
            "cuLaunchKernel",
            "cuLaunchCooperativeKernel",
            "cuLaunchCooperativeKernelMultiDevice",
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
            "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
            "cuOccupancyMaxPotentialBlockSize",
            "cuOccupancyMaxPotentialBlockSizeWithFlags",
            "cuModuleUnload",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_linker_runtime_conversion(self):
        code = """
        CUresult linkModule(
            const char* path,
            void* image,
            size_t imageBytes,
            unsigned int optionCount,
            CUjit_option* options,
            void** optionValues,
            bool ok
        ) {
            CUlinkState linkState;
            void* cubin;
            size_t cubinSize;

            CUresult status = cuLinkCreate(
                optionCount,
                options,
                optionValues,
                &linkState
            );
            status = cuLinkAddData(
                linkState,
                CU_JIT_INPUT_PTX,
                image,
                imageBytes,
                "kernel.ptx",
                optionCount,
                options,
                optionValues
            );
            status = cuLinkAddFile(
                linkState,
                CU_JIT_INPUT_PTX,
                path,
                optionCount,
                options,
                optionValues
            );
            status = cuLinkComplete(linkState, &cubin, &cubinSize);
            cuLinkDestroy(linkState);

            if (cuLinkAddFile(
                    linkState,
                    CU_JIT_INPUT_CUBIN,
                    path,
                    0,
                    options,
                    optionValues
                ) != CUDA_SUCCESS) {
                return status;
            }

            checkStatus(cuLinkComplete(linkState, &cubin, &cubinSize));
            return ok ? cuLinkDestroy(linkState) : status;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver linker create: output: linkState, "
            "option count: optionCount, options: options, option values: optionValues"
        ) in result
        assert (
            "// CUDA driver linker add data: state: linkState, type: CU_JIT_INPUT_PTX, "
            'data: image, bytes: imageBytes, name: "kernel.ptx", '
            "option count: optionCount, options: options, option values: optionValues"
        ) in result
        assert (
            "// CUDA driver linker add file: state: linkState, type: CU_JIT_INPUT_PTX, "
            "path: path, option count: optionCount, options: options, "
            "option values: optionValues"
        ) in result
        assert (
            "// CUDA driver linker complete: state: linkState, cubin output: cubin, "
            "size output: cubinSize"
        ) in result
        assert "// CUDA driver linker destroy: state: linkState" in result
        assert (
            "/* CUDA driver linker add file: state: linkState, "
            "type: CU_JIT_INPUT_CUBIN, path: path, option count: 0, "
            "options: options, option values: optionValues */ CUDA_SUCCESS) "
            "!= CUDA_SUCCESS"
        ) in result
        assert (
            "checkStatus((/* CUDA driver linker complete: state: linkState, "
            "cubin output: cubin, size output: cubinSize */ CUDA_SUCCESS));" in result
        )
        assert (
            "return (ok ? (/* CUDA driver linker destroy: state: linkState */ "
            "CUDA_SUCCESS) : status);"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuLinkCreate",
            "cuLinkAddData",
            "cuLinkAddFile",
            "cuLinkComplete",
            "cuLinkDestroy",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_function_configuration_runtime_conversion(self):
        code = """
        CUresult configureFunction(CUfunction function) {
            int maxThreads;
            CUresult status = cuFuncGetAttribute(
                &maxThreads,
                CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                function
            );
            status = cuFuncSetAttribute(
                function,
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                49152
            );
            status = cuFuncSetCacheConfig(
                function,
                CU_FUNC_CACHE_PREFER_SHARED
            );
            cuFuncSetCacheConfig(function, CU_FUNC_CACHE_PREFER_L1);
            status = cuFuncSetSharedMemConfig(
                function,
                CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE
            );

            return cuFuncSetSharedMemConfig(
                function,
                CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE
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
            "// CUDA driver function get attribute: output: maxThreads, "
            "attribute: CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, "
            "function: function"
        ) in result
        assert (
            "// CUDA driver function set attribute: function: function, "
            "attribute: CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, "
            "value: 49152"
        ) in result
        assert (
            "// CUDA driver function set cache config: function: function, "
            "config: CU_FUNC_CACHE_PREFER_SHARED"
        ) in result
        assert (
            "// CUDA driver function set cache config: function: function, "
            "config: CU_FUNC_CACHE_PREFER_L1"
        ) in result
        assert (
            "// CUDA driver function set shared memory config: function: function, "
            "config: CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE"
        ) in result
        assert (
            "// CUDA driver function set shared memory config: function: function, "
            "config: CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE"
        ) in result
        assert "return CUDA_SUCCESS;" in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuFuncGetAttribute",
            "cuFuncSetAttribute",
            "cuFuncSetCacheConfig",
            "cuFuncSetSharedMemConfig",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_context_lifecycle_runtime_conversion(self):
        code = """
        CUresult manageContext(CUdevice device, CUcontext peer) {
            CUcontext context;
            CUcontext current;
            CUdevice activeDevice;
            unsigned int flags;
            unsigned int primaryFlags;
            int active;
            unsigned long long contextId;
            size_t limitBytes;
            CUfunc_cache cacheConfig;
            CUsharedconfig sharedConfig;

            CUresult status = cuCtxCreate(
                &context,
                CU_CTX_SCHED_AUTO,
                device
            );
            status = cuCtxCreate_v2(
                &current,
                CU_CTX_SCHED_BLOCKING_SYNC,
                device
            );
            status = cuCtxSetCurrent(context);
            status = cuCtxGetCurrent(&current);
            status = cuCtxPushCurrent(context);
            status = cuCtxPopCurrent(&current);
            status = cuCtxGetDevice(&activeDevice);
            status = cuCtxGetFlags(&flags);
            status = cuCtxGetId(context, &contextId);
            status = cuCtxGetLimit(&limitBytes, CU_LIMIT_STACK_SIZE);
            status = cuCtxSetLimit(CU_LIMIT_STACK_SIZE, limitBytes);
            status = cuCtxGetCacheConfig(&cacheConfig);
            status = cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1);
            status = cuCtxGetSharedMemConfig(&sharedConfig);
            status = cuCtxSetSharedMemConfig(
                CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE
            );
            status = cuCtxEnablePeerAccess(peer, 0);
            status = cuCtxDisablePeerAccess(peer);
            status = cuCtxSynchronize();
            status = cuDevicePrimaryCtxRetain(&context, device);
            status = cuDevicePrimaryCtxGetState(
                device,
                &primaryFlags,
                &active
            );
            status = cuDevicePrimaryCtxSetFlags(device, CU_CTX_SCHED_SPIN);
            status = cuDevicePrimaryCtxSetFlags_v2(
                device,
                CU_CTX_SCHED_BLOCKING_SYNC
            );
            status = cuDevicePrimaryCtxRelease(device);
            cuDevicePrimaryCtxReset_v2(device);
            cuCtxDestroy(context);
            status = cuCtxDestroy_v2(current);

            return cuDevicePrimaryCtxRelease_v2(device);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver context create: output: context, "
            "flags: CU_CTX_SCHED_AUTO, device: device"
        ) in result
        assert (
            "// CUDA driver context create: output: current, "
            "flags: CU_CTX_SCHED_BLOCKING_SYNC, device: device"
        ) in result
        assert "// CUDA driver context set current: context" in result
        assert "// CUDA driver context get current: output: current" in result
        assert "// CUDA driver context push current: context" in result
        assert "// CUDA driver context pop current: output: current" in result
        assert "// CUDA driver context get device: output: activeDevice" in result
        assert "// CUDA driver context get flags: output: flags" in result
        assert (
            "// CUDA driver context get id: context: context, " "output: contextId"
        ) in result
        assert (
            "// CUDA driver context get limit: output: limitBytes, "
            "limit: CU_LIMIT_STACK_SIZE"
        ) in result
        assert (
            "// CUDA driver context set limit: limit: CU_LIMIT_STACK_SIZE, "
            "value: limitBytes"
        ) in result
        assert (
            "// CUDA driver context get cache config: output: cacheConfig"
        ) in result
        assert (
            "// CUDA driver context set cache config: "
            "config: CU_FUNC_CACHE_PREFER_L1"
        ) in result
        assert (
            "// CUDA driver context get shared memory config: " "output: sharedConfig"
        ) in result
        assert (
            "// CUDA driver context set shared memory config: "
            "config: CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE"
        ) in result
        assert (
            "// CUDA driver context enable peer access: peer: peer, flags: 0"
        ) in result
        assert "// CUDA driver context disable peer access: peer: peer" in result
        assert "// CUDA driver context synchronize" in result
        assert (
            "// CUDA driver device primary context retain: output: context, "
            "device: device"
        ) in result
        assert (
            "// CUDA driver device primary context get state: device: device, "
            "flags output: primaryFlags, active output: active"
        ) in result
        assert (
            "// CUDA driver device primary context set flags: device: device, "
            "flags: CU_CTX_SCHED_SPIN"
        ) in result
        assert (
            "// CUDA driver device primary context set flags: device: device, "
            "flags: CU_CTX_SCHED_BLOCKING_SYNC"
        ) in result
        assert "// CUDA driver device primary context release: device: device" in result
        assert "// CUDA driver device primary context reset: device: device" in result
        assert "// CUDA driver context destroy: context" in result
        assert "// CUDA driver context destroy: current" in result
        assert "return CUDA_SUCCESS;" in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuCtxCreate",
            "cuCtxCreate_v2",
            "cuCtxSetCurrent",
            "cuCtxGetCurrent",
            "cuCtxPushCurrent",
            "cuCtxPopCurrent",
            "cuCtxGetDevice",
            "cuCtxGetFlags",
            "cuCtxGetId",
            "cuCtxGetLimit",
            "cuCtxSetLimit",
            "cuCtxGetCacheConfig",
            "cuCtxSetCacheConfig",
            "cuCtxGetSharedMemConfig",
            "cuCtxSetSharedMemConfig",
            "cuCtxEnablePeerAccess",
            "cuCtxDisablePeerAccess",
            "cuCtxSynchronize",
            "cuDevicePrimaryCtxRetain",
            "cuDevicePrimaryCtxGetState",
            "cuDevicePrimaryCtxSetFlags",
            "cuDevicePrimaryCtxSetFlags_v2",
            "cuDevicePrimaryCtxRelease",
            "cuDevicePrimaryCtxRelease_v2",
            "cuDevicePrimaryCtxReset_v2",
            "cuCtxDestroy",
            "cuCtxDestroy_v2",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_device_inventory_runtime_conversion(self):
        code = """
        bool checkStatus(CUresult status) {
            return status == CUDA_SUCCESS;
        }

        CUresult inspectDevices(int ordinal) {
            CUdevice device;
            CUdevice alternate;
            int driverVersion;
            int count;
            char name[256];
            CUuuid uuid;
            size_t bytes;
            size_t bytesV2;
            int major;
            int minor;
            int multiprocessors;

            CUresult status = cuInit(0);
            status = cuDriverGetVersion(&driverVersion);
            status = cuDeviceGet(&device, ordinal);
            status = cuDeviceGetCount(&count);
            cuDeviceGet(&alternate, 0);
            status = cuDeviceGetName(name, 256, device);
            status = cuDeviceGetUuid(&uuid, device);
            status = cuDeviceTotalMem(&bytes, device);
            status = cuDeviceTotalMem_v2(&bytesV2, device);
            status = cuDeviceGetAttribute(
                &multiprocessors,
                CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                device
            );
            status = cuDeviceComputeCapability(&major, &minor, device);

            if (cuDeviceGetCount(&count) != CUDA_SUCCESS) {
                return CUDA_ERROR_UNKNOWN;
            }

            bool ok = checkStatus(cuDriverGetVersion(&driverVersion));
            return ok ? cuDeviceGetAttribute(
                &major,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                device
            ) : status;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// CUDA driver initialize: flags: 0" in result
        assert ("// CUDA driver get version: output: driverVersion") in result
        assert ("// CUDA driver get device: output: device, ordinal: ordinal") in result
        assert "// CUDA driver get device count: output: count" in result
        assert ("// CUDA driver get device: output: alternate, ordinal: 0") in result
        assert (
            "// CUDA driver get device name: output: name, "
            "length: 256, device: device"
        ) in result
        assert (
            "// CUDA driver get device UUID: output: uuid, device: device"
        ) in result
        assert (
            "// CUDA driver get device total memory: output: bytes, " "device: device"
        ) in result
        assert (
            "// CUDA driver get device total memory: output: bytesV2, " "device: device"
        ) in result
        assert (
            "// CUDA driver get device attribute: output: multiprocessors, "
            "attribute: CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device: device"
        ) in result
        assert (
            "// CUDA driver get device compute capability: "
            "major output: major, minor output: minor, device: device"
        ) in result
        assert (
            "if (((/* CUDA driver get device count: output: count */ "
            "CUDA_SUCCESS) != CUDA_SUCCESS))"
        ) in result
        assert (
            "checkStatus((/* CUDA driver get version: output: driverVersion */ "
            "CUDA_SUCCESS))"
        ) in result
        assert (
            "(ok ? (/* CUDA driver get device attribute: output: major, "
            "attribute: CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, "
            "device: device */ CUDA_SUCCESS) : status)"
        ) in result
        assert "return CUDA_SUCCESS;" not in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuInit",
            "cuDriverGetVersion",
            "cuDeviceGet",
            "cuDeviceGetCount",
            "cuDeviceGetName",
            "cuDeviceGetUuid",
            "cuDeviceTotalMem",
            "cuDeviceTotalMem_v2",
            "cuDeviceGetAttribute",
            "cuDeviceComputeCapability",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_device_property_query_runtime_conversion(self):
        code = """
        bool checkStatus(CUresult status) {
            return status == CUDA_SUCCESS;
        }

        CUresult queryDeviceProperties(CUdevice device, CUdevice peer) {
            CUdevprop props;
            int peerValue;
            char luid[8];
            unsigned int nodeMask;
            CUdevice busDevice;
            char pciBusId[32];

            CUresult status = cuDeviceGetProperties(&props, device);
            status = cuDeviceGetP2PAttribute(
                &peerValue,
                CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK,
                device,
                peer
            );
            status = cuDeviceCanAccessPeer(&peerValue, device, peer);
            status = cuDeviceGetLuid(luid, &nodeMask, device);
            status = cuDeviceGetByPCIBusId(&busDevice, "0000:65:00.0");
            cuDeviceGetPCIBusId(pciBusId, 32, device);

            if (cuDeviceCanAccessPeer(&peerValue, device, peer) != CUDA_SUCCESS) {
                return CUDA_ERROR_UNKNOWN;
            }

            bool ok = checkStatus(cuDeviceGetPCIBusId(pciBusId, 32, device));
            return ok ? status : cuDeviceGetProperties(&props, peer);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver get device properties: output: props, device: device"
        ) in result
        assert (
            "// CUDA driver get device P2P attribute: output: peerValue, "
            "attribute: CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK, "
            "source device: device, destination device: peer"
        ) in result
        assert (
            "// CUDA driver device peer access query: output: peerValue, "
            "device: device, peer device: peer"
        ) in result
        assert (
            "// CUDA driver get device LUID: output: luid, node mask output: nodeMask, "
            "device: device"
        ) in result
        assert (
            '// CUDA driver get device by PCI bus ID: output: busDevice, bus ID: "0000:65:00.0"'
        ) in result
        assert (
            "// CUDA driver get device PCI bus ID: output: pciBusId, "
            "length: 32, device: device"
        ) in result
        assert (
            "if (((/* CUDA driver device peer access query: output: peerValue, "
            "device: device, peer device: peer */ CUDA_SUCCESS) != CUDA_SUCCESS))"
        ) in result
        assert (
            "checkStatus((/* CUDA driver get device PCI bus ID: output: pciBusId, "
            "length: 32, device: device */ CUDA_SUCCESS))"
        ) in result
        assert (
            "(ok ? status : (/* CUDA driver get device properties: output: props, "
            "device: peer */ CUDA_SUCCESS))"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuDeviceGetProperties",
            "cuDeviceGetP2PAttribute",
            "cuDeviceCanAccessPeer",
            "cuDeviceGetLuid",
            "cuDeviceGetByPCIBusId",
            "cuDeviceGetPCIBusId",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_graph_node_creation_dependency_runtime_conversion(self):
        code = """
        void buildGraph(
            cudaGraph_t graph,
            cudaGraph_t child,
            cudaGraphNode_t* deps,
            size_t count,
            cudaEvent_t event
        ) {
            cudaGraphNode_t kernelNode;
            cudaGraphNode_t memcpyNode;
            cudaGraphNode_t memsetNode;
            cudaGraphNode_t hostNode;
            cudaGraphNode_t childNode;
            cudaGraphNode_t emptyNode;
            cudaGraphNode_t eventRecordNode;
            cudaGraphNode_t eventWaitNode;
            cudaGraphNode_t foundNode;
            cudaKernelNodeParams kernelParams;
            cudaMemcpy3DParms copyParams;
            cudaMemsetParams memsetParams;
            cudaHostNodeParams hostParams;
            cudaGraphNodeType nodeType;

            cudaError_t err = cudaGraphAddKernelNode(
                &kernelNode,
                graph,
                deps,
                count,
                &kernelParams
            );
            err = cudaGraphAddMemcpyNode(&memcpyNode, graph, deps, count, &copyParams);
            err = cudaGraphAddMemsetNode(
                &memsetNode,
                graph,
                deps,
                count,
                &memsetParams
            );
            err = cudaGraphAddHostNode(&hostNode, graph, deps, count, &hostParams);
            err = cudaGraphAddChildGraphNode(&childNode, graph, deps, count, child);
            err = cudaGraphAddEmptyNode(&emptyNode, graph, deps, count);
            err = cudaGraphAddEventRecordNode(
                &eventRecordNode,
                graph,
                deps,
                count,
                event
            );
            err = cudaGraphAddEventWaitNode(&eventWaitNode, graph, deps, count, event);
            err = cudaGraphAddDependencies(graph, deps, &kernelNode, count);
            err = cudaGraphRemoveDependencies(graph, deps, &kernelNode, count);
            err = cudaGraphNodeFindInClone(&foundNode, kernelNode, graph);
            err = cudaGraphNodeGetType(kernelNode, &nodeType);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA graph add kernel node: output: kernelNode, graph: graph, "
            "dependencies: deps, dependency count: count, params: (&kernelParams)"
        ) in result
        assert (
            "// CUDA graph add memcpy node: output: memcpyNode, graph: graph, "
            "dependencies: deps, dependency count: count, params: (&copyParams)"
        ) in result
        assert (
            "// CUDA graph add memset node: output: memsetNode, graph: graph, "
            "dependencies: deps, dependency count: count, params: (&memsetParams)"
        ) in result
        assert (
            "// CUDA graph add host node: output: hostNode, graph: graph, "
            "dependencies: deps, dependency count: count, params: (&hostParams)"
        ) in result
        assert (
            "// CUDA graph add child graph node: output: childNode, graph: graph, "
            "dependencies: deps, dependency count: count, child graph: child"
        ) in result
        assert (
            "// CUDA graph add empty node: output: emptyNode, graph: graph, "
            "dependencies: deps, dependency count: count"
        ) in result
        assert (
            "// CUDA graph add event record node: output: eventRecordNode, "
            "graph: graph, dependencies: deps, dependency count: count, event: event"
        ) in result
        assert (
            "// CUDA graph add event wait node: output: eventWaitNode, "
            "graph: graph, dependencies: deps, dependency count: count, event: event"
        ) in result
        assert (
            "// CUDA graph add dependencies: graph: graph, from: deps, "
            "to: (&kernelNode), count: count"
        ) in result
        assert (
            "// CUDA graph remove dependencies: graph: graph, from: deps, "
            "to: (&kernelNode), count: count"
        ) in result
        assert (
            "// CUDA graph node find in clone: output: foundNode, "
            "original node: kernelNode, clone graph: graph"
        ) in result
        assert (
            "// CUDA graph node get type: node: kernelNode, output: nodeType" in result
        )
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "err = cudaSuccess;" in result
        for function_name in [
            "cudaGraphAddKernelNode",
            "cudaGraphAddMemcpyNode",
            "cudaGraphAddMemsetNode",
            "cudaGraphAddHostNode",
            "cudaGraphAddChildGraphNode",
            "cudaGraphAddEmptyNode",
            "cudaGraphAddEventRecordNode",
            "cudaGraphAddEventWaitNode",
            "cudaGraphAddDependencies",
            "cudaGraphRemoveDependencies",
            "cudaGraphNodeFindInClone",
            "cudaGraphNodeGetType",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_graph_node_creation_dependency_runtime_conversion(self):
        code = """
        bool checkStatus(CUresult status) {
            return status == CUDA_SUCCESS;
        }

        CUresult buildDriverGraph(
            CUgraph graph,
            CUgraph child,
            CUgraphNode* deps,
            size_t count,
            CUcontext context,
            bool ok
        ) {
            CUgraphNode kernelNode;
            CUgraphNode memcpyNode;
            CUgraphNode memsetNode;
            CUgraphNode hostNode;
            CUgraphNode childNode;
            CUgraphNode emptyNode;
            CUgraphNode foundNode;
            CUDA_KERNEL_NODE_PARAMS kernelParams;
            CUDA_MEMCPY3D copyParams;
            CUDA_MEMSET_NODE_PARAMS memsetParams;
            CUDA_HOST_NODE_PARAMS hostParams;
            CUgraphNodeType nodeType;

            CUresult status = cuGraphAddKernelNode(
                &kernelNode,
                graph,
                deps,
                count,
                &kernelParams
            );
            status = cuGraphAddMemcpyNode(
                &memcpyNode,
                graph,
                deps,
                count,
                &copyParams,
                context
            );
            status = cuGraphAddMemsetNode(
                &memsetNode,
                graph,
                deps,
                count,
                &memsetParams,
                context
            );
            status = cuGraphAddHostNode(&hostNode, graph, deps, count, &hostParams);
            status = cuGraphAddChildGraphNode(&childNode, graph, deps, count, child);
            cuGraphAddEmptyNode(&emptyNode, graph, deps, count);
            status = cuGraphRemoveDependencies(graph, deps, &kernelNode, count);

            if (cuGraphAddDependencies(graph, deps, &kernelNode, count) != CUDA_SUCCESS) {
                return status;
            }

            bool found = checkStatus(
                cuGraphNodeFindInClone(&foundNode, kernelNode, graph)
            );
            return found && ok ? cuGraphNodeGetType(kernelNode, &nodeType) : status;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver graph add kernel node: output: kernelNode, "
            "graph: graph, dependencies: deps, dependency count: count, "
            "params: (&kernelParams)"
        ) in result
        assert (
            "// CUDA driver graph add memcpy node: output: memcpyNode, "
            "graph: graph, dependencies: deps, dependency count: count, "
            "params: (&copyParams), context: context"
        ) in result
        assert (
            "// CUDA driver graph add memset node: output: memsetNode, "
            "graph: graph, dependencies: deps, dependency count: count, "
            "params: (&memsetParams), context: context"
        ) in result
        assert (
            "// CUDA driver graph add host node: output: hostNode, graph: graph, "
            "dependencies: deps, dependency count: count, params: (&hostParams)"
        ) in result
        assert (
            "// CUDA driver graph add child graph node: output: childNode, "
            "graph: graph, dependencies: deps, dependency count: count, "
            "child graph: child"
        ) in result
        assert (
            "// CUDA driver graph add empty node: output: emptyNode, graph: graph, "
            "dependencies: deps, dependency count: count"
        ) in result
        assert (
            "// CUDA driver graph remove dependencies: graph: graph, from: deps, "
            "to: (&kernelNode), count: count"
        ) in result
        assert (
            "if (((/* CUDA driver graph add dependencies: graph: graph, "
            "from: deps, to: (&kernelNode), count: count */ CUDA_SUCCESS) "
            "!= CUDA_SUCCESS))"
        ) in result
        assert (
            "checkStatus((/* CUDA driver graph node find in clone: "
            "output: foundNode, original node: kernelNode, clone graph: graph */ "
            "CUDA_SUCCESS))"
        ) in result
        assert (
            "(found && ok) ? (/* CUDA driver graph node get type: "
            "node: kernelNode, output: nodeType */ CUDA_SUCCESS) : status"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuGraphAddKernelNode",
            "cuGraphAddMemcpyNode",
            "cuGraphAddMemsetNode",
            "cuGraphAddHostNode",
            "cuGraphAddChildGraphNode",
            "cuGraphAddEmptyNode",
            "cuGraphAddDependencies",
            "cuGraphRemoveDependencies",
            "cuGraphNodeFindInClone",
            "cuGraphNodeGetType",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_graph_external_event_child_runtime_conversion(self):
        code = """
        bool checkStatus(CUresult status) {
            return status == CUDA_SUCCESS;
        }

        CUresult configureDriverInteropGraph(
            CUgraph graph,
            CUgraphExec exec,
            CUgraph child,
            CUgraphNode* deps,
            size_t count,
            CUevent event,
            bool ok
        ) {
            CUgraphNode signalNode;
            CUgraphNode waitNode;
            CUgraphNode eventRecordNode;
            CUgraphNode eventWaitNode;
            CUgraphNode childNode;
            CUDA_EXT_SEM_SIGNAL_NODE_PARAMS signalParams;
            CUDA_EXT_SEM_WAIT_NODE_PARAMS waitParams;
            CUevent eventOut;
            CUgraph childOut;

            CUresult status = cuGraphAddEventRecordNode(
                &eventRecordNode,
                graph,
                deps,
                count,
                event
            );
            status = cuGraphAddEventWaitNode(
                &eventWaitNode,
                graph,
                deps,
                count,
                event
            );
            status = cuGraphAddExternalSemaphoresSignalNode(
                &signalNode,
                graph,
                deps,
                count,
                &signalParams
            );
            status = cuGraphAddExternalSemaphoresWaitNode(
                &waitNode,
                graph,
                deps,
                count,
                &waitParams
            );
            status = cuGraphExternalSemaphoresSignalNodeGetParams(
                signalNode,
                &signalParams
            );
            status = cuGraphExternalSemaphoresSignalNodeSetParams(
                signalNode,
                &signalParams
            );
            status = cuGraphExternalSemaphoresWaitNodeGetParams(
                waitNode,
                &waitParams
            );
            status = cuGraphExternalSemaphoresWaitNodeSetParams(
                waitNode,
                &waitParams
            );
            status = cuGraphEventRecordNodeGetEvent(eventRecordNode, &eventOut);
            status = cuGraphEventRecordNodeSetEvent(eventRecordNode, event);
            status = cuGraphEventWaitNodeGetEvent(eventWaitNode, &eventOut);
            cuGraphEventWaitNodeSetEvent(eventWaitNode, event);
            status = cuGraphChildGraphNodeGetGraph(childNode, &childOut);
            status = cuGraphExecExternalSemaphoresWaitNodeSetParams(
                exec,
                waitNode,
                &waitParams
            );
            status = cuGraphExecEventRecordNodeSetEvent(
                exec,
                eventRecordNode,
                event
            );

            if (
                cuGraphExecChildGraphNodeSetParams(exec, childNode, child)
                != CUDA_SUCCESS
            ) {
                return status;
            }

            bool eventSet = checkStatus(
                cuGraphExecEventWaitNodeSetEvent(exec, eventWaitNode, event)
            );
            return eventSet && ok
                ? cuGraphExecExternalSemaphoresSignalNodeSetParams(
                    exec,
                    signalNode,
                    &signalParams
                )
                : status;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver graph add event record node: output: eventRecordNode, "
            "graph: graph, dependencies: deps, dependency count: count, event: event"
        ) in result
        assert (
            "// CUDA driver graph add event wait node: output: eventWaitNode, "
            "graph: graph, dependencies: deps, dependency count: count, event: event"
        ) in result
        assert (
            "// CUDA driver graph add external semaphore signal node: "
            "output: signalNode, graph: graph, dependencies: deps, "
            "dependency count: count, params: (&signalParams)"
        ) in result
        assert (
            "// CUDA driver graph add external semaphore wait node: "
            "output: waitNode, graph: graph, dependencies: deps, "
            "dependency count: count, params: (&waitParams)"
        ) in result
        assert (
            "// CUDA driver graph external semaphore signal node get params: "
            "node: signalNode, params: (&signalParams)"
        ) in result
        assert (
            "// CUDA driver graph external semaphore signal node set params: "
            "node: signalNode, params: (&signalParams)"
        ) in result
        assert (
            "// CUDA driver graph external semaphore wait node get params: "
            "node: waitNode, params: (&waitParams)"
        ) in result
        assert (
            "// CUDA driver graph external semaphore wait node set params: "
            "node: waitNode, params: (&waitParams)"
        ) in result
        assert (
            "// CUDA driver graph event record node get event: "
            "node: eventRecordNode, output: eventOut"
        ) in result
        assert (
            "// CUDA driver graph event record node set event: "
            "node: eventRecordNode, event: event"
        ) in result
        assert (
            "// CUDA driver graph event wait node get event: "
            "node: eventWaitNode, output: eventOut"
        ) in result
        assert (
            "// CUDA driver graph event wait node set event: "
            "node: eventWaitNode, event: event"
        ) in result
        assert (
            "// CUDA driver graph child graph node get graph: "
            "node: childNode, output: childOut"
        ) in result
        assert (
            "// CUDA driver graph exec set external semaphore wait node params: "
            "exec: exec, node: waitNode, params: (&waitParams)"
        ) in result
        assert (
            "// CUDA driver graph exec set event record node event: "
            "exec: exec, node: eventRecordNode, event: event"
        ) in result
        assert (
            "if (((/* CUDA driver graph exec set child graph node params: "
            "exec: exec, node: childNode, child graph: child */ CUDA_SUCCESS) "
            "!= CUDA_SUCCESS))"
        ) in result
        assert (
            "checkStatus((/* CUDA driver graph exec set event wait node event: "
            "exec: exec, node: eventWaitNode, event: event */ CUDA_SUCCESS))"
        ) in result
        assert (
            "(eventSet && ok) ? (/* CUDA driver graph exec set external "
            "semaphore signal node params: exec: exec, node: signalNode, "
            "params: (&signalParams) */ CUDA_SUCCESS) : status"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuGraphAddEventRecordNode",
            "cuGraphAddEventWaitNode",
            "cuGraphAddExternalSemaphoresSignalNode",
            "cuGraphAddExternalSemaphoresWaitNode",
            "cuGraphExternalSemaphoresSignalNodeGetParams",
            "cuGraphExternalSemaphoresSignalNodeSetParams",
            "cuGraphExternalSemaphoresWaitNodeGetParams",
            "cuGraphExternalSemaphoresWaitNodeSetParams",
            "cuGraphEventRecordNodeGetEvent",
            "cuGraphEventRecordNodeSetEvent",
            "cuGraphEventWaitNodeGetEvent",
            "cuGraphEventWaitNodeSetEvent",
            "cuGraphChildGraphNodeGetGraph",
            "cuGraphExecExternalSemaphoresWaitNodeSetParams",
            "cuGraphExecEventRecordNodeSetEvent",
            "cuGraphExecChildGraphNodeSetParams",
            "cuGraphExecEventWaitNodeSetEvent",
            "cuGraphExecExternalSemaphoresSignalNodeSetParams",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_graph_memory_user_object_runtime_conversion(self):
        code = """
        bool checkStatus(CUresult status) {
            return status == CUDA_SUCCESS;
        }

        CUresult manageDriverGraphResources(
            CUdevice device,
            CUgraph graph,
            CUgraphNode* deps,
            size_t count,
            CUdeviceptr pointer,
            void* payload,
            CUhostFn destroy,
            bool ok
        ) {
            CUgraphNode allocNode;
            CUgraphNode freeNode;
            CUDA_MEM_ALLOC_NODE_PARAMS allocParams;
            CUdeviceptr freedPtr;
            CUuserObject object;
            CUuserObject created;
            size_t usedBytes;
            size_t highWatermark;

            CUresult status = cuGraphAddMemAllocNode(
                &allocNode,
                graph,
                deps,
                count,
                &allocParams
            );
            status = cuGraphAddMemFreeNode(&freeNode, graph, deps, count, pointer);
            status = cuGraphMemAllocNodeGetParams(allocNode, &allocParams);
            status = cuGraphMemFreeNodeGetParams(freeNode, &freedPtr);
            status = cuUserObjectCreate(
                &created,
                payload,
                destroy,
                2u,
                CU_USER_OBJECT_NO_DESTRUCTOR_SYNC
            );
            status = cuUserObjectRetain(created, 3u);
            status = cuGraphRetainUserObject(
                graph,
                created,
                1u,
                CU_GRAPH_USER_OBJECT_MOVE
            );
            status = cuGraphRetainUserObject(graph, object, 1u, 0);
            status = cuGraphReleaseUserObject(graph, created, 1u);
            status = cuDeviceGetGraphMemAttribute(
                device,
                CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT,
                &usedBytes
            );
            status = cuDeviceSetGraphMemAttribute(
                device,
                CU_GRAPH_MEM_ATTR_USED_MEM_HIGH,
                &highWatermark
            );
            cuDeviceGraphMemTrim(device);

            if (cuGraphMemFreeNodeGetParams(freeNode, &freedPtr) != CUDA_SUCCESS) {
                return status;
            }

            bool released = checkStatus(
                cuGraphReleaseUserObject(graph, object, 1u)
            );
            return released && ok ? cuUserObjectRelease(object, 2u) : status;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver graph add memory alloc node: output: allocNode, "
            "graph: graph, dependencies: deps, dependency count: count, "
            "params: (&allocParams)"
        ) in result
        assert (
            "// CUDA driver graph add memory free node: output: freeNode, "
            "graph: graph, dependencies: deps, dependency count: count, "
            "pointer: pointer"
        ) in result
        assert (
            "// CUDA driver graph memory alloc node get params: node: allocNode, "
            "params: (&allocParams)"
        ) in result
        assert (
            "// CUDA driver graph memory free node get params: node: freeNode, "
            "output: freedPtr"
        ) in result
        assert (
            "// CUDA driver user object create: output: created, payload: payload, "
            "destroy callback: destroy, initial references: 2u, "
            "flags: CU_USER_OBJECT_NO_DESTRUCTOR_SYNC"
        ) in result
        assert (
            "// CUDA driver user object retain: object: created, references: 3u"
            in result
        )
        assert (
            "// CUDA driver graph retain user object: graph: graph, "
            "object: created, references: 1u, flags: CU_GRAPH_USER_OBJECT_MOVE"
        ) in result
        assert (
            "// CUDA driver graph retain user object: graph: graph, "
            "object: object, references: 1u, flags: 0"
        ) in result
        assert (
            "// CUDA driver graph release user object: graph: graph, "
            "object: created, references: 1u"
        ) in result
        assert (
            "// CUDA driver device get graph memory attribute: device: device, "
            "attribute: CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT, output: usedBytes"
        ) in result
        assert (
            "// CUDA driver device set graph memory attribute: device: device, "
            "attribute: CU_GRAPH_MEM_ATTR_USED_MEM_HIGH, value: highWatermark"
        ) in result
        assert "// CUDA driver device graph memory trim: device: device" in result
        assert (
            "if (((/* CUDA driver graph memory free node get params: "
            "node: freeNode, output: freedPtr */ CUDA_SUCCESS) != CUDA_SUCCESS))"
        ) in result
        assert (
            "checkStatus((/* CUDA driver graph release user object: "
            "graph: graph, object: object, references: 1u */ CUDA_SUCCESS))"
        ) in result
        assert (
            "(released && ok) ? (/* CUDA driver user object release: "
            "object: object, references: 2u */ CUDA_SUCCESS) : status"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuGraphAddMemAllocNode",
            "cuGraphAddMemFreeNode",
            "cuGraphMemAllocNodeGetParams",
            "cuGraphMemFreeNodeGetParams",
            "cuUserObjectCreate",
            "cuUserObjectRetain",
            "cuGraphRetainUserObject",
            "cuGraphReleaseUserObject",
            "cuDeviceGetGraphMemAttribute",
            "cuDeviceSetGraphMemAttribute",
            "cuDeviceGraphMemTrim",
            "cuUserObjectRelease",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_graph_memcpy_upload_memory_runtime_conversion(self):
        code = """
        void configureGraph(
            cudaGraph_t graph,
            cudaGraphExec_t exec,
            cudaGraphNode_t* deps,
            size_t count,
            cudaStream_t stream,
            void* dst,
            void* src,
            size_t bytes,
            size_t offset
        ) {
            cudaGraphNode_t memcpy1DNode;
            cudaGraphNode_t fromSymbolNode;
            cudaGraphNode_t toSymbolNode;
            cudaGraphNode_t allocNode;
            cudaGraphNode_t freeNode;
            cudaMemAllocNodeParams allocParams;
            void* dptr;
            void* freedPtr;
            int symbolData;

            cudaError_t err = cudaGraphAddMemcpyNode1D(
                &memcpy1DNode,
                graph,
                deps,
                count,
                dst,
                src,
                bytes,
                cudaMemcpyDeviceToDevice
            );
            err = cudaGraphAddMemcpyNodeFromSymbol(
                &fromSymbolNode,
                graph,
                deps,
                count,
                dst,
                symbolData,
                bytes,
                offset,
                cudaMemcpyDeviceToDevice
            );
            err = cudaGraphAddMemcpyNodeToSymbol(
                &toSymbolNode,
                graph,
                deps,
                count,
                symbolData,
                src,
                bytes,
                offset,
                cudaMemcpyDeviceToDevice
            );
            err = cudaGraphUpload(exec, stream);
            err = cudaGraphExecUpload(exec, stream);
            err = cudaGraphAddMemAllocNode(&allocNode, graph, deps, count, &allocParams);
            err = cudaGraphAddMemFreeNode(&freeNode, graph, deps, count, dptr);
            err = cudaGraphMemAllocNodeGetParams(allocNode, &allocParams);
            err = cudaGraphMemFreeNodeGetParams(freeNode, &freedPtr);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA graph add memcpy 1D node: output: memcpy1DNode, "
            "graph: graph, dependencies: deps, dependency count: count, "
            "dst: dst, src: src, byte count: bytes, kind: cudaMemcpyDeviceToDevice"
        ) in result
        assert (
            "// CUDA graph add memcpy-from-symbol node: output: fromSymbolNode, "
            "graph: graph, dependencies: deps, dependency count: count, dst: dst, "
            "symbol: symbolData, byte count: bytes, offset: offset, "
            "kind: cudaMemcpyDeviceToDevice"
        ) in result
        assert (
            "// CUDA graph add memcpy-to-symbol node: output: toSymbolNode, "
            "graph: graph, dependencies: deps, dependency count: count, "
            "symbol: symbolData, src: src, byte count: bytes, offset: offset, "
            "kind: cudaMemcpyDeviceToDevice"
        ) in result
        assert "// CUDA graph upload: exec: exec, stream: stream" in result
        assert "// CUDA graph exec upload: exec: exec, stream: stream" in result
        assert (
            "// CUDA graph add memory alloc node: output: allocNode, "
            "graph: graph, dependencies: deps, dependency count: count, "
            "params: (&allocParams)"
        ) in result
        assert (
            "// CUDA graph add memory free node: output: freeNode, graph: graph, "
            "dependencies: deps, dependency count: count, pointer: dptr"
        ) in result
        assert (
            "// CUDA graph memory alloc node get params: node: allocNode, "
            "params: (&allocParams)"
        ) in result
        assert (
            "// CUDA graph memory free node get params: node: freeNode, "
            "output: freedPtr"
        ) in result
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "err = cudaSuccess;" in result
        for function_name in [
            "cudaGraphAddMemcpyNode1D",
            "cudaGraphAddMemcpyNodeFromSymbol",
            "cudaGraphAddMemcpyNodeToSymbol",
            "cudaGraphUpload",
            "cudaGraphExecUpload",
            "cudaGraphAddMemAllocNode",
            "cudaGraphAddMemFreeNode",
            "cudaGraphMemAllocNodeGetParams",
            "cudaGraphMemFreeNodeGetParams",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_graph_user_object_memory_attribute_runtime_conversion(self):
        code = """
        void manageGraphResources(
            cudaGraph_t graph,
            cudaUserObject_t object,
            void* payload,
            cudaHostFn_t destroy
        ) {
            cudaUserObject_t created;
            size_t graphBytes;
            size_t highWatermark;

            cudaError_t err = cudaUserObjectCreate(
                &created,
                payload,
                destroy,
                2u,
                cudaUserObjectNoDestructorSync
            );
            err = cudaUserObjectRetain(created, 3u);
            err = cudaUserObjectRetain(created);
            err = cudaGraphRetainUserObject(
                graph,
                created,
                1u,
                cudaGraphUserObjectMove
            );
            err = cudaGraphRetainUserObject(graph, object);
            err = cudaGraphReleaseUserObject(graph, created, 1u);
            err = cudaGraphReleaseUserObject(graph, object);
            err = cudaUserObjectRelease(created, 2u);
            err = cudaUserObjectRelease(object);
            err = cudaDeviceGetGraphMemAttribute(
                0,
                cudaGraphMemAttrUsedMemCurrent,
                &graphBytes
            );
            err = cudaDeviceSetGraphMemAttribute(
                0,
                cudaGraphMemAttrUsedMemHigh,
                &highWatermark
            );
            err = cudaDeviceGraphMemTrim(0);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA user object create: output: created, payload: payload, "
            "destroy callback: destroy, initial references: 2u, "
            "flags: cudaUserObjectNoDestructorSync"
        ) in result
        assert "// CUDA user object retain: object: created, references: 3u" in result
        assert "// CUDA user object retain: object: created, references: 1" in result
        assert (
            "// CUDA graph retain user object: graph: graph, object: created, "
            "references: 1u, flags: cudaGraphUserObjectMove"
        ) in result
        assert (
            "// CUDA graph retain user object: graph: graph, object: object, "
            "references: 1, flags: 0"
        ) in result
        assert (
            "// CUDA graph release user object: graph: graph, object: created, "
            "references: 1u"
        ) in result
        assert (
            "// CUDA graph release user object: graph: graph, object: object, "
            "references: 1"
        ) in result
        assert "// CUDA user object release: object: created, references: 2u" in result
        assert "// CUDA user object release: object: object, references: 1" in result
        assert (
            "// CUDA device get graph memory attribute: device: 0, "
            "attribute: cudaGraphMemAttrUsedMemCurrent, output: graphBytes"
        ) in result
        assert (
            "// CUDA device set graph memory attribute: device: 0, "
            "attribute: cudaGraphMemAttrUsedMemHigh, value: highWatermark"
        ) in result
        assert "// CUDA device graph memory trim: device: 0" in result
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "err = cudaSuccess;" in result
        for function_name in [
            "cudaUserObjectCreate",
            "cudaUserObjectRetain",
            "cudaGraphRetainUserObject",
            "cudaGraphReleaseUserObject",
            "cudaUserObjectRelease",
            "cudaDeviceGetGraphMemAttribute",
            "cudaDeviceSetGraphMemAttribute",
            "cudaDeviceGraphMemTrim",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_graph_external_semaphore_event_node_runtime_conversion(self):
        code = """
        void configureInteropGraph(
            cudaGraph_t graph,
            cudaGraphExec_t exec,
            cudaGraph_t child,
            cudaGraphNode_t* deps,
            size_t count,
            cudaEvent_t event
        ) {
            cudaGraphNode_t signalNode;
            cudaGraphNode_t waitNode;
            cudaGraphNode_t eventRecordNode;
            cudaGraphNode_t eventWaitNode;
            cudaGraphNode_t childNode;
            cudaExternalSemaphoreSignalNodeParams signalParams;
            cudaExternalSemaphoreWaitNodeParams waitParams;
            cudaEvent_t eventOut;
            cudaGraph_t childOut;

            cudaError_t err = cudaGraphAddExternalSemaphoresSignalNode(
                &signalNode,
                graph,
                deps,
                count,
                &signalParams
            );
            err = cudaGraphAddExternalSemaphoresWaitNode(
                &waitNode,
                graph,
                deps,
                count,
                &waitParams
            );
            err = cudaGraphExternalSemaphoresSignalNodeGetParams(
                signalNode,
                &signalParams
            );
            err = cudaGraphExternalSemaphoresSignalNodeSetParams(
                signalNode,
                &signalParams
            );
            err = cudaGraphExternalSemaphoresWaitNodeGetParams(waitNode, &waitParams);
            err = cudaGraphExternalSemaphoresWaitNodeSetParams(waitNode, &waitParams);
            err = cudaGraphExecExternalSemaphoresSignalNodeSetParams(
                exec,
                signalNode,
                &signalParams
            );
            err = cudaGraphExecExternalSemaphoresWaitNodeSetParams(
                exec,
                waitNode,
                &waitParams
            );
            err = cudaGraphEventRecordNodeGetEvent(eventRecordNode, &eventOut);
            err = cudaGraphEventRecordNodeSetEvent(eventRecordNode, event);
            err = cudaGraphEventWaitNodeGetEvent(eventWaitNode, &eventOut);
            err = cudaGraphEventWaitNodeSetEvent(eventWaitNode, event);
            err = cudaGraphExecEventRecordNodeSetEvent(exec, eventRecordNode, event);
            err = cudaGraphExecEventWaitNodeSetEvent(exec, eventWaitNode, event);
            err = cudaGraphChildGraphNodeGetGraph(childNode, &childOut);
            err = cudaGraphExecChildGraphNodeSetParams(exec, childNode, child);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA graph add external semaphore signal node: "
            "output: signalNode, graph: graph, dependencies: deps, "
            "dependency count: count, params: (&signalParams)"
        ) in result
        assert (
            "// CUDA graph add external semaphore wait node: output: waitNode, "
            "graph: graph, dependencies: deps, dependency count: count, "
            "params: (&waitParams)"
        ) in result
        assert (
            "// CUDA graph external semaphore signal node get params: "
            "node: signalNode, params: (&signalParams)"
        ) in result
        assert (
            "// CUDA graph external semaphore signal node set params: "
            "node: signalNode, params: (&signalParams)"
        ) in result
        assert (
            "// CUDA graph external semaphore wait node get params: "
            "node: waitNode, params: (&waitParams)"
        ) in result
        assert (
            "// CUDA graph external semaphore wait node set params: "
            "node: waitNode, params: (&waitParams)"
        ) in result
        assert (
            "// CUDA graph exec set external semaphore signal node params: "
            "exec: exec, node: signalNode, params: (&signalParams)"
        ) in result
        assert (
            "// CUDA graph exec set external semaphore wait node params: "
            "exec: exec, node: waitNode, params: (&waitParams)"
        ) in result
        assert (
            "// CUDA graph event record node get event: "
            "node: eventRecordNode, output: eventOut"
        ) in result
        assert (
            "// CUDA graph event record node set event: "
            "node: eventRecordNode, event: event"
        ) in result
        assert (
            "// CUDA graph event wait node get event: "
            "node: eventWaitNode, output: eventOut"
        ) in result
        assert (
            "// CUDA graph event wait node set event: "
            "node: eventWaitNode, event: event"
        ) in result
        assert (
            "// CUDA graph exec set event record node event: "
            "exec: exec, node: eventRecordNode, event: event"
        ) in result
        assert (
            "// CUDA graph exec set event wait node event: "
            "exec: exec, node: eventWaitNode, event: event"
        ) in result
        assert (
            "// CUDA graph child graph node get graph: "
            "node: childNode, output: childOut"
        ) in result
        assert (
            "// CUDA graph exec set child graph node params: "
            "exec: exec, node: childNode, child graph: child"
        ) in result
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "err = cudaSuccess;" in result
        for function_name in [
            "cudaGraphAddExternalSemaphoresSignalNode",
            "cudaGraphAddExternalSemaphoresWaitNode",
            "cudaGraphExternalSemaphoresSignalNodeGetParams",
            "cudaGraphExternalSemaphoresSignalNodeSetParams",
            "cudaGraphExternalSemaphoresWaitNodeGetParams",
            "cudaGraphExternalSemaphoresWaitNodeSetParams",
            "cudaGraphExecExternalSemaphoresSignalNodeSetParams",
            "cudaGraphExecExternalSemaphoresWaitNodeSetParams",
            "cudaGraphEventRecordNodeGetEvent",
            "cudaGraphEventRecordNodeSetEvent",
            "cudaGraphEventWaitNodeGetEvent",
            "cudaGraphEventWaitNodeSetEvent",
            "cudaGraphExecEventRecordNodeSetEvent",
            "cudaGraphExecEventWaitNodeSetEvent",
            "cudaGraphChildGraphNodeGetGraph",
            "cudaGraphExecChildGraphNodeSetParams",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_graph_generic_node_query_runtime_conversion(self):
        code = """
        void inspectGraph(
            cudaGraph_t graph,
            cudaGraphExec_t exec,
            cudaGraphNode_t node,
            cudaGraphNode_t* deps,
            cudaGraphEdgeData* edgeData,
            size_t count,
            char* path
        ) {
            cudaGraphNode_t genericNode;
            cudaGraphNode_t* nodes;
            cudaGraphNode_t* rootNodes;
            cudaGraphNode_t* fromNodes;
            cudaGraphNode_t* toNodes;
            cudaGraphNodeParams params;
            cudaGraph_t containingGraph;
            unsigned long long execFlags;
            unsigned long long toolsNodeId;
            unsigned int graphId;
            unsigned int execId;
            unsigned int nodeId;
            unsigned int enabled;
            size_t nodeCount;
            size_t rootCount;
            size_t edgeCount;
            size_t dependencyCount;
            size_t dependentCount;

            cudaError_t err = cudaGraphAddNode(
                &genericNode,
                graph,
                deps,
                edgeData,
                count,
                &params
            );
            err = cudaGraphAddDependencies(
                graph,
                deps,
                &genericNode,
                edgeData,
                count
            );
            err = cudaGraphRemoveDependencies(
                graph,
                deps,
                &genericNode,
                edgeData,
                count
            );
            err = cudaGraphNodeGetParams(node, &params);
            err = cudaGraphNodeSetParams(node, &params);
            err = cudaGraphExecNodeSetParams(exec, node, &params);
            err = cudaGraphDebugDotPrint(
                graph,
                path,
                cudaGraphDebugDotFlagsVerbose
            );
            err = cudaGraphExecGetFlags(exec, &execFlags);
            err = cudaGraphExecGetId(exec, &execId);
            err = cudaGraphGetId(graph, &graphId);
            err = cudaGraphGetEdges(
                graph,
                fromNodes,
                toNodes,
                edgeData,
                &edgeCount
            );
            err = cudaGraphGetNodes(graph, nodes, &nodeCount);
            err = cudaGraphGetRootNodes(graph, rootNodes, &rootCount);
            err = cudaGraphNodeGetContainingGraph(node, &containingGraph);
            err = cudaGraphNodeGetDependencies(
                node,
                deps,
                edgeData,
                &dependencyCount
            );
            err = cudaGraphNodeGetDependentNodes(
                node,
                deps,
                edgeData,
                &dependentCount
            );
            err = cudaGraphNodeGetEnabled(exec, node, &enabled);
            err = cudaGraphNodeSetEnabled(exec, node, enabled);
            err = cudaGraphNodeGetLocalId(node, &nodeId);
            err = cudaGraphNodeGetToolsId(node, &toolsNodeId);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA graph add generic node: output: genericNode, graph: graph, "
            "dependencies: deps, edge data: edgeData, dependency count: count, "
            "params: (&params)"
        ) in result
        assert (
            "// CUDA graph add dependencies: graph: graph, from: deps, "
            "to: (&genericNode), edge data: edgeData, count: count"
        ) in result
        assert (
            "// CUDA graph remove dependencies: graph: graph, from: deps, "
            "to: (&genericNode), edge data: edgeData, count: count"
        ) in result
        assert "// CUDA graph node get params: node: node, params: (&params)" in result
        assert "// CUDA graph node set params: node: node, params: (&params)" in result
        assert (
            "// CUDA graph exec set node params: exec: exec, node: node, "
            "params: (&params)"
        ) in result
        assert (
            "// CUDA graph debug DOT print: graph: graph, path: path, "
            "flags: cudaGraphDebugDotFlagsVerbose"
        ) in result
        assert "// CUDA graph exec get flags: exec: exec, output: execFlags" in result
        assert "// CUDA graph exec get id: exec: exec, output: execId" in result
        assert "// CUDA graph get id: graph: graph, output: graphId" in result
        assert (
            "// CUDA graph get edges: graph: graph, from output: fromNodes, "
            "to output: toNodes, edge data: edgeData, count output: edgeCount"
        ) in result
        assert (
            "// CUDA graph get nodes: graph: graph, nodes output: nodes, "
            "count output: nodeCount"
        ) in result
        assert (
            "// CUDA graph get root nodes: graph: graph, nodes output: rootNodes, "
            "count output: rootCount"
        ) in result
        assert (
            "// CUDA graph node get containing graph: node: node, "
            "output: containingGraph"
        ) in result
        assert (
            "// CUDA graph node get dependencies: node: node, "
            "dependencies output: deps, edge data: edgeData, "
            "count output: dependencyCount"
        ) in result
        assert (
            "// CUDA graph node get dependent nodes: node: node, "
            "dependent nodes output: deps, edge data: edgeData, "
            "count output: dependentCount"
        ) in result
        assert (
            "// CUDA graph node get enabled: exec: exec, node: node, " "output: enabled"
        ) in result
        assert (
            "// CUDA graph node set enabled: exec: exec, node: node, "
            "enabled: enabled"
        ) in result
        assert "// CUDA graph node get local id: node: node, output: nodeId" in result
        assert (
            "// CUDA graph node get tools id: node: node, output: toolsNodeId" in result
        )
        assert "var err: cudaError_t = cudaSuccess;" in result
        assert "err = cudaSuccess;" in result
        for function_name in [
            "cudaGraphAddNode",
            "cudaGraphAddDependencies",
            "cudaGraphRemoveDependencies",
            "cudaGraphNodeGetParams",
            "cudaGraphNodeSetParams",
            "cudaGraphExecNodeSetParams",
            "cudaGraphDebugDotPrint",
            "cudaGraphExecGetFlags",
            "cudaGraphExecGetId",
            "cudaGraphGetId",
            "cudaGraphGetEdges",
            "cudaGraphGetNodes",
            "cudaGraphGetRootNodes",
            "cudaGraphNodeGetContainingGraph",
            "cudaGraphNodeGetDependencies",
            "cudaGraphNodeGetDependentNodes",
            "cudaGraphNodeGetEnabled",
            "cudaGraphNodeSetEnabled",
            "cudaGraphNodeGetLocalId",
            "cudaGraphNodeGetToolsId",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_graph_query_update_runtime_conversion(self):
        code = """
        bool checkStatus(CUresult status) {
            return status == CUDA_SUCCESS;
        }

        CUresult inspectDriverGraph(
            CUgraph graph,
            CUgraphExec exec,
            CUgraphNode node,
            CUgraphNode* deps,
            CUDA_GRAPH_EDGE_DATA* edgeData,
            char* path,
            bool ok
        ) {
            CUgraphNode* nodes;
            CUgraphNode* rootNodes;
            CUgraphNode* fromNodes;
            CUgraphNode* toNodes;
            CUDA_GRAPH_NODE_PARAMS params;
            CUgraph containingGraph;
            unsigned long long execFlags;
            unsigned long long toolsNodeId;
            unsigned int graphId;
            unsigned int execId;
            unsigned int nodeId;
            unsigned int enabled;
            size_t nodeCount;
            size_t rootCount;
            size_t edgeCount;
            size_t dependencyCount;
            size_t dependentCount;

            CUresult status = cuGraphNodeGetParams(node, &params);
            status = cuGraphNodeSetParams(node, &params);
            status = cuGraphExecNodeSetParams(exec, node, &params);
            status = cuGraphDebugDotPrint(graph, path, CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE);
            status = cuGraphExecGetFlags(exec, &execFlags);
            status = cuGraphExecGetId(exec, &execId);
            status = cuGraphGetId(graph, &graphId);
            status = cuGraphGetEdges(graph, fromNodes, toNodes, edgeData, &edgeCount);
            status = cuGraphGetNodes(graph, nodes, &nodeCount);
            status = cuGraphGetRootNodes(graph, rootNodes, &rootCount);
            status = cuGraphNodeGetContainingGraph(node, &containingGraph);
            status = cuGraphNodeGetDependencies(
                node,
                deps,
                edgeData,
                &dependencyCount
            );
            status = cuGraphNodeGetDependentNodes(
                node,
                deps,
                edgeData,
                &dependentCount
            );
            status = cuGraphNodeGetEnabled(exec, node, &enabled);
            cuGraphNodeSetEnabled(exec, node, enabled);

            if (cuGraphNodeGetLocalId(node, &nodeId) != CUDA_SUCCESS) {
                return status;
            }

            bool tools = checkStatus(cuGraphNodeGetToolsId(node, &toolsNodeId));
            return tools && ok ? cuGraphExecNodeSetParams(exec, node, &params) : status;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver graph node get params: node: node, params: (&params)"
            in result
        )
        assert (
            "// CUDA driver graph node set params: node: node, params: (&params)"
            in result
        )
        assert (
            "// CUDA driver graph exec set node params: exec: exec, node: node, "
            "params: (&params)"
        ) in result
        assert (
            "// CUDA driver graph debug DOT print: graph: graph, path: path, "
            "flags: CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE"
        ) in result
        assert (
            "// CUDA driver graph exec get flags: exec: exec, output: execFlags"
            in result
        )
        assert "// CUDA driver graph exec get id: exec: exec, output: execId" in result
        assert "// CUDA driver graph get id: graph: graph, output: graphId" in result
        assert (
            "// CUDA driver graph get edges: graph: graph, from output: fromNodes, "
            "to output: toNodes, edge data: edgeData, count output: edgeCount"
        ) in result
        assert (
            "// CUDA driver graph get nodes: graph: graph, nodes output: nodes, "
            "count output: nodeCount"
        ) in result
        assert (
            "// CUDA driver graph get root nodes: graph: graph, "
            "nodes output: rootNodes, count output: rootCount"
        ) in result
        assert (
            "// CUDA driver graph node get containing graph: node: node, "
            "output: containingGraph"
        ) in result
        assert (
            "// CUDA driver graph node get dependencies: node: node, "
            "dependencies output: deps, edge data: edgeData, "
            "count output: dependencyCount"
        ) in result
        assert (
            "// CUDA driver graph node get dependent nodes: node: node, "
            "dependent nodes output: deps, edge data: edgeData, "
            "count output: dependentCount"
        ) in result
        assert (
            "// CUDA driver graph node get enabled: exec: exec, node: node, "
            "output: enabled"
        ) in result
        assert (
            "// CUDA driver graph node set enabled: exec: exec, node: node, "
            "enabled: enabled"
        ) in result
        assert (
            "if (((/* CUDA driver graph node get local id: "
            "node: node, output: nodeId */ CUDA_SUCCESS) != CUDA_SUCCESS))"
        ) in result
        assert (
            "checkStatus((/* CUDA driver graph node get tools id: "
            "node: node, output: toolsNodeId */ CUDA_SUCCESS))"
        ) in result
        assert (
            "(tools && ok) ? (/* CUDA driver graph exec set node params: "
            "exec: exec, node: node, params: (&params) */ CUDA_SUCCESS) : status"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuGraphNodeGetParams",
            "cuGraphNodeSetParams",
            "cuGraphExecNodeSetParams",
            "cuGraphDebugDotPrint",
            "cuGraphExecGetFlags",
            "cuGraphExecGetId",
            "cuGraphGetId",
            "cuGraphGetEdges",
            "cuGraphGetNodes",
            "cuGraphGetRootNodes",
            "cuGraphNodeGetContainingGraph",
            "cuGraphNodeGetDependencies",
            "cuGraphNodeGetDependentNodes",
            "cuGraphNodeGetEnabled",
            "cuGraphNodeSetEnabled",
            "cuGraphNodeGetLocalId",
            "cuGraphNodeGetToolsId",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_graph_conditional_instantiate_runtime_conversion(self):
        code = """
        bool checkStatus(CUresult status) {
            return status == CUDA_SUCCESS;
        }

        CUresult buildDriverConditionalGraph(
            CUgraph graph,
            CUgraphExec exec,
            const CUgraphNode* deps,
            const CUgraphEdgeData* edgeData,
            size_t count,
            CUcontext context
        ) {
            CUgraphConditionalHandle handle;
            CUgraphNode conditionalNode;
            CUgraphNodeParams params;
            CUDA_GRAPH_INSTANTIATE_PARAMS instantiateParams;

            CUresult status = cuGraphConditionalHandleCreate(
                &handle,
                graph,
                context,
                1u,
                CU_GRAPH_COND_ASSIGN_DEFAULT
            );
            status = cuGraphInstantiateWithParams(
                &exec,
                graph,
                &instantiateParams
            );
            status = cuGraphAddNode(
                &conditionalNode,
                graph,
                deps,
                edgeData,
                count,
                &params
            );

            if (cuGraphConditionalHandleCreate(&handle, graph, context, 0u, 0) != CUDA_SUCCESS) {
                return status;
            }

            bool ok = checkStatus(cuGraphInstantiateWithParams(
                &exec,
                graph,
                &instantiateParams
            ));
            return ok ? cuGraphAddNode(
                &conditionalNode,
                graph,
                deps,
                edgeData,
                count,
                &params
            ) : status;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver graph conditional handle create: output: handle, "
            "graph: graph, context: context, default launch value: 1u, "
            "flags: CU_GRAPH_COND_ASSIGN_DEFAULT"
        ) in result
        assert (
            "// CUDA driver graph instantiate with params: output: exec, "
            "graph: graph, params: (&instantiateParams)"
        ) in result
        assert (
            "// CUDA driver graph add generic node: output: conditionalNode, "
            "graph: graph, dependencies: deps, edge data: edgeData, "
            "dependency count: count, params: (&params)"
        ) in result
        assert (
            "if (((/* CUDA driver graph conditional handle create: "
            "output: handle, graph: graph, context: context, "
            "default launch value: 0u, flags: 0 */ CUDA_SUCCESS) != CUDA_SUCCESS))"
        ) in result
        assert (
            "checkStatus((/* CUDA driver graph instantiate with params: "
            "output: exec, graph: graph, params: (&instantiateParams) */ "
            "CUDA_SUCCESS))"
        ) in result
        assert (
            "ok ? (/* CUDA driver graph add generic node: output: conditionalNode, "
            "graph: graph, dependencies: deps, edge data: edgeData, "
            "dependency count: count, params: (&params) */ CUDA_SUCCESS) : status"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuGraphConditionalHandleCreate",
            "cuGraphInstantiateWithParams",
            "cuGraphAddNode",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_driver_graph_specialized_node_params_runtime_conversion(self):
        code = """
        bool checkStatus(CUresult status) {
            return status == CUDA_SUCCESS;
        }

        CUresult updateDriverGraphNodes(
            CUgraph graph,
            CUgraphExec exec,
            CUgraphNode kernelNode,
            CUgraphNode memcpyNode,
            CUgraphNode memsetNode,
            CUgraphNode hostNode,
            CUgraphNode targetNode,
            CUcontext context,
            bool ok
        ) {
            CUDA_KERNEL_NODE_PARAMS kernelParams;
            CUDA_MEMCPY3D copyParams;
            CUDA_MEMSET_NODE_PARAMS memsetParams;
            CUDA_HOST_NODE_PARAMS hostParams;
            CUkernelNodeAttrValue attrValue;
            CUgraphNode errorNode;
            CUgraphExecUpdateResult updateResult;
            CUgraphExecUpdateResultInfo resultInfo;

            CUresult status = cuGraphKernelNodeGetParams(kernelNode, &kernelParams);
            status = cuGraphKernelNodeSetParams(kernelNode, &kernelParams);
            status = cuGraphMemcpyNodeGetParams(memcpyNode, &copyParams);
            status = cuGraphMemcpyNodeSetParams(memcpyNode, &copyParams);
            status = cuGraphMemsetNodeGetParams(memsetNode, &memsetParams);
            status = cuGraphMemsetNodeSetParams(memsetNode, &memsetParams);
            status = cuGraphHostNodeGetParams(hostNode, &hostParams);
            cuGraphHostNodeSetParams(hostNode, &hostParams);
            status = cuGraphExecUpdate(exec, graph, &errorNode, &updateResult);
            status = cuGraphExecUpdate_v2(exec, graph, &resultInfo);
            status = cuGraphExecKernelNodeSetParams(
                exec,
                kernelNode,
                &kernelParams
            );
            status = cuGraphExecMemcpyNodeSetParams(
                exec,
                memcpyNode,
                &copyParams,
                context
            );
            status = cuGraphExecMemsetNodeSetParams(
                exec,
                memsetNode,
                &memsetParams,
                context
            );
            status = cuGraphExecHostNodeSetParams(exec, hostNode, &hostParams);
            status = cuGraphKernelNodeSetAttribute(
                kernelNode,
                CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                &attrValue
            );

            if (
                cuGraphKernelNodeGetAttribute(
                    kernelNode,
                    CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                    &attrValue
                ) != CUDA_SUCCESS
            ) {
                return status;
            }

            if (
                cuGraphExecMemcpyNodeSetParams(
                    exec,
                    memcpyNode,
                    &copyParams,
                    context
                ) != CUDA_SUCCESS
            ) {
                return status;
            }

            bool memsetUpdated = checkStatus(
                cuGraphExecMemsetNodeSetParams(
                    exec,
                    memsetNode,
                    &memsetParams,
                    context
                )
            );
            bool copied = checkStatus(
                cuGraphKernelNodeCopyAttributes(kernelNode, targetNode)
            );
            return copied && memsetUpdated && ok
                ? cuGraphExecMemcpyNodeSetParams(
                    exec,
                    memcpyNode,
                    &copyParams,
                    context
                )
                : status;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert (
            "// CUDA driver graph kernel node get params: node: kernelNode, "
            "params: (&kernelParams)"
        ) in result
        assert (
            "// CUDA driver graph kernel node set params: node: kernelNode, "
            "params: (&kernelParams)"
        ) in result
        assert (
            "// CUDA driver graph memcpy node get params: node: memcpyNode, "
            "params: (&copyParams)"
        ) in result
        assert (
            "// CUDA driver graph memcpy node set params: node: memcpyNode, "
            "params: (&copyParams)"
        ) in result
        assert (
            "// CUDA driver graph memset node get params: node: memsetNode, "
            "params: (&memsetParams)"
        ) in result
        assert (
            "// CUDA driver graph memset node set params: node: memsetNode, "
            "params: (&memsetParams)"
        ) in result
        assert (
            "// CUDA driver graph host node get params: node: hostNode, "
            "params: (&hostParams)"
        ) in result
        assert (
            "// CUDA driver graph host node set params: node: hostNode, "
            "params: (&hostParams)"
        ) in result
        assert (
            "// CUDA driver graph exec update: exec: exec, graph: graph, "
            "error node output: errorNode, result output: updateResult"
        ) in result
        assert (
            "// CUDA driver graph exec update v2: exec: exec, graph: graph, "
            "result info output: resultInfo"
        ) in result
        assert (
            "// CUDA driver graph exec set kernel node params: exec: exec, "
            "node: kernelNode, params: (&kernelParams)"
        ) in result
        assert (
            "// CUDA driver graph exec set memcpy node params: exec: exec, "
            "node: memcpyNode, params: (&copyParams), context: context"
        ) in result
        assert (
            "// CUDA driver graph exec set memset node params: exec: exec, "
            "node: memsetNode, params: (&memsetParams), context: context"
        ) in result
        assert (
            "// CUDA driver graph exec set host node params: exec: exec, "
            "node: hostNode, params: (&hostParams)"
        ) in result
        assert (
            "// CUDA driver graph kernel node set attribute: node: kernelNode, "
            "attribute: CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW, "
            "value: (&attrValue)"
        ) in result
        assert (
            "if (((/* CUDA driver graph kernel node get attribute: "
            "node: kernelNode, "
            "attribute: CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW, "
            "output: attrValue */ CUDA_SUCCESS) != CUDA_SUCCESS))"
        ) in result
        assert (
            "checkStatus((/* CUDA driver graph kernel node copy attributes: "
            "source: kernelNode, destination: targetNode */ CUDA_SUCCESS))"
        ) in result
        assert (
            "if (((/* CUDA driver graph exec set memcpy node params: "
            "exec: exec, node: memcpyNode, params: (&copyParams), "
            "context: context */ CUDA_SUCCESS) != CUDA_SUCCESS))"
        ) in result
        assert (
            "checkStatus((/* CUDA driver graph exec set memset node params: "
            "exec: exec, node: memsetNode, params: (&memsetParams), "
            "context: context */ CUDA_SUCCESS))"
        ) in result
        assert (
            "((copied && memsetUpdated) && ok) ? "
            "(/* CUDA driver graph exec set memcpy node params: exec: exec, "
            "node: memcpyNode, params: (&copyParams), context: context */ "
            "CUDA_SUCCESS) : status"
        ) in result
        assert "var status: CUresult = CUDA_SUCCESS;" in result
        assert "status = CUDA_SUCCESS;" in result
        for function_name in [
            "cuGraphKernelNodeGetParams",
            "cuGraphKernelNodeSetParams",
            "cuGraphMemcpyNodeGetParams",
            "cuGraphMemcpyNodeSetParams",
            "cuGraphMemsetNodeGetParams",
            "cuGraphMemsetNodeSetParams",
            "cuGraphHostNodeGetParams",
            "cuGraphHostNodeSetParams",
            "cuGraphExecUpdate",
            "cuGraphExecUpdate_v2",
            "cuGraphExecKernelNodeSetParams",
            "cuGraphExecMemcpyNodeSetParams",
            "cuGraphExecMemsetNodeSetParams",
            "cuGraphExecHostNodeSetParams",
            "cuGraphKernelNodeGetAttribute",
            "cuGraphKernelNodeSetAttribute",
            "cuGraphKernelNodeCopyAttributes",
        ]:
            assert f"{function_name}(" not in result

    def test_cuda_runtime_event_api_conversion(self):
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

    def test_single_trailing_restrict_pointer_qualifier_conversion(self):
        code = """
        static __global__ void kernel(float2 *__restrict out,
                                      const int *__restrict indices) {
            out[threadIdx.x] = out[threadIdx.x];
        }
        void host(float* data) {
            float *__restrict a = data, *__restrict b = data;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "out: array<vec2<f32> >" in result
        assert "indices: array<i32>" in result
        assert "var a: ptr<f32> = data;" in result
        assert "var b: ptr<f32> = data;" in result
        assert "__restrict" not in result

    def test_rvalue_reference_declarations_conversion(self):
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

    def test_cuda_texture_gather_helpers_convert(self):
        code = """
        void gatherOps(
            texture<float4, cudaTextureType2D> tex,
            cudaTextureObject_t objectTex,
            bool* resident,
            float2 uv
        ) {
            float4 gathered = tex2Dgather<float4>(tex, uv.x, uv.y);
            float4 gatheredComponent = tex2Dgather<float4>(
                tex,
                uv.x,
                uv.y,
                2
            );
            float4 gatheredObject = tex2Dgather<float4>(
                objectTex,
                uv.x,
                uv.y,
                1
            );
            float4 sparseGather = tex2Dgather<float4>(
                objectTex,
                uv.x,
                uv.y,
                resident,
                3
            );
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        result = CudaToCrossGLConverter().generate(ast)

        assert "sampler2D tex" in result
        assert "sampler2D objectTex" in result
        assert (
            "var gathered: vec4<f32> = textureGather("
            "tex, vec2<f32>(uv.x, uv.y));" in result
        )
        assert (
            "var gatheredComponent: vec4<f32> = textureGather("
            "tex, vec2<f32>(uv.x, uv.y), 2);" in result
        )
        assert (
            "var gatheredObject: vec4<f32> = textureGather("
            "objectTex, vec2<f32>(uv.x, uv.y), 1);" in result
        )
        assert (
            "var sparseGather: vec4<f32> = "
            "(/* cuda texture.tex2Dgather sparse residency not directly "
            "supported in CrossGL */ vec4<f32>(0.0, 0.0, 0.0, 0.0));" in result
        )
        assert "tex2Dgather<float4>(" not in result

    def test_cuda_texture_fetch_helpers_convert(self):
        code = """
        void fetchOps(
            texture<float4, cudaTextureType1D> lineTex,
            cudaTextureObject_t objectTex,
            bool* resident,
            int x
        ) {
            float4 fetched = tex1Dfetch<float4>(lineTex, x);
            float4 objectFetched = tex1Dfetch<float4>(objectTex, x);
            float4 sparseFetched = tex1Dfetch<float4>(objectTex, x, resident);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        result = CudaToCrossGLConverter().generate(ast)

        assert "sampler1D lineTex" in result
        assert "sampler1D objectTex" in result
        assert "var fetched: vec4<f32> = texelFetch(lineTex, x, 0);" in result
        assert "var objectFetched: vec4<f32> = texelFetch(objectTex, x, 0);" in result
        assert (
            "var sparseFetched: vec4<f32> = "
            "(/* cuda texture.tex1Dfetch sparse residency not directly "
            "supported in CrossGL */ vec4<f32>(0.0, 0.0, 0.0, 0.0));" in result
        )
        assert "tex1Dfetch<float4>(" not in result

    def test_cuda_sparse_texture_fetch_helpers_emit_diagnostics(self):
        code = """
        void sparseFetchOps(
            cudaTextureObject_t objectTex,
            bool* resident,
            float2 uv,
            float lod
        ) {
            float4 sparseSample = tex2D<float4>(
                objectTex,
                uv.x,
                uv.y,
                resident
            );
            float4 sparseLod = tex2DLod<float4>(
                objectTex,
                uv.x,
                uv.y,
                lod,
                resident
            );
            float4 sparseGrad = tex2DGrad<float4>(
                objectTex,
                uv.x,
                uv.y,
                uv,
                uv,
                resident
            );
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        result = CudaToCrossGLConverter().generate(ast)

        assert "sampler2D objectTex" in result
        for name, helper in (
            ("sparseSample", "tex2D"),
            ("sparseLod", "tex2DLod"),
            ("sparseGrad", "tex2DGrad"),
        ):
            assert (
                f"var {name}: vec4<f32> = "
                f"(/* cuda texture.{helper} sparse residency not directly "
                "supported in CrossGL */ vec4<f32>(0.0, 0.0, 0.0, 0.0));"
            ) in result
        assert "tex2D<float4>(" not in result
        assert "tex2DLod<float4>(" not in result
        assert "tex2DGrad<float4>(" not in result

    def test_cuda_sparse_layered_texture_helpers_emit_diagnostics(self):
        code = """
        void sparseLayeredFetchOps(
            cudaTextureObject_t objectLayers,
            bool* resident,
            float2 uv,
            int layer,
            float lod
        ) {
            float4 sparseLayer = tex2DLayered<float4>(
                objectLayers,
                uv.x,
                uv.y,
                layer,
                resident
            );
            float4 sparseLayerLod = tex2DLayeredLod<float4>(
                objectLayers,
                uv.x,
                uv.y,
                layer,
                lod,
                resident
            );
            float4 sparseLayerGrad = tex2DLayeredGrad<float4>(
                objectLayers,
                uv.x,
                uv.y,
                layer,
                uv,
                uv,
                resident
            );
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        result = CudaToCrossGLConverter().generate(ast)

        assert "sampler2DArray objectLayers" in result
        for name, helper in (
            ("sparseLayer", "tex2DLayered"),
            ("sparseLayerLod", "tex2DLayeredLod"),
            ("sparseLayerGrad", "tex2DLayeredGrad"),
        ):
            assert (
                f"var {name}: vec4<f32> = "
                f"(/* cuda texture.{helper} sparse residency not directly "
                "supported in CrossGL */ vec4<f32>(0.0, 0.0, 0.0, 0.0));"
            ) in result
        assert "tex2DLayered<float4>(" not in result
        assert "tex2DLayeredLod<float4>(" not in result
        assert "tex2DLayeredGrad<float4>(" not in result

    def test_cuda_sparse_3d_texture_helpers_emit_diagnostics(self):
        code = """
        void sparseVolumeFetchOps(
            cudaTextureObject_t objectVolume,
            bool* resident,
            float3 uvw,
            float lod
        ) {
            float4 sparseVolume = tex3D<float4>(
                objectVolume,
                uvw.x,
                uvw.y,
                uvw.z,
                resident
            );
            float4 sparseVolumeLod = tex3DLod<float4>(
                objectVolume,
                uvw.x,
                uvw.y,
                uvw.z,
                lod,
                resident
            );
            float4 sparseVolumeGrad = tex3DGrad<float4>(
                objectVolume,
                uvw.x,
                uvw.y,
                uvw.z,
                uvw,
                uvw,
                resident
            );
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        result = CudaToCrossGLConverter().generate(ast)

        assert "sampler3D objectVolume" in result
        for name, helper in (
            ("sparseVolume", "tex3D"),
            ("sparseVolumeLod", "tex3DLod"),
            ("sparseVolumeGrad", "tex3DGrad"),
        ):
            assert (
                f"var {name}: vec4<f32> = "
                f"(/* cuda texture.{helper} sparse residency not directly "
                "supported in CrossGL */ vec4<f32>(0.0, 0.0, 0.0, 0.0));"
            ) in result
        assert "tex3D<float4>(" not in result
        assert "tex3DLod<float4>(" not in result
        assert "tex3DGrad<float4>(" not in result

    def test_cuda_sparse_1d_and_cube_texture_helpers_emit_diagnostics(self):
        code = """
        void sparseLineAndCubeFetchOps(
            cudaTextureObject_t objectLine,
            cudaTextureObject_t objectLineLayers,
            cudaTextureObject_t objectCube,
            cudaTextureObject_t objectCubeLayers,
            bool* resident,
            float x,
            int layer,
            float3 dir,
            float lod
        ) {
            float4 sparseLine = tex1D<float4>(objectLine, x, resident);
            float4 sparseLineLod = tex1DLod<float4>(
                objectLine,
                x,
                lod,
                resident
            );
            float4 sparseLineGrad = tex1DGrad<float4>(
                objectLine,
                x,
                x,
                x,
                resident
            );
            float4 sparseLineLayer = tex1DLayered<float4>(
                objectLineLayers,
                x,
                layer,
                resident
            );
            float4 sparseLineLayerLod = tex1DLayeredLod<float4>(
                objectLineLayers,
                x,
                layer,
                lod,
                resident
            );
            float4 sparseLineLayerGrad = tex1DLayeredGrad<float4>(
                objectLineLayers,
                x,
                layer,
                x,
                x,
                resident
            );
            float4 sparseCube = texCubemap<float4>(
                objectCube,
                dir.x,
                dir.y,
                dir.z,
                resident
            );
            float4 sparseCubeLod = texCubemapLod<float4>(
                objectCube,
                dir.x,
                dir.y,
                dir.z,
                lod,
                resident
            );
            float4 sparseCubeGrad = texCubemapGrad<float4>(
                objectCube,
                dir.x,
                dir.y,
                dir.z,
                dir,
                dir,
                resident
            );
            float4 sparseCubeLayer = texCubemapLayered<float4>(
                objectCubeLayers,
                dir.x,
                dir.y,
                dir.z,
                layer,
                resident
            );
            float4 sparseCubeLayerLod = texCubemapLayeredLod<float4>(
                objectCubeLayers,
                dir.x,
                dir.y,
                dir.z,
                layer,
                lod,
                resident
            );
            float4 sparseCubeLayerGrad = texCubemapLayeredGrad<float4>(
                objectCubeLayers,
                dir.x,
                dir.y,
                dir.z,
                layer,
                dir,
                dir,
                resident
            );
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        result = CudaToCrossGLConverter().generate(ast)

        assert "sampler1D objectLine" in result
        assert "sampler1DArray objectLineLayers" in result
        assert "samplerCube objectCube" in result
        assert "samplerCubeArray objectCubeLayers" in result
        for name, helper in (
            ("sparseLine", "tex1D"),
            ("sparseLineLod", "tex1DLod"),
            ("sparseLineGrad", "tex1DGrad"),
            ("sparseLineLayer", "tex1DLayered"),
            ("sparseLineLayerLod", "tex1DLayeredLod"),
            ("sparseLineLayerGrad", "tex1DLayeredGrad"),
            ("sparseCube", "texCubemap"),
            ("sparseCubeLod", "texCubemapLod"),
            ("sparseCubeGrad", "texCubemapGrad"),
            ("sparseCubeLayer", "texCubemapLayered"),
            ("sparseCubeLayerLod", "texCubemapLayeredLod"),
            ("sparseCubeLayerGrad", "texCubemapLayeredGrad"),
        ):
            assert (
                f"var {name}: vec4<f32> = "
                f"(/* cuda texture.{helper} sparse residency not directly "
                "supported in CrossGL */ vec4<f32>(0.0, 0.0, 0.0, 0.0));"
            ) in result
            assert f"{helper}<float4>(" not in result

    def test_qualified_resource_object_pointer_array_conversion(self):
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

    def test_cub_dependent_shared_temp_storage_conversion(self):
        code = """
        using WarpReduce = cub::WarpReduce<int>;

        __global__ void kernel(int* out, int value) {
            __shared__ typename WarpReduce::TempStorage temp_storage[4];
            out[threadIdx.x] = value;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "typedef cub::WarpReduce<int> WarpReduce;" in result
        assert (
            "var<workgroup> temp_storage: array<WarpReduce::TempStorage, 4>;" in result
        )
        assert "typename" not in result

    def test_typedef_multi_declarator_alias_conversion(self):
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
        code = ""
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "// CUDA to CrossGL conversion" in result

    def test_fixed_array_initializer_conversion(self):
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

    def test_character_literal_conversion(self):
        code = r"""
        char helper() {
            char c = 'x';
            char escaped = '\n';
            char hex = '\x7f';
            char oct = '\377';
            return c;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        codegen = CudaToCrossGLConverter()
        result = codegen.generate(ast)

        assert "var c: i8 = 'x';" in result
        assert "var escaped: i8 = '\\n';" in result
        assert "var hex: i8 = '\\x7f';" in result
        assert "var oct: i8 = '\\377';" in result
        assert "return c;" in result

    def test_long_long_type_conversion(self):
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
