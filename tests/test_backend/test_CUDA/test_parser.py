import pytest

from crosstl.backend.CUDA.CudaAst import (
    ArrayAccessNode,
    AssignmentNode,
    AtomicOperationNode,
    BinaryOpNode,
    CastNode,
    ConstantMemoryNode,
    CudaAsmNode,
    CudaBuiltinNode,
    DeleteNode,
    DesignatedInitializerNode,
    DoWhileNode,
    EnumNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    InitializerListNode,
    KernelLaunchNode,
    KernelNode,
    MemberAccessNode,
    NewNode,
    PreprocessorNode,
    RangeForNode,
    ReturnNode,
    ShaderNode,
    SharedMemoryNode,
    StructNode,
    SwitchNode,
    SyncNode,
    TernaryOpNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
)
from crosstl.backend.CUDA.CudaLexer import CudaLexer
from crosstl.backend.CUDA.CudaParser import CudaParser


class TestCudaParser:
    def test_simple_kernel_parsing(self):
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

        assert isinstance(ast, ShaderNode)
        assert len(ast.kernels) == 1
        assert ast.kernels[0].name == "simple_kernel"

    def test_block_preprocessor_pragmas_parsing(self):
        code = """
        __global__ void kernel(float* data) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                data[i] = data[i] + 1.0f;
            }
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        pragma = ast.kernels[0].body[0]
        assert isinstance(pragma, PreprocessorNode)
        assert pragma.directive == "other"
        assert pragma.content == "#pragma unroll"

    def test_adjacent_string_literal_arguments_parsing(self):
        code = r"""
        void host() {
            printf("first "
                   "second\n");
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        call = ast.functions[0].body[0]
        assert isinstance(call, FunctionCallNode)
        assert call.args == ['"first second\\n"']

    def test_empty_statements_in_public_sample_patterns_are_skipped(self):
        code = """
        void host() {
            int value = 0;
            value = value + 1;;
            if (value > 0) {
                value = value - 1;
            };
            for (value = 0; value < 4; value++);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        body = ast.functions[0].body

        assert len(body) == 4
        assert isinstance(body[0], VariableNode)
        assert isinstance(body[1], AssignmentNode)
        assert isinstance(body[2], IfNode)
        assert isinstance(body[3], ForNode)
        assert body[3].body is None

    def test_device_function_parsing(self):
        code = """
        __device__ float add(float a, float b) {
            return a + b;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert isinstance(ast, ShaderNode)
        assert len(ast.functions) == 1
        assert ast.functions[0].name == "add"
        assert "__device__" in ast.functions[0].qualifiers

    def test_function_parameter_defaults_are_skipped(self):
        code = """
        __global__ void shfl_scan_test(int *data,
                                       int width,
                                       int *partial_sums = NULL) {
            int lane_id = threadIdx.x % warpSize;
        }

        static int test(bool automaticLaunchConfig,
                        const int count = max(32, 1000000)) {
            return count;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        kernel_params = ast.kernels[0].params
        assert [(param.vtype, param.name) for param in kernel_params] == [
            ("int *", "data"),
            ("int", "width"),
            ("int *", "partial_sums"),
        ]

        function_params = ast.functions[0].params
        assert [(param.vtype, param.name) for param in function_params] == [
            ("bool", "automaticLaunchConfig"),
            ("const int", "count"),
        ]

    def test_builtin_named_struct_member_access_from_occupancy_sample(self):
        code = """
        static double reportPotentialOccupancy() {
            cudaDeviceProp prop;
            int activeWarps = 32 / prop.warpSize;
            int lane = threadIdx.x % warpSize;
            return (double)activeWarps / lane;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        function = ast.functions[0]
        active_warps = function.body[1]
        lane = function.body[2]

        assert isinstance(active_warps.value.right, MemberAccessNode)
        assert active_warps.value.right.object == "prop"
        assert active_warps.value.right.member == "warpSize"
        assert isinstance(lane.value.right, CudaBuiltinNode)
        assert lane.value.right.builtin_name == "warpSize"

    def test_public_cuda_samples_builtin_named_members_parse_as_fields(self):
        code = """
        void configure() {
            cudaGraphKernelNodeParams params;
            params.gridDim = dim3(2, 1, 1);
            params.blockDim = dim3(128, 1, 1);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        grid_assignment = ast.functions[0].body[1]
        block_assignment = ast.functions[0].body[2]

        assert isinstance(grid_assignment.left, MemberAccessNode)
        assert grid_assignment.left.object == "params"
        assert grid_assignment.left.member == "gridDim"
        assert isinstance(block_assignment.left, MemberAccessNode)
        assert block_assignment.left.object == "params"
        assert block_assignment.left.member == "blockDim"

    def test_public_cuda_samples_const_dim3_constructor_global_parses(self):
        code = """
        const dim3 windowSize(512, 512);
        const dim3 windowBlockSize(16, 16, 1);
        const dim3 windowGridSize(windowSize.x / windowBlockSize.x,
                                  windowSize.y / windowBlockSize.y);
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        window_size = ast.global_variables[0]

        assert isinstance(window_size, VariableNode)
        assert window_size.vtype == "const dim3"
        assert window_size.name == "windowSize"
        assert isinstance(window_size.value, FunctionCallNode)
        assert window_size.value.name == "dim3"
        assert window_size.value.args == ["512", "512"]

        window_grid_size = ast.global_variables[2]
        assert isinstance(window_grid_size, VariableNode)
        assert window_grid_size.name == "windowGridSize"
        assert isinstance(window_grid_size.value, FunctionCallNode)
        assert window_grid_size.value.name == "dim3"
        assert len(window_grid_size.value.args) == 2

    def test_public_cuda_samples_unnamed_parameters_and_keyword_names(self):
        code = """
        struct Ray { float x; };
        __global__ void computeAngles_kernel(const Ray, float*, cudaTextureObject_t);
        __global__ void cuda_kernel_texture_2d(unsigned char* surface,
                                               int width,
                                               cudaExternalMemory_t& externalMemory);
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        first_params = ast.kernels[0].params
        assert [(param.vtype, param.name) for param in first_params] == [
            ("const Ray", ""),
            ("float *", ""),
            ("cudaTextureObject_t", ""),
        ]

        second_params = ast.kernels[1].params
        assert [(param.vtype, param.name) for param in second_params] == [
            ("unsigned char *", "surface"),
            ("int", "width"),
            ("cudaExternalMemory_t &", "externalMemory"),
        ]

    def test_public_cuda_samples_comma_expression_statements_parse_separately(self):
        code = """
        void init(int* I, int* J) {
            I[0] = 0, J[0] = 0, J[1] = 1;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        body = ast.functions[0].body
        assert len(body) == 3
        assert all(isinstance(stmt, AssignmentNode) for stmt in body)
        assert body[0].left.array == "I"
        assert body[1].left.array == "J"
        assert body[2].left.array == "J"

    def test_parenthesized_comma_expression_statement_from_hpc_training(self):
        code = """
        void swap_arrays(float* x, float* xnew) {
            float* xtmp;
            (xtmp = xnew, xnew = x, x = xtmp);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        body = ast.functions[0].body
        assert len(body) == 4
        assert all(isinstance(stmt, AssignmentNode) for stmt in body[1:])
        assert body[1].left == "xtmp"
        assert body[2].left == "xnew"
        assert body[3].left == "x"

    def test_public_cuda_samples_macro_expanded_empty_call_arguments_are_skipped(self):
        code = """
        void report() {
            fprintf(stderr,
                    "CUDART version %d.%d does not support "
                    "<cudaDeviceProp.canMapHostMemory> field\\n",
                    , CUDART_VERSION / 1000);
            __checkCudaErrors(, "", 319);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        fprintf_call = ast.functions[0].body[0]
        check_call = ast.functions[0].body[1]

        assert isinstance(fprintf_call, FunctionCallNode)
        assert len(fprintf_call.args) == 3
        assert fprintf_call.args[0] == "stderr"
        assert isinstance(fprintf_call.args[2], BinaryOpNode)
        assert isinstance(check_call, FunctionCallNode)
        assert check_call.args == ['""', "319"]

    def test_public_cuda_samples_out_of_class_template_constructor_is_skipped(self):
        code = """
        template <typename Real>
        PiEstimator<Real>::PiEstimator(unsigned int numSims,
                                       unsigned int device,
                                       unsigned int threadBlockSize,
                                       unsigned int seed)
            : m_numSims(numSims),
              m_device(device),
              m_threadBlockSize(threadBlockSize),
              m_seed(seed) {
            initialize();
        }

        __global__ void estimate(float* output) {
            output[0] = 0.0f;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert len(ast.kernels) == 1
        assert ast.kernels[0].name == "estimate"
        assert ast.global_variables == []

    def test_public_cuda_samples_specifier_prefixed_template_constructor_is_skipped(
        self,
    ):
        code = """
        template <class T>
        class interval_gpu {
        public:
            __device__ __host__ interval_gpu();
        };

        template <class T>
        inline __device__ __host__ interval_gpu<T>::interval_gpu() {
        }

        void host() {
            int value = 1;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert len(ast.structs) == 1
        assert ast.structs[0].name == "interval_gpu"
        assert ast.functions[0].name == "host"

    def test_public_cuda_samples_return_type_scoped_template_member_is_skipped(
        self,
    ):
        code = """
        template <class T>
        class interval_gpu {
        public:
            __device__ __host__ T const& lower() const;
        };

        template <class T>
        inline __device__ __host__ T const& interval_gpu<T>::lower() const {
            return low;
        }

        void host() {
            int value = 1;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert len(ast.structs) == 1
        assert ast.structs[0].name == "interval_gpu"
        assert ast.functions[0].name == "host"

    def test_public_cuda_samples_virtual_operator_members_are_skipped(self):
        code = """
        template <typename T>
        struct SharedMemory {
            virtual __device__ T& operator*() = 0;
            virtual __device__ T& operator[](int i) = 0;
        };

        __global__ void kernel(float* output) {
            output[threadIdx.x] = 0.0f;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert len(ast.structs) == 1
        assert ast.structs[0].name == "SharedMemory"
        assert ast.structs[0].members == []
        assert ast.kernels[0].name == "kernel"

    def test_public_cccl_templated_attributed_functor_member_is_skipped(self):
        code = """
        template <class T>
        struct plus_one
        {
          template <class U>
          [[nodiscard]] _CCCL_API constexpr T operator()(const U val) const noexcept
          {
            return static_cast<T>(val + 1);
          }
        };

        template <typename T>
        static void unary(nvbench::state& state, nvbench::type_list<T>)
        {
          plus_one<T> op;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert len(ast.structs) == 1
        assert ast.structs[0].name == "plus_one"
        assert ast.structs[0].members == []
        assert ast.functions[0].name == "unary"
        assert ast.functions[0].body[0].vtype == "plus_one<T>"
        assert ast.functions[0].body[0].name == "op"

    def test_public_cccl_maybe_unused_parameters_parse(self):
        code = """
        int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
        {
          stackable_ctx ctx;
          return argc;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        main = ast.functions[0]
        assert main.name == "main"
        assert [(param.vtype, param.name) for param in main.params] == [
            ("int", "argc"),
            ("char * *", "argv"),
        ]
        assert main.body[0].vtype == "stackable_ctx"

    def test_public_cuda_template_disambiguator_calls_parse(self):
        # Inspired by LiterateLB CUDA template kernels using
        # OPERATOR::template apply<T>(...) inside __global__ wrappers.
        code = """
        template <typename OPERATOR, typename F>
        __global__ void call_operator(F f) {
            OPERATOR::template apply<float>(threadIdx.x);
            f.template operator()<float>(threadIdx.x);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        scope_call = ast.kernels[0].body[0]
        member_call = ast.kernels[0].body[1]

        assert isinstance(scope_call, FunctionCallNode)
        assert scope_call.name == "OPERATOR::apply<float>"
        assert isinstance(member_call, FunctionCallNode)
        assert isinstance(member_call.name, MemberAccessNode)
        assert member_call.name.object == "f"
        assert member_call.name.member == "operator()<float>"

    def test_public_cub_macro_test_blocks_are_skipped_before_kernels(self):
        code = """
        C2H_TEST("DeviceAdjacentDifference::SubtractRight works with iterators",
                 "[device][adjacent_difference]",
                 types)
        {
          using type = typename c2h::get<0, TestType>;
        }

        __global__ void kernel(int* out) {
            out[threadIdx.x] = 0;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert [kernel.name for kernel in ast.kernels] == ["kernel"]

    def test_public_cuda_samples_braced_template_temporaries_parse_as_calls(self):
        code = """
        __global__ void mdspan_kernel(int* smem_storage, __half* X, int idx) {
            cuda::shared_memory_mdspan smem(
                smem_storage,
                cuda::std::dextents<cuda::std::size_t, 2>{8, 8}
            );
            X[idx] = __half{float(idx) - 3.5f};
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        smem = ast.kernels[0].body[0]
        assert smem.vtype == "cuda::shared_memory_mdspan"
        assert isinstance(smem.value.args[1], FunctionCallNode)
        assert smem.value.args[1].name == "cuda::std::dextents<cuda::std::size_t, 2>"
        assert smem.value.args[1].args == ["8", "8"]

        assignment = ast.kernels[0].body[1]
        assert isinstance(assignment.right, FunctionCallNode)
        assert assignment.right.name == "__half"
        assert isinstance(assignment.right.args[0], BinaryOpNode)

    def test_public_cuda_samples_function_pointer_typedef_parsing(self):
        code = """
        typedef unsigned char (*blockFunction_t)(
            unsigned char,
            unsigned char,
            unsigned char
        );

        blockFunction_t choose();
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        alias = ast.typedefs[0]
        assert isinstance(alias, TypeAliasNode)
        assert alias.name == "blockFunction_t"
        assert alias.alias_type == "unsigned char (*)"
        assert [(param.vtype, param.name) for param in alias.params] == [
            ("unsigned char", ""),
            ("unsigned char", ""),
            ("unsigned char", ""),
        ]
        assert ast.functions[0].return_type == "blockFunction_t"

    def test_public_cuda_samples_function_pointer_headers_before_kernels(self):
        code = """
        int setup(int (*combine)(int, int), int count);

        __device__ int (*select_block_function(int mode))(int) {
            return 0;
        }

        __global__ void function_pointer_kernel(int *out) {
            out[threadIdx.x] = 0;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert ast.functions[0].name == "setup"
        assert [(param.vtype, param.name) for param in ast.functions[0].params] == [
            ("int (*)", "combine"),
            ("int", "count"),
        ]
        assert [kernel.name for kernel in ast.kernels] == ["function_pointer_kernel"]

    def test_public_cuda_samples_elaborated_and_qualified_type_declarations(self):
        code = """
        void configure(int n) {
            struct cudaDeviceProp properties;
            const Real *pointx = points + n;
            std::size_t count = std::size_t(n) * 4;
            int bytes = sizeof(unsigned);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        body = ast.functions[0].body
        assert body[0].vtype == "struct cudaDeviceProp"
        assert body[0].name == "properties"
        assert body[1].vtype == "const Real *"
        assert body[1].name == "pointx"
        assert body[2].vtype == "std::size_t"
        assert isinstance(body[2].value.left, FunctionCallNode)
        assert body[2].value.left.name == "std::size_t"
        assert body[3].value.name == "sizeof"
        assert body[3].value.args == ["unsigned int"]

    def test_public_cuda_strided_access_throw_runtime_error_statements(self):
        code = """
        inline void cuda_last_error_check() {
            cudaError_t error_code = cudaGetLastError();

            if (cudaSuccess != error_code) {
                std::stringstream ss;
                ss << "CUDA Runtime API error " << error_code << ": "
                   << cudaGetErrorString(error_code) << std::endl;
                throw std::runtime_error(ss.str());
            }
        }

        int main() {
            cudaError_t err = cudaGetDeviceProperties(&prop, 0);
            if (err != cudaSuccess)
                throw std::runtime_error("Failed to get CUDA device name");
            return EXIT_SUCCESS;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        error_check_body = ast.functions[0].body
        braced_if = error_check_body[1]
        braced_throw = braced_if.if_body[2]

        assert isinstance(braced_if, IfNode)
        assert isinstance(braced_throw, FunctionCallNode)
        assert braced_throw.name == "throw"
        assert isinstance(braced_throw.args[0], FunctionCallNode)
        assert braced_throw.args[0].name == "std::runtime_error"

        main_body = ast.functions[1].body
        unbraced_throw = main_body[1].if_body
        assert isinstance(unbraced_throw, FunctionCallNode)
        assert unbraced_throw.name == "throw"
        assert unbraced_throw.args[0].name == "std::runtime_error"

    def test_public_cuda_samples_typedef_enum_aliases_and_forward_structs(self):
        code = """
        typedef enum memAllocType_enum {
            MEMALLOC_TYPE_START,
            USE_MANAGED_MEMORY = MEMALLOC_TYPE_START,
            MEMALLOC_TYPE_END = USE_MANAGED_MEMORY
        } MemAllocType;

        struct resultsData;

        MemAllocType currentType;
        struct resultsData *results;
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        enum = ast.structs[0]
        assert isinstance(enum, EnumNode)
        assert enum.name == "MemAllocType"
        assert enum.tag_name == "memAllocType_enum"
        assert ast.structs[1].name == "resultsData"
        assert ast.structs[1].members == []
        assert ast.global_variables[0].vtype == "MemAllocType"
        assert ast.global_variables[1].vtype == "struct resultsData *"

    def test_public_cuda_samples_templated_class_declarations(self):
        code = """
        template <typename Real> class PiEstimator
        {
        public:
            PiEstimator(unsigned int numSims, unsigned int device);
            Real operator()();

        private:
            unsigned int m_seed;
            unsigned int m_numSims;
        };
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        klass = ast.structs[0]
        assert isinstance(klass, StructNode)
        assert klass.name == "PiEstimator"
        assert [member.name for member in klass.members] == ["m_seed", "m_numSims"]
        assert [member.vtype for member in klass.members] == [
            "unsigned int",
            "unsigned int",
        ]

    def test_public_cuda_samples_qualified_c_style_cast_parsing(self):
        code = """
        constexpr int BATCH = 1;
        constexpr int HEADS = 8;
        constexpr std::size_t Q_SIZE = (std::size_t)BATCH * HEADS;
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        q_size = ast.global_variables[2]
        assert q_size.vtype == "std::size_t"
        assert isinstance(q_size.value, BinaryOpNode)
        assert isinstance(q_size.value.left, CastNode)
        assert q_size.value.left.target_type == "std::size_t"
        assert q_size.value.left.expression == "BATCH"

    def test_public_cuda_samples_local_anonymous_enum_declaration(self):
        code = """
        void configure() {
            enum {
                SHMEM_SZ = 16 * 256,
                BLOCKS = 8,
            };
            int size = SHMEM_SZ;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        enum = ast.functions[0].body[0]
        assert isinstance(enum, EnumNode)
        assert enum.name is None
        assert enum.members[0][0] == "SHMEM_SZ"
        assert isinstance(enum.members[0][1], BinaryOpNode)
        assert ast.functions[0].body[1].value == "SHMEM_SZ"

    def test_public_cuda_samples_goto_and_labels_are_skipped(self):
        code = """
        int host(int *p) {
            int rc = -1;
            if (p == NULL)
                goto Exit;
            rc = 0;
        Exit:
            rc += 1;
            return rc;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        body = ast.functions[0].body
        assert isinstance(body[1], IfNode)
        assert body[1].if_body is None
        assert isinstance(body[3], AssignmentNode)
        assert body[3].left == "rc"
        assert body[3].operator == "+="
        assert isinstance(body[4], ReturnNode)

    def test_public_cuda_samples_global_qualified_braced_temporaries(self):
        code = """
        void host(int ordinal) {
            s.device = ::DLDevice{::kDLCUDA, ordinal};
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assignment = ast.functions[0].body[0]
        assert isinstance(assignment.left, MemberAccessNode)
        assert isinstance(assignment.right, FunctionCallNode)
        assert assignment.right.name == "::DLDevice"
        assert assignment.right.args == ["::kDLCUDA", "ordinal"]

    def test_public_cuda_samples_device_global_variables_parse_as_globals(self):
        code = """
        __device__ int g_uids = 0;
        __device__ double grid_dot_result = 0.0;
        __device__ cufftCallbackLoadC myOwnCallbackPtr = ComplexPointwiseMulAndScale;

        __device__ float add(float a, float b) {
            return a + b;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert [var.name for var in ast.global_variables] == [
            "g_uids",
            "grid_dot_result",
            "myOwnCallbackPtr",
        ]
        assert [var.vtype for var in ast.global_variables] == [
            "int",
            "double",
            "cufftCallbackLoadC",
        ]
        assert all(var.qualifiers == ["__device__"] for var in ast.global_variables)
        assert len(ast.functions) == 1
        assert ast.functions[0].name == "add"

    def test_cuda_constant_memory_optional_device_qualifier_parse_as_globals(self):
        # CUDA C++ Programming Guide 7.2.2 documents __constant__ with optional
        # __device__ as a variable memory space specifier.
        code = """
        __device__ __constant__ float stencil[4];
        __constant__ __device__ int offsets[4];
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert [var.name for var in ast.global_variables] == ["stencil", "offsets"]
        assert [var.vtype for var in ast.global_variables] == ["float[4]", "int[4]"]
        assert all(isinstance(var, ConstantMemoryNode) for var in ast.global_variables)
        assert ast.global_variables[0].qualifiers == ["__device__", "__constant__"]
        assert ast.global_variables[1].qualifiers == ["__constant__", "__device__"]

    def test_fixed_width_pointer_declaration_lists_parse_as_variables(self):
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

        kernel = ast.kernels[0]
        d_local, d_shadow, assignment = kernel.body

        assert kernel.params[0].vtype == "uint32_t *"
        assert kernel.params[0].name == "d_output"
        assert kernel.params[1].vtype == "const uint32_t *"
        assert kernel.params[1].name == "d_input"
        assert isinstance(d_local, VariableNode)
        assert d_local.vtype == "uint32_t *"
        assert d_local.name == "d_local"
        assert isinstance(d_shadow, VariableNode)
        assert d_shadow.vtype == "uint32_t *"
        assert d_shadow.name == "d_shadow"
        assert isinstance(assignment, AssignmentNode)
        assert assignment.right.op == ">>"

    def test_enum_class_declarations_parse_before_functions(self):
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

        enum = ast.structs[0]
        function = ast.functions[0]

        assert isinstance(enum, EnumNode)
        assert enum.name == "MemoryMode"
        assert enum.members == [("PAGED", None), ("PINNED", None)]
        assert enum.underlying_type == "unsigned int"
        assert enum.is_scoped is True
        assert function.params[0].vtype == "const MemoryMode"
        assert function.params[0].name == "memory_mode"

    def test_void_parameter_list_and_bodyless_prototypes_parsing(self):
        code = """
        void runTest(int argc, char **argv);
        extern "C" bool computeGold(int *gpuData, const int len);

        int main(void) {
            const unsigned blocks = 512;
            return 0;
        }

        void runTest(int argc, char **argv) {
            return;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert ast.functions[0].name == "runTest"
        assert ast.functions[0].body is None
        assert ast.functions[1].name == "computeGold"
        assert ast.functions[1].body is None
        assert ast.functions[2].name == "main"
        assert ast.functions[2].params == []
        assert ast.functions[2].body[0].vtype == "const unsigned int"
        assert ast.functions[3].body is not None

    def test_public_cuda_samples_composite_scalar_type_parsing(self):
        code = """
        void host() {
            unsigned long int counter = 0;
            long double avgElapsedClocks = 0;
            short int lanes = 32;
            avgElapsedClocks += (long double)(counter);
            counter += sizeof(unsigned long int);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        body = ast.functions[0].body
        assert body[0].vtype == "unsigned long int"
        assert body[1].vtype == "long double"
        assert body[2].vtype == "short int"
        assert isinstance(body[3].right, CastNode)
        assert body[3].right.target_type == "long double"
        assert body[4].right.args == ["unsigned long int"]

    def test_public_cuda_samples_global_declaration_lists_parsing(self):
        code = """
        __device__ static unsigned int numErrors = 0, errorFound = 0;
        cudaEvent_t start, stop;
        cudaArray *d_array, *d_tempArray;
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert [(var.vtype, var.name, var.value) for var in ast.global_variables] == [
            ("unsigned int", "numErrors", "0"),
            ("unsigned int", "errorFound", "0"),
            ("cudaEvent_t", "start", None),
            ("cudaEvent_t", "stop", None),
            ("cudaArray *", "d_array", None),
            ("cudaArray *", "d_tempArray", None),
        ]
        assert ast.global_variables[0].qualifiers == ["__device__", "static"]
        assert ast.global_variables[1].qualifiers == ["__device__", "static"]

    def test_public_cuda_samples_unknown_pointer_return_types_and_reference_params(
        self,
    ):
        code = """
        extern "C" Volume *VolumeFilter_runFilter(Volume *input,
                                                  Volume *output,
                                                  int numWeights) {
            Volume *swap = 0;
            return input;
        }

        Vertex *cudaImportVertexBuffer(void *sharedHandle,
                                       cudaExternalMemory_t &externalMemory) {
            Vertex *cudaDevVertptr = NULL;
            return cudaDevVertptr;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        volume_filter = ast.functions[0]
        import_buffer = ast.functions[1]

        assert volume_filter.return_type == "Volume *"
        assert volume_filter.name == "VolumeFilter_runFilter"
        assert volume_filter.linkage == "C"
        assert [(param.vtype, param.name) for param in volume_filter.params] == [
            ("Volume *", "input"),
            ("Volume *", "output"),
            ("int", "numWeights"),
        ]
        assert volume_filter.body[0].vtype == "Volume *"
        assert volume_filter.body[0].name == "swap"

        assert import_buffer.return_type == "Vertex *"
        assert import_buffer.params[1].vtype == "cudaExternalMemory_t &"
        assert import_buffer.params[1].name == "externalMemory"
        assert import_buffer.body[0].vtype == "Vertex *"
        assert import_buffer.body[0].name == "cudaDevVertptr"

    def test_nvidia_helper_cuda_drvapi_post_return_inline_function_parsing(
        self,
    ):
        code = """
        bool inline findFatbinPath(const char *module_file,
                                   std::string &module_path,
                                   char **argv,
                                   std::ostringstream &ostrm) {
            return true;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        function = ast.functions[0]
        assert function.name == "findFatbinPath"
        assert function.return_type == "bool"
        assert function.qualifiers == ["inline"]
        assert [(param.vtype, param.name) for param in function.params] == [
            ("const char *", "module_file"),
            ("std::string &", "module_path"),
            ("char * *", "argv"),
            ("std::ostringstream &", "ostrm"),
        ]

    def test_public_cuda_samples_constructor_style_globals_are_not_functions(self):
        code = """
        int N = 1 << 22;
        dim3 block(512);
        dim3 grid;

        float processWithStreams(int streams_used);
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert [var.name for var in ast.global_variables] == ["N", "block", "grid"]
        assert ast.global_variables[1].vtype == "dim3"
        assert isinstance(ast.global_variables[1].value, FunctionCallNode)
        assert ast.global_variables[1].value.name == "dim3"
        assert ast.global_variables[1].value.args == ["512"]
        assert [function.name for function in ast.functions] == ["processWithStreams"]

    def test_public_cuda_samples_nested_struct_aggregates_and_friend_declarations(self):
        code = """
        struct BlockDXT1 {
            Color16 col0;
            Color16 col1;
            union {
                unsigned char row[4];
                unsigned int indices;
            };
            void decompress(Color32 colors[16]) const;
        };

        class Segmentation {
        private:
            class Level {
            public:
                Level(uint totalNodes) : nodes_(totalNodes) {}
            private:
                friend class Pyramid;
                vector<uint> nodes_;
            };
        };
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        block = ast.structs[0]
        assert [member.name for member in block.members] == [
            "col0",
            "col1",
            "row",
            "indices",
        ]
        assert block.members[2].vtype == "unsigned char[4]"
        assert block.members[3].vtype == "unsigned int"
        assert ast.structs[1].name == "Segmentation"

    def test_public_cuda_samples_function_pointer_locals_and_if_constexpr(self):
        code = """
        void host() {
            void (*kernel)(float *, float *, int, int);
            if constexpr (COMPUTE_MEAN_AND_RSTD) {
                kernel = copy;
            }
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        pointer = ast.functions[0].body[0]
        branch = ast.functions[0].body[1]

        assert isinstance(pointer, VariableNode)
        assert pointer.vtype == "void (*)"
        assert pointer.name == "kernel"
        assert [(param.vtype, param.name) for param in pointer.params] == [
            ("float *", ""),
            ("float *", ""),
            ("int", ""),
            ("int", ""),
        ]
        assert isinstance(branch, IfNode)
        assert branch.is_constexpr is True

    def test_public_cuda_samples_vulkan_callback_macros_and_try_catch(self):
        code = """
        struct App {
            static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
                const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
                void *pUserData) {
                return VK_FALSE;
            }
        };

        int main() {
            vulkanImageCUDA app;
            try {
                app.run();
            } catch (const std::exception &e) {
                return EXIT_FAILURE;
            }
            return 0;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        main = ast.functions[0]
        assert main.name == "main"
        assert main.body[0].vtype == "vulkanImageCUDA"
        assert isinstance(main.body[1], FunctionCallNode)
        assert isinstance(main.body[1].name, MemberAccessNode)
        assert main.body[1].name.member == "run"
        assert isinstance(main.body[2], ReturnNode)

    def test_public_cuda_samples_comma_initializer_and_uint_pointer_lists(self):
        code = """
        void host(uint *src, int start, int end) {
            uint *ikey, *ival, *okey, *oval;
            int i, sum = 0;

            for (sum = 0, i = start; i < end; i++) {
                sink(src[i]);
            }
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        params = ast.functions[0].params
        assert params[0].vtype == "uint *"

        body = ast.functions[0].body
        assert [var.vtype for var in body[:4]] == [
            "uint *",
            "uint *",
            "uint *",
            "uint *",
        ]
        assert [var.name for var in body[:4]] == ["ikey", "ival", "okey", "oval"]
        assert [var.name for var in body[4:6]] == ["i", "sum"]
        assert body[5].value == "0"

        loop = body[6]
        assert isinstance(loop, ForNode)
        assert isinstance(loop.init, list)
        assert len(loop.init) == 2
        assert loop.init[0].left == "sum"
        assert loop.init[1].left == "i"

    def test_public_cccl_decltype_types_parse_as_declarations_and_returns(self):
        code = """
        struct __static_bounds {
            [[nodiscard]] _CCCL_API static constexpr decltype(_Lower) lower() noexcept;
        };

        _CCCL_API constexpr decltype(_Lower) lower_value() noexcept {
            return _Lower;
        }

        _CCCL_HOST_API decltype(auto) adapt_resource(Resource&& resource) noexcept {
            return resource;
        }

        void shift_right(int __n, int __last, int __first) {
            decltype(__n) __d = __last - __first;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert ast.structs[0].name == "__static_bounds"
        assert ast.structs[0].members == []

        lower_value, adapt_resource, shift_right = ast.functions
        assert lower_value.return_type == "decltype(_Lower)"
        assert lower_value.name == "lower_value"
        assert lower_value.qualifiers == ["_CCCL_API", "constexpr"]

        assert adapt_resource.return_type == "decltype(auto)"
        assert adapt_resource.name == "adapt_resource"
        assert adapt_resource.params[0].vtype == "Resource &&"
        assert adapt_resource.params[0].name == "resource"

        decl = shift_right.body[0]
        assert isinstance(decl, VariableNode)
        assert decl.vtype == "decltype(__n)"
        assert decl.name == "__d"
        assert isinstance(decl.value, BinaryOpNode)
        assert decl.value.op == "-"

    def test_public_cuda_samples_decltype_lambda_parameters_parse(self):
        code = """
        void select_devices() {
            auto bestFit = std::make_pair(it, it);
            auto distance = [](decltype(bestFit) p) {
                return p;
            };
            auto access = [](decltype(*itr) mapPair) {
                return mapPair.second;
            };
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        distance = ast.functions[0].body[1]
        access = ast.functions[0].body[2]

        assert isinstance(distance.value, FunctionCallNode)
        assert distance.value.name == "lambda"
        assert distance.value.args[0].vtype == "decltype(bestFit)"
        assert distance.value.args[0].name == "p"

        assert isinstance(access.value, FunctionCallNode)
        assert access.value.args[0].vtype == "decltype(*itr)"
        assert access.value.args[0].name == "mapPair"

    def test_shared_memory_parsing(self):
        code = """
        __global__ void kernel() {
            __shared__ float shared_data[256];
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert isinstance(ast, ShaderNode)
        assert len(ast.kernels) == 1

    def test_dynamic_shared_memory_parsing_marks_extern_unsized_array(self):
        code = """
        __global__ void kernel() {
            extern __shared__ float shared[];
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        declaration = ast.kernels[0].body[0]
        assert isinstance(declaration, SharedMemoryNode)
        assert declaration.vtype == "float[]"
        assert declaration.is_extern_shared_memory is True
        assert declaration.is_dynamic_shared_memory is True

    def test_public_cuda_samples_shared_memory_qualifier_variants(self):
        code = """
        __global__ void kernel() {
            extern float __shared__ smem[];
            __shared__ alignas(alignof(float4)) float As[32];
            volatile __shared__ float scores[16];
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        extern_smem, aligned_smem, volatile_smem = ast.kernels[0].body
        assert isinstance(extern_smem, SharedMemoryNode)
        assert extern_smem.vtype == "float[]"
        assert extern_smem.is_extern_shared_memory is True
        assert extern_smem.is_dynamic_shared_memory is True
        assert isinstance(aligned_smem, SharedMemoryNode)
        assert aligned_smem.vtype == "float[32]"
        assert isinstance(volatile_smem, SharedMemoryNode)
        assert volatile_smem.vtype == "float[16]"

    def test_builtin_variables_parsing(self):
        code = """
        __global__ void kernel() {
            int x = threadIdx.x;
            int y = blockIdx.y;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert isinstance(ast, ShaderNode)
        assert len(ast.kernels) == 1

    def test_vector_types_parsing(self):
        code = """
        __global__ void kernel(float2* data) {
            float4 vec = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert isinstance(ast, ShaderNode)
        assert len(ast.kernels) == 1

    def test_fp16_pointer_array_declarations_parse_as_variables(self):
        code = """
        void host(size_t size) {
            half2 *vec[2];
            half2 *const devVec[2];
            checkCudaErrors(cudaMallocHost((void **)&vec[0],
                                           size * sizeof *vec[0]));
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        body = ast.functions[0].body
        assert isinstance(body[0], VariableNode)
        assert body[0].vtype == "half2 *[2]"
        assert body[0].name == "vec"
        assert isinstance(body[1], VariableNode)
        assert body[1].vtype == "half2 * const[2]"
        assert body[1].name == "devVec"
        assert isinstance(body[2], FunctionCallNode)

    def test_constructor_style_vector_declarations_parsing(self):
        code = """
        void launch() {
            dim3 grid(16, 8, 1);
            dim3 block(32);
            float3 v(1.0f, 2.0f, 3.0f);
            uint4 ids = make_uint4(1u, 2u, 3u, 4u);
            uchar2 bytes(1, 2);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        body = ast.functions[0].body
        assert body[0].vtype == "dim3"
        assert body[0].name == "grid"
        assert isinstance(body[0].value, FunctionCallNode)
        assert body[0].value.name == "dim3"
        assert body[0].value.args == ["16", "8", "1"]
        assert body[1].value.args == ["32"]
        assert body[2].vtype == "float3"
        assert body[2].value.name == "float3"
        assert body[3].vtype == "uint4"
        assert body[3].value.name == "make_uint4"
        assert body[4].vtype == "uchar2"
        assert body[4].value.name == "uchar2"

    def test_kernel_launch_parsing(self):
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

        launch = ast.functions[0].body[2]
        assert isinstance(launch, KernelLaunchNode)
        assert launch.kernel_name == "kernel"
        assert launch.blocks == "grid"
        assert launch.threads == "block"
        assert launch.shared_mem == "128"
        assert launch.stream == "stream"
        assert launch.args == ["data", "1"]

    def test_templated_kernel_launch_parsing(self):
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

        host = next(function for function in ast.functions if function.name == "host")
        launch = host.body[0]
        assert isinstance(launch, KernelLaunchNode)
        assert launch.kernel_name == "scale<float>"
        assert launch.blocks == "1"
        assert launch.threads == "32"
        assert launch.shared_mem is None
        assert launch.stream is None
        assert launch.args == ["data", "2.0f"]

    def test_single_value_kernel_launch_config_parsing(self):
        code = """
        void host(float* d_x) {
            tileKernel<<<1>>>(d_x);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        launch = ast.functions[0].body[0]
        assert isinstance(launch, KernelLaunchNode)
        assert launch.kernel_name == "tileKernel"
        assert launch.blocks == "1"
        assert launch.threads is None
        assert launch.shared_mem is None
        assert launch.stream is None
        assert launch.args == ["d_x"]

    def test_template_identifier_c_style_cast_parsing(self):
        code = """
        template <class T>
        __global__ void kernel(T* out, unsigned int num_threads) {
            out[threadIdx.x] = (T)num_threads;
            out = (T *)malloc(num_threads);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assignment = ast.kernels[0].body[0]
        assert isinstance(assignment.right, CastNode)
        assert assignment.right.target_type == "T"
        assert assignment.right.expression == "num_threads"
        allocation = ast.kernels[0].body[1]
        assert isinstance(allocation.right, CastNode)
        assert allocation.right.target_type == "T *"

    def test_computed_kernel_launch_config_parsing(self):
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

        launch = ast.functions[0].body[1]
        assert isinstance(launch, KernelLaunchNode)
        assert isinstance(launch.blocks, BinaryOpNode)
        assert launch.blocks.op == "/"
        assert launch.threads == "blockSize"
        assert isinstance(launch.shared_mem, BinaryOpNode)
        assert launch.shared_mem.op == "*"
        assert isinstance(launch.shared_mem.left, FunctionCallNode)
        assert launch.shared_mem.left.name == "sizeof"
        assert launch.shared_mem.left.args == ["float"]
        assert launch.stream == "stream"
        assert launch.args == ["data", "n"]

    def test_grid_dim_kernel_launch_config_expressions_parsing(self):
        code = """
        void host(float* data) {
            kernel<<<dim3(gridDim.x, gridDim.y, 1),
                     dim3(blockDim.x, blockDim.y, 1)>>>(data);
            kernel<<<gridDim.x + gridDim.y, blockDim.x * blockDim.y>>>(data);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        first_launch = ast.functions[0].body[0]
        second_launch = ast.functions[0].body[1]
        assert isinstance(first_launch, KernelLaunchNode)
        assert isinstance(first_launch.blocks, FunctionCallNode)
        assert first_launch.blocks.name == "dim3"
        assert isinstance(first_launch.blocks.args[0], CudaBuiltinNode)
        assert first_launch.blocks.args[0].builtin_name == "gridDim"
        assert isinstance(first_launch.threads, FunctionCallNode)
        assert first_launch.threads.name == "dim3"

        assert isinstance(second_launch, KernelLaunchNode)
        assert isinstance(second_launch.blocks, BinaryOpNode)
        assert second_launch.blocks.op == "+"
        assert isinstance(second_launch.threads, BinaryOpNode)
        assert second_launch.threads.op == "*"

    def test_dim3_declarations_can_shadow_builtin_launch_names(self):
        code = """
        void host(float* data) {
            dim3 gridDim;
            dim3 blockDim;
            blockDim.x = 128;
            gridDim.x = (1024 + blockDim.x - 1) / blockDim.x;
            kernel<<<gridDim, blockDim>>>(data);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        grid_decl = ast.functions[0].body[0]
        block_decl = ast.functions[0].body[1]
        launch = ast.functions[0].body[4]
        assert grid_decl.name == "gridDim"
        assert block_decl.name == "blockDim"
        assert isinstance(launch, KernelLaunchNode)
        assert isinstance(launch.blocks, CudaBuiltinNode)
        assert launch.blocks.builtin_name == "gridDim"
        assert isinstance(launch.threads, CudaBuiltinNode)
        assert launch.threads.builtin_name == "blockDim"

    def test_cuda_launch_kernel_api_parsing(self):
        code = """
        void host(float* data, int n, int stream) {
            dim3 grid(16);
            dim3 block(32);
            void* packedArgs[] = { &data, &n };
            cudaLaunchKernel((const void*)kernel, grid, block, packedArgs, 0, stream);
            cudaLaunchKernel(kernel);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        packed_args = ast.functions[0].body[2]
        assert packed_args.vtype == "void *[]"
        assert isinstance(packed_args.value, InitializerListNode)
        assert packed_args.value.elements[0].op == "&"
        assert packed_args.value.elements[0].operand == "data"
        assert packed_args.value.elements[1].operand == "n"

        launch = ast.functions[0].body[3]
        assert isinstance(launch, KernelLaunchNode)
        assert launch.kernel_name == "kernel"
        assert launch.blocks == "grid"
        assert launch.threads == "block"
        assert launch.shared_mem == "0"
        assert launch.stream == "stream"
        assert launch.args == ["packedArgs"]
        assert isinstance(ast.functions[0].body[4], FunctionCallNode)
        assert ast.functions[0].body[4].name == "cudaLaunchKernel"

    def test_cuda_runtime_launch_function_addresses_are_unwrapped(self):
        code = """
        void host(void** packedArgs, int stream) {
            cudaLaunchKernel((const void*)&kernel, 1, 32, packedArgs, 0, stream);
            cudaLaunchCooperativeKernel(
                (void*)&cooperativeKernel<float>, 1, 32, packedArgs, 0, stream);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        first_launch = ast.functions[0].body[0]
        second_launch = ast.functions[0].body[1]

        assert isinstance(first_launch, KernelLaunchNode)
        assert first_launch.kernel_name == "kernel"
        assert isinstance(second_launch, KernelLaunchNode)
        assert second_launch.kernel_name == "cooperativeKernel<float>"

    def test_cuda_launch_cooperative_kernel_api_parsing(self):
        code = """
        void host(void** params) {
            dim3 grid(16);
            dim3 block(32);
            cudaLaunchCooperativeKernel((void*)k, grid, block, params, 0, NULL);
            cudaLaunchCooperativeKernel(k);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        launch = ast.functions[0].body[2]
        assert isinstance(launch, KernelLaunchNode)
        assert launch.kernel_name == "k"
        assert launch.blocks == "grid"
        assert launch.threads == "block"
        assert launch.shared_mem == "0"
        assert launch.stream == "NULL"
        assert launch.args == ["params"]
        assert isinstance(ast.functions[0].body[3], FunctionCallNode)
        assert ast.functions[0].body[3].name == "cudaLaunchCooperativeKernel"

    def test_cuda_launch_kernel_casted_packed_args_parsing(self):
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

        packed_args = ast.functions[0].body[2]
        assert packed_args.vtype == "void * *"
        assert isinstance(packed_args.value, InitializerListNode)

        launch = ast.functions[0].body[3]
        assert isinstance(launch, KernelLaunchNode)
        assert launch.kernel_name == "kernel"
        assert len(launch.args) == 1
        assert isinstance(launch.args[0], CastNode)
        assert launch.args[0].target_type == "void * *"
        assert launch.args[0].expression == "packedArgs"

    def test_cuda_launch_kernel_compound_literal_args_parsing(self):
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

        launch = ast.functions[0].body[2]
        assert isinstance(launch, KernelLaunchNode)
        assert len(launch.args) == 1
        assert isinstance(launch.args[0], CastNode)
        assert launch.args[0].target_type == "void * []"
        assert isinstance(launch.args[0].expression, InitializerListNode)
        assert launch.args[0].expression.elements[0].op == "&"
        assert launch.args[0].expression.elements[0].operand == "data"
        assert launch.args[0].expression.elements[1].operand == "n"

    def test_std_chrono_benchmark_expressions_parsing(self):
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

        body = ast.functions[0].body
        assert body[0].vtype == "auto"
        assert body[0].value.name == "std::chrono::high_resolution_clock::now"
        assert isinstance(body[1], KernelLaunchNode)
        assert body[2].value.name == "std::chrono::high_resolution_clock::now"
        assert isinstance(body[3].value.name, MemberAccessNode)
        assert body[3].value.name.member == "count"
        duration_call = body[3].value.name.object
        assert (
            duration_call.name
            == "std::chrono::duration_cast<std::chrono::microseconds>"
        )
        assert isinstance(duration_call.args[0], BinaryOpNode)
        assert body[4].value.op == "<"

    def test_device_lambda_expression_parsing(self):
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

        body = ast.functions[0].body
        folded_lambda = body[0].value.args[2]
        assert isinstance(folded_lambda, FunctionCallNode)
        assert folded_lambda.name == "lambda"
        assert [(arg.vtype, arg.name) for arg in folded_lambda.args[:-1]] == [
            ("int", "acc"),
            ("int", "x"),
        ]
        assert isinstance(folded_lambda.args[-1], BinaryOpNode)

        mapped_lambda = body[1].value.args[1]
        assert isinstance(mapped_lambda, FunctionCallNode)
        assert mapped_lambda.name == "lambda"
        assert isinstance(mapped_lambda.args[0], VariableNode)
        assert mapped_lambda.args[0].vtype == "float3"
        assert mapped_lambda.args[0].name == "color"
        assert mapped_lambda.args[-1] == "{ prepare(color); return color; }"

    def test_std_vector_host_buffer_parsing(self):
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

        body = ast.functions[0].body
        assert body[0].vtype == "std::vector<float>"
        assert body[0].name == "h"
        assert body[0].value.name == "std::vector<float>"
        copy_call = body[1]
        assert isinstance(copy_call, FunctionCallNode)
        assert copy_call.name == "cudaMemcpy"
        assert isinstance(copy_call.args[1].name, MemberAccessNode)
        assert copy_call.args[1].name.member == "data"
        assert isinstance(copy_call.args[2], BinaryOpNode)
        assert isinstance(copy_call.args[2].left.name, MemberAccessNode)
        assert copy_call.args[2].left.name.member == "size"
        assert body[2].value.op == "<"
        assert isinstance(body[3], FunctionCallNode)
        assert body[3].name == "std::chrono::high_resolution_clock::now"

    def test_std_array_host_buffer_parsing(self):
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

        body = ast.functions[0].body
        assert body[0].vtype == "std::array<float, 4>"
        assert body[0].name == "h"
        assert isinstance(body[0].value, InitializerListNode)
        assert body[0].value.elements == [
            "1.0f",
            "2.0f",
            "3.0f",
            "4.0f",
        ]
        assert body[1].vtype == "std::array<float, 4>"
        assert body[1].name == "zeros"
        assert isinstance(body[1].value, InitializerListNode)
        assert body[1].value.elements == []
        copy_call = body[2]
        assert isinstance(copy_call, FunctionCallNode)
        assert copy_call.name == "cudaMemcpy"
        assert isinstance(copy_call.args[1].name, MemberAccessNode)
        assert copy_call.args[1].name.member == "data"
        assert isinstance(copy_call.args[2], BinaryOpNode)
        assert isinstance(copy_call.args[2].left.name, MemberAccessNode)
        assert copy_call.args[2].left.name.member == "size"

    def test_host_index_fill_scalar_constructor_parsing(self):
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

        body = ast.functions[0].body
        assert isinstance(body[1], ForNode)
        assignment = body[1].body[0]
        assert isinstance(assignment, AssignmentNode)
        assert isinstance(assignment.right, FunctionCallNode)
        assert assignment.right.name == "float"
        assert assignment.right.args == ["i"]

    def test_reference_host_helper_parameters_parsing(self):
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

        assert ast.functions[0].params[0].vtype == "std::vector<float> &"
        assert ast.functions[1].return_type == "size_t"
        assert ast.functions[1].params[0].vtype == "const std::vector<float> &"

    def test_restrict_pointer_qualifier_parsing(self):
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

        assert [param.vtype for param in ast.kernels[0].params] == [
            "const float * __restrict__",
            "float __restrict__ *",
        ]

        body = ast.functions[0].body
        assert [var.vtype for var in body] == [
            "float * __restrict__",
            "const float * __restrict__",
            "float * __restrict__",
            "float * __restrict__",
        ]
        assert [var.name for var in body] == ["p", "cp", "a", "b"]

    def test_single_trailing_restrict_pointer_qualifier_parsing(self):
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

        assert [param.vtype for param in ast.kernels[0].params] == [
            "float2 * __restrict",
            "const int * __restrict",
        ]

        body = ast.functions[0].body
        assert [var.vtype for var in body] == [
            "float * __restrict",
            "float * __restrict",
        ]
        assert [var.name for var in body] == ["a", "b"]

    def test_launch_bounds_kernel_attribute_parsing(self):
        code = """
        __launch_bounds__(128) __global__ void BlackScholesGPU(
            float2 *__restrict d_CallResult,
            float2 *__restrict d_PutResult) {
            d_CallResult[threadIdx.x] = d_PutResult[threadIdx.x];
        }

        extern "C" __launch_bounds__(256, 2) __global__ void bounded(float* data) {
            data[threadIdx.x] = 0.0f;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert [kernel.name for kernel in ast.kernels] == [
            "BlackScholesGPU",
            "bounded",
        ]
        assert ast.kernels[0].attributes == ["__launch_bounds__(128)"]
        assert ast.kernels[1].attributes == ["__launch_bounds__(256, 2)"]
        assert ast.kernels[0].params[0].vtype == "float2 * __restrict"
        assert ast.kernels[1].qualifiers == ["extern", "__global__"]
        assert ast.kernels[1].linkage == "C"

    def test_cuda_function_attribute_parsing(self):
        code = """
        __cluster_dims__(2, 1, 1) __block_size__(128) __global__ void clustered(
            float* out) {
            out[threadIdx.x] = 1.0f;
        }

        extern "C" __global__ void __block_size__((256, 1, 1), (2, 2, 2))
        shaped(float* out) {
            out[threadIdx.x] = 2.0f;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert [kernel.name for kernel in ast.kernels] == ["clustered", "shaped"]
        assert ast.kernels[0].attributes == [
            "__cluster_dims__(2, 1, 1)",
            "__block_size__(128)",
        ]
        assert ast.kernels[1].attributes == ["__block_size__((256, 1, 1), (2, 2, 2))"]
        assert ast.kernels[1].qualifiers == ["extern", "__global__"]
        assert ast.kernels[1].linkage == "C"

    def test_c_linkage_block_preserves_cuda_function_metadata(self):
        code = """
        extern "C" {
        __global__ void kernel(float* out) {
            out[threadIdx.x] = 1.0f;
        }

        __device__ float add(float a, float b) {
            return a + b;
        }
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        kernel = ast.kernels[0]
        device_function = ast.functions[0]

        assert kernel.name == "kernel"
        assert kernel.qualifiers == ["extern", "__global__"]
        assert kernel.linkage == "C"
        assert device_function.name == "add"
        assert device_function.qualifiers == ["extern", "__device__"]
        assert device_function.linkage == "C"

    def test_grid_constant_kernel_parameter_parsing(self):
        code = """
        struct Params {
            int value;
        };

        __global__ void kernelDefault(__grid_constant__ const Params p, int* out) {
            out[threadIdx.x] = p.value;
        }

        __global__ void kernelGuide(const __grid_constant__ int scale, int* out) {
            out[threadIdx.x] = scale;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert [kernel.name for kernel in ast.kernels] == [
            "kernelDefault",
            "kernelGuide",
        ]
        assert ast.kernels[0].params[0].vtype == "__grid_constant__ const Params"
        assert ast.kernels[0].params[0].name == "p"
        assert ast.kernels[1].params[0].vtype == "const __grid_constant__ int"
        assert ast.kernels[1].params[0].name == "scale"

    def test_rvalue_reference_declarations_parsing(self):
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

        assert [param.vtype for param in ast.functions[0].params] == [
            "float &&",
            "const float &&",
        ]

        body = ast.functions[1].body
        assert [stmt.vtype for stmt in body[:4]] == [
            "auto &&",
            "const auto &&",
            "float &&",
            "bool",
        ]
        assert body[3].value.op == "&&"
        assert isinstance(body[4], RangeForNode)
        assert body[4].vtype == "auto &&"

    def test_cpp_named_casts_parse_as_cast_nodes(self):
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

        body = ast.functions[0].body
        assert isinstance(body[0].value, CastNode)
        assert body[0].value.target_type == "float"
        assert body[0].value.expression == "i"
        assert isinstance(body[1].value, CastNode)
        assert body[1].value.target_type == "float*"
        assert body[1].value.expression == "input"
        assert isinstance(body[2].args[0], CastNode)
        assert body[2].args[0].target_type == "void**"
        assert body[2].args[0].expression.op == "&"
        assert body[2].args[0].expression.operand == "data"

    def test_new_delete_host_allocation_parsing(self):
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

        body = ast.functions[0].body
        assert isinstance(body[0].value, NewNode)
        assert body[0].value.target_type == "float"
        assert body[0].value.size == "n"
        assert body[0].value.is_array is True
        assert isinstance(body[1].value, NewNode)
        assert body[1].value.target_type == "int"
        assert body[1].value.args == []
        assert body[1].value.is_array is False
        assert body[2].value.args == ["1.0f"]
        assert isinstance(body[3], AssignmentNode)
        assert isinstance(body[4], DeleteNode)
        assert body[4].expression == "h"
        assert body[4].is_array is True
        assert isinstance(body[5], DeleteNode)
        assert body[5].expression == "q"
        assert body[5].is_array is False

    def test_public_cuda_samples_placement_new_and_variadic_parameters(self):
        code = """
        template <typename T>
        struct Vector {
            int size;
        };

        void error_exit(const char *format, ...) {
            return;
        }

        __global__ void construct(char* s_buffer, int* s_data) {
            Vector<int>* s_vector;
            s_vector = new (s_buffer) Vector<int>(1024, s_data);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert ast.functions[0].params[-1].vtype == "..."
        assert ast.functions[0].params[-1].name == ""

        allocation = ast.kernels[0].body[1]
        assert isinstance(allocation, AssignmentNode)
        assert isinstance(allocation.right, NewNode)
        assert allocation.right.target_type == "Vector<int>"
        assert allocation.right.placement_args == ["s_buffer"]
        assert allocation.right.args == ["1024", "s_data"]

    def test_cuda_parameter_pack_reference_from_llm_metal_corpus(self):
        code = """
        template<class Kernel, class... KernelArgs>
        float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kernel_args) {
            kernel(std::forward<KernelArgs>(kernel_args)...);
            return 0.0f;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert [(param.vtype, param.name) for param in ast.functions[0].params] == [
            ("int", "repeats"),
            ("Kernel", "kernel"),
            ("KernelArgs && ...", "kernel_args"),
        ]
        call = ast.functions[0].body[0]
        assert isinstance(call, FunctionCallNode)
        assert call.name == "kernel"
        assert isinstance(call.args[0], UnaryOpNode)
        assert call.args[0].op == "post..."

    def test_public_cuda_samples_dependent_qualified_typename_parsing(self):
        code = """
        template <typename T>
        __device__ typename vec3<T>::Type
        bodyBodyInteraction(typename vec3<T>::Type ai,
                            typename vec4<T>::Type *positions) {
            typename vec3<T>::Type r;
            positions = (typename vec4<T>::Type *)positions;
            return ai;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        function = ast.functions[0]
        assert function.return_type == "typename vec3<T>::Type"
        assert [(param.vtype, param.name) for param in function.params] == [
            ("typename vec3<T>::Type", "ai"),
            ("typename vec4<T>::Type *", "positions"),
        ]
        assert function.body[0].vtype == "typename vec3<T>::Type"
        assert isinstance(function.body[1].right, CastNode)
        assert function.body[1].right.target_type == "typename vec4<T>::Type *"

    def test_public_cuda_samples_tile_attributes_bindings_and_udl_literals(self):
        code = """
        template <int NUM_CTAS>
        [[using cutile: hint(0, num_cta_in_cga = NUM_CTAS),
                         hint(0, occupancy = 1)]]
        __global__ void tiled(float* _C) {
            using namespace cuda::experimental::stf::literals;
            float* C = ct::assume_aligned(_C, 16_ic);
            auto [pid_m, pid_n, dummy] = ct::bid();
            auto acc = ct::zeros<ct::tile<float, ct::shape<16, 16>>>();
            [[cutile::hint(0, latency = 4)]] acc = acc;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert len(ast.kernels) == 1
        body = ast.kernels[0].body
        assert body[0].value.name == "ct::assume_aligned"
        assert body[0].value.args == ["_C", "16_ic"]
        assert body[1].vtype == "auto"
        assert body[1].name == "[pid_m, pid_n, dummy]"
        assert body[1].value.name == "ct::bid"
        assert body[2].value.name == "ct::zeros<ct::tile<float, ct::shape<16, 16>>>"
        assert isinstance(body[3], AssignmentNode)
        assert body[3].left == "acc"

    def test_public_cuda_samples_tile_global_kernel_parsing(self):
        code = """
        template <int NUM_CTAS>
        __tile_global__ void matmul_naive(float* C,
                                          const __half* A,
                                          int M,
                                          int K) {
            auto a_span = ct::tensor_span{A, ct::extents{M, K}};
            auto [pid_m, pid_n, dummy] = ct::bid();
            C[pid_m] = 0.0f;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert len(ast.kernels) == 1
        assert ast.functions == []

        kernel = ast.kernels[0]
        assert isinstance(kernel, KernelNode)
        assert kernel.name == "matmul_naive"
        assert kernel.qualifiers == ["__tile_global__"]
        assert kernel.params[1].vtype == "const __half *"
        assert kernel.body[0].value.name == "ct::tensor_span"
        assert kernel.body[1].name == "[pid_m, pid_n, dummy]"

    def test_cuda_programming_guide_tile_function_parsing(self):
        # Source pattern:
        # https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html
        code = """
        #include "cuda_tile.h"
        __tile__ float helper(float x, float y) {
            return x + y;
        }
        __tile_global__ void my_kernel(float* a, float* b, float* c) {
            c[0] = helper(a[0], b[0]);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert len(ast.functions) == 1
        assert ast.functions[0].name == "helper"
        assert ast.functions[0].qualifiers == ["__tile__"]
        assert ast.kernels[0].name == "my_kernel"
        assert ast.kernels[0].qualifiers == ["__tile_global__"]

    def test_unique_ptr_host_allocation_parsing(self):
        code = """
        void host(int n) {
            std::unique_ptr<float[]> h = std::make_unique<float[]>(n);
            std::unique_ptr<int> q = std::make_unique<int>();
            std::unique_ptr<float[]> owned(new float[n]);
            float* raw = h.get();
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        body = ast.functions[0].body
        assert body[0].vtype == "std::unique_ptr<float[]>"
        assert isinstance(body[0].value, FunctionCallNode)
        assert body[0].value.name == "std::make_unique<float[]>"
        assert body[0].value.args == ["n"]
        assert body[1].vtype == "std::unique_ptr<int>"
        assert body[1].value.name == "std::make_unique<int>"
        assert body[2].value.name == "std::unique_ptr<float[]>"
        assert isinstance(body[2].value.args[0], NewNode)
        assert body[2].value.args[0].is_array is True
        assert isinstance(body[3].value, FunctionCallNode)
        assert isinstance(body[3].value.name, MemberAccessNode)
        assert body[3].value.name.object == "h"
        assert body[3].value.name.member == "get"

    def test_qualified_template_argument_spacing_parsing(self):
        code = """
        void host(int n) {
            std::unique_ptr<const float[]> h =
                std::make_unique<const float[]>(n);
            std::array<unsigned int, 4> ids{};
            std::vector<const unsigned int> flags;
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                elapsed
            ).count();
            texture<float4, 2> tex2d;
            texture<float4, cudaTextureType2DLayered> layeredTex;
            surface<void, cudaSurfaceType1DLayered> lineLayerSurface;
            surface<void, cudaSurfaceType3D> volumeSurface;
            surface<void, cudaSurfaceTypeCubemap> cubeSurface;
            surface<void, cudaSurfaceTypeCubemapLayered> cubeLayerSurface;
            cudaArray_t arrayRef;
            cudaArray* rawArray;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        body = ast.functions[0].body
        assert body[0].vtype == "std::unique_ptr<const float[]>"
        assert body[0].value.name == "std::make_unique<const float[]>"
        assert body[1].vtype == "std::array<unsigned int, 4>"
        assert body[2].vtype == "std::vector<const unsigned int>"
        assert body[3].value.name.member == "count"
        assert (
            body[3].value.name.object.name
            == "std::chrono::duration_cast<std::chrono::microseconds>"
        )
        assert body[4].vtype == "texture<float4, 2>"
        assert body[5].vtype == "texture<float4, cudaTextureType2DLayered>"
        assert body[6].vtype == "surface<void, cudaSurfaceType1DLayered>"
        assert body[7].vtype == "surface<void, cudaSurfaceType3D>"
        assert body[8].vtype == "surface<void, cudaSurfaceTypeCubemap>"
        assert body[9].vtype == "surface<void, cudaSurfaceTypeCubemapLayered>"
        assert body[10].vtype == "cudaArray_t"
        assert body[11].vtype == "cudaArray *"

    def test_resource_object_pointer_and_legacy_texture_template_parsing(self):
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
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert ast.global_variables[0].vtype == (
            "texture<float4, cudaTextureType2D, cudaReadModeElementType>"
        )
        assert ast.global_variables[0].name == "texRef"
        assert ast.global_variables[1].vtype == "surface<void, cudaSurfaceType2D>"
        assert ast.global_variables[1].name == "surfaceRef"

        params = ast.functions[0].params
        assert params[0].vtype == "cudaTextureObject_t *"
        assert params[0].name == "textureObjects"
        assert params[1].vtype == "cudaSurfaceObject_t *"
        assert params[1].name == "surfaceObjects"
        assert params[2].vtype == (
            "texture<float4, cudaTextureType2D, cudaReadModeElementType>"
        )
        assert params[2].name == "legacyTex"
        assert params[3].vtype == "surface<void, cudaSurfaceType2D>"
        assert params[3].name == "legacySurface"

        body = ast.functions[0].body
        assert body[0].vtype == "float4"
        assert body[0].value.name == "tex2D<float4>"
        assert isinstance(body[0].value.args[0], ArrayAccessNode)
        assert body[0].value.args[0].array == "textureObjects"
        assert body[1].value.name == "tex2D<float4>"
        assert body[1].value.args[0] == "legacyTex"
        assert body[2].value.name == "surf2Dread<float4>"
        assert isinstance(body[2].value.args[0], ArrayAccessNode)
        assert body[2].value.args[0].array == "surfaceObjects"
        assert body[3].name == "surf2Dwrite"
        assert isinstance(body[3].args[1], ArrayAccessNode)
        assert body[4].value.name == "surf2Dread<float4>"
        assert body[4].value.args[0] == "legacySurface"

    def test_cuda_texture_gather_helper_parsing(self):
        code = """
        void gatherOps(
            texture<float4, cudaTextureType2D> tex,
            cudaTextureObject_t objectTex,
            bool* resident,
            float2 uv
        ) {
            float4 gathered = tex2Dgather<float4>(tex, uv.x, uv.y);
            float4 gatheredComponent = tex2Dgather<float4>(
                objectTex,
                uv.x,
                uv.y,
                2
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

        params = ast.functions[0].params
        assert params[0].vtype == "texture<float4, cudaTextureType2D>"
        assert params[1].vtype == "cudaTextureObject_t"
        assert params[2].vtype == "bool *"

        body = ast.functions[0].body
        assert body[0].value.name == "tex2Dgather<float4>"
        assert body[0].value.args[0] == "tex"
        assert body[0].value.args[1].object == "uv"
        assert body[0].value.args[1].member == "x"
        assert body[0].value.args[2].object == "uv"
        assert body[0].value.args[2].member == "y"
        assert body[1].value.name == "tex2Dgather<float4>"
        assert body[1].value.args[0] == "objectTex"
        assert body[1].value.args[1].object == "uv"
        assert body[1].value.args[1].member == "x"
        assert body[1].value.args[2].object == "uv"
        assert body[1].value.args[2].member == "y"
        assert body[1].value.args[3] == "2"
        assert body[2].value.name == "tex2Dgather<float4>"
        assert body[2].value.args[0] == "objectTex"
        assert body[2].value.args[1].object == "uv"
        assert body[2].value.args[1].member == "x"
        assert body[2].value.args[2].object == "uv"
        assert body[2].value.args[2].member == "y"
        assert body[2].value.args[3:] == ["resident", "3"]

    def test_cuda_texture_fetch_helper_parsing(self):
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

        params = ast.functions[0].params
        assert params[0].vtype == "texture<float4, cudaTextureType1D>"
        assert params[1].vtype == "cudaTextureObject_t"
        assert params[2].vtype == "bool *"

        body = ast.functions[0].body
        assert body[0].value.name == "tex1Dfetch<float4>"
        assert body[0].value.args == ["lineTex", "x"]
        assert body[1].value.name == "tex1Dfetch<float4>"
        assert body[1].value.args == ["objectTex", "x"]
        assert body[2].value.name == "tex1Dfetch<float4>"
        assert body[2].value.args == ["objectTex", "x", "resident"]

    def test_qualified_resource_object_pointer_array_parsing(self):
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

        assert [(var.vtype, var.name) for var in ast.global_variables] == [
            ("const cudaTextureObject_t[2]", "globalConstTextures"),
            ("cudaSurfaceObject_t * __restrict__[2]", "globalSurfaceRows"),
        ]

        params = ast.functions[0].params
        assert [param.vtype for param in params[:4]] == [
            "const cudaTextureObject_t * __restrict__",
            "cudaSurfaceObject_t * __restrict__",
            "const cudaTextureObject_t[2]",
            "cudaSurfaceObject_t * __restrict__[2]",
        ]
        assert [param.name for param in params[:4]] == [
            "textureObjects",
            "surfaceObjects",
            "textureArray",
            "surfaceRows",
        ]

        body = ast.functions[0].body
        assert isinstance(body[0].value.args[0], ArrayAccessNode)
        assert body[0].value.args[0].array == "textureObjects"
        assert isinstance(body[1].value.args[0], ArrayAccessNode)
        assert body[1].value.args[0].array == "textureArray"
        assert isinstance(body[2].value.args[0], ArrayAccessNode)
        assert body[2].value.args[0].array == "globalConstTextures"
        assert isinstance(body[3].value.args[0], ArrayAccessNode)
        assert body[3].value.args[0].array == "surfaceObjects"
        assert isinstance(body[5].value.args[0], ArrayAccessNode)
        assert isinstance(body[5].value.args[0].array, ArrayAccessNode)
        assert body[5].value.args[0].array.array == "surfaceRows"
        assert isinstance(body[7].value.args[0], ArrayAccessNode)
        assert isinstance(body[7].value.args[0].array, ArrayAccessNode)
        assert body[7].value.args[0].array.array == "globalSurfaceRows"

    def test_nested_template_argument_parsing(self):
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

        body = ast.functions[0].body
        assert body[0].vtype == "std::vector<std::array<unsigned int, 4>>"
        assert body[1].vtype == "std::vector<std::vector<float>>"
        assert body[2].vtype == "std::unique_ptr<const float*>"
        assert body[2].value.name == "std::make_unique<const float*>"
        assert body[3].vtype == "std::unique_ptr<float[], HostDeleter>"
        assert body[3].value.name == "std::unique_ptr<float[], HostDeleter>"
        assert isinstance(body[3].value.args[0], NewNode)

    def test_type_alias_parsing(self):
        code = """
        using HostBuffer = std::unique_ptr<float[]>;
        typedef std::vector<std::array<unsigned int, 4>> Table;
        using namespace std;
        using std::cerr;
        using std::endl;

        void host(int n) {
            using LocalBuffer = std::unique_ptr<float[], HostDeleter>;
            using std::min;
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

        assert len(ast.typedefs) == 2
        assert all(isinstance(alias, TypeAliasNode) for alias in ast.typedefs)
        assert ast.typedefs[0].name == "HostBuffer"
        assert ast.typedefs[0].alias_type == "std::unique_ptr<float[]>"
        assert ast.typedefs[1].name == "Table"
        assert ast.typedefs[1].alias_type == "std::vector<std::array<unsigned int, 4>>"

        body = ast.functions[0].body
        assert isinstance(body[0], TypeAliasNode)
        assert body[0].name == "LocalBuffer"
        assert body[0].alias_type == "std::unique_ptr<float[], HostDeleter>"
        assert body[1].vtype == "HostBuffer"
        assert body[2].vtype == "HostBuffer *"
        assert body[3].vtype == "LocalBuffer"
        assert body[4].vtype == "Table"
        assert body[5].name == "consume"

    def test_public_cuda_function_pointer_using_alias_parsing(self):
        code = """
        using matmul_fn_ptr = void(*)(float* p,
                                      int ps,
                                      const float* k,
                                      int ks,
                                      const float* q,
                                      int qs,
                                      int T,
                                      int hs,
                                      float alpha);

        __global__ void trimul_global(float* out) {
            out[threadIdx.x] = 0.0f;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        alias = ast.typedefs[0]
        assert isinstance(alias, TypeAliasNode)
        assert alias.name == "matmul_fn_ptr"
        assert alias.alias_type == "void (*)"
        assert [(param.vtype, param.name) for param in alias.params] == [
            ("float *", "p"),
            ("int", "ps"),
            ("const float *", "k"),
            ("int", "ks"),
            ("const float *", "q"),
            ("int", "qs"),
            ("int", "T"),
            ("int", "hs"),
            ("float", "alpha"),
        ]
        assert ast.kernels[0].name == "trimul_global"

    def test_cub_dependent_shared_temp_storage_parsing(self):
        code = """
        using WarpReduce = cub::WarpReduce<int>;

        __global__ void kernel(int* out, int value) {
            __shared__ typename WarpReduce::TempStorage temp_storage[4];
            int warp_id = threadIdx.x / 32;
            int aggregate = WarpReduce(temp_storage[warp_id]).Sum(value);
            out[threadIdx.x] = aggregate;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert ast.typedefs[0].name == "WarpReduce"
        assert ast.typedefs[0].alias_type == "cub::WarpReduce<int>"

        body = ast.kernels[0].body
        assert isinstance(body[0], SharedMemoryNode)
        assert body[0].vtype == "typename WarpReduce::TempStorage[4]"
        assert body[0].name == "temp_storage"
        assert body[1].vtype == "int"
        assert isinstance(body[2].value, FunctionCallNode)
        assert isinstance(body[2].value.name, MemberAccessNode)
        assert body[2].value.name.member == "Sum"
        storage_arg = body[2].value.name.object.args[0]
        assert isinstance(storage_arg, ArrayAccessNode)
        assert storage_arg.array == "temp_storage"
        assert storage_arg.index == "warp_id"

    def test_typedef_multi_declarator_alias_parsing(self):
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

        assert [(alias.name, alias.alias_type) for alias in ast.typedefs] == [
            ("Real", "float"),
            ("RealPtr", "float *"),
            ("Tile", "float[16]"),
            ("Buffer", "std::unique_ptr<float[]>"),
            ("BufferPtr", "std::unique_ptr<float[]> *"),
        ]

        function = ast.functions[0]
        assert function.params[1].vtype == "Real"
        assert [stmt.vtype for stmt in function.body[:5]] == [
            "Real",
            "RealPtr",
            "Tile",
            "Buffer",
            "BufferPtr",
        ]
        assert function.body[5].name == "consume"

    def test_typedef_struct_with_alignment_parsing(self):
        code = """
        typedef struct Existing *ExistingPtr;
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

        assert len(ast.typedefs) == 1
        assert ast.typedefs[0].name == "ExistingPtr"
        assert ast.typedefs[0].alias_type == "struct Existing *"
        assert len(ast.structs) == 1
        pair = ast.structs[0]
        assert isinstance(pair, StructNode)
        assert pair.name == "Pair"
        assert pair.attributes == ["__align__(8)"]
        assert [(member.vtype, member.name) for member in pair.members] == [
            ("unsigned int", "x"),
            ("unsigned int", "y"),
        ]
        assert ast.kernels[0].params[0].vtype == "Pair *"
        assert ast.kernels[0].params[0].name == "pairs"

    def test_cuda_vector_types_header_struct_attributes_parse(self):
        code = """
        struct __VECTOR_TYPE_DEPRECATED__("use float4_16a")
               __device_builtin__ __builtin_align__(16) float4 {
            float x, y, z, w;
        };
        typedef __device_builtin__ struct float4 float4;
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert len(ast.structs) == 1
        vector = ast.structs[0]
        assert isinstance(vector, StructNode)
        assert vector.name == "float4"
        assert vector.attributes == ["__builtin_align__(16)"]
        assert [(member.vtype, member.name) for member in vector.members] == [
            ("float", "x"),
            ("float", "y"),
            ("float", "z"),
            ("float", "w"),
        ]
        assert len(ast.typedefs) == 1
        assert ast.typedefs[0].alias_type == "struct float4"
        assert ast.typedefs[0].name == "float4"

    def test_aligned_anonymous_aggregate_members_parse(self):
        code = """
        struct Packet {
            int header;
            struct __align__(16) {
                float4 value;
            } payload;
            union alignas(8) {
                int i;
                float f;
            };
        };
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        packet = ast.structs[0]
        assert packet.name == "Packet"
        assert [(member.vtype, member.name) for member in packet.members] == [
            ("int", "header"),
            ("struct", "payload"),
            ("int", "i"),
            ("float", "f"),
        ]
        assert packet.members[1].attributes == ["__align__(16)"]

    def test_cuda_struct_conversion_operator_methods_are_skipped(self):
        code = """
        template <class T> struct SharedMemory {
            int tag;

            __device__ inline operator T *() {
                extern __shared__ int __smem[];
                return (T *)__smem;
            }

            __device__ inline operator const T *() const {
                extern __shared__ int __smem[];
                return (T *)__smem;
            }
        };

        template <> struct SharedMemory<double> {
            __device__ inline operator double *() {
                extern __shared__ double __smem_d[];
                return (double *)__smem_d;
            }
        };

        template <class T, unsigned int blockSize, bool nIsPow2>
        __global__ void reduce(T *out) {
            T *sdata = SharedMemory<T>();
            out[threadIdx.x] = sdata[threadIdx.x];
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert ast.structs[0].name == "SharedMemory"
        assert [(member.vtype, member.name) for member in ast.structs[0].members] == [
            ("int", "tag")
        ]
        assert ast.structs[1].name == "SharedMemory<double>"
        assert ast.structs[1].members == []
        assert ast.kernels[0].name == "reduce"

    def test_cuda_struct_defaulted_template_constructor_from_generated_matrix_helpers(
        self,
    ):
        code = """
        struct float2x2 {
            float m[4];
            static const int CGL_COLUMNS = 2;

            template <typename Matrix, typename = decltype(Matrix::CGL_COLUMNS)>
            __host__ __device__ explicit float2x2(const Matrix& source) {}
        };
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert ast.structs[0].name == "float2x2"
        assert [(member.vtype, member.name) for member in ast.structs[0].members] == [
            ("float[4]", "m"),
            ("const int", "CGL_COLUMNS"),
        ]

    def test_public_cuda_samples_typedef_typename_inside_template_struct(self):
        code = """
        template <class RNG>
        struct PiEstimator {
            typedef typename RNG::value_type value_type;
            int samples;
        };

        __global__ void estimate_pi(int *out) {
            out[threadIdx.x] = 0;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert ast.structs[0].name == "PiEstimator"
        assert [(member.vtype, member.name) for member in ast.structs[0].members] == [
            ("int", "samples")
        ]
        assert ast.kernels[0].name == "estimate_pi"

    def test_public_cuda_samples_function_template_specialization_header(self):
        code = """
        template <class T, unsigned int blockSize, bool nIsPow2>
        __global__ void reduce<T, blockSize, nIsPow2>(T *out) {
            extern __shared__ unsigned int sdata[];
            out[threadIdx.x] = sdata[threadIdx.x];
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        kernel = ast.kernels[0]
        assert kernel.name == "reduce<T, blockSize, nIsPow2>"
        assert [(param.vtype, param.name) for param in kernel.params] == [
            ("T *", "out")
        ]
        assert isinstance(kernel.body[0], SharedMemoryNode)
        assert kernel.body[0].is_extern_shared_memory is True

    def test_constexpr_local_declaration_in_switch_case_parsing(self):
        code = """
        void launch(int mode) {
            switch (mode) {
            case 9:
                constexpr int numGroups = 2;
                break;
            }
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        declaration = ast.functions[0].body[0].cases[0].body[0]
        assert isinstance(declaration, VariableNode)
        assert declaration.vtype == "int"
        assert declaration.name == "numGroups"
        assert declaration.qualifiers == ["constexpr"]

    def test_explicit_template_instantiation_declaration_is_skipped(self):
        code = """
        template <class T>
        void reduce(int size, T *data) {
            data[0] = (T)size;
        }

        template void reduce<int>(int size, int *data);
        template void reduce<float>(int size, float *data);
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert [function.name for function in ast.functions] == ["reduce"]
        assert ast.global_variables == []

    def test_type_alias_c_style_cast_parsing(self):
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

        body = ast.functions[0].body
        assert len(body) == 1
        assert isinstance(body[0], ReturnNode)
        assert isinstance(body[0].value, CastNode)
        assert body[0].value.target_type == "LaneMask"
        assert body[0].value.expression == "x"

    def test_cuda_samples_const_unknown_pointer_pointer_c_style_cast_parsing(self):
        code = """
        void compile_shader(char* data, int size) {
            glShaderSource(shader, 1, (const GLchar **)&data, &size);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        call = ast.functions[0].body[0]
        shader_source = call.args[2]
        size_arg = call.args[3]

        assert isinstance(call, FunctionCallNode)
        assert isinstance(shader_source, CastNode)
        assert shader_source.target_type == "const GLchar * *"
        assert isinstance(shader_source.expression, UnaryOpNode)
        assert shader_source.expression.op == "&"
        assert shader_source.expression.operand == "data"
        assert isinstance(size_arg, UnaryOpNode)
        assert size_arg.op == "&"
        assert size_arg.operand == "size"

    def test_auto_pointer_reference_local_declarations_parsing(self):
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

        body = ast.functions[0].body
        assert isinstance(body[2], BinaryOpNode)
        assert [var.vtype for var in body[3:]] == [
            "auto &",
            "auto *",
            "const auto *",
            "auto *",
            "auto *",
        ]
        assert [var.name for var in body[3:]] == ["x", "p", "cp", "q", "r"]
        assert body[3].value.array == "h"
        assert body[3].value.index == "0"
        assert [var.value for var in body[4:]] == ["data", "data", "data", "data"]

    def test_multi_declarator_host_setup_parsing(self):
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

        body = ast.functions[0].body
        assert [var.name for var in body] == [
            "a",
            "b",
            "c",
            "d_a",
            "d_b",
            "x",
            "y",
            "z",
            "weights",
            "bias",
        ]
        assert body[0].vtype == "std::vector<float>"
        assert body[0].value.name == "std::vector<float>"
        assert body[1].value.args == ["n"]
        assert body[3].vtype == "float *"
        assert body[4].vtype == "float *"
        assert body[5].value == "1"
        assert body[6].value == "2"
        assert body[7].value is None
        assert body[8].vtype == "float[2]"
        assert isinstance(body[8].value, InitializerListNode)
        assert body[9].vtype == "float"
        assert body[9].value == "3.0f"

    def test_multi_declarator_for_initializer_parsing(self):
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

        first_loop = ast.functions[0].body[0]
        assert isinstance(first_loop, ForNode)
        assert [var.name for var in first_loop.init] == ["i", "j"]
        assert [var.vtype for var in first_loop.init] == ["int", "int"]
        assert first_loop.init[0].value == "0"
        assert first_loop.init[1].value == "n"

        second_loop = ast.functions[0].body[1]
        assert isinstance(second_loop, ForNode)
        assert [var.name for var in second_loop.init] == ["pa", "pb"]
        assert [var.vtype for var in second_loop.init] == ["float *", "float *"]
        assert second_loop.init[0].value == "a"
        assert second_loop.init[1].value == "b"

    def test_multi_expression_for_update_parsing(self):
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

        loop = ast.functions[0].body[0]
        assert isinstance(loop, ForNode)
        assert isinstance(loop.update, list)
        assert len(loop.update) == 2
        assert isinstance(loop.update[0], UnaryOpNode)
        assert loop.update[0].op == "++"
        assert loop.update[0].operand == "i"
        assert isinstance(loop.update[1], UnaryOpNode)
        assert loop.update[1].op == "--"
        assert loop.update[1].operand == "j"

    def test_grid_stride_for_update_compound_assignment_parsing(self):
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

        loop = ast.kernels[0].body[0]
        assert isinstance(loop, ForNode)
        assert isinstance(loop.update, AssignmentNode)
        assert loop.update.left == "i"
        assert loop.update.operator == "+="
        assert isinstance(loop.update.right, BinaryOpNode)
        assert loop.update.right.op == "*"
        assert isinstance(loop.update.right.left, CudaBuiltinNode)
        assert loop.update.right.left.builtin_name == "blockDim"
        assert loop.update.right.left.component == "x"
        assert isinstance(loop.update.right.right, CudaBuiltinNode)
        assert loop.update.right.right.builtin_name == "gridDim"
        assert loop.update.right.right.component == "x"

    def test_assignment_expression_is_right_associative(self):
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

        body = ast.functions[0].body
        chained = body[3]
        assert isinstance(chained, AssignmentNode)
        assert chained.left == "a"
        assert chained.operator == "="
        assert isinstance(chained.right, AssignmentNode)
        assert chained.right.left == "b"
        assert chained.right.right == "c"

        initializer = body[4].value
        assert isinstance(initializer, AssignmentNode)
        assert initializer.left == "a"
        assert initializer.right == "b"

        call = body[5]
        assert isinstance(call, FunctionCallNode)
        assert isinstance(call.args[0], AssignmentNode)
        assert call.args[0].left == "a"
        assert call.args[0].right == "1"

    def test_range_based_for_loop_parsing(self):
        code = """
        void host(std::vector<float>& h) {
            for (auto& x : h) {
                x = 1.0f;
            }
            for (const auto& x : h) {
                sink(x);
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

        loops = ast.functions[0].body
        assert all(isinstance(loop, RangeForNode) for loop in loops)
        assert [(loop.vtype, loop.name, loop.iterable) for loop in loops] == [
            ("auto &", "x", "h"),
            ("const auto &", "x", "h"),
            ("float", "y", "h"),
        ]

    def test_atomic_operations_parsing(self):
        code = """
        __global__ void kernel(int* counter) {
            atomicAdd(counter, 1);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert isinstance(ast, ShaderNode)
        assert len(ast.kernels) == 1

    def test_atomic_address_taken_pointer_array_and_shared_targets_parsing(self):
        code = """
        __global__ void kernel(int* values, int* expected, int* desired, int index) {
            __shared__ int sharedCounts[32];
            atomicAdd(&values[index], 1);
            int old = atomicCAS(&values[index], expected[index], desired[index]);
            atomicMax(&sharedCounts[threadIdx.x], old);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        body = ast.kernels[0].body
        atomic_add = body[1]
        atomic_cas = body[2].value
        atomic_max = body[3]

        assert isinstance(atomic_add, AtomicOperationNode)
        assert atomic_add.operation == "atomicAdd"
        assert isinstance(atomic_add.args[0], UnaryOpNode)
        assert atomic_add.args[0].op == "&"
        assert isinstance(atomic_add.args[0].operand, ArrayAccessNode)
        assert atomic_add.args[0].operand.array == "values"
        assert atomic_add.args[0].operand.index == "index"

        assert isinstance(atomic_cas, AtomicOperationNode)
        assert atomic_cas.operation == "atomicCAS"
        assert isinstance(atomic_cas.args[0], UnaryOpNode)
        assert isinstance(atomic_cas.args[1], ArrayAccessNode)
        assert atomic_cas.args[1].array == "expected"
        assert isinstance(atomic_cas.args[2], ArrayAccessNode)
        assert atomic_cas.args[2].array == "desired"

        assert isinstance(atomic_max, AtomicOperationNode)
        assert atomic_max.operation == "atomicMax"
        assert isinstance(atomic_max.args[0], UnaryOpNode)
        shared_target = atomic_max.args[0].operand
        assert isinstance(shared_target, ArrayAccessNode)
        assert shared_target.array == "sharedCounts"
        assert isinstance(shared_target.index, CudaBuiltinNode)
        assert shared_target.index.builtin_name == "threadIdx"
        assert shared_target.index.component == "x"

    def test_bitwise_atomic_operations_parsing(self):
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

        body = ast.kernels[0].body
        atomic_and = body[1].value
        atomic_or = body[2].value
        atomic_xor = body[3]

        assert isinstance(atomic_and, AtomicOperationNode)
        assert atomic_and.operation == "atomicAnd"
        assert isinstance(atomic_and.args[0], UnaryOpNode)
        assert atomic_and.args[0].op == "&"
        assert isinstance(atomic_and.args[0].operand, ArrayAccessNode)
        assert atomic_and.args[0].operand.array == "values"

        assert isinstance(atomic_or, AtomicOperationNode)
        assert atomic_or.operation == "atomicOr"
        assert isinstance(atomic_or.args[0], UnaryOpNode)
        shared_target = atomic_or.args[0].operand
        assert isinstance(shared_target, ArrayAccessNode)
        assert shared_target.array == "sharedMasks"
        assert isinstance(shared_target.index, CudaBuiltinNode)
        assert shared_target.index.builtin_name == "threadIdx"

        assert isinstance(atomic_xor, AtomicOperationNode)
        assert atomic_xor.operation == "atomicXor"
        assert isinstance(atomic_xor.args[0], UnaryOpNode)
        assert isinstance(atomic_xor.args[0].operand, ArrayAccessNode)

    def test_bounded_wrap_atomic_operations_parsing(self):
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

        body = ast.kernels[0].body
        atomic_inc = body[1].value
        atomic_dec = body[2].value

        assert isinstance(atomic_inc, AtomicOperationNode)
        assert atomic_inc.operation == "atomicInc"
        assert isinstance(atomic_inc.args[0], UnaryOpNode)
        assert atomic_inc.args[0].op == "&"
        assert isinstance(atomic_inc.args[0].operand, ArrayAccessNode)
        assert atomic_inc.args[0].operand.array == "values"

        assert isinstance(atomic_dec, AtomicOperationNode)
        assert atomic_dec.operation == "atomicDec"
        assert isinstance(atomic_dec.args[0], UnaryOpNode)
        shared_target = atomic_dec.args[0].operand
        assert isinstance(shared_target, ArrayAccessNode)
        assert shared_target.array == "sharedCounters"
        assert isinstance(shared_target.index, CudaBuiltinNode)
        assert shared_target.index.builtin_name == "threadIdx"

    def test_user_defined_atomic_name_is_not_parsed_as_builtin_atomic(self):
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

        assert ast.functions[0].name == "atomicExch"
        shadow_call = ast.kernels[0].body[0].value
        builtin_call = ast.kernels[0].body[1]

        assert isinstance(shadow_call, FunctionCallNode)
        assert not isinstance(shadow_call, AtomicOperationNode)
        assert shadow_call.name == "atomicExch"
        assert isinstance(builtin_call, AtomicOperationNode)

    def test_user_defined_atomic_name_declared_later_is_not_parsed_as_builtin_atomic(
        self,
    ):
        """Test later user-defined atomic names shadow CUDA atomic parsing."""
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

        assert ast.functions[0].name == "atomicExch"
        shadow_call = ast.kernels[0].body[0].value
        builtin_call = ast.kernels[0].body[1]

        assert isinstance(shadow_call, FunctionCallNode)
        assert not isinstance(shadow_call, AtomicOperationNode)
        assert shadow_call.name == "atomicExch"
        assert isinstance(builtin_call, AtomicOperationNode)

    def test_sync_parsing(self):
        code = """
        __global__ void kernel() {
            __syncthreads();
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert isinstance(ast, ShaderNode)
        assert len(ast.kernels) == 1

    def test_masked_syncwarp_parsing(self):
        code = """
        __global__ void kernel(unsigned int mask) {
            __syncwarp(mask);
            __syncwarp();
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        masked_sync = ast.kernels[0].body[0]
        unmasked_sync = ast.kernels[0].body[1]

        assert isinstance(masked_sync, SyncNode)
        assert masked_sync.sync_type == "__syncwarp"
        assert masked_sync.args == ["mask"]
        assert isinstance(unmasked_sync, SyncNode)
        assert unmasked_sync.sync_type == "__syncwarp"
        assert unmasked_sync.args == []

    def test_inline_ptx_asm_parsing(self):
        code = r"""
        __global__ void asmKernel(unsigned int* out, unsigned int in) {
            unsigned int lane = 0;
            asm volatile("bar.sync 0;");
            asm("mov.u32 %0, %%tid.x;" : "=r"(lane));
            asm volatile(
                "add.u32 %0, %1, 1;"
                : [result] "=r"(lane)
                : [source] "r"(in)
                : "memory"
            );
            asm __volatile__("membar.gl;" ::: "memory");
            out[threadIdx.x] = lane;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        asm_statements = [
            stmt for stmt in ast.kernels[0].body if isinstance(stmt, CudaAsmNode)
        ]
        assert len(asm_statements) == 4
        assert asm_statements[0].is_volatile is True
        assert asm_statements[0].template == '"bar.sync 0;"'
        assert asm_statements[1].outputs[0].constraint == '"=r"'
        assert asm_statements[1].outputs[0].expression == "lane"
        assert asm_statements[2].outputs[0].symbolic_name == "result"
        assert asm_statements[2].inputs[0].symbolic_name == "source"
        assert asm_statements[2].clobbers == ['"memory"']
        assert asm_statements[3].outputs == []
        assert asm_statements[3].inputs == []
        assert asm_statements[3].clobbers == ['"memory"']

    def test_cooperative_groups_thread_block_parsing(self):
        code = """
        __global__ void kernel() {
            cooperative_groups::thread_block block =
                cooperative_groups::this_thread_block();
            block.sync();
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        declaration = ast.kernels[0].body[0]
        sync_call = ast.kernels[0].body[1]

        assert isinstance(declaration, VariableNode)
        assert declaration.vtype == "cooperative_groups::thread_block"
        assert isinstance(declaration.value, FunctionCallNode)
        assert declaration.value.name == "cooperative_groups::this_thread_block"
        assert isinstance(sync_call, FunctionCallNode)
        assert isinstance(sync_call.name, MemberAccessNode)
        assert sync_call.name.object == "block"
        assert sync_call.name.member == "sync"

    def test_cooperative_groups_rank_and_tile_parsing(self):
        code = """
        __global__ void kernel() {
            cooperative_groups::thread_block block =
                cooperative_groups::this_thread_block();
            auto tile = cooperative_groups::tiled_partition<32>(block);
            unsigned int rank = block.thread_rank();
            unsigned int width = tile.size();
            unsigned int lane = tile.thread_rank();
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        body = ast.kernels[0].body
        tile = body[1]
        rank = body[2]
        width = body[3]
        lane = body[4]

        assert isinstance(tile, VariableNode)
        assert isinstance(tile.value, FunctionCallNode)
        assert tile.value.name == "cooperative_groups::tiled_partition<32>"
        assert tile.value.args == ["block"]

        for declaration, owner, member in [
            (rank, "block", "thread_rank"),
            (width, "tile", "size"),
            (lane, "tile", "thread_rank"),
        ]:
            assert isinstance(declaration, VariableNode)
            assert isinstance(declaration.value, FunctionCallNode)
            assert isinstance(declaration.value.name, MemberAccessNode)
            assert declaration.value.name.object == owner
            assert declaration.value.name.member == member

    def test_cooperative_groups_sync_and_member_alias_parsing(self):
        code = """
        namespace cg = cooperative_groups;

        __global__ void kernel() {
            auto block = cg::this_thread_block();
            cg::sync(cg::this_thread_block());
            unsigned int count = block.num_threads();
            dim3 local = block.thread_index();
            dim3 dims = block.dim_threads();
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        body = ast.kernels[0].body
        sync_call = body[1]
        count = body[2]
        local = body[3]
        dims = body[4]

        assert isinstance(sync_call, FunctionCallNode)
        assert sync_call.name == "cg::sync"
        assert isinstance(sync_call.args[0], FunctionCallNode)
        assert sync_call.args[0].name == "cg::this_thread_block"

        for declaration, owner, member in [
            (count, "block", "num_threads"),
            (local, "block", "thread_index"),
            (dims, "block", "dim_threads"),
        ]:
            assert isinstance(declaration, VariableNode)
            assert isinstance(declaration.value, FunctionCallNode)
            assert isinstance(declaration.value.name, MemberAccessNode)
            assert declaration.value.name.object == owner
            assert declaration.value.name.member == member

    def test_namespace_alias_metadata_is_preserved_for_codegen(self):
        code = """
        namespace cstd = cuda::std;

        __device__ float kernel(float x, float y) {
            return cstd::fminf(x, y);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        value = ast.functions[0].body[0].value

        assert ast.namespace_aliases == {"cstd": "cuda::std"}
        assert isinstance(value, FunctionCallNode)
        assert value.name == "cstd::fminf"

    def test_cooperative_groups_async_copy_wait_parsing(self):
        code = """
        namespace cg = cooperative_groups;

        __global__ void kernel(int* shared, const int* global) {
            auto block = cg::this_thread_block();
            cg::memcpy_async(block, shared, global, sizeof(int) * block.size());
            cg::wait(block);
            cooperative_groups::wait_prior<1>(block);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        body = ast.kernels[0].body
        async_copy = body[1]
        wait = body[2]
        wait_prior = body[3]

        assert isinstance(async_copy, FunctionCallNode)
        assert async_copy.name == "cg::memcpy_async"
        assert async_copy.args[:3] == ["block", "shared", "global"]
        assert isinstance(async_copy.args[3], BinaryOpNode)
        assert isinstance(async_copy.args[3].right, FunctionCallNode)
        assert isinstance(async_copy.args[3].right.name, MemberAccessNode)
        assert async_copy.args[3].right.name.object == "block"
        assert async_copy.args[3].right.name.member == "size"

        assert isinstance(wait, FunctionCallNode)
        assert wait.name == "cg::wait"
        assert wait.args == ["block"]

        assert isinstance(wait_prior, FunctionCallNode)
        assert wait_prior.name == "cooperative_groups::wait_prior<1>"
        assert wait_prior.args == ["block"]

    def test_cuda_barrier_pipeline_async_copy_parsing(self):
        code = """
        __global__ void kernel(int* shared, const int* global) {
            __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> state;
            cuda::barrier<cuda::thread_scope_block> blockBarrier;
            init(&blockBarrier, blockDim.x);
            cuda::memcpy_async(
                shared,
                global,
                sizeof(int) * blockDim.x,
                blockBarrier
            );
            auto pipe = cuda::make_pipeline();
            pipe.producer_acquire();
            cuda::memcpy_async(shared, global, 16, pipe);
            pipe.consumer_wait();
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        body = ast.kernels[0].body
        assert isinstance(body[0], SharedMemoryNode)
        assert (
            body[0].vtype == "cuda::pipeline_shared_state<cuda::thread_scope_block, 2>"
        )
        assert body[0].name == "state"

        barrier = body[1]
        assert isinstance(barrier, VariableNode)
        assert barrier.vtype == "cuda::barrier<cuda::thread_scope_block>"
        assert barrier.name == "blockBarrier"

        init_call = body[2]
        assert isinstance(init_call, FunctionCallNode)
        assert init_call.name == "init"
        assert isinstance(init_call.args[0], UnaryOpNode)
        assert init_call.args[0].op == "&"
        assert init_call.args[0].operand == "blockBarrier"
        assert isinstance(init_call.args[1], CudaBuiltinNode)

        barrier_copy = body[3]
        assert isinstance(barrier_copy, FunctionCallNode)
        assert barrier_copy.name == "cuda::memcpy_async"
        assert barrier_copy.args[0:2] == ["shared", "global"]
        assert isinstance(barrier_copy.args[2], BinaryOpNode)
        assert barrier_copy.args[3] == "blockBarrier"

        pipe = body[4]
        assert isinstance(pipe, VariableNode)
        assert pipe.vtype == "auto"
        assert isinstance(pipe.value, FunctionCallNode)
        assert pipe.value.name == "cuda::make_pipeline"

        acquire = body[5]
        assert isinstance(acquire, FunctionCallNode)
        assert isinstance(acquire.name, MemberAccessNode)
        assert acquire.name.object == "pipe"
        assert acquire.name.member == "producer_acquire"

        pipeline_copy = body[6]
        assert isinstance(pipeline_copy, FunctionCallNode)
        assert pipeline_copy.name == "cuda::memcpy_async"
        assert pipeline_copy.args == ["shared", "global", "16", "pipe"]

        wait = body[7]
        assert isinstance(wait, FunctionCallNode)
        assert isinstance(wait.name, MemberAccessNode)
        assert wait.name.object == "pipe"
        assert wait.name.member == "consumer_wait"

    def test_cuda_pipeline_primitive_intrinsic_parsing(self):
        code = """
        __global__ void kernel(int* shared, const int* global, __mbarrier_t* barrier) {
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

        params = ast.kernels[0].params
        assert [(param.vtype, param.name) for param in params] == [
            ("int *", "shared"),
            ("const int *", "global"),
            ("__mbarrier_t *", "barrier"),
        ]

        body = ast.kernels[0].body
        first_copy = body[0]
        second_copy = body[1]
        commit = body[2]
        wait = body[3]
        arrive = body[4]

        assert isinstance(first_copy, FunctionCallNode)
        assert first_copy.name == "__pipeline_memcpy_async"
        assert first_copy.args == ["shared", "global", "16"]

        assert isinstance(second_copy, FunctionCallNode)
        assert second_copy.name == "__pipeline_memcpy_async"
        assert isinstance(second_copy.args[0], BinaryOpNode)
        assert isinstance(second_copy.args[1], BinaryOpNode)
        assert second_copy.args[2:] == ["16", "0"]

        assert isinstance(commit, FunctionCallNode)
        assert commit.name == "__pipeline_commit"
        assert commit.args == []

        assert isinstance(wait, FunctionCallNode)
        assert wait.name == "__pipeline_wait_prior"
        assert wait.args == ["1"]

        assert isinstance(arrive, FunctionCallNode)
        assert arrive.name == "__pipeline_arrive_on"
        assert arrive.args == ["barrier"]

    def test_fixed_arrays_and_initializer_lists_parsing(self):
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

        assert ast.global_variables[0].vtype == "float[4]"
        assert isinstance(ast.global_variables[0].value, InitializerListNode)
        assert ast.structs[0].members[0].vtype == "float[3]"
        assert ast.structs[0].members[1].vtype == "float[2][2]"
        assert ast.kernels[0].params[0].vtype == "float[4]"
        assert ast.kernels[0].body[0].vtype == "float[2]"
        assert isinstance(ast.kernels[0].body[0].value, InitializerListNode)

    def test_designated_initializer_lists_parsing(self):
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

        body = ast.functions[0].body
        values = body[0].value.elements
        assert isinstance(values[0], DesignatedInitializerNode)
        assert values[0].designators == [("index", "2")]
        assert values[0].value == "7"
        assert values[1].designators == [("index", "0")]
        assert values[1].value == "1"

        point = body[1].value.elements
        assert point[0].designators == [("field", "y")]
        assert point[0].value == "2.0f"
        assert point[1].designators == [("field", "x")]
        assert point[1].value == "1.0f"

        points = body[2].value.elements
        assert points[0].designators == [("index", "0"), ("field", "y")]
        assert points[0].value == "2.0f"
        assert points[1].designators == [("index", "1"), ("field", "x")]
        assert points[1].value == "3.0f"

    def test_qualified_declarations_parsing(self):
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

        assert ast.global_variables[0].vtype == "float"
        assert ast.global_variables[1].vtype == "unsigned int"
        assert ast.kernels[0].params[0].vtype == "unsigned int *"
        assert ast.kernels[0].params[1].vtype == "const float"
        assert ast.kernels[0].body[0].vtype == "const int"
        assert ast.kernels[0].body[1].vtype == "unsigned int"
        assert ast.kernels[0].body[2].vtype == "float"

    def test_identifier_multiply_expression_is_not_declaration(self):
        code = """
        __global__ void kernel() {
            a * b;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        assert isinstance(ast.kernels[0].body[0], BinaryOpNode)

    def test_local_pointer_declarations_and_unary_pointer_expressions(self):
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

        body = ast.functions[0].body
        assert body[0].vtype == "float *"
        assert body[1].vtype == "const float *"
        assert body[2].vtype == "unsigned int *"
        assert isinstance(body[4].value, UnaryOpNode)
        assert body[4].value.op == "&"
        assert isinstance(body[5], AssignmentNode)
        assert isinstance(body[5].left, UnaryOpNode)
        assert body[5].left.op == "*"
        assert isinstance(body[5].right, UnaryOpNode)
        assert body[5].right.op == "*"

    def test_long_long_type_parsing(self):
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

        kernel = ast.kernels[0]
        assert kernel.params[0].vtype == "unsigned long long *"
        assert kernel.params[1].vtype == "long long"
        assert kernel.body[0].vtype == "unsigned long long"
        assert kernel.body[1].vtype == "long long"
        assert isinstance(kernel.body[1].value, CastNode)
        assert kernel.body[1].value.target_type == "long long"

    def test_pointer_member_access_operator_parsing(self):
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

        body = ast.functions[0].body
        pointer_read = body[0].value
        value_read = body[1].value
        pointer_write = body[2].left

        assert isinstance(pointer_read, MemberAccessNode)
        assert pointer_read.object == "p"
        assert pointer_read.member == "value"
        assert pointer_read.is_pointer is True
        assert isinstance(value_read, MemberAccessNode)
        assert value_read.object == "v"
        assert value_read.member == "value"
        assert value_read.is_pointer is False
        assert isinstance(pointer_write, MemberAccessNode)
        assert pointer_write.object == "p"
        assert pointer_write.member == "value"
        assert pointer_write.is_pointer is True

    def test_resource_member_access_through_cast_and_dereference_parsing(self):
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

        body = ast.functions[0].body
        sampled_target = body[0].value.args[0]
        loaded_target = body[1].value.args[0]
        write_target = body[2].args[1]

        assert isinstance(sampled_target, MemberAccessNode)
        assert sampled_target.member == "tex"
        assert sampled_target.is_pointer is True
        assert isinstance(sampled_target.object, CastNode)
        assert sampled_target.object.target_type == "ResourcePair *"
        assert sampled_target.object.expression == "raw"
        assert isinstance(loaded_target, MemberAccessNode)
        assert loaded_target.member == "surf"
        assert loaded_target.is_pointer is False
        assert isinstance(loaded_target.object, UnaryOpNode)
        assert loaded_target.object.op == "*"
        assert loaded_target.object.operand == "pairPtr"
        assert isinstance(write_target, MemberAccessNode)
        assert write_target.member == "surf"
        assert write_target.is_pointer is True
        assert isinstance(write_target.object, CastNode)
        assert write_target.object.target_type == "ResourcePair *"
        assert write_target.object.expression == "raw"

    def test_bitwise_logical_and_shift_expression_parsing(self):
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

        body = ast.functions[0].body
        assert isinstance(body[0].value, BinaryOpNode)
        assert body[0].value.op == "|"
        assert body[0].value.left.op == "&"
        assert body[0].value.right.op == "^"
        assert body[1].value.op == "<<"
        assert body[2].value.op == ">>"
        assert body[3].condition.op == "||"
        assert body[3].condition.left.op == "&&"
        assert body[3].if_body[0].operator == "<<="
        assert body[3].if_body[1].operator == ">>="

    def test_numeric_literal_parsing(self):
        code = """
        unsigned int helper() {
            unsigned int mask = 0xffu;
            unsigned int bits = 0b1010u;
            unsigned int oct = 0777u;
            float x = 1e-3f;
            float y = .5f;
            int batch = 1'000;
            int tokens = 1'000'000;
            return mask | bits | oct;
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        body = ast.functions[0].body
        assert body[0].value == "0xffu"
        assert body[1].value == "0b1010u"
        assert body[2].value == "0777u"
        assert body[3].value == "1e-3f"
        assert body[4].value == ".5f"
        assert body[5].value == "1'000"
        assert body[6].value == "1'000'000"

    def test_character_literal_parsing(self):
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

        body = ast.functions[0].body
        assert body[0].value == "'x'"
        assert body[1].value == "'\\n'"
        assert body[2].value == "'\\x7f'"
        assert body[3].value == "'\\377'"

    def test_qualified_and_pointer_return_functions_parsing(self):
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

        assert ast.functions[0].return_type == "unsigned int"
        assert ast.functions[1].return_type == "float *"
        assert ast.functions[1].params[0].vtype == "float *"
        assert ast.functions[2].return_type == "const float *"
        assert ast.functions[2].params[0].vtype == "const float *"
        assert ast.functions[3].return_type == "unsigned int"
        assert "static" in ast.functions[3].qualifiers
        assert "inline" in ast.functions[3].qualifiers

    def test_cuda_vector_return_helpers_with_inline_spelling_parsing(self):
        code = """
        __host__ __device__ __inline__ uint2 encodeTextureObject(
            cudaTextureObject_t tex, bool normalized
        ) {
            return uint2();
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        helper = ast.functions[0]
        assert helper.name == "encodeTextureObject"
        assert helper.return_type == "uint2"
        assert "__inline__" in helper.qualifiers
        assert helper.params[0].vtype == "cudaTextureObject_t"
        assert helper.params[1].vtype == "bool"

    def test_cuda_overloaded_operator_function_parsing(self):
        code = """
        __forceinline__ __device__ float2 operator+(float2 a, float2 b) {
            return make_float2(a.x + b.x, a.y + b.y);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        overload = ast.functions[0]
        assert overload.name == "operator+"
        assert overload.return_type == "float2"
        assert overload.params[0].vtype == "float2"
        assert overload.params[1].vtype == "float2"
        assert isinstance(overload.body[0], ReturnNode)

    def test_control_flow_and_cast_expression_parsing(self):
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

        body = ast.functions[0].body
        assert isinstance(body[1], DoWhileNode)
        assert isinstance(body[2], SwitchNode)
        assert len(body[2].cases) == 1
        assert len(body[2].default_case) == 2
        assert isinstance(body[3].value, TernaryOpNode)
        assert isinstance(body[3].value.true_expr, CastNode)

    def test_empty_default_switch_parsing_preserves_default_case(self):
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

        switch = ast.functions[0].body[0]
        assert isinstance(switch, SwitchNode)
        assert len(switch.cases) == 1
        assert switch.default_case == []

    def test_address_of_function_call_argument_is_not_function_pointer_declaration(
        self,
    ):
        code = """
        void bench() {
            cudaEvent_t start;
            cudaEventCreate(&start);
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        call = ast.functions[0].body[1]
        assert isinstance(call, FunctionCallNode)
        assert call.name == "cudaEventCreate"
        assert call.args[0].op == "&"
        assert call.args[0].operand == "start"

    def test_switch_parsing_preserves_default_before_later_case_order(self):
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

        switch = ast.functions[0].body[0]
        assert isinstance(switch, SwitchNode)
        assert [case.value for case in switch.ordered_cases] == [None, "1"]
        assert len(switch.cases) == 1
        assert len(switch.default_case) == 1

    def test_switch_parsing_rejects_duplicate_default_labels(self):
        code = """
        void f(int value) {
            switch (value) {
                default:
                    value += 1;
                case 1:
                    value += 2;
                default:
                    value += 3;
            }
        }
        """
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)

        with pytest.raises(SyntaxError, match="duplicate default"):
            parser.parse()
