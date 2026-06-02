import pytest

from crosstl.backend.HIP.HipAst import (
    ArrayAccessNode,
    AssignmentNode,
    AtomicOperationNode,
    BinaryOpNode,
    CastNode,
    DeleteNode,
    DesignatedInitializerNode,
    DoWhileNode,
    EnumNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    HipAsmNode,
    HipBuiltinNode,
    IfNode,
    InitializerListNode,
    KernelLaunchNode,
    KernelNode,
    MemberAccessNode,
    NewNode,
    RangeForNode,
    ReturnNode,
    StructNode,
    SwitchNode,
    SyncNode,
    TernaryOpNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from crosstl.backend.HIP.HipLexer import HipLexer
from crosstl.backend.HIP.HipParser import HipParser, HipProgramNode


class TestHipParser:
    def parse_code(self, code):
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        return parser.parse()

    def test_missing_semicolon_after_expression_statement_errors(self):
        code = """
        void host() {
            sink()
        }
        """

        with pytest.raises(SyntaxError, match="Expected SEMICOLON"):
            self.parse_code(code)

    def test_missing_semicolon_between_assignments_errors(self):
        code = """
        void host() {
            x = 1 y = 2;
        }
        """

        with pytest.raises(SyntaxError, match="Expected SEMICOLON"):
            self.parse_code(code)

    def test_missing_semicolon_between_declarations_errors(self):
        code = """
        void host() {
            int x = 1 float y = 2;
        }
        """

        with pytest.raises(SyntaxError, match="Expected SEMICOLON"):
            self.parse_code(code)

    def test_for_initializer_declaration_keeps_internal_semicolon_parsing(self):
        code = """
        void host(int n) {
            for (int i = 0; i < n; i++) {
                sink(i);
            }
        }
        """
        ast = self.parse_code(code)

        loop = ast.statements[0].body[0]
        assert isinstance(loop, ForNode)
        assert isinstance(loop.init, VariableNode)
        assert loop.init.name == "i"
        assert isinstance(loop.body[0], FunctionCallNode)

    def test_for_header_expressions_can_continue_after_newlines(self):
        code = """
        __global__ void strided(float* out, unsigned int size) {
            for (unsigned int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
                              i < size;
                              i += hipGridDim_x * hipBlockDim_x) {
                out[i] = i;
            }
        }
        """
        ast = self.parse_code(code)

        loop = ast.statements[0].body[0]
        assert isinstance(loop, ForNode)
        assert isinstance(loop.condition, BinaryOpNode)
        assert loop.condition.op == "<"
        assert isinstance(loop.update, AssignmentNode)
        assert loop.update.operator == "+="

    def test_cpp_stream_expression_can_continue_after_newline(self):
        code = """
        void host() {
            constexpr size_t elements_to_print = 10;
            std::cout << "First " << elements_to_print << " elements: "
                      << format_range(begin, end) << std::endl;
        }
        """
        ast = self.parse_code(code)

        body = ast.statements[0].body
        assert isinstance(body[1], BinaryOpNode)
        assert body[1].op == "<<"

    def test_templated_kernel_launch_can_start_on_next_line(self):
        code = """
        template <int width>
        __global__ void matrix_transpose_kernel(float* out, float* input) {
        }

        void host(float* out, float* input) {
            matrix_transpose_kernel<16>
                <<<grid_dim, block_dim, 0, hipStreamDefault>>>(out, input);
        }
        """
        ast = self.parse_code(code)

        launch = ast.statements[1].body[0]
        assert isinstance(launch, KernelLaunchNode)
        assert launch.kernel_name == "matrix_transpose_kernel<16>"

    def test_else_can_follow_block_on_next_line(self):
        code = """
        void host(unsigned int errors) {
            if (errors) {
                return;
            }
            else {
                report_success();
            }
        }
        """
        ast = self.parse_code(code)

        branch = ast.statements[0].body[0]
        assert isinstance(branch, IfNode)
        assert branch.else_body

    def test_hip_flat_builtin_alias_parsing(self):
        code = """
        __global__ void kernel(float* out) {
            out[hipThreadIdx_x] = hipBlockDim_y + warpSize;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        assignment = ast.statements[0].body[0]
        index = assignment.left.index
        left_value = assignment.right.left
        right_value = assignment.right.right

        assert isinstance(index, HipBuiltinNode)
        assert index.builtin_name == "threadIdx"
        assert index.component == "x"
        assert isinstance(left_value, HipBuiltinNode)
        assert left_value.builtin_name == "blockDim"
        assert left_value.component == "y"
        assert isinstance(right_value, HipBuiltinNode)
        assert right_value.builtin_name == "warpSize"
        assert right_value.component is None

    def test_public_rocm_examples_device_global_variables_parse_as_globals(self):
        code = """
        __device__ auto load_callback_dev = load_callback;
        __device__ __constant__ constexpr float c0 = 299792458.0f;

        __device__ float add(float a, float b) {
            return a + b;
        }
        """
        ast = self.parse_code(code)

        callback = ast.statements[0]
        constant = ast.statements[1]
        function = ast.statements[2]

        assert isinstance(callback, VariableNode)
        assert callback.name == "load_callback_dev"
        assert callback.vtype == "auto"
        assert callback.qualifiers == ["__device__"]
        assert callback.value == "load_callback"

        assert isinstance(constant, VariableNode)
        assert constant.name == "c0"
        assert constant.vtype == "float"
        assert constant.qualifiers == ["__device__", "__constant__", "constexpr"]
        assert constant.value == "299792458.0f"

        assert isinstance(function, FunctionNode)
        assert function.name == "add"
        assert "__device__" in function.qualifiers

    def test_public_hpc_training_const_static_declarations_parse(self):
        code = """
        const static int BLOCKSIZE = 1024;
        const static int GRIDSIZE = 1024, N = 128e07;

        __global__ void reduction(double* output) {
            const static int LOCAL_SIZE = 256;
            __shared__ double local_sum[BLOCKSIZE];
            output[0] = LOCAL_SIZE;
        }
        """
        ast = self.parse_code(code)

        blocksize = ast.statements[0]
        gridsize = ast.statements[1]
        n_value = ast.statements[2]
        kernel = ast.statements[3]
        local_size = kernel.body[0]
        local_sum = kernel.body[1]

        assert isinstance(blocksize, VariableNode)
        assert blocksize.vtype == "const int"
        assert blocksize.qualifiers == ["static"]
        assert blocksize.name == "BLOCKSIZE"

        assert gridsize.vtype == "const int"
        assert gridsize.qualifiers == ["static"]
        assert gridsize.name == "GRIDSIZE"
        assert n_value.vtype == "const int"
        assert n_value.qualifiers == ["static"]
        assert n_value.name == "N"

        assert isinstance(kernel, KernelNode)
        assert local_size.vtype == "const int"
        assert local_size.qualifiers == ["static"]
        assert local_sum.vtype == "double[BLOCKSIZE]"
        assert local_sum.qualifiers == ["__shared__"]

    def test_public_rocm_bit_extract_fixed_width_pointer_declarations_parse(self):
        code = """
        __global__ void bit_extract_kernel(
            uint32_t* d_output,
            const uint32_t* d_input,
            size_t size) {
            uint32_t *d_local, *d_shadow;
            d_output[0] = ((d_input[0] & 0xf00) >> 8);
        }
        """
        ast = self.parse_code(code)

        kernel = ast.statements[0]
        d_local, d_shadow, assignment = kernel.body

        assert kernel.params[0] == {"type": "uint32_t *", "name": "d_output"}
        assert kernel.params[1] == {"type": "const uint32_t *", "name": "d_input"}
        assert isinstance(d_local, VariableNode)
        assert d_local.vtype == "uint32_t *"
        assert d_local.name == "d_local"
        assert isinstance(d_shadow, VariableNode)
        assert d_shadow.vtype == "uint32_t *"
        assert d_shadow.name == "d_shadow"
        assert isinstance(assignment, AssignmentNode)
        assert assignment.right.op == ">>"

    def test_public_hpc_training_template_constant_symbols_parse(self):
        code = """
        template <int R> __constant__ float d_dx[2 * R + 1];

        template <int R>
        __device__ float *scratch;

        using vec = __attribute__((__vector_size__(4 * sizeof(float)))) float;

        __inline__ int ceil(int x, int y) {
            return (x - 1) / y + 1;
        }

        template <int R>
        __global__ void stencil(float* out, int line, int
                slice) {
            out[0] = d_dx<R>[threadIdx.x];
        }
        """
        ast = self.parse_code(code)

        constant = ast.statements[0]
        scratch = ast.statements[1]
        vec_alias = ast.statements[2]
        ceil = ast.statements[3]
        kernel = ast.statements[4]
        assignment = kernel.body[0]

        assert isinstance(constant, VariableNode)
        assert constant.name == "d_dx"
        assert constant.vtype == "float[((2 * R) + 1)]"
        assert constant.qualifiers == ["__constant__"]

        assert isinstance(scratch, VariableNode)
        assert scratch.name == "scratch"
        assert scratch.vtype == "float *"
        assert scratch.qualifiers == ["__device__"]

        assert isinstance(vec_alias, TypeAliasNode)
        assert vec_alias.name == "vec"
        assert (
            "__attribute__((__vector_size__(4*sizeof(float))))" in vec_alias.alias_type
        )
        assert vec_alias.alias_type.endswith("float")

        assert isinstance(ceil, FunctionNode)
        assert ceil.name == "ceil"
        assert ceil.qualifiers == ["__inline__"]

        assert isinstance(kernel, KernelNode)
        assert kernel.params[1] == {"type": "int", "name": "line"}
        assert kernel.params[2] == {"type": "int", "name": "slice"}
        assert isinstance(assignment, AssignmentNode)
        assert isinstance(assignment.right, ArrayAccessNode)
        assert assignment.right.array == "d_dx<R>"
        assert isinstance(assignment.right.index, HipBuiltinNode)
        assert assignment.right.index.builtin_name == "threadIdx"
        assert assignment.right.index.component == "x"

    def test_public_rocm_bandwidth_enum_class_parse_as_top_level_declaration(self):
        code = """
        enum class MemoryMode : unsigned int
        {
            PAGED,
            PINNED
        };

        void run_bandwidth_host_device(const MemoryMode memory_mode) {
            if (memory_mode == MemoryMode::PAGED) {
                return;
            }
        }
        """
        ast = self.parse_code(code)

        enum = ast.statements[0]
        function = ast.statements[1]

        assert isinstance(enum, EnumNode)
        assert enum.name == "MemoryMode"
        assert enum.members == [("PAGED", None), ("PINNED", None)]
        assert enum.underlying_type == "unsigned int"
        assert enum.is_scoped is True
        assert function.params[0]["type"] == "const MemoryMode"
        assert function.params[0]["name"] == "memory_mode"

    def test_public_rocm_bandwidth_multiline_vector_return_type_parsing(self):
        code = """
        std::vector<double>
            run_bandwidth_host_device(const std::vector<size_t>& sizes,
                                      const unsigned int trails) {
            const size_t size_in_bytes = 256;
            const double bandwidth_achieved
                = ((size_in_bytes * trails) / 1e9) / elapsed;
        }
        """
        ast = self.parse_code(code)

        function = ast.statements[0]
        body = function.body

        assert isinstance(function, FunctionNode)
        assert function.return_type == "std::vector<double>"
        assert function.name == "run_bandwidth_host_device"
        assert body[1].vtype == "const double"
        assert body[1].name == "bandwidth_achieved"
        assert isinstance(body[1].value, BinaryOpNode)

    def test_public_rocm_template_braced_construction_and_global_calls_parse(self):
        code = """
        void host(float ref, float error) {
            auto block_sizes = std::vector<std::size_t>{0};
            float norm_ref = ::sqrtf((float)ref);
            float norm_error = ::sqrtf((float)error);
            bool ok = norm_ref > 0.0f ||
                      norm_error < 1.0f;
            double range = double(rangeMax - rangeMin) +
                           1.0;
        }
        """
        ast = self.parse_code(code)

        body = ast.statements[0].body
        assert body[0].name == "block_sizes"
        assert isinstance(body[0].value, FunctionCallNode)
        assert body[0].value.name == "std::vector<std::size_t>"
        assert isinstance(body[1].value, FunctionCallNode)
        assert body[1].value.name == "::sqrtf"
        assert isinstance(body[3].value, BinaryOpNode)
        assert body[3].value.op == "||"
        assert isinstance(body[4].value, BinaryOpNode)
        assert body[4].value.op == "+"

    def test_public_rocm_function_pointer_parameter_and_throw_parse(self):
        code = """
        template <typename T>
        std::string toString(
            T value,
            std::ios_base& (*r)(std::ios_base&) = std::dec) {
            throw std::string("failed");
        }
        """
        ast = self.parse_code(code)

        function = ast.statements[0]
        assert function.name == "toString"
        assert function.params[1]["name"] == "r"
        assert function.params[1]["type"] == "std::ios_base & (*)"
        assert isinstance(function.body[0], FunctionCallNode)
        assert function.body[0].name == "throw"

    def test_public_rocm_static_struct_instance_and_class_members_parse(self):
        code = """
        static struct sdkVersionStr
        {
            int major;
            inline sdkVersionStr() {
                major = 2;
            }
        } sdkVerStr;

        class HIPCommandArgs : public SDKCmdArgsParser
        {
        public:
            virtual ~HIPCommandArgs() {
            }

            HIPCommandArgs() {
                deviceId = 0;
            }

            int parseCommandLine(int argc, char** argv) {
                return 0;
            }
        };
        """
        ast = self.parse_code(code)

        static_struct = ast.statements[0]
        class_node = ast.statements[1]
        assert isinstance(static_struct, StructNode)
        assert static_struct.name == "sdkVersionStr"
        assert isinstance(class_node, StructNode)
        assert class_node.name == "HIPCommandArgs"
        member_names = [member.name for member in class_node.members]
        assert "~HIPCommandArgs" in member_names
        assert "HIPCommandArgs" in member_names
        assert "parseCommandLine" in member_names

    def test_public_rocm_composable_kernel_struct_inheritance_parse(self):
        code = """
        struct topk_softmax_kargs : public ck_tile::TopkSoftmaxHostArgs
        {
            static constexpr ck_tile::index_t M_Tile = 128;
        };

        template <typename PrecType>
        struct GemmConfigComputeV3_2 : public GemmConfigBase
        {
            static constexpr ck_tile::index_t N_Tile = 256;
        };
        """
        ast = self.parse_code(code)

        first_struct = ast.statements[0]
        templated_struct = ast.statements[1]

        assert isinstance(first_struct, StructNode)
        assert first_struct.name == "topk_softmax_kargs"
        assert first_struct.members[0].qualifiers == ["static", "constexpr"]
        assert first_struct.members[0].vtype == "ck_tile::index_t"
        assert first_struct.members[0].name == "M_Tile"

        assert isinstance(templated_struct, StructNode)
        assert templated_struct.name == "GemmConfigComputeV3_2"
        assert templated_struct.members[0].name == "N_Tile"

    def test_public_hip_elaborated_struct_types_parse(self):
        code = """
        struct usageStatistics
        {
            std::uint64_t usedMemCurrent;
        };

        void cpu_timer_start(struct timespec *tstart_cpu);
        double cpu_timer_stop(struct timespec tstart_cpu);

        void getUsageStatistics(hipMemPool_t memPool,
                                struct usageStatistics *statistics)
        {
            size_t bytes = sizeof(struct usageStatistics);
            HIP_CHECK(hipMemPoolGetAttribute(
                memPool,
                hipMemPoolAttrUsedMemCurrent,
                &statistics->usedMemCurrent));
        }

        static int indexPlusValue_comp(const void *a, const void *b)
        {
            const struct indexPlusValue *da = (indexPlusValue *)a;
            const struct indexPlusValue *db = (indexPlusValue *)b;

            return da->idx < db->idx ? -1 : da->idx > db->idx;
        }
        """
        ast = self.parse_code(code)

        timer_start = ast.statements[1]
        timer_stop = ast.statements[2]
        usage_stats = ast.statements[3]
        comparator = ast.statements[4]

        assert timer_start.params[0] == {
            "type": "struct timespec *",
            "name": "tstart_cpu",
        }
        assert timer_stop.params[0] == {
            "type": "struct timespec",
            "name": "tstart_cpu",
        }
        assert usage_stats.params[1] == {
            "type": "struct usageStatistics *",
            "name": "statistics",
        }
        sizeof_decl = usage_stats.body[0]
        assert isinstance(sizeof_decl.value, FunctionCallNode)
        assert sizeof_decl.value.name == "sizeof"
        assert sizeof_decl.value.args == ["struct usageStatistics"]

        assert comparator.qualifiers == ["static"]
        assert comparator.body[0].vtype == "const struct indexPlusValue *"
        assert comparator.body[0].name == "da"
        assert isinstance(comparator.body[0].value, CastNode)
        assert comparator.body[0].value.target_type == "indexPlusValue *"
        assert comparator.body[1].vtype == "const struct indexPlusValue *"

    def test_public_rocm_runtime_compilation_adjacent_raw_kernel_string_parsing(self):
        code = r"""
        static constexpr auto saxpy_kernel{
            "#include \"test_header.h\"\n"
            "#include \"test_header1.h\"\n"
            R"(
        extern "C" __global__ void saxpy_kernel()
        {
        }
        )"};
        """
        ast = self.parse_code(code)

        declaration = ast.statements[0]
        initializer = declaration.value

        assert isinstance(declaration, VariableNode)
        assert isinstance(initializer, InitializerListNode)
        assert len(initializer.elements) == 1
        assert "test_header1.h" in initializer.elements[0]
        assert 'extern "C" __global__ void saxpy_kernel' in initializer.elements[0]

    def test_public_rocm_opengl_interop_raw_shader_string_parsing(self):
        code = """
        constexpr const char* vertex_shader = R"(
        #version 330 core

        in float in_height;

        void main()
        {
            gl_Position = vec4(in_height, 0, 1, 1);
        }
        )";
        """
        ast = self.parse_code(code)

        declaration = ast.statements[0]

        assert isinstance(declaration, VariableNode)
        assert declaration.name == "vertex_shader"
        assert declaration.vtype == "const char *"
        assert declaration.value.startswith('R"(')
        assert "#version 330 core" in declaration.value

    def test_public_rocm_opengl_interop_unknown_pointer_return_type_parsing(self):
        code = """
        GLFWwindow* create_window(const int initial_width, const int initial_height)
        {
            return nullptr;
        }
        """
        ast = self.parse_code(code)

        function = ast.statements[0]

        assert isinstance(function, FunctionNode)
        assert function.return_type == "GLFWwindow *"
        assert function.name == "create_window"
        assert function.params[0] == {"type": "const int", "name": "initial_width"}

    def test_public_rocm_opengl_interop_unknown_pointer_local_declaration(self):
        code = """
        void host() {
            GLFWwindow* const window = create_window(initial_width, initial_height);
            foo * bar;
        }
        """
        ast = self.parse_code(code)

        declaration = ast.statements[0].body[0]
        expression = ast.statements[0].body[1]

        assert isinstance(declaration, VariableNode)
        assert declaration.vtype == "GLFWwindow * const"
        assert declaration.name == "window"
        assert isinstance(declaration.value, FunctionCallNode)
        assert isinstance(expression, BinaryOpNode)
        assert expression.op == "*"

    def test_public_rocm_opengl_interop_struct_constructor_body_skips_balanced(self):
        code = """
        struct renderer
        {
            GLuint vao;

            renderer()
            {
                glGenVertexArrays(1, &this->vao);
            }

            renderer(const renderer&) = delete;
        };

        int main() {
            return 0;
        }
        """
        ast = self.parse_code(code)

        struct = ast.statements[0]
        function = ast.statements[1]

        assert isinstance(struct, StructNode)
        assert struct.name == "renderer"
        assert len(struct.members) == 1
        assert struct.members[0].name == "vao"
        assert isinstance(function, FunctionNode)
        assert function.name == "main"

    def test_public_rocm_macro_call_statement_can_omit_semicolon(self):
        code = """
        void host(int num_streams, hipStream_t* streams) {
            HIP_CHECK(hipGetLastError())
            HIP_CHECK(hipMemcpy(dst, src, size, hipMemcpyDeviceToHost));
            for(int i = 0; i < num_streams; i++)
            {
                HIP_CHECK(hipStreamDestroy(streams[i]))
            }
        }
        """
        ast = self.parse_code(code)

        first_call = ast.statements[0].body[0]
        second_call = ast.statements[0].body[1]
        loop = ast.statements[0].body[2]
        statement = loop.body[0]

        assert isinstance(first_call, FunctionCallNode)
        assert first_call.name == "HIP_CHECK"
        assert isinstance(second_call, FunctionCallNode)
        assert second_call.name == "HIP_CHECK"
        assert isinstance(loop, ForNode)
        assert isinstance(statement, FunctionCallNode)
        assert statement.name == "HIP_CHECK"

    def test_public_rocm_hipblaslt_templated_constructor_declaration(self):
        code = """
        int main() {
            runner<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, float>
                runner(1024, 512, 1024, 1, 1.f, 1.f, 32 * 1024 * 1024);
            runner.run([&runner] { gemm_ext(runner.handle); });
        }
        """
        ast = self.parse_code(code)

        declaration = ast.statements[0].body[0]
        run_call = ast.statements[0].body[1]

        assert isinstance(declaration, VariableNode)
        assert declaration.vtype == (
            "runner<hipblasLtHalf, hipblasLtHalf, hipblasLtHalf, float, float>"
        )
        assert declaration.name == "runner"
        assert isinstance(declaration.value, FunctionCallNode)
        assert declaration.value.name == declaration.vtype
        assert isinstance(run_call, FunctionCallNode)
        assert isinstance(run_call.name, MemberAccessNode)
        assert run_call.name.object == "runner"
        assert run_call.name.member == "run"

    def test_public_rocm_hip_doc_constexpr_qualified_size_t_global(self):
        code = """
        constexpr std::size_t const_array_size = 32;
        __constant__ double const_array[const_array_size];

        void set_constant_memory(double* values) {
            HIP_CHECK(hipMemcpyToSymbol(
                const_array, values, const_array_size * sizeof(double)));
        }
        """
        ast = self.parse_code(code)

        size = ast.statements[0]
        const_array = ast.statements[1]
        call = ast.statements[2].body[0]

        assert isinstance(size, VariableNode)
        assert size.qualifiers == ["constexpr"]
        assert size.vtype == "std::size_t"
        assert size.name == "const_array_size"
        assert isinstance(const_array, VariableNode)
        assert const_array.vtype == "double[const_array_size]"
        assert const_array.qualifiers == ["__constant__"]
        assert isinstance(call, FunctionCallNode)
        assert call.name == "HIP_CHECK"

    def test_public_rocm_reduction_structured_binding_and_templated_lambda(self):
        code = """
        template<uint32_t WarpSize>
        void host(int dev_id, unsigned* front, std::size_t count) {
            auto [name, block_size, warp_size] = [dev_id]()
            {
                return std::make_tuple(std::string{prop.name}, 256, 64);
            }();

            auto kernel_op = [] __device__(unsigned lhs, unsigned rhs) {
                return max(lhs, rhs);
            };

            tmp::static_for<WarpSize, tmp::not_equal<0>, tmp::divide<2>>(
                [&]<int I>()
                {
                    sink(I);
                });

            HIP_CHECK(hipMalloc((void**)&front, sizeof(unsigned) * count));
        }
        """
        ast = self.parse_code(code)

        function = ast.statements[0]
        binding = function.body[0]
        kernel_op = function.body[1]
        static_for = function.body[2]
        hip_check = function.body[3]

        assert isinstance(binding, VariableNode)
        assert binding.vtype == "auto"
        assert binding.name == "[name, block_size, warp_size]"
        assert isinstance(binding.value, FunctionCallNode)
        assert isinstance(kernel_op.value, FunctionCallNode)
        assert kernel_op.value.name == "lambda"
        assert isinstance(static_for, FunctionCallNode)
        assert static_for.name == (
            "tmp::static_for<WarpSize, tmp::not_equal<0>, tmp::divide<2>>"
        )
        assert isinstance(static_for.args[0], FunctionCallNode)
        assert static_for.args[0].name == "lambda"
        assert isinstance(hip_check, FunctionCallNode)
        assert hip_check.name == "HIP_CHECK"

    def test_dynamic_shared_memory_parsing_marks_extern_unsized_array(self):
        code = """
        __global__ void kernel() {
            extern __shared__ float shared[];
        }
        """
        ast = self.parse_code(code)

        declaration = ast.statements[0].body[0]
        assert isinstance(declaration, VariableNode)
        assert declaration.vtype == "float[]"
        assert declaration.qualifiers == ["extern", "__shared__"]
        assert declaration.is_extern_shared_memory is True
        assert declaration.is_dynamic_shared_memory is True

    def test_hip_device_property_member_names_can_match_builtin_tokens(self):
        code = """
        void host(hipDeviceProp_t* props_ptr) {
            hipDeviceProp_t props;
            int warp = props.warpSize;
            int pointer_warp = props_ptr->warpSize;
        }
        """
        ast = self.parse_code(code)

        body = ast.statements[0].body
        warp_value = body[1].value
        pointer_warp_value = body[2].value

        assert isinstance(warp_value, MemberAccessNode)
        assert warp_value.object == "props"
        assert warp_value.member == "warpSize"
        assert not warp_value.is_pointer
        assert isinstance(pointer_warp_value, MemberAccessNode)
        assert pointer_warp_value.object == "props_ptr"
        assert pointer_warp_value.member == "warpSize"
        assert pointer_warp_value.is_pointer

    def test_newline_split_initializer_and_builtin_named_members(self):
        code = """
        void host() {
            hipChannelFormatDesc channel_desc
                = hipCreateChannelDesc(8, 0, 0, 0, hipChannelFormatKindUnsigned);
            hipKernelNodeParams params{};
            params.gridDim = dim3(1, 1, 1);
            params.blockDim = dim3(32, 1, 1);
        }
        """
        ast = self.parse_code(code)

        body = ast.statements[0].body
        assert body[0].name == "channel_desc"
        assert isinstance(body[0].value, FunctionCallNode)
        assert body[0].value.name == "hipCreateChannelDesc"
        assert isinstance(body[2].left, MemberAccessNode)
        assert body[2].left.member == "gridDim"
        assert isinstance(body[3].left, MemberAccessNode)
        assert body[3].left.member == "blockDim"

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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        assert isinstance(ast, HipProgramNode)
        assert ast.statements[0].vtype == "float[4]"
        assert isinstance(ast.statements[0].value, InitializerListNode)
        assert ast.statements[1].members[0].vtype == "float[3]"
        assert ast.statements[1].members[1].vtype == "float[2][2]"
        assert ast.statements[2].params[0]["type"] == "float[4]"
        assert ast.statements[2].body[0].vtype == "float[2]"
        assert isinstance(ast.statements[2].body[0].value, InitializerListNode)

    def test_long_long_int_declarations_parsing(self):
        code = """
        void kernel() {
            long long int start = 0;
            unsigned long long int ticks = 1;
        }
        """
        ast = self.parse_code(code)

        body = ast.statements[0].body
        assert body[0].vtype == "long long"
        assert body[0].name == "start"
        assert body[1].vtype == "unsigned long long"
        assert body[1].name == "ticks"

    def test_cpp_function_declarator_spacing_and_qualifiers(self):
        code = """
        template <typename T>
        __global__
        __launch_bounds__(512, 4)
        void
        vector_square(T *C_d, size_t N) {
            C_d[0] = C_d[0];
        }

        __device__ void init_array(float * const a, const unsigned int arraySize) {
            return;
        }

        __host__ __device__ constexpr int round_up(int number, int multiple) {
            return number;
        }

        int main(void) {
            constexpr unsigned int size = 1;
            const unsigned blocks = 512;
            return 0;
        }
        """
        ast = self.parse_code(code)

        kernel = ast.statements[0]
        init_array = ast.statements[1]
        round_up = ast.statements[2]
        main = ast.statements[3]

        assert kernel.name == "vector_square"
        assert kernel.attributes == ["__launch_bounds__(512, 4)"]
        assert kernel.params[0]["type"] == "T *"
        assert init_array.params[0]["type"] == "float * const"
        assert init_array.params[1]["type"] == "const unsigned int"
        assert round_up.name == "round_up"
        assert "constexpr" in round_up.qualifiers
        assert main.params == []
        assert main.body[0].vtype == "unsigned int"
        assert "constexpr" in main.body[0].qualifiers
        assert main.body[1].vtype == "const unsigned int"

    def test_user_defined_atomic_name_is_not_parsed_as_builtin_atomic(self):
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

        helper = ast.statements[0]
        kernel = ast.statements[1]
        shadow_call = kernel.body[0].value
        builtin_call = kernel.body[1]

        assert helper.name == "hipAtomicExch"
        assert isinstance(shadow_call, FunctionCallNode)
        assert not isinstance(shadow_call, AtomicOperationNode)
        assert shadow_call.name == "hipAtomicExch"
        assert isinstance(builtin_call, AtomicOperationNode)

    def test_user_defined_atomic_name_declared_later_is_not_parsed_as_builtin_atomic(
        self,
    ):
        """Test later user-defined atomic names shadow HIP atomic parsing."""
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

        kernel = ast.statements[0]
        helper = ast.statements[1]
        shadow_call = kernel.body[0].value
        builtin_call = kernel.body[1]

        assert helper.name == "hipAtomicExch"
        assert isinstance(shadow_call, FunctionCallNode)
        assert not isinstance(shadow_call, AtomicOperationNode)
        assert shadow_call.name == "hipAtomicExch"
        assert isinstance(builtin_call, AtomicOperationNode)

    def test_multiline_initializer_lists_parsing(self):
        code = """
        void host(float* data, int n) {
            void* packedArgs[] = {
                &data,
                &n,
            };
            float matrix[2][2] = {
                {1.0f, 2.0f},
                {3.0f, 4.0f},
            };
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
        packed_args = body[0]
        assert packed_args.vtype == "void *[]"
        assert isinstance(packed_args.value, InitializerListNode)
        assert packed_args.value.elements[0].op == "&"
        assert packed_args.value.elements[0].operand == "data"
        assert packed_args.value.elements[1].op == "&"
        assert packed_args.value.elements[1].operand == "n"

        matrix = body[1]
        assert matrix.vtype == "float[2][2]"
        assert isinstance(matrix.value, InitializerListNode)
        assert isinstance(matrix.value.elements[0], InitializerListNode)
        assert isinstance(matrix.value.elements[1], InitializerListNode)
        assert matrix.value.elements[0].elements == ["1.0f", "2.0f"]
        assert matrix.value.elements[1].elements == ["3.0f", "4.0f"]

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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[1].body
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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        launch = ast.statements[0].body[2]
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
            scale<float><<<dim3(1), dim3(32)>>>(data, 2.0f);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        host = next(
            statement for statement in ast.statements if statement.name == "host"
        )
        launch = host.body[0]
        assert isinstance(launch, KernelLaunchNode)
        assert launch.kernel_name == "scale<float>"
        assert isinstance(launch.blocks, FunctionCallNode)
        assert launch.blocks.name == "dim3"
        assert launch.blocks.args == ["1"]
        assert isinstance(launch.threads, FunctionCallNode)
        assert launch.threads.name == "dim3"
        assert launch.threads.args == ["32"]
        assert launch.shared_mem is None
        assert launch.stream is None
        assert launch.args == ["data", "2.0f"]

    def test_braced_dim3_kernel_launch_parsing(self):
        code = """
        __global__ void kernel() {
        }

        void host() {
            kernel<<<dim3{1, 1, 1}, dim3{32, 1, 1}>>>();
            auto block = dim3{64, 1, 1};
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        host = next(
            statement for statement in ast.statements if statement.name == "host"
        )
        launch = host.body[0]
        assert isinstance(launch, KernelLaunchNode)
        assert isinstance(launch.blocks, FunctionCallNode)
        assert launch.blocks.name == "dim3"
        assert launch.blocks.args == ["1", "1", "1"]
        assert isinstance(launch.threads, FunctionCallNode)
        assert launch.threads.name == "dim3"
        assert launch.threads.args == ["32", "1", "1"]
        assert isinstance(host.body[1].value, FunctionCallNode)
        assert host.body[1].value.name == "dim3"
        assert host.body[1].value.args == ["64", "1", "1"]

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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        launch = ast.statements[0].body[1]
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

    def test_hip_launch_kernel_ggl_parsing(self):
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

        packed_args = ast.statements[0].body[2]
        assert packed_args.vtype == "void *[]"
        assert isinstance(packed_args.value, InitializerListNode)
        assert packed_args.value.elements[0].op == "&"
        assert packed_args.value.elements[0].operand == "data"
        assert packed_args.value.elements[1].operand == "n"

        launch = ast.statements[0].body[3]
        assert isinstance(launch, KernelLaunchNode)
        assert launch.kernel_name == "kernel"
        assert launch.blocks == "grid"
        assert launch.threads == "block"
        assert launch.shared_mem == "0"
        assert launch.stream == "0"
        assert launch.args == ["packedArgs"]

    def test_hip_launch_kernel_parenthesized_kernel_name_from_corpus(self):
        code = """
        void host(hipStream_t stream) {
            hipLaunchKernelGGL((NormKernel1), dim3(1), dim3(128), 0, stream, 1);
        }
        """

        ast = self.parse_code(code)

        launch = ast.statements[0].body[0]
        assert isinstance(launch, KernelLaunchNode)
        assert launch.kernel_name == "NormKernel1"
        assert isinstance(launch.blocks, FunctionCallNode)
        assert launch.blocks.name == "dim3"
        assert launch.blocks.args == ["1"]
        assert isinstance(launch.threads, FunctionCallNode)
        assert launch.threads.name == "dim3"
        assert launch.threads.args == ["128"]
        assert launch.shared_mem == "0"
        assert launch.stream == "stream"
        assert launch.args == ["1"]

    def test_hip_launch_kernel_api_parsing(self):
        code = """
        void host(float* data, int n, int stream) {
            dim3 grid(16);
            dim3 block(32);
            void* packedArgs[] = { &data, &n };
            hipLaunchKernel((const void*)kernel, grid, block, packedArgs, 0, stream);
            hipLaunchKernel(kernel);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        packed_args = ast.statements[0].body[2]
        assert packed_args.vtype == "void *[]"
        assert isinstance(packed_args.value, InitializerListNode)
        assert packed_args.value.elements[0].op == "&"
        assert packed_args.value.elements[0].operand == "data"
        assert packed_args.value.elements[1].operand == "n"

        launch = ast.statements[0].body[3]
        assert isinstance(launch, KernelLaunchNode)
        assert launch.kernel_name == "kernel"
        assert launch.blocks == "grid"
        assert launch.threads == "block"
        assert launch.shared_mem == "0"
        assert launch.stream == "stream"
        assert launch.args == ["packedArgs"]
        assert isinstance(ast.statements[0].body[4], FunctionCallNode)
        assert ast.statements[0].body[4].name == "hipLaunchKernel"

    def test_templated_hip_launch_kernel_ggl_parsing(self):
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

        host = next(
            statement for statement in ast.statements if statement.name == "host"
        )
        launch = host.body[0]
        assert isinstance(launch, KernelLaunchNode)
        assert launch.kernel_name == "scale<float>"
        assert isinstance(launch.blocks, FunctionCallNode)
        assert launch.blocks.name == "dim3"
        assert launch.blocks.args == ["1"]
        assert isinstance(launch.threads, FunctionCallNode)
        assert launch.threads.name == "dim3"
        assert launch.threads.args == ["32"]
        assert launch.shared_mem == "0"
        assert launch.stream == "0"
        assert launch.args == ["data", "2.0f"]

    def test_hip_launch_kernel_casted_packed_args_parsing(self):
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

        packed_args = ast.statements[0].body[2]
        assert packed_args.vtype == "void * *"
        assert isinstance(packed_args.value, InitializerListNode)

        launch = ast.statements[0].body[3]
        assert isinstance(launch, KernelLaunchNode)
        assert launch.kernel_name == "kernel"
        assert len(launch.args) == 1
        assert isinstance(launch.args[0], CastNode)
        assert launch.args[0].target_type == "void * *"
        assert launch.args[0].expression == "packedArgs"

    def test_hip_launch_kernel_compound_literal_args_parsing(self):
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

        launch = ast.statements[0].body[2]
        assert isinstance(launch, KernelLaunchNode)
        assert len(launch.args) == 1
        assert isinstance(launch.args[0], CastNode)
        assert launch.args[0].target_type == "void * []"
        assert isinstance(launch.args[0].expression, InitializerListNode)
        assert launch.args[0].expression.elements[0].op == "&"
        assert launch.args[0].expression.elements[0].operand == "data"
        assert launch.args[0].expression.elements[1].operand == "n"

    def test_runtime_error_status_parsing(self):
        code = """
        void host(float* data, int n) {
            hipError_t err = hipMalloc((void**)&data, n * sizeof(float));
            if (err != hipSuccess) { return; }
            err = hipDeviceSynchronize();
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
        assert body[0].vtype == "hipError_t"
        assert isinstance(body[0].value, FunctionCallNode)
        assert body[0].value.name == "hipMalloc"
        assert isinstance(body[1], IfNode)
        assert body[1].condition.right == "hipSuccess"
        assert isinstance(body[2], AssignmentNode)
        assert isinstance(body[2].right, FunctionCallNode)
        assert body[2].right.name == "hipDeviceSynchronize"

    def test_cpp17_if_initializer_parsing(self):
        code = """
        int main() {
            if(auto err = hipDeviceSynchronize(); err != hipSuccess)
                return 1;
            return 0;
        }
        """
        ast = self.parse_code(code)

        body = ast.statements[0].body
        assert isinstance(body[0], VariableNode)
        assert body[0].vtype == "auto"
        assert body[0].name == "err"
        assert isinstance(body[0].value, FunctionCallNode)
        assert body[0].value.name == "hipDeviceSynchronize"
        assert isinstance(body[1], IfNode)
        assert body[1].condition.left == "err"
        assert body[1].condition.op == "!="
        assert body[1].condition.right == "hipSuccess"
        assert isinstance(body[1].if_body, ReturnNode)
        assert body[2].value == "0"

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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
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

        body = ast.statements[0].body
        assert body[0].vtype == "std::vector<float>"
        assert body[0].name == "h"
        assert body[0].value.name == "std::vector<float>"
        copy_call = body[1]
        assert isinstance(copy_call, FunctionCallNode)
        assert copy_call.name == "hipMemcpy"
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
            hipMemcpy(d, h.data(), h.size() * sizeof(float),
                      hipMemcpyHostToDevice);
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
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
        assert copy_call.name == "hipMemcpy"
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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        assert ast.statements[0].params[0]["type"] == "std::vector<float> &"
        assert ast.statements[1].return_type == "size_t"
        assert ast.statements[1].params[0]["type"] == "const std::vector<float> &"

    def test_pointer_parameter_declarator_reuses_base_type(self):
        code = """
        void prepare(__half *d_in1, *d_in2) {
            sink(d_in1, d_in2);
        }
        """

        ast = self.parse_code(code)

        function = ast.statements[0]
        assert function.params == [
            {"type": "__half *", "name": "d_in1"},
            {"type": "__half *", "name": "d_in2"},
        ]

    def test_public_rocm_vulkan_qualified_constructor_parameters_parsing(self):
        code = """
        base_dispatch::base_dispatch(PFN_vkGetInstanceProcAddr loader)
            : get_instance_proc_addr(loader), status(0)
        {
            this->get_instance_proc_addr = loader;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        constructor = ast.statements[0]
        assert constructor.name == "base_dispatch::base_dispatch"
        assert constructor.return_type == ""
        assert constructor.params == [
            {"type": "PFN_vkGetInstanceProcAddr", "name": "loader"}
        ]

    def test_default_parameter_values_are_skipped_in_parameter_lists(self):
        code = """
        VkInstance create_instance(const bool with_validation = true,
                                   const int flags = make_flags(1, 2)) {
            return instance;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        function = ast.statements[0]
        assert function.params == [
            {"type": "const bool", "name": "with_validation"},
            {"type": "const int", "name": "flags"},
        ]

    def test_resource_keyword_can_be_contextual_variable_name(self):
        code = """
        VkSurfaceKHR create_surface() {
            VkSurfaceKHR surface;
            sink(&surface);
            return surface;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
        assert body[0].vtype == "VkSurfaceKHR"
        assert body[0].name == "surface"
        assert body[2].value == "surface"

    def test_scoped_member_function_definition_with_trailing_const(self):
        code = """
        VkSurfaceFormatKHR graphics_context::find_surface_format() const {
            return format;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        function = ast.statements[0]
        assert function.return_type == "VkSurfaceFormatKHR"
        assert function.name == "graphics_context::find_surface_format"

    def test_restrict_pointer_qualifier_parsing(self):
        code = """
        __global__ void kernel(const float* __restrict__ input,
                               float __restrict__* output,
                               const double* __restrict public_input) {
            output[threadIdx.x] = input[threadIdx.x];
        }
        void host(float* data) {
            float* __restrict__ p = data;
            const float* __restrict__ cp = data;
            float *__restrict__ a = data, *__restrict__ b = data;
            double* __restrict public_ptr = public_input;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        assert [param["type"] for param in ast.statements[0].params] == [
            "const float * __restrict__",
            "float __restrict__ *",
            "const double * __restrict",
        ]

        body = ast.statements[1].body
        assert [var.vtype for var in body] == [
            "float * __restrict__",
            "const float * __restrict__",
            "float * __restrict__",
            "float * __restrict__",
            "double * __restrict",
        ]
        assert [var.name for var in body] == ["p", "cp", "a", "b", "public_ptr"]

    def test_multiline_parameter_lists_parsing(self):
        code = """
        void resource_lifecycle(
            hipResourceDesc* resourceDesc,
            hipTextureDesc* textureDesc
        ) {
            sink(resourceDesc, textureDesc);
        }

        __global__ void kernel(
            const float* input,
            float* output,
            int n
        ) {
            output[threadIdx.x] = input[threadIdx.x] + n;
        }
        """

        ast = self.parse_code(code)

        host = ast.statements[0]
        assert isinstance(host, FunctionNode)
        assert host.name == "resource_lifecycle"
        assert host.params == [
            {"type": "hipResourceDesc *", "name": "resourceDesc"},
            {"type": "hipTextureDesc *", "name": "textureDesc"},
        ]

        kernel = ast.statements[1]
        assert isinstance(kernel, KernelNode)
        assert kernel.name == "kernel"
        assert kernel.params == [
            {"type": "const float *", "name": "input"},
            {"type": "float *", "name": "output"},
            {"type": "int", "name": "n"},
        ]

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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        assert [param["type"] for param in ast.statements[0].params] == [
            "float &&",
            "const float &&",
        ]

        body = ast.statements[1].body
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
            hipMalloc(reinterpret_cast<void**>(&data), n * sizeof(float));
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
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

    def test_unique_ptr_host_allocation_parsing(self):
        code = """
        void host(int n) {
            std::unique_ptr<float[]> h = std::make_unique<float[]>(n);
            std::unique_ptr<int> q = std::make_unique<int>();
            std::unique_ptr<float[]> owned(new float[n]);
            float* raw = h.get();
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
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
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
        assert body[0].vtype == "std::unique_ptr<const float[]>"
        assert body[0].value.name == "std::make_unique<const float[]>"
        assert body[1].vtype == "std::array<unsigned int, 4>"
        assert body[2].vtype == "std::vector<const unsigned int>"
        assert body[3].value.name.member == "count"
        assert (
            body[3].value.name.object.name
            == "std::chrono::duration_cast<std::chrono::microseconds>"
        )

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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
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

        assert isinstance(ast.statements[0], TypeAliasNode)
        assert ast.statements[0].name == "HostBuffer"
        assert ast.statements[0].alias_type == "std::unique_ptr<float[]>"
        assert isinstance(ast.statements[1], TypeAliasNode)
        assert ast.statements[1].name == "Table"
        assert (
            ast.statements[1].alias_type == "std::vector<std::array<unsigned int, 4>>"
        )

        body = ast.statements[2].body
        assert isinstance(body[0], TypeAliasNode)
        assert body[0].name == "LocalBuffer"
        assert body[0].alias_type == "std::unique_ptr<float[], HostDeleter>"
        assert body[1].vtype == "HostBuffer"
        assert body[2].vtype == "HostBuffer *"
        assert body[3].vtype == "LocalBuffer"
        assert body[4].vtype == "Table"
        assert body[5].name == "consume"

    def test_public_rocm_device_query_namespace_block_parsing(self):
        code = """
        namespace
        {
        constexpr unsigned int col_w = 26;

        double khz_to_mhz(size_t f)
        {
            return f / 1000.;
        }
        }

        namespace helpers
        {
        __device__ float add(float a, float b)
        {
            return a + b;
        }
        }

        int main()
        {
            return 0;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        assert len(ast.statements) == 4
        assert isinstance(ast.statements[0], VariableNode)
        assert ast.statements[0].name == "col_w"
        assert ast.statements[0].qualifiers == ["constexpr"]
        assert isinstance(ast.statements[1], FunctionNode)
        assert ast.statements[1].name == "khz_to_mhz"
        assert isinstance(ast.statements[2], FunctionNode)
        assert ast.statements[2].name == "add"
        assert "__device__" in ast.statements[2].qualifiers
        assert isinstance(ast.statements[3], FunctionNode)
        assert ast.statements[3].name == "main"

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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        assert [(alias.name, alias.alias_type) for alias in ast.statements[:5]] == [
            ("Real", "float"),
            ("RealPtr", "float *"),
            ("Tile", "float[16]"),
            ("Buffer", "std::unique_ptr<float[]>"),
            ("BufferPtr", "std::unique_ptr<float[]> *"),
        ]

        function = ast.statements[5]
        assert function.params[1]["type"] == "Real"
        assert [stmt.vtype for stmt in function.body[:5]] == [
            "Real",
            "RealPtr",
            "Tile",
            "Buffer",
            "BufferPtr",
        ]
        assert function.body[5].name == "consume"

    def test_typedef_struct_alias_parsing(self):
        code = """
        typedef struct Existing *ExistingPtr;
        typedef struct {
            unsigned int x;
            unsigned int y;
        } Pair;

        __global__ void kernel(Pair* pairs) {
            pairs[threadIdx.x].x = 1;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        alias = ast.statements[0]
        pair = ast.statements[1]
        kernel = ast.statements[2]
        assert isinstance(alias, TypeAliasNode)
        assert alias.name == "ExistingPtr"
        assert alias.alias_type == "struct Existing *"
        assert isinstance(pair, StructNode)
        assert pair.name == "Pair"
        assert [(member.vtype, member.name) for member in pair.members] == [
            ("unsigned int", "x"),
            ("unsigned int", "y"),
        ]
        assert isinstance(kernel, KernelNode)
        assert kernel.params[0]["type"] == "Pair *"
        assert kernel.params[0]["name"] == "pairs"

    def test_type_alias_c_style_cast_parsing(self):
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

        body = ast.statements[1].body
        assert len(body) == 1
        assert isinstance(body[0], ReturnNode)
        assert isinstance(body[0].value, CastNode)
        assert body[0].value.target_type == "LaneMask"
        assert body[0].value.expression == "x"

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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        first_loop = ast.statements[0].body[0]
        assert isinstance(first_loop, ForNode)
        assert [var.name for var in first_loop.init] == ["i", "j"]
        assert [var.vtype for var in first_loop.init] == ["int", "int"]
        assert first_loop.init[0].value == "0"
        assert first_loop.init[1].value == "n"

        second_loop = ast.statements[0].body[1]
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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        loop = ast.statements[0].body[0]
        assert isinstance(loop, ForNode)
        assert isinstance(loop.update, list)
        assert len(loop.update) == 2
        assert isinstance(loop.update[0], UnaryOpNode)
        assert loop.update[0].op == "++"
        assert loop.update[0].operand == "i"
        assert isinstance(loop.update[1], UnaryOpNode)
        assert loop.update[1].op == "--"
        assert loop.update[1].operand == "j"

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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        loops = ast.statements[0].body
        assert all(isinstance(loop, RangeForNode) for loop in loops)
        assert [(loop.vtype, loop.name, loop.iterable) for loop in loops] == [
            ("auto &", "x", "h"),
            ("const auto &", "x", "h"),
            ("float", "y", "h"),
        ]

    def test_qualified_declarations_parsing(self):
        code = """
        static float cached = 1.0f;
        unsigned int mask = 3u;
        signed int signedMask = -1;
        long long wide = 2ll;
        unsigned long long uwide = 3ull;
        hipArray_t array;

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

        assert ast.statements[0].vtype == "float"
        assert ast.statements[1].vtype == "unsigned int"
        assert ast.statements[2].vtype == "signed int"
        assert ast.statements[3].vtype == "long long"
        assert ast.statements[4].vtype == "unsigned long long"
        assert ast.statements[5].vtype == "hipArray_t"
        assert ast.statements[6].params[0]["type"] == "unsigned int *"
        assert ast.statements[6].params[1]["type"] == "const float"
        assert ast.statements[6].params[2]["type"] == "long long"
        assert ast.statements[6].body[0].vtype == "const int"
        assert ast.statements[6].body[1].vtype == "unsigned int"
        assert ast.statements[6].body[2].vtype == "unsigned long long"
        assert ast.statements[6].body[3].vtype == "long long"
        assert ast.statements[6].body[3].value.target_type == "long long"
        assert ast.statements[6].body[4].vtype == "float"

    def test_qualified_and_pointer_return_functions_parsing(self):
        code = """
        unsigned int lane_mask() { return 3u; }
        float* get_data(float* data) { return data; }
        const float* get_const_data(const float* data) { return data; }
        static inline unsigned int helper(unsigned int x) { return x; }
        __forceinline__ __device__ float fast_add(float a, float b) { return a + b; }
        __noinline__ __device__ float slow_sub(float a, float b) { return a - b; }
        __launch_bounds__(256, 2) __global__ void bounded(float* data) {
            data[threadIdx.x] = 0.0f;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        assert ast.statements[0].return_type == "unsigned int"
        assert ast.statements[1].return_type == "float *"
        assert ast.statements[1].params[0]["type"] == "float *"
        assert ast.statements[2].return_type == "const float *"
        assert ast.statements[2].params[0]["type"] == "const float *"
        assert ast.statements[3].return_type == "unsigned int"
        assert "static" in ast.statements[3].qualifiers
        assert "inline" in ast.statements[3].qualifiers
        assert ast.statements[4].name == "fast_add"
        assert "__forceinline__" in ast.statements[4].qualifiers
        assert "__device__" in ast.statements[4].qualifiers
        assert ast.statements[5].name == "slow_sub"
        assert "__noinline__" in ast.statements[5].qualifiers
        assert "__device__" in ast.statements[5].qualifiers
        assert ast.statements[6].name == "bounded"
        assert ast.statements[6].attributes == ["__launch_bounds__(256, 2)"]

    def test_launch_bounds_after_return_type_kernel_parsing(self):
        code = """
        __global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EXECUTION_UNIT)
        documented_kernel(float* out) {
            out[threadIdx.x] = 1.0f;
        }

        __global__ static __launch_bounds__(BlockSize) void reduction_kernel(float* out) {
            out[threadIdx.x] = 2.0f;
        }
        """
        ast = self.parse_code(code)

        documented_kernel = ast.statements[0]
        reduction_kernel = ast.statements[1]

        assert isinstance(documented_kernel, KernelNode)
        assert documented_kernel.name == "documented_kernel"
        assert documented_kernel.attributes == [
            "__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EXECUTION_UNIT)"
        ]
        assert documented_kernel.params[0]["type"] == "float *"

        assert isinstance(reduction_kernel, KernelNode)
        assert reduction_kernel.name == "reduction_kernel"
        assert reduction_kernel.attributes == ["__launch_bounds__(BlockSize)"]
        assert reduction_kernel.params[0]["type"] == "float *"

    def test_rocwmma_kernel_macro_with_launch_bounds_from_corpus(self):
        code = """
        ROCWMMA_KERNEL void __launch_bounds__(256) gemm(float* out) {
            out[threadIdx.x] = 0.0f;
        }
        """

        ast = self.parse_code(code)
        kernel = ast.statements[0]

        assert isinstance(kernel, KernelNode)
        assert kernel.name == "gemm"
        assert kernel.attributes == ["__launch_bounds__(256)"]
        assert kernel.params[0]["type"] == "float *"

    def test_trailing_return_type_functions_from_rocm_corpus(self):
        code = """
        __device__ auto device_value(unsigned long long dim) -> float {
            return 0.0f;
        }

        auto host_value() -> unsigned long long {
            return 1ull;
        }
        """

        ast = self.parse_code(code)
        device_value = ast.statements[0]
        host_value = ast.statements[1]

        assert isinstance(device_value, FunctionNode)
        assert device_value.return_type == "float"
        assert "__device__" in device_value.qualifiers
        assert device_value.params[0]["type"] == "unsigned long long"
        assert isinstance(device_value.body[0], ReturnNode)

        assert isinstance(host_value, FunctionNode)
        assert host_value.return_type == "unsigned long long"
        assert isinstance(host_value.body[0], ReturnNode)

    def test_c_linkage_kernel_parsing(self):
        code = """
        extern "C"
        __global__ void vector_add(float* output, float* input1, float* input2, size_t size) {
            int i = threadIdx.x;
            if (i < size) {
                output[i] = input1[i] + input2[i];
            }
        }
        """
        ast = self.parse_code(code)

        kernel = ast.statements[0]
        assert isinstance(kernel, KernelNode)
        assert kernel.name == "vector_add"
        assert kernel.qualifiers == ["extern", "__global__"]
        assert kernel.linkage == "C"
        assert kernel.params == [
            {"type": "float *", "name": "output"},
            {"type": "float *", "name": "input1"},
            {"type": "float *", "name": "input2"},
            {"type": "size_t", "name": "size"},
        ]

    def test_c_linkage_block_parsing(self):
        code = """
        extern "C" {
        __global__ void kernel(float* out) {
            out[threadIdx.x] = 1.0f;
        }
        void launch_wrapper() {}
        }
        """
        ast = self.parse_code(code)

        kernel = ast.statements[0]
        wrapper = ast.statements[1]
        assert isinstance(kernel, KernelNode)
        assert kernel.qualifiers == ["extern", "__global__"]
        assert kernel.linkage == "C"
        assert isinstance(wrapper, FunctionNode)
        assert wrapper.qualifiers == ["extern"]
        assert wrapper.linkage == "C"

    def test_hip_opaque_and_declared_type_pointer_declarations_parsing(self):
        code = """
        struct Pair {
            float x;
        };

        hipGraph_t* getGraph(hipGraph_t* graph) {
            return graph;
        }

        Pair* choosePair(Pair* pair) {
            return pair;
        }

        void host() {
            hipGraph_t* graphPtr;
            hipGraphExec_t *execPtr = nullptr;
            Pair* pairPtr = choosePair(nullptr);
            const hipStream_t* streams = nullptr;
            graphPtr = getGraph(graphPtr);
        }

        void expressionHost() {
            foo * bar;
        }
        """

        ast = self.parse_code(code)

        assert ast.statements[0].name == "Pair"
        get_graph = ast.statements[1]
        assert get_graph.return_type == "hipGraph_t *"
        assert get_graph.params == [{"type": "hipGraph_t *", "name": "graph"}]

        choose_pair = ast.statements[2]
        assert choose_pair.return_type == "Pair *"
        assert choose_pair.params == [{"type": "Pair *", "name": "pair"}]

        host_body = ast.statements[3].body
        assert [stmt.vtype for stmt in host_body[:4]] == [
            "hipGraph_t *",
            "hipGraphExec_t *",
            "Pair *",
            "const hipStream_t *",
        ]
        assert host_body[0].name == "graphPtr"
        assert host_body[1].value == "nullptr"
        assert host_body[2].value.name == "choosePair"
        assert host_body[3].value == "nullptr"
        assert isinstance(host_body[4], AssignmentNode)

        expression = ast.statements[4].body[0]
        assert isinstance(expression, BinaryOpNode)
        assert expression.left == "foo"
        assert expression.op == "*"
        assert expression.right == "bar"

    def test_template_prefixed_kernel_parsing(self):
        code = """
        template <typename T>
        __global__ void fill(T* data, T value) {
            data[threadIdx.x] = value;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        kernel = ast.statements[0]
        assignment = kernel.body[0]

        assert isinstance(kernel, KernelNode)
        assert kernel.name == "fill"
        assert kernel.params == [
            {"type": "T *", "name": "data"},
            {"type": "T", "name": "value"},
        ]
        assert isinstance(assignment, AssignmentNode)
        assert isinstance(assignment.left, ArrayAccessNode)
        assert assignment.left.array == "data"

    def test_template_prefixed_host_function_parsing(self):
        code = """
        template <class T>
        __host__ T clampValue(T x) {
            return x;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        function = ast.statements[0]

        assert isinstance(function, FunctionNode)
        assert function.return_type == "T"
        assert function.name == "clampValue"
        assert "__host__" in function.qualifiers
        assert function.params == [{"type": "T", "name": "x"}]
        assert isinstance(function.body[0], ReturnNode)
        assert function.body[0].value == "x"

    def test_template_prefixed_using_alias_with_dependent_type_parsing(self):
        code = """
        template <std::uint32_t WarpSize>
        using lane_mask_t = typename std::conditional<
            WarpSize == 32,
            std::uint32_t,
            std::uint64_t
        >::type;

        template <std::uint32_t WarpSize>
        __global__ void block_reduce(lane_mask_t<WarpSize> mask) {
        }
        """
        ast = self.parse_code(code)

        alias = ast.statements[0]
        kernel = ast.statements[1]

        assert isinstance(alias, TypeAliasNode)
        assert alias.name == "lane_mask_t"
        assert alias.alias_type == (
            "std::conditional<WarpSize==32, std::uint32_t, std::uint64_t>::type"
        )
        assert isinstance(kernel, KernelNode)
        assert kernel.params == [{"type": "lane_mask_t<WarpSize>", "name": "mask"}]

    def test_template_explicit_specialization_function_name_suffix_parsing(self):
        code = """
        template <>
        auto get_elimit<ck_tile::bf16_t>() {
            return 1;
        }
        """
        ast = self.parse_code(code)

        function = ast.statements[0]

        assert isinstance(function, FunctionNode)
        assert function.return_type == "auto"
        assert function.name == "get_elimit<ck_tile::bf16_t>"
        assert isinstance(function.body[0], ReturnNode)
        assert function.body[0].value == "1"

    def test_public_rocm_if_constexpr_condition_parsing(self):
        code = """
        template <typename T>
        int pick_type() {
            if constexpr(std::is_same_v<T, float>) {
                return 1;
            }
            else if constexpr(T::value) {
                return 2;
            }
            return 0;
        }
        """
        ast = self.parse_code(code)

        function = ast.statements[0]
        branch = function.body[0]

        assert isinstance(branch, IfNode)
        assert branch.condition == "std::is_same_v<T, float>"
        assert isinstance(branch.else_body, IfNode)
        assert branch.else_body.condition == "T::value"

    def test_public_rocm_scoped_using_declarations_are_skipped(self):
        code = """
        namespace client
        {
        namespace
        {
        using common::call_stack_t;
        using std::begin, std::end;
        using Alias = common::Thing;
        }
        }
        """
        ast = self.parse_code(code)

        alias = ast.statements[0]

        assert isinstance(alias, TypeAliasNode)
        assert alias.name == "Alias"
        assert alias.alias_type == "common::Thing"

    def test_public_rocm_template_struct_specialization_parsing(self):
        code = """
        template <typename T>
        struct DataTypeTraits;

        template <>
        struct DataTypeTraits<ck_tile::half_t>
        {
            using value_type = ck_tile::half_t;
            static constexpr const char* name = "fp16";
        };
        """
        ast = self.parse_code(code)

        forward_decl = ast.statements[0]
        specialization = ast.statements[1]

        assert isinstance(forward_decl, StructNode)
        assert forward_decl.name == "DataTypeTraits"
        assert isinstance(specialization, StructNode)
        assert specialization.name == "DataTypeTraits<ck_tile::half_t>"
        assert specialization.members[-1].name == "name"
        assert specialization.members[-1].qualifiers == ["static", "constexpr"]

    def test_public_rocm_qualified_operator_call_definition_parsing(self):
        code = """
        struct TensorLayout;
        struct SampleRunner {
            bool operator()(const TensorLayout& layout);
        };

        template <typename InputType>
        bool SampleRunner::operator()(const TensorLayout& layout)
        {
            const auto inputType = getDataTypeEnumFromType<InputType>();
            return true;
        }
        """
        ast = self.parse_code(code)

        function = ast.statements[2]

        assert isinstance(function, FunctionNode)
        assert function.name == "SampleRunner::operator()"
        assert function.params == [{"type": "const TensorLayout &", "name": "layout"}]
        assert function.body[0].vtype == "const auto"
        assert function.body[0].name == "inputType"
        assert function.body[0].value.name == "getDataTypeEnumFromType<InputType>"

    def test_public_rocm_try_block_with_const_local_declarations_parsing(self):
        code = """
        int main() {
            try {
                const auto value = make_value();
                return value;
            }
            catch (...) {
                return -1;
            }
        }
        """
        ast = self.parse_code(code)

        body = ast.statements[0].body

        assert isinstance(body[0], VariableNode)
        assert body[0].vtype == "const auto"
        assert body[0].name == "value"
        assert isinstance(body[1], ReturnNode)
        assert isinstance(body[2], ReturnNode)

    def test_explicit_template_instantiation_parsing(self):
        code = """
        template float moe_smoothquant_<
            trait_<ck_tile::fp16_t, ck_tile::int8_t, 1, true>
        >(const S&, A);
        """
        ast = self.parse_code(code)

        function = ast.statements[0]

        assert isinstance(function, FunctionNode)
        assert function.return_type == "float"
        assert function.name == (
            "moe_smoothquant_<trait_<ck_tile::fp16_t, " "ck_tile::int8_t, 1, true>>"
        )
        assert function.qualifiers == ["template"]
        assert function.params == [
            {"type": "const S &", "name": ""},
            {"type": "A", "name": ""},
        ]
        assert function.body is None

    def test_braced_for_body_and_sync_statement_parsing(self):
        code = """
        void helper(int n) {
            for (int j = 0; j < n; j++) {
                sink(j);
            }
            __syncthreads();
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
        assert isinstance(body[0], ForNode)
        assert isinstance(body[0].init, VariableNode)
        assert body[0].init.name == "j"
        assert isinstance(body[0].body[0], FunctionCallNode)
        assert isinstance(body[1], SyncNode)

    def test_masked_syncwarp_parsing(self):
        code = """
        __global__ void kernel(unsigned int mask) {
            __syncwarp(mask);
            __syncwarp();
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        kernel = ast.statements[0]
        masked_sync = kernel.body[0]
        unmasked_sync = kernel.body[1]

        assert isinstance(kernel, KernelNode)
        assert isinstance(masked_sync, SyncNode)
        assert masked_sync.sync_type == "__syncwarp"
        assert masked_sync.args == ["mask"]
        assert isinstance(unmasked_sync, SyncNode)
        assert unmasked_sync.sync_type == "__syncwarp"
        assert unmasked_sync.args == []

    def test_inline_assembly_parsing(self):
        code = r"""
        __global__ void asmKernel(float* out, float in) {
            asm volatile("v_mov_b32_e32 %0, %1" : "=v"(out[0]) : "v"(in));
            asm __volatile__("s_waitcnt vmcnt(0)" ::: "memory");
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        kernel = ast.statements[0]
        asm_statements = [stmt for stmt in kernel.body if isinstance(stmt, HipAsmNode)]
        assert len(asm_statements) == 2
        assert asm_statements[0].is_volatile is True
        assert asm_statements[0].template == '"v_mov_b32_e32 %0, %1"'
        assert asm_statements[0].outputs[0].constraint == '"=v"'
        assert asm_statements[0].outputs[0].expression.array == "out"
        assert asm_statements[0].inputs[0].constraint == '"v"'
        assert asm_statements[0].inputs[0].expression == "in"
        assert asm_statements[1].outputs == []
        assert asm_statements[1].inputs == []
        assert asm_statements[1].clobbers == ['"memory"']

    def test_c_style_for_structured_assignment_updates_parsing(self):
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

        array_loop = ast.statements[0].body[1]
        assert isinstance(array_loop, ForNode)
        assert isinstance(array_loop.update, AssignmentNode)
        assert array_loop.update.operator == "+="
        assert isinstance(array_loop.update.left, ArrayAccessNode)
        assert array_loop.update.left.array == "values"
        assert array_loop.update.left.index == "i"

        member_loop = ast.statements[0].body[2]
        assert isinstance(member_loop, ForNode)
        assert isinstance(member_loop.update, AssignmentNode)
        assert member_loop.update.operator == "="
        assert isinstance(member_loop.update.left, MemberAccessNode)
        assert member_loop.update.left.object == "object"
        assert member_loop.update.left.member == "field"

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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
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
            a * b;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
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
        assert isinstance(body[7], BinaryOpNode)

    def test_pointer_member_access_operator_parsing(self):
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

        body = ast.statements[1].body
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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
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
            unsigned int max_extent = 0xFFFF'FFFF;
            unsigned int bits = 0b1010u;
            unsigned int lanes = 0b1010'0101u;
            unsigned int oct = 0777u;
            unsigned int count = 1'000u;
            float x = 1e-3f;
            float y = .5f;
            return mask | bits | oct;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
        assert body[0].value == "0xffu"
        assert body[1].value == "0xFFFF'FFFF"
        assert body[2].value == "0b1010u"
        assert body[3].value == "0b1010'0101u"
        assert body[4].value == "0777u"
        assert body[5].value == "1'000u"
        assert body[6].value == "1e-3f"
        assert body[7].value == ".5f"

    def test_boolean_null_and_character_literal_parsing(self):
        code = r"""
        bool helper(int* ptr) {
            bool yes = true;
            bool no = false;
            char c = 'x';
            char escaped = '\n';
            char hex = '\x7f';
            char oct = '\377';
            int* p = nullptr;
            int* q = NULL;
            return yes && !no;
        }
        """
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        body = ast.statements[0].body
        assert body[0].value == "true"
        assert body[1].value == "false"
        assert body[2].value == "'x'"
        assert body[3].value == "'\\n'"
        assert body[4].value == "'\\x7f'"
        assert body[5].value == "'\\377'"
        assert body[6].value == "nullptr"
        assert body[7].value == "NULL"
        assert body[8].value.op == "&&"

    def test_control_flow_and_cast_expression_parsing(self):
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

        body = ast.statements[0].body
        assert isinstance(body[1], WhileNode)
        assert isinstance(body[2], DoWhileNode)
        assert isinstance(body[3], SwitchNode)
        assert len(body[3].cases) == 1
        assert len(body[3].default_case) == 2
        assert isinstance(body[4].value, TernaryOpNode)
        assert isinstance(body[4].value.true_expr, CastNode)

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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        switch = ast.statements[0].body[0]
        assert isinstance(switch, SwitchNode)
        assert len(switch.cases) == 1
        assert switch.default_case == []

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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        switch = ast.statements[0].body[0]
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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)

        with pytest.raises(SyntaxError, match="duplicate default"):
            parser.parse()
