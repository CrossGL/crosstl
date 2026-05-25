"""Test HIP Parser"""

import pytest

from crosstl.backend.HIP.HipAst import (
    AssignmentNode,
    ArrayAccessNode,
    AtomicOperationNode,
    BinaryOpNode,
    CastNode,
    DeleteNode,
    DesignatedInitializerNode,
    DoWhileNode,
    ForNode,
    FunctionNode,
    FunctionCallNode,
    HipBuiltinNode,
    IfNode,
    InitializerListNode,
    KernelNode,
    KernelLaunchNode,
    MemberAccessNode,
    NewNode,
    RangeForNode,
    SyncNode,
    SwitchNode,
    TernaryOpNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
    ReturnNode,
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
        """Test expression statements require semicolons."""
        code = """
        void host() {
            sink()
        }
        """

        with pytest.raises(SyntaxError, match="Expected SEMICOLON"):
            self.parse_code(code)

    def test_missing_semicolon_between_assignments_errors(self):
        """Test adjacent assignments cannot parse without separators."""
        code = """
        void host() {
            x = 1 y = 2;
        }
        """

        with pytest.raises(SyntaxError, match="Expected SEMICOLON"):
            self.parse_code(code)

    def test_missing_semicolon_between_declarations_errors(self):
        """Test adjacent declarations cannot parse without separators."""
        code = """
        void host() {
            int x = 1 float y = 2;
        }
        """

        with pytest.raises(SyntaxError, match="Expected SEMICOLON"):
            self.parse_code(code)

    def test_for_initializer_declaration_keeps_internal_semicolon_parsing(self):
        """Test for-loop declarations still leave semicolons to the for parser."""
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

    def test_hip_flat_builtin_alias_parsing(self):
        """Test HIP flat builtin aliases normalize to dotted builtin nodes."""
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

    def test_fixed_arrays_and_initializer_lists_parsing(self):
        """Test fixed arrays and brace initializer lists"""
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

    def test_user_defined_atomic_name_is_not_parsed_as_builtin_atomic(self):
        """Test user-defined atomic names shadow HIP atomic parsing."""
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
        """Test multiline brace initializer lists"""
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
        """Test C99 designated initializer lists"""
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
        """Test HIP constructor-style local vector declarations"""
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
        """Test HIP kernel launch configuration parsing"""
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
        """Test HIP template-id kernel launch parsing"""
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

    def test_computed_kernel_launch_config_parsing(self):
        """Test HIP computed kernel launch configuration parsing"""
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
        """Test hipLaunchKernelGGL parses as a kernel launch"""
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

    def test_hip_launch_kernel_api_parsing(self):
        """Test hipLaunchKernel parses as a kernel launch"""
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
        """Test hipLaunchKernelGGL accepts a template-id kernel argument"""
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
        """Test casted packed args parse in hipLaunchKernelGGL"""
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
        """Test compound literal packed args parse in hipLaunchKernelGGL"""
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
        """Test HIP runtime status types and literals parse"""
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

    def test_std_chrono_benchmark_expressions_parsing(self):
        """Test namespace-qualified chrono timing expressions"""
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
        """Test HIP device lambdas parse into CrossGL pseudo-lambda calls."""
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
        """Test scoped template host vector declarations and methods"""
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
        """Test scoped template host array declarations and methods"""
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
        """Test benchmark-style host fills with scalar type constructors"""
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
        """Test STL host helper parameters with lvalue references"""
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

    def test_restrict_pointer_qualifier_parsing(self):
        """Test __restrict__ pointer qualifiers in parameters and locals"""
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

        assert [param["type"] for param in ast.statements[0].params] == [
            "const float * __restrict__",
            "float __restrict__ *",
        ]

        body = ast.statements[1].body
        assert [var.vtype for var in body] == [
            "float * __restrict__",
            "const float * __restrict__",
            "float * __restrict__",
            "float * __restrict__",
        ]
        assert [var.name for var in body] == ["p", "cp", "a", "b"]

    def test_rvalue_reference_declarations_parsing(self):
        """Test rvalue references in parameters, locals, and range loops"""
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
        """Test C++ named casts parse into CastNode"""
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
        """Test C++ new/delete host allocation syntax"""
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
        """Test common std::unique_ptr host allocation syntax"""
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
        """Test template arguments preserve spaces between type tokens"""
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
        """Test nested template arguments that close with >>"""
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
        """Test typedef and using aliases for host helper types"""
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

    def test_typedef_multi_declarator_alias_parsing(self):
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

    def test_type_alias_c_style_cast_parsing(self):
        """Test C-style casts to typedef aliases parse as cast nodes"""
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
        """Test auto pointer and reference local declarations"""
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
        """Test comma-separated host setup declarations"""
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
        """Test comma-separated declarations in for initializers"""
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
        """Test comma-separated expressions in for updates"""
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
        """Test C++ range-based for declarations"""
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
        """Test const, unsigned, and static declarations"""
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
        lexer = HipLexer(code)
        tokens = lexer.tokenize()
        parser = HipParser(tokens)
        ast = parser.parse()

        assert ast.statements[0].vtype == "float"
        assert ast.statements[1].vtype == "unsigned int"
        assert ast.statements[2].params[0]["type"] == "unsigned int *"
        assert ast.statements[2].params[1]["type"] == "const float"
        assert ast.statements[2].body[0].vtype == "const int"
        assert ast.statements[2].body[1].vtype == "unsigned int"
        assert ast.statements[2].body[2].vtype == "float"

    def test_qualified_and_pointer_return_functions_parsing(self):
        """Test qualified scalar and pointer return types"""
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

        assert ast.statements[0].return_type == "unsigned int"
        assert ast.statements[1].return_type == "float *"
        assert ast.statements[1].params[0]["type"] == "float *"
        assert ast.statements[2].return_type == "const float *"
        assert ast.statements[2].params[0]["type"] == "const float *"
        assert ast.statements[3].return_type == "unsigned int"
        assert "static" in ast.statements[3].qualifiers
        assert "inline" in ast.statements[3].qualifiers

    def test_template_prefixed_kernel_parsing(self):
        """Test C++ template-prefixed HIP kernels"""
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
        """Test C++ template-prefixed HIP host functions"""
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

    def test_braced_for_body_and_sync_statement_parsing(self):
        """Test braced for bodies and sync calls parse as statements"""
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

    def test_c_style_for_structured_assignment_updates_parsing(self):
        """Test array and member assignment targets in for updates"""
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
        """Test assignments parse inside expressions and associate to the right"""
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
        """Test bitwise, shift, logical, and compound shift expressions"""
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
        """Test integer mask and float suffix literals parse intact"""
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

        body = ast.statements[0].body
        assert body[0].value == "0xffu"
        assert body[1].value == "0b1010u"
        assert body[2].value == "0777u"
        assert body[3].value == "1e-3f"
        assert body[4].value == ".5f"

    def test_boolean_null_and_character_literal_parsing(self):
        """Test bool, null, nullptr, and character literals parse intact"""
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

        body = ast.statements[0].body
        assert body[0].value == "true"
        assert body[1].value == "false"
        assert body[2].value == "'x'"
        assert body[3].value == "'\\n'"
        assert body[4].value == "nullptr"
        assert body[5].value == "NULL"
        assert body[6].value.op == "&&"

    def test_control_flow_and_cast_expression_parsing(self):
        """Test HIP control-flow nodes and cast expressions"""
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
        """Test empty default labels remain distinguishable from no default"""
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
        """Test source switch label order is retained when default is first."""
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
        """Test duplicate switch default labels are rejected."""
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
