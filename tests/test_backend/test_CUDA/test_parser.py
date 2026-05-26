"""Test CUDA Parser"""

import pytest
from crosstl.backend.CUDA.CudaLexer import CudaLexer
from crosstl.backend.CUDA.CudaParser import CudaParser
from crosstl.backend.CUDA.CudaAst import (
    ArrayAccessNode,
    AssignmentNode,
    AtomicOperationNode,
    BinaryOpNode,
    CastNode,
    CudaBuiltinNode,
    DeleteNode,
    DesignatedInitializerNode,
    DoWhileNode,
    ForNode,
    FunctionCallNode,
    InitializerListNode,
    KernelLaunchNode,
    MemberAccessNode,
    NewNode,
    RangeForNode,
    ReturnNode,
    ShaderNode,
    SwitchNode,
    TernaryOpNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
)


class TestCudaParser:
    def test_simple_kernel_parsing(self):
        """Test parsing a simple CUDA kernel"""
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

    def test_device_function_parsing(self):
        """Test parsing a device function"""
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

    def test_shared_memory_parsing(self):
        """Test parsing shared memory declaration"""
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

    def test_builtin_variables_parsing(self):
        """Test parsing CUDA built-in variables"""
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
        """Test parsing CUDA vector types"""
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

    def test_constructor_style_vector_declarations_parsing(self):
        """Test CUDA constructor-style local vector declarations"""
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
        """Test CUDA kernel launch configuration parsing"""
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
        """Test CUDA template-id kernel launch parsing"""
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

    def test_computed_kernel_launch_config_parsing(self):
        """Test CUDA computed kernel launch configuration parsing"""
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

    def test_cuda_launch_kernel_api_parsing(self):
        """Test cudaLaunchKernel parses as a kernel launch"""
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

    def test_cuda_launch_kernel_casted_packed_args_parsing(self):
        """Test casted packed args parse in cudaLaunchKernel"""
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
        """Test compound literal packed args parse in cudaLaunchKernel"""
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
        """Test CUDA device lambdas parse into CrossGL pseudo-lambda calls."""
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
        """Test scoped template host vector declarations and methods"""
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
        """Test scoped template host array declarations and methods"""
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
        """Test benchmark-style host fills with scalar type constructors"""
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
        """Test STL host helper parameters with lvalue references"""
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
        """Test C++ named casts parse into CastNode"""
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
        """Test CUDA resource object pointers and three-argument textures parse."""
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

    def test_qualified_resource_object_pointer_array_parsing(self):
        """Test global CUDA resource object pointer arrays with qualifiers parse."""
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

    def test_type_alias_c_style_cast_parsing(self):
        """Test C-style casts to typedef aliases parse as cast nodes"""
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
        """Test comma-separated host setup declarations"""
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
        """Test comma-separated expressions in for updates"""
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
        """Test CUDA grid-stride for-loop updates"""
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
        """Test parsing atomic operations"""
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
        """Test CUDA atomics preserve address-taken lvalue target shape."""
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
        """Test CUDA bitwise atomic operations parse as atomic nodes."""
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

    def test_user_defined_atomic_name_is_not_parsed_as_builtin_atomic(self):
        """Test user-defined atomic names shadow CUDA atomic parsing."""
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
        """Test parsing synchronization functions"""
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
        """Test expression statements using * are not parsed as declarations"""
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
        """Test scalar long long and unsigned long long declarations"""
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
        """Test casted and dereferenced struct resource member access parsing."""
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

    def test_qualified_and_pointer_return_functions_parsing(self):
        """Test qualified scalar and pointer return types"""
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

    def test_control_flow_and_cast_expression_parsing(self):
        """Test CUDA control-flow nodes and cast expressions"""
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)
        ast = parser.parse()

        switch = ast.functions[0].body[0]
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
        lexer = CudaLexer(code)
        tokens = lexer.tokenize()
        parser = CudaParser(tokens)

        with pytest.raises(SyntaxError, match="duplicate default"):
            parser.parse()
