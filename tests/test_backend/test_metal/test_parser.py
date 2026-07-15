from typing import List

import pytest

from crosstl.backend.common_ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    CallNode,
    CastNode,
    DiscardNode,
    DoWhileNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    InitializerListNode,
    MemberAccessNode,
    MethodCallNode,
    RangeForNode,
    TextureSampleNode,
    UnaryOpNode,
    VectorConstructorNode,
    WhileNode,
)
from crosstl.backend.Metal.MetalAst import (
    BlockNode,
    CallableTypeAliasNode,
    EnumNode,
    LambdaNode,
    TypeAliasNode,
)
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import (
    MetalCallOperatorAssociationError,
    MetalParser,
)


def tokenize_code(code: str) -> List:
    lexer = MetalLexer(code)
    return lexer.tokenize()


def parse_code(code: str):
    tokens = tokenize_code(code)
    parser = MetalParser(tokens)
    return parser.parse()


def parse_ok(code: str):
    ast = parse_code(code)
    assert ast is not None
    return ast


def parse_fails(code: str):
    with pytest.raises(SyntaxError):
        parse_code(code)


def iter_ast_nodes(node):
    if node is None or isinstance(node, (str, int, float, bool)):
        return
    if isinstance(node, dict):
        for value in node.values():
            yield from iter_ast_nodes(value)
        return
    if isinstance(node, (list, tuple, set)):
        for value in node:
            yield from iter_ast_nodes(value)
        return
    yield node
    for value in getattr(node, "__dict__", {}).values():
        yield from iter_ast_nodes(value)


def test_parse_numeric_heavy_generated_identifier_function_and_call():
    code = """
    float nvfp4_quantize_float_gs_16_b_4(float value) {
        return value + 1.0;
    }

    kernel void k(device float* out [[buffer(0)]]) {
        out[0] = nvfp4_quantize_float_gs_16_b_4(2.0);
    }
    """
    ast = parse_ok(code)
    helper, kernel = ast.functions
    call = kernel.body[0].right

    assert helper.name == "nvfp4_quantize_float_gs_16_b_4"
    assert isinstance(call, FunctionCallNode)
    assert call.name == "nvfp4_quantize_float_gs_16_b_4"


def test_parse_owner_dependent_constexpr_static_helper_contract():
    code = """
    template <typename Scalar, int Bits, int Word = 8>
    inline constexpr short pack_factor() {
        return Word / Bits;
    }

    template <typename T, int Bits>
    struct Loader {
        static constexpr short factor = pack_factor<T, Bits>();
    };
    """

    ast = parse_ok(code)
    helper = ast.functions[0]
    owner = ast.structs[0]
    factor = owner.members[0]

    assert helper.name == "pack_factor"
    assert helper.template_parameters == [
        ("typename", "Scalar"),
        ("value", "Bits"),
        ("value", "Word"),
    ]
    assert helper.template_parameter_defaults == {"Word": "8"}
    assert helper.declaration_qualifiers == ["inline", "constexpr"]
    assert owner.template_parameters == [
        ("typename", "T"),
        ("value", "Bits"),
    ]
    assert isinstance(factor.default_value, FunctionCallNode)
    assert factor.default_value.name == "pack_factor<T,Bits>"
    assert factor.default_value.args == []


def test_numeric_leading_generated_identifier_reports_specific_error():
    code = """
    float 16_b_4(float value) {
        return value;
    }
    """

    with pytest.raises(SyntaxError) as excinfo:
        parse_code(code)

    error = excinfo.value
    assert "Invalid Metal identifier '16_b_4'" in str(error)
    assert "must start with a letter or underscore" in str(error)
    assert getattr(error, "raw_message", "").startswith("Invalid Metal identifier")


def test_adjacent_numeric_suffix_in_declaration_reports_fragment_and_position():
    code = """
    float generated_quantize_float_gs_16_b_4(float value) {
        float scaled = value 16_b_4;
        return scaled;
    }
    """

    with pytest.raises(SyntaxError) as excinfo:
        parse_code(code)

    error = excinfo.value
    assert "Invalid adjacent numeric suffix fragment '16_b_4'" in str(error)
    assert "function generated_quantize_float_gs_16_b_4" in str(error)
    assert getattr(error, "raw_message", "").startswith(
        "Invalid adjacent numeric suffix fragment"
    )
    assert getattr(error, "source_location", {})["line"] == 3
    assert getattr(error, "source_location", {})["column"] == 30


def test_parse_vertex_fragment_program():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct VertexInput {
        float3 position [[attribute(0)]];
        float2 uv [[attribute(5)]];
    };

    struct VertexOutput {
        float4 position [[position]];
        float2 uv;
    };

    vertex VertexOutput vertex_main(VertexInput in [[stage_in]]) {
        VertexOutput out;
        out.position = float4(in.position, 1.0);
        out.uv = in.uv;
        return out;
    }

    fragment float4 fragment_main(VertexOutput in [[stage_in]]) {
        return float4(in.uv, 0.0, 1.0);
    }
    """
    parse_ok(code)


def test_parse_control_flow_constructs():
    code = """
    void main() {
        int i = 0;
        if (i < 1) {
            i++;
        } else if (i == 1) {
            i += 2;
        } else {
            i -= 1;
        }

        for (int j = 0; j < 4; j++) {
            if (j == 2) {
                continue;
            }
        }

        while (i < 10) {
            i++;
        }

        do {
            i--;
        } while (i > 0);

        switch (i) {
            case 0:
                i = 1;
                break;
            case 1:
                i = 2;
                break;
            default:
                i = 3;
                break;
        }

        return;
    }
    """
    parse_ok(code)


def test_parse_static_branch_attribute_after_if_condition_from_blender_shader():
    # Reduced from:
    # Repo: https://github.com/blender/blender
    # Commit: e5fc656cdab0e682296f8dd024b942b548e788f4
    # Path: source/blender/gpu/shaders/gpu_shader_2D_widget_base.bsl.hh
    code = """
    void draw_widget(bool instanced) {
        if (instanced) [[static_branch]] {
            return;
        }
    }
    """
    ast = parse_ok(code)
    if_node = ast.functions[0].body[0]

    assert isinstance(if_node, IfNode)
    assert if_node.if_chain[0][0].name == "instanced"


def test_parse_top_level_materialized_conversion_operator_fragments():
    code = """
    inline float softmax_exp_float(float x) {
        return x;
    }

    constexpr operator float() const constant {
        return static_cast<float>(real);
    }

    inline half softmax_exp_half(half x) {
        return x;
    }

    constexpr operator half() const constant {
        return static_cast<half>(real);
    }

    inline half use_softmax_exp(half value) {
        return softmax_exp_half(value);
    }
    """
    ast = parse_ok(code)

    assert [function.name for function in ast.functions] == [
        "softmax_exp_float",
        "softmax_exp_half",
        "use_softmax_exp",
    ]


def test_parse_nested_unbraced_for_loops_from_public_msl_example():
    code = """
    fragment float4 shader_day53(float4 pixPos [[position]]) {
        const float PIXEL_SIZE = 40.0;
        float4 col = float4(0.0);
        for (int i = 0; i < int(PIXEL_SIZE); i++)
            for (int j = 0; j < int(PIXEL_SIZE); j++)
                col += float4(float(i + j));
        return col;
    }
    """
    ast = parse_ok(code)
    loops = [node for node in iter_ast_nodes(ast) if isinstance(node, ForNode)]

    assert len(loops) == 2
    assert isinstance(loops[0].body[0], ForNode)


def test_parse_unbraced_while_body_from_msl_cxx14_statement_grammar():
    code = """
    kernel void normalize(device float* values [[buffer(0)]], uint count) {
        uint i = 0;
        while (i < count)
            values[i++] = 0.0f;
    }
    """
    ast = parse_ok(code)
    body = ast.functions[0].body

    assert isinstance(body[1], WhileNode)
    assert len(body[1].body) == 1
    assert isinstance(body[1].body[0], AssignmentNode)


def test_parse_unbraced_do_while_body_from_msl_cxx14_statement_grammar():
    code = """
    kernel void normalize(device float* values [[buffer(0)]], uint count) {
        uint i = 0;
        do
            values[i++] = 0.0f;
        while (i < count);
    }
    """
    ast = parse_ok(code)
    body = ast.functions[0].body

    assert isinstance(body[1], DoWhileNode)
    assert len(body[1].body) == 1
    assert isinstance(body[1].body[0], AssignmentNode)


def test_parse_templated_call_assignment_lhs_from_pytorch_mps_binary_kernel():
    # Reduced from:
    # Repo: https://github.com/pytorch/pytorch
    # Path: aten/src/ATen/native/mps/kernels/BinaryKernel.metal
    code = """
    template<typename T>
    device T& ref_at_offs(device T* ptr, long off);

    template<typename T>
    T val_at_offs(device T* ptr, long off);

    kernel void lerp_kernel(device float* out_ptr [[buffer(0)]],
                            device float* self_ptr [[buffer(1)]],
                            long out_off,
                            long self_off) {
        ref_at_offs<float>(out_ptr, long(out_off)) =
            val_at_offs<float>(self_ptr, long(self_off));
    }
    """
    ast = parse_ok(code)
    assignment = ast.functions[0].body[0]

    assert isinstance(assignment, AssignmentNode)
    assert isinstance(assignment.left, FunctionCallNode)
    assert assignment.left.name == "ref_at_offs<float>"


def test_parse_indexed_reinterpret_cast_assignment_from_pytorch_mps_quantized():
    # Reduced from:
    # Repo: https://github.com/pytorch/pytorch
    # Path: aten/src/ATen/native/mps/kernels/Quantized.metal
    code = """
    struct vecT { float value; };

    kernel void int4pack_mm(device char* output_data [[buffer(0)]],
                            uint m,
                            uint n,
                            uint N,
                            float result) {
        reinterpret_cast<device vecT*>(output_data + m * N)[n / 4] =
            vecT(result);
    }
    """
    ast = parse_ok(code)
    assignment = ast.functions[0].body[0]
    casts = [node for node in iter_ast_nodes(assignment) if isinstance(node, CastNode)]

    assert isinstance(assignment, AssignmentNode)
    assert isinstance(assignment.left, ArrayAccessNode)
    assert len(casts) == 1
    assert casts[0].target_type == "vecT*"
    assert casts[0].qualifiers == ["device"]


def test_parse_reinterpret_cast_retains_pointer_qualifiers():
    code = """
    void aliases(thread float* values) {
        thread float* mutable_values =
            reinterpret_cast<thread float*>(values);
        const thread float* const_values =
            reinterpret_cast<const thread float*>(values);
    }
    """

    ast = parse_ok(code)
    casts = [node for node in iter_ast_nodes(ast) if isinstance(node, CastNode)]

    assert [cast.target_type for cast in casts] == ["float*", "float*"]
    assert [cast.qualifiers for cast in casts] == [
        ["thread"],
        ["const", "thread"],
    ]


def test_parse_global_scoped_metal_array_declaration_from_pytorch_mps_scatter():
    # Reduced from:
    # Repo: https://github.com/pytorch/pytorch
    # Path: aten/src/ATen/native/mps/kernels/ScatterGather.metal
    code = """
    constant uint max_ndim = 16;

    template<typename T>
    void pos_from_thread_index(long tid, thread T* pos, constant T* sizes);

    kernel void scatter(long thread_index,
                        long tid_offset,
                        constant long* index_sizes [[buffer(0)]]) {
        long tid = long(thread_index) + tid_offset;
        ::metal::array<long, max_ndim> pos;
        pos_from_thread_index<long>(tid, &pos[0], index_sizes);
    }
    """
    ast = parse_ok(code)
    body = ast.functions[0].body

    assert body[1].vtype == "metal::array<long,max_ndim>"
    assert body[1].name == "pos"


def test_parse_resource_bindings_and_address_spaces():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Uniforms {
        float4x4 mvp;
    };

    kernel void compute_main(device float* data [[buffer(0)]],
                             constant Uniforms& uniforms [[buffer(1)]],
                             uint tid [[thread_position_in_grid]]) {
        threadgroup float shared[16];
        float value = data[tid];
        data[tid] = (uniforms.mvp[0].x + value);
        shared[tid % 16] = value;
    }
    """
    parse_ok(code)


def test_parse_return_type_before_stage_qualifier_from_metal_cpp_sample():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct v2f
    {
        float4 position [[position]];
        half3 color;
    };

    v2f vertex vertexMain(uint vertexId [[vertex_id]],
                          device const float3* positions [[buffer(0)]],
                          device const float3* colors [[buffer(1)]])
    {
        v2f o;
        o.position = float4(positions[vertexId], 1.0);
        o.color = half3(colors[vertexId]);
        return o;
    }

    half4 fragment fragmentMain(v2f in [[stage_in]])
    {
        return half4(in.color, 1.0);
    }
    """
    ast = parse_ok(code)
    vertex, fragment = ast.functions

    assert vertex.return_type == "v2f"
    assert vertex.qualifier == "vertex"
    assert vertex.name == "vertexMain"
    assert fragment.return_type == "half4"
    assert fragment.qualifier == "fragment"
    assert fragment.name == "fragmentMain"


def test_parse_gnu_attribute_before_local_declaration():
    code = """
    void main(float b) {
        __attribute__((unused)) float zero = b;
    }
    """
    ast = parse_ok(code)

    assignment = ast.functions[0].body[0]
    variable = assignment.left
    assert variable.name == "zero"
    assert variable.vtype == "float"
    assert variable.attributes[0].name == "unused"
    assert variable.attributes[0].args == []


def test_parse_gnu_attribute_before_threadgroup_local_declaration():
    code = """
    void main() {
        __attribute__((unused)) threadgroup float scratch[16];
    }
    """
    ast = parse_ok(code)

    variable = ast.functions[0].body[0]
    assert variable.name == "scratch"
    assert variable.vtype == "float"
    assert variable.qualifiers == ["threadgroup"]
    assert variable.array_sizes == ["16"]
    assert variable.attributes[0].name == "unused"
    assert variable.attributes[0].args == []


def test_parse_gnu_attribute_before_global_constant_declaration():
    code = """
    __attribute__((unused)) constant int MathError_DivisionByZero = 0;
    """
    ast = parse_ok(code)

    constant = ast.global_variables[0]
    assert isinstance(constant, AssignmentNode)
    assert constant.left.name == "MathError_DivisionByZero"
    assert constant.left.vtype == "int"
    assert constant.left.qualifiers == ["constant"]
    assert constant.left.attributes[0].name == "unused"
    assert constant.left.attributes[0].args == []
    assert constant.right == "0"


def test_parse_block_scope_using_decltype_alias_from_mlx_steel_attention():
    code = """
    void resolve_mask() {
        // Reduced from MLX Steel attention mask handling:
        // using stile_t = decltype(Stile);
        // constexpr auto neg_inf = Limits<selem_t>::finite_min;
        int Stile;
        using stile_t = decltype(Stile);
        using selem_t = typename stile_t::elem_type;
        constexpr auto neg_inf = Limits<selem_t>::finite_min;
        constexpr short kRowsPT = stile_t::kRowsPerThread;
    }
    """
    ast = parse_ok(code)

    body = ast.functions[0].body
    # Body-local ``using X = Y;`` aliases are retained as TypeAliasNode
    # statements so codegen can resolve later declarations that reference them.
    aliases = [stmt for stmt in body if isinstance(stmt, TypeAliasNode)]
    assert [alias.name for alias in aliases] == ["stile_t", "selem_t"]
    statements = [stmt for stmt in body if not isinstance(stmt, TypeAliasNode)]
    assert statements[1].left.name == "neg_inf"
    assert statements[1].left.vtype == "auto"
    assert statements[1].right.name == "Limits<selem_t>::finite_min"
    assert statements[2].left.name == "kRowsPT"
    assert statements[2].right.name == "stile_t::kRowsPerThread"


def test_parse_block_scope_decltype_variable_declaration():
    code = """
    void prepare_shape() {
        int lane_count = 4;
        decltype(lane_count) active_lanes = lane_count;
    }
    """
    ast = parse_ok(code)

    decl = ast.functions[0].body[1]
    assert isinstance(decl, AssignmentNode)
    assert decl.left.vtype == "decltype(lane_count)"
    assert decl.left.name == "active_lanes"
    assert decl.right.name == "lane_count"


def test_parse_block_scope_typedef_alias_from_mlx_fp_quantized():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Commit: b155224b9963cd9476363b464a559232a0868000
    # Path: mlx/backend/metal/kernels/fp_quantized.h
    code = """
    void load_quantized() {
        constexpr int values_per_thread = 4;
        typedef float U;
        thread U x_thread[values_per_thread];
        thread U result[8] = {0};
    }
    """
    ast = parse_ok(code)

    body = ast.functions[0].body
    alias = body[1]
    assert isinstance(alias, TypeAliasNode)
    assert alias.name == "U"
    assert alias.alias_type == "float"
    assert alias.source_location is not None
    assert body[2].vtype == "U"
    assert body[2].qualifiers == ["thread"]
    assert body[2].array_sizes[0].name == "values_per_thread"
    assert body[3].left.vtype == "U"
    assert body[3].left.qualifiers == ["thread"]
    assert body[3].left.array_sizes == ["8"]


def test_parse_block_scope_typedef_preserves_declarator_metadata():
    ast = parse_ok("""
        void prepare() {
            typedef const device float* DeviceValues;
            typedef float Lanes[4];
        }
        """)

    pointer_alias, array_alias = ast.functions[0].body
    assert pointer_alias.alias_type == "float*"
    assert pointer_alias.qualifiers == ["const", "device"]
    assert pointer_alias.array_sizes == []
    assert pointer_alias.source_location is not None
    assert array_alias.alias_type == "float"
    assert len(array_alias.array_sizes) == 1
    assert array_alias.array_sizes[0] == "4"
    assert array_alias.source_location is not None


def test_parse_block_scope_typedef_does_not_leak_to_sibling_function():
    code = """
    void first() {
        typedef float LocalScalar;
        LocalScalar value = 1.0f;
    }

    void second() {
        LocalScalar value = 2.0f;
    }
    """

    parser = MetalParser(tokenize_code(code))
    parser.parse()

    assert "LocalScalar" not in parser.known_types


def test_parse_switch_typedef_scope_spans_cases_only():
    source = """
        void select(int mode) {
            switch (mode) {
                case 0:
                    typedef int CaseValue;
                    break;
                case 1:
                    CaseValue value = 1;
                    break;
            }
            int after = 2;
        }
        """
    parser = MetalParser(tokenize_code(source))
    ast = parser.parse()

    switch = ast.functions[0].body[0]
    assert isinstance(switch.cases[0].statements[0], TypeAliasNode)
    assert switch.cases[1].statements[0].left.vtype == "CaseValue"
    assert ast.functions[0].body[1].left.vtype == "int"
    assert "CaseValue" not in parser.known_types


def test_parse_dependent_numeric_template_call_from_mlx_fp_quantized():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Path: mlx/backend/metal/kernels/fp_quantized.h
    code = """
    template <typename U, int values_per_thread, int bits>
    inline U qdot(thread U* result);

    template <typename U, const int group_size, int bits>
    void fp_qvm_impl(thread U* result) {
        constexpr int pack_factor = 8 / bits;
        constexpr int tn = group_size / pack_factor;
        qdot<U, tn * pack_factor, bits>(result);
    }
    """
    ast = parse_ok(code)
    call = ast.functions[0].body[2]

    assert isinstance(call, FunctionCallNode)
    assert call.name == "qdot<U,tn*pack_factor,bits>"


def test_parse_numeric_template_call_without_prior_template_declaration():
    code = """
    void run(device float* out) {
        project_helper<16, 4>(out);
    }
    """
    ast = parse_ok(code)
    call = ast.functions[0].body[0]

    assert isinstance(call, FunctionCallNode)
    assert call.name == "project_helper<16,4>"


def test_parse_numeric_template_braced_functor_call_statement():
    code = """
    struct Quantize {};

    void run(float value) {
        Quantize<4>{}(value);
    }
    """
    ast = parse_ok(code)
    call = ast.functions[0].body[0]

    assert isinstance(call, CallNode)
    assert isinstance(call.callee, FunctionCallNode)
    assert call.callee.name == "Quantize<4>"


def test_parse_struct_empty_member_declaration_left_by_functor_materialization():
    code = """
    struct LogAddExp {
        ;

        float operator()(float x, float y) {
            return x + y;
        }
    };
    """
    ast = parse_ok(code)

    assert [struct.name for struct in ast.structs] == ["LogAddExp"]
    assert ast.structs[0].members == []


def test_numeric_template_parse_error_reports_context_and_position():
    code = """
    template <int N>
    void broken() {
        g<N * 2>(;
    }
    """

    with pytest.raises(SyntaxError) as excinfo:
        parse_code(code)

    error = excinfo.value
    assert "function broken" in str(error)
    assert "line" in str(error)
    assert "column" in str(error)
    assert getattr(error, "declaration_context", None) == "function broken"
    location = getattr(error, "source_location", {})
    assert location["line"] >= 4
    assert location["column"] >= 1


def test_parse_template_struct_default_bool_relational_from_tinygrad_metal():
    # Reduced from:
    # Repo: https://github.com/tinygrad/tinygrad
    # Commit: f9d88d3c3a6536ae28b054fe8881d1c3064e25fd
    # Path: extra/thunder/metal/include/common/common.metal
    code = """
    template<int Start, int End, int Stride, bool=(Start<End)>
    struct unroll_i_in_range {
        template<class F, typename... Args>
        static METAL_FUNC void run(F f, Args... args) {
            return;
        }
    };
    """
    ast = parse_ok(code)
    struct = ast.structs[0]

    assert struct.name == "unroll_i_in_range"
    assert struct.template_parameters == [
        ("value", "Start"),
        ("value", "End"),
        ("value", "Stride"),
    ]


def test_parse_comma_separated_pointer_declarators_keep_own_suffixes():
    code = """
    void main() {
        thread float *a, *b;
        thread float *c, d;
    }
    """
    ast = parse_ok(code)
    body = ast.functions[0].body

    assert [(decl.name, decl.vtype, decl.qualifiers) for decl in body] == [
        ("a", "float*", ["thread"]),
        ("b", "float*", ["thread"]),
        ("c", "float*", ["thread"]),
        ("d", "float", ["thread"]),
    ]


def test_parse_gnu_attribute_before_global_constant_multiline_initializer():
    code = """
    __attribute__((unused)) constant float VALUES[2] = {
        1.0,
        2.0,
    };
    """
    ast = parse_ok(code)

    constant = ast.global_variables[0]
    assert isinstance(constant, AssignmentNode)
    assert constant.left.name == "VALUES"
    assert constant.left.qualifiers == ["constant"]
    assert constant.left.array_sizes == ["2"]
    assert constant.left.attributes[0].name == "unused"
    assert isinstance(constant.right, InitializerListNode)
    assert constant.right.elements == ["1.0", "2.0"]


def test_parse_gnu_attribute_after_function_return_type():
    code = """
    static inline float4 __attribute__((unused))
    helper(float4 value) {
        return value;
    }
    """
    ast = parse_ok(code)

    function = ast.functions[0]
    assert function.name == "helper"
    assert function.return_type == "float4"
    assert function.attributes[0].name == "unused"
    assert function.attributes[0].args == []


def test_parse_gnu_attribute_between_function_qualifiers_and_return_type_from_spirv_cross():
    # Reduced from:
    # Repo: https://github.com/KhronosGroup/SPIRV-Cross
    # Path: reference/shaders-ue4/asm/vert/texture-buffer.asm.vert
    code = """
    static inline __attribute__((always_inline))
    uint2 spvTexelBufferCoord(uint tc) {
        return uint2(tc % 4096, tc / 4096);
    }
    """
    ast = parse_ok(code)
    function = ast.functions[0]

    assert function.name == "spvTexelBufferCoord"
    assert function.return_type == "uint2"
    assert function.attributes[0].name == "always_inline"
    assert function.attributes[0].args == []
    assert function.body[0].value.type_name == "uint2"


def test_parse_coherent_memory_qualifier_from_mlx_fence_kernel():
    code = """
    [[kernel]] void input_coherent(
        volatile coherent(system) device uint* input [[buffer(0)]],
        const constant uint& size [[buffer(1)]],
        uint index [[thread_position_in_grid]]) {
      if (index < size) {
        input[index] = input[index];
      }
    }
    """
    ast = parse_ok(code)
    param = ast.functions[0].params[0]

    assert param.vtype == "uint*"
    assert param.name == "input"
    assert param.qualifiers == ["volatile", "coherent(system)", "device"]
    assert param.attributes[0].name == "buffer"
    assert param.attributes[0].args == ["0"]


def test_parse_resource_memory_qualifiers_on_alias_and_post_pointer_declarations():
    code = """
    using FencePointer = volatile coherent(system) device atomic_uint*;

    kernel void qualify(
        FencePointer aliased [[buffer(0)]],
        device atomic_uint* volatile coherent(device) direct [[buffer(1)]],
        device uint* plain [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
      uint local = index;
    }
    """

    ast = parse_ok(code)
    alias = ast.typedefs[0]
    aliased, direct, plain, index = ast.functions[0].params

    assert alias.name == "FencePointer"
    assert alias.alias_type == "atomic_uint*"
    assert alias.qualifiers == ["volatile", "coherent(system)", "device"]
    assert aliased.vtype == "FencePointer"
    assert aliased.qualifiers == []
    assert direct.qualifiers == ["device", "volatile", "coherent(device)"]
    assert plain.qualifiers == ["device"]
    assert index.qualifiers == []
    assert ast.functions[0].body[0].left.qualifiers == []


def test_parse_scoped_atomic_thread_fence_call_from_mlx_kernel():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void fence() {
        metal::atomic_thread_fence(metal::mem_flags::mem_device);
    }
    """
    ast = parse_ok(code)

    call = ast.functions[0].body[0]
    assert isinstance(call, FunctionCallNode)
    assert call.name == "metal::atomic_thread_fence"
    assert call.args[0].name == "metal::mem_flags::mem_device"


def test_parse_arrays_and_indexing():
    code = """
    struct Data {
        float values[4];
    };

    void main() {
        Data d;
        float v = d.values[2];
        float arr[3];
        arr[1] = v;
    }
    """
    parse_ok(code)


def test_parse_imageblock_member_array_after_attribute_from_apple_sample():
    code = """
    struct TransparentFragmentValues {
        rgba8unorm<half4> colors [[raster_order_group(0)]] [kNumLayers];
        half depths [[raster_order_group(0)]] [kNumLayers];
    };
    """
    ast = parse_ok(code)
    members = ast.structs[0].members

    assert members[0].vtype == "rgba8unorm<half4>"
    assert members[0].array_sizes[0].name == "kNumLayers"
    assert members[1].array_sizes[0].name == "kNumLayers"


def test_parse_typedef_struct_gnu_attribute_from_apple_deferred_sample():
    code = """
    typedef struct __attribute__ ((packed)) packed_float3 {
        float x;
        float y;
        float z;
    } packed_float3;
    """
    ast = parse_ok(code)
    struct = ast.structs[0]

    assert struct.name == "packed_float3"
    assert struct.typedef_tag == "packed_float3"
    assert struct.attributes[0].name == "packed"
    assert [(member.vtype, member.name) for member in struct.members] == [
        ("float", "x"),
        ("float", "y"),
        ("float", "z"),
    ]


def test_parse_struct_named_packed_vector_token_from_apple_raytracing_sample():
    # Reduced from:
    # Repo: https://github.com/donaldwuid/apple_metal_sample_code
    # Commit: 0bc50e5b3670b3169855ab260e8da5ff07b53749
    # Path:
    # MetalSampleCodeLibrary/RayTracing/AcceleratingRayTracingUsingMetal/
    # Renderer/ShaderTypes.h
    code = """
    struct packed_float3 {
        float x;
        float y;
        float z;
    };

    struct Sphere {
        packed_float3 origin;
        float radiusSquared;
        packed_float3 color;
        float radius;
    };
    """
    ast = parse_ok(code)

    assert [struct.name for struct in ast.structs] == ["packed_float3", "Sphere"]
    assert ast.structs[1].members[0].vtype == "packed_float3"
    assert ast.structs[1].members[2].vtype == "packed_float3"


def test_parse_leading_global_scope_expression_from_apple_hdr_sample():
    code = """
    struct GaussSample {
        float2 offset;
        float weight;
    };

    constant GaussSample GaussKernelX[] = {
        {{-2.06278f, 0.f}, 0.05092f},
        {{ 0.53805f, 0.f}, 0.44908f}
    };
    constant size_t GAUSS_KERNEL_SIZE_X = sizeof(GaussKernelX) / sizeof(GaussKernelX[0]);

    half3 BlurredSampleX(float2 texCoords) {
        half3 finalColor = half3(0.f);

        for(uint32_t gaussSampleIdx = 0ul; gaussSampleIdx < ::GAUSS_KERNEL_SIZE_X; ++gaussSampleIdx) {
            constant GaussSample & gaussSample = ::GaussKernelX[gaussSampleIdx];
            finalColor += half3(gaussSample.weight);
        }

        return finalColor;
    }
    """
    ast = parse_ok(code)
    array_accesses = [
        node for node in iter_ast_nodes(ast) if isinstance(node, ArrayAccessNode)
    ]

    assert any(
        getattr(node.array, "name", None) == "GaussKernelX" for node in array_accesses
    )


def test_parse_fragment_output_color_and_depth_attributes():
    code = """
    struct FragmentOutput {
        float4 color0 [[color(0)]];
        float depth [[depth(any)]];
    };
    """
    ast = parse_ok(code)
    members = ast.structs[0].members

    assert members[0].name == "color0"
    assert members[0].attributes[0].name == "color"
    assert members[0].attributes[0].args == ["0"]
    assert members[1].name == "depth"
    assert members[1].attributes[0].name == "depth"
    assert members[1].attributes[0].args == ["any"]


def test_parse_struct_member_default_initializers_from_metal4_basics_pbr():
    code = """
    typedef struct FragmentMaterial {
        float4 baseColor { 1.0f, 1.0f, 1.0f, 1.0f };
        float3 c_diff { 1.0f, 1.0f, 1.0f };
        float metalness = 1.0f;
        float perceptualRoughness = 1.0f;
    } FragmentMaterial;
    """
    ast = parse_ok(code)
    members = ast.structs[0].members

    assert members[0].name == "baseColor"
    assert isinstance(members[0].default_value, InitializerListNode)
    assert members[0].default_value.elements == ["1.0f", "1.0f", "1.0f", "1.0f"]
    assert members[1].name == "c_diff"
    assert isinstance(members[1].default_value, InitializerListNode)
    assert members[2].name == "metalness"
    assert members[2].default_value == "1.0f"
    assert members[3].name == "perceptualRoughness"
    assert members[3].default_value == "1.0f"


def test_parse_argument_buffer_array_of_device_pointers_from_apple_sample():
    code = """
    struct FragmentShaderArguments {
        array<texture2d<float>, AAPLNumTextureArguments> exampleTextures
            [[id(AAPLArgumentBufferIDExampleTextures)]];
        array<device float *, AAPLNumBufferArguments> exampleBuffers
            [[id(AAPLArgumentBufferIDExampleBuffers)]];
        array<uint32_t, AAPLNumBufferArguments> exampleConstants
            [[id(AAPLArgumentBufferIDExampleConstants)]];
    };
    """
    ast = parse_ok(code)
    members = ast.structs[0].members

    assert members[0].vtype == "array<texture2d<float>,AAPLNumTextureArguments>"
    assert members[1].vtype == "array<device float*,AAPLNumBufferArguments>"
    assert members[1].attributes[0].name == "id"
    assert members[1].attributes[0].args == ["AAPLArgumentBufferIDExampleBuffers"]
    assert members[2].vtype == "array<uint32_t,AAPLNumBufferArguments>"


def test_parse_ternary_and_bitwise_expressions():
    code = """
    void main() {
        int a = 3;
        int b = 5;
        int c = (a > b) ? a : b;
        int d = (a & b) | (a ^ b);
        int e = ~a;
        int f = a << 1;
        int g = b >> 1;
    }
    """
    parse_ok(code)


def test_parse_argument_buffers_and_function_constants():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Args {
        float4x4 mvp;
    };

    constant Args& args [[buffer(0), id(3)]];

    constant int gMode [[function_constant(0)]];

    vertex float4 vertex_main(float3 pos [[attribute(0)]]) {
        if (gMode == 1) {
            return args.mvp * float4(pos, 1.0);
        }
        return float4(pos, 1.0);
    }
    """
    parse_ok(code)


def test_parse_host_name_function_attribute_from_apple_spec():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    [[host_name("api_kernel")]]
    kernel void source_kernel(device float* data [[buffer(0)]]) {
        data[0] = 1.0;
    }
    """
    ast = parse_ok(code)
    function = ast.functions[0]

    assert function.name == "source_kernel"
    assert function.qualifier == "kernel"
    assert function.attributes[0].name == "host_name"
    assert function.attributes[0].args == ['"api_kernel"']


def test_parse_argument_buffer_reference_array_parameter():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    fragment float4 my_fragment(
        constant texture2d<float> & texturesAB1 [[buffer(0)]],
        constant texture2d<float> & texturesAB2[10] [[buffer(1)]],
        array<texture2d<float>, 10> texturesArray [[texture(0)]]) {
        return float4(1.0);
    }
    """
    ast = parse_ok(code)
    params = ast.functions[0].params

    assert params[0].vtype == "texture2d<float>&"
    assert params[0].qualifiers == ["constant"]
    assert params[1].vtype == "texture2d<float>&"
    assert params[1].name == "texturesAB2"
    assert params[1].array_sizes == ["10"]
    assert params[1].attributes[0].name == "buffer"
    assert params[1].attributes[0].args == ["1"]
    assert params[2].vtype == "array<texture2d<float>,10>"
    assert params[2].name == "texturesArray"


def test_parse_shader_parameter_attribute_on_following_line_and_texture_arrays():
    code = """
    fragment float4 fragment_main(
        constant LightingModel &lighting_model
        [[buffer(3)]],
        array<texture2d<float>, MAX_SHADOW_CASCADES> shadow_maps [[texture(6)]],
        texture2d_array<float, access::read_write> irradiance_map [[texture(11)]]) {
        return float4(1.0);
    }
    """
    ast = parse_ok(code)
    params = ast.functions[0].params

    assert params[0].name == "lighting_model"
    assert params[0].vtype == "LightingModel&"
    assert params[0].qualifiers == ["constant"]
    assert params[0].attributes[0].name == "buffer"
    assert params[0].attributes[0].args == ["3"]
    assert params[1].name == "shadow_maps"
    assert params[1].vtype == "array<texture2d<float>,MAX_SHADOW_CASCADES>"
    assert params[1].attributes[0].name == "texture"
    assert params[1].attributes[0].args == ["6"]
    assert params[2].name == "irradiance_map"
    assert params[2].vtype == "texture2d_array<float,access::read_write>"
    assert params[2].attributes[0].name == "texture"
    assert params[2].attributes[0].args == ["11"]


def test_parse_fragment_entry_name_on_following_line():
    code = """
    fragment FragmentOutput
    fragment_main(FragmentInput input [[stage_in]]) {
        return FragmentOutput();
    }
    """
    ast = parse_ok(code)
    function = ast.functions[0]

    assert function.qualifier == "fragment"
    assert function.return_type == "FragmentOutput"
    assert function.name == "fragment_main"
    assert function.params[0].attributes[0].name == "stage_in"


def test_parse_defaulted_function_constant_preserves_attribute():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    constant bool useFastPath [[function_constant(3)]] = true;

    fragment float4 fragment_main() {
        if (useFastPath) {
            return float4(1.0);
        }
        return float4(0.0);
    }
    """
    ast = parse_ok(code)
    constant = ast.global_variables[0]
    assert isinstance(constant, AssignmentNode)
    assert constant.left.name == "useFastPath"
    assert constant.left.vtype == "bool"
    assert "constant" in constant.left.qualifiers
    assert constant.left.attributes[0].name == "function_constant"
    assert constant.left.attributes[0].args == ["3"]
    assert constant.right == "true"


def test_parse_global_multi_declarators_from_msl_cxx_declaration_semantics():
    # Apple documents MSL as C++-based; top-level declarations therefore use
    # C++ init-declarator-list semantics rather than requiring one name per line.
    code = """
    #include <metal_stdlib>
    using namespace metal;

    constant float exposure = 1.0f, gamma = 2.2f;
    constant bool useToneMap [[function_constant(0)]],
                  useDither [[function_constant(1)]] = true;

    fragment float4 fragment_main() {
        return float4(exposure + gamma + float(useToneMap || useDither));
    }
    """
    ast = parse_ok(code)
    exposure, gamma, use_tone_map, use_dither = ast.global_variables

    assert isinstance(exposure, AssignmentNode)
    assert exposure.left.name == "exposure"
    assert exposure.left.vtype == "float"
    assert exposure.left.qualifiers == ["constant"]
    assert exposure.right == "1.0f"

    assert isinstance(gamma, AssignmentNode)
    assert gamma.left.name == "gamma"
    assert gamma.left.vtype == "float"
    assert gamma.left.qualifiers == ["constant"]
    assert gamma.right == "2.2f"

    assert use_tone_map.name == "useToneMap"
    assert use_tone_map.vtype == "bool"
    assert use_tone_map.attributes[0].name == "function_constant"
    assert use_tone_map.attributes[0].args == ["0"]

    assert isinstance(use_dither, AssignmentNode)
    assert use_dither.left.name == "useDither"
    assert use_dither.left.attributes[0].name == "function_constant"
    assert use_dither.left.attributes[0].args == ["1"]
    assert use_dither.right == "true"


def test_parse_mlx_steel_const_function_constant_macro_qualifier():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Commit: e9e20fa69184bd38cc0ca12bd9a854c059e59588
    # Path: mlx/backend/metal/kernels/fft.h
    code = """
    #include <metal_common>
    #include "mlx/backend/metal/kernels/steel/defines.h"
    using namespace metal;

    STEEL_CONST bool inv_ [[function_constant(0)]];
    STEEL_CONST int elems_per_thread_ [[function_constant(2)]];
    """
    ast = parse_ok(code)
    inv, elems_per_thread = ast.global_variables

    assert inv.name == "inv_"
    assert inv.vtype == "bool"
    assert inv.qualifiers == ["constant"]
    assert inv.attributes[0].name == "function_constant"
    assert inv.attributes[0].args == ["0"]

    assert elems_per_thread.name == "elems_per_thread_"
    assert elems_per_thread.vtype == "int"
    assert elems_per_thread.qualifiers == ["constant"]
    assert elems_per_thread.attributes[0].name == "function_constant"
    assert elems_per_thread.attributes[0].args == ["2"]


def test_parse_fragment_early_tests_attribute():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    [[early_fragment_tests]]
    fragment float4 fragment_main() {
        return float4(1.0);
    }
    """
    ast = parse_ok(code)
    function = ast.functions[0]

    assert function.qualifier == "fragment"
    assert function.attributes[0].name == "early_fragment_tests"
    assert function.attributes[0].args == []


def test_parse_access_qualified_textures_and_methods():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    fragment float4 fragment_main(texture2d<float, access::read_write> tex [[texture(0)]],
                                  sampler samp [[sampler(0)]]) {
        float4 c = tex.sample(samp, float2(0.5, 0.5));
        float4 r = tex.read(uint2(0, 0));
        tex.write(c, uint2(0, 0));
        float s = tex.sample_compare(samp, float2(0.5, 0.5), 0.5);
        float4 g = tex.gather(samp, float2(0.5, 0.5));
        return c + r + g + float4(s);
    }
    """
    parse_ok(code)


def test_parse_texture_method_ast_shapes():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    fragment float4 fragment_main(texture2d<float, access::read_write> tex [[texture(0)]],
                                  sampler samp [[sampler(0)]]) {
        float4 base = tex.sample(
            samp,
            float2(0.5, 0.5)
        );
        float4 mip = tex.sample(
            samp,
            float2(0.5, 0.5),
            1.0
        );
        float4 readValue = tex.read(uint2(0, 0));
        tex.write(
            base,
            uint2(0, 0)
        );
        float compareValue = tex.sample_compare_level(
            samp, float2(0.5, 0.5), 0.5, 0.0
        );
        float4 gathered = tex.gather_compare(
            samp,
            float2(0.5, 0.5),
            0.5
        );
        return base + mip + readValue + gathered + float4(compareValue);
    }
    """
    ast = parse_ok(code)
    nodes = list(iter_ast_nodes(ast))

    samples = [node for node in nodes if isinstance(node, TextureSampleNode)]
    assert len(samples) == 2
    assert samples[0].lod is None
    assert samples[1].lod is not None

    methods = [node.method for node in nodes if isinstance(node, MethodCallNode)]
    assert {"read", "write", "sample_compare_level", "gather_compare"}.issubset(
        set(methods)
    )
    assert "sample" not in methods


def test_parse_packed_and_simd_types():
    code = """
    struct PackedTypes {
        packed_float4 p4;
        packed_half2 ph2;
        packed_uint3 pu3;
        simd_float3 s3;
        simd_float4x4 s44;
    };
    """
    parse_ok(code)


def test_parse_common_vector_type_aliases():
    code = """
    using Half2 = half2;
    using Float4 = float4;
    typedef packed_float3 PackedFloat3;

    struct AliasedTypes {
        Half2 uv;
        Float4 color;
        PackedFloat3 position;
    };
    """
    ast = parse_ok(code)
    members = ast.structs[0].members

    assert [alias.name for alias in ast.typedefs] == [
        "Half2",
        "Float4",
        "PackedFloat3",
    ]
    assert [alias.alias_type for alias in ast.typedefs] == [
        "half2",
        "float4",
        "packed_float3",
    ]
    assert [member.vtype for member in members] == [
        "Half2",
        "Float4",
        "PackedFloat3",
    ]


def test_parse_ms_and_depth_textures():
    code = """
    fragment float4 fragment_main(texture2d_ms<float> msTex [[texture(0)]],
                                  depth2d_array<float> depthArr [[texture(1)]],
                                  texturecube_array<float> cubeArr [[texture(2)]],
                                  uint2 coord [[position]]) {
        float4 c = msTex.read(coord, 0);
        return c;
    }
    """
    parse_ok(code)


def test_parse_function_tables_and_acceleration_structures():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Payload {
        float3 color;
    };

    visible_function_table<void(Payload&)> vft [[buffer(0), id(1)]];
    intersection_function_table<void(Payload&)> ift [[buffer(1), id(2)]];
    acceleration_structure accel [[buffer(2)]];

    kernel void main(device float* outData [[buffer(3)]]) {
        // No-op
    }
    """
    parse_ok(code)


def test_parse_metal_namespace_types():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    fragment float4 fragment_main(metal::texture2d<float> tex [[texture(0)]],
                                  metal::sampler samp [[sampler(0)]]) {
        return tex.sample(samp, float2(0.5, 0.5));
    }
    """
    parse_ok(code)


def test_parse_scoped_return_type_prototype_from_msl_examples():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    metal::float2x2 rot(float radian);

    metal::float2x2 make_rot(float radian) {
        return metal::float2x2(
            cos(radian), -sin(radian),
            sin(radian), cos(radian)
        );
    }
    """
    ast = parse_ok(code)

    assert [func.name for func in ast.functions] == ["make_rot"]
    assert ast.functions[0].return_type == "metal::float2x2"


def test_parse_raytracing_qualifiers_and_types():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    intersection void isect(raytracing::ray r, intersector inter) {
        // no-op
    }

    anyhit void any_hit() { }
    closesthit void closest_hit() { }
    miss void miss_main() { }
    callable void callable_main() { }
    """
    parse_ok(code)


def test_parse_enum_and_typedef():
    code = """
    typedef int32_t MyInt;
    enum Mode { Off, On = 2, Auto };

    void main() {
        MyInt v = 1;
        Mode m = Auto;
    }
    """
    parse_ok(code)


def test_parse_local_enum_declaration_from_metal_function_body():
    code = """
    fragment float4 local_enum_frag(float4 color [[stage_in]]) {
        enum Mode { ModeA = 0, ModeB = 1 };
        Mode mode = ModeA;
        return mode == ModeA ? color : float4(0.0);
    }
    """
    ast = parse_ok(code)
    function = ast.functions[0]
    local_enum = function.body[0]
    mode_declaration = function.body[1]

    assert isinstance(local_enum, EnumNode)
    assert local_enum.name == "Mode"
    assert local_enum.members == [("ModeA", "0"), ("ModeB", "1")]
    assert isinstance(mode_declaration, AssignmentNode)
    assert mode_declaration.left.vtype == "Mode"
    assert mode_declaration.left.name == "mode"
    assert mode_declaration.right.name == "ModeA"


def test_parse_scoped_enum_with_underlying_type_from_metal4_basics():
    code = """
    enum class BRDF : uint {
        Lambert,
        TrowbridgeReitz,
    };
    """
    ast = parse_ok(code)
    enum = ast.enums[0]

    assert enum.name == "BRDF"
    assert enum.is_scoped is True
    assert enum.underlying_type == "uint"
    assert enum.members == [("Lambert", None), ("TrowbridgeReitz", None)]


def test_parse_class_helper_with_access_labels_from_public_metal_shader():
    # Reduced from:
    # Repo: https://github.com/lambdaclass/lambdaworks
    # Commit: 3c8d8f65546cde6e847dd29b2ef6aefc38c0895a
    # Path: crates/math/src/gpu/metal/shaders/field/mersenne31.h.metal
    code = """
    #include <metal_stdlib>

    class FpMersenne31 {
    public:
        FpMersenne31() = default;
        constexpr FpMersenne31(uint32_t v) : inner(v) {}

        constexpr explicit operator uint32_t() const { return inner; }

        static FpMersenne31 zero() { return FpMersenne31(0); }

        FpMersenne31 operator+(const FpMersenne31 rhs) const {
            return FpMersenne31(inner + rhs.inner);
        }

    private:
        uint32_t inner;

        static uint32_t weak_reduce(uint32_t n) {
            return n;
        }
    };

    kernel void use_field(device uint32_t* out [[buffer(0)]]) {
        FpMersenne31 value;
        out[0] = uint32_t(0);
    }
    """
    ast = parse_ok(code)
    class_node = ast.structs[0]

    assert class_node.name == "FpMersenne31"
    assert getattr(class_node, "aggregate_kind", None) == "class"
    assert [(member.vtype, member.name) for member in class_node.members] == [
        ("uint32_t", "inner")
    ]
    assert ast.functions[0].params[0].vtype == "uint32_t*"
    assert ast.functions[0].body[0].vtype == "FpMersenne31"


def test_parse_virtual_methods_and_destructor_from_current_mlx_allocator_header():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Commit: 01368d8e7888d6989969aa82bc36f2ba09dc5ced
    # Path: mlx/backend/metal/allocator.h
    code = """
    class MetalAllocator : public allocator::Allocator {
    public:
        virtual Buffer malloc(size_t size) override;
        virtual size_t size(Buffer buffer) const override;

    private:
        ~MetalAllocator();
        size_t active_memory_{0};
    };
    """
    ast = parse_ok(code)
    class_node = ast.structs[0]

    assert class_node.name == "MetalAllocator"
    assert getattr(class_node, "aggregate_kind", None) == "class"
    assert class_node.base_types == ["public allocator::Allocator"]
    assert [(member.vtype, member.name) for member in class_node.members] == [
        ("size_t", "active_memory_")
    ]
    assert isinstance(class_node.members[0].default_value, InitializerListNode)


def test_parse_scoped_call_operator_definition_from_mlx_unary_ops():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Commit: 8f0e8b14e0fc028df8618684583af9bef44647b8
    # Path: mlx/backend/metal/kernels/unary_ops.h
    code = """
    struct ArcCos {
        complex64_t operator()(complex64_t x);
    };

    struct Log {
        complex64_t operator()(complex64_t x);
    };

    struct Sqrt {
        complex64_t operator()(complex64_t x);
    };

    complex64_t ArcCos::operator()(complex64_t x) {
        auto i = complex64_t{0.0, 1.0};
        auto y = Log{}(x + i * Sqrt{}(1.0 - x * x));
        return {y.imag, -y.real};
    };
    """
    ast = parse_ok(code)
    function = ast.functions[0]

    assert function.name == "ArcCos::operator()"
    assert function.return_type == "complex64_t"
    assert function.params[0].vtype == "complex64_t"
    assert function.params[0].name == "x"


def test_parse_associates_declared_out_of_line_call_operator_with_trailing_const():
    source = """struct complex64_t { float real; float imag; };
struct ArcCos {
    complex64_t operator()(complex64_t declared_value) const;
};

complex64_t ArcCos::operator()(complex64_t x) const {
    return x;
}
"""
    tokens = MetalLexer(
        source, preprocess=False, file_path="mlx-unary-const.metal"
    ).tokenize()
    ast = MetalParser(tokens, file_path="mlx-unary-const.metal").parse()
    owner = next(struct for struct in ast.structs if struct.name == "ArcCos")
    declaration = owner.call_operator_declarations[0]
    definition = ast.functions[0]

    assert declaration.name == "operator()"
    assert declaration.body is None
    assert declaration.method_qualifiers == ["const"]
    assert declaration.owner_name == "ArcCos"
    assert declaration.declaration_source_location["line"] == 3
    assert definition.name == "ArcCos::operator()"
    assert definition.method_qualifiers == ["const"]
    assert definition.out_of_line_call_operator_owner == "ArcCos"
    assert definition.out_of_line_call_operator_declaration == {
        "owner": "ArcCos",
        "signature": "operator()(complex64_t) const",
        "source_location": declaration.declaration_source_location,
    }
    assert declaration not in definition.__dict__.values()
    assert owner not in definition.__dict__.values()
    assert len(list(iter_ast_nodes(ast))) < 40


def test_parse_rejects_missing_out_of_line_call_operator_declaration_with_location():
    source = """struct complex64_t { float real; float imag; };
struct ArcCos {
    float operator()(float x);
};

complex64_t ArcCos::operator()(complex64_t x) {
    return x;
}
"""
    tokens = MetalLexer(
        source, preprocess=False, file_path="mlx-unary-missing.metal"
    ).tokenize()

    with pytest.raises(MetalCallOperatorAssociationError) as exc_info:
        MetalParser(tokens, file_path="mlx-unary-missing.metal").parse()

    error = exc_info.value
    assert error.project_diagnostic_code == (
        "project.translate.metal-call-operator-association-unresolved"
    )
    assert error.owner == "ArcCos"
    assert error.reason == "no declaration matches the definition contract"
    assert error.source_location["file"] == "mlx-unary-missing.metal"
    assert error.source_location["line"] == 6
    assert error.declaration_locations[0]["line"] == 3
    assert error.candidates == ("operator()(float)",)


def test_parse_rejects_ambiguous_out_of_line_call_operator_declaration():
    source = """struct ArcCos {
    float operator()(float first) const;
    float operator()(float second) const;
};

float ArcCos::operator()(float x) const {
    return x;
}
"""
    tokens = MetalLexer(
        source, preprocess=False, file_path="mlx-unary-ambiguous.metal"
    ).tokenize()

    with pytest.raises(MetalCallOperatorAssociationError) as exc_info:
        MetalParser(tokens, file_path="mlx-unary-ambiguous.metal").parse()

    error = exc_info.value
    assert error.reason == "multiple declarations match the definition contract"
    assert error.source_location["line"] == 6
    assert [location["line"] for location in error.declaration_locations] == [2, 3]
    assert error.candidates == (
        "operator()(float) const",
        "operator()(float) const",
    )


def test_skip_struct_builtin_conversion_operators_from_mlx_fp4_header():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Commit: b155224b9963cd9476363b464a559232a0868000
    # Path: mlx/backend/metal/kernels/fp4.h
    code = """
    struct fp4_e2m1 {
        operator float16_t() {
            half converted = as_type<half>(ushort((bits & 7) << 9));
            return bits & 8 ? -converted : converted;
        }

        operator float() {
            return static_cast<float>(this->operator float16_t());
        }

        operator bfloat16_t() {
            return static_cast<bfloat16_t>(this->operator float16_t());
        }

        uint8_t bits;
    };
    """
    ast = parse_ok(code)
    struct = ast.structs[0]

    assert struct.name == "fp4_e2m1"
    assert [(member.vtype, member.name) for member in struct.members] == [
        ("uint8_t", "bits")
    ]


def test_parse_unscoped_enum_with_underlying_type_from_metal_splatter():
    code = """
    enum BufferIndex: int32_t
    {
        BufferIndexMeshPositions = 0,
        BufferIndexMeshGenerics  = 1,
        BufferIndexUniforms      = 2,
    };
    """
    ast = parse_ok(code)
    enum = ast.enums[0]

    assert enum.name == "BufferIndex"
    assert enum.is_scoped is False
    assert enum.underlying_type == "int32_t"
    assert enum.members[1] == ("BufferIndexMeshGenerics", "1")


def test_parse_anonymous_enum_from_metal_base_effect():
    code = """
    enum {
        VertexAttributePosition,
        VertexAttributeNormal,
        VertexAttributeColor,
        VertexAttributeTexCoord0,
    };
    """
    ast = parse_ok(code)
    enum = ast.enums[0]

    assert enum.name is None
    assert enum.underlying_type is None
    assert enum.members[0] == ("VertexAttributePosition", None)
    assert enum.members[-1] == ("VertexAttributeTexCoord0", None)


def test_parse_typedef_enum_with_tag_and_alias_from_satin_constants():
    code = """
    typedef enum ComputeBufferIndex {
        ComputeBufferUniforms = 0,
        ComputeBufferCustom0 = 1,
        ComputeBufferCustom1 = 2
    } ComputeBufferIndex;

    void main() {
        ComputeBufferIndex index = ComputeBufferCustom0;
    }
    """
    ast = parse_ok(code)

    enum = ast.enums[0]
    assert enum.name == "ComputeBufferIndex"
    assert enum.typedef_tag == "ComputeBufferIndex"
    assert enum.members[0][0] == "ComputeBufferUniforms"


def test_parse_top_level_material_chunk_statement_from_satin():
    code = """
    const float2 baseColorTexcoord =
        (uniforms.baseColorTexcoordTransform * float3(in.texcoord, 1.0)).xy;
    pixel.material.baseColor = baseColorMap.sample(baseColorSampler, baseColorTexcoord).rgb;
    pixel.material.baseColor *= uniforms.baseColor.rgb;
    """

    ast = parse_ok(code)

    assert len(ast.global_variables) == 1
    assert ast.global_variables[0].left.name == "baseColorTexcoord"


def test_parse_stage_keyword_names_in_helpers_and_parameters():
    code = """
    float intersection(float a, float b) { return max(a, b); }

    float fresnel(float3 eyeVector, float3 worldNormal, float amount = 3.0) {
        return pow(1.0 + dot(eyeVector, worldNormal), amount);
    }

    typedef struct {
        float time;
    } SpriteUniforms;

    vertex float4 spriteVertex(constant SpriteUniforms &compute [[buffer(1)]])
    {
        return float4(compute.time);
    }
    """
    ast = parse_ok(code)

    assert ast.functions[0].name == "intersection"
    assert ast.functions[1].params[2].name == "amount"
    assert ast.functions[1].params[2].default_value == "3.0"
    assert ast.functions[2].params[0].name == "compute"


def test_parse_object_data_reference_parameter_from_satin_mesh_shader():
    code = """
    struct Payload {
        uint indices[3];
    };

    [[object, max_total_threads_per_threadgroup(1)]]
    void customObject(
        object_data Payload &payload [[payload]],
        mesh_grid_properties mgp
    ) {
        payload.indices[0] = 0;
    }
    """
    ast = parse_ok(code)
    param = ast.functions[0].params[0]

    assert param.vtype == "Payload&"
    assert param.name == "payload"
    assert param.qualifiers == ["object_data"]
    assert param.attributes[0].name == "payload"


def test_parse_top_level_texture_parameter_fragment_from_satin_pbr_chunk():
    code = """
    texture2d<float> baseColorMap [[texture(PBRTextureBaseColor)]],
        texture2d<float> subsurfaceMap [[texture(PBRTextureSubsurface)]],
        texturecube<float> reflectionMap [[texture(PBRTextureReflection)]],
    """
    ast = parse_ok(code)

    assert ast.global_variables == []


def test_parse_sizeof_and_cast():
    code = """
    void main() {
        int a = sizeof(int);
        int b = alignof(float4);
        float3 v = (float3)(1.0);
    }
    """
    parse_ok(code)


def test_parse_sizeof_dependent_typename_from_tinygrad_tile_copy():
    code = """
    template<typename ST>
    METAL_FUNC void load(threadgroup ST &dst) {
        constexpr const int elem_per_memcpy =
            sizeof(read_vector) / sizeof(typename ST::dtype);
        return;
    }
    """
    ast = parse_ok(code)

    sizeof_calls = [
        node
        for node in iter_ast_nodes(ast)
        if isinstance(node, FunctionCallNode) and node.name == "sizeof"
    ]
    assert any(call.args == ["ST::dtype"] for call in sizeof_calls)


def test_parse_pointer_member_access():
    code = """
    struct Uniforms {
        float4x4 mvp;
    };

    void main(constant Uniforms* uniforms) {
        float4 position = uniforms->mvp * float4(1.0);
    }
    """
    ast = parse_ok(code)
    member_accesses = [
        node for node in iter_ast_nodes(ast) if isinstance(node, MemberAccessNode)
    ]

    assert any(node.member == "mvp" and node.is_pointer for node in member_accesses)


def test_parse_stage_keyword_member_access_from_naga_ray_query_msl():
    # Reduced from:
    # Repo: https://github.com/gfx-rs/naga
    # Commit: d0f28c0b1a3c772e55e68db1c47eff5131cb6732
    # Path: tests/out/msl/ray-query.msl
    code = """
    struct Result {
        uint type;
        float distance;
    };
    struct RayQuery {
        Result intersection;
    };
    void main_() {
        RayQuery rq;
        uint kind = rq.intersection.type;
        float t = rq.intersection.distance;
    }
    """
    ast = parse_ok(code)
    member_accesses = [
        node for node in iter_ast_nodes(ast) if isinstance(node, MemberAccessNode)
    ]

    assert any(node.member == "intersection" for node in member_accesses)
    assert any(node.member == "type" for node in member_accesses)
    assert any(node.member == "distance" for node in member_accesses)


def test_parse_single_statement_if_with_discard_fragment():
    code = """
    fragment half4 fragment_main(float4 color) {
        if (color.a < 0.5)
            discard_fragment();

        return half4(color);
    }
    """
    ast = parse_ok(code)
    if_nodes = [node for node in iter_ast_nodes(ast) if isinstance(node, IfNode)]

    assert len(if_nodes) == 1
    assert isinstance(if_nodes[0].if_body[0], DiscardNode)


def test_parse_alignas_and_static_assert():
    code = """
    alignas(16) float4 alignedValue;
    static_assert(1 == 1, "ok");

    void main() {
        alignas(float4) int v = 0;
        static_assert(sizeof(int) == 4);
    }
    """
    parse_ok(code)


def test_parse_alignas_after_struct_keyword_from_apple_msl_spec():
    # Provenance: Apple Metal Shading Language Specification section 2.5
    # permits alignas on structure declarations.
    code = """
    struct alignas(16) UniformBlock {
        float4 color;
    };
    """
    ast = parse_ok(code)
    struct = ast.structs[0]

    assert struct.name == "UniformBlock"
    assert struct.alignas == ["16"]
    assert struct.members[0].name == "color"
    assert struct.members[0].vtype == "float4"


def test_parse_using_alias():
    code = """
    using Index = uint;
    void main() {
        Index i = 0;
    }
    """
    parse_ok(code)


def test_parse_metal_namespace_using_declarations_from_chromium_hdr_shader():
    # Reduced from:
    # Repo: https://chromium.googlesource.com/chromium/src
    # Commit: 137.0.7151.119
    # Path: components/metal_util/hdr_copier_layer.mm
    code = """
    #include <metal_stdlib>
    #include <simd/simd.h>
    using metal::float2;
    using metal::float3;
    using metal::float4;
    using metal::sampler;
    using metal::texture2d;
    using metal::abs;
    using metal::max;
    using metal::pow;
    using metal::sign;

    typedef struct {
        float4 clipSpacePosition [[position]];
        float2 texcoord;
    } RasterizerData;

    float ToLinearPQ(float v) {
        v = max(0.0f, v);
        float p = pow(v, 0.5f);
        return sign(p) * abs(p);
    }

    fragment float4 fragmentShader(RasterizerData in [[stage_in]],
                                   texture2d<float> plane0 [[texture(0)]]) {
        constexpr sampler s(metal::mag_filter::nearest,
                            metal::min_filter::nearest);
        float4 color = plane0.sample(s, in.texcoord);
        color.xyz = float3(ToLinearPQ(color.x));
        return color;
    }
    """
    ast = parse_ok(code)

    assert ast.structs[0].name == "RasterizerData"
    assert [func.name for func in ast.functions] == ["ToLinearPQ", "fragmentShader"]


def test_parse_gpuimage_typedef_struct_and_parenthesized_identifier_expression():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    typedef struct {
        float reductionFactor;
    } LuminanceUniform;

    fragment half4 luminanceRangeReduction(
        texture2d<half> inputTexture [[texture(0)]],
        constant LuminanceUniform& uniform [[buffer(1)]]
    ) {
        half4 color;
        half luminanceRatio;
        return half4(half3((color.rgb) + (luminanceRatio)), color.w);
    }
    """
    parse_ok(code)


def test_parse_stage_attributes_and_trailing_const_pointer_qualifiers():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct ShaderVertex {
        float4 position [[position]];
        float4 color;
    };

    [[vertex]] ShaderVertex main_vertex(
        device ShaderVertex const* const vertices [[buffer(0)]],
        uint vid [[vertex_id]]
    ) {
        return vertices[vid];
    }
    """
    ast = parse_ok(code)

    assert ast.functions[0].qualifier == "vertex"


def test_parse_keyword_named_sampler_and_buffer_from_apple_samples():
    code = """
    fragment half4 texturedQuadFragment(texture2d<half> texture [[texture(0)]],
                                        device float* values [[buffer(0)]],
                                        uint index) {
        constexpr sampler sampler(min_filter::linear, mag_filter::linear);
        device float* buffer = values;
        half4 color = texture.sample(sampler, float2(buffer[index]));
        return color;
    }
    """
    ast = parse_ok(code)
    body = ast.functions[0].body

    sampler_decl = body[0]
    buffer_decl = body[1]
    assert sampler_decl.left.name == "sampler"
    assert sampler_decl.left.vtype == "sampler"
    assert buffer_decl.left.name == "buffer"
    assert buffer_decl.left.vtype == "float*"


def test_parse_function_prototypes_macro_expanded_empty_statements_and_comma_update():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    half lum(half3 c);

    half lum(half3 c) {
        return dot(c, half3(0.3, 0.59, 0.11));
    }

    kernel void reduce(device uint* values [[buffer(0)]], uint histSize) {
        uint residual = 4;
        uint residualStep = 2;
        for (uint i = 0; i < histSize && residual > 0; i += residualStep, residual--) {
            ; ;
            values[i]++;
        }
    }
    """
    parse_ok(code)


def test_parse_multi_declarator_for_header_from_mlx_conv_loader():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Commit: 6ea7a00d05d548219864d10ff6c013b7544b13ea
    # Path: mlx/backend/metal/kernels/steel/conv/loaders/loader_general.h
    code = """
    void load_unsafe(short n_rows, short TROWS) {
        for (short i = 0, is = 0; i < n_rows; ++i, is += TROWS) {
            short row = is;
        }
    }
    """
    ast = parse_ok(code)
    loop = ast.functions[0].body[0]

    assert isinstance(loop, ForNode)
    assert isinstance(loop.init, list)
    assert len(loop.init) == 2
    assert loop.init[0].left.name == "i"
    assert loop.init[0].left.vtype == "short"
    assert loop.init[1].left.name == "is"
    assert isinstance(loop.update, BinaryOpNode)
    assert loop.update.op == ","


def test_parse_metalpetal_namespace_macro_qualifier_and_unsigned_int():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    namespace metalpetal {
        namespace yuv2rgbconvert {
            typedef struct {
                packed_float2 position;
            } Vertex;

            METAL_FUNC float helper(float value) {
                using namespace metalpetal::yuv2rgbconvert;
                return value;
            }

            vertex float4 colorConversionVertex(
                const device Vertex * vertices [[buffer(0)]],
                unsigned int vid [[vertex_id]]
            ) {
                return float4(float2(vertices[vid].position), helper(1.0), 1.0);
            }
        }
    }
    """
    parse_ok(code)


def test_parse_gnu_inline_unsigned_helpers_from_strelka_random_shader():
    # Reduced from:
    # Repo: https://github.com/arhix52/Strelka
    # Commit: 3eec7fa260e7d598911053f9e0f38054ce1c4f60
    # Path: src/render/metal/shaders/random.h
    code = """
    inline unsigned pcg_hash(unsigned seed) {
        unsigned state = seed * 747796405u + 2891336453u;
        return state;
    }

    template<unsigned int N>
    static __inline__ unsigned int tea(unsigned int val0, unsigned int val1) {
        unsigned int v0 = val0;
        return v0;
    }

    static __inline__ unsigned int lcg(thread unsigned int &prev) {
        prev = prev * 1664525u + 1013904223u;
        return prev & 0x00FFFFFF;
    }
    """
    ast = parse_ok(code)
    pcg_hash, tea, lcg = ast.functions

    assert pcg_hash.return_type == "uint"
    assert pcg_hash.name == "pcg_hash"
    assert pcg_hash.body[0].left.vtype == "uint"
    assert tea.return_type == "uint"
    assert tea.name == "tea"
    assert tea.template_parameters == [("value", "N")]
    assert tea.params[0].vtype == "uint"
    assert lcg.return_type == "uint"
    assert lcg.params[0].vtype == "uint&"
    assert lcg.params[0].qualifiers == ["thread"]


def test_parse_function_table_call_and_icb_methods():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Payload { float3 color; };

    visible_function_table<void(Payload&)> vft [[buffer(0)]];
    indirect_command_buffer icb [[buffer(1)]];

    kernel void main(device float* outData [[buffer(2)]]) {
        Payload p;
        vft[0](p);
        icb.reset();
        icb.draw_primitives(3, 1, 0, 0);
    }
    """
    parse_ok(code)


def test_parse_visible_function_table_using_signature_alias_from_apple_wwdc():
    # Apple WWDC20 "Get to know Metal function pointers" uses function-signature
    # aliases before passing them to visible_function_table<T>.
    # https://developer.apple.com/videos/play/wwdc2020/10013/
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Light { uint index; };
    struct Lighting { float3 color; };
    struct Material { uint index; };
    struct TriangleIntersectionData { float3 normal; };

    using LightingFunction = Lighting(Light, TriangleIntersectionData);
    using MaterialFunction = float3(Material, Lighting, TriangleIntersectionData);

    kernel void shade(
        visible_function_table<LightingFunction> lightingFunctions [[buffer(1)]],
        visible_function_table<MaterialFunction> materialFunctions [[buffer(2)]],
        device float3* output [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
        Light light;
        Material material;
        TriangleIntersectionData triangleIntersection;
        Lighting lighting = lightingFunctions[light.index](light, triangleIntersection);
        output[tid] = materialFunctions[material.index](
            material, lighting, triangleIntersection);
    }
    """
    ast = parse_ok(code)

    assert [alias.name for alias in ast.typedefs] == [
        "LightingFunction",
        "MaterialFunction",
    ]
    assert [alias.alias_type for alias in ast.typedefs] == ["Lighting", "float3"]
    assert all(getattr(alias, "is_function_type", False) for alias in ast.typedefs)
    params = ast.functions[0].params
    assert params[0].vtype == "visible_function_table<LightingFunction>"
    assert params[1].vtype == "visible_function_table<MaterialFunction>"


def test_parse_icb_extended_methods():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    indirect_command_buffer icb [[buffer(0)]];

    kernel void main() {
        icb.reset();
        icb.draw_primitives(3, 1, 0, 0);
        icb.draw_indexed_primitives(3, 1, 0, 0, 0);
        icb.draw_patches(3, 1, 0, 0);
        icb.compute_dispatch(uint3(1, 1, 1));
    }
    """
    parse_ok(code)


def test_parse_payload_and_hit_attributes():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct Payload { float3 color; };
    struct HitAttrib { float2 bary; };

    anyhit void any_hit(Payload& payload [[payload]],
                        HitAttrib attr [[hit_attribute]]) { }

    closesthit void closest_hit(Payload& payload [[payload]],
                                HitAttrib attr [[hit_attribute]]) { }
    """
    parse_ok(code)


def test_parse_mesh_object_io():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    struct ObjPayload { float4 data; };

    object void object_main(threadgroup ObjPayload* payload [[payload]],
                            uint3 gid [[threadgroup_position_in_grid]]) { }

    mesh void mesh_main(threadgroup ObjPayload* payload [[payload]],
                        uint3 tid [[thread_position_in_threadgroup]]) { }
    """
    parse_ok(code)


def test_parse_mesh_output_functions():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    mesh void mesh_main() {
        SetMeshOutputCounts(64, 32);
        SetVertex(0, float3(0.0));
        SetPrimitive(0, 0);
    }
    """
    parse_ok(code)


def test_parse_atomic_operations():
    code = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void main() {
        atomic_int counter;
        int old = atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
        old = atomic_exchange_explicit(counter, 0, memory_order_relaxed);
    }
    """
    parse_ok(code)


def test_parse_leading_decimal_float_literals_in_constexpr_arrays():
    code = """
    constexpr constant static float kvalues_mxfp4_f[4] = {0, .5f, 1.f, -.5f};
    """
    parse_ok(code)


def test_parse_cxx14_digit_separator_numeric_literals_from_msl_spec():
    # The Metal Shading Language Specification defines MSL as C++14 based.
    code = """
    kernel void main(device uint* out [[buffer(0)]]) {
        uint count = 1'024u;
        uint mask = 0xFF'00u;
        uint bits = 0b1010'0011u;
        float coeff = 1.602'176e-19f;
        out[0] = count + mask + bits + uint(coeff);
    }
    """
    ast = parse_ok(code)
    body = ast.functions[0].body

    assert body[0].right == "1'024u"
    assert body[1].right == "0xFF'00u"
    assert body[2].right == "0b1010'0011u"
    assert body[3].right == "1.602'176e-19f"


def test_parse_hex_float_literals_from_msl_cxx_base():
    # MSL is C++14 based, so hexadecimal floating literals use a p/P exponent.
    code = """
    kernel void main(device float* out [[buffer(0)]]) {
        float tiny = 0x1.0p-14f;
        half one = 0x1p+0h;
        float separated = 0x1'0.8p+2f;
        out[0] = tiny + float(one) + separated;
    }
    """
    ast = parse_ok(code)
    body = ast.functions[0].body

    assert body[0].right == "0x1.0p-14f"
    assert body[1].right == "0x1p+0h"
    assert body[2].right == "0x1'0.8p+2f"


def test_parse_bfloat_literal_suffixes_from_msl_spec():
    # Apple MSL Specification, section 2.2, documents 0.5bf and 0.5BF.
    code = """
    void main() {
        bfloat lo = 0.5bf;
        bfloat hi = 0.5BF;
    }
    """
    ast = parse_ok(code)
    body = ast.functions[0].body

    assert body[0].right == "0.5bf"
    assert body[1].right == "0.5BF"


def test_parse_as_type_template_call():
    code = """
    static inline float fp32_from_bits(uint32_t bits) {
        return as_type<float>(bits);
    }
    """
    ast = parse_ok(code)
    calls = [node for node in iter_ast_nodes(ast) if isinstance(node, FunctionCallNode)]

    assert any(node.name == "as_type<float>" for node in calls)


def test_parse_static_cast_from_apple_compute_sample():
    code = """
    kernel void process(uint2 gid [[thread_position_in_grid]]) {
        float2 p0 = static_cast<float2>(gid);
    }
    """
    ast = parse_ok(code)
    casts = [node for node in iter_ast_nodes(ast) if isinstance(node, CastNode)]

    assert len(casts) == 1
    assert casts[0].target_type == "float2"


def test_parse_template_dequantize_helper_from_llama_cpp():
    code = """
    template <typename type4x4>
    void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
        reg = (type4x4)(*src);
    }
    """
    ast = parse_ok(code)
    casts = [node for node in iter_ast_nodes(ast) if isinstance(node, CastNode)]

    assert [func.name for func in ast.functions] == ["dequantize_f32"]
    assert ast.functions[0].generics == ["type4x4"]
    assert ast.functions[0].params[0].vtype == "float4x4*"
    assert ast.functions[0].params[0].qualifiers == ["device", "const"]
    assert ast.functions[0].params[2].vtype == "type4x4&"
    assert ast.functions[0].params[2].qualifiers == ["thread"]
    assert any(node.target_type == "type4x4" for node in casts)


def test_parse_template_kernel_typename_without_space_from_llama_cpp():
    code = """
    template<typename T>
    kernel void kernel_memset(
            constant ggml_metal_kargs_memset & args,
            device T * dst,
            uint tpig [[thread_position_in_grid]]) {
        dst[tpig] = args.val;
    }
    """
    ast = parse_ok(code)

    assert [func.name for func in ast.functions] == ["kernel_memset"]
    assert ast.functions[0].qualifier == "kernel"
    assert ast.functions[0].generics == ["T"]
    assert ast.functions[0].params[1].vtype == "T*"
    assert ast.functions[0].params[1].qualifiers == ["device"]


def test_parse_explicit_template_specialization_from_blender_texture_read():
    code = """
    template<typename T> T convert_type(float type)
    {
        return T(type);
    }

    template<> uint convert_type<uint>(float val)
    {
        return uint(val * float(0xFFFFFFFFu));
    }
    """
    ast = parse_ok(code)

    assert [func.name for func in ast.functions] == [
        "convert_type",
        "convert_type<uint>",
    ]
    assert ast.functions[0].generics == ["T"]
    assert ast.functions[1].return_type == "uint"
    assert ast.functions[1].generics == []


def test_parse_nested_template_closers_from_spirv_cross_descriptor_array():
    # Reduced from:
    # Repo: https://github.com/KhronosGroup/SPIRV-Cross
    # Commit: 146679ff8255a6068518685599d7fb8761d1b570
    # Path: reference/shaders-msl/frag/runtime_array_as_argument_buffer.msl3.argument-tier-1.rich-descriptor.frag
    code = """
    void implicit_texture(thread uint& inputId,
                          const spvDescriptorArray<texture2d<float>> textures,
                          const spvDescriptorArray<sampler> smp) {
        return;
    }
    """
    ast = parse_ok(code)
    params = ast.functions[0].params

    assert params[0].vtype == "uint&"
    assert params[0].qualifiers == ["thread"]
    assert params[1].vtype == "spvDescriptorArray<texture2d<float>>"
    assert params[1].qualifiers == ["const"]
    assert params[2].vtype == "spvDescriptorArray<sampler>"
    assert params[2].qualifiers == ["const"]


def test_parse_nested_template_closers_in_struct_specialization_from_spirv_cross():
    # Reduced from:
    # Repo: https://github.com/KhronosGroup/SPIRV-Cross
    # Commit: 146679ff8255a6068518685599d7fb8761d1b570
    # Path: reference/shaders-msl/asm/comp/quantize.asm.comp
    code = """
    template<uint N>
    struct SpvHalfTypeSelector<vec<float, N>> {
        using H = vec<half, N>;
    };
    """
    ast = parse_ok(code)
    struct = ast.structs[0]

    assert struct.name == "SpvHalfTypeSelector<vec<float,N>>"
    assert struct.template_parameters == [("value", "N")]
    assert struct.members == []


def test_parse_template_struct_from_mlx_arg_reduce():
    code = """
    template <typename U>
    struct IndexValPair {
        uint32_t index;
        U val;
    };
    """
    ast = parse_ok(code)
    struct = ast.structs[0]

    assert struct.name == "IndexValPair"
    assert struct.generics == ["U"]
    assert struct.template_parameters == [("typename", "U")]
    assert [member.name for member in struct.members] == ["index", "val"]
    assert [member.vtype for member in struct.members] == ["uint32_t", "U"]


def test_parse_struct_constructor_with_initializer_list_and_trailing_semicolon_from_mlx():
    code = """
    struct complex64_t {
        float real;
        float imag;
        constexpr complex64_t(float real, float imag) : real(real), imag(imag) {};
    };
    """
    ast = parse_ok(code)
    struct = ast.structs[0]

    assert struct.name == "complex64_t"
    assert [member.name for member in struct.members] == ["real", "imag"]
    assert len(struct.constructors) == 1
    constructor = struct.constructors[0]
    assert constructor.owner_name == "complex64_t"
    assert [parameter.name for parameter in constructor.params] == ["real", "imag"]
    assert [initializer.target for initializer in constructor.initializers] == [
        "real",
        "imag",
    ]
    assert constructor.body == []
    assert constructor.source_location["line"] == 5


def test_parse_function_body_pragma_from_llama_cpp():
    code = """
    void quantize_q4_0(device const float* src, device block_q4_0& dst) {
        #pragma METAL fp math_mode(safe)
        float amax = 0.0f;
        dst.d = amax;
    }
    """
    ast = parse_ok(code)

    assert [func.name for func in ast.functions] == ["quantize_q4_0"]
    assert len(ast.functions[0].body) == 2


def test_parse_comma_assignment_statement_from_llama_cpp():
    code = """
    void dequantize(device const float* values) {
        float dl = 0.0f;
        float ml = 0.0f;
        dl = values[0], ml = values[1];
    }
    """
    ast = parse_ok(code)
    statement = ast.functions[0].body[2]

    assert isinstance(statement, BinaryOpNode)
    assert statement.op == ","
    assert isinstance(statement.left, AssignmentNode)
    assert isinstance(statement.right, AssignmentNode)
    assert statement.left.left.name == "dl"
    assert statement.right.left.name == "ml"


def test_parse_braced_uchar_vector_constructor_from_llama_cpp():
    code = """
    static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
        return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                     : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                              uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
    }
    """
    ast = parse_ok(code)
    braced_constructors = [
        node
        for node in iter_ast_nodes(ast)
        if isinstance(node, FunctionCallNode)
        and node.name == "uchar2"
        and getattr(node, "is_braced_constructor", False)
    ]
    scalar_constructors = [
        node
        for node in iter_ast_nodes(ast)
        if isinstance(node, VectorConstructorNode) and node.type_name == "uchar"
    ]

    assert len(braced_constructors) == 2
    assert len(scalar_constructors) == 4


def test_parse_direct_list_declaration_initializers_from_book_of_shaders_metal():
    code = """
    constant float3 colorA { 0.000f, 0.129f, 0.647f };
    constant float3 colorB { 0.980f, 0.275f, 0.090f };

    fragment float4 fragment_main() {
        float3 color { colorA.x, colorB.y, 1.0f };
        float2 st { 0.0f, 1.0f }, offset { 1.0f, 0.0f };
        return float4(color + float3(st.x, offset.y, 0.0f), 1.0f);
    }
    """
    ast = parse_ok(code)

    assert [node.left.name for node in ast.global_variables] == ["colorA", "colorB"]
    assert all(
        isinstance(node.right, InitializerListNode) for node in ast.global_variables
    )
    assert [len(node.right.elements) for node in ast.global_variables] == [3, 3]

    body = ast.functions[0].body
    assert isinstance(body[0].right, InitializerListNode)
    assert [node.left.name for node in body[1:3]] == ["st", "offset"]
    assert all(isinstance(node.right, InitializerListNode) for node in body[1:3])


def test_parse_standalone_scoped_block_from_llama_cpp():
    code = """
    void FC_unary_op(device const float* src0, device float* dst, uint i0) {
        {
            if (i0 >= 4) {
                return;
            }

            const float x = src0[i0];
            dst[i0] = x;
        }
    }
    """
    ast = parse_ok(code)
    block = ast.functions[0].body[0]

    assert isinstance(block, BlockNode)
    assert len(block.statements) == 3
    assert isinstance(block.statements[0], IfNode)
    assert isinstance(block.statements[2], AssignmentNode)


def test_parse_statement_expression_block_from_angle_generated_shader():
    # Reduced from:
    # Repo: https://android.googlesource.com/platform/external/angle
    # Commit: 282a5fb4ad
    # Path: src/libANGLE/renderer/metal/shaders/mtl_internal_shaders_autogen.metal
    code = """
    void outputPrimitive(bool use16,
                         bool use32,
                         device ushort* out16,
                         device uint* out32,
                         thread uint& onOutIndex,
                         uint tmpIndex) {
        ({
            if (use16) {
                out16[(onOutIndex)] = tmpIndex;
            }
            if (use32) {
                out32[(onOutIndex)] = tmpIndex;
            }
            onOutIndex++;
        });
    }
    """
    ast = parse_ok(code)
    block = ast.functions[0].body[0]

    assert isinstance(block, BlockNode)
    assert len(block.statements) == 2
    assert isinstance(block.statements[0], IfNode)
    assert len(block.statements[0].if_chain) == 2
    assert getattr(block.statements[1], "op", None) == "++"


def test_parse_decltype_template_typedef_and_explicit_instantiations_from_llama_cpp():
    code = """
    template [[host_name("kernel_unary_f32_f32")]]
    kernel void kernel_unary_impl(device const float* src, device float* dst) {
        dst[0] = erf_approx<float>(src[0]);
    }

    typedef decltype(kernel_unary_impl<float>) kernel_unary_t;
    template [[host_name("kernel_unary_f32_f32")]]
    kernel kernel_unary_t kernel_unary_impl<float>;
    """
    ast = parse_ok(code)

    assert ast.typedefs[0].name == "kernel_unary_t"
    assert ast.typedefs[0].alias_type == "decltype(kernel_unary_impl<float>)"
    assert len(ast.functions) == 1
    assert ast.functions[0].name == "kernel_unary_impl"


def test_parse_decltype_kernel_template_id_instantiation_from_mlx_jit_indexing():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Path: mlx/backend/metal/jit/indexing.h
    code = """
    template <typename T>
    [[kernel]] void slice_update_op_impl(device T* out [[buffer(0)]]) {
        out[0] = T(0);
    }

    [[kernel]] decltype(slice_update_op_impl<float>) slice_update_op_impl<float>;
    """
    ast = parse_ok(code)

    assert len(ast.functions) == 1
    assert ast.functions[0].name == "slice_update_op_impl"
    assert ast.global_variables == []


def test_parse_deleted_template_function_declaration_from_vllm_metal():
    # Reduced from:
    # Repo: https://github.com/vllm-project/vllm-metal
    # Path: vllm_metal/metal/kernels_v2/reshape_and_cache.metal
    code = """
    template <typename KV_T, typename CACHE_T>
    inline CACHE_T to_cache(KV_T v) = delete;

    template <>
    inline float to_cache<float, float>(float v) {
        return v;
    }

    kernel void real_kernel(device float* out [[buffer(0)]]) {
        out[0] = to_cache<float, float>(1.0f);
    }
    """
    ast = parse_ok(code)

    assert [func.name for func in ast.functions] == [
        "to_cache<float,float>",
        "real_kernel",
    ]


def test_parse_restrict_parameter_qualifier_from_vllm_metal():
    # Reduced from:
    # Repo: https://github.com/vllm-project/vllm-metal
    # Path: vllm_metal/metal/kernels_v2/reshape_and_cache.metal
    code = """
    kernel void reshape_and_cache(
        const device float *__restrict__ key [[buffer(0)]],
        device float* out [[buffer(1)]]
    ) {
        out[0] = key[0];
    }
    """
    ast = parse_ok(code)

    params = ast.functions[0].params
    assert params[0].name == "key"
    assert params[0].vtype == "float*"
    assert params[0].qualifiers == ["const", "device", "__restrict__"]
    assert params[1].name == "out"


def test_parse_materialx_out_parameter_qualifier_from_xcode_genmsl():
    # Reduced from Xcode's bundled MaterialX MSL library:
    # /Applications/Xcode.app/.../USDLib_FormatLoaderProxy_Xcode.framework/
    # Resources/libraries/stdlib/genmsl/mx_normalmap.metal
    code = """
    void mx_normalmap_vector2(vec3 value,
                              int map_space,
                              vec2 normal_scale,
                              out vec3 result) {
        result = value;
    }
    """
    ast = parse_ok(code)
    result_param = ast.functions[0].params[-1]

    assert result_param.vtype == "vec3"
    assert result_param.name == "result"
    assert result_param.qualifiers == ["out"]


def test_parse_pragma_and_type_trait_expression_from_llama_cpp():
    code = """
    void reduce(uint j, uint limit, device float* dst_row) {
        for (int i = 0; i < limit; i++) {
            _Pragma("clang loop unroll(full)")
            dst_row[i] = erf_approx<float>(dst_row[i]);
        }

        if (is_same<float4, T0>::value) {
            dst_row[0] = 0.0f;
        }
    }
    """
    ast = parse_ok(code)
    body = ast.functions[0].body

    assert isinstance(body[0], ForNode)
    assert len(body[0].body) == 1
    assert isinstance(body[1], IfNode)


def test_parse_empty_for_condition_before_type_trait_from_llama_cpp():
    code = """
    void flash_attn_ext(uint iwg, uint sgitg) {
        for (int ic0 = iwg * NSG + sgitg; ; ic0 += NWG * NSG) {
            int ic = ic0;
            if (ic0 >= D) {
                break;
            }
        }

        if (is_same<float4, T0>::value) {
            return;
        }
    }
    """
    ast = parse_ok(code)
    loop = ast.functions[0].body[0]

    assert isinstance(loop, ForNode)
    assert loop.condition is None
    assert isinstance(ast.functions[0].body[1], IfNode)


def test_parse_qualified_casts_and_range_designator_from_llama_cpp():
    code = """
    void load_block(device const void* src0, uint offset0, short r1ptg) {
        device const block_q1_0* block =
            (device const block_q1_0*)((device char*)src0 + offset0);
        float sumf[8] = {[0 ... r1ptg - 1] = 0.0f};
    }
    """
    ast = parse_ok(code)
    cast_assignment = ast.functions[0].body[0]
    range_assignment = ast.functions[0].body[1]
    initializer = range_assignment.right
    designator = initializer.elements[0]

    assert isinstance(cast_assignment.right, CastNode)
    assert cast_assignment.right.target_type == "block_q1_0*"
    assert cast_assignment.right.qualifiers == ["device", "const"]
    assert designator.designators[0][0] == "range"
    assert designator.designators[0][1] == "0"
    assert designator.designators[0][2] == "r1ptg-1"


def test_parse_function_pointer_typedef_from_llama_cpp():
    code = """
    typedef void (im2col_t)(constant ggml_metal_kargs_im2col& args,
                            device const float* x,
                            device char* dst,
                            uint3 tgpig [[threadgroup_position_in_grid]]);
    """
    ast = parse_ok(code)

    alias = ast.typedefs[0]
    assert isinstance(alias, CallableTypeAliasNode)
    assert alias.name == "im2col_t"
    assert alias.alias_type == "void"
    assert alias.indirection == ""
    assert [parameter.vtype for parameter in alias.parameters] == [
        "ggml_metal_kargs_im2col&",
        "float*",
        "char*",
        "uint3",
    ]
    assert [parameter.qualifiers for parameter in alias.parameters] == [
        ["constant"],
        ["device", "const"],
        ["device"],
        [],
    ]


def test_parse_mlx_callable_pointer_typedef_preserves_signature_contract():
    code = """
    typedef void (*RadixFunc)(thread float2*, thread float2*);
    using AlternateRadixFunc = void (*)(thread float2* lhs,
                                        thread float2* rhs);
    """
    ast = parse_ok(code)

    pointer_typedef, using_alias = ast.typedefs
    assert isinstance(pointer_typedef, CallableTypeAliasNode)
    assert pointer_typedef.name == "RadixFunc"
    assert pointer_typedef.return_type == "void"
    assert pointer_typedef.indirection == "*"
    assert pointer_typedef.is_function_pointer is True
    assert [parameter.vtype for parameter in pointer_typedef.parameters] == [
        "float2*",
        "float2*",
    ]
    assert [parameter.qualifiers for parameter in pointer_typedef.parameters] == [
        ["thread"],
        ["thread"],
    ]
    assert pointer_typedef.source_location["line"] == 2

    assert isinstance(using_alias, CallableTypeAliasNode)
    assert using_alias.name == "AlternateRadixFunc"
    assert using_alias.indirection == "*"
    assert [parameter.name for parameter in using_alias.parameters] == ["lhs", "rhs"]


def test_parse_enum_return_prototype_from_llama_cpp_context_header():
    code = """
    typedef struct ggml_metal * ggml_metal_t;

    enum ggml_status ggml_metal_graph_compute(
        ggml_metal_t ctx,
        struct ggml_cgraph * gf);
    """
    ast = parse_ok(code)

    assert ast.functions == []
    assert ast.structs == []
    assert ast.enums == []


def test_parse_struct_forward_declaration_from_mlx_complex_header():
    code = """
    struct complex64_t;

    void use_complex(thread complex64_t& value) {
        return;
    }
    """
    ast = parse_ok(code)

    assert ast.structs == []
    assert ast.functions[0].params[0].vtype == "complex64_t&"


def test_parse_template_struct_base_clause_from_mlx_type_traits():
    code = """
    namespace metal {
    template <typename T>
    struct is_empty : metal::bool_constant<__is_empty(T)> {};

    template <typename... Ts>
    struct make_void {
      typedef void type;
    };

    template <typename Head, typename... Tail>
    using selected_t = typename make_void<Head, Tail...>::type;

    template <class T>
    struct is_static : metal::bool_constant<is_empty<remove_cv_t<T>>::value> {};

    void consume(selected_t<int, float> value) {}
    }
    """
    ast = parse_ok(code)

    assert [struct.name for struct in ast.structs] == [
        "is_empty",
        "make_void",
        "is_static",
    ]
    assert ast.structs[0].base_types == ["metal::bool_constant<__is_empty(T)>"]
    assert ast.structs[1].members == []
    assert [
        (alias.name, alias.alias_type) for alias in ast.structs[1].type_aliases
    ] == [("type", "void")]
    assert ast.structs[2].base_types == [
        "metal::bool_constant<is_empty<remove_cv_t<T>>::value>"
    ]
    alias = ast.typedefs[0]
    assert alias.name == "selected_t"
    assert alias.qualified_name == "metal::selected_t"
    assert alias.alias_type == "make_void<Head,Tail...>::type"
    assert alias.template_parameters == [
        ("typename", "Head"),
        ("typename...", "Tail"),
    ]
    assert alias.is_template_alias is True
    assert ast.functions[0].namespace == "metal"


def test_parse_variadic_function_parameter_pack_from_mlx_integral_constant():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Commit: 6ea7a00d05d548219864d10ff6c013b7544b13ea
    # Path: mlx/backend/metal/kernels/steel/utils/integral_constant.h
    code = """
    template <typename T, typename... Us>
    METAL_FUNC constexpr auto sum(T x, Us... us) {
        return x + sum(us...);
    }
    """
    ast = parse_ok(code)
    function = ast.functions[0]
    returned = function.body[0].value
    pack_call = returned.right

    assert function.template_parameters == [("typename", "T"), ("typename...", "Us")]
    assert [(param.vtype, param.name) for param in function.params] == [
        ("T", "x"),
        ("Us...", "us"),
    ]
    assert isinstance(pack_call, FunctionCallNode)
    assert isinstance(pack_call.args[0], UnaryOpNode)
    assert pack_call.args[0].op == "post..."
    assert pack_call.args[0].operand.name == "us"


def test_parse_defaulted_value_template_parameters_and_source_location():
    code = """
    template <bool UseAlternate = false, int Width = 2 + 1>
    float select_value(float primary, float alternate) {
        return UseAlternate ? alternate : primary + Width;
    }
    """
    function = parse_ok(code).functions[0]

    assert function.template_parameters == [
        ("value", "UseAlternate"),
        ("value", "Width"),
    ]
    assert function.template_parameter_defaults == {
        "UseAlternate": "false",
        "Width": "2+1",
    }
    assert function.template_source_location == function.source_location
    assert function.source_location["line"] == 2
    assert function.source_location["end_line"] == 5
    assert function.source_location["length"] > 0


def test_parse_template_declaration_preserves_existing_source_location(monkeypatch):
    existing_location = {
        "file": "existing.metal",
        "line": 40,
        "column": 7,
        "end_line": 42,
        "end_column": 2,
    }
    original_parse_function = MetalParser.parse_function

    def parse_function_with_existing_location(parser):
        function = original_parse_function(parser)
        function.source_location = existing_location
        return function

    monkeypatch.setattr(
        MetalParser, "parse_function", parse_function_with_existing_location
    )

    source = """template <bool Enabled = false>
bool selected() { return Enabled; }
"""
    function = parse_ok(source).functions[0]

    assert function.source_location is existing_location
    assert function.template_source_location["line"] == 1
    assert function.template_source_location["column"] == 1
    assert function.template_source_location["end_line"] == 2


def test_parse_dependent_value_template_default_keeps_declared_parameter_name():
    code = """
    template <typename T, int Count = T::extent>
    int count_value() {
        return Count;
    }
    """
    function = parse_ok(code).functions[0]

    assert function.template_parameters == [("typename", "T"), ("value", "Count")]
    assert function.template_parameter_defaults == {"Count": "T::extent"}


def test_parse_dependent_enable_if_return_type_from_tinygrad_metal():
    # Reduced from:
    # Repo: https://github.com/tinygrad/tinygrad
    # Commit: 12addee14f1d728793648ceca307a5fde2b24cea
    # Path: extra/thunder/metal/include/ops/group/memory/tile/shared_to_register.metal
    code = """
    template<typename RT, typename ST>
    METAL_FUNC static typename metal::enable_if<
        ducks::is_row_register_tile<RT>() && ducks::is_shared_tile<ST>(),
        void>::type
    load(thread RT &dst, threadgroup const ST &src, const int threadIdx) {
        return;
    }
    """
    ast = parse_ok(code)
    function = ast.functions[0]

    assert function.name == "load"
    assert function.template_parameters == [("typename", "RT"), ("typename", "ST")]
    assert function.return_type == (
        "metal::enable_if<"
        "ducks::is_row_register_tile<RT>()&&ducks::is_shared_tile<ST>(),void"
        ">::type"
    )
    assert [
        (param.vtype, param.name, param.qualifiers) for param in function.params
    ] == [
        ("RT&", "dst", ["thread"]),
        ("ST&", "src", ["threadgroup", "const"]),
        ("int", "threadIdx", ["const"]),
    ]


def test_parse_multiline_macro_invocation_from_mlx_bf16_math_header():
    code = """
    #define METAL_FUNC inline
    #define instantiate_metal_math_funcs(itype, otype, ctype, mfast) \\
      METAL_FUNC otype abs(itype x) { \\
        return static_cast<otype>(__metal_fabs(static_cast<ctype>(x), mfast)); \\
      }

    namespace metal {
    instantiate_metal_math_funcs(
        bfloat16_t,
        bfloat16_t,
        float,
        __METAL_MAYBE_FAST_MATH__);
    }

    kernel void real_kernel(device float* out [[buffer(0)]]) {
        out[0] = 1.0f;
    }
    """
    ast = parse_ok(code)

    assert [func.name for func in ast.functions] == ["abs", "real_kernel"]
    overload = ast.functions[0]
    assert overload.namespace == "metal"
    assert overload.return_type == "bfloat16_t"
    assert [(param.vtype, param.name) for param in overload.params] == [
        ("bfloat16_t", "x")
    ]


def test_parse_metal_mesh_scoped_type_from_public_samples():
    code = """
    using TriangleMeshType = metal::mesh<VertexOut, PrimOut, 4, 2, topology::line>;

    kernel void real_kernel(device float* out [[buffer(0)]]) {
        out[0] = 1.0f;
    }
    """
    ast = parse_ok(code)

    assert [func.name for func in ast.functions] == ["real_kernel"]


def test_parse_char_literal_initializer_from_public_metal_samples():
    code = """
    constant char PATTERNS[][2] = {
        { 'M', 'S' },
        { 'S', 'M' }
    };

    kernel void real_kernel(device float* out [[buffer(0)]]) {
        out[0] = 1.0f;
    }
    """
    ast = parse_ok(code)

    assert ast.global_variables[0].left.name == "PATTERNS"
    assert ast.global_variables[0].left.array_sizes == [None, "2"]
    assert [func.name for func in ast.functions] == ["real_kernel"]


def test_parse_member_template_disambiguator_from_mlx_arg_reduce():
    code = """
    void reduce(thread Reducer& op) {
        int best = 0;
        int vals[2] = {0, 1};
        best = op.template reduce_many<N_READS>(best, vals, 0);
    }
    """
    ast = parse_ok(code)

    calls = [node for node in iter_ast_nodes(ast) if isinstance(node, MethodCallNode)]
    assert calls[0].method == "reduce_many<N_READS>"


def test_parse_standalone_for_loop_macro_prefix_from_mlx_conv():
    code = """
    void load_tile(device float* out) {
        MLX_MTL_PRAGMA_UNROLL
        for (int cc = 0; cc < 4; ++cc) {
            out[cc] = float(cc);
        }
    }
    """
    ast = parse_ok(code)

    loops = [node for node in iter_ast_nodes(ast) if isinstance(node, ForNode)]
    assert len(loops) == 1


def test_parse_template_qualified_static_member_declaration_from_mlx_conv():
    code = """
    constant constexpr const float WinogradTransforms<6, 3, 8>::wt_transform[8][8];
    """
    ast = parse_ok(code)

    variable = ast.global_variables[0]
    assert variable.name == "WinogradTransforms<6,3,8>::wt_transform"
    assert len(variable.array_sizes) == 2


def test_parse_scoped_variable_template_expression_from_mlx_gemv_masked():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Commit: b155224b9963cd9476363b464a559232a0868000
    # Path: mlx/backend/metal/kernels/gemv_masked.h
    code = """
    using namespace metal;

    template <typename out_mask_t, typename op_mask_t>
    struct GEMVKernel {
      static constant constexpr const bool has_operand_mask =
          !metal::is_same_v<op_mask_t, nomask_t>;
      static constant constexpr const bool has_mul_operand_mask =
          has_operand_mask && !metal::is_same_v<op_mask_t, bool>;
    };
    """
    ast = parse_ok(code)

    members = ast.structs[0].members
    first_value = members[0].default_value
    assert isinstance(first_value, UnaryOpNode)
    assert first_value.op == "!"
    assert first_value.operand.name == "metal::is_same_v<op_mask_t,nomask_t>"

    second_value = members[1].default_value
    assert isinstance(second_value, BinaryOpNode)
    assert second_value.op == "&&"
    assert second_value.right.operand.name == "metal::is_same_v<op_mask_t,bool>"


def test_parse_parenthesized_scoped_constant_expressions_from_mlx_steel():
    code = """
    struct Base {
      static constant constexpr const int kFragRows = 8;
    };

    struct Tile {
      static constant constexpr const int direct = (Base::kFragRows);
      static constant constexpr const int product = 4 * (Base::kFragRows);
      static constant constexpr const int sum = (Base::kFragRows) + 1;
      static constant constexpr const int difference = (Base::kFragRows) - 1;
      static constant constexpr const int scaled = (Base::kFragRows) * 2;
      static constant constexpr const int grouped_arithmetic =
          (Base::kFragRows + 1);
      static constant constexpr const int nested =
          ((outer::inner::Limits::value));
      static constant constexpr const int negated = -(Base::kFragRows);
      static constant constexpr const int selected =
          true ? (Base::kFragRows) : 0;
    };

    void consume(int value) {}

    void use_constant() {
      consume((Base::kFragRows));
    }
    """
    ast = parse_ok(code)

    members = {member.name: member.default_value for member in ast.structs[1].members}
    direct = members["direct"]
    assert direct.name == "Base::kFragRows"
    assert direct.source_location["file"] is None
    assert direct.group_source_location["offset"] < direct.source_location["offset"]
    assert direct.group_source_location["end_offset"] > (
        direct.source_location["end_offset"]
    )
    assert members["product"].right.name == "Base::kFragRows"
    assert members["sum"].left.name == "Base::kFragRows"
    assert members["difference"].left.name == "Base::kFragRows"
    assert members["scaled"].left.name == "Base::kFragRows"
    grouped_arithmetic = members["grouped_arithmetic"]
    assert grouped_arithmetic.left.name == "Base::kFragRows"
    assert grouped_arithmetic.group_source_location["offset"] < (
        grouped_arithmetic.left.source_location["offset"]
    )
    nested = members["nested"]
    assert nested.name == "outer::inner::Limits::value"
    assert len(nested.group_source_locations) == 2
    inner_group, outer_group = nested.group_source_locations
    assert outer_group["offset"] < inner_group["offset"]
    assert outer_group["end_offset"] > inner_group["end_offset"]
    assert members["negated"].operand.name == "Base::kFragRows"
    assert members["selected"].true_expr.name == "Base::kFragRows"

    call = ast.functions[1].body[0]
    assert isinstance(call, FunctionCallNode)
    assert call.args[0].name == "Base::kFragRows"


def test_parenthesized_scoped_values_remain_distinct_from_casts_and_constructors():
    code = """
    namespace outer {
    struct Payload {
      int value;
    };
    }

    void convert(outer::Payload input) {
      outer::Payload qualified = (outer::Payload)input;
      outer::Payload unary_qualified = (outer::Payload)-input;
      int scalar = (int)-1.5;
      int constructed = int(2.5);
    }

    template <typename T>
    T negate(T input) {
      return (T)-input;
    }
    """
    ast = parse_ok(code)

    body = ast.functions[0].body
    assert isinstance(body[0].right, CastNode)
    assert body[0].right.target_type == "outer::Payload"
    assert isinstance(body[1].right, CastNode)
    assert body[1].right.target_type == "outer::Payload"
    assert isinstance(body[1].right.expression, UnaryOpNode)
    assert isinstance(body[2].right, CastNode)
    assert body[2].right.target_type == "int"
    assert isinstance(body[2].right.expression, UnaryOpNode)
    assert isinstance(body[3].right, VectorConstructorNode)
    assert body[3].right.vector_type == "int"

    template_casts = [
        node for node in iter_ast_nodes(ast.functions[1]) if isinstance(node, CastNode)
    ]
    assert len(template_casts) == 1
    assert template_casts[0].target_type == "T"
    assert isinstance(template_casts[0].expression, UnaryOpNode)


def test_parse_typename_qualified_threadgroup_type_from_mlx_gemv():
    code = """
    void load_tile() {
        threadgroup typename gemv_kernel::acc_type tgp_memory[4];
    }
    """
    ast = parse_ok(code)

    variable = ast.functions[0].body[0]
    assert variable.vtype == "gemv_kernel::acc_type"
    assert variable.name == "tgp_memory"
    assert variable.qualifiers == ["threadgroup"]


def test_parse_multi_declarator_struct_members_from_cxx_msl_headers():
    code = """
    struct KernelParams {
        float scale = 1.0, bias = 0.0;
        uint2 extent, stride;
        device float* values, fallback;
    };
    """
    ast = parse_ok(code)
    members = ast.structs[0].members

    assert [(member.vtype, member.name) for member in members] == [
        ("float", "scale"),
        ("float", "bias"),
        ("uint2", "extent"),
        ("uint2", "stride"),
        ("float*", "values"),
        ("float", "fallback"),
    ]
    assert members[0].default_value == "1.0"
    assert members[1].default_value == "0.0"
    assert members[4].qualifiers == ["device"]
    assert members[5].qualifiers == ["device"]


def test_parse_union_declaration_from_mlx_random():
    code = """
    union rbits {
        uint2 val;
        uchar4 bytes[2];
    };

    rbits make_bits() {
        rbits v;
        for (auto r : rotations[0]) {
            v.val.x += r;
        }
        return v;
    }
    """
    ast = parse_ok(code)

    union = ast.structs[0]
    assert union.name == "rbits"
    assert getattr(union, "aggregate_kind", None) == "union"
    assert [(member.vtype, member.name) for member in union.members] == [
        ("uint2", "val"),
        ("uchar4", "bytes"),
    ]
    assert union.members[1].array_sizes == ["2"]
    assert ast.functions[0].return_type == "rbits"
    assert ast.functions[0].body[0].vtype == "rbits"
    loop = ast.functions[0].body[1]
    assert isinstance(loop, RangeForNode)
    assert loop.vtype == "auto"
    assert loop.name == "r"


def test_parse_using_union_alias_from_mlx_cexpf_header():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Commit: b155224b9963cd9476363b464a559232a0868000
    # Path: mlx/backend/metal/kernels/cexpf.h
    code = """
    using ieee_float_shape_type = union {
      float value;
      uint32_t word;
    };

    inline void get_float_word(thread uint32_t& i, float d) {
      ieee_float_shape_type gf_u;
      gf_u.value = (d);
      (i) = gf_u.word;
    }
    """
    ast = parse_ok(code)

    union = ast.structs[0]
    assert union.name == "ieee_float_shape_type"
    assert getattr(union, "aggregate_kind", None) == "union"
    assert getattr(union, "using_alias", False) is True
    assert [(member.vtype, member.name) for member in union.members] == [
        ("float", "value"),
        ("uint32_t", "word"),
    ]
    assert ast.functions[0].body[0].vtype == "ieee_float_shape_type"


def test_parse_if_constexpr_from_mlx_fp_quantized():
    code = """
    template <typename T, int group_size>
    static inline T dequantize_scale(uint8_t s) {
        if constexpr (group_size == 16) {
            return T(*(thread fp8_e4m3*)(&s));
        } else if constexpr (group_size == 32) {
            return T(*(thread fp8_e8m0*)(&s));
        } else {
            return T(s);
        }
    }
    """
    ast = parse_ok(code)
    if_node = ast.functions[0].body[0]

    assert isinstance(if_node, IfNode)
    assert isinstance(if_node.condition, BinaryOpNode)
    assert len(if_node.else_if_chain) == 1
    assert if_node.else_body


def test_parse_lambda_argument_from_mlx_fp_quantized_nax():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Commit: b155224b9963cd9476363b464a559232a0868000
    # Path: mlx/backend/metal/kernels/fp_quantized_nax.h
    code = """
    void run(bool is_unaligned_sm) {
      dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
        if constexpr (kAlignedM.value) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      });
    }
    """
    ast = parse_ok(code)

    call = ast.functions[0].body[0]
    assert isinstance(call, FunctionCallNode)
    assert call.name == "dispatch_bool"
    lambda_arg = call.args[1]
    assert isinstance(lambda_arg, LambdaNode)
    assert lambda_arg.capture == "&"
    assert lambda_arg.params[0].vtype == "auto"
    assert lambda_arg.params[0].name == "kAlignedM"
    assert lambda_arg.source_location["line"] == 3
    assert lambda_arg.source_location["end_line"] == 7
    assert lambda_arg.source_location["length"] > 0

    barrier_calls = [
        node
        for node in iter_ast_nodes(lambda_arg)
        if isinstance(node, FunctionCallNode) and node.name == "threadgroup_barrier"
    ]
    assert barrier_calls


def test_parse_template_id_value_expression_with_member_args_from_mlx_gemm_gather_nax():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Commit: 8f0e8b14e0fc028df8618684583af9bef44647b8
    # Path: mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_gather_nax.h
    code = """
    void run(bool is_unaligned_sm) {
      dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
        auto do_gemm = gemm_loop<
            T,
            SM,
            kAlignedM.value,
            AccumType>;
        if constexpr (kAlignedM.value) {
          do_gemm();
        }
      });
    }
    """
    ast = parse_ok(code)
    lambda_arg = ast.functions[0].body[0].args[1]
    assignment = lambda_arg.body[0]
    if_node = lambda_arg.body[1]

    assert isinstance(assignment, AssignmentNode)
    assert assignment.left.name == "do_gemm"
    assert assignment.right.name == "gemm_loop<T,SM,kAlignedM.value,AccumType>"
    assert isinstance(if_node, IfNode)
    assert if_node.condition.member == "value"


def test_parse_raw_string_view_shader_template_from_mlx_jit_indexing_header():
    # Reduced from:
    # Repo: https://github.com/ml-explore/mlx
    # Commit: 4367c73b60541ddd5a266ce4644fd93d20223b6e
    # Path: mlx/backend/metal/jit/indexing.h
    code = """
    constexpr std::string_view masked_assign_kernel = R"(
    template [[host_name("{0}")]] [[kernel]]
    decltype(masked_assign_impl<{1}, {2}>) masked_assign_impl<{1}, {2}>;
    )";
    """
    ast = parse_ok(code)
    assignment = ast.global_variables[0]

    assert isinstance(assignment, AssignmentNode)
    assert assignment.left.vtype == "std::string_view"
    assert assignment.left.name == "masked_assign_kernel"
    assert str(assignment.right).startswith('R"(')


def test_parse_preprocessor_define():
    code = """
    #define FOO 1
    void main() {
        int x = FOO;
    }
    """
    parse_ok(code)


@pytest.mark.parametrize(
    "code",
    [
        "struct S { float x };",  # Missing semicolon in member declaration
        "struct S { float x;",  # Unclosed struct
        "float4 pos [[position];",  # Unterminated attribute brackets
        "vertex float4 main(float4 v [[stage_in]] { return v; }",  # Missing ')'
        "void main() { case 0: break; }",  # Case outside switch
        "void main() { switch (1) { case 0 return; } }",  # Missing ':'
    ],
)
def test_parse_invalid_syntax_cases(code):
    parse_fails(code)


if __name__ == "__main__":
    pytest.main()
