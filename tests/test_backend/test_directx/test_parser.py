import math
import textwrap

import pytest

from crosstl.backend.common_ast import (
    AssignmentNode,
    CastNode,
    FunctionCallNode,
    FunctionNode,
    InitializerListNode,
    MemberAccessNode,
    SwitchNode,
    TextureSampleNode,
    VectorConstructorNode,
)
from crosstl.backend.DirectX.DirectxAst import (
    ForNode,
    IfNode,
    PragmaNode,
    StructNode,
    VariableNode,
    WhileNode,
)
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.backend.DirectX.preprocessor import HLSLPreprocessor

VERTEX_PIXEL_HLSL = textwrap.dedent("""
    cbuffer CameraBuffer : register(b0) {
        float4x4 viewProj;
        float3 eyePos;
        float padding;
    };

    Texture2D tex0 : register(t0);
    SamplerState samp0 : register(s0);

    struct VSInput {
        float3 position : POSITION;
        float3 normal : NORMAL;
        float2 uv : TEXCOORD0;
    };

    struct VSOutput {
        float4 position : SV_Position;
        float3 normal : TEXCOORD1;
        float2 uv : TEXCOORD0;
    };

    VSOutput VSMain(VSInput input) {
        VSOutput output;
        output.position = mul(viewProj, float4(input.position, 1.0));
        output.normal = input.normal;
        output.uv = input.uv;
        return output;
    }

    struct PSInput {
        float4 position : SV_Position;
        float3 normal : TEXCOORD1;
        float2 uv : TEXCOORD0;
    };

    struct PSOutput {
        float4 color : SV_Target0;
    };

    float4 Lighting(float3 n, float2 uv) {
        float3 lightDir = normalize(float3(0.0, 1.0, 0.0));
        float ndotl = max(dot(n, lightDir), 0.0);
        float4 texColor = tex0.Sample(samp0, uv);
        return texColor * ndotl;
    }

    PSOutput PSMain(PSInput input) {
        PSOutput output;
        output.color = Lighting(input.normal, input.uv);
        return output;
    }
    """).strip()

CONTROL_FLOW_HLSL = textwrap.dedent("""
    int ControlFlow(int a, int b) {
        int sum = 0;
        for (int i = 0; i < 4; ++i) {
            if (i % 2 == 0) {
                continue;
            } else if (i == 3) {
                break;
            }
            sum += i;
        }

        int j = 0;
        while (j < 3) {
            sum += j;
            j++;
        }

        int k = 0;
        do {
            sum += k;
            k++;
        } while (k < 2);

        switch (a) {
            case 0:
                sum += 1;
                break;
            case 1:
                sum += 2;
                // fallthrough
            default:
                sum += 3;
                break;
        }

        int ternaryVal = (a > b) ? a : b;
        return sum + ternaryVal;
    }
    """).strip()

ARRAYS_HLSL = textwrap.dedent("""
    float4 colors[4];
    float3 grid[2][3];

    float sumArray(float4 values[4]) {
        float total = 0.0;
        for (int i = 0; i < 4; i++) {
            total += values[i].x;
        }
        return total;
    }

    float useGrid() {
        return grid[1][2].x;
    }
    """).strip()

RESOURCES_HLSL = textwrap.dedent("""
    Texture2D<float4> tex1 : register(t1);
    SamplerComparisonState sampComp : register(s1);
    RWTexture2D<float4> outputTex : register(u0);
    StructuredBuffer<float4> data : register(t2);
    RWStructuredBuffer<float4> outData : register(u1);

    float4 SampleTex(float2 uv) : SV_Target0 {
        float4 value = tex1.SampleCmpLevelZero(sampComp, uv, 0.5);
        outData[0] = value;
        return value;
    }
    """).strip()

OVERLOADS_HLSL = textwrap.dedent("""
    float4 Blend(float4 a, float4 b) {
        return a + b;
    }

    float4 Blend(float4 a, float4 b, float t) {
        return lerp(a, b, t);
    }

    float4 UseBlend(float4 a, float4 b) {
        return Blend(a, b, 0.5);
    }
    """).strip()

COMPUTE_HLSL = textwrap.dedent("""
    RWTexture2D<float4> outputTex : register(u0);

    [numthreads(8, 8, 1)]
    void CSMain(uint3 dtid : SV_DispatchThreadID) {
        outputTex[dtid.xy] = float4(1.0, 0.0, 0.0, 1.0);
    }
    """).strip()

PREPROCESSOR_HLSL = textwrap.dedent("""
    #define USE_LIGHTING 1
    #if USE_LIGHTING
    float3 Lighting(float3 n) { return n; }
    #endif

    float4 main() : SV_Target0 {
        return float4(1.0, 1.0, 1.0, 1.0);
    }
    """).strip()


def tokenize_code(code: str):
    lexer = HLSLLexer(code)
    return lexer.tokenize()


def parse_code(code: str):
    tokens = tokenize_code(code)
    parser = HLSLParser(tokens)
    return parser.parse()


def assert_parses(code: str):
    try:
        parse_code(code)
    except SyntaxError as exc:
        pytest.fail(f"Expected code to parse, but got SyntaxError: {exc}")


def assert_parse_error(code: str):
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


def test_parse_vertex_pixel_shader():
    assert_parses(VERTEX_PIXEL_HLSL)


def test_parse_main_struct_inout_parameters_infer_stage_qualifiers():
    vertex_ast = parse_code(textwrap.dedent("""
            struct VSInput {
                float3 position : ATTRIB0;
                float4 color : COLOR0;
            };

            struct VSOutput {
                float4 position : SV_POSITION;
                float4 color : COLOR0;
            };

            void main(in VSInput input, out VSOutput output) {
                output.position = float4(input.position, 1.0);
                output.color = input.color;
            }
            """))

    fragment_ast = parse_code(textwrap.dedent("""
            struct PSInput {
                float4 position : SV_POSITION;
                float4 color : COLOR0;
            };

            struct PSOutput {
                float4 color : SV_TARGET;
            };

            void main(in PSInput input, out PSOutput output) {
                output.color = input.color;
            }
            """))

    vertex_main = vertex_ast.functions[0]
    fragment_main = fragment_ast.functions[0]

    assert vertex_main.qualifier == "vertex"
    assert vertex_main.qualifiers == ["vertex"]
    assert fragment_main.qualifier == "fragment"
    assert fragment_main.qualifiers == ["fragment"]


def test_parse_brace_initializer_declarations():
    ast = parse_code(textwrap.dedent("""
            struct MyPayload {
                int val;
            };

            struct RayDesc {
                float3 Origin;
                float TMin;
                float3 Direction;
                float TMax;
            };

            void RayGen() {
                float3 origin = float3(0.0, 0.0, 0.0);
                float3 rayDir = float3(0.0, 0.0, 1.0);
                RayDesc myRay = { origin, 0.0f, rayDir, 10000.0f };
                MyPayload payload = { 0 };
            }
            """))

    raygen = next(function for function in ast.functions if function.name == "RayGen")
    my_ray = next(stmt for stmt in raygen.body if getattr(stmt, "name", "") == "myRay")
    payload = next(
        stmt for stmt in raygen.body if getattr(stmt, "name", "") == "payload"
    )

    assert isinstance(my_ray.value, InitializerListNode)
    assert len(my_ray.value.elements) == 4
    assert isinstance(payload.value, InitializerListNode)
    assert payload.value.elements == [0]


def test_parse_vulkan_samples_relaxed_instruction_class_method_shader():
    ast = parse_code(textwrap.dedent("""
            class A {
              void foo(uint v) {
                printf("relaxed-ext-inst demo: value = %u", v);
              }
            };

            [numthreads(1, 1, 1)]
            void main(uint3 gid : SV_DispatchThreadID) {
              A a;
              if (all(gid == uint3(0, 0, 0))) { a.foo(1); }
            }
            """))

    assert [function.name for function in ast.functions] == ["main"]
    main = ast.functions[0]
    assert main.qualifier == "compute"
    assert any(getattr(stmt, "name", "") == "a" for stmt in main.body)


def test_parse_class_qualified_method_definition_from_sdk_samples():
    ast = parse_code(textwrap.dedent("""
            interface iBaseLight {
                float3 IlluminateAmbient(float3 normal);
            };

            class cAmbientLight : iBaseLight {
                float3 m_vLightColor;
                bool m_bEnable;
                float3 IlluminateAmbient(float3 normal);
            };

            float3 cAmbientLight::IlluminateAmbient(float3 normal) {
                return m_vLightColor * m_bEnable;
            }
            """))

    assert [function.name for function in ast.functions] == [
        "cAmbientLight::IlluminateAmbient"
    ]
    method = ast.functions[0]
    assert method.return_type == "float3"
    assert [param.name for param in method.params] == ["normal"]


def test_parse_const_qualified_member_functions_from_hlsl_specs():
    # Source: https://microsoft.github.io/hlsl-specs/proposals/0007-const-member-functions/
    ast = parse_code(textwrap.dedent("""
            struct Hat {
                int getFeathers() const;
                int Feathers;
            };

            struct Pupper {
                void Wag() const { }
            };

            int Hat::getFeathers() const {
                return Feathers;
            }
            """))

    structs_by_name = {struct.name: struct for struct in ast.structs}
    hat_method = structs_by_name["Hat"].methods[0]
    wag_method = structs_by_name["Pupper"].methods[0]
    definition = ast.functions[0]

    assert hat_method.name == "getFeathers"
    assert hat_method.is_prototype is True
    assert hat_method.qualifiers == ["const"]
    assert wag_method.name == "Wag"
    assert wag_method.qualifiers == ["const"]
    assert definition.name == "Hat::getFeathers"
    assert definition.qualifiers == ["const"]


def test_parse_struct_base_lists_from_dxc_rewriter():
    ast = parse_code(textwrap.dedent("""
            interface my_interface {
            };

            class my_class {
            };

            struct my_struct_4 : my_interface {
            };

            struct my_struct_5 : my_class, my_interface {
            };
            """))

    structs = {struct.name: struct for struct in ast.structs}

    assert structs["my_struct_4"].base_classes == ["my_interface"]
    assert structs["my_struct_5"].base_classes == ["my_class", "my_interface"]


def test_parse_templated_struct_base_from_dxc_templates():
    # Source: microsoft/DirectXShaderCompiler@8ed708842c1ccb24bd914eff03125c837a01be71
    # tools/clang/test/CodeGenDXIL/templates/incomplete-target-in-CanConvert.hlsl
    ast = parse_code(textwrap.dedent("""
            template<typename T> struct Wrapper;
            float get(float x) { return x; }
            float get(Wrapper<float> o);

            template<typename T> struct Wrapper2;
            float get(Wrapper2<float> o);

            template<typename T>
            struct Wrapper {
                T value;
                void set(float x) {
                    value = get(x);
                }
            };

            float get(Wrapper<float> o) { return o.value; }

            template<typename T>
            struct Wrapper2 : Wrapper<T> {
            };

            float get(Wrapper2<float> o) { return o.value; }

            float main(float x : IN) : OUT {
                Wrapper2<float> w;
                w.set(x);
                return get(w);
            }
            """))

    structs = {struct.name: struct for struct in ast.structs}
    main = next(function for function in ast.functions if function.name == "main")
    local = next(stmt for stmt in main.body if getattr(stmt, "name", "") == "w")

    assert structs["Wrapper2"].base_classes == ["Wrapper<T>"]
    assert local.vtype == "Wrapper2<float>"


def test_parse_elaborated_struct_function_signature_and_local_from_vkd3d():
    ast = parse_code(textwrap.dedent("""
            struct input {
                struct {
                    float4 texcoord : texcoord;
                } m;
            };

            struct output {
                struct {
                    float4 color : sv_target;
                } m;
            };

            struct output main(struct input i) {
                struct output o;
                o.m.color = i.m.texcoord;
                return o;
            }
            """))

    function = ast.functions[0]
    local_output = function.body[0]

    assert function.return_type == "output"
    assert function.params[0].vtype == "input"
    assert local_output.vtype == "output"
    assert local_output.name == "o"


def test_parse_elaborated_struct_member_from_dxc_cast_subscript():
    # Source: microsoft/DirectXShaderCompiler@main
    # tools/clang/test/HLSLFileCheck/hlsl/types/cast/mat_cast_sub_write.hlsl
    ast = parse_code(textwrap.dedent("""
            struct B {
                uint4 ui;
            };

            struct M {
                struct B base;
                float4 color;
            };
            """))

    members = ast.structs[1].members

    assert members[0].vtype == "B"
    assert members[0].name == "base"
    assert members[1].vtype == "float4"
    assert members[1].name == "color"


def test_parse_anonymous_embedded_struct_members_from_dxc_workgraphs():
    # Source: microsoft/DirectXShaderCompiler@main
    # tools/clang/test/HLSLFileCheck/hlsl/workgraph/nested_sv_dispatchgrid.hlsl
    ast = parse_code(textwrap.dedent("""
            struct Record1 {
                struct {
                    uint3 grid : SV_DispatchGrid;
                };
            };
            """))

    record = ast.structs[0]

    assert len(record.members) == 1
    assert record.members[0].vtype == "uint3"
    assert record.members[0].name == "grid"
    assert record.members[0].semantic == "SV_DispatchGrid"


def test_parse_elaborated_struct_declarations_from_dxc_rewriter():
    ast = parse_code(textwrap.dedent("""
            struct my_struct_type_decl {
                int a;
            };

            const struct my_struct_type_decl my_struct_var_decl;

            struct my_struct_type_init {
                int a;
            };

            const struct my_struct_type_init my_struct_type_init_one = { 1 };
            """))

    globals_by_name = {variable.name: variable for variable in ast.global_variables}

    assert globals_by_name["my_struct_var_decl"].vtype == "my_struct_type_decl"
    assert globals_by_name["my_struct_var_decl"].qualifiers == ["const"]
    assert globals_by_name["my_struct_type_init_one"].vtype == "my_struct_type_init"
    assert isinstance(
        globals_by_name["my_struct_type_init_one"].value, InitializerListNode
    )


def test_parse_control_flow_declaration_conditions_from_dxc_rewriter():
    ast = parse_code(textwrap.dedent("""
            int global_fn() {
                return 1;
            }

            void statements() {
                int local_i = 1;

                if (int my_if_local = global_fn()) {
                    my_if_local++;
                } else {
                    my_if_local--;
                }

                switch (int my_switch_local = global_fn()) {
                case 0:
                    my_switch_local--;
                    return;
                }

                while (int my_while_local = global_fn()) {
                    my_while_local--;
                }
            }
            """))

    body = ast.functions[1].body
    if_node = next(statement for statement in body if isinstance(statement, IfNode))
    switch_node = next(
        statement for statement in body if isinstance(statement, SwitchNode)
    )
    while_node = next(
        statement for statement in body if isinstance(statement, WhileNode)
    )

    assert if_node.condition.name == "my_if_local"
    assert switch_node.expression.name == "my_switch_local"
    assert while_node.condition.name == "my_while_local"


def test_parse_hex_escape_char_literals_from_dxc_rewriter():
    ast = parse_code(r"""
        void expressions() {
            int local_i;
            local_i = 'c';
            local_i = '\xff';
            local_i = '\x94';
        }
    """)

    assignments = [
        statement
        for statement in ast.functions[0].body
        if isinstance(statement, AssignmentNode)
    ]

    assert [assignment.right for assignment in assignments] == [
        "'c'",
        r"'\xff'",
        r"'\x94'",
    ]


def test_parse_global_static_const_array_initializer():
    ast = parse_code("static const float Weights[2] = { 0.25f, 0.75f };")

    weights = ast.global_variables[0]
    assert weights.name == "Weights"
    assert weights.vtype == "float"
    assert weights.qualifiers == ["static", "const"]
    assert weights.is_const is True
    assert weights.array_sizes == [2]
    assert isinstance(weights.value, InitializerListNode)
    assert weights.value.elements == [0.25, 0.75]


def test_parse_control_flow_and_operators():
    assert_parses(CONTROL_FLOW_HLSL)


def test_parse_arrays_and_indexing():
    assert_parses(ARRAYS_HLSL)


def test_parse_scalar_literal_swizzle_from_saschawillems_input_attachment():
    ast = parse_code("""
    float4 main() : SV_Target0 {
        return 0.xxxx;
    }
    """)

    value = ast.functions[0].body[0].value
    assert isinstance(value, MemberAccessNode)
    assert value.object == 0
    assert value.member == "xxxx"


def test_parse_resources_and_bindings():
    assert_parses(RESOURCES_HLSL)


def test_parse_function_overloads_and_calls():
    assert_parses(OVERLOADS_HLSL)


def test_parse_compute_attributes_and_semantics():
    assert_parses(COMPUTE_HLSL)


def test_parse_geometry_primitive_before_direction_from_dxc_spirv():
    # Source: microsoft/DirectXShaderCompiler@517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # tools/clang/test/CodeGenSPIRV/primitive.point.gs.hlsl
    ast = parse_code("""
    struct S { float4 val : VAL; };

    [maxvertexcount(3)]
    void main(point in uint id[1] : VertexID, inout LineStream<S> outData) {
    }
    """)

    function = ast.functions[0]
    id_param, stream_param = function.params

    assert function.qualifier == "geometry"
    assert id_param.vtype == "uint"
    assert id_param.name == "id"
    assert id_param.qualifiers == ["in"]
    assert id_param.array_sizes == [1]
    assert id_param.semantic == "VertexID"
    assert [(attr.name, attr.args) for attr in id_param.attributes] == [
        ("primitive", ["point"])
    ]
    assert stream_param.vtype == "LineStream<S>"
    assert stream_param.qualifiers == ["inout"]


def test_parse_noperspective_interpolation_modifier_from_hlsl_docs():
    ast = parse_code("""
    struct PSInput {
        centroid noperspective float2 uv : TEXCOORD0;
        noperspective float4 color : COLOR0;
    };

    float4 PSMain(noperspective float4 color : COLOR0) : SV_Target0 {
        return color;
    }
    """)

    struct = ast.structs[0]
    uv, color = struct.members
    param = ast.functions[0].params[0]

    assert uv.vtype == "float2"
    assert uv.name == "uv"
    assert uv.semantic == "TEXCOORD0"
    assert uv.qualifiers == ["centroid", "noperspective"]
    assert color.vtype == "float4"
    assert color.qualifiers == ["noperspective"]
    assert param.vtype == "float4"
    assert param.qualifiers == ["noperspective"]


def test_parse_post_semantic_interpolation_modifiers_from_hlsl_function_docs():
    # Source: https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-function-parameters
    ast = parse_code("""
    float4 PSMain(
        float4 color : COLOR0 noperspective,
        uint materialId : TEXCOORD1 : nointerpolation
    ) : SV_Target0 {
        return color + float4(materialId, 0, 0, 0);
    }
    """)

    color, material_id = ast.functions[0].params

    assert color.vtype == "float4"
    assert color.name == "color"
    assert color.semantic == "COLOR0"
    assert color.qualifiers == ["noperspective"]
    assert material_id.vtype == "uint"
    assert material_id.name == "materialId"
    assert material_id.semantic == "TEXCOORD1"
    assert material_id.qualifiers == ["nointerpolation"]


def test_parse_clipplanes_function_modifier_from_hlsl_function_docs():
    # Source: https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-function-syntax
    ast = parse_code("""
    inline clipplanes(userClip0, userClip1) precise float4 VSMain(
        float3 position : POSITION
    ) : SV_Position {
        return float4(position, 1.0);
    }
    """)

    function = ast.functions[0]

    assert function.name == "VSMain"
    assert function.qualifiers == ["inline", "precise", "vertex"]
    assert [(attr.name, attr.args) for attr in function.attributes] == [
        ("clipplanes", ["userClip0", "userClip1"])
    ]


def test_parse_center_interpolation_modifier_from_dxc_center_keyword():
    # Source: microsoft/DirectXShaderCompiler@517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # tools/clang/test/HLSLFileCheck/hlsl/types/modifiers/center/center_kwd.hlsl
    ast = parse_code("""
    float main(center float t : T) : SV_TARGET
    {
        float center = 10.0f;
        return center * 2;
    }
    """)

    param = ast.functions[0].params[0]
    local = ast.functions[0].body[0]

    assert param.vtype == "float"
    assert param.name == "t"
    assert param.semantic == "T"
    assert param.qualifiers == ["center"]
    assert local.vtype == "float"
    assert local.name == "center"
    assert local.qualifiers == []


def test_parse_contextual_shared_storage_modifier_from_hlsl_docs():
    ast = parse_code("""
    shared float cachedWeight;
    Texture2D<float> shared : register(t0);

    float4 PSMain(float2 uv : TEXCOORD0) : SV_Target0 {
        return shared.SampleLevel(samplerState, uv, 0.0);
    }
    """)

    cached_weight = ast.global_variables[0]
    shared_texture = ast.global_variables[1]

    assert cached_weight.name == "cachedWeight"
    assert cached_weight.vtype == "float"
    assert cached_weight.qualifiers == ["shared"]
    assert shared_texture.name == "shared"
    assert shared_texture.vtype == "Texture2D<float>"
    assert shared_texture.qualifiers == []


def test_parse_snorm_unorm_scalar_modifiers_from_dxc_rewriter_samples():
    ast = parse_code("""
    snorm float globalWeight;

    void main() {
        float left;
        unorm min16float right;
        left = right;
    }
    """)

    global_weight = ast.global_variables[0]
    left, right, assignment = ast.functions[0].body

    assert global_weight.name == "globalWeight"
    assert global_weight.qualifiers == ["snorm"]
    assert left.name == "left"
    assert left.qualifiers == []
    assert right.name == "right"
    assert right.vtype == "min16float"
    assert right.qualifiers == ["unorm"]
    assert assignment.operator == "="


def test_parse_post_type_const_modifiers_from_dxc_rewriter_gold_samples():
    ast = parse_code("""
    row_major float2x3 const g_row;
    snorm float const g_scalar;

    cbuffer CBInit {
        row_major float2x3 const g_row_init = g_row;
    };

    typedef row_major float2x3 const RowMajorConstMatrix;

    float3 foo(row_major float2x3 const val) {
        return val[0];
    }
    """)

    g_row, g_scalar = ast.global_variables
    cbuffer_member = ast.cbuffers[0].members[0]
    typedef = ast.typedefs[0]
    param = ast.functions[0].params[0]

    assert g_row.qualifiers == ["row_major", "const"]
    assert g_scalar.qualifiers == ["snorm", "const"]
    assert cbuffer_member.qualifiers == ["row_major", "const"]
    assert typedef.qualifiers == ["row_major", "const"]
    assert param.qualifiers == ["row_major", "const"]


def test_parse_rootsignature_macro_adjacent_string_literals():
    code = r"""
    #define RootSig \
        "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT|ALLOW_STREAM_OUTPUT)," \
        "DescriptorTable(SRV(t0, numDescriptors=1))," \
        "DescriptorTable(UAV(u0, numDescriptors=2))," \
        "DescriptorTable(Sampler(s0, numDescriptors=2))"

    [RootSignature(RootSig)]
    float4 RootSignaturePS(float4 pos : SV_POSITION) : SV_TARGET {
        return pos;
    }
    """

    ast = parse_code(code)
    attributes = ast.functions[0].attributes

    assert attributes[0].name == "RootSignature"
    assert attributes[0].args == [
        '"RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT|ALLOW_STREAM_OUTPUT),'
        "DescriptorTable(SRV(t0, numDescriptors=1)),"
        "DescriptorTable(UAV(u0, numDescriptors=2)),"
        'DescriptorTable(Sampler(s0, numDescriptors=2))"'
    ]


def test_parse_interpolation_intrinsics_keep_free_function_calls():
    code = """
    float4 main(float4 color : COLOR0, uint sampleIndex : SV_SampleIndex) : SV_Target0 {
        int2 snappedOffset = int2(1, -1);
        float4 atSample = EvaluateAttributeAtSample(color, sampleIndex);
        float4 atOffset = EvaluateAttributeSnapped(color, snappedOffset);
        float4 atCentroid = EvaluateAttributeCentroid(color);
        return atSample + atOffset + atCentroid;
    }
    """

    ast = parse_code(code)
    free_calls = [
        node.name
        for node in iter_ast_nodes(ast)
        if isinstance(node, FunctionCallNode) and isinstance(node.name, str)
    ]

    assert "EvaluateAttributeAtSample" in free_calls
    assert "EvaluateAttributeSnapped" in free_calls
    assert "EvaluateAttributeCentroid" in free_calls


def test_parse_namespace_block_flattens_and_preserves_scoped_call_name():
    code = """
    using namespace dx;

    namespace CrossBilateral {
        float Weight(float value) {
            return value;
        }
    }

    float UseWeight(float value) {
        return CrossBilateral::Weight(value);
    }
    """

    ast = parse_code(code)
    calls = [
        node.name
        for node in iter_ast_nodes(ast)
        if isinstance(node, FunctionCallNode) and isinstance(node.name, str)
    ]

    assert [function.name for function in ast.functions] == ["Weight", "UseWeight"]
    assert "CrossBilateral::Weight" in calls


def test_parse_global_scope_hlsl_namespace_intrinsic_calls():
    code = """
    float4 main(float4 color : COLOR0) : SV_Target0 {
        float a = hlsl::saturate(color.x);
        float b = ::hlsl::lerp(a, color.y, 0.5);
        return float4(b, a, 0.0, 1.0);
    }
    """

    ast = parse_code(code)
    calls = [
        node.name
        for node in iter_ast_nodes(ast)
        if isinstance(node, FunctionCallNode) and isinstance(node.name, str)
    ]

    assert "hlsl::saturate" in calls
    assert "hlsl::lerp" in calls


def test_parse_clip_intrinsic_expression_statement():
    code = """
    float4 PSMain(float4 color : COLOR0) : SV_Target {
        clip(color.a < 0.1f ? -1 : 1);
        return color;
    }
    """

    ast = parse_code(code)
    calls = [
        node
        for node in iter_ast_nodes(ast)
        if isinstance(node, FunctionCallNode) and node.name == "clip"
    ]

    assert len(calls) == 1


def test_parse_clip_identifier_from_vkd3d_clip_distance_sample():
    code = """
    struct vertex {
        float4 position : SV_POSITION;
        float clip : SV_CLIPDISTANCE;
        float cull : SV_CULLDISTANCE1;
    };

    void main(float4 position : POSITION, out vertex vertex) {
        vertex.position = position;
        vertex.clip = position.y;
        vertex.cull = position.x;
        clip(vertex.clip - 0.5f);
    }
    """

    ast = parse_code(code)
    vertex_struct = ast.structs[0]
    main = ast.functions[0]
    clip_assignment = main.body[1]
    clip_calls = [
        node
        for node in iter_ast_nodes(ast)
        if isinstance(node, FunctionCallNode) and node.name == "clip"
    ]

    assert vertex_struct.members[1].name == "clip"
    assert vertex_struct.members[1].semantic == "SV_CLIPDISTANCE"
    assert vertex_struct.members[2].name == "cull"
    assert vertex_struct.members[2].semantic == "SV_CULLDISTANCE1"
    assert isinstance(clip_assignment.left, MemberAccessNode)
    assert clip_assignment.left.member == "clip"
    assert len(clip_calls) == 1


def test_parse_export_function_specifier():
    ast = parse_code("export void LogTraceRayStart() { }")

    assert ast.functions[0].name == "LogTraceRayStart"
    assert ast.functions[0].return_type == "void"
    assert "export" in ast.functions[0].qualifiers


def test_parse_preprocessor_directives():
    assert_parses(PREPROCESSOR_HLSL)


def test_parse_pragma_once_from_directx_fallback_samples():
    ast = parse_code("""
    #pragma once

    struct RWByteAddressBufferPointer {
        RWByteAddressBuffer buffer;
        uint offsetInBytes;
    };
    """)

    pragma = ast.structs[0]

    assert pragma.directive == "once"
    assert pragma.value is None


def test_preprocessor_evaluates_conditionals():
    code = """
    #define ENABLE_BAD 0
    #if ENABLE_BAD
    int broken = ;
    #endif

    int ok() { return 1; }
    """
    assert_parses(code)


def test_preprocessor_ifdef_trailing_comment_keeps_enabled_branch():
    code = """
    #ifdef _WAVE_OP // SM 6.0
    uint WaveOr(uint mask) { return mask; }
    #endif
    """
    tokens = HLSLLexer(code, defines={"_WAVE_OP": "1"}).tokenize()
    ast = HLSLParser(tokens).parse()

    assert [function.name for function in ast.functions] == ["WaveOr"]


def test_preprocessor_preserves_unresolved_system_include():
    code = """
    #include <d3dcommon.h>
    float4 main() : SV_Target { return float4(1.0, 0.0, 0.0, 1.0); }
    """

    processed = HLSLPreprocessor().preprocess(textwrap.dedent(code))

    assert "#include <d3dcommon.h>" in processed
    assert "float4 main()" in processed


def test_parse_unsigned_long_long_integer_suffixes_from_directx_samples():
    code = """
    uint64_t f() {
        uint64_t laneBit = 1ull << WaveGetLaneIndex();
        uint64_t otherBit = 1llu;
        uint64_t hexBit = 0xffull;
        return laneBit | otherBit | hexBit;
    }
    """
    ast = parse_code(code)

    assert ast.functions[0].name == "f"


def test_parse_enum_and_typedef():
    code = """
    enum BlendMode {
        BlendOpaque = 0,
        BlendAdd = 1,
    };
    typedef float4 Color;
    typedef row_major float2x3 RowMajorMatrix;
    typedef precise const float2 PreciseConstVector;
    Color main(Color input) : SV_Target0 {
        return input;
    }
    """
    ast = parse_code(code)

    assert [typedef.name for typedef in ast.typedefs] == [
        "Color",
        "RowMajorMatrix",
        "PreciseConstVector",
    ]
    assert ast.typedefs[0].alias_type == "float4"
    assert ast.typedefs[0].qualifiers == []
    assert ast.typedefs[1].alias_type == "float2x3"
    assert ast.typedefs[1].qualifiers == ["row_major"]
    assert ast.typedefs[2].alias_type == "float2"
    assert ast.typedefs[2].qualifiers == ["precise", "const"]


def test_parse_const_incomplete_array_typedef_from_dxc():
    # Source: microsoft/DirectXShaderCompiler@d6e0ca4a0c25b13ed676c8ba16839c3eb9fcc652
    # tools/clang/test/HLSLFileCheck/hlsl/types/array/incomp_array.hlsl
    ast = parse_code("typedef const int inta[];")

    typedef = ast.typedefs[0]
    assert typedef.name == "inta"
    assert typedef.alias_type == "int"
    assert typedef.qualifiers == ["const"]
    assert typedef.array_sizes == [None]


def test_parse_using_alias_declarations_from_hlsl_spec():
    ast = parse_code("""
    using Float = float;
    using Color = vector<float, 4>;
    using RowMajorMatrix = row_major float2x3 const;
    using IndexPair = int[2];
    using namespace dx;
    using FilterKernel::Radius;

    Color main(Color input) : SV_Target0 {
        return input;
    }
    """)

    assert [typedef.name for typedef in ast.typedefs] == [
        "Float",
        "Color",
        "RowMajorMatrix",
        "IndexPair",
    ]
    assert ast.typedefs[0].alias_type == "float"
    assert ast.typedefs[1].alias_type == "vector<float, 4>"
    assert ast.typedefs[2].alias_type == "float2x3"
    assert ast.typedefs[2].qualifiers == ["row_major", "const"]
    assert ast.typedefs[3].alias_type == "int"
    assert ast.typedefs[3].array_sizes == [2]
    assert [function.name for function in ast.functions] == ["main"]


def test_parse_struct_member_using_declarations_from_dxc_cpp_style():
    ast = parse_code("""
    struct Base {
        float value;
    };

    struct Derived : Base {
        using Base::value;
        float other;
    };

    float main(Derived input) : SV_Target0 {
        return input.other;
    }
    """)

    derived = next(struct for struct in ast.structs if struct.name == "Derived")
    assert [member.name for member in derived.members] == ["other"]
    assert ast.functions[0].params[0].vtype == "Derived"


def test_parse_block_scope_typedef_from_dxc_linalg_vectors():
    code = """
    ByteAddressBuffer BAB : register(t0);

    [numthreads(4, 4, 4)]
    void main(uint ID : SV_GroupID) {
      typedef vector<half, 16> half16;
      half16 srcF16 = BAB.Load<half16>(128);
    }
    """
    ast = parse_code(code)

    assert ast.typedefs == []
    assert len(ast.functions[0].body) == 1
    assert ast.functions[0].body[0].vtype == "half16"
    assert ast.functions[0].body[0].name == "srcF16"


def test_parse_block_scope_class_declarations_from_dxc_spec():
    # Source: https://github.com/microsoft/DirectXShaderCompiler
    # Commit: 517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # Path: tools/clang/test/SemaHLSL/spec.hlsl
    ast = parse_code("""
    namespace ns_general {
      void subobjects() {
        class Class { uint field; };
        class SuperClass { Class C; uint field; };

        Class C = { 0 };
        SuperClass SC = { 0, 0 };
      }
    }
    """)

    body = ast.functions[0].body

    assert [(statement.vtype, statement.name) for statement in body] == [
        ("Class", "C"),
        ("SuperClass", "SC"),
    ]


def test_parse_enum_underlying_type_from_dxc_sema_enums():
    # Source: https://github.com/microsoft/DirectXShaderCompiler
    # Commit: 517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # Path: tools/clang/test/SemaHLSL/enums.hlsl
    ast = parse_code("""
    enum MyEnumUInt : uint {
        ZEROU,
        ONEU,
        FOURU = 4,
    };

    enum class MyEnumMin16int : min16int {
        ZEROMIN16INT,
    };
    """)

    unsigned_enum, scoped_enum = ast.enums

    assert unsigned_enum.name == "MyEnumUInt"
    assert unsigned_enum.underlying_type == "uint"
    assert unsigned_enum.is_scoped is False
    assert [name for name, _ in unsigned_enum.members] == [
        "ZEROU",
        "ONEU",
        "FOURU",
    ]
    assert scoped_enum.name == "MyEnumMin16int"
    assert scoped_enum.underlying_type == "min16int"
    assert scoped_enum.is_scoped is True


def test_parse_anonymous_enum_constants_inside_namespace_from_directx_samples():
    code = """
    namespace SMem
    {
        namespace Size
        {
            enum {
                Histogram = NUM_KEYS,
            };
        }

        namespace Offset
        {
            enum {
                Histogram = 0,
                Key8b = Size::Histogram,
            };
        }
    }
    """
    ast = parse_code(code)

    assert len(ast.enums) == 2
    assert ast.enums[0].name == "AnonymousEnum_1"
    assert ast.enums[1].name == "AnonymousEnum_2"
    assert [name for name, _ in ast.enums[0].members] == ["Histogram"]
    assert [name for name, _ in ast.enums[1].members] == ["Histogram", "Key8b"]


def test_parse_namespaces_inside_cbuffer_from_dxc_cbuffer_layout():
    # Source: microsoft/DirectXShaderCompiler@517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # tools/clang/test/HLSLFileCheck/hlsl/objects/Cbuffer/namespace_in_cb.hlsl
    ast = parse_code("""
    namespace N {
    float a;
    }

    cbuffer B {
      namespace N {
      float c;
      }
      namespace N2 {
        float d;
      }
    }

    namespace N2 {
      float b;
    }

    float main() : SV_Target {
      return N::a + N::c + N2::d + N2::b;
    }
    """)

    assert [variable.name for variable in ast.global_variables] == ["a", "b"]
    assert ast.cbuffers[0].name == "B"
    assert [member.name for member in ast.cbuffers[0].members] == ["c", "d"]
    assert [member.vtype for member in ast.cbuffers[0].members] == ["float", "float"]
    assert ast.functions[0].name == "main"


def test_parse_resource_arrays_and_register_space():
    code = """
    Texture2D textures[4] : register(t0, space1);
    RWTexture1DArray rwTexArray[2] : register(u1, space2);
    SamplerState samplers[4] : register(s0);
    float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
        return textures[0].Sample(samplers[0], uv);
    }
    """
    assert_parses(code)


def test_parse_sampler_state_initializer_blocks_from_microsoft_docs():
    code = """
    SamplerState MeshTextureSampler
    {
        Filter = MIN_MAG_MIP_LINEAR;
        AddressU = Wrap;
        AddressV = Wrap;
    };

    SamplerComparisonState ShadowSampler
    {
        Filter = COMPARISON_MIN_MAG_LINEAR_MIP_POINT;
        AddressU = Clamp;
        ComparisonFunc = LESS;
    };
    """
    ast = parse_code(code)
    mesh_sampler, shadow_sampler = ast.global_variables

    assert mesh_sampler.vtype == "SamplerState"
    assert mesh_sampler.sampler_state == [
        ("Filter", "MIN_MAG_MIP_LINEAR"),
        ("AddressU", "Wrap"),
        ("AddressV", "Wrap"),
    ]
    assert shadow_sampler.vtype == "SamplerComparisonState"
    assert shadow_sampler.sampler_state == [
        ("Filter", "COMPARISON_MIN_MAG_LINEAR_MIP_POINT"),
        ("AddressU", "Clamp"),
        ("ComparisonFunc", "LESS"),
    ]


def test_parse_legacy_d3d9_sampler_state_initializer_texture_binding():
    code = """
    texture g_MeshTexture;
    sampler MeshTextureSampler = sampler_state {
        Texture = <g_MeshTexture>;
    };
    """
    ast = parse_code(code)
    texture_decl, sampler_decl = ast.global_variables

    assert texture_decl.vtype == "texture"
    assert sampler_decl.vtype == "sampler"
    assert sampler_decl.value is None
    assert sampler_decl.sampler_state == [("Texture", "g_MeshTexture")]


def test_skip_deprecated_effect_annotations_and_state_blocks_from_dxc_rewriter():
    code = """
    Texture2D tex : register(t1), tex2 : register(t2)
    < int foo=1; >
    {
        Texture = tex;
        Filter = MIN_MAG_MIP_LINEAR;
    }, texa[3]
    <
        string Name = "texa";
        int ArraySize = 3;
    >;

    SamplerState samLinear : register(s7)
    < bool foo=1 > 2; >
    {
        Texture = tex;
        Filter = MIN_MAG_MIP_LINEAR;
    };

    float3 annotatedColor : Ambient
    <
        string SasUiLabel = "Ambient";
        string SasUiControl = "ColorPicker";
    > = { 0.2f, 0.2f, 0.2f };

    float4 main() : SV_Target {
        Texture2D localTex { state=foo; };
        int foobar {blah=foo;};
        return tex.Sample(samLinear, float2(0.1, 0.2));
    }

    technique T0 {
        pass {}
    }

    Technique {
        pass {}
    }

    technique10 T10 {
        pass {}
    }

    technique11 T11 {
        pass {}
    }

    technique PostProcess
    <
        string Parameter0 = "BloomScale";
        float4 Parameter0Def = float4(1.5f, 0, 0, 0);
        int Parameter0Size = 1;
        string Parameter0Desc = " (float)";
    >
    {
        pass p0
        {
            VertexShader = null;
            PixelShader = compile ps_2_0 main();
            ZEnable = false;
        }
    }
    """

    ast = parse_code(code)

    assert [variable.name for variable in ast.global_variables] == [
        "tex",
        "tex2",
        "texa",
        "samLinear",
        "annotatedColor",
    ]
    assert ast.global_variables[0].register == "t1"
    assert ast.global_variables[1].register == "t2"
    assert ast.global_variables[2].array_sizes == [3]
    assert ast.global_variables[3].register == "s7"
    assert isinstance(ast.global_variables[4].value, InitializerListNode)
    assert [statement.name for statement in ast.functions[0].body[:2]] == [
        "localTex",
        "foobar",
    ]


def test_skip_top_level_pmfx_style_effect_metadata_blocks():
    ast = parse_code("""
    state default {
        DepthEnable = true;
    }

    program p0 {
        vs = vs_main;
        ps = ps_main;
    }

    fxgroup PostProcess {
        technique10 Render {
            pass P0 {
                PixelShader = compile ps_5_0 ps_main();
            }
        }
    }

    pass ExtractedPass {
        PixelShader = compile ps_5_0 ps_main();
    }

    float4 ps_main() : SV_Target {
        return 1;
    }
    """)

    assert [function.name for function in ast.functions] == ["ps_main"]
    assert ast.global_variables == []


def test_parse_rasterizer_ordered_resources_and_register_space():
    code = """
    RasterizerOrderedTexture2D<uint> counters : register(u0, space1);
    RasterizerOrderedTexture2DArray<float4> layers[2] : register(u1, space2);
    RasterizerOrderedBuffer<uint> bins : register(u3);
    RasterizerOrderedStructuredBuffer<int> values : register(u4);
    RasterizerOrderedByteAddressBuffer bytes : register(u5, space3);

    float4 PSMain(uint2 pixel : TEXCOORD0, uint layer : TEXCOORD1) : SV_Target0 {
        uint oldCount;
        InterlockedAdd(counters[pixel], 1u, oldCount);
        values[0] = int(oldCount);
        bytes.Store(0, oldCount);
        return layers[0][uint3(pixel, layer)];
    }
    """

    ast = parse_code(code)
    globals_by_name = {node.name: node for node in ast.global_variables}

    assert globals_by_name["counters"].vtype == "RasterizerOrderedTexture2D<uint>"
    assert globals_by_name["counters"].register == "u0, space1"
    assert globals_by_name["layers"].vtype == "RasterizerOrderedTexture2DArray<float4>"
    assert globals_by_name["layers"].array_sizes == [2]
    assert globals_by_name["layers"].register == "u1, space2"
    assert globals_by_name["bins"].vtype == "RasterizerOrderedBuffer<uint>"
    assert globals_by_name["values"].vtype == "RasterizerOrderedStructuredBuffer<int>"
    assert globals_by_name["bytes"].vtype == "RasterizerOrderedByteAddressBuffer"
    assert globals_by_name["bytes"].register == "u5, space3"


def test_parse_min_precision_vector_and_matrix_types():
    code = """
    struct MinPrecisionData {
        min16float3 hdr : COLOR0;
        min10float2 uv : TEXCOORD0;
        min16float2x3 colorMatrix;
    };

    min16float3 Shade(
        min16float3 color,
        min12int2 offset,
        min16uint4 mask
    ) {
        min10float2 localUv = min10float2(0.0, 1.0);
        return min16float3(color.x, localUv.x, localUv.y);
    }
    """

    ast = parse_code(code)
    struct = ast.structs[0]
    func = ast.functions[0]

    assert [member.vtype for member in struct.members] == [
        "min16float3",
        "min10float2",
        "min16float2x3",
    ]
    assert func.return_type == "min16float3"
    assert [param.vtype for param in func.params] == [
        "min16float3",
        "min12int2",
        "min16uint4",
    ]
    local_uv = next(
        node
        for node in iter_ast_nodes(func)
        if getattr(node, "name", None) == "localUv"
    )
    assert local_uv.vtype == "min10float2"


def test_parse_exact_16_bit_scalar_types_from_hlsl_docs():
    # Source: https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-scalar
    # Source: https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-vector
    ast = parse_code("""
    float16_t halfValue;
    int16_t signedValue;
    uint16_t unsignedValue;

    vector<float16_t> MakeHalfVector(float16_t seed) {
        float16_t local = float16_t(seed);
        return vector<float16_t>(local, local, local, local);
    }
    """)

    assert [variable.vtype for variable in ast.global_variables] == [
        "float16_t",
        "int16_t",
        "uint16_t",
    ]
    function = ast.functions[0]
    assert function.return_type == "vector<float16_t>"
    assert function.params[0].vtype == "float16_t"
    assert function.body[0].vtype == "float16_t"
    assert isinstance(function.body[0].value, VectorConstructorNode)
    assert function.body[0].value.type_name == "float16_t"


def test_parse_exact_32_bit_scalar_types_from_hlsl_docs():
    # Source: https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-scalar
    ast = parse_code("""
    float32_t fullFloat;
    int32_t signedWord;
    uint32_t unsignedWord;

    float32_t Promote(uint32_t seed) {
        int32_t local = int32_t(seed);
        return float32_t(local);
    }
    """)

    assert [variable.vtype for variable in ast.global_variables] == [
        "float32_t",
        "int32_t",
        "uint32_t",
    ]
    function = ast.functions[0]
    assert function.return_type == "float32_t"
    assert function.params[0].vtype == "uint32_t"
    assert function.body[0].vtype == "int32_t"
    assert isinstance(function.body[0].value, VectorConstructorNode)
    assert function.body[0].value.type_name == "int32_t"
    assert isinstance(function.body[1].value, VectorConstructorNode)
    assert function.body[1].value.type_name == "float32_t"


def test_parse_template_style_vector_matrix_types_and_constructors():
    ast = parse_code("""
    struct TemplateTypes {
        vector<float, 3> normal : NORMAL;
        matrix<float, 3, 3> basis;
    };

    vector<double, 4> MakeTemplateVector(
        vector<float, 3> input : TEXCOORD0
    ) : SV_Target0 {
        matrix<float, 2, 3> localBasis;
        vector<float> defaultWidth = vector<float>(1.0, 2.0, 3.0, 4.0);
        return vector<double, 4>(input.x, input.y, input.z, 1.0);
    }
    """)

    struct = ast.structs[0]
    func = ast.functions[0]
    default_width = next(
        node
        for node in iter_ast_nodes(func)
        if getattr(node, "name", None) == "defaultWidth"
    )
    local_basis = next(
        node
        for node in iter_ast_nodes(func)
        if getattr(node, "name", None) == "localBasis"
    )

    assert [member.vtype for member in struct.members] == [
        "vector<float, 3>",
        "matrix<float, 3, 3>",
    ]
    assert func.return_type == "vector<double, 4>"
    assert func.params[0].vtype == "vector<float, 3>"
    assert local_basis.vtype == "matrix<float, 2, 3>"
    assert default_width.vtype == "vector<float>"
    assert isinstance(default_width.value, VectorConstructorNode)
    assert default_width.value.type_name == "vector<float>"
    assert isinstance(func.body[-1].value, VectorConstructorNode)
    assert func.body[-1].value.type_name == "vector<double, 4>"


def test_parse_template_vector_constant_expression_dimensions_from_dxc():
    # Source: microsoft/DirectXShaderCompiler@8ed708842c1ccb24bd914eff03125c837a01be71
    # tools/clang/test/SemaHLSL/vector-syntax.hlsl
    ast = parse_code("""
    vector<float, 1+1> literalSize;
    static const int i = 1;
    vector<float, i+i> constSize;
    """)

    assert [variable.vtype for variable in ast.global_variables] == [
        "vector<float, 1 + 1>",
        "int",
        "vector<float, i + i>",
    ]


def test_parse_default_template_style_vector_matrix_constructors_from_hlsl_docs():
    # Source: https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-vector
    # Source: https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-matrix
    ast = parse_code("""
    vector MakeDefaultVector(float value) {
        vector local = vector(value, value, value, value);
        return local;
    }

    matrix MakeDefaultMatrix(float value) {
        matrix local = matrix(
            value, value, value, value,
            value, value, value, value,
            value, value, value, value,
            value, value, value, value
        );
        return local;
    }
    """)

    vector_function, matrix_function = ast.functions
    vector_local = vector_function.body[0]
    matrix_local = matrix_function.body[0]

    assert vector_function.return_type == "vector"
    assert vector_local.vtype == "vector"
    assert isinstance(vector_local.value, VectorConstructorNode)
    assert vector_local.value.type_name == "vector"
    assert matrix_function.return_type == "matrix"
    assert matrix_local.vtype == "matrix"
    assert isinstance(matrix_local.value, VectorConstructorNode)
    assert matrix_local.value.type_name == "matrix"


def test_parse_function_array_return_declarator_from_dxc_longvec_decls():
    # Source: microsoft/DirectXShaderCompiler@517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # tools/clang/test/CodeGenDXIL/hlsl/types/longvec-decls.hlsl
    ast = parse_code("""
    vector<float, 4> lv_param_arr_passthru(vector<float, 4> vec[10])[10] {
        return vec;
    }
    """)

    func = ast.functions[0]

    assert func.return_type == "vector<float, 4>"
    assert func.array_sizes == [10]
    assert func.params[0].vtype == "vector<float, 4>"
    assert func.params[0].array_sizes == [10]


def test_parse_one_row_column_matrix_aliases_from_dxc_matrix_syntax():
    # Source: microsoft/DirectXShaderCompiler@517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # tools/clang/test/SemaHLSL/matrix-syntax.hlsl
    ast = parse_code("""
    void matrix_on_demand() {
        float1x2 f12;
        bool2x1 boolMatrix;
        unsigned int4x2 unsignedMatrix;
    }
    """)

    declarations = ast.functions[0].body

    assert [declaration.vtype for declaration in declarations] == [
        "float1x2",
        "bool2x1",
        "unsigned int4x2",
    ]
    assert [declaration.name for declaration in declarations] == [
        "f12",
        "boolMatrix",
        "unsignedMatrix",
    ]


def test_parse_template_function_prefix_from_raytracing_sample():
    ast = parse_code("""
    template<typename T>
    T InterpolateAttribute(T vertexAttribute[3], float2 barycentrics)
    {
        return vertexAttribute[0] +
            barycentrics.x * (vertexAttribute[1] - vertexAttribute[0]) +
            barycentrics.y * (vertexAttribute[2] - vertexAttribute[0]);
    }
    """)

    function = ast.functions[0]
    assert function.name == "InterpolateAttribute"
    assert function.return_type == "T"
    assert function.params[0].vtype == "T"
    assert function.params[0].array_sizes == [3]


def test_parse_struct_forward_declarations_from_dxc_incomplete_type_tests():
    # Sources:
    # microsoft/DirectXShaderCompiler tools/clang/test/SemaHLSL/sizeof-requires-complete-type.hlsl
    # microsoft/DirectXShaderCompiler tools/clang/test/CodeGenDXIL/templates/incomplete-target-in-CanConvert.hlsl
    ast = parse_code("""
    struct Incomplete;
    template<typename T> struct Wrapper;

    struct Complete {
        float value;
    };

    float get(Wrapper<float> o);
    float main(Complete c) : OUT {
        return c.value;
    }
    """)

    forward_declarations = [
        struct
        for struct in ast.structs
        if getattr(struct, "is_forward_declaration", False)
    ]
    definitions = [
        struct
        for struct in ast.structs
        if not getattr(struct, "is_forward_declaration", False)
    ]

    assert [struct.name for struct in forward_declarations] == [
        "Incomplete",
        "Wrapper",
    ]
    assert [struct.name for struct in definitions] == ["Complete"]
    assert definitions[0].members[0].name == "value"
    assert ast.functions[0].name == "get"
    assert ast.functions[0].is_prototype is True
    assert ast.functions[0].params[0].vtype == "Wrapper<float>"


def test_parse_nested_template_arguments_closed_by_shift_right_token_from_dxc():
    ast = parse_code("""
    template<typename T> struct Wrapper;

    float main(Wrapper<vector<float, 4>> input) : SV_Target0 {
        return 1.0;
    }
    """)

    assert ast.functions[0].params[0].vtype == "Wrapper<vector<float, 4>>"


def test_parse_hitobject_array_byval_parameter_modifier_macro_from_dxc():
    # microsoft/DirectXShaderCompiler
    # tools/clang/test/CodeGenDXIL/hlsl/objects/HitObject/hitobject-array-byval.hlsl
    ast = parse_code("""
    void MakeNop(MOD dx::HitObject obj[2]) {
      obj[0] = dx::HitObject::MakeNop();
      obj[1] = dx::HitObject::MakeNop();
    }
    """)

    param = ast.functions[0].params[0]

    assert param.vtype == "dx::HitObject"
    assert param.name == "obj"
    assert param.array_sizes == [2]
    assert param.qualifiers == []


def test_parse_struct_template_methods_from_dxc_spirv_resource_array():
    # Source: microsoft/DirectXShaderCompiler@517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # tools/clang/test/CodeGenSPIRV/use.rvalue.for.member-expr.of.array-subscript.hlsl
    ast = parse_code("""
    [[vk::binding(0, 0)]] ByteAddressBuffer babuf[]: register(t0, space0);
    [[vk::binding(0, 0)]] RWByteAddressBuffer rwbuf[]: register(u0, space0);

    struct BufferAccess {
      uint handle;

      template<typename T>
      T load(uint index) {
        return babuf[this.handle].Load<T>(index * sizeof(T));
      }

      template<typename T>
      void store(uint index, T value) {
        return rwbuf[this.handle].Store<T>(index * sizeof(T), value);
      }
    };

    struct A {
      uint x;
    };

    [numthreads(1, 1, 1)]
    void main(uint tid : SV_DispatchThreadId) {
      BufferAccess buf;
      A b = buf.load<A>(0);
      buf.store<A>(0, b);
    }
    """)

    babuf, rwbuf = ast.global_variables
    buffer_access = next(
        struct for struct in ast.structs if struct.name == "BufferAccess"
    )
    load_method, store_method = buffer_access.methods
    main = ast.functions[0]

    assert babuf.array_sizes == [None]
    assert babuf.register == "t0, space0"
    assert rwbuf.array_sizes == [None]
    assert rwbuf.register == "u0, space0"
    assert [(attr.name, attr.args) for attr in babuf.attributes] == [
        ("vk::binding", [0, 0])
    ]
    assert [method.name for method in buffer_access.methods] == ["load", "store"]
    assert load_method.return_type == "T"
    assert [param.vtype for param in load_method.params] == ["uint"]
    assert store_method.return_type == "void"
    assert [param.vtype for param in store_method.params] == ["uint", "T"]
    assert main.qualifier == "compute"
    assert main.params[0].semantic == "SV_DispatchThreadId"


def test_parse_template_specialized_struct_declarations_from_dxc_sfinae():
    # Source: microsoft/DirectXShaderCompiler@517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # tools/clang/test/SemaHLSL/template-implicit-this-sfinae.hlsl
    ast = parse_code("""
    template <typename T>
    struct is_arithmetic {
        static const bool value = false;
    };

    template <> struct is_arithmetic<float> {
        static const bool value = true;
    };
    """)

    assert [struct.name for struct in ast.structs] == [
        "is_arithmetic",
        "is_arithmetic",
    ]
    assert ast.structs[0].members[0].name == "value"
    assert ast.structs[0].members[0].value is False
    assert ast.structs[1].members[0].name == "value"
    assert ast.structs[1].members[0].value is True


def test_parse_elaborated_template_struct_variable_from_dxc_template_struct():
    # Source: microsoft/DirectXShaderCompiler@d6e0ca4a0c25b13ed676c8ba16839c3eb9fcc652
    # tools/clang/test/HLSLFileCheck/hlsl/template/templateStruct.hlsl
    ast = parse_code("""
    template<typename T>
    struct TS {
      T t;
    };

    struct TS<float4> ts;

    float4 main() : SV_Target {
      return ts.t;
    }
    """)

    assert [struct.name for struct in ast.structs] == ["TS"]
    assert ast.global_variables[0].vtype == "TS<float4>"
    assert ast.global_variables[0].name == "ts"
    assert ast.functions[0].name == "main"


def test_parse_dependent_typename_scoped_template_return_from_dxc_sfinae():
    # Source: microsoft/DirectXShaderCompiler@8ed708842c1ccb24bd914eff03125c837a01be71
    # tools/clang/test/SemaHLSL/template-implicit-this-sfinae.hlsl
    ast = parse_code("""
    namespace hlsl {
    template <bool B, typename T> struct enable_if {};
    template <typename T> struct enable_if<true, T> {
        using type = T;
    };
    template <typename T> struct is_arithmetic {
        static const bool value = false;
    };
    }

    template <typename T>
    struct Wrapper {
        static const bool IsArithmetic = hlsl::is_arithmetic<T>::value;

        template <typename U = T>
        typename hlsl::enable_if<IsArithmetic, U>::type Get() {
            return val;
        }

        T val;
    };
    """)

    wrapper = next(struct for struct in ast.structs if struct.name == "Wrapper")

    assert wrapper.methods[0].return_type == "hlsl::enable_if<IsArithmetic, U>::type"
    assert wrapper.methods[0].name == "Get"
    assert wrapper.members[-1].name == "val"


def test_parse_dependent_scoped_template_call_from_dxc_udt_validation():
    # Source: microsoft/DirectXShaderCompiler@d6e0ca4a0c25b13ed676c8ba16839c3eb9fcc652
    # tools/clang/test/HLSLFileCheck/hlsl/template/4771-udt-parameter-validation.hlsl
    ast = parse_code("""
    template<typename T>
    struct Leg {
        static T zero() {
            return (T)0;
        }
    };

    template<class Animal>
    typename Animal::LegType getLegs(Animal A) {
      return A.Legs + Leg<typename Animal::LegType>::zero();
    }

    struct Pup {
      using LegType = int;
      LegType Legs;
    };
    """)

    function = ast.functions[0]
    returned = function.body[0].value

    assert function.return_type == "Animal::LegType"
    assert returned.right.name == "Leg<typename Animal::LegType>::zero"


def test_parse_variable_template_id_expressions_from_dxc_var_template():
    # Source: microsoft/DirectXShaderCompiler@main
    # tools/clang/test/CodeGenSPIRV/var.template.hlsl
    ast = parse_code("""
    template <class, class>
    static const bool is_same_v = false;

    template <class T>
    static const bool is_same_v<T, T> = true;

    RWStructuredBuffer<bool> outs;

    void main() {
      outs[0] = is_same_v<int, bool>;
      outs[1] = is_same_v<int, int>;
    }
    """)

    assert ast.global_variables[-1].name == "outs"
    assert ast.functions[0].body[0].right == "is_same_v<int, bool>"
    assert ast.functions[0].body[1].right == "is_same_v<int, int>"


def test_parse_unity_shaderlab_program_blocks_import_embedded_hlsl():
    ast = parse_code("""
    Shader "Custom/ExtractedProgram"
    {
        Properties
        {
            _Color("Color", Color) = (1, 1, 1, 1)
        }

        SubShader
        {
            HLSLINCLUDE
            struct appdata {
                float4 vertex : POSITION;
            };

            struct v2f {
                float4 position : SV_POSITION;
            };

            float4 ApplyTint(float4 color) {
                return color;
            }
            ENDHLSL

            Pass
            {
                HLSLPROGRAM
                #pragma vertex vert
                #pragma fragment frag

                v2f vert(appdata v) {
                    v2f o;
                    o.position = v.vertex;
                    return o;
                }

                float4 frag(v2f i) : SV_Target {
                    return ApplyTint(float4(1, 1, 1, 1));
                }
                ENDHLSL
            }
        }
    }
    """)

    struct_names = [node.name for node in ast.structs if isinstance(node, StructNode)]
    pragmas = [node for node in ast.structs if isinstance(node, PragmaNode)]

    assert struct_names == ["appdata", "v2f"]
    assert [(pragma.directive, pragma.value) for pragma in pragmas] == [
        ("vertex", "vert"),
        ("fragment", "frag"),
    ]
    assert [function.name for function in ast.functions] == [
        "ApplyTint",
        "vert",
        "frag",
    ]


def test_parse_unity_shaderlab_without_hlsl_program_blocks_skips_non_hlsl_content():
    # Reduced from Unity-Built-in-Shaders:
    # DefaultResourcesExtra/Mobile/Mobile-Particle-Add.shader
    ast = parse_code("""
    Shader "Mobile/Particles/Additive" {
        Properties {
            _MainTex ("Particle Texture", 2D) = "white" {}
        }

        SubShader {
            Pass {
                SetTexture [_MainTex] {
                    combine texture * primary
                }
            }
        }
    }
    """)

    assert ast.structs == []
    assert ast.functions == []
    assert ast.global_variables == []


def test_parse_unity_unexpanded_struct_member_macros_from_builtin_shaders():
    ast = parse_code("""
    struct SpeedTreeVB {
        float4 vertex : POSITION;
        UNITY_VERTEX_INPUT_INSTANCE_ID
    };

    struct TerrainInput {
        float4 tc;
        UNITY_FOG_COORDS(0)
    };
    """)

    structs = {node.name: node for node in ast.structs if isinstance(node, StructNode)}

    assert [member.name for member in structs["SpeedTreeVB"].members] == ["vertex"]
    assert [member.name for member in structs["TerrainInput"].members] == ["tc"]


def test_parse_compact_shaderlab_program_markers_import_embedded_hlsl():
    ast = parse_code("""
    Shader "Custom/CompactProgram" { SubShader { Pass { CGPROGRAM
    #pragma vertex vert
    #pragma fragment frag
    struct v2f { float4 position : SV_POSITION; };
    v2f vert(float4 position : POSITION) { v2f o; o.position = position; return o; }
    float4 frag(v2f i) : SV_Target { return float4(1, 1, 1, 1); } ENDCG } } }
    """)

    pragmas = [node for node in ast.structs if isinstance(node, PragmaNode)]

    assert [(pragma.directive, pragma.value) for pragma in pragmas] == [
        ("vertex", "vert"),
        ("fragment", "frag"),
    ]
    assert [function.name for function in ast.functions] == ["vert", "frag"]


def test_skip_top_level_raw_text_from_public_raytracing_samples():
    ast = parse_code("""
    ToDo fix or remove
    Texture2D<float> g_inValue : register(t0);

    On change, update triangle/vertex definitiions.
    static const float GRASS_X[3][7] = {
        {-0.329877, 0.329877, -0.212571, 0.212571, -0.173286, 0.173286, 0.000000 }
    };
    """)

    names = [variable.name for variable in ast.global_variables]
    assert names == ["g_inValue", "GRASS_X"]


def test_parse_cbuffer_preserves_buffer_and_member_bindings():
    code = """
    cbuffer FrameData : register(b0, space1) {
        row_major float4x4 viewProj : packoffset(c0);
        float4 tint : packoffset(c4);
    };
    """

    ast = parse_code(code)
    cbuffer = ast.cbuffers[0]

    assert cbuffer.register == "b0, space1"
    assert cbuffer.packoffset is None
    assert cbuffer.members[0].name == "viewProj"
    assert cbuffer.members[0].packoffset == "c0"
    assert cbuffer.members[0].register is None
    assert cbuffer.members[1].name == "tint"
    assert cbuffer.members[1].packoffset == "c4"
    assert cbuffer.members[1].register is None


def test_parse_cbuffer_effect_annotation_before_body_from_fx_style_sample():
    ast = parse_code("""
    cbuffer FrameData : register(b0, space1)
    <
        string UIName = "Frame";
        bool Visible = true;
    >
    {
        row_major float4x4 viewProj : packoffset(c0);
        precise float exposure : packoffset(c4.x);
    };

    tbuffer LookupData : register(t3, space2)
    <
        string UIName = "Lookup";
    >
    {
        float4 values[4] : packoffset(c0);
    };
    """)

    frame, lookup = ast.cbuffers

    assert frame.name == "FrameData"
    assert frame.register == "b0, space1"
    assert [member.name for member in frame.members] == ["viewProj", "exposure"]
    assert frame.members[0].packoffset == "c0"
    assert frame.members[1].qualifiers == ["precise"]
    assert frame.members[1].packoffset == "c4.x"
    assert lookup.name == "LookupData"
    assert lookup.is_tbuffer is True
    assert lookup.register == "t3, space2"
    assert lookup.members[0].array_sizes == [4]
    assert lookup.members[0].packoffset == "c0"


def test_parse_anonymous_old_style_cbuffer_uses_synthetic_name():
    code = """
    cbuffer : register(b1)
    {
        float4 a;
        int2 b;
    };
    """

    ast = parse_code(code)
    cbuffer = ast.cbuffers[0]

    assert cbuffer.name == "AnonymousCBuffer_b1"
    assert cbuffer.register == "b1"
    assert [member.name for member in cbuffer.members] == ["a", "b"]
    assert [member.vtype for member in cbuffer.members] == ["float4", "int2"]


def test_parse_nested_cbuffer_members_from_dxc_packreg_sample():
    ast = parse_code("""
    cbuffer OuterBuffer {
        float OuterItem0;
        cbuffer InnerBuffer {
            float InnerItem0;
        };
        float OuterItem1;
    };
    """)

    outer = ast.cbuffers[0]
    nested = outer.members[1]

    assert outer.name == "OuterBuffer"
    assert [member.name for member in outer.members] == [
        "OuterItem0",
        "InnerBuffer",
        "OuterItem1",
    ]
    assert nested.is_cbuffer is True
    assert nested.name == "InnerBuffer"
    assert [member.name for member in nested.members] == ["InnerItem0"]


def test_parse_tbuffer_preserves_texture_buffer_metadata():
    ast = parse_code("""
    tbuffer LookupData : register(t3, space2) {
        float4 values[4] : packoffset(c0);
        float scale : packoffset(c4.x);
    };
    """)

    tbuffer = ast.cbuffers[0]

    assert tbuffer.name == "LookupData"
    assert tbuffer.buffer_kind == "tbuffer"
    assert tbuffer.is_tbuffer is True
    assert tbuffer.register == "t3, space2"
    assert [member.name for member in tbuffer.members] == ["values", "scale"]
    assert tbuffer.members[0].array_sizes == [4]
    assert tbuffer.members[0].packoffset == "c0"
    assert tbuffer.members[1].packoffset == "c4.x"


def test_parse_cbuffer_and_tbuffer_methods_from_dxc_spirv():
    # Source: microsoft/DirectXShaderCompiler
    # tools/clang/test/CodeGenSPIRV/fn.ctbuffer.hlsl
    ast = parse_code("""
    cbuffer MyCBuffer {
        float4 cb_val;

        float4 get_cb_val() { return cb_val; }
    }

    struct S {
        float3 s_val;

        float3 get_s_val() { return s_val; }
    };

    tbuffer MyTBuffer {
        float tb_val;
        S tb_s;

        float get_tb_val() { return tb_val; }
    }

    float4 main() : SV_Target {
        return get_cb_val() + float4(tb_s.get_s_val(), 0.0) * get_tb_val();
    }
    """)

    cbuffer, tbuffer = ast.cbuffers

    assert [member.name for member in cbuffer.members] == ["cb_val"]
    assert [method.name for method in cbuffer.methods] == ["get_cb_val"]
    assert cbuffer.methods[0].return_type == "float4"
    assert len(cbuffer.methods[0].body) == 1

    assert [member.name for member in tbuffer.members] == ["tb_val", "tb_s"]
    assert [method.name for method in tbuffer.methods] == ["get_tb_val"]
    assert tbuffer.methods[0].return_type == "float"
    assert len(tbuffer.methods[0].body) == 1


def test_parse_object_style_buffer_resource_templates():
    ast = parse_code("""
    struct FrameConstants {
        float4 tint;
    };
    ConstantBuffer<FrameConstants> frame : register(b2, space1);
    TextureBuffer<FrameConstants> lookup : register(t5, space3);
    """)

    globals_by_name = {node.name: node for node in ast.global_variables}

    assert globals_by_name["frame"].vtype == "ConstantBuffer<FrameConstants>"
    assert globals_by_name["frame"].register == "b2, space1"
    assert globals_by_name["lookup"].vtype == "TextureBuffer<FrameConstants>"
    assert globals_by_name["lookup"].register == "t5, space3"


def test_parse_cxx11_namespaced_attribute_on_cbuffer():
    ast = parse_code("""
    [[vk::push_constant]]
    cbuffer PushConstants : register(b0, space1) {
        float4 tint : packoffset(c0);
    };
    """)

    cbuffer = ast.cbuffers[0]

    assert cbuffer.name == "PushConstants"
    assert cbuffer.register == "b0, space1"
    assert cbuffer.members[0].name == "tint"
    assert cbuffer.members[0].packoffset == "c0"
    assert len(cbuffer.attributes) == 1
    assert cbuffer.attributes[0].name == "vk::push_constant"
    assert cbuffer.attributes[0].args == []


def test_parse_geometry_shader():
    code = """
    struct GSInput {
        float4 pos : SV_Position;
    };
    struct GSOutput {
        float4 pos : SV_Position;
    };

    [maxvertexcount(3)]
    void GSMain(triangle GSInput input[3], inout TriangleStream<GSOutput> triStream) {
        GSOutput outVert;
        outVert.pos = input[0].pos;
        triStream.Append(outVert);
        triStream.RestartStrip();
    }
    """
    assert_parses(code)


def test_parse_tessellation_shaders():
    code = """
    struct HSInput {
        float4 pos : SV_Position;
    };
    struct HSOutput {
        float4 pos : SV_Position;
    };

    struct HSConstData {
        float edges[3] : SV_TessFactor;
        float inside : SV_InsideTessFactor;
    };

    [domain(\"tri\")]
    [partitioning(\"fractional_even\")]
    [outputtopology(\"triangle_cw\")]
    [outputcontrolpoints(3)]
    [patchconstantfunc(\"HSConst\")]
    HSOutput HSMain(InputPatch<HSInput, 3> patch, uint id : SV_OutputControlPointID) {
        HSOutput output;
        output.pos = patch[id].pos;
        return output;
    }

    [domain(\"tri\")]
    float4 DSMain(HSConstData data, const OutputPatch<HSOutput, 3> patch, float3 uvw : SV_DomainLocation) : SV_Position {
        return patch[0].pos;
    }
    """
    assert_parses(code)


def test_parse_mesh_task_shaders():
    code = """
    [shader(\"amplification\")]
    void ASMain() {
        DispatchMesh(1, 1, 1);
    }

    [shader(\"mesh\")]
    void MSMain() {
        SetMeshOutputCounts(1, 1);
    }
    """
    assert_parses(code)


def test_parse_pascal_case_mesh_attributes_infer_mesh_stage():
    code = """
    struct VertexOut {
        float4 position : SV_Position;
    };

    [NumThreads(128, 1, 1)]
    [OutputTopology(\"triangle\")]
    void main(
        out indices uint3 tris[1],
        out vertices VertexOut verts[1]
    ) {
        SetMeshOutputCounts(1, 1);
    }
    """

    ast = parse_code(code)
    function = ast.functions[0]

    assert function.name == "main"
    assert function.qualifier == "mesh"
    assert [attribute.name for attribute in function.attributes] == [
        "NumThreads",
        "OutputTopology",
    ]
    assert [attribute.name for attribute in function.params[0].attributes] == [
        "indices"
    ]
    assert [attribute.name for attribute in function.params[1].attributes] == [
        "vertices"
    ]


def test_parse_raytracing_shader():
    code = """
    RaytracingAccelerationStructure accel : register(t0, space1);

    [shader(\"raygeneration\")]
    void RayGen() {
        TraceRay(accel, 0, 0xFF, 0, 1, 0, float3(0.0, 0.0, 0.0), 0.0, float3(0.0, 0.0, 1.0), 100.0, 0);
    }
    """
    assert_parses(code)


def test_parse_raytracing_shader_stages():
    code = """
    [shader(\"intersection\")]
    void IsMain() { }
    [shader(\"closesthit\")]
    void ChMain() { }
    [shader(\"anyhit\")]
    void AhMain() { }
    [shader(\"miss\")]
    void MsMain() { }
    [shader(\"callable\")]
    void ClMain() { }
    """
    assert_parses(code)


def test_parse_raytracing_payload_parameter_named_payload():
    ast = parse_code("""
    struct Payload {
        float3 hitValue;
        bool shadowed;
    };

    struct Attributes {
        float2 bary;
    };

    [shader(\"closesthit\")]
    void ClosestHit(inout Payload payload, in Attributes attribs) {
        payload.shadowed = true;
    }

    [shader(\"miss\")]
    void Miss(inout Payload payload) {
        payload.shadowed = false;
    }
    """)

    closest_hit = ast.functions[0]
    miss = ast.functions[1]

    assert closest_hit.qualifier == "ray_closest_hit"
    assert closest_hit.params[0].vtype == "Payload"
    assert closest_hit.params[0].name == "payload"
    assert closest_hit.params[0].qualifiers == ["inout"]
    assert closest_hit.params[0].attributes == []
    assert closest_hit.params[1].vtype == "Attributes"
    assert closest_hit.params[1].name == "attribs"
    assert closest_hit.params[1].qualifiers == ["in"]
    assert miss.qualifier == "ray_miss"
    assert miss.params[0].vtype == "Payload"
    assert miss.params[0].name == "payload"
    assert miss.params[0].qualifiers == ["inout"]
    assert miss.params[0].attributes == []


def test_parse_additional_attributes():
    code = """
    [earlydepthstencil]
    float4 PSMain(float4 pos : SV_Position) : SV_Target0 { return pos; }

    [unroll]
    void CSMain() { }

    [branch]
    float4 BRMain(float4 pos : SV_Position) : SV_Target0 { return pos; }

    [loop]
    void LoopMain() { }

    [flatten]
    float4 FLMain(float4 pos : SV_Position) : SV_Target0 { return pos; }

    [maxtessfactor(16)]
    [instance(2)]
    [fastopt]
    [allow_uav_condition]
    void AttrMain() { }
    """
    assert_parses(code)


def test_parse_cxx11_namespaced_attribute_on_top_level_resource_declaration():
    ast = parse_code("""
    [[vk::binding(3, 1)]]
    Texture2D<float4> texture2 : register(t0, space0);
    """)

    resource = ast.global_variables[0]

    assert resource.name == "texture2"
    assert resource.vtype == "Texture2D<float4>"
    assert resource.register == "t0, space0"
    assert len(resource.attributes) == 1
    assert resource.attributes[0].name == "vk::binding"
    assert resource.attributes[0].args == [3, 1]


def test_parse_wave_intrinsics():
    code = """
    uint WaveMain(uint value) {
        bool predicate = value != 0u;
        uint lane = WaveGetLaneIndex();
        uint laneCount = WaveGetLaneCount();
        bool first = WaveIsFirstLane();
        uint sum = WaveActiveSum(value);
        uint product = WaveActiveProduct(value);
        uint andValue = WaveActiveBitAnd(value);
        uint orValue = WaveActiveBitOr(value);
        uint xorValue = WaveActiveBitXor(value);
        bool allTrue = WaveActiveAllTrue(predicate);
        bool anyTrue = WaveActiveAnyTrue(predicate);
        uint4 ballot = WaveActiveBallot(predicate);
        uint laneValue = WaveReadLaneAt(value, 0u);
        uint firstValue = WaveReadLaneFirst(value);
        uint prefixSum = WavePrefixSum(value);
        uint prefixProduct = WavePrefixProduct(value);
        uint4 matchMask = WaveMatch(value);
        uint multiSum = WaveMultiPrefixSum(value, ballot);
        uint quadX = QuadReadAcrossX(value);
        uint quadY = QuadReadAcrossY(value);
        uint quadDiagonal = QuadReadAcrossDiagonal(value);
        uint quadLane = QuadReadLaneAt(value, 2u);
        return lane + laneCount + sum + product + andValue + orValue + xorValue
            + laneValue + firstValue + prefixSum + prefixProduct + matchMask.x
            + multiSum + quadX + quadY + quadDiagonal + quadLane + ballot.x
            + (first ? 1u : 0u) + (allTrue ? 1u : 0u) + (anyTrue ? 1u : 0u);
    }
    """
    assert_parses(code)


def test_parse_texture_methods():
    code = """
    Texture2D tex : register(t0);
    SamplerState samp : register(s0);
    float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
        float4 a = tex.Sample(samp, uv);
        float4 b = tex.SampleLevel(samp, uv, 0.0);
        float4 c = tex.SampleGrad(samp, uv, float2(1.0, 0.0), float2(0.0, 1.0));
        float4 d = tex.SampleBias(samp, uv, 0.5);
        float4 e = tex.SampleCmpLevelZero(samp, uv, 0.5);
        float4 f = tex.GatherRed(samp, uv);
        return a + b + c + d + e + f;
    }
    """
    assert_parses(code)


def test_parse_texture_sample_offset_methods_keep_member_calls():
    code = """
    Texture2D tex : register(t0);
    SamplerState samp : register(s0);

    float4 main(
        float2 uv : TEXCOORD0,
        float lod : TEXCOORD1,
        float2 ddx : TEXCOORD2,
        float2 ddy : TEXCOORD3,
        int2 offset : TEXCOORD4
    ) : SV_Target0 {
        float4 plain = tex.Sample(samp, uv, offset);
        float4 mip = tex.SampleLevel(samp, uv, lod, offset);
        float4 grad = tex.SampleGrad(samp, uv, ddx, ddy, offset);
        return plain + mip + grad;
    }
    """

    ast = parse_code(code)
    nodes = list(iter_ast_nodes(ast))

    assert not [node for node in nodes if isinstance(node, TextureSampleNode)]
    members = [
        node.name.member
        for node in nodes
        if isinstance(node, FunctionCallNode)
        and isinstance(node.name, MemberAccessNode)
    ]
    assert {"Sample", "SampleLevel", "SampleGrad"}.issubset(set(members))


def test_parse_texture_compare_and_gather_offset_methods_keep_member_calls():
    code = """
    Texture2D colorMap : register(t0);
    Texture2D<float> shadowMap : register(t1);
    SamplerState linearSampler : register(s0);
    SamplerComparisonState compareSampler : register(s1);

    float4 main(
        float2 uv : TEXCOORD0,
        float depth : TEXCOORD1,
        int2 offset : TEXCOORD2
    ) : SV_Target0 {
        float cmp = shadowMap.SampleCmp(compareSampler, uv, depth, offset);
        float cmpZero = shadowMap.SampleCmpLevelZero(
            compareSampler, uv, depth, offset
        );
        float4 gather = colorMap.GatherRed(linearSampler, uv, offset);
        float4 gatherAny = colorMap.Gather(linearSampler, uv, offset);
        float4 gatherCmp = shadowMap.GatherCmp(
            compareSampler, uv, depth, offset
        );
        float4 gatherCmpRed = shadowMap.GatherCmpRed(
            compareSampler, uv, depth, offset
        );
        return gather + gatherAny + gatherCmp + gatherCmpRed + float4(cmp + cmpZero);
    }
    """

    ast = parse_code(code)
    nodes = list(iter_ast_nodes(ast))

    members = [
        node.name.member
        for node in nodes
        if isinstance(node, FunctionCallNode)
        and isinstance(node.name, MemberAccessNode)
    ]
    assert {
        "SampleCmp",
        "SampleCmpLevelZero",
        "GatherRed",
        "Gather",
        "GatherCmp",
        "GatherCmpRed",
    }.issubset(set(members))


def test_parse_texture_status_and_clamp_overloads_keep_member_calls():
    code = """
    Texture2D colorMap : register(t0);
    Texture2D<float> shadowMap : register(t1);
    SamplerState linearSampler : register(s0);
    SamplerComparisonState compareSampler : register(s1);

    float4 main(
        float2 uv : TEXCOORD0,
        float depth : TEXCOORD1,
        float lod : TEXCOORD2,
        float bias : TEXCOORD3,
        float2 ddx : TEXCOORD4,
        float2 ddy : TEXCOORD5,
        int2 offset : TEXCOORD6,
        uint status : TEXCOORD7
    ) : SV_Target0 {
        float4 plain = colorMap.Sample(
            linearSampler, uv, offset, 0.0, status
        );
        float4 biased = colorMap.SampleBias(
            linearSampler, uv, bias, offset, 0.0, status
        );
        float4 mip = colorMap.SampleLevel(
            linearSampler, uv, lod, offset, status
        );
        float4 grad = colorMap.SampleGrad(
            linearSampler, uv, ddx, ddy, offset, 0.0, status
        );
        float cmp = shadowMap.SampleCmp(
            compareSampler, uv, depth, offset, 0.0, status
        );
        float cmpZero = shadowMap.SampleCmpLevelZero(
            compareSampler, uv, depth, offset, status
        );
        float4 gather = colorMap.Gather(linearSampler, uv, offset, status);
        float4 gatherRed = colorMap.GatherRed(linearSampler, uv, offset, status);
        float4 gatherOffsets = colorMap.GatherRed(
            linearSampler, uv, offset, offset, offset, offset, status
        );
        float4 gatherCmp = shadowMap.GatherCmp(
            compareSampler, uv, depth, offset, status
        );
        float4 gatherCmpGreen = shadowMap.GatherCmpGreen(
            compareSampler, uv, depth, offset
        );
        return (
            plain + biased + mip + grad + gather + gatherRed + gatherOffsets
            + gatherCmp + gatherCmpGreen + float4(cmp + cmpZero)
        );
    }
    """

    ast = parse_code(code)
    nodes = list(iter_ast_nodes(ast))

    assert not [node for node in nodes if isinstance(node, TextureSampleNode)]
    members = [
        node.name.member
        for node in nodes
        if isinstance(node, FunctionCallNode)
        and isinstance(node.name, MemberAccessNode)
    ]
    assert {
        "Sample",
        "SampleBias",
        "SampleLevel",
        "SampleGrad",
        "SampleCmp",
        "SampleCmpLevelZero",
        "Gather",
        "GatherRed",
        "GatherCmp",
        "GatherCmpGreen",
    }.issubset(set(members))


def test_parse_tiled_resource_status_loads_and_checks_keep_calls():
    code = """
    Texture2D colorMap : register(t0);
    Texture2DMS<float4> msMap : register(t1);
    RWTexture2D<float4> outputImage : register(u0);

    float4 main(
        int2 pixel : TEXCOORD0,
        int sampleIndex : TEXCOORD1,
        int2 offset : TEXCOORD2
    ) : SV_Target0 {
        uint status = 0;
        float4 fetched = colorMap.Load(int3(pixel, 0), offset, status);
        float4 stored = outputImage.Load(pixel, status);
        float4 ms = msMap.Load(pixel, sampleIndex, offset, status);
        bool mapped = CheckAccessFullyMapped(status);
        return mapped ? fetched + stored + ms : float4(0.0, 0.0, 0.0, 0.0);
    }
    """

    ast = parse_code(code)
    nodes = list(iter_ast_nodes(ast))

    member_calls = [
        node.name.member
        for node in nodes
        if isinstance(node, FunctionCallNode)
        and isinstance(node.name, MemberAccessNode)
    ]
    free_calls = [
        node.name
        for node in nodes
        if isinstance(node, FunctionCallNode) and isinstance(node.name, str)
    ]
    assert member_calls.count("Load") == 3
    assert "CheckAccessFullyMapped" in free_calls


def test_parse_get_dimensions_overloads_keep_member_calls():
    code = """
    Texture2D<float4> colorMap : register(t0);
    Texture2DArray<float4> layerMap : register(t1);
    Texture2DMS<float4> msMap : register(t2);
    RWTexture3D<float4> volume : register(u0);
    StructuredBuffer<float4> structs : register(t3);

    void main(uint lod : TEXCOORD0) {
        uint width;
        uint height;
        uint depth;
        uint elements;
        uint levels;
        uint samples;
        uint count;
        uint stride;
        colorMap.GetDimensions(width, height, levels);
        layerMap.GetDimensions(lod, width, height, elements, levels);
        msMap.GetDimensions(width, height, samples);
        volume.GetDimensions(width, height, depth);
        structs.GetDimensions(count, stride);
    }
    """

    ast = parse_code(code)
    nodes = list(iter_ast_nodes(ast))

    member_calls = [
        node.name.member
        for node in nodes
        if isinstance(node, FunctionCallNode)
        and isinstance(node.name, MemberAccessNode)
    ]
    assert member_calls.count("GetDimensions") == 5


def test_parse_get_dimensions_edge_overloads_keep_member_calls():
    code = """
    Texture1D<float4> lineMap : register(t0);
    Texture1DArray<float4> lineArray : register(t1);
    TextureCube<float4> cubeMap : register(t2);
    TextureCubeArray<float4> cubeArray : register(t3);
    RWTexture1DArray<float4> imageArray : register(u0);
    Texture2DMSArray<float4> msArray : register(t4);
    RWTexture2DMSArray<float4> msImage : register(u1);

    void main(uint lod : TEXCOORD0) {
        uint width;
        uint height;
        uint elements;
        uint levels;
        uint samples;
        lineMap.GetDimensions(width, levels);
        lineArray.GetDimensions(lod, width, elements, levels);
        cubeMap.GetDimensions(width, height, levels);
        cubeArray.GetDimensions(lod, width, height, elements, levels);
        imageArray.GetDimensions(width, elements);
        msArray.GetDimensions(width, height, elements, samples);
        msImage.GetDimensions(width, height, elements, samples);
    }
    """

    ast = parse_code(code)
    nodes = list(iter_ast_nodes(ast))

    member_calls = [
        node.name.member
        for node in nodes
        if isinstance(node, FunctionCallNode)
        and isinstance(node.name, MemberAccessNode)
    ]
    assert member_calls.count("GetDimensions") == 7


def test_parse_texture_lod_query_methods_keep_member_calls():
    code = """
    Texture2D<float4> colorMap : register(t0);
    Texture2DMS<float4> msMap : register(t1);
    SamplerState linearSampler : register(s0);

    float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
        float clamped = colorMap.CalculateLevelOfDetail(linearSampler, uv);
        float unclamped = colorMap.CalculateLevelOfDetailUnclamped(
            linearSampler, uv
        );
        float msLod = msMap.CalculateLevelOfDetail(linearSampler, uv);
        return float4(clamped + unclamped + msLod);
    }
    """

    ast = parse_code(code)
    nodes = list(iter_ast_nodes(ast))

    member_calls = [
        node.name.member
        for node in nodes
        if isinstance(node, FunctionCallNode)
        and isinstance(node.name, MemberAccessNode)
    ]
    assert member_calls.count("CalculateLevelOfDetail") == 2
    assert member_calls.count("CalculateLevelOfDetailUnclamped") == 1


def test_parse_texture_sample_position_query_keeps_member_calls():
    code = """
    Texture2DMS<float4> msMap : register(t0);
    Texture2DMSArray<float4> msArray : register(t1);
    Texture2D<float4> colorMap : register(t2);

    float4 main(uint sampleIndex : SV_SampleIndex) : SV_Target0 {
        float2 pos = msMap.GetSamplePosition(sampleIndex);
        float2 arrayPos = msArray.GetSamplePosition(sampleIndex);
        float2 invalid = colorMap.GetSamplePosition(sampleIndex);
        return float4(pos + arrayPos + invalid, 0.0, 1.0);
    }
    """

    ast = parse_code(code)
    nodes = list(iter_ast_nodes(ast))

    member_calls = [
        node.name.member
        for node in nodes
        if isinstance(node, FunctionCallNode)
        and isinstance(node.name, MemberAccessNode)
    ]
    assert member_calls.count("GetSamplePosition") == 3


def test_parse_resource_method_ast_shapes():
    code = """
    Texture2D tex : register(t0);
    SamplerState samp : register(s0);
    RWStructuredBuffer<int> buffer : register(u0);
    AppendStructuredBuffer<int> appendBuf : register(u1);
    ConsumeStructuredBuffer<int> consumeBuf : register(u2);

    float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
        float4 base = tex.Sample(
            samp,
            uv
        );
        float4 mip = tex.SampleLevel(
            samp,
            uv,
            1.0
        );
        float4 grad = tex.SampleGrad(
            samp,
            uv,
            float2(1.0, 0.0),
            float2(0.0, 1.0)
        );
        float cmp = tex.SampleCmpLevelZero(samp, uv, 0.5);
        float cmpLod = tex.SampleCmpLevel(samp, uv, 0.5, 1.0);
        float cmpGrad = tex.SampleCmpGrad(
            samp,
            uv,
            0.5,
            float2(1.0, 0.0),
            float2(0.0, 1.0)
        );
        float cmpBias = tex.SampleCmpBias(samp, uv, 0.5, 0.25);
        int loaded = buffer.Load(0);
        buffer.Store(
            1,
            loaded
        );
        appendBuf.Append(
            loaded
        );
        int consumed = consumeBuf.Consume();
        return base + mip + grad + float4(cmp + loaded + consumed);
    }
    """
    ast = parse_code(code)
    nodes = list(iter_ast_nodes(ast))

    samples = [node for node in nodes if isinstance(node, TextureSampleNode)]
    assert len(samples) == 2
    assert samples[0].lod is None
    assert samples[1].lod is not None

    members = [
        node.name.member
        for node in nodes
        if isinstance(node, FunctionCallNode)
        and isinstance(node.name, MemberAccessNode)
    ]
    assert "Sample" not in members
    assert "SampleLevel" not in members
    assert {
        "SampleGrad",
        "SampleCmpLevelZero",
        "SampleCmpLevel",
        "SampleCmpGrad",
        "SampleCmpBias",
        "Load",
        "Store",
        "Append",
        "Consume",
    }.issubset(set(members))


def test_parse_typed_resource_method_calls():
    code = """
    RWByteAddressBuffer rawBytes : register(u1);

    void main(uint ix : IX) {
        uint loaded = rawBytes.Load<uint>(ix);
        rawBytes.Store<uint>(ix, loaded);
        bool inRange = ix < 4u;
    }
    """
    ast = parse_code(code)
    nodes = list(iter_ast_nodes(ast))

    members = [
        node.name.member
        for node in nodes
        if isinstance(node, FunctionCallNode)
        and isinstance(node.name, MemberAccessNode)
    ]
    assert "Load<uint>" in members
    assert "Store<uint>" in members


def test_parse_rwbyteaddressbuffer_interlocked_add_from_microsoft_docs():
    code = """
    RWByteAddressBuffer rawBytes : register(u1);

    void main(uint offset : TEXCOORD0) {
        uint original;
        rawBytes.InterlockedAdd(offset, 1u, original);
    }
    """
    ast = parse_code(code)
    nodes = list(iter_ast_nodes(ast))

    members = [
        node.name.member
        for node in nodes
        if isinstance(node, FunctionCallNode)
        and isinstance(node.name, MemberAccessNode)
    ]
    assert "InterlockedAdd" in members


def test_parse_rwbyteaddressbuffer_interlocked_family_from_microsoft_docs():
    code = """
    RWByteAddressBuffer rawBytes : register(u1);

    void main(uint offset : TEXCOORD0) {
        uint original;
        rawBytes.InterlockedMax(offset, 5u, original);
        rawBytes.InterlockedCompareExchange(offset + 4u, 3u, 7u, original);
    }
    """
    ast = parse_code(code)
    nodes = list(iter_ast_nodes(ast))

    members = [
        node.name.member
        for node in nodes
        if isinstance(node, FunctionCallNode)
        and isinstance(node.name, MemberAccessNode)
    ]
    assert "InterlockedMax" in members
    assert "InterlockedCompareExchange" in members


def test_parse_upstream_default_parameter_value():
    ast = parse_code("""
    float4 ToRGBM(float3 rgb, float PeakValue = 255.0 / 16.0) {
        return float4(rgb, PeakValue);
    }
    """)

    func = ast.functions[0]

    assert func.params[1].name == "PeakValue"
    assert func.params[1].value is not None


def test_parse_upstream_comma_separated_declarations():
    ast = parse_code("""
    cbuffer CSConstants : register(b0) {
        uint ViewportWidth, ViewportHeight;
    };

    float main() : SV_Target0 {
        float x = 1.0, y = 2.0;
        return x + y;
    }
    """)

    assert [member.name for member in ast.cbuffers[0].members] == [
        "ViewportWidth",
        "ViewportHeight",
    ]

    main_func = ast.functions[0]
    locals_ = [stmt for stmt in main_func.body if isinstance(stmt, VariableNode)]
    assert [node.name for node in locals_] == ["x", "y"]
    assert [node.value for node in locals_] == [1.0, 2.0]


def test_parse_upstream_statement_attributes_and_for_update_sequences():
    ast = parse_code("""
    void main(uint count) {
        uint tileLightLoadOffset = 0;
        [unroll]
        for (uint n = 0; n < count; n++, tileLightLoadOffset += 4) {
        }
    }
    """)

    loop = next(node for node in iter_ast_nodes(ast) if isinstance(node, ForNode))

    assert [attr.name for attr in loop.attributes] == ["unroll"]
    assert isinstance(loop.update, list)
    assert len(loop.update) == 2


def test_parse_upstream_else_if_chain_with_final_else():
    ast = parse_code("""
    float TestSamples(uint x, uint y) {
        if (y == 0) {
            return 0.5;
        } else if (x == y) {
            return 0.25;
        } else {
            return 0.125;
        }
    }
    """)

    first_if = next(node for node in iter_ast_nodes(ast) if isinstance(node, IfNode))

    assert isinstance(first_if.else_body, IfNode)
    assert first_if.else_body.else_body is not None


def test_parse_upstream_bare_scope_block():
    ast = parse_code("""
    float main(float ao) : SV_Target0 {
        float colorSum = 0.0;
        {
            float ambient = ao;
            colorSum += ambient;
        }
        return colorSum;
    }
    """)

    body_names = [
        node.name for node in ast.functions[0].body if isinstance(node, VariableNode)
    ]
    assert body_names == ["colorSum", "ambient"]


def test_parse_parenthesized_identifier_after_unary_minus_not_cast():
    ast = parse_code("""
    float Sigmoid(float v) {
        return 1.0 / (1.0 + exp(-(v)));
    }
    """)

    assert ast.functions[0].name == "Sigmoid"


def test_parse_function_prototype_declaration():
    ast = parse_code("""
    void TraceRay_OnMiss(inout uint rngState);

    void main() {
        TraceRay_OnMiss(0);
    }
    """)

    assert ast.functions[0].name == "TraceRay_OnMiss"
    assert ast.functions[0].body == []
    assert ast.functions[0].is_prototype is True


def test_parse_relational_condition_before_parenthesized_expression_from_blur_shader():
    ast = parse_code("""
    float4 PSSimpleBlur(float2 uv) : SV_TARGET {
        float3 textureColor = float3(1.0f, 0.0f, 0.0f);
        if (uv.x > (blurXOffset + 0.005f)) {
            for (int i = 1; i < 3; i++) {
                textureColor += textureColor;
            }
        }
        return float4(textureColor, 1.0);
    }
    """)

    condition = next(
        node.condition for node in iter_ast_nodes(ast) if isinstance(node, IfNode)
    )

    assert condition.op == ">"
    assert isinstance(condition.left, MemberAccessNode)


def test_parse_raw_bvh_macro_invocations_inside_switch_from_directx_graphics_samples():
    code = """
    #define GetBVHSize(ID) case ID: size = GetBVHSize(BVH##ID); break

    void main(uint3 DTid : SV_DispatchThreadID) {
        uint size = 0;
        switch (DTid.x + 1) {
            GetBVHSize(1);
            GetBVHSize(2);
        }
    }
    """

    tokens = HLSLLexer(textwrap.dedent(code), preprocess=False).tokenize()
    ast = HLSLParser(tokens).parse()

    switch = next(node for node in iter_ast_nodes(ast) if isinstance(node, SwitchNode))

    assert [call.name for call in switch.default_case] == ["GetBVHSize", "GetBVHSize"]
    assert [call.args for call in switch.default_case] == [[1], [2]]


def test_preprocess_self_referential_bvh_function_macro_from_directx_graphics_samples():
    ast = parse_code("""
    #define GetBVHSize(ID) case ID: size = GetBVHSize(BVH##ID); break

    void main(uint3 DTid : SV_DispatchThreadID) {
        uint size = 0;
        switch (DTid.x + 1) {
            GetBVHSize(1);
            GetBVHSize(2);
        }
    }
    """)

    switch = next(node for node in iter_ast_nodes(ast) if isinstance(node, SwitchNode))

    assert [case.value for case in switch.cases] == [1, 2]
    assert [case.body[0].right.name for case in switch.cases] == [
        "GetBVHSize",
        "GetBVHSize",
    ]
    assert [case.body[0].right.args for case in switch.cases] == [
        ["BVH1"],
        ["BVH2"],
    ]


def test_parse_ray_payload_struct_attributes_from_directx_graphics_samples():
    ast = parse_code("""
    struct [raypayload] RayPayload {
        float4 color : write(caller, closesthit, miss) : read(caller);
        uint iterations : write(caller) : read(closesthit);
    };
    """)

    payload = ast.structs[0]

    assert payload.name == "RayPayload"
    assert [attribute.name for attribute in payload.attributes] == ["raypayload"]
    assert (
        payload.members[0].semantic == "write(caller, closesthit, miss): read(caller)"
    )
    assert payload.members[1].semantic == "write(caller): read(closesthit)"


def test_parse_scoped_enum_parameter_type_from_directx_graphics_samples():
    ast = parse_code("""
    float GetDistanceFromSignedDistancePrimitive(
        in float3 position,
        in SignedDistancePrimitive::Enum sdPrimitive);
    """)

    function = ast.functions[0]

    assert function.is_prototype is True
    assert function.params[1].vtype == "SignedDistancePrimitive::Enum"
    assert function.params[1].name == "sdPrimitive"


def test_parse_scoped_local_variable_type_from_directx_graphics_samples():
    ast = parse_code("""
    struct AabbCB {
        uint primitiveType;
    };

    float ResolvePrimitive(AabbCB cb) {
        AnalyticPrimitive::Enum primitiveType =
            (AnalyticPrimitive::Enum) cb.primitiveType;
        CrossBilateral::BilinearDepthNormal::Parameters params;
        params.Depth.Sigma = 1.0;
        return params.Depth.Sigma + primitiveType;
    }
    """)

    function = ast.functions[0]
    primitive_decl = next(
        stmt for stmt in function.body if getattr(stmt, "name", "") == "primitiveType"
    )
    params_decl = next(
        stmt for stmt in function.body if getattr(stmt, "name", "") == "params"
    )

    assert primitive_decl.vtype == "AnalyticPrimitive::Enum"
    assert primitive_decl.name == "primitiveType"
    assert isinstance(primitive_decl.value, CastNode)
    assert primitive_decl.value.target_type == "AnalyticPrimitive::Enum"
    assert params_decl.vtype == "CrossBilateral::BilinearDepthNormal::Parameters"
    assert params_decl.name == "params"


def test_parse_unsigned_vector_cast_from_directx_graphics_samples():
    ast = parse_code("""
    RWTexture2D<float4> RT : register(u0);

    void raygen_main() {
        RT[(unsigned int2)DispatchRaysIndex()] = float4(1, 1, 1, 1);
    }
    """)

    assignment = ast.functions[0].body[0]
    index = assignment.left.index

    assert isinstance(index, CastNode)
    assert index.target_type == "unsigned int2"
    assert isinstance(index.expression, FunctionCallNode)
    assert index.expression.name == "DispatchRaysIndex"


def test_parse_array_type_casts_from_directx_sdk_samples():
    ast = parse_code("""
    void main(int4 packedIndices, float4 packedWeights) {
        int indexArray[4] = (int[4])packedIndices;
        float blendWeightsArray[4] = (float[4])packedWeights;
    }
    """)

    index_array, blend_weights = ast.functions[0].body

    assert index_array.array_sizes == [4]
    assert isinstance(index_array.value, CastNode)
    assert index_array.value.target_type == "int[4]"
    assert index_array.value.expression == "packedIndices"

    assert blend_weights.array_sizes == [4]
    assert isinstance(blend_weights.value, CastNode)
    assert blend_weights.value.target_type == "float[4]"
    assert blend_weights.value.expression == "packedWeights"


def test_parse_legacy_effect_compile_array_initializer_from_directx_sdk_samples():
    ast = parse_code("""
    VertexShader vsArray20[2] =
    {
        compile vs_2_0 VertSkinning(1),
        compile vs_2_0 VertSkinning(2)
    };
    """)

    shader_array = ast.global_variables[0]

    assert shader_array.vtype == "VertexShader"
    assert shader_array.array_sizes == [2]
    assert isinstance(shader_array.value, InitializerListNode)
    assert [element.name for element in shader_array.value.elements] == [
        "compile",
        "compile",
    ]
    assert shader_array.value.elements[0].args[0] == "vs_2_0"
    assert isinstance(shader_array.value.elements[0].args[1], FunctionCallNode)
    assert shader_array.value.elements[0].args[1].name == "VertSkinning"


def test_parse_inline_struct_array_member_in_cbuffer_from_directx_sdk_samples():
    ast = parse_code("""
    cbuffer cbPerLight : register(b1) {
        struct LightDataStruct {
            matrix m_mLightViewProj;
            float4 m_vLightPos;
        } g_LightData[g_iNumLights] : packoffset(c4);
    };
    """)

    light_struct = ast.structs[0]
    cbuffer = ast.cbuffers[0]
    light_data = cbuffer.members[0]

    assert light_struct.name == "LightDataStruct"
    assert [member.name for member in light_struct.members] == [
        "m_mLightViewProj",
        "m_vLightPos",
    ]
    assert cbuffer.name == "cbPerLight"
    assert light_data.vtype == "LightDataStruct"
    assert light_data.name == "g_LightData"
    assert light_data.array_sizes == ["g_iNumLights"]
    assert light_data.packoffset == "c4"


def test_parse_parenthesized_vector_values_from_directx_graphics_samples():
    ast = parse_code("""
    struct MyPayload {
        float4 color;
    };

    void anyhit_main(inout MyPayload payload) {
        payload.color += (0.1, 0.1, 0.1, 1);
        payload.color = (1, 0, 0, 1);
    }
    """)

    plus_assign, assign = ast.functions[0].body

    assert isinstance(plus_assign.right, InitializerListNode)
    assert plus_assign.right.elements == [0.1, 0.1, 0.1, 1]
    assert isinstance(assign.right, InitializerListNode)
    assert assign.right.elements == [1, 0, 0, 1]


def test_parse_local_struct_declaration_from_directx_graphics_samples():
    ast = parse_code("""
    void ReformTree(in uint groupThreadId) {
        if (groupThreadId != 0) {
            return;
        }

        struct PartitionEntry {
            uint Mask;
            uint NodeIndex;
        };

        uint nodesAllocated = 1;
        PartitionEntry partitionStack[FullTreeletSize];
        partitionStack[0].Mask = 0;
    }
    """)

    function = ast.functions[0]
    local_struct = next(stmt for stmt in function.body if isinstance(stmt, StructNode))
    partition_stack = next(
        stmt for stmt in function.body if getattr(stmt, "name", "") == "partitionStack"
    )

    assert local_struct.name == "PartitionEntry"
    assert [member.name for member in local_struct.members] == ["Mask", "NodeIndex"]
    assert partition_stack.vtype == "PartitionEntry"
    assert partition_stack.array_sizes == ["FullTreeletSize"]


def test_parse_sample_contextual_identifier_from_directx_graphics_samples():
    ast = parse_code("""
    struct SampleValue {
        float3 value;
    };

    StructuredBuffer<SampleValue> g_sampleSets : register(t0);

    float3 GenerateRayDirection(
        uint sampleSetJump,
        uint sampleJump,
        uint numSamplesPerSet,
        float3 u,
        float3 v,
        float3 w
    ) {
        float3 sample =
            g_sampleSets[sampleSetJump + (sampleJump % numSamplesPerSet)].value;
        float3 rayDirection =
            normalize(sample.x * u + sample.y * v + sample.z * w);
        return rayDirection;
    }
    """)

    function = ast.functions[0]
    sample_decl = next(
        stmt for stmt in function.body if getattr(stmt, "name", "") == "sample"
    )

    assert sample_decl.vtype == "float3"
    assert sample_decl.name == "sample"


def test_parse_legacy_special_float_literal_from_directx_graphics_samples():
    ast = parse_code("""
    bool RayAABBIntersectionTest(float3 rayDirection) {
        const float FLT_INFINITY = 1.#INF;
        float3 invRayDirection = rayDirection != 0
            ? 1 / rayDirection
            : float3(FLT_INFINITY, FLT_INFINITY, FLT_INFINITY);
        return true;
    }
    """)

    function = ast.functions[0]
    infinity = next(
        stmt for stmt in function.body if getattr(stmt, "name", "") == "FLT_INFINITY"
    )

    assert math.isinf(infinity.value)


def test_parse_double_suffix_float_literals_from_dxc_intrinsics():
    # Source: microsoft/DirectXShaderCompiler@d6e0ca4a0c25b13ed676c8ba16839c3eb9fcc652
    # tools/clang/test/HLSLFileCheck/hlsl/intrinsics/basic/intrinsic-examples_Mod.hlsl
    # tools/clang/test/HLSLFileCheck/hlsl/intrinsics/compound/pow-mulonly-lit-types.hlsl
    ast = parse_code("""
    double overload1(double d) { return 1.0l; }

    float main(float4x4 b : B) : SV_Target
    {
        return pow(b, -131072.0L)[0][0];
    }
    """)

    assert [function.name for function in ast.functions] == ["overload1", "main"]


def test_parse_unsigned_int_namespace_constants_from_directx_graphics_samples():
    ast = parse_code("""
    namespace FilterKernel
    {
        static const unsigned int Radius = 1;
        static const unsigned int Width = 1 + 2 * Radius;
        static const float Kernel1D[Width] = { 0.27901, 0.44198, 0.27901 };
    }

    Texture2D<float> g_inDepth : register(t1);
    """)

    radius = next(
        variable
        for variable in ast.global_variables
        if getattr(variable, "name", "") == "Radius"
    )
    width = next(
        variable
        for variable in ast.global_variables
        if getattr(variable, "name", "") == "Width"
    )

    assert radius.vtype == "unsigned int"
    assert width.vtype == "unsigned int"
    assert [variable.name for variable in ast.global_variables[-1:]] == ["g_inDepth"]


def test_parse_struct_methods_from_dxc_rewriter_samples():
    code = textwrap.dedent("""
        struct MyTestStruct
        {
            uint4 data[2];

            uint getData1() {
                return uint(data[1].z);
            }

            float3 getDataAsFloat() {
                return float3(asfloat(data[0].x), asfloat(data[0].y), asfloat(data[0].z));
            }
        };
        """)

    ast = parse_code(code)
    struct = ast.structs[0]

    assert isinstance(struct, StructNode)
    assert [member.name for member in struct.members] == ["data"]
    assert [method.name for method in struct.methods] == [
        "getData1",
        "getDataAsFloat",
    ]
    assert all(isinstance(method, FunctionNode) for method in struct.methods)
    assert struct.methods[0].return_type == "uint"
    assert isinstance(struct.methods[0].body[0].value, VectorConstructorNode)


def test_parse_nested_struct_method_prototype_from_wickedengine_sh_lite():
    # Source: https://github.com/turanszkij/WickedEngine/blob/9df7a530aed53cc59b345f751939e513170ddf3c/WickedEngine/shaders/SH_Lite.hlsli
    ast = parse_code("""
        namespace SH
        {
            struct L1
            {
                static const uint NumCoefficients = 4;
                half C[NumCoefficients];

                struct Packed
                {
                    uint C[NumCoefficients / 2];
                    L1 Unpack();
                };
            };

            L1 L1::Packed::Unpack()
            {
                L1 ret;
                return ret;
            }
        }
    """)

    packed = next(struct for struct in ast.structs if struct.name == "Packed")

    assert [member.name for member in packed.members] == ["C"]
    assert [method.name for method in packed.methods] == ["Unpack"]
    assert packed.methods[0].return_type == "L1"
    assert packed.methods[0].is_prototype is True
    assert [function.name for function in ast.functions] == ["L1::Packed::Unpack"]


def test_parse_normalized_rwtexture_element_type_from_wickedengine_rtao():
    # Source: https://github.com/turanszkij/WickedEngine/blob/9df7a530aed53cc59b345f751939e513170ddf3c/WickedEngine/shaders/rtao_denoise_filterCS.hlsl
    ast = parse_code("""
        RWTexture2D<unorm float> output : register(u1);
    """)

    assert ast.global_variables[0].vtype == "RWTexture2D<unorm float>"


def test_parse_reordercoherent_local_resource_from_dxc_dxil_69():
    # Source: microsoft/DirectXShaderCompiler@517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # tools/clang/test/CodeGenDXIL/hlsl/attributes/reordercoherent_uav.hlsl
    ast = parse_code("""
        RWTexture1D<float4> uav1 : register(u3);

        [shader("raygeneration")]
        void main()
        {
          reordercoherent RWTexture1D<float4> uav3 = uav1;
          uav3[0] = float4(5.0, 0.0, 0.0, 1.0);
        }
    """)

    local_resource = ast.functions[0].body[0]

    assert local_resource.vtype == "RWTexture1D<float4>"
    assert local_resource.name == "uav3"
    assert local_resource.qualifiers == ["reordercoherent"]
    assert local_resource.value == "uav1"
    assert isinstance(ast.functions[0].body[1], AssignmentNode)


def test_parse_globallycoherent_uav_parameter_from_hlsl_specs():
    # Source: Microsoft HLSL Specifications proposal 0021, vk cooperative matrix.
    # It declares CoherentStore(globallycoherent RWStructuredBuffer<Type> data, ...).
    ast = parse_code("""
        struct Payload {
            uint value;
        };

        void CoherentStore(
            globallycoherent RWStructuredBuffer<Payload> data,
            uint index
        ) {
            data[index].value = 1u;
        }
    """)

    param = ast.functions[0].params[0]

    assert param.vtype == "RWStructuredBuffer<Payload>"
    assert param.name == "data"
    assert param.qualifiers == ["globallycoherent"]


def test_parse_struct_operator_methods_from_dxc_intrinsics_tests():
    code = textwrap.dedent("""
        struct Vector {
            float2 v;

            Vector operator+(Vector vec) {
                Vector ret;
                ret.v = v + vec.v;
                return ret;
            }
        };
        """)

    ast = parse_code(code)
    struct = ast.structs[0]

    assert [member.name for member in struct.members] == ["v"]
    assert [method.name for method in struct.methods] == ["operator+"]
    assert struct.methods[0].return_type == "Vector"


def test_parse_struct_constructors_and_destructors_from_dxc_style_helpers():
    ast = parse_code("""
        struct MaterialSample {
            float roughness;
            float3 normal;

            MaterialSample(float value, float3 n) : roughness(value), normal(n) {}

            ~MaterialSample() {
                roughness = 0.0;
            }
        };
    """)

    struct = ast.structs[0]
    constructor, destructor = struct.methods

    assert [member.name for member in struct.members] == ["roughness", "normal"]
    assert constructor.name == "MaterialSample"
    assert constructor.return_type == ""
    assert constructor.is_constructor is True
    assert [param.name for param in constructor.params] == ["value", "n"]
    assert constructor.body == []
    assert destructor.name == "~MaterialSample"
    assert destructor.is_destructor is True
    assert isinstance(destructor.body[0], AssignmentNode)


def test_parse_struct_bitfield_members_from_dxc_swizzle_fixture():
    ast = parse_code("""
        struct MyStruct
        {
            uint v0: 5;
            uint v1: 15;
            uint v2: 12;
        };
    """)

    struct = ast.structs[0]

    assert [member.name for member in struct.members] == ["v0", "v1", "v2"]
    assert [member.vtype for member in struct.members] == ["uint", "uint", "uint"]
    assert [member.bit_width for member in struct.members] == [5, 15, 12]
    assert [member.semantic for member in struct.members] == [None, None, None]


def test_parse_anonymous_and_packed_struct_bitfields_from_dxc_debug_fixture():
    # Source: microsoft/DirectXShaderCompiler@8ed708842c1ccb24bd914eff03125c837a01be71
    # tools/clang/test/HLSLFileCheck/hlsl/types/struct/bitfields.hlsl
    ast = parse_code("""
        struct foo {
          int x : 8;
          int : 8;
          int y : 16;
        };

        struct P1 {
          uint l_Packed;
          uint k_Packed : 6,
            i_Packed : 15,
            j_Packed : 11;
        };
    """)

    foo, packed = ast.structs

    assert [member.name for member in foo.members] == ["x", "y"]
    assert [member.bit_width for member in foo.members] == [8, 16]
    assert [member.name for member in packed.members] == [
        "l_Packed",
        "k_Packed",
        "i_Packed",
        "j_Packed",
    ]
    assert [member.bit_width for member in packed.members] == [None, 6, 15, 11]


def test_parse_min_precision_scalar_constructor_from_dxc_tests():
    ast = parse_code("""
        int main(min16int a : A) : SV_Target {
            min16int q = a + 2;
            if (q == min16int(7)) {
                return q - 3;
            }
            return 1;
        }
    """)

    main = ast.functions[0]
    branch = next(stmt for stmt in main.body if isinstance(stmt, IfNode))

    assert branch.condition.op == "=="
    assert isinstance(branch.condition.right, VectorConstructorNode)
    assert branch.condition.right.type_name == "min16int"


def test_parse_fixed_width_vector_alias_constructors_from_hlsl_docs():
    # Sources:
    # https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-scalar
    # https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-vector
    ast = parse_code("""
        float16_t4 MakeHalf(float16_t value : TEXCOORD0) : SV_Target0 {
            float16_t4 halfColor = float16_t4(value, value, value, value);
            int32_t4 signedLanes = int32_t4(1, 2, 3, 4);
            uint32_t2 unsignedPair = uint32_t2(1u, 2u);
            return halfColor + float16_t4(
                signedLanes.x, signedLanes.y, unsignedPair.x, unsignedPair.y
            );
        }
    """)

    half_color, signed_lanes, unsigned_pair, return_stmt = ast.functions[0].body

    assert half_color.vtype == "float16_t4"
    assert isinstance(half_color.value, VectorConstructorNode)
    assert half_color.value.type_name == "float16_t4"
    assert signed_lanes.vtype == "int32_t4"
    assert isinstance(signed_lanes.value, VectorConstructorNode)
    assert signed_lanes.value.type_name == "int32_t4"
    assert unsigned_pair.vtype == "uint32_t2"
    assert isinstance(unsigned_pair.value, VectorConstructorNode)
    assert unsigned_pair.value.type_name == "uint32_t2"
    assert isinstance(return_stmt.value.right, VectorConstructorNode)
    assert return_stmt.value.right.type_name == "float16_t4"


def test_parse_array_template_argument_from_dxc_buffer_tests():
    ast = parse_code("""
        StructuredBuffer<float[2]> ArrBuf : register(t3);
        float2 main(uint ix0 : IX0) : SV_Target {
            return (float2)ArrBuf.Load(ix0 + 1);
        }
    """)

    arr_buf = ast.global_variables[0]

    assert arr_buf.vtype == "StructuredBuffer<float[2]>"


def test_parse_top_level_anonymous_struct_variable_from_dxc_rewriter_samples():
    code = textwrap.dedent("""
        SamplerState ss;

        static const struct {
            float a;
            SamplerState s;
        } A = {1.2, ss};

        float4 main() : SV_Target {
            return A.a;
        }
        """)

    ast = parse_code(code)
    anonymous = ast.structs[0]

    assert anonymous.name == "AnonymousStruct_A"
    assert [member.name for member in anonymous.members] == ["a", "s"]
    assert anonymous.variables == ["A"]
    assert anonymous.variable_declarations[0].name == "A"
    assert isinstance(anonymous.variable_declarations[0].value, InitializerListNode)


def test_parse_anonymous_struct_array_typedef_from_dxc_codegen_debug_tests():
    # Source: microsoft/DirectXShaderCompiler
    # tools/clang/test/CodeGenHLSL/debug/locals/array_of_structs_nested_noopt.hlsl
    ast = parse_code("""
        typedef struct { int a[4]; float2 b[2]; } type[3];

        int main() : OUT {
            type var = (type)0;
            return var[0].a[0];
        }
        """)

    typedef = ast.typedefs[0]
    anonymous = ast.structs[0]
    local_var = ast.functions[0].body[0]

    assert typedef.name == "type"
    assert typedef.alias_type == "AnonymousStruct_type"
    assert typedef.array_sizes == [3]
    assert anonymous.name == "AnonymousStruct_type"
    assert [member.name for member in anonymous.members] == ["a", "b"]
    assert anonymous.members[0].array_sizes == [4]
    assert anonymous.members[1].array_sizes == [2]
    assert local_var.vtype == "type"
    assert isinstance(local_var.value, CastNode)
    assert local_var.value.target_type == "type"


def test_hlsl_define_skips_cpp_compatibility_branch_from_directx_samples():
    code = textwrap.dedent("""
        #ifndef HLSL
        struct float2 { float x, y; };
        #endif

        float main() {
            return 0.0;
        }
        """)

    ast = parse_code(code)

    assert [function.name for function in ast.functions] == ["main"]
    assert ast.structs == []


def test_parse_linalg_post_type_attributes_from_dxc():
    # Source: microsoft/DirectXShaderCompiler@517dd5eb5d8cbb46c15fc1230acac1d2f4779092
    # tools/clang/test/CodeGenDXIL/hlsl/linalg/attr-matrix-type.hlsl
    code = textwrap.dedent("""
        typedef __builtin_LinAlgMatrix [[__LinAlgMatrix_Attributes(ComponentType::F32, 10, 20, MatrixUse::A, MatrixScope::Thread)]] Mat10by20;

        void f2(__builtin_LinAlgMatrix [[__LinAlgMatrix_Attributes(ComponentType::I32, 4, 5, MatrixUse::B, MatrixScope::ThreadGroup)]] mat2) {
            __builtin_LinAlgMatrix [[__LinAlgMatrix_Attributes(ComponentType::I16, 2, 3, MatrixUse::Accumulator, MatrixScope::ThreadGroup)]] mat1;
        }
        """)

    ast = parse_code(code)
    typedef = ast.typedefs[0]
    function = ast.functions[0]
    param = function.params[0]
    local = function.body[0]

    assert typedef.name == "Mat10by20"
    assert typedef.alias_type == "__builtin_LinAlgMatrix"
    assert typedef.attributes[0].name == "__LinAlgMatrix_Attributes"
    assert param.name == "mat2"
    assert param.attributes[0].name == "__LinAlgMatrix_Attributes"
    assert local.name == "mat1"
    assert local.attributes[0].name == "__LinAlgMatrix_Attributes"


def test_parse_linalg_using_alias_post_type_attributes_from_dxc():
    # Source: microsoft/DirectXShaderCompiler
    # tools/clang/test/CodeGenDXIL/hlsl/linalg/matrix-target-type-in-struct.hlsl
    code = textwrap.dedent("""
        using MyHandleT = __builtin_LinAlgMatrix [[__LinAlgMatrix_Attributes(9, 4, 4, 0, 1)]];

        class MyMatrix {
          MyHandleT handle;

          static MyMatrix Splat(float Val) {
            MyMatrix Result;
            __builtin_LinAlg_FillMatrix(Result.handle, Val);
            return Result;
          }
        };

        [numthreads(4, 4, 4)]
        void main() {
          MyMatrix MatA = MyMatrix::Splat(1.0f);
        }
        """)

    ast = parse_code(code)

    alias = ast.typedefs[0]
    local = ast.functions[0].body[0]
    assert alias.name == "MyHandleT"
    assert alias.alias_type == "__builtin_LinAlgMatrix"
    assert alias.attributes[0].name == "__LinAlgMatrix_Attributes"
    assert [arg for arg in alias.attributes[0].args] == [9, 4, 4, 0, 1]
    assert local.name == "MatA"


@pytest.mark.parametrize(
    "code",
    [
        "float4 main() : SV_Target0 { float x = 1.0 return float4(x, 0, 0, 1); }",
        "struct Foo { float4 a; ",
        "void main() { for (int i = 0; i < 4 i++) { } }",
    ],
)
def test_parse_invalid_syntax(code):
    assert_parse_error(code)


if __name__ == "__main__":
    pytest.main()
