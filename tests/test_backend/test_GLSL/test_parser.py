import textwrap
from typing import List

import pytest

from crosstl.backend.GLSL.OpenglAst import (
    BinaryOpNode,
    InitializerListNode,
    VariableNode,
)
from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser


def tokenize_code(code: str) -> List:
    lexer = GLSLLexer(code)
    return lexer.tokenize()


def parse_code(code: str, shader_type: str = "vertex"):
    tokens = tokenize_code(code)
    parser = GLSLParser(tokens, shader_type)
    return parser.parse()


def parse_ok(code: str, shader_type: str = "vertex"):
    ast = parse_code(code, shader_type)
    assert ast is not None
    return ast


def parse_fails(code: str, shader_type: str = "vertex"):
    with pytest.raises(SyntaxError):
        parse_code(code, shader_type)


def test_parse_vertex_shader_with_layout_and_io():
    code = textwrap.dedent("""
        #version 450 core
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec2 uv;
        layout(location = 0) out vec2 vUV;
        uniform mat4 uMVP;

        void main() {
            vUV = uv;
            gl_Position = uMVP * vec4(position, 1.0);
        }
        """)
    parse_ok(code, "vertex")


def test_parse_fragment_shader_with_discard():
    code = textwrap.dedent("""
        #version 450 core
        layout(location = 0) in vec2 vUV;
        layout(location = 0) out vec4 fragColor;
        uniform sampler2D uTexture;

        void main() {
            vec4 color = texture(uTexture, vUV);
            if (color.a < 0.1) {
                discard;
            }
            fragColor = color;
        }
        """)
    parse_ok(code, "fragment")


def test_parse_function_body_with_brace_on_next_line():
    code = textwrap.dedent("""
        #version 320 es
        precision mediump float;
        layout(location = 0) in vec3 in_color;
        layout(location = 0) out vec4 out_color;

        void main()
        {
            out_color = vec4(in_color, 1.0);
        }
        """)

    ast = parse_ok(code, "fragment")

    assert any(function.name == "main" for function in ast.functions)


def test_parse_main_with_void_parameter_list():
    code = textwrap.dedent("""
        #version 320 es
        precision highp float;

        void main(void)
        {
        }
        """)

    ast = parse_ok(code, "fragment")
    main = next(function for function in ast.functions if function.name == "main")

    assert main.params == []


def test_parse_function_parameters_without_names():
    code = textwrap.dedent("""
        #version 400 core

        void ftd(int, float, double) {}
        void itf(int, double, int);

        void main()
        {
            ftd(1, 1.0, 2.0);
        }
        """)

    ast = parse_ok(code, "vertex")
    ftd = next(function for function in ast.functions if function.name == "ftd")
    itf = next(function for function in ast.functions if function.name == "itf")

    assert [(param.vtype, param.name) for param in ftd.params] == [
        ("int", "_param0"),
        ("float", "_param1"),
        ("double", "_param2"),
    ]
    assert [(param.vtype, param.name) for param in itf.params] == [
        ("int", "_param0"),
        ("double", "_param1"),
        ("int", "_param2"),
    ]


def test_parse_struct_with_brace_on_next_line():
    code = textwrap.dedent("""
        #version 450
        struct Particle
        {
            vec4 pos;
            vec4 vel;
        };

        void main()
        {
        }
        """)

    ast = parse_ok(code, "compute")

    assert ast.structs[0].name == "Particle"
    assert [member.name for member in ast.structs[0].members] == ["pos", "vel"]


def test_parse_interface_block_with_newline_brace_and_instance():
    code = textwrap.dedent("""
        #version 450
        layout(push_constant) uniform Registers
        {
            uvec2 resolution;
            vec2 inv_resolution;
        }
        registers;

        void main()
        {
        }
        """)

    ast = parse_ok(code, "compute")

    assert ast.structs[0].name == "Registers"
    assert ast.structs[0].interface_block is True
    assert ast.uniforms[0].name == "registers"


def test_parse_control_flow_with_brace_on_next_line():
    code = textwrap.dedent("""
        #version 450

        void main()
        {
            if (true)
            {
                int value = 1;
            }
        }
        """)

    parse_ok(code, "fragment")


def test_parse_single_statement_control_bodies():
    code = textwrap.dedent("""
        #version 450

        void main()
        {
            int value = 0;
            if (value == 0)
                return;
            else if (value == 1)
                value = 2;
            else
                value = 3;

            for (int i = 0; i < 4; i++)
                value += i;

            while (value < 8)
                value++;
        }
        """)

    parse_ok(code, "compute")


def test_parse_structs_and_arrays():
    code = textwrap.dedent("""
        #version 450 core
        struct Light {
            vec3 position;
            vec3 color;
            float intensity;
        };

        uniform Light lights[4];
        uniform float weights[4];
        layout(location = 0) out vec3 vColor;

        void main() {
            vec3 color = lights[0].color * weights[1];
            vColor = color;
            gl_Position = vec4(1.0);
        }
    """)
    parse_ok(code, "vertex")


def test_parse_float_suffix_literals():
    code = textwrap.dedent("""
        #version 450
        void main(void)
        {
            float normalLength = 0.1f;
            vec3 pos = vec3(1.0f, 0.0f, 0.0f) / 3.0f;
        }
        """)

    parse_ok(code, "geometry")


def test_parse_const_array_initializers():
    code = textwrap.dedent("""
        #version 460 core
        void main() {
            const ivec2 offsets[4] = {
                ivec2(-1, 0),
                ivec2(1, 0),
                ivec2(0, -1),
                ivec2(0, 1),
            };
            const ivec2 ctorOffsets[4] = ivec2[4](
                ivec2(-1, -1),
                ivec2(1, -1),
                ivec2(-1, 1),
                ivec2(1, 1)
            );
            const ivec2 nested[2][2] = {
                { ivec2(-1, 0), ivec2(1, 0) },
                { ivec2(0, -1), ivec2(0, 1) },
            };
        }
        """)

    ast = parse_ok(code, "fragment")
    main = next(function for function in ast.functions if function.name == "main")
    declarations = [stmt for stmt in main.body if isinstance(stmt, VariableNode)]

    offsets, ctor_offsets, nested = declarations
    assert offsets.name == "offsets"
    assert offsets.is_const is True
    assert offsets.is_array is True
    assert offsets.array_size.value == "4"
    assert [size.value for size in offsets.array_sizes] == ["4"]
    assert isinstance(offsets.value, InitializerListNode)
    assert len(offsets.value.elements) == 4

    assert ctor_offsets.name == "ctorOffsets"
    assert ctor_offsets.is_const is True
    assert ctor_offsets.is_array is True
    assert ctor_offsets.array_size.value == "4"
    assert [size.value for size in ctor_offsets.array_sizes] == ["4"]
    assert isinstance(ctor_offsets.value, InitializerListNode)
    assert len(ctor_offsets.value.elements) == 4

    assert nested.name == "nested"
    assert nested.is_const is True
    assert nested.is_array is True
    assert nested.array_size.value == "2"
    assert [size.value for size in nested.array_sizes] == ["2", "2"]
    assert isinstance(nested.value, InitializerListNode)
    assert len(nested.value.elements) == 2
    assert all(
        isinstance(element, InitializerListNode) for element in nested.value.elements
    )


def test_parse_vulkan_block_member_layouts_and_multiline_declarators():
    code = textwrap.dedent("""
        #version 450
        layout(binding = 2) buffer FrequencyInformation
        {
            uvec4 settings;
            uvec2 rates[];
        }
        params;

        layout(push_constant) uniform PushConsts {
            layout(offset = 12) float roughness;
            layout(offset = 16) float metallic;
        } material;

        void main() {
            const uint x0 = gl_GlobalInvocationID.x,
                       y0 = gl_GlobalInvocationID.y,
                       output_width = params.settings[2];
            const float min_rate = 1,
                        max_rate = max(params.settings.x, params.settings.y);
        }
        """)

    ast = parse_ok(code, "compute")
    push_consts = next(struct for struct in ast.structs if struct.name == "PushConsts")

    assert [member.name for member in push_consts.members] == [
        "roughness",
        "metallic",
    ]
    assert push_consts.members[0].layout == {"offset": "12"}
    assert push_consts.members[1].layout == {"offset": "16"}

    main = next(function for function in ast.functions if function.name == "main")
    declarations = [stmt for stmt in main.body if isinstance(stmt, VariableNode)]
    assert [decl.name for decl in declarations] == [
        "x0",
        "y0",
        "output_width",
        "min_rate",
        "max_rate",
    ]


def test_parse_array_type_declarators_and_newline_array_constructors():
    code = textwrap.dedent("""
        #version 450
        const vec2[3] positions = vec2[]
        (
            vec2(-1, -1),
            vec2(-1,  3),
            vec2( 3, -1)
        );

        float conv(in float[9] kernel, in float[9] data) {
            float[9] local;
            return kernel[0] + data[0] + local[0];
        }

        void main() {
            gl_Position = vec4(positions[0], 0.0, 1.0);
        }
        """)

    ast = parse_ok(code, "vertex")

    positions = next(var for var in ast.constant if var.name == "positions")
    assert [size.value for size in positions.array_sizes] == ["3"]
    assert isinstance(positions.value, InitializerListNode)

    conv = next(function for function in ast.functions if function.name == "conv")
    assert [size.value for size in conv.params[0].array_sizes] == ["9"]
    assert [size.value for size in conv.params[1].array_sizes] == ["9"]
    local = next(stmt for stmt in conv.body if isinstance(stmt, VariableNode))
    assert [size.value for size in local.array_sizes] == ["9"]


def test_parse_vulkan_extension_types_suffixes_and_qualifiers():
    code = textwrap.dedent("""
        #version 460
        #extension GL_ARM_tensors : enable
        #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

        layout(set = 0, binding = 0) writeonly uniform tensorARM<float, 4> output_tensor;
        layout(location = 0) in pervertexEXT vec3 inColor[];

        void main() {
            vec3 sample[9];
            f16vec4 rg_offset = f16vec4(0.95hf, 1.0hf, 0.95hf, 1.0hf);
            sample[0] = inColor[0];
        }
        """)

    ast = parse_ok(code, "fragment")

    tensor = next(var for var in ast.uniforms if var.name == "output_tensor")
    assert tensor.vtype == "tensorARM<float, 4>"

    in_color = next(var for var in ast.io_variables if var.name == "inColor")
    assert "pervertexEXT" in in_color.qualifiers
    assert in_color.is_array is True


def test_parse_empty_statement_after_control_block():
    code = textwrap.dedent("""
        #version 450
        void main() {
            if (gl_FrontFacing) {
                gl_FragDepth = 1.0;
            };
        }
        """)

    parse_ok(code, "fragment")


def test_parse_control_flow_constructs():
    code = textwrap.dedent("""
        #version 450 core
        layout(location = 0) in vec3 position;

        void main() {
            int i = 0;
            for (int j = 0; j < 4; j++) {
                if (j == 2) {
                    continue;
                }
                i += j;
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
                default:
                    break;
            }

            if (i > 2) {
                i = i * 2;
            } else {
                i = 1;
            }

            gl_Position = vec4(position, 1.0);
        }
        """)
    parse_ok(code, "vertex")


def test_parse_ternary_and_bitwise_expressions():
    code = textwrap.dedent("""
        #version 450 core
        void main() {
            int a = 3;
            int b = 5;
            int c = (a > b) ? a : b;
            int d = (a & b) | (a ^ b);
            int e = ~a;
            bool f = (a != b) && (a <= b || b >= a);
            gl_Position = vec4(float(c + d + e));
        }
        """)
    parse_ok(code, "vertex")


def test_parse_logical_xor_precedence():
    code = textwrap.dedent("""
        #version 450 core
        void main() {
            bool a = true;
            bool b = false;
            bool c = true;
            bool d = false;
            bool result = a || b ^^ c && d;
        }
        """)

    ast = parse_ok(code, "fragment")
    main = next(function for function in ast.functions if function.name == "main")
    result = next(
        stmt
        for stmt in main.body
        if isinstance(stmt, VariableNode) and stmt.name == "result"
    )

    assert isinstance(result.value, BinaryOpNode)
    assert result.value.op == "||"
    assert isinstance(result.value.right, BinaryOpNode)
    assert result.value.right.op == "^^"
    assert isinstance(result.value.right.right, BinaryOpNode)
    assert result.value.right.right.op == "&&"


def test_parse_parenthesized_comma_assignment_expression():
    code = textwrap.dedent("""
        #version 450 core
        void main() {
            int a = 0;
            int b = (a = 1, a + 2);
            sink((a = 3, b), b);
        }
        """)

    ast = parse_ok(code, "fragment")
    main = next(function for function in ast.functions if function.name == "main")
    b_decl = next(
        stmt
        for stmt in main.body
        if isinstance(stmt, VariableNode) and stmt.name == "b"
    )
    call = main.body[2]

    assert isinstance(b_decl.value, BinaryOpNode)
    assert b_decl.value.op == ","
    assert b_decl.value.left.operator == "="
    assert isinstance(b_decl.value.right, BinaryOpNode)
    assert b_decl.value.right.op == "+"

    assert call.name.name == "sink"
    assert len(call.args) == 2
    assert isinstance(call.args[0], BinaryOpNode)
    assert call.args[0].op == ","


def test_parse_matrix_vector_operations():
    code = textwrap.dedent("""
        #version 450 core
        void main() {
            mat4 m = mat4(1.0);
            vec4 v = vec4(1.0, 2.0, 3.0, 1.0);
            vec4 r = m * v;
            vec3 n = normalize(v.xyz);
            gl_Position = r + vec4(n, 0.0);
        }
        """)
    parse_ok(code, "vertex")


def test_parse_precision_qualifiers():
    code = textwrap.dedent("""
        #version 300 es
        precision highp float;
        precision mediump sampler2D;
        in vec2 vUV;
        out vec4 fragColor;
        uniform sampler2D uTexture;

        void main() {
            mediump vec4 color = texture(uTexture, vUV);
            fragColor = color;
        }
        """)
    parse_ok(code, "fragment")


def test_parse_preprocessor_directives():
    code = textwrap.dedent("""
        #version 450 core
        #define USE_LIGHTING 1
        #if USE_LIGHTING
        #define FACTOR 0.5
        #endif

        void main() {
            float value = FACTOR;
            gl_Position = vec4(value);
        }
        """)
    parse_ok(code, "vertex")


def test_parse_function_overloads():
    code = textwrap.dedent("""
        #version 450 core
        float compute(vec3 n, vec3 l) {
            return max(dot(n, l), 0.0);
        }

        float compute(vec3 n, vec3 l, float intensity) {
            return compute(n, l) * intensity;
        }

        void main() {
            float v = compute(vec3(0.0), vec3(1.0), 0.5);
            gl_Position = vec4(v);
        }
        """)
    parse_ok(code, "vertex")


def test_parse_interface_blocks_and_uniform_block():
    code = textwrap.dedent("""
        #version 450 core
        layout(std140, binding = 0) uniform Globals {
            mat4 mvp[2][3];
            vec4 baseColor[2][2];
        } globals[2][3];

        in VertexIn {
            vec3 position[2][3];
            vec2 uv[2];
        } vin[2];

        out VertexOut {
            vec4 color[2][3];
        } vout;

        void main() {
            vout.color[1][2] = vec4(vin[1].position[0][1], 1.0);
            gl_Position = globals[1][2].mvp[0][1] * vec4(vin[1].position[0][1], 1.0);
        }
        """)
    ast = parse_ok(code, "vertex")

    globals_struct = next(struct for struct in ast.structs if struct.name == "Globals")
    vertex_in = next(struct for struct in ast.structs if struct.name == "VertexIn")
    vertex_out = next(struct for struct in ast.structs if struct.name == "VertexOut")

    assert [size.value for size in globals_struct.interface_array_sizes] == ["2", "3"]
    assert [
        [size.value for size in member.array_sizes] for member in globals_struct.members
    ] == [["2", "3"], ["2", "2"]]
    assert [size.value for size in vertex_in.interface_array_sizes] == ["2"]
    assert [
        [size.value for size in member.array_sizes] for member in vertex_in.members
    ] == [
        ["2", "3"],
        ["2"],
    ]
    assert [size.value for size in vertex_out.members[0].array_sizes] == ["2", "3"]

    globals_var = next(var for var in ast.uniforms if var.name == "globals")
    vin_var = next(var for var in ast.io_variables if var.name == "vin")
    assert [size.value for size in globals_var.array_sizes] == ["2", "3"]
    assert [size.value for size in vin_var.array_sizes] == ["2"]


def test_parse_compute_layout_qualifier():
    code = textwrap.dedent("""
        #version 430 core
        layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
        void main() {
            // No-op
        }
        """)
    parse_ok(code, "compute")


def test_parse_subroutine_declaration():
    code = textwrap.dedent("""
        #version 450 core
        subroutine vec4 ShadeFunc(vec3 n);
        subroutine uniform ShadeFunc shade;
        void main() { }
        """)
    parse_ok(code, "fragment")


def test_parse_swizzle_and_constructors():
    code = textwrap.dedent("""
        #version 450 core
        void main() {
            vec4 v = vec4(1.0, 2.0, 3.0, 4.0);
            vec2 xy = v.xy;
            vec3 rgb = v.rgb;
            mat3 m = mat3(1.0);
            float arr[] = float[](1.0, 2.0, 3.0);
        }
        """)
    parse_ok(code, "vertex")


@pytest.mark.parametrize(
    "code",
    [
        "void main() { int a = 0 }",  # Missing semicolon
        "void main() { if (true) { gl_Position = vec4(1.0); }",  # Missing brace
        "void main() { else { gl_Position = vec4(1.0); } }",  # Else without if
        "void main() { case 0: break; }",  # Case outside switch
        "void main() { switch (1) { case 0 return; } }",  # Missing colon
        "layout(location =) in vec3 position; void main() { gl_Position = vec4(position, 1.0); }",  # Bad layout
        "void main( { gl_Position = vec4(1.0); }",  # Unterminated signature
    ],
)
def test_parse_invalid_syntax_cases(code):
    parse_fails(code, "vertex")


if __name__ == "__main__":
    pytest.main()
