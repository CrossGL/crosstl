from crosstl.translator.lexer import Lexer
import pytest
import shutil
import subprocess
from typing import List
from crosstl.translator.parser import Parser
from crosstl.translator.ast import (
    ArrayAccessNode,
    AtomicOpNode,
    BlockNode,
    FunctionNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    PrimitiveType,
)
from crosstl.translator.codegen.slang_codegen import SlangCodeGen


def tokenize_code(code: str) -> List:
    """Helper function to tokenize code."""
    lexer = Lexer(code)
    return lexer.get_tokens()


def parse_code(tokens: List):
    """Helper function to parse tokens into an AST.

    Args:
        tokens (List): The list of tokens to parse
    Returns:
        AST: The abstract syntax tree generated from the parser
    """
    parser = Parser(tokens)
    return parser.parse()


def generate_code(ast_node):
    """Test the code generator
    Args:
        ast_node: The abstract syntax tree generated from the code
    Returns:
        str: The generated code from the abstract syntax tree
    """
    codegen = SlangCodeGen()
    return codegen.generate(ast_node)


def compile_generated_slang(generated_code, tmp_path, stage, profile="sm_6_0"):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")

    source_path = tmp_path / f"smoke_{stage}.slang"
    output_path = tmp_path / f"smoke_{stage}.hlsl"
    source_path.write_text(generated_code, encoding="utf-8")

    result = subprocess.run(
        [
            slangc,
            "-target",
            "hlsl",
            "-entry",
            "main",
            "-stage",
            stage,
            "-profile",
            profile,
            "-o",
            str(output_path),
            str(source_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert output_path.exists()


def test_struct():
    code = """
    struct VSInput {
        vec2 texCoord @ TEXCOORD0;
    };
    struct VSOutput {
        vec4 color @ COLOR;
    };
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        assert code is not None
    except SyntaxError:
        pytest.fail("Slang struct codegen not implemented.")


def test_basic_shader():
    code = """
    shader main {
        struct VSInput {
            vec2 texCoord @ TEXCOORD0;
        };
        struct VSOutput {
            vec4 color @ COLOR;
        };
        vertex {
            VSOutput main(VSInput input) {
                VSOutput output;
                output.color = vec4(input.texCoord, 0.0, 1.0);
                return output;
            }
        }
        fragment {
            vec4 main(VSOutput input) @ gl_FragColor {
                return input.color;
            }
        }
    }
    """
    try:
        tokens = tokenize_code(code)
        ast = parse_code(tokens)
        code = generate_code(ast)
        assert code is not None
    except SyntaxError:
        pytest.fail("Slang basic shader codegen not implemented.")


def test_stage_only_shader_emits_slang_entry_point():
    code = """
    shader main {
        vertex {
            void main() {
                gl_Position = vec4(1.0, 0.0, 0.0, 1.0);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "// Vertex Shader" in generated_code
    assert '[shader("vertex")]' in generated_code
    assert "float4 main() : SV_Position" in generated_code
    assert "return float4(1.0, 0.0, 0.0, 1.0);" in generated_code
    assert "gl_Position =" not in generated_code


def test_stage_local_functions_and_initialized_variables_emit():
    code = """
    shader main {
        vertex {
            float helper(float x) {
                return x;
            }

            void main() {
                float y = helper(1.0);
                gl_Position = vec4(y, y, y, 1.0);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float helper(float x)" in generated_code
    assert "return x;" in generated_code
    assert "float y = helper(1.0);" in generated_code
    assert "float4 main() : SV_Position" in generated_code
    assert "return float4(y, y, y, 1.0);" in generated_code
    assert "gl_Position =" not in generated_code


def test_vertex_position_assignment_rewrites_only_final_top_level_assignment():
    code = """
    shader main {
        vertex {
            void main() {
                gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
                float y = 1.0;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "float4 main() : SV_Position" in generated_code
    assert "float4 cgl_Position;" in generated_code
    assert "cgl_Position = float4(0.0, 0.0, 0.0, 1.0);" in generated_code
    assert "float y = 1.0;" in generated_code
    assert "return cgl_Position;" in generated_code
    assert "\n    gl_Position =" not in generated_code


def test_fragment_output_assignments_rewrite_to_slang_return_semantics():
    code = """
    shader main {
        fragment {
            void main() {
                gl_FragColor = vec4(1.0, 0.5, 0.25, 1.0);
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "float4 main() : SV_Target" in generated_code
    assert "return float4(1.0, 0.5, 0.25, 1.0);" in generated_code
    assert "gl_FragColor =" not in generated_code


def test_fragment_depth_assignment_rewrites_to_slang_return_semantic():
    code = """
    shader main {
        fragment {
            void main() {
                float depth = 0.25;
                gl_FragDepth = depth;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "float main() : SV_Depth" in generated_code
    assert "float depth = 0.25;" in generated_code
    assert "return depth;" in generated_code
    assert "gl_FragDepth =" not in generated_code


def test_control_flow_fragment_output_assignments_rewrite_to_local_return():
    code = """
    shader main {
        fragment {
            void main() {
                if (gl_FrontFacing) {
                    gl_FragColor = vec4(1.0);
                } else {
                    gl_FragColor = vec4(0.0);
                }
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "float4 main(bool gl_FrontFacing : SV_IsFrontFace) : SV_Target"
        in generated_code
    )
    assert "float4 cgl_FragColor;" in generated_code
    assert "if (gl_FrontFacing)" in generated_code
    assert "cgl_FragColor = float4(1.0);" in generated_code
    assert "cgl_FragColor = float4(0.0);" in generated_code
    assert "return cgl_FragColor;" in generated_code
    assert "\n        gl_FragColor =" not in generated_code


def test_fragment_color_and_depth_outputs_rewrite_to_slang_output_struct():
    code = """
    shader main {
        fragment {
            void main() {
                gl_FragColor = vec4(1.0, 0.5, 0.25, 1.0);
                gl_FragDepth = 0.5;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct CGLPSMainOutput" in generated_code
    assert "float4 cgl_FragColor : SV_Target;" in generated_code
    assert "float cgl_FragDepth : SV_Depth;" in generated_code
    assert "CGLPSMainOutput main()" in generated_code
    assert "CGLPSMainOutput cgl_Output;" in generated_code
    assert "cgl_Output.cgl_FragColor = float4(1.0, 0.5, 0.25, 1.0);" in generated_code
    assert "cgl_Output.cgl_FragDepth = 0.5;" in generated_code
    assert "return cgl_Output;" in generated_code
    assert "\n    gl_FragColor =" not in generated_code
    assert "\n    gl_FragDepth =" not in generated_code


def test_fragment_multiple_color_outputs_rewrite_to_slang_output_struct():
    code = """
    shader main {
        fragment {
            void main() {
                gl_FragColor0 = vec4(1.0);
                gl_FragColor1 = vec4(0.0);
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "float4 cgl_FragColor0 : SV_Target0;" in generated_code
    assert "float4 cgl_FragColor1 : SV_Target1;" in generated_code
    assert "cgl_Output.cgl_FragColor0 = float4(1.0);" in generated_code
    assert "cgl_Output.cgl_FragColor1 = float4(0.0);" in generated_code
    assert "return cgl_Output;" in generated_code


def test_fragment_frag_data_output_rewrites_to_slang_target_semantic():
    code = """
    shader main {
        fragment {
            void main() {
                gl_FragData[0] = vec4(1.0);
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "float4 main() : SV_Target0" in generated_code
    assert "return float4(1.0);" in generated_code
    assert "gl_FragData" not in generated_code


def test_fragment_multiple_frag_data_outputs_rewrite_to_slang_output_struct():
    code = """
    shader main {
        fragment {
            void main() {
                gl_FragData[0] = vec4(1.0);
                gl_FragData[1] = vec4(0.0);
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "float4 cgl_FragColor0 : SV_Target0;" in generated_code
    assert "float4 cgl_FragColor1 : SV_Target1;" in generated_code
    assert "cgl_Output.cgl_FragColor0 = float4(1.0);" in generated_code
    assert "cgl_Output.cgl_FragColor1 = float4(0.0);" in generated_code
    assert "return cgl_Output;" in generated_code
    assert "gl_FragData" not in generated_code


def test_fragment_non_output_array_assignment_does_not_crash_output_scan():
    code = """
    shader main {
        vec4 colors[2];

        fragment {
            void main() {
                colors[0] = vec4(1.0);
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "float4 colors[2];" in generated_code
    assert "void main()" in generated_code
    assert "colors[0] = float4(1.0);" in generated_code


@pytest.mark.parametrize(
    "target_index",
    [
        "targetIndex",
        "-1",
    ],
)
def test_fragment_dynamic_frag_data_output_index_raises_clear_error(target_index):
    code = f"""
    shader main {{
        fragment {{
            void main() {{
                int targetIndex = 0;
                gl_FragData[{target_index}] = vec4(1.0);
            }}
        }}
    }}
    """
    with pytest.raises(
        ValueError,
        match=("gl_FragData requires a non-negative literal " "render-target index"),
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_vertex_position_and_point_size_outputs_rewrite_to_slang_output_struct():
    code = """
    shader main {
        vertex {
            void main() {
                gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
                gl_PointSize = 4.0;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct CGLVSMainOutput" in generated_code
    assert "float4 cgl_Position : SV_Position;" in generated_code
    assert "float cgl_PointSize : PSIZE;" in generated_code
    assert "CGLVSMainOutput main()" in generated_code
    assert "cgl_Output.cgl_Position = float4(0.0, 0.0, 0.0, 1.0);" in generated_code
    assert "cgl_Output.cgl_PointSize = 4.0;" in generated_code
    assert "return cgl_Output;" in generated_code


def test_vertex_layer_and_viewport_outputs_rewrite_to_slang_output_struct():
    code = """
    shader main {
        vertex {
            void main() {
                gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
                gl_Layer = 1u;
                gl_ViewportIndex = 2u;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "struct CGLVSMainOutput" in generated_code
    assert "float4 cgl_Position : SV_Position;" in generated_code
    assert "uint cgl_Layer : SV_RenderTargetArrayIndex;" in generated_code
    assert "uint cgl_ViewportIndex : SV_ViewportArrayIndex;" in generated_code
    assert "cgl_Output.cgl_Position = float4(0.0, 0.0, 0.0, 1.0);" in generated_code
    assert "cgl_Output.cgl_Layer = 1u;" in generated_code
    assert "cgl_Output.cgl_ViewportIndex = 2u;" in generated_code
    assert "return cgl_Output;" in generated_code
    assert "\n    gl_Layer =" not in generated_code
    assert "\n    gl_ViewportIndex =" not in generated_code


def test_slangc_smoke_compiles_generated_vertex_output_rewrite(tmp_path):
    code = """
    shader SmokeVertex {
        vertex {
            void main() {
                gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    compile_generated_slang(generated_code, tmp_path, "vertex")


def test_slangc_smoke_compiles_generated_fragment_output_struct(tmp_path):
    code = """
    shader SmokeFragment {
        fragment {
            void main() {
                gl_FragColor = vec4(1.0, 0.5, 0.25, 1.0);
                gl_FragDepth = 0.5;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    compile_generated_slang(generated_code, tmp_path, "fragment")


def test_slangc_smoke_compiles_generated_compute_system_values(tmp_path):
    code = """
    shader SmokeCompute {
        compute {
            void main() {
                uint globalX = gl_GlobalInvocationID.x;
                uint localIndex = gl_LocalInvocationIndex;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    compile_generated_slang(generated_code, tmp_path, "compute")


def test_compute_system_value_builtins_emit_implicit_slang_parameters():
    code = """
    shader main {
        compute {
            void main() {
                uint globalX = gl_GlobalInvocationID.x;
                uint localIndex = gl_LocalInvocationIndex;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "void main(uint3 gl_GlobalInvocationID : SV_DispatchThreadID, "
        "uint gl_LocalInvocationIndex : SV_GroupIndex)" in generated_code
    )
    assert "uint globalX = gl_GlobalInvocationID.x;" in generated_code
    assert "uint localIndex = gl_LocalInvocationIndex;" in generated_code


def test_system_value_builtin_uses_existing_semantic_parameter_alias():
    code = """
    shader main {
        compute {
            void main(uvec3 tid @ gl_GlobalInvocationID) {
                uint globalX = gl_GlobalInvocationID.x;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "void main(uint3 tid : SV_DispatchThreadID)" in generated_code
    assert "uint3 gl_GlobalInvocationID : SV_DispatchThreadID" not in generated_code
    assert "uint globalX = tid.x;" in generated_code
    assert "gl_GlobalInvocationID.x" not in generated_code


def test_vertex_system_value_builtin_combines_with_position_return_rewrite():
    code = """
    shader main {
        vertex {
            void main() {
                float x = float(gl_VertexID);
                gl_Position = vec4(x, 0.0, 0.0, 1.0);
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "float4 main(uint gl_VertexID : SV_VertexID) : SV_Position" in generated_code
    assert "float x = float(gl_VertexID);" in generated_code
    assert "return float4(x, 0.0, 0.0, 1.0);" in generated_code


def test_vertex_draw_metadata_builtins_emit_implicit_slang_parameters():
    code = """
    shader main {
        vertex {
            void main() {
                int absoluteVertex = int(gl_VertexID) + gl_BaseVertex;
                uint instance = gl_InstanceID + gl_BaseInstance;
                uint draw = gl_DrawID;
                float x = float(absoluteVertex) + float(instance + draw);
                gl_Position = vec4(x, 0.0, 0.0, 1.0);
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "float4 main(uint gl_VertexID : SV_VertexID, "
        "uint gl_InstanceID : SV_InstanceID, "
        "int gl_BaseVertex : SV_StartVertexLocation, "
        "uint gl_BaseInstance : SV_StartInstanceLocation, "
        "uint gl_DrawID : SV_DrawID) : SV_Position" in generated_code
    )
    assert "int absoluteVertex = int(gl_VertexID) + gl_BaseVertex;" in generated_code
    assert "uint instance = gl_InstanceID + gl_BaseInstance;" in generated_code
    assert "uint draw = gl_DrawID;" in generated_code
    assert "return float4(x, 0.0, 0.0, 1.0);" in generated_code


def test_vertex_draw_metadata_uses_existing_semantic_parameter_aliases():
    code = """
    shader main {
        vertex {
            void main(
                int baseVertex @ gl_BaseVertex,
                uint baseInstance @ gl_BaseInstance,
                uint drawIndex @ gl_DrawID
            ) {
                int absoluteVertex = baseVertex + gl_BaseVertex;
                uint instance = baseInstance + gl_BaseInstance;
                uint draw = drawIndex + gl_DrawID;
                gl_Position = vec4(float(absoluteVertex) + float(instance + draw));
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "int baseVertex : SV_StartVertexLocation" in generated_code
    assert "uint baseInstance : SV_StartInstanceLocation" in generated_code
    assert "uint drawIndex : SV_DrawID" in generated_code
    assert "int gl_BaseVertex : SV_StartVertexLocation" not in generated_code
    assert "uint gl_BaseInstance : SV_StartInstanceLocation" not in generated_code
    assert "uint gl_DrawID : SV_DrawID" not in generated_code
    assert "int absoluteVertex = baseVertex + baseVertex;" in generated_code
    assert "uint instance = baseInstance + baseInstance;" in generated_code
    assert "uint draw = drawIndex + drawIndex;" in generated_code


def test_fragment_system_value_builtin_combines_with_color_return_rewrite():
    code = """
    shader main {
        fragment {
            void main() {
                bool front = gl_FrontFacing;
                gl_FragColor = front ? vec4(1.0) : vec4(0.0);
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "float4 main(bool gl_FrontFacing : SV_IsFrontFace) : SV_Target"
        in generated_code
    )
    assert "bool front = gl_FrontFacing;" in generated_code
    assert "return (front ? float4(1.0) : float4(0.0));" in generated_code


def test_fragment_sample_layer_and_viewport_builtins_emit_implicit_slang_parameters():
    code = """
    shader main {
        fragment {
            void main() {
                uint sampleId = gl_SampleID;
                uint layer = gl_Layer;
                uint viewport = gl_ViewportIndex;
                gl_FragColor = vec4(float(sampleId + layer + viewport));
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "float4 main(uint gl_SampleID : SV_SampleIndex, "
        "uint gl_Layer : SV_RenderTargetArrayIndex, "
        "uint gl_ViewportIndex : SV_ViewportArrayIndex) : SV_Target" in generated_code
    )
    assert "uint sampleId = gl_SampleID;" in generated_code
    assert "uint layer = gl_Layer;" in generated_code
    assert "uint viewport = gl_ViewportIndex;" in generated_code
    assert "return float4(float(sampleId + layer + viewport));" in generated_code


def test_sample_system_value_builtin_uses_existing_semantic_parameter_alias():
    code = """
    shader main {
        fragment {
            void main(uint sampleIndex @ gl_SampleID) {
                uint sampleId = gl_SampleID;
                gl_FragColor = vec4(float(sampleId));
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "float4 main(uint sampleIndex : SV_SampleIndex) : SV_Target" in generated_code
    )
    assert "uint gl_SampleID : SV_SampleIndex" not in generated_code
    assert "uint sampleId = sampleIndex;" in generated_code
    assert "gl_SampleID" not in generated_code


def test_geometry_stage_builtins_emit_stage_specific_slang_parameters():
    code = """
    shader main {
        geometry {
            void main() {
                uint primitive = gl_PrimitiveIDIn;
                uint invocation = gl_InvocationID;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "void main(uint gl_PrimitiveIDIn : SV_PrimitiveID, "
        "uint gl_InvocationID : SV_GSInstanceID)" in generated_code
    )
    assert "uint primitive = gl_PrimitiveIDIn;" in generated_code
    assert "uint invocation = gl_InvocationID;" in generated_code


def test_geometry_invocation_builtin_uses_existing_semantic_parameter_alias():
    code = """
    shader main {
        geometry {
            void main(uint gsInstance @ gl_InvocationID) {
                uint invocation = gl_InvocationID;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "void main(uint gsInstance : SV_GSInstanceID)" in generated_code
    assert "uint gl_InvocationID : SV_GSInstanceID" not in generated_code
    assert "uint invocation = gsInstance;" in generated_code


def test_tessellation_control_builtins_emit_hull_slang_parameters():
    code = """
    shader main {
        tessellation_control {
            void main() {
                uint controlPoint = gl_InvocationID;
                uint primitive = gl_PrimitiveID;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "void main(uint gl_InvocationID : SV_OutputControlPointID, "
        "uint gl_PrimitiveID : SV_PrimitiveID)" in generated_code
    )
    assert "uint controlPoint = gl_InvocationID;" in generated_code
    assert "uint primitive = gl_PrimitiveID;" in generated_code


def test_tessellation_invocation_builtin_uses_hull_semantic_alias():
    code = """
    shader main {
        tessellation_control {
            void main(uint controlPoint @ gl_InvocationID) {
                uint index = gl_InvocationID;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "void main(uint controlPoint : SV_OutputControlPointID)" in generated_code
    assert "uint gl_InvocationID : SV_OutputControlPointID" not in generated_code
    assert "uint index = controlPoint;" in generated_code
    assert "SV_GSInstanceID" not in generated_code


def test_tessellation_evaluation_builtins_emit_domain_slang_parameters():
    code = """
    shader main {
        tessellation_evaluation {
            void main() {
                vec3 coord = gl_TessCoord;
                uint primitive = gl_PrimitiveID;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "void main(float3 gl_TessCoord : SV_DomainLocation, "
        "uint gl_PrimitiveID : SV_PrimitiveID)" in generated_code
    )
    assert "float3 coord = gl_TessCoord;" in generated_code
    assert "uint primitive = gl_PrimitiveID;" in generated_code


def test_tessellation_factor_semantics_map_to_slang_patch_constant_outputs():
    code = """
    shader main {
        struct PatchConstants {
            float outer[3] @ gl_TessLevelOuter;
            float inner[1] @ gl_TessLevelInner;
        };

        tessellation_control {
            PatchConstants HSConst() {
                PatchConstants patch;
                return patch;
            }

            void main()
                @domain(tri)
                @partitioning(integer)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "float outer[3] : SV_TessFactor;" in generated_code
    assert "float inner[1] : SV_InsideTessFactor;" in generated_code
    assert "PatchConstants HSConst()" in generated_code
    assert '[patchconstantfunc("HSConst")]' in generated_code
    assert ": gl_TessLevelOuter" not in generated_code
    assert ": gl_TessLevelInner" not in generated_code


def test_tessellation_domain_struct_semantics_map_without_stage_context():
    code = """
    struct DomainInput {
        vec3 coord @ gl_TessCoord;
        uint primitive @ gl_PrimitiveIDIn;
    };
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "float3 coord : SV_DomainLocation;" in generated_code
    assert "uint primitive : SV_PrimitiveID;" in generated_code
    assert ": gl_TessCoord" not in generated_code
    assert ": gl_PrimitiveIDIn" not in generated_code


def test_tessellation_patch_generic_literal_type_arguments_emit():
    code = """
    shader main {
        struct VSOut {
            vec4 position @ gl_Position;
        };

        tessellation_evaluation {
            void main(OutputPatch<VSOut, 3> patch) @domain(tri) {
                VSOut first = patch[0];
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "void main(OutputPatch<VSOut, 3> patch)" in generated_code
    assert "VSOut first = patch[0];" in generated_code
    assert "None" not in generated_code


def test_tessellation_control_gl_in_aliases_explicit_input_patch_parameter():
    code = """
    shader main {
        struct VSOut {
            vec4 position @ gl_Position;
        };

        tessellation_control {
            void main(InputPatch<VSOut, 3> inputPatch) {
                VSOut first = gl_in[0];
                VSOut current = gl_in[gl_InvocationID];
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "void main(InputPatch<VSOut, 3> inputPatch, "
        "uint gl_InvocationID : SV_OutputControlPointID)" in generated_code
    )
    assert "VSOut first = inputPatch[0];" in generated_code
    assert "VSOut current = inputPatch[gl_InvocationID];" in generated_code
    assert "gl_in" not in generated_code


def test_tessellation_evaluation_gl_in_aliases_explicit_output_patch_parameter():
    code = """
    shader main {
        struct HSOut {
            vec4 position @ gl_Position;
        };

        tessellation_evaluation {
            void main(OutputPatch<HSOut, 3> patch) {
                HSOut first = gl_in[0];
                vec3 coord = gl_TessCoord;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "void main(OutputPatch<HSOut, 3> patch, "
        "float3 gl_TessCoord : SV_DomainLocation)" in generated_code
    )
    assert "HSOut first = patch[0];" in generated_code
    assert "float3 coord = gl_TessCoord;" in generated_code
    assert "gl_in" not in generated_code


def test_tessellation_control_gl_out_output_patch_lowers_to_hull_return():
    code = """
    shader main {
        struct VSOut {
            vec4 position @ gl_Position;
        };

        struct HSOut {
            vec4 position @ gl_Position;
            vec4 color @ TEXCOORD0;
        };

        tessellation_control {
            void main(
                InputPatch<VSOut, 3> inputPatch,
                OutputPatch<HSOut, 3> gl_out
            ) {
                gl_out[gl_InvocationID].position =
                    inputPatch[gl_InvocationID].position;
                gl_out[gl_InvocationID].color = vec4(1.0);
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "HSOut main(InputPatch<VSOut, 3> inputPatch, "
        "uint gl_InvocationID : SV_OutputControlPointID)" in generated_code
    )
    assert "OutputPatch<HSOut, 3> gl_out" not in generated_code
    assert "HSOut cgl_OutputControlPoint;" in generated_code
    assert (
        "cgl_OutputControlPoint.position = "
        "inputPatch[gl_InvocationID].position;" in generated_code
    )
    assert "cgl_OutputControlPoint.color = float4(1.0);" in generated_code
    assert "return cgl_OutputControlPoint;" in generated_code
    assert "gl_out" not in generated_code


def test_tessellation_control_gl_out_non_void_return_lowers_assignments():
    code = """
    shader main {
        struct VSOut {
            vec4 position @ gl_Position;
        };

        struct HSOut {
            vec4 position @ gl_Position;
        };

        tessellation_control {
            HSOut main(InputPatch<VSOut, 3> inputPatch) {
                gl_out[gl_InvocationID].position =
                    inputPatch[gl_InvocationID].position;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "HSOut main(InputPatch<VSOut, 3> inputPatch, "
        "uint gl_InvocationID : SV_OutputControlPointID)" in generated_code
    )
    assert "HSOut cgl_OutputControlPoint;" in generated_code
    assert (
        "cgl_OutputControlPoint.position = "
        "inputPatch[gl_InvocationID].position;" in generated_code
    )
    assert "return cgl_OutputControlPoint;" in generated_code
    assert "gl_out" not in generated_code


def test_tessellation_control_gl_out_uses_existing_control_point_parameter():
    code = """
    shader main {
        struct HSOut {
            vec4 position @ gl_Position;
        };

        tessellation_control {
            void main(
                OutputPatch<HSOut, 3> outputs,
                uint controlPoint @ gl_InvocationID
            ) {
                gl_out[controlPoint].position = vec4(1.0);
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "HSOut main(uint controlPoint : SV_OutputControlPointID)" in generated_code
    assert "uint gl_InvocationID : SV_OutputControlPointID" not in generated_code
    assert "cgl_OutputControlPoint.position = float4(1.0);" in generated_code
    assert "gl_out" not in generated_code


def test_tessellation_control_gl_out_whole_control_point_assignment_lowers():
    code = """
    shader main {
        struct VSOut {
            vec4 position @ gl_Position;
        };

        struct HSOut {
            vec4 position @ gl_Position;
            vec4 color @ TEXCOORD0;
        };

        tessellation_control {
            void main(
                InputPatch<VSOut, 3> inputPatch,
                OutputPatch<HSOut, 3> gl_out
            ) {
                HSOut output;
                output.position = inputPatch[gl_InvocationID].position;
                output.color = vec4(1.0);
                gl_out[gl_InvocationID] = output;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "HSOut main(InputPatch<VSOut, 3> inputPatch, "
        "uint gl_InvocationID : SV_OutputControlPointID)" in generated_code
    )
    assert "cgl_OutputControlPoint = output;" in generated_code
    assert "return cgl_OutputControlPoint;" in generated_code
    assert "OutputPatch<HSOut, 3> gl_out" not in generated_code
    assert "gl_out" not in generated_code


def test_tessellation_control_gl_out_current_reads_alias_to_hull_return_local():
    code = """
    shader main {
        struct HSOut {
            vec4 position @ gl_Position;
        };

        tessellation_control {
            void main(OutputPatch<HSOut, 3> gl_out) {
                gl_out[gl_InvocationID].position = vec4(1.0);
                vec4 copied = gl_out[gl_InvocationID].position;
                gl_out[gl_InvocationID].position.x = copied.y;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "float4 copied = cgl_OutputControlPoint.position;" in generated_code
    assert "cgl_OutputControlPoint.position.x = copied.y;" in generated_code
    assert "return cgl_OutputControlPoint;" in generated_code
    assert "gl_out" not in generated_code


@pytest.mark.parametrize(
    ("stage_source", "message"),
    [
        (
            """
            tessellation_control {
                void main() {
                    gl_out[gl_InvocationID].position = vec4(1.0);
                }
            }
            """,
            "gl_out requires a non-void return type or exactly one explicit OutputPatch",
        ),
        (
            """
            tessellation_control {
                void main(OutputPatch<HSOut, 3> outputs) {
                    gl_out[0].position = vec4(1.0);
                }
            }
            """,
            "gl_out writes must target gl_out\\[gl_InvocationID\\]",
        ),
        (
            """
            tessellation_control {
                void main(OutputPatch<HSOut, 3> outputs) {
                    gl_out[gl_InvocationID].position = vec4(1.0);
                    HSOut previous = gl_out[0];
                }
            }
            """,
            "gl_out accesses must target gl_out\\[gl_InvocationID\\]",
        ),
        (
            """
            tessellation_control {
                void main(OutputPatch<HSOut, 3> outputs) {
                    gl_out[gl_InvocationID].position = vec4(1.0);
                    HSOut previous = gl_out;
                }
            }
            """,
            "gl_out must be indexed as gl_out\\[gl_InvocationID\\]",
        ),
    ],
)
def test_tessellation_control_invalid_gl_out_raises(stage_source, message):
    code = f"""
    shader main {{
        struct HSOut {{
            vec4 position @ gl_Position;
        }};

        {stage_source}
    }}
    """
    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(code)))


@pytest.mark.parametrize(
    ("stage_source", "message"),
    [
        (
            """
            tessellation_control {
                void main() {
                    VSOut first = gl_in[0];
                }
            }
            """,
            "hull stage gl_in requires exactly one explicit InputPatch",
        ),
        (
            """
            tessellation_evaluation {
                void main() {
                    HSOut first = gl_in[0];
                }
            }
            """,
            "domain stage gl_in requires exactly one explicit OutputPatch",
        ),
    ],
)
def test_tessellation_gl_in_without_explicit_patch_parameter_raises(
    stage_source, message
):
    code = f"""
    shader main {{
        struct VSOut {{
            vec4 position @ gl_Position;
        }};

        struct HSOut {{
            vec4 position @ gl_Position;
        }};

        {stage_source}
    }}
    """
    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(code)))


def test_tessellation_output_patch_generic_literal_return_type_emits():
    code = """
    shader main {
        struct VSOut {
            vec4 position @ gl_Position;
        };

        tessellation_control {
            OutputPatch<VSOut, 3> makePatch() {
                OutputPatch<VSOut, 3> patch;
                return patch;
            }

            void main() {
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "OutputPatch<VSOut, 3> makePatch()" in generated_code
    assert "OutputPatch<VSOut, 3> patch;" in generated_code
    assert "return patch;" in generated_code
    assert "None" not in generated_code


def test_tessellation_patch_constant_function_uses_hull_context_without_shader_attr():
    code = """
    shader main {
        struct VSOut {
            vec4 position @ gl_Position;
        };

        struct PatchConstants {
            float outer[3] @ gl_TessLevelOuter;
            float inner[1] @ gl_TessLevelInner;
        };

        tessellation_control {
            PatchConstants HSConst(InputPatch<VSOut, 3> inputPatch) {
                PatchConstants patch;
                VSOut first = gl_in[0];
                uint primitive = gl_PrimitiveID;
                patch.outer[0] = first.position.x + float(primitive);
                return patch;
            }

            void main()
                @domain(tri)
                @partitioning(integer)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "PatchConstants HSConst(InputPatch<VSOut, 3> inputPatch, "
        "uint gl_PrimitiveID : SV_PrimitiveID)" in generated_code
    )
    assert "VSOut first = inputPatch[0];" in generated_code
    assert "uint primitive = gl_PrimitiveID;" in generated_code
    assert "gl_in" not in generated_code
    assert generated_code.count('[shader("hull")]') == 1
    assert '[shader("hull")]\nPatchConstants HSConst' not in generated_code


def test_tessellation_patch_constant_function_uses_existing_primitive_alias():
    code = """
    shader main {
        struct VSOut {
            vec4 position @ gl_Position;
        };

        struct PatchConstants {
            float outer[3] @ gl_TessLevelOuter;
            float inner[1] @ gl_TessLevelInner;
        };

        tessellation_control {
            PatchConstants HSConst(
                InputPatch<VSOut, 3> inputPatch,
                uint patchID @ gl_PrimitiveID
            ) {
                PatchConstants patch;
                VSOut first = gl_in[0];
                uint primitive = gl_PrimitiveID;
                patch.outer[0] = first.position.x + float(primitive);
                return patch;
            }

            void main()
                @domain(tri)
                @partitioning(integer)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "PatchConstants HSConst(InputPatch<VSOut, 3> inputPatch, "
        "uint patchID : SV_PrimitiveID)" in generated_code
    )
    assert "uint gl_PrimitiveID : SV_PrimitiveID" not in generated_code
    assert "VSOut first = inputPatch[0];" in generated_code
    assert "uint primitive = patchID;" in generated_code
    assert "gl_in" not in generated_code


def test_tessellation_patch_constant_input_patch_matches_hull_entry_shape():
    code = """
    shader main {
        struct VSOut {
            vec4 position @ gl_Position;
        };

        struct PatchConstants {
            float outer[3] @ gl_TessLevelOuter;
            float inner[1] @ gl_TessLevelInner;
        };

        tessellation_control {
            PatchConstants HSConst(InputPatch<VSOut, 3> constantsPatch) {
                PatchConstants patch;
                VSOut first = gl_in[0];
                patch.outer[0] = first.position.x;
                return patch;
            }

            void main(InputPatch<VSOut, 3> inputPatch)
                @domain(tri)
                @partitioning(integer)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
                VSOut current = inputPatch[gl_InvocationID];
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "PatchConstants HSConst(InputPatch<VSOut, 3> constantsPatch)" in generated_code
    )
    assert (
        "void main(InputPatch<VSOut, 3> inputPatch, "
        "uint gl_InvocationID : SV_OutputControlPointID)" in generated_code
    )
    assert "VSOut first = constantsPatch[0];" in generated_code
    assert "gl_in" not in generated_code


@pytest.mark.parametrize(
    ("constants_source", "domain", "topology", "control_points", "message"),
    [
        (
            "float inner[1] @ gl_TessLevelInner;",
            "tri",
            "triangle_cw",
            3,
            "SV_TessFactor",
        ),
        (
            "vec3 outer @ gl_TessLevelOuter;",
            "tri",
            "triangle_cw",
            3,
            "SV_InsideTessFactor",
        ),
        (
            """
            vec4 outer @ gl_TessLevelOuter;
            float inner @ gl_TessLevelInner;
            """,
            "tri",
            "triangle_cw",
            3,
            "3 SV_TessFactor",
        ),
        (
            """
            vec4 outer @ gl_TessLevelOuter;
            float inner @ gl_TessLevelInner;
            """,
            "quad",
            "triangle_cw",
            4,
            "2 SV_InsideTessFactor",
        ),
        (
            """
            vec2 outer @ gl_TessLevelOuter;
            float inner @ gl_TessLevelInner;
            """,
            "isoline",
            "line",
            2,
            "must not return SV_InsideTessFactor",
        ),
    ],
)
def test_tessellation_patch_constant_factor_semantics_validate_domain_counts(
    constants_source,
    domain,
    topology,
    control_points,
    message,
):
    code = f"""
    shader main {{
        struct VSOut {{
            vec4 position @ gl_Position;
        }};

        struct HSOut {{
            vec4 position @ gl_Position;
        }};

        struct PatchConstants {{
            {constants_source}
        }};

        tessellation_control {{
            PatchConstants HSConst(InputPatch<VSOut, {control_points}> inputPatch) {{
                PatchConstants patch;
                return patch;
            }}

            HSOut main(InputPatch<VSOut, {control_points}> inputPatch)
                @domain({domain})
                @partitioning(integer)
                @outputtopology({topology})
                @outputcontrolpoints({control_points})
                @patchconstantfunc(HSConst) {{
                HSOut output;
                return output;
            }}
        }}
    }}
    """
    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(code)))


@pytest.mark.parametrize(
    ("constants_source", "domain", "topology", "control_points", "message"),
    [
        (
            """
            int outer[3] @ gl_TessLevelOuter;
            float inner[1] @ gl_TessLevelInner;
            """,
            "tri",
            "triangle_cw",
            3,
            "SV_TessFactor member 'outer'.*floating scalar",
        ),
        (
            """
            vec3 outer @ gl_TessLevelOuter;
            uint inner @ gl_TessLevelInner;
            """,
            "tri",
            "triangle_cw",
            3,
            "SV_InsideTessFactor member 'inner'.*floating scalar",
        ),
        (
            "vec2 outer[1] @ gl_TessLevelOuter;",
            "isoline",
            "line",
            2,
            "SV_TessFactor member 'outer'.*floating scalar",
        ),
    ],
)
def test_tessellation_patch_constant_factor_semantics_reject_non_floating_types(
    constants_source,
    domain,
    topology,
    control_points,
    message,
):
    code = f"""
    shader main {{
        struct VSOut {{
            vec4 position @ gl_Position;
        }};

        struct HSOut {{
            vec4 position @ gl_Position;
        }};

        struct PatchConstants {{
            {constants_source}
        }};

        tessellation_control {{
            PatchConstants HSConst(InputPatch<VSOut, {control_points}> inputPatch) {{
                PatchConstants patch;
                return patch;
            }}

            HSOut main(InputPatch<VSOut, {control_points}> inputPatch)
                @domain({domain})
                @partitioning(integer)
                @outputtopology({topology})
                @outputcontrolpoints({control_points})
                @patchconstantfunc(HSConst) {{
                HSOut output;
                return output;
            }}
        }}
    }}
    """
    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(code)))


def test_tessellation_patch_constant_factor_semantics_allow_isoline_outer_only():
    code = """
    shader main {
        struct VSOut {
            vec4 position @ gl_Position;
        };

        struct HSOut {
            vec4 position @ gl_Position;
        };

        struct PatchConstants {
            vec2 outer @ gl_TessLevelOuter;
        };

        tessellation_control {
            PatchConstants HSConst(InputPatch<VSOut, 2> inputPatch) {
                PatchConstants patch;
                return patch;
            }

            HSOut main(InputPatch<VSOut, 2> inputPatch)
                @domain(isoline)
                @partitioning(integer)
                @outputtopology(line)
                @outputcontrolpoints(2)
                @patchconstantfunc(HSConst) {
                HSOut output;
                return output;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "float2 outer : SV_TessFactor;" in generated_code
    assert "SV_InsideTessFactor" not in generated_code


@pytest.mark.parametrize(
    ("patch_constant_source", "message"),
    [
        (
            """
            PatchConstants HSConst(InputPatch<OtherOut, 3> constantsPatch) {
                PatchConstants patch;
                return patch;
            }
            """,
            "InputPatch<OtherOut, 3> must match hull entry InputPatch<VSOut, 3>",
        ),
        (
            """
            PatchConstants HSConst(InputPatch<VSOut, 4> constantsPatch) {
                PatchConstants patch;
                return patch;
            }
            """,
            "InputPatch<VSOut, 4> must match hull entry InputPatch<VSOut, 3>",
        ),
        (
            """
            PatchConstants HSConst(
                InputPatch<VSOut, 3> firstPatch,
                InputPatch<VSOut, 3> secondPatch
            ) {
                PatchConstants patch;
                return patch;
            }
            """,
            "requires at most one InputPatch",
        ),
        (
            """
            void HSConst(InputPatch<VSOut, 3> constantsPatch) {
            }
            """,
            "requires a non-void return type",
        ),
    ],
)
def test_tessellation_patch_constant_invalid_input_patch_shape_raises(
    patch_constant_source, message
):
    code = f"""
    shader main {{
        struct VSOut {{
            vec4 position @ gl_Position;
        }};

        struct OtherOut {{
            vec4 position @ gl_Position;
        }};

        struct PatchConstants {{
            float outer[3] @ gl_TessLevelOuter;
            float inner[1] @ gl_TessLevelInner;
        }};

        tessellation_control {{
            {patch_constant_source}

            void main(InputPatch<VSOut, 3> inputPatch)
                @domain(tri)
                @partitioning(integer)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {{
            }}
        }}
    }}
    """
    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(code)))


def test_tessellation_domain_output_patch_matches_hull_output_shape():
    code = """
    shader main {
        struct VSOut {
            vec4 position @ gl_Position;
        };

        struct HSOut {
            vec4 position @ gl_Position;
        };

        struct PatchConstants {
            float outer[3] @ gl_TessLevelOuter;
            float inner[1] @ gl_TessLevelInner;
        };

        tessellation_control {
            PatchConstants HSConst(InputPatch<VSOut, 3> inputPatch) {
                PatchConstants patch;
                return patch;
            }

            HSOut main(InputPatch<VSOut, 3> inputPatch)
                @domain(tri)
                @partitioning(integer)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
                HSOut output;
                output.position = inputPatch[gl_InvocationID].position;
                return output;
            }
        }

        tessellation_evaluation {
            vec4 main(OutputPatch<HSOut, 3> patch)
                @domain(tri)
                @gl_Position {
                HSOut first = gl_in[0];
                vec3 coord = gl_TessCoord;
                return first.position + vec4(coord, 0.0);
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "HSOut HSMain(InputPatch<VSOut, 3> inputPatch" in generated_code
    assert (
        "float4 DSMain(OutputPatch<HSOut, 3> patch, "
        "float3 gl_TessCoord : SV_DomainLocation) : SV_Position" in generated_code
    )
    assert "HSOut first = patch[0];" in generated_code
    assert "gl_in" not in generated_code


@pytest.mark.parametrize(
    ("domain_source", "message"),
    [
        (
            """
            tessellation_evaluation {
                vec4 main(OutputPatch<OtherOut, 3> patch) @domain(tri) @gl_Position {
                    return patch[0].position;
                }
            }
            """,
            "OutputPatch<OtherOut, 3> must match tessellation_control output "
            "OutputPatch<HSOut, 3>",
        ),
        (
            """
            tessellation_evaluation {
                vec4 main(OutputPatch<HSOut, 4> patch) @domain(tri) @gl_Position {
                    return patch[0].position;
                }
            }
            """,
            "OutputPatch<HSOut, 4> must match",
        ),
        (
            """
            tessellation_evaluation {
                vec4 main(OutputPatch<HSOut, 3> patch) @domain(quad) @gl_Position {
                    return patch[0].position;
                }
            }
            """,
            "domain 'quad' must match tessellation_control domain 'tri'",
        ),
        (
            """
            tessellation_evaluation {
                vec4 main(OutputPatch<HSOut, 3> patch) @gl_Position {
                    return patch[0].position;
                }
            }
            """,
            "tessellation_evaluation stage requires a domain attribute",
        ),
        (
            """
            tessellation_evaluation {
                vec4 main(
                    OutputPatch<HSOut, 3> firstPatch,
                    OutputPatch<HSOut, 3> secondPatch
                ) @domain(tri) @gl_Position {
                    return firstPatch[0].position + secondPatch[0].position;
                }
            }
            """,
            "tessellation_evaluation stage requires at most one OutputPatch",
        ),
    ],
)
def test_tessellation_domain_invalid_output_patch_shape_raises(domain_source, message):
    code = f"""
    shader main {{
        struct VSOut {{
            vec4 position @ gl_Position;
        }};

        struct HSOut {{
            vec4 position @ gl_Position;
        }};

        struct OtherOut {{
            vec4 position @ gl_Position;
        }};

        struct PatchConstants {{
            float outer[3] @ gl_TessLevelOuter;
            float inner[1] @ gl_TessLevelInner;
        }};

        tessellation_control {{
            PatchConstants HSConst(InputPatch<VSOut, 3> inputPatch) {{
                PatchConstants patch;
                return patch;
            }}

            HSOut main(InputPatch<VSOut, 3> inputPatch)
                @domain(tri)
                @partitioning(integer)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {{
                HSOut output;
                return output;
            }}
        }}

        {domain_source}
    }}
    """
    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(code)))


def test_tessellation_domain_attribute_requires_matching_hull_domain():
    code = """
    shader main {
        struct HSOut {
            vec4 position @ gl_Position;
        };

        tessellation_control {
            HSOut main()
                @partitioning(integer)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3) {
                HSOut output;
                return output;
            }
        }

        tessellation_evaluation {
            vec4 main(OutputPatch<HSOut, 3> patch)
                @domain(tri)
                @gl_Position {
                return patch[0].position;
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="tessellation_control stage requires a domain attribute",
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_tessellation_control_output_patch_size_matches_outputcontrolpoints():
    code = """
    shader main {
        struct HSOut {
            vec4 position @ gl_Position;
        };

        struct PatchConstants {
            float outer[3] @ gl_TessLevelOuter;
            float inner[1] @ gl_TessLevelInner;
        };

        tessellation_control {
            PatchConstants HSConst() {
                PatchConstants patch;
                return patch;
            }

            void main(OutputPatch<HSOut, 4> gl_out)
                @domain(tri)
                @partitioning(integer)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
                gl_out[gl_InvocationID].position = vec4(1.0);
            }
        }
    }
    """
    with pytest.raises(
        ValueError,
        match="OutputPatch<HSOut, 4> must match outputcontrolpoints\\(3\\)",
    ):
        generate_code(parse_code(tokenize_code(code)))


@pytest.mark.parametrize(
    ("stage_source", "message"),
    [
        (
            """
            tessellation_control {
                vec4 main() @gl_Position {
                    return vec4(1.0);
                }
            }
            """,
            "tessellation_control stage returns must put semantics",
        ),
        (
            """
            tessellation_control {
                PatchConstants HSConst() @gl_Position {
                    PatchConstants patch;
                    return patch;
                }

                void main()
                    @domain(tri)
                    @partitioning(integer)
                    @outputtopology(triangle_cw)
                    @outputcontrolpoints(3)
                    @patchconstantfunc(HSConst) {
                }
            }
            """,
            "patch constant function returns must put semantics",
        ),
    ],
)
def test_tessellation_control_return_semantics_are_rejected(stage_source, message):
    code = f"""
    shader main {{
        struct PatchConstants {{
            float outer[3] @ gl_TessLevelOuter;
            float inner[1] @ gl_TessLevelInner;
        }};

        {stage_source}
    }}
    """
    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(code)))


@pytest.mark.parametrize(
    ("attribute", "message"),
    [
        ("@patchconstantfunc(Missing)", "does not reference a generated function"),
        ("@patchconstantfunc(HSConst, Other)", "requires exactly one function name"),
    ],
)
def test_tessellation_patchconstantfunc_references_generated_function(
    attribute, message
):
    code = f"""
    shader main {{
        tessellation_control {{
            PatchConstants HSConst() {{
                PatchConstants patch;
                return patch;
            }}

            void main()
                @domain(tri)
                @partitioning(integer)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                {attribute} {{
            }}
        }}
    }}
    """
    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(code)))


@pytest.mark.parametrize(
    "patch_constant_source",
    [
        """
        PatchConstants HSConst() {
            PatchConstants patch;
            uint controlPoint = gl_InvocationID;
            return patch;
        }
        """,
        """
        PatchConstants HSConst(uint controlPoint @ gl_InvocationID) {
            PatchConstants patch;
            return patch;
        }
        """,
    ],
)
def test_tessellation_patch_constant_function_rejects_control_point_id(
    patch_constant_source,
):
    code = f"""
    shader main {{
        struct PatchConstants {{
            float outer[3] @ gl_TessLevelOuter;
            float inner[1] @ gl_TessLevelInner;
        }};

        tessellation_control {{
            {patch_constant_source}

            void main()
                @domain(tri)
                @partitioning(integer)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {{
            }}
        }}
    }}
    """
    with pytest.raises(
        ValueError,
        match="patch constant functions cannot use gl_InvocationID",
    ):
        generate_code(parse_code(tokenize_code(code)))


@pytest.mark.parametrize(
    "stage_source",
    [
        """
        tessellation_control {
            void main() {
                gl_TessLevelOuter[0] = 1.0;
            }
        }
        """,
        """
        tessellation_evaluation {
            void main() {
                float outer = gl_TessLevelOuter[0];
            }
        }
        """,
    ],
)
def test_implicit_tessellation_factor_builtins_raise_clear_error(stage_source):
    code = f"""
    shader main {{
        {stage_source}
    }}
    """
    with pytest.raises(
        ValueError,
        match=(
            "gl_TessLevelOuter and gl_TessLevelInner require an explicit "
            "patch constant function"
        ),
    ):
        generate_code(parse_code(tokenize_code(code)))


def test_mix_builtin_lowers_to_lerp_without_affecting_resources_or_constructors():
    code = """
    shader MixBug {
        sampler2d tex;

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.5);
                float x = mix(0.0, 1.0, 0.25);
                vec4 c = vec4(x, x, x, 1.0);
                vec4 sampled = texture(tex, uv);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float x = lerp(0.0, 1.0, 0.25);" in generated_code
    assert "float4 c = float4(x, x, x, 1.0);" in generated_code
    assert "float4 sampled = tex.Sample(uv);" in generated_code
    assert "mix(" not in generated_code
    assert "texture(" not in generated_code


def test_bool_mix_lowers_to_slang_selectors():
    code = """
    shader BoolMix {
        compute {
            float nextX() {
                return 0.0;
            }

            float nextY() {
                return 1.0;
            }

            bool nextFlag() {
                return true;
            }

            vec3 makeA() {
                return vec3(1.0, 2.0, 3.0);
            }

            vec3 makeB() {
                return vec3(4.0, 5.0, 6.0);
            }

            bvec3 makeMask() {
                return bvec3(true, false, true);
            }

            void main() {
                bool flag = true;
                float literal = mix(0.0, 1.0, flag);
                float complexValue = mix(nextX(), nextY(), nextFlag());
                vec3 a = vec3(1.0, 2.0, 3.0);
                vec3 b = vec3(4.0, 5.0, 6.0);
                bvec3 mask = bvec3(true, false, true);
                vec3 selected = mix(a, b, mask);
                vec3 ternarySelected = mask ? b : a;
                vec3 complexSelected = mix(makeA(), makeB(), makeMask());
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float3 _crossgl_select_bool3_float3("
        "bool3 mask, float3 trueValue, float3 falseValue)" in generated_code
    )
    assert (
        "return float3((mask.x ? trueValue.x : falseValue.x), "
        "(mask.y ? trueValue.y : falseValue.y), "
        "(mask.z ? trueValue.z : falseValue.z));" in generated_code
    )
    assert "float literal = (flag ? 1.0 : 0.0);" in generated_code
    assert "float complexValue = (nextFlag() ? nextY() : nextX());" in generated_code
    assert (
        "float3 selected = _crossgl_select_bool3_float3(mask, b, a);" in generated_code
    )
    assert (
        "float3 ternarySelected = _crossgl_select_bool3_float3(mask, b, a);"
        in generated_code
    )
    assert (
        "float3 complexSelected = "
        "_crossgl_select_bool3_float3(makeMask(), makeB(), makeA());" in generated_code
    )
    assert "lerp(0.0, 1.0, flag)" not in generated_code
    assert "lerp(nextX(), nextY(), nextFlag())" not in generated_code
    assert "lerp(a, b, mask)" not in generated_code
    assert "lerp(makeA(), makeB(), makeMask())" not in generated_code
    assert "(mask ? b : a)" not in generated_code


def test_generated_helper_names_avoid_user_symbol_collisions():
    code = """
    shader HelperCollisions {
        vec3 _crossgl_select_bool3_float3;
        int cgl_textureSize_sampler2D;
        int cgl_textureQueryLevels_sampler2D;
        uint cgl_imageAtomicAdd_original;
        sampler2d colorMap;
        uimage2D counters @r32ui;

        compute {
            void main() {
                vec3 a = vec3(1.0, 2.0, 3.0);
                vec3 b = vec3(4.0, 5.0, 6.0);
                bvec3 mask = bvec3(true, false, true);
                vec3 selected = mix(a, b, mask);
                ivec2 size = textureSize(colorMap, 0);
                int levels = textureQueryLevels(colorMap);
                uint oldValue = imageAtomicAdd(counters, ivec2(0, 0), 1u);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float3 _crossgl_select_bool3_float3;" in generated_code
    assert "int cgl_textureSize_sampler2D;" in generated_code
    assert "int cgl_textureQueryLevels_sampler2D;" in generated_code
    assert "uint cgl_imageAtomicAdd_original;" in generated_code
    assert (
        "float3 _crossgl_select_bool3_float3_1("
        "bool3 mask, float3 trueValue, float3 falseValue)" in generated_code
    )
    assert (
        "int2 cgl_textureSize_sampler2D_1("
        "Sampler2D<float4> tex, uint mipLevel)" in generated_code
    )
    assert (
        "int cgl_textureQueryLevels_sampler2D_1(Sampler2D<float4> tex)"
        in generated_code
    )
    assert (
        "float3 selected = _crossgl_select_bool3_float3_1(mask, b, a);"
        in generated_code
    )
    assert "int2 size = cgl_textureSize_sampler2D_1(colorMap, 0);" in generated_code
    assert (
        "int levels = cgl_textureQueryLevels_sampler2D_1(colorMap);" in generated_code
    )
    assert "uint cgl_imageAtomicAdd_original_1;" in generated_code
    assert (
        "InterlockedAdd(counters[int2(0, 0)], 1u, "
        "cgl_imageAtomicAdd_original_1);" in generated_code
    )
    assert "uint oldValue = cgl_imageAtomicAdd_original_1;" in generated_code
    assert "float3 _crossgl_select_bool3_float3(" not in generated_code
    assert "int2 cgl_textureSize_sampler2D(" not in generated_code
    assert "int cgl_textureQueryLevels_sampler2D(" not in generated_code
    assert generated_code.count("uint cgl_imageAtomicAdd_original;") == 1


def test_lambda_call_emits_slang_lambda_for_explicit_simple_parameters():
    code = """
    shader LambdaShader {
        compute {
            void main() {
                let folded = fold(values, 0, lambda(int acc, int x, (acc + x)));
                let mapped = map(colors, lambda(vec3 color, { return color; }));
                let genericVector = map(colors, lambda(vec3<f32> color, color));
                let always = lambda(true);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "fold(values, 0, (int acc, int x) => (acc + x))" in generated_code
    assert "map(colors, (float3 color) => { return color; })" in generated_code
    assert "map(colors, (float3 color) => color)" in generated_code
    assert "() => true" in generated_code
    assert "lambda(" not in generated_code


def test_lambda_call_preserves_unsupported_slang_lambda_shapes():
    code = """
    shader LambdaFallbackShader {
        compute {
            void main() {
                let mapped = map(values, lambda(x, (x + 1)));
                let generic = map(values, lambda(Result<i32, i32> value, { return value; }));
                let tupled = map(values, lambda((i32, i32) pair, { return pair; }));
                let qualified = map(values, lambda(const int value, value));
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "lambda(x, (x + 1))" in generated_code
    assert "lambda(Result<i32, i32> value, { return value; })" in generated_code
    assert "lambda((i32, i32) pair, { return pair; })" in generated_code
    assert "lambda(const int value, value)" in generated_code
    assert "=>" not in generated_code


def test_function_parameter_modifiers_are_preserved_for_slang():
    code = """
    shader ParameterQualifiers {
        float adjust(
            const float bias,
            in float source,
            out float result,
            inout float accumulated
        ) {
            result = source + bias;
            accumulated = accumulated + result;
            return result;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float adjust(const float bias, in float source, out float result, "
        "inout float accumulated)" in generated_code
    )
    assert (
        "float adjust(float bias, float source, float result, float accumulated)"
        not in (generated_code)
    )


def test_user_defined_texture_function_shadows_resource_lowering():
    code = """
    shader TextureShadow {
        sampler2d tex;

        fragment {
            vec4 texture(sampler2d src, vec2 uv) {
                return vec4(uv, 0.0, 1.0);
            }

            void main() {
                vec2 uv = vec2(0.25, 0.5);
                vec4 color = texture(tex, uv);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float4 texture(Sampler2D<float4> src, float2 uv)" in generated_code
    assert "float4 color = texture(tex, uv);" in generated_code
    assert "float4 color = tex.Sample(uv);" not in generated_code


def test_mod_builtin_lowers_to_slang_fmod():
    code = """
    shader BuiltinGap {
        compute {
            void main() {
                float wrapped = mod(5.0, 2.0);
                vec2 v = vec2(5.0, 7.0);
                vec2 wrappedVec = mod(v, vec2(2.0));
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float wrapped = fmod(5.0, 2.0);" in generated_code
    assert "float2 wrappedVec = fmod(v, float2(2.0));" in generated_code
    assert "float2 v = float2(5.0, 7.0);" in generated_code
    assert "wrapped = mod(" not in generated_code
    assert "wrappedVec = mod(" not in generated_code


def test_fract_builtin_lowers_to_slang_frac():
    code = """
    shader BuiltinGap {
        compute {
            void main() {
                float wrapped = fract(1.25);
                vec2 v = fract(vec2(1.25, 2.5));
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float wrapped = frac(1.25);" in generated_code
    assert "float2 v = frac(float2(1.25, 2.5));" in generated_code
    assert "fract(" not in generated_code


def test_inversesqrt_builtin_lowers_to_slang_rsqrt():
    code = """
    shader BuiltinGap {
        compute {
            void main() {
                float inv = inversesqrt(x);
                vec2 invVec = inversesqrt(vec2(4.0, 9.0));
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float inv = rsqrt(x);" in generated_code
    assert "float2 invVec = rsqrt(float2(4.0, 9.0));" in generated_code
    assert "inversesqrt(" not in generated_code


def test_saturate_builtin_lowers_to_slang_clamp():
    code = """
    shader BuiltinGap {
        compute {
            void main() {
                float saturated = saturate(1.25);
                vec2 saturatedVec = saturate(vec2(-1.0, 2.0));
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float saturated = clamp(1.25, 0.0, 1.0);" in generated_code
    assert "float2 saturatedVec = clamp(float2(-1.0, 2.0), 0.0, 1.0);" in generated_code
    assert "saturate(" not in generated_code


def test_user_defined_saturate_function_is_not_lowered_to_clamp():
    code = """
    shader BuiltinGap {
        compute {
            float saturate(float x) {
                return x + 1.0;
            }

            void main() {
                float adjusted = saturate(0.5);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float saturate(float x)" in generated_code
    assert "float adjusted = saturate(0.5);" in generated_code
    assert "float adjusted = clamp(0.5, 0.0, 1.0);" not in generated_code


def test_user_defined_mix_function_is_not_lowered_to_lerp():
    code = """
    shader BuiltinGap {
        compute {
            float mix(float x, float y, float t) {
                return x + y + t;
            }

            void main() {
                float adjusted = mix(0.0, 1.0, 0.25);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float mix(float x, float y, float t)" in generated_code
    assert "float adjusted = mix(0.0, 1.0, 0.25);" in generated_code
    assert "float adjusted = lerp(0.0, 1.0, 0.25);" not in generated_code


def test_workgroup_barrier_lowers_to_slang_group_sync():
    code = """
    shader BarrierGap {
        compute {
            void main() {
                workgroupBarrier();
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "GroupMemoryBarrierWithGroupSync();" in generated_code
    assert "workgroupBarrier();" not in generated_code


def test_user_defined_workgroup_barrier_is_not_lowered_to_group_sync():
    code = """
    shader BarrierGap {
        compute {
            void workgroupBarrier() {
                return;
            }

            void main() {
                workgroupBarrier();
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "void workgroupBarrier()" in generated_code
    assert "workgroupBarrier();" in generated_code
    assert "GroupMemoryBarrierWithGroupSync();" not in generated_code


def test_threadgroup_memory_qualifiers_emit_slang_groupshared():
    code = """
    shader SharedMemoryQualifiers {
        compute {
            void main() {
                groupshared float scratch[64];
                shared uint bins[4];
                threadgroup vec4 cachedColor[2];
                workgroup int flags[8];
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "groupshared float scratch[64];" in generated_code
    assert "groupshared uint bins[4];" in generated_code
    assert "groupshared float4 cachedColor[2];" in generated_code
    assert "groupshared int flags[8];" in generated_code
    assert "\n    float scratch[64];" not in generated_code
    assert "\n    uint bins[4];" not in generated_code
    assert "\n    float4 cachedColor[2];" not in generated_code
    assert "\n    int flags[8];" not in generated_code


def test_wave_intrinsics_lower_to_native_slang_calls():
    code = """
    shader SlangWaveIntrinsics {
        compute {
            void main() {
                uint lane = WaveGetLaneIndex();
                uint count = WaveGetLaneCount();
                bool first = WaveIsFirstLane();
                uint sumValue = WaveActiveSum(lane);
                uint productValue = WaveActiveProduct(lane + 1u);
                uint minValue = WaveActiveMin(sumValue);
                uint maxValue = WaveActiveMax(productValue);
                uint andValue = WaveActiveBitAnd(maxValue);
                uint orValue = WaveActiveBitOr(andValue);
                uint xorValue = WaveActiveBitXor(orValue);
                uint prefixSum = WavePrefixSum(xorValue);
                uint prefixProduct = WavePrefixProduct(lane + 1u);
                bool anyLane = WaveActiveAnyTrue(prefixSum > 0u);
                bool allLane = WaveActiveAllTrue(prefixProduct > 0u);
                uvec4 ballot = WaveActiveBallot(anyLane);
                uvec4 comparisonBallot = WaveActiveBallot(lane != 0u);
                uint broadcast = WaveReadLaneAt(prefixSum, 0u);
                uint firstValue = WaveReadLaneFirst(broadcast);
                uint quadX = QuadReadAcrossX(firstValue);
                uint quadY = QuadReadAcrossY(quadX);
                uint quadDiagonal = QuadReadAcrossDiagonal(quadY);
                uint quadLane = QuadReadLaneAt(quadDiagonal, 0u);
                uvec4 matchMask = WaveMatch(lane);
                uint multiSum = WaveMultiPrefixSum(lane, matchMask);
                uint multiProduct = WaveMultiPrefixProduct(lane + 1u, matchMask);
                uint multiAnd = WaveMultiPrefixBitAnd(multiProduct, matchMask);
                uint multiOr = WaveMultiPrefixBitOr(multiAnd, matchMask);
                uint multiXor = WaveMultiPrefixBitXor(multiOr, matchMask);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "uint lane = WaveGetLaneIndex();" in generated_code
    assert "uint count = WaveGetLaneCount();" in generated_code
    assert "bool first = WaveIsFirstLane();" in generated_code
    assert "uint sumValue = WaveActiveSum(lane);" in generated_code
    assert "uint productValue = WaveActiveProduct(lane + 1u);" in generated_code
    assert "uint minValue = WaveActiveMin(sumValue);" in generated_code
    assert "uint maxValue = WaveActiveMax(productValue);" in generated_code
    assert "uint andValue = WaveActiveBitAnd(maxValue);" in generated_code
    assert "uint orValue = WaveActiveBitOr(andValue);" in generated_code
    assert "uint xorValue = WaveActiveBitXor(orValue);" in generated_code
    assert "uint prefixSum = WavePrefixSum(xorValue);" in generated_code
    assert "uint prefixProduct = WavePrefixProduct(lane + 1u);" in generated_code
    assert "bool anyLane = WaveActiveAnyTrue(prefixSum > 0u);" in generated_code
    assert "bool allLane = WaveActiveAllTrue(prefixProduct > 0u);" in generated_code
    assert "uint4 ballot = WaveActiveBallot(anyLane);" in generated_code
    assert "uint4 comparisonBallot = WaveActiveBallot(lane != 0u);" in generated_code
    assert "uint broadcast = WaveReadLaneAt(prefixSum, 0u);" in generated_code
    assert "uint firstValue = WaveReadLaneFirst(broadcast);" in generated_code
    assert "uint quadX = QuadReadAcrossX(firstValue);" in generated_code
    assert "uint quadY = QuadReadAcrossY(quadX);" in generated_code
    assert "uint quadDiagonal = QuadReadAcrossDiagonal(quadY);" in generated_code
    assert "uint quadLane = QuadReadLaneAt(quadDiagonal, 0u);" in generated_code
    assert "uint4 matchMask = WaveMatch(lane);" in generated_code
    assert "uint multiSum = WaveMultiPrefixSum(lane, matchMask);" in generated_code
    assert (
        "uint multiProduct = WaveMultiPrefixProduct(lane + 1u, matchMask);"
        in generated_code
    )
    assert "uint multiAnd = WaveMultiPrefixBitAnd(multiProduct, matchMask);" in (
        generated_code
    )
    assert "uint multiOr = WaveMultiPrefixBitOr(multiAnd, matchMask);" in (
        generated_code
    )
    assert "uint multiXor = WaveMultiPrefixBitXor(multiOr, matchMask);" in (
        generated_code
    )
    assert "WaveOpNode" not in generated_code
    assert "unsupported Slang wave intrinsic" not in generated_code


def test_wave_intrinsics_compile_with_slangc_if_available(tmp_path):
    code = """
    shader SlangWaveCompileSmoke {
        compute {
            void main() {
                uint lane = WaveGetLaneIndex();
                uint sumValue = WaveActiveSum(lane);
                bool anyLane = WaveActiveAnyTrue(sumValue > 0u);
                uvec4 ballot = WaveActiveBallot(anyLane);
                uint broadcast = WaveReadLaneAt(sumValue, 0u);
                uint firstValue = WaveReadLaneFirst(broadcast);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    compile_generated_slang(generated_code, tmp_path, "compute")


def test_wave_intrinsic_invalid_arities_emit_slang_diagnostics():
    code = """
    shader InvalidSlangWaveIntrinsics {
        compute {
            void main() {
                uint lane = WaveGetLaneIndex(1u);
                bool first = WaveIsFirstLane(false);
                uvec4 ballot = WaveActiveBallot();
                uint multi = WaveMultiPrefixSum(lane);
                uint quad = QuadReadLaneAt(lane);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "uint lane = /* unsupported Slang wave intrinsic: WaveGetLaneIndex "
        "expects 0 arguments, got 1 */ 0;" in generated_code
    )
    assert (
        "bool first = /* unsupported Slang wave intrinsic: WaveIsFirstLane "
        "expects 0 arguments, got 1 */ false;" in generated_code
    )
    assert (
        "uint4 ballot = /* unsupported Slang wave intrinsic: WaveActiveBallot "
        "expects 1 arguments, got 0 */ uint4(0);" in generated_code
    )
    assert (
        "uint multi = /* unsupported Slang wave intrinsic: WaveMultiPrefixSum "
        "expects 2 arguments, got 1 */ 0;" in generated_code
    )
    assert (
        "uint quad = /* unsupported Slang wave intrinsic: QuadReadLaneAt "
        "expects 2 arguments, got 1 */ 0;" in generated_code
    )
    assert "WaveOpNode" not in generated_code


def test_wave_vote_intrinsics_validate_predicate_types():
    code = """
    shader InvalidSlangWaveVotePredicates {
        compute {
            void main() {
                uint value = 1u;
                bvec2 vectorMask = bvec2(true, false);
                uvec2 lanes = uvec2(value, value);
                bool badAnyValue = WaveActiveAnyTrue(value);
                bool badAllVector = WaveActiveAllTrue(vectorMask);
                uvec4 badBallotVectorComparison =
                    WaveActiveBallot(lanes == uvec2(1u, 2u));
                uvec4 badBallotNumeric = WaveActiveBallot(WaveActiveSum(value));
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "bool badAnyValue = /* unsupported Slang wave intrinsic: "
        "WaveActiveAnyTrue predicate must be scalar bool, got uint */ false;"
        in generated_code
    )
    assert (
        "bool badAllVector = /* unsupported Slang wave intrinsic: "
        "WaveActiveAllTrue predicate must be scalar bool, got bool2 */ false;"
        in generated_code
    )
    assert (
        "uint4 badBallotVectorComparison = /* unsupported Slang wave intrinsic: "
        "WaveActiveBallot predicate must be scalar bool, got bool2 */ uint4(0);"
        in generated_code
    )
    assert (
        "uint4 badBallotNumeric = /* unsupported Slang wave intrinsic: "
        "WaveActiveBallot predicate must be scalar bool, got uint */ uint4(0);"
        in generated_code
    )
    assert "WaveActiveAnyTrue(value)" not in generated_code
    assert "WaveActiveAllTrue(vectorMask)" not in generated_code
    assert "WaveActiveBallot(lanes == uint2(1u, 2u))" not in generated_code
    assert "WaveActiveBallot(WaveActiveSum(value))" not in generated_code
    assert "WaveOpNode" not in generated_code


def test_wave_intrinsic_invalid_argument_types_emit_slang_diagnostics():
    code = """
    shader InvalidSlangWaveArgumentTypes {
        compute {
            void main() {
                uint value = 1u;
                uvec4 mask = uvec4(1u, 0u, 0u, 0u);
                uint badLane = WaveReadLaneAt(value, vec2(0.0, 1.0));
                uint badQuadLane = QuadReadLaneAt(value, false);
                uint badPartition = WaveMultiPrefixSum(value, vec3(1.0, 0.0, 0.0));
                uint badPartitionCall = WaveMultiPrefixSum(value, WaveActiveSum(value));
                uint badBitValue = WaveMultiPrefixBitAnd(false, mask);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "uint badLane = /* unsupported Slang wave intrinsic: WaveReadLaneAt "
        "lane index must be scalar int or uint, got float2 */ 0;" in generated_code
    )
    assert (
        "uint badQuadLane = /* unsupported Slang wave intrinsic: QuadReadLaneAt "
        "lane index must be scalar int or uint, got bool */ 0;" in generated_code
    )
    assert (
        "uint badPartition = /* unsupported Slang wave intrinsic: "
        "WaveMultiPrefixSum partition mask must be uint4, got float3 */ 0;"
        in generated_code
    )
    assert (
        "uint badPartitionCall = /* unsupported Slang wave intrinsic: "
        "WaveMultiPrefixSum partition mask must be uint4, got uint */ 0;"
        in generated_code
    )
    assert (
        "uint badBitValue = /* unsupported Slang wave intrinsic: "
        "WaveMultiPrefixBitAnd value must be scalar or vector int or uint, "
        "got bool */ 0;" in generated_code
    )
    assert "WaveReadLaneAt(value, float2(0.0, 1.0))" not in generated_code
    assert "QuadReadLaneAt(value, false)" not in generated_code
    assert "WaveMultiPrefixSum(value, float3(1.0, 0.0, 0.0))" not in generated_code
    assert "WaveOpNode" not in generated_code


def test_floating_binary_modulo_lowers_to_slang_fmod():
    code = """
    shader BinaryModuloGap {
        compute {
            void main() {
                float a = 5.0;
                float wrapped = a % 2.0;
                a %= 3.0;
                vec2 v = vec2(5.0, 7.0);
                vec2 wrappedVec = v % vec2(2.0);
                v %= vec2(3.0);
                double d = 5.0;
                double wrappedDouble = d % 2.0;
                d %= 3.0;
                int i = 5 % 2;
                i %= 2;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float wrapped = fmod(a, 2.0);" in generated_code
    assert "a = fmod(a, 3.0);" in generated_code
    assert "float2 wrappedVec = fmod(v, float2(2.0));" in generated_code
    assert "v = fmod(v, float2(3.0));" in generated_code
    assert "double wrappedDouble = fmod(d, 2.0);" in generated_code
    assert "d = fmod(d, 3.0);" in generated_code
    assert "int i = 5 % 2;" in generated_code
    assert "i %= 2;" in generated_code
    assert "float wrapped = a % 2.0;" not in generated_code
    assert "float2 wrappedVec = v % float2(2.0);" not in generated_code
    assert "double wrappedDouble = d % 2.0;" not in generated_code
    assert "a %= 3.0;" not in generated_code
    assert "v %= float2(3.0);" not in generated_code
    assert "d %= 3.0;" not in generated_code


def test_multiple_stage_entry_points_emit():
    code = """
    shader main {
        vertex {
            void main() {
            }
        }
        fragment {
            void main() {
            }
        }
        compute {
            void main() {
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert '[shader("vertex")]' in generated_code
    assert '[shader("fragment")]' in generated_code
    assert '[shader("compute")]' in generated_code


def test_duplicate_stage_entry_names_emit_unique_slang_functions():
    code = """
    shader main {
        vertex {
            float VSMain(float x) {
                return x;
            }

            void main() {
                float x = VSMain(1.0);
            }
        }
        fragment {
            void main() {
            }
        }
        compute {
            void main() {
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "float VSMain(float x)" in generated_code
    assert "void VSMain_1()" in generated_code
    assert "void PSMain()" in generated_code
    assert "void CSMain()" in generated_code
    assert generated_code.count("void main()") == 0


def test_advanced_stage_attributes_emit_slang_shader_names():
    code = """
    shader advanced {
        geometry {
            void main() {
            }
        }
        tessellation_control {
            void main() {
            }
        }
        tessellation_evaluation {
            void main() {
            }
        }
        mesh {
            void main() {
            }
        }
        task {
            void main() {
            }
        }
        ray_generation {
            void main() {
            }
        }
        ray_closest_hit {
            void main() {
            }
        }
        ray_any_hit {
            void main() {
            }
        }
        ray_miss {
            void main() {
            }
        }
        ray_callable {
            void main() {
            }
        }
        ray_intersection {
            void main() {
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    expected_shader_attributes = [
        '[shader("geometry")]',
        '[shader("hull")]',
        '[shader("domain")]',
        '[shader("mesh")]',
        '[shader("amplification")]',
        '[shader("raygeneration")]',
        '[shader("closesthit")]',
        '[shader("anyhit")]',
        '[shader("miss")]',
        '[shader("callable")]',
        '[shader("intersection")]',
    ]
    for attribute in expected_shader_attributes:
        assert attribute in generated_code

    raw_crossgl_stage_attributes = [
        '[shader("tessellation_control")]',
        '[shader("tessellation_evaluation")]',
        '[shader("task")]',
        '[shader("ray_generation")]',
        '[shader("ray_closest_hit")]',
        '[shader("ray_any_hit")]',
        '[shader("ray_miss")]',
        '[shader("ray_callable")]',
        '[shader("ray_intersection")]',
    ]
    for attribute in raw_crossgl_stage_attributes:
        assert attribute not in generated_code


def test_compute_stage_emits_default_numthreads():
    code = """
    shader main {
        compute {
            void main() {
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "[numthreads(1, 1, 1)]" in generated_code
    assert '[numthreads(1, 1, 1)]\n[shader("compute")]' in generated_code


def test_compute_stage_uses_execution_layout_numthreads():
    code = """
    shader main {
        compute {
            layout(local_size_x = 8, local_size_y = 4, local_size_z = 2) in;
            void main() {
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "[numthreads(8, 4, 2)]" in generated_code
    assert "[numthreads(1, 1, 1)]" not in generated_code


def test_mesh_and_task_stages_emit_execution_layout_numthreads():
    code = """
    shader MeshTask {
        task {
            layout(local_size_x = 32, local_size_y = 2, local_size_z = 1) in;
            void main() {
            }
        }
        mesh {
            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
            void main() {
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert '[numthreads(32, 2, 1)]\n[shader("amplification")]' in generated_code
    assert '[numthreads(64, 1, 1)]\n[shader("mesh")]' in generated_code


def test_mesh_and_task_intrinsics_lower_to_native_slang_calls():
    code = """
    shader SlangMeshTaskIntrinsics {
        task {
            void main() {
                DispatchMesh(2, 3, 4);
            }
        }
        mesh {
            void main() @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert '[shader("amplification")]' in generated_code
    assert '[shader("mesh")]' in generated_code
    assert "void ASMain()" in generated_code
    assert "void MSMain()" in generated_code
    assert "DispatchMesh(2, 3, 4);" in generated_code
    assert "SetMeshOutputCounts(3, 1);" in generated_code
    assert "MeshOpNode" not in generated_code
    assert "unsupported Slang mesh intrinsic" not in generated_code


def test_mesh_payload_parameter_lowers_to_native_slang_payload_qualifier():
    code = """
    shader SlangMeshPayload {
        struct MeshPayload {
            uint meshlet;
        };

        groupshared MeshPayload payload;

        task {
            void main() @numthreads(1, 1, 1) {
                payload.meshlet = 7u;
                DispatchMesh(1, 1, 1, payload);
            }
        }

        mesh {
            void main(
                @mesh_payload in MeshPayload payload
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
                uint meshlet = payload.meshlet;
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "groupshared MeshPayload payload;" in generated_code
    assert "DispatchMesh(1, 1, 1, payload);" in generated_code
    assert "void MSMain(in payload MeshPayload payload)" in generated_code
    assert ": mesh_payload" not in generated_code
    assert "MeshOpNode" not in generated_code
    assert "unsupported Slang mesh intrinsic" not in generated_code


def test_mesh_output_signature_and_helper_roles_lower_to_native_slang():
    code = """
    shader SlangMeshOutputRoles {
        struct MeshVertex {
            vec4 position @ SV_Position;
            vec2 uv @ TEXCOORD0;
        };

        struct MeshPrimitive {
            bool culled @ SV_CullPrimitive;
            uint layer @ SV_RenderTargetArrayIndex;
        };

        void writeVertex(
            @vertices inout MeshVertex verts[3],
            uint idx
        ) {
            verts[idx].position = vec4(0.0, 0.0, 0.0, 1.0);
            verts[idx].uv = vec2(0.5, 0.25);
        }

        void writePrimitive(
            @indices out uvec3 tris[1],
            @primitives out MeshPrimitive prims[1]
        ) {
            tris[0] = uvec3(0u, 1u, 2u);
            prims[0].culled = false;
            prims[0].layer = 0u;
        }

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1],
                @primitives out MeshPrimitive prims[1]
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
                writeVertex(verts, 0u);
                writePrimitive(tris, prims);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "void writeVertex(inout MeshVertex verts[3], uint idx)" in generated_code
    assert (
        "void writePrimitive(out uint3 tris[1], out MeshPrimitive prims[1])"
        in generated_code
    )
    assert (
        "void main(out vertices MeshVertex verts[3], "
        "out indices uint3 tris[1], "
        "out primitives MeshPrimitive prims[1])" in generated_code
    )
    assert "verts[idx].position = float4(0.0, 0.0, 0.0, 1.0);" in generated_code
    assert "verts[idx].uv = float2(0.5, 0.25);" in generated_code
    assert "tris[0] = uint3(0u, 1u, 2u);" in generated_code
    assert "prims[0].culled = false;" in generated_code
    assert "prims[0].layer = 0u;" in generated_code
    assert ": vertices" not in generated_code
    assert ": indices" not in generated_code
    assert ": primitives" not in generated_code


@pytest.mark.parametrize(
    "parameter_source",
    [
        "@vertices MeshVertex verts[3]",
        "@vertices in MeshVertex verts[3]",
        "@vertices inout MeshVertex verts[3]",
        "@vertices const MeshVertex verts[3]",
    ],
)
def test_mesh_output_entry_parameters_require_out_qualifier(parameter_source):
    code = f"""
    shader InvalidSlangMeshOutputRole {{
        struct MeshVertex {{
            vec4 position @ SV_Position;
        }};

        mesh {{
            void main(
                {parameter_source},
                @indices out uvec3 tris[1]
            ) @numthreads(1, 1, 1) @outputtopology(triangle) {{
                SetMeshOutputCounts(1, 1);
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match="must use the out qualifier"):
        generate_code(parse_code(tokenize_code(code)))


def test_mesh_output_entry_parameters_reject_non_mesh_stages():
    code = """
    shader InvalidSlangMeshOutputRoleStage {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        compute {
            void main(@vertices out MeshVertex verts[1]) @numthreads(1, 1, 1) {
            }
        }
    }
    """

    with pytest.raises(ValueError, match="compute stage cannot declare mesh vertices"):
        generate_code(parse_code(tokenize_code(code)))


def test_mesh_output_signature_validates_roles_counts_and_topology():
    missing_vertices_code = """
    shader MissingSlangMeshVertices {
        mesh {
            void main(@indices out uvec3 tris[1])
                @numthreads(1, 1, 1)
                @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="out vertices"):
        generate_code(parse_code(tokenize_code(missing_vertices_code)))

    duplicate_indices_code = """
    shader DuplicateSlangMeshIndices {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1],
                @indices out uvec3 otherTris[1]
            ) @numthreads(1, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="at most one indices"):
        generate_code(parse_code(tokenize_code(duplicate_indices_code)))

    unsized_vertices_code = """
    shader UnsizedSlangMeshVertices {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[],
                @indices out uvec3 tris[1]
            ) @numthreads(1, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="vertices.*static array size"):
        generate_code(parse_code(tokenize_code(unsized_vertices_code)))

    wrong_index_type_code = """
    shader WrongSlangMeshIndexType {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec2 tris[1]
            ) @numthreads(1, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="triangle.*uint3"):
        generate_code(parse_code(tokenize_code(wrong_index_type_code)))

    mismatched_primitives_code = """
    shader MismatchedSlangMeshPrimitiveCount {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        struct MeshPrimitive {
            bool culled @ SV_CullPrimitive;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[2],
                @primitives out MeshPrimitive prims[1]
            ) @numthreads(1, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="primitives.*match.*indices"):
        generate_code(parse_code(tokenize_code(mismatched_primitives_code)))

    missing_counts_code = """
    shader MissingSlangMeshOutputCounts {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(1, 1, 1) @outputtopology(triangle) {
                tris[0] = uvec3(0u, 1u, 2u);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="SetMeshOutputCounts exactly once"):
        generate_code(parse_code(tokenize_code(missing_counts_code)))

    duplicate_counts_code = """
    shader DuplicateSlangMeshOutputCounts {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(1, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
                SetMeshOutputCounts(3, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="SetMeshOutputCounts exactly once"):
        generate_code(parse_code(tokenize_code(duplicate_counts_code)))

    malformed_counts_code = """
    shader MalformedSlangMeshOutputCounts {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(1, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="requires exactly two arguments"):
        generate_code(parse_code(tokenize_code(malformed_counts_code)))

    oversized_vertex_count_code = """
    shader OversizedSlangMeshVertexCount {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[2],
                @indices out uvec3 tris[1]
            ) @numthreads(1, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="vertex count exceeds"):
        generate_code(parse_code(tokenize_code(oversized_vertex_count_code)))

    oversized_primitive_count_code = """
    shader OversizedSlangMeshPrimitiveCount {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(1, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 2);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="primitive count exceeds.*indices"):
        generate_code(parse_code(tokenize_code(oversized_primitive_count_code)))


def test_mesh_output_signature_validates_write_order_and_literal_indices():
    valid_code = """
    shader ValidSlangMeshOutputWrites {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(1, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
                tris[0] = uvec3(0u, 1u, 2u);
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(valid_code)))
    assert "SetMeshOutputCounts(3, 1);" in generated_code
    assert "verts[0].position = float4(0.0, 0.0, 0.0, 1.0);" in generated_code
    assert "tris[0] = uint3(0u, 1u, 2u);" in generated_code

    write_before_counts_code = """
    shader EarlySlangMeshOutputWrite {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(1, 1, 1) @outputtopology(triangle) {
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
                SetMeshOutputCounts(3, 1);
                tris[0] = uvec3(0u, 1u, 2u);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="writes must occur after"):
        generate_code(parse_code(tokenize_code(write_before_counts_code)))

    helper_before_counts_code = """
    shader EarlySlangMeshOutputHelper {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        void writeVertex(@vertices inout MeshVertex verts[3]) {
            verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
        }

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(1, 1, 1) @outputtopology(triangle) {
                writeVertex(verts);
                SetMeshOutputCounts(3, 1);
                tris[0] = uvec3(0u, 1u, 2u);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="helper calls before SetMeshOutputCounts"):
        generate_code(parse_code(tokenize_code(helper_before_counts_code)))

    vertex_index_code = """
    shader OversizedSlangMeshVertexWrite {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(1, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
                verts[3].position = vec4(0.0, 0.0, 0.0, 1.0);
                tris[0] = uvec3(0u, 1u, 2u);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="vertices output literal index"):
        generate_code(parse_code(tokenize_code(vertex_index_code)))

    index_array_code = """
    shader OversizedSlangMeshIndexWrite {
        struct MeshVertex {
            vec4 position @ SV_Position;
        };

        mesh {
            void main(
                @vertices out MeshVertex verts[3],
                @indices out uvec3 tris[1]
            ) @numthreads(1, 1, 1) @outputtopology(triangle) {
                SetMeshOutputCounts(3, 1);
                verts[0].position = vec4(0.0, 0.0, 0.0, 1.0);
                tris[1] = uvec3(0u, 1u, 2u);
            }
        }
    }
    """
    with pytest.raises(ValueError, match="indices output literal index"):
        generate_code(parse_code(tokenize_code(index_array_code)))


@pytest.mark.parametrize(
    ("parameter_source", "message"),
    [
        (
            "@mesh_payload out MeshPayload payload",
            "mesh payload parameter 'payload' must use the in qualifier",
        ),
        (
            "@mesh_payload inout MeshPayload payload",
            "mesh payload parameter 'payload' must use the in qualifier",
        ),
        (
            "@mesh_payload in float payload",
            "mesh payload parameter 'payload' must use a user-defined struct type",
        ),
    ],
)
def test_mesh_payload_parameter_rejects_invalid_slang_shapes(parameter_source, message):
    code = f"""
    shader InvalidSlangMeshPayload {{
        struct MeshPayload {{
            uint meshlet;
        }};

        mesh {{
            void main(
                {parameter_source}
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {{
                SetMeshOutputCounts(3, 1);
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(code)))


@pytest.mark.parametrize(
    ("payload_source", "mesh_parameter_source", "message"),
    [
        (
            "MeshPayload payload;",
            "",
            "DispatchMesh payload argument requires a mesh stage @mesh_payload parameter",
        ),
        (
            "OtherPayload payload;",
            "@mesh_payload in MeshPayload payload",
            "DispatchMesh payload argument type OtherPayload must match mesh payload type MeshPayload",
        ),
    ],
)
def test_dispatch_mesh_payload_argument_matches_mesh_payload_parameter(
    payload_source, mesh_parameter_source, message
):
    code = f"""
    shader InvalidSlangDispatchMeshPayload {{
        struct MeshPayload {{
            uint meshlet;
        }};
        struct OtherPayload {{
            uint meshlet;
        }};

        groupshared {payload_source}

        task {{
            void main() @numthreads(1, 1, 1) {{
                DispatchMesh(1, 1, 1, payload);
            }}
        }}

        mesh {{
            void main(
                {mesh_parameter_source}
            ) @numthreads(32, 1, 1) @outputtopology(triangle) {{
                SetMeshOutputCounts(3, 1);
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(code)))


def test_mesh_intrinsic_invalid_arities_emit_slang_diagnostics():
    code = """
    shader InvalidSlangMeshIntrinsics {
        mesh {
            void main() @outputtopology(triangle) {
                SetMeshOutputCounts(1);
                DispatchMesh(1, 1);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "/* unsupported Slang mesh intrinsic: SetMeshOutputCounts "
        "expects 2 arguments, got 1 */ 0;" in generated_code
    )
    assert (
        "/* unsupported Slang mesh intrinsic: DispatchMesh "
        "expects 3 or 4 arguments, got 2 */ 0;" in generated_code
    )
    assert "MeshOpNode" not in generated_code


def test_set_mesh_output_counts_validates_count_argument_types():
    code = """
    shader InvalidSlangMeshOutputCountTypes {
        mesh {
            void main() @outputtopology(triangle) {
                uint vertexCount = 3u;
                bool useMeshlet = true;
                vec2 primitiveCounts = vec2(1.0, 2.0);
                SetMeshOutputCounts(vertexCount, 1u);
                SetMeshOutputCounts(useMeshlet, 1u);
                SetMeshOutputCounts(3u, primitiveCounts);
                SetMeshOutputCounts(1.5, 1u);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "SetMeshOutputCounts(vertexCount, 1u);" in generated_code
    assert (
        "/* unsupported Slang mesh intrinsic: SetMeshOutputCounts vertex count "
        "must be scalar int or uint, got bool */ 0;" in generated_code
    )
    assert (
        "/* unsupported Slang mesh intrinsic: SetMeshOutputCounts primitive count "
        "must be scalar int or uint, got float2 */ 0;" in generated_code
    )
    assert (
        "/* unsupported Slang mesh intrinsic: SetMeshOutputCounts vertex count "
        "must be scalar int or uint, got float */ 0;" in generated_code
    )
    assert "SetMeshOutputCounts(useMeshlet, 1u)" not in generated_code
    assert "SetMeshOutputCounts(3u, primitiveCounts)" not in generated_code
    assert "SetMeshOutputCounts(1.5, 1u)" not in generated_code
    assert "MeshOpNode" not in generated_code


def test_dispatch_mesh_validates_thread_count_argument_types():
    code = """
    shader InvalidSlangDispatchMeshCountTypes {
        task {
            void main() {
                uint dispatchX = 2u;
                bool enabled = true;
                float yCount = 1.5;
                vec2 zCounts = vec2(1.0, 2.0);
                DispatchMesh(dispatchX, 1u, 1u);
                DispatchMesh(enabled, 1u, 1u);
                DispatchMesh(1u, yCount, 1u);
                DispatchMesh(1u, 1u, zCounts);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "DispatchMesh(dispatchX, 1u, 1u);" in generated_code
    assert (
        "/* unsupported Slang mesh intrinsic: DispatchMesh x count "
        "must be scalar int or uint, got bool */ 0;" in generated_code
    )
    assert (
        "/* unsupported Slang mesh intrinsic: DispatchMesh y count "
        "must be scalar int or uint, got float */ 0;" in generated_code
    )
    assert (
        "/* unsupported Slang mesh intrinsic: DispatchMesh z count "
        "must be scalar int or uint, got float2 */ 0;" in generated_code
    )
    assert "DispatchMesh(enabled, 1u, 1u)" not in generated_code
    assert "DispatchMesh(1u, yCount, 1u)" not in generated_code
    assert "DispatchMesh(1u, 1u, zCounts)" not in generated_code
    assert "MeshOpNode" not in generated_code


@pytest.mark.parametrize(
    ("stage_source", "message"),
    [
        (
            "compute { void main() { SetMeshOutputCounts(1, 1); } }",
            "compute stage cannot call SetMeshOutputCounts",
        ),
        (
            "mesh { void main() @outputtopology(triangle) { DispatchMesh(1, 1, 1); } }",
            "mesh stage cannot call DispatchMesh",
        ),
    ],
)
def test_mesh_intrinsics_reject_invalid_slang_stages(stage_source, message):
    code = f"""
    shader InvalidSlangMeshIntrinsicStage {{
        {stage_source}
    }}
    """

    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(code)))


def test_stage_numthreads_attribute_overrides_default_numthreads():
    code = """
    shader StageNumthreads {
        compute {
            void main() @numthreads(8, 4, 2) {
            }
        }
        task {
            void main() @numthreads(32, 1, 1) {
            }
        }
        mesh {
            void main() @numthreads(64, 2, 1) {
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert '[numthreads(8, 4, 2)]\n[shader("compute")]' in generated_code
    assert '[numthreads(32, 1, 1)]\n[shader("amplification")]' in generated_code
    assert '[numthreads(64, 2, 1)]\n[shader("mesh")]' in generated_code
    assert "[numthreads(1, 1, 1)]" not in generated_code
    assert " : numthreads" not in generated_code


def test_stage_numthreads_attribute_accepts_default_dimensions():
    code = """
    shader StageNumthreadsDefaults {
        compute {
            void main() @numthreads(8) {
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert '[numthreads(8, 1, 1)]\n[shader("compute")]' in generated_code


def test_stage_attributes_lower_to_slang_attributes_not_return_semantics():
    code = """
    shader StageAttributes {
        struct PatchConstants {
            float outer[3] @ gl_TessLevelOuter;
            float inner[1] @ gl_TessLevelInner;
        };

        geometry {
            void main() @maxvertexcount(3) {
            }
        }
        tessellation_control {
            PatchConstants HSConst() {
                PatchConstants patch;
                return patch;
            }

            void main()
                @domain(triangle)
                @partitioning(fractional_odd)
                @outputtopology(triangle_cw)
                @outputcontrolpoints(3)
                @patchconstantfunc(HSConst) {
            }
        }
        tessellation_evaluation {
            void main() @domain(triangle) {
            }
        }
        mesh {
            void main() @outputtopology(triangle) {
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "[maxvertexcount(3)]" in generated_code
    assert '[domain("tri")]' in generated_code
    assert '[partitioning("fractional_odd")]' in generated_code
    assert '[outputtopology("triangle_cw")]' in generated_code
    assert "[outputcontrolpoints(3)]" in generated_code
    assert '[patchconstantfunc("HSConst")]' in generated_code
    assert '[outputtopology("triangle")]' in generated_code
    assert '[domain("triangle")]' not in generated_code
    assert " : maxvertexcount" not in generated_code
    assert " : domain" not in generated_code
    assert " : outputtopology" not in generated_code


def test_semantics_map_to_slang_system_values():
    code = """
    shader SemanticShader {
        struct VSOut {
            vec4 position @ gl_Position;
            vec4 color @ TEXCOORD0;
        };

        vertex {
            VSOut main(uint vertexId @ gl_VertexID) {
                VSOut out;
                return out;
            }
        }

        fragment {
            vec4 main(VSOut input) @ gl_FragColor {
                return input.color;
            }
        }

        compute {
            void main(uvec3 tid @ gl_GlobalInvocationID) {
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "float4 position : SV_Position;" in generated_code
    assert "float4 color : TEXCOORD0;" in generated_code
    assert "VSOut VSMain(uint vertexId : SV_VertexID)" in generated_code
    assert "float4 PSMain(VSOut input) : SV_Target" in generated_code
    assert "void CSMain(uint3 tid : SV_DispatchThreadID)" in generated_code
    assert ": gl_Position" not in generated_code
    assert ": gl_FragColor" not in generated_code
    assert ": gl_GlobalInvocationID" not in generated_code


@pytest.mark.parametrize(
    ("shader_type", "semantic", "message"),
    [
        (
            "fragment",
            "gl_FragColor",
            "fragment stage void return cannot use return semantic gl_FragColor",
        ),
        (
            None,
            "COLOR",
            "void function return cannot use return semantic COLOR",
        ),
    ],
)
def test_void_return_semantics_raise_for_slang(shader_type, semantic, message):
    function = FunctionNode(
        name="main",
        return_type=PrimitiveType("void"),
        parameters=[],
        body=BlockNode([]),
    )
    function.semantic = semantic

    with pytest.raises(ValueError, match=message):
        SlangCodeGen().generate_function(function, shader_type=shader_type)


@pytest.mark.parametrize("metadata", ["compute", "workgroup_size"])
def test_stage_marker_metadata_is_not_treated_as_slang_return_semantic(metadata):
    function = FunctionNode(
        name="kernel",
        return_type=PrimitiveType("void"),
        parameters=[],
        body=BlockNode([]),
    )
    function.semantic = metadata

    generated_code = SlangCodeGen().generate_function(function)

    assert "void kernel()" in generated_code
    assert f" : {metadata}" not in generated_code


def test_ray_stage_builtins_lower_to_slang_intrinsics():
    code = """
    shader RayBuiltins {
        ray_generation {
            void main() {
                uvec3 launch = gl_LaunchIDEXT;
                uint launchX = gl_LaunchIDEXT.x;
                uvec3 launchSize = gl_LaunchSizeEXT;
            }
        }

        ray_closest_hit {
            void main() {
                vec3 worldRay = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT;
                vec3 objectRay = gl_ObjectRayOriginEXT + gl_ObjectRayDirectionEXT;
                float range = gl_RayTmaxEXT - gl_RayTminEXT + gl_HitTEXT;
                uint hitKind = gl_HitKindEXT;
                uint customInstance = gl_InstanceCustomIndexEXT;
                uint instance = gl_InstanceID;
                uint primitive = gl_PrimitiveID;
                uint geometryIndex = gl_GeometryIndexEXT;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "uint3 launch = DispatchRaysIndex();" in generated_code
    assert "uint launchX = DispatchRaysIndex().x;" in generated_code
    assert "uint3 launchSize = DispatchRaysDimensions();" in generated_code
    assert "float3 worldRay = WorldRayOrigin() + WorldRayDirection();" in generated_code
    assert (
        "float3 objectRay = ObjectRayOrigin() + ObjectRayDirection();" in generated_code
    )
    assert "float range = RayTCurrent() - RayTMin() + RayTCurrent();" in generated_code
    assert "uint hitKind = HitKind();" in generated_code
    assert "uint customInstance = InstanceIndex();" in generated_code
    assert "uint instance = InstanceID();" in generated_code
    assert "uint primitive = PrimitiveIndex();" in generated_code
    assert "uint geometryIndex = GeometryIndex();" in generated_code
    assert "gl_LaunchIDEXT" not in generated_code
    assert "gl_LaunchSizeEXT" not in generated_code
    assert "gl_WorldRayOriginEXT" not in generated_code
    assert "gl_InstanceCustomIndexEXT" not in generated_code
    assert "gl_PrimitiveID" not in generated_code
    assert "gl_GeometryIndexEXT" not in generated_code
    assert ": SV_" not in generated_code


def test_ray_stage_intrinsic_builtins_respect_local_shadowing():
    code = """
    shader RayBuiltinShadowing {
        ray_miss {
            void main() {
                uint gl_HitKindEXT = 7;
                uint hitKind = gl_HitKindEXT;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "uint gl_HitKindEXT = 7;" in generated_code
    assert "uint hitKind = gl_HitKindEXT;" in generated_code
    assert "HitKind()" not in generated_code


def test_ray_stage_semantic_parameters_emit_slang_directions():
    code = """
    shader RayStageSemantics {
        struct RayPayload {
            vec3 color;
        };

        struct HitAttributes {
            vec2 barycentrics;
        };

        struct CallableData {
            uint value;
        };

        ray_closest_hit {
            void main(
                RayPayload payload @ payload,
                HitAttributes attributes @ hit_attribute
            ) {
                payload.color = vec3(attributes.barycentrics, 1.0);
            }
        }

        ray_miss {
            void main(RayPayload payload @ rayPayloadInEXT) {
                payload.color = vec3(0.0, 0.0, 0.0);
            }
        }

        ray_callable {
            void main(CallableData data @ callableDataInEXT) {
                data.value = 1u;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "void ClosestHitMain(inout RayPayload payload, "
        "in HitAttributes attributes)" in generated_code
    )
    assert "void MissMain(inout RayPayload payload)" in generated_code
    assert "void CallableMain(inout CallableData data)" in generated_code
    assert "payload.color = float3(attributes.barycentrics, 1.0);" in generated_code
    assert "payload.color = float3(0.0, 0.0, 0.0);" in generated_code
    assert "data.value = 1u;" in generated_code
    assert " : payload" not in generated_code
    assert " : hit_attribute" not in generated_code
    assert "rayPayloadInEXT" not in generated_code
    assert "callableDataInEXT" not in generated_code


def test_ray_stage_local_interface_variables_become_slang_parameters():
    code = """
    shader RayStageLocalInterfaces {
        struct RayPayload {
            vec3 color;
        };

        struct CallableData {
            uint value;
        };

        ray_generation {
            layout(location = 0) @rayPayloadEXT RayPayload rayPayload;
            layout(location = 1) @callableDataEXT CallableData callableData;

            void main() {
                rayPayload.color = vec3(1.0, 0.0, 0.0);
                callableData.value = 7u;
            }
        }

        ray_closest_hit {
            layout(location = 0) @rayPayloadInEXT RayPayload closestPayload;
            @hitAttributeEXT vec2 hitAttributes;

            void main() {
                closestPayload.color = vec3(hitAttributes, 1.0);
            }
        }

        ray_miss {
            @rayPayloadInEXT RayPayload missPayload;

            void main() {
                missPayload.color = vec3(0.0, 0.0, 0.0);
            }
        }

        ray_callable {
            @callableDataInEXT CallableData callableInput;

            void main() {
                callableInput.value = 1u;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "void RayGenMain(inout RayPayload rayPayload, "
        "inout CallableData callableData)" in generated_code
    )
    assert (
        "void ClosestHitMain(inout RayPayload closestPayload, "
        "in float2 hitAttributes)" in generated_code
    )
    assert "void MissMain(inout RayPayload missPayload)" in generated_code
    assert "void CallableMain(inout CallableData callableInput)" in generated_code
    assert "rayPayload.color = float3(1.0, 0.0, 0.0);" in generated_code
    assert "callableData.value = 7u;" in generated_code
    assert "closestPayload.color = float3(hitAttributes, 1.0);" in generated_code
    assert "missPayload.color = float3(0.0, 0.0, 0.0);" in generated_code
    assert "callableInput.value = 1u;" in generated_code
    assert "RayPayload rayPayload;" not in generated_code
    assert "CallableData callableData;" not in generated_code
    assert "RayPayload closestPayload;" not in generated_code
    assert "float2 hitAttributes;" not in generated_code
    assert "rayPayloadEXT" not in generated_code
    assert "rayPayloadInEXT" not in generated_code
    assert "hitAttributeEXT" not in generated_code
    assert "callableDataInEXT" not in generated_code


def test_ray_stage_builtin_triangle_hit_attributes_emit_slang_input_parameter():
    code = """
    shader RayBuiltinHitAttributes {
        struct RayPayload {
            vec3 color;
        };

        ray_closest_hit {
            void main(
                RayPayload payload @ payload,
                BuiltInTriangleIntersectionAttributes attributes @ hit_attribute
            ) {
                payload.color = vec3(attributes.barycentrics, 1.0);
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "void main(inout RayPayload payload, "
        "in BuiltInTriangleIntersectionAttributes attributes)" in generated_code
    )
    assert " : hit_attribute" not in generated_code


def test_slang_ray_tracing_intrinsics_emit_native_calls_and_validate_shapes():
    code = """
    shader SlangRayIntrinsics {
        accelerationStructureEXT scene;

        struct RayPayload {
            vec3 color;
        };

        struct CallableData {
            uint value;
        };

        struct HitAttributes {
            vec2 barycentrics;
        };

        ray_generation {
            void main() {
                RayDesc ray;
                RayPayload payload;
                CallableData callableData;
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
                TraceRay(
                    scene,
                    0,
                    0xFF,
                    0,
                    1,
                    0,
                    vec3(0.0, 0.0, 0.0),
                    0.0,
                    vec3(0.0, 0.0, 1.0),
                    100.0,
                    payload
                );
                CallShader(1, callableData);
            }
        }

        ray_intersection {
            void main() {
                HitAttributes attributes;
                bool accepted = ReportHit(1.0, 0, attributes);
                bool acceptedWithoutAttributes = ReportHit(2.0, 1);
            }
        }

        ray_any_hit {
            void main(RayPayload payload @ payload, HitAttributes attributes @ hit_attribute) {
                IgnoreHit();
                AcceptHitAndEndSearch();
            }
        }

        ray_callable {
            void main(CallableData data @ callableDataInEXT) {
                data.value = data.value + 1u;
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "[[vk::binding(0, 0)]] RaytracingAccelerationStructure scene "
        ": register(t0);" in generated_code
    )
    assert "TraceRay(scene, 0, 255, 0, 1, 0, ray, payload);" in generated_code
    assert (
        "TraceRay(scene, 0, 255, 0, 1, 0, "
        "RayDesc(float3(0.0, 0.0, 0.0), 0.0, "
        "float3(0.0, 0.0, 1.0), 100.0), payload);"
    ) in generated_code
    assert "CallShader(1, callableData);" in generated_code
    assert "bool accepted = ReportHit(1.0, 0, attributes);" in generated_code
    assert "bool acceptedWithoutAttributes = ReportHit(2.0, 1);" in generated_code
    assert "IgnoreHit();" in generated_code
    assert "AcceptHitAndEndSearch();" in generated_code
    assert "void CallableMain(inout CallableData data)" in generated_code
    assert "data.value = data.value + 1u;" in generated_code
    assert "RayTracingOpNode" not in generated_code
    assert "accelerationStructureEXT" not in generated_code


def test_slang_report_hit_helper_return_attributes_emit_native_calls():
    code = """
    shader SlangReportHitHelperReturnAttributes {
        struct RayPayload {
            vec3 color;
        };

        struct HitAttributes {
            vec2 barycentrics;
        };

        HitAttributes makeAttributes(float u) {
            HitAttributes attributes;
            attributes.barycentrics = vec2(u, 1.0 - u);
            return attributes;
        }

        HitAttributes relayAttributes() {
            return makeAttributes(0.25);
        }

        ray_intersection {
            void main() {
                bool directAccepted = ReportHit(1.0, 0, makeAttributes(0.5));
                bool relayAccepted = ReportHit(2.0, 1, relayAttributes());
            }
        }

        ray_closest_hit {
            void main(
                RayPayload payload @ payload,
                HitAttributes attributes @ hit_attribute
            ) { }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "HitAttributes makeAttributes(float u)" in generated_code
    assert "attributes.barycentrics = float2(u, 1.0 - u);" in generated_code
    assert "HitAttributes relayAttributes()" in generated_code
    assert "return makeAttributes(0.25);" in generated_code
    assert (
        "bool directAccepted = ReportHit(1.0, 0, makeAttributes(0.5));"
        in generated_code
    )
    assert (
        "bool relayAccepted = ReportHit(2.0, 1, relayAttributes());" in generated_code
    )


def test_slang_ray_tracing_intrinsic_helpers_validate_in_stage_context():
    code = """
    shader SlangRayIntrinsicHelpers {
        accelerationStructureEXT scene;

        struct RayPayload {
            vec3 color;
        };

        struct CallableData {
            uint value;
        };

        struct HitAttributes {
            vec2 barycentrics;
        };

        void traceScene(
            RaytracingAccelerationStructure accelerationStructure,
            RayDesc ray,
            inout RayPayload payload
        ) {
            TraceRay(accelerationStructure, 0, 0xFF, 0, 1, 0, ray, payload);
        }

        void invokeCallable(uint shaderIndex, inout CallableData data) {
            CallShader(shaderIndex, data);
        }

        bool reportHit(HitAttributes attributes) {
            return ReportHit(1.0, 0, attributes);
        }

        HitAttributes makeAttributes() {
            HitAttributes attributes;
            return attributes;
        }

        ray_generation {
            void main() {
                RayDesc ray;
                RayPayload payload;
                CallableData data;
                traceScene(scene, ray, payload);
                invokeCallable(1, data);
            }
        }

        ray_intersection {
            void main() {
                HitAttributes attributes;
                bool accepted = reportHit(attributes);
                bool helperReturnAccepted = reportHit(makeAttributes());
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "void traceScene(RaytracingAccelerationStructure accelerationStructure, "
        "RayDesc ray, inout RayPayload payload)" in generated_code
    )
    assert (
        "TraceRay(accelerationStructure, 0, 255, 0, 1, 0, ray, payload);"
        in generated_code
    )
    assert (
        "void invokeCallable(uint shaderIndex, inout CallableData data)"
        in generated_code
    )
    assert "CallShader(shaderIndex, data);" in generated_code
    assert "bool reportHit(HitAttributes attributes)" in generated_code
    assert "return ReportHit(1.0, 0, attributes);" in generated_code
    assert "HitAttributes makeAttributes()" in generated_code
    assert "traceScene(scene, ray, payload);" in generated_code
    assert "invokeCallable(1, data);" in generated_code
    assert "bool accepted = reportHit(attributes);" in generated_code
    assert "bool helperReturnAccepted = reportHit(makeAttributes());" in generated_code


def test_slang_ray_tracing_interface_member_lvalues_emit_native_calls():
    code = """
    shader SlangRayInterfaceLvalues {
        accelerationStructureEXT scene;

        struct RayPayload {
            vec3 color;
        };

        struct CallableData {
            uint value;
        };

        struct HitAttributes {
            vec2 barycentrics;
        };

        struct RayPayloadHolder {
            RayPayload primary;
            RayPayload payloads[2];
        };

        struct CallableDataHolder {
            CallableData primary;
            CallableData items[2];
        };

        struct HitAttributeHolder {
            HitAttributes primary;
            HitAttributes items[2];
        };

        ray_generation {
            void main() {
                RayDesc ray;
                RayPayloadHolder rays;
                CallableDataHolder callables;
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, rays.primary);
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, rays.payloads[1]);
                CallShader(0, callables.primary);
                CallShader(1, callables.items[0]);
            }
        }

        ray_intersection {
            void main() {
                HitAttributeHolder hits;
                bool memberAccepted = ReportHit(1.0, 0, hits.primary);
                bool indexedAccepted = ReportHit(2.0, 1, hits.items[1]);
            }
        }

        ray_miss {
            void main(RayPayload payload @ rayPayloadInEXT) { }
        }

        ray_closest_hit {
            void main(RayPayload payload @ payload, HitAttributes attributes @ hit_attribute) { }
        }

        ray_callable {
            void main(CallableData data @ callableDataInEXT) { }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "TraceRay(scene, 0, 255, 0, 1, 0, ray, rays.primary);" in generated_code
    assert "TraceRay(scene, 0, 255, 0, 1, 0, ray, rays.payloads[1]);" in generated_code
    assert "CallShader(0, callables.primary);" in generated_code
    assert "CallShader(1, callables.items[0]);" in generated_code
    assert "bool memberAccepted = ReportHit(1.0, 0, hits.primary);" in generated_code
    assert "bool indexedAccepted = ReportHit(2.0, 1, hits.items[1]);" in generated_code


def test_slang_ray_tracing_interface_pointer_and_reference_aliases_emit_native_calls():
    code = """
    shader SlangRayInterfacePointerReferenceAliases {
        accelerationStructureEXT scene;

        struct RayPayload {
            vec3 color;
        };

        struct CallableData {
            uint value;
        };

        struct HitAttributes {
            vec2 barycentrics;
        };

        struct RayPayloadHolder {
            RayPayload primary;
            RayPayload payloads[2];
        };

        struct CallableDataHolder {
            CallableData primary;
            CallableData items[2];
        };

        struct HitAttributeHolder {
            HitAttributes primary;
            HitAttributes items[2];
        };

        ray_generation {
            void main() {
                RayDesc ray;
                RayPayload payload;
                RayPayload& payloadRef = payload;
                RayPayloadHolder rays;
                RayPayloadHolder* rayPtr = &rays;
                RayPayloadHolder& rayRef = rays;
                CallableData callableData;
                CallableData& callableDataRef = callableData;
                CallableDataHolder callables;
                CallableDataHolder* callablePtr = &callables;
                CallableDataHolder& callableRef = callables;
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, rayPtr->primary);
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, rayRef.payloads[1]);
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payloadRef);
                CallShader(0, callablePtr->primary);
                CallShader(1, callableRef.items[0]);
                CallShader(2, callableDataRef);
            }
        }

        ray_intersection {
            void main() {
                HitAttributes attributes;
                HitAttributes& attributesRef = attributes;
                HitAttributeHolder hits;
                HitAttributeHolder* hitPtr = &hits;
                HitAttributeHolder& hitRef = hits;
                bool pointerAccepted = ReportHit(1.0, 0, hitPtr->primary);
                bool indexedReferenceAccepted = ReportHit(2.0, 1, hitRef.items[1]);
                bool directReferenceAccepted = ReportHit(3.0, 2, attributesRef);
            }
        }

        ray_miss {
            void main(RayPayload payload @ rayPayloadInEXT) { }
        }

        ray_closest_hit {
            void main(RayPayload payload @ payload, HitAttributes attributes @ hit_attribute) { }
        }

        ray_callable {
            void main(CallableData data @ callableDataInEXT) { }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "RayPayload& payloadRef = payload;" in generated_code
    assert "RayPayloadHolder* rayPtr = &rays;" in generated_code
    assert "RayPayloadHolder& rayRef = rays;" in generated_code
    assert "CallableData& callableDataRef = callableData;" in generated_code
    assert "CallableDataHolder* callablePtr = &callables;" in generated_code
    assert "CallableDataHolder& callableRef = callables;" in generated_code
    assert "HitAttributes& attributesRef = attributes;" in generated_code
    assert "HitAttributeHolder* hitPtr = &hits;" in generated_code
    assert "HitAttributeHolder& hitRef = hits;" in generated_code
    assert "TraceRay(scene, 0, 255, 0, 1, 0, ray, rayPtr->primary);" in generated_code
    assert (
        "TraceRay(scene, 0, 255, 0, 1, 0, ray, rayRef.payloads[1]);" in generated_code
    )
    assert "TraceRay(scene, 0, 255, 0, 1, 0, ray, payloadRef);" in generated_code
    assert "CallShader(0, callablePtr->primary);" in generated_code
    assert "CallShader(1, callableRef.items[0]);" in generated_code
    assert "CallShader(2, callableDataRef);" in generated_code
    assert (
        "bool pointerAccepted = ReportHit(1.0, 0, hitPtr->primary);" in generated_code
    )
    assert (
        "bool indexedReferenceAccepted = ReportHit(2.0, 1, hitRef.items[1]);"
        in generated_code
    )
    assert (
        "bool directReferenceAccepted = ReportHit(3.0, 2, attributesRef);"
        in generated_code
    )


def test_slang_ray_tracing_helper_pointer_holder_aliases_emit_native_calls():
    code = """
    shader SlangRayPointerHolderHelpers {
        RaytracingAccelerationStructure scene;

        struct RayPayload {
            vec3 color;
        };

        struct CallableData {
            uint value;
        };

        struct HitAttributes {
            vec2 barycentrics;
        };

        struct RayPayloadHolder {
            RayPayload primary;
            RayPayload payloads[2];
        };

        struct CallableDataHolder {
            CallableData primary;
            CallableData items[2];
        };

        struct HitAttributeHolder {
            HitAttributes primary;
            HitAttributes items[2];
        };

        void traceHolder(
            RaytracingAccelerationStructure accelerationStructure,
            RayDesc ray,
            RayPayloadHolder* rays
        ) {
            RayPayload& alias = rays->primary;
            TraceRay(accelerationStructure, 0, 0xFF, 0, 1, 0, ray, alias);
            TraceRay(accelerationStructure, 0, 0xFF, 0, 1, 0, ray, rays->payloads[1]);
        }

        void invokeHolder(uint shaderIndex, CallableDataHolder* callables) {
            CallableData& alias = callables->items[0];
            CallShader(shaderIndex, alias);
            CallShader(shaderIndex + 1u, callables->primary);
        }

        bool reportHolder(HitAttributeHolder* hits) {
            HitAttributes& alias = hits->items[1];
            return ReportHit(1.0, 0, alias) || ReportHit(2.0, 1, hits->primary);
        }

        ray_generation {
            void main() {
                RayDesc ray;
                RayPayloadHolder rays;
                RayPayloadHolder* rayPtr = &rays;
                CallableDataHolder callables;
                CallableDataHolder* callablePtr = &callables;
                traceHolder(scene, ray, rayPtr);
                invokeHolder(0, callablePtr);
            }
        }

        ray_intersection {
            void main() {
                HitAttributeHolder hits;
                HitAttributeHolder* hitPtr = &hits;
                bool accepted = reportHolder(hitPtr);
            }
        }

        ray_miss {
            void main(RayPayload payload @ rayPayloadInEXT) { }
        }

        ray_closest_hit {
            void main(RayPayload payload @ payload, HitAttributes attributes @ hit_attribute) { }
        }

        ray_callable {
            void main(CallableData data @ callableDataInEXT) { }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "void traceHolder(RaytracingAccelerationStructure accelerationStructure, "
        "RayDesc ray, RayPayloadHolder* rays)" in generated_code
    )
    assert "RayPayload& alias = rays->primary;" in generated_code
    assert (
        "TraceRay(accelerationStructure, 0, 255, 0, 1, 0, ray, alias);"
        in generated_code
    )
    expected_indexed_trace = "".join(
        [
            "TraceRay(accelerationStructure, 0, 255, 0, 1, 0, ray, ",
            "rays->payloads[1]);",
        ]
    )
    assert expected_indexed_trace in generated_code
    assert (
        "void invokeHolder(uint shaderIndex, CallableDataHolder* callables)"
        in generated_code
    )
    assert "CallableData& alias = callables->items[0];" in generated_code
    assert "CallShader(shaderIndex, alias);" in generated_code
    assert "CallShader(shaderIndex + 1u, callables->primary);" in generated_code
    assert "bool reportHolder(HitAttributeHolder* hits)" in generated_code
    assert "HitAttributes& alias = hits->items[1];" in generated_code
    assert (
        "return ReportHit(1.0, 0, alias) || ReportHit(2.0, 1, hits->primary);"
        in generated_code
    )
    assert "traceHolder(scene, ray, rayPtr);" in generated_code
    assert "invokeHolder(0, callablePtr);" in generated_code
    assert "bool accepted = reportHolder(hitPtr);" in generated_code
    assert "RayTracingOpNode" not in generated_code


def test_slang_ray_tracing_helper_address_of_pointer_relays_emit_native_calls():
    code = """
    shader SlangRayAddressOfPointerRelays {
        RaytracingAccelerationStructure scene;

        struct RayPayload {
            vec3 color;
        };

        struct CallableData {
            uint value;
        };

        struct HitAttributes {
            vec2 barycentrics;
        };

        struct RayPayloadHolder {
            RayPayload primary;
        };

        struct CallableDataHolder {
            CallableData primary;
        };

        struct HitAttributeHolder {
            HitAttributes primary;
        };

        void traceLeaf(
            RaytracingAccelerationStructure accelerationStructure,
            RayDesc ray,
            inout RayPayload payload
        ) {
            TraceRay(accelerationStructure, 0, 0xFF, 0, 1, 0, ray, payload);
        }

        void tracePointerRelay(
            RaytracingAccelerationStructure accelerationStructure,
            RayDesc ray,
            RayPayloadHolder* rays
        ) {
            traceLeaf(accelerationStructure, ray, rays->primary);
        }

        void callableLeaf(uint shaderIndex, inout CallableData data) {
            CallShader(shaderIndex, data);
        }

        void callablePointerRelay(
            uint shaderIndex,
            CallableDataHolder* callables
        ) {
            callableLeaf(shaderIndex, callables->primary);
        }

        bool reportPointerRelay(HitAttributeHolder* hits) {
            return ReportHit(1.0, 0, hits->primary);
        }

        ray_generation {
            void main() {
                RayDesc ray;
                RayPayloadHolder rays;
                CallableDataHolder callables;
                tracePointerRelay(scene, ray, &rays);
                callablePointerRelay(2, &callables);
            }
        }

        ray_intersection {
            void main() {
                HitAttributeHolder hits;
                bool accepted = reportPointerRelay(&hits);
            }
        }

        ray_miss {
            void main(RayPayload payload @ rayPayloadInEXT) { }
        }

        ray_closest_hit {
            void main(RayPayload payload @ payload, HitAttributes attributes @ hit_attribute) { }
        }

        ray_callable {
            void main(CallableData data @ callableDataInEXT) { }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "tracePointerRelay(scene, ray, &rays);" in generated_code
    assert "callablePointerRelay(2, &callables);" in generated_code
    assert "bool accepted = reportPointerRelay(&hits);" in generated_code
    assert "traceLeaf(accelerationStructure, ray, rays->primary);" in generated_code
    assert "callableLeaf(shaderIndex, callables->primary);" in generated_code
    assert "return ReportHit(1.0, 0, hits->primary);" in generated_code
    assert "RayTracingOpNode" not in generated_code


def test_slang_nested_ray_interface_relay_helpers_preserve_lvalues():
    code = """
    shader SlangRayNestedRelayHelpers {
        RaytracingAccelerationStructure scene;

        struct RayPayload {
            vec3 color;
        };

        struct CallableData {
            uint value;
        };

        struct RayPayloadHolder {
            RayPayload primary;
            RayPayload payloads[2];
        };

        struct CallableDataHolder {
            CallableData primary;
            CallableData items[2];
        };

        void traceLeaf(
            RaytracingAccelerationStructure accelerationStructure,
            RayDesc ray,
            inout RayPayload payload
        ) {
            TraceRay(accelerationStructure, 0, 0xFF, 0, 1, 0, ray, payload);
        }

        void traceReferenceRelay(
            RaytracingAccelerationStructure accelerationStructure,
            RayDesc ray,
            RayPayloadHolder& rays
        ) {
            traceLeaf(accelerationStructure, ray, rays.payloads[1]);
        }

        void tracePointerRelay(
            RaytracingAccelerationStructure accelerationStructure,
            RayDesc ray,
            RayPayloadHolder* rays
        ) {
            traceLeaf(accelerationStructure, ray, rays->primary);
        }

        void callableLeaf(uint shaderIndex, inout CallableData data) {
            CallShader(shaderIndex, data);
        }

        void callableReferenceRelay(uint shaderIndex, CallableData& data) {
            callableLeaf(shaderIndex, data);
        }

        void callablePointerRelay(
            uint shaderIndex,
            CallableDataHolder* callables
        ) {
            callableLeaf(shaderIndex + 1u, callables->items[0]);
        }

        ray_generation {
            void main() {
                RayDesc ray;
                RayPayloadHolder rays;
                RayPayloadHolder& rayRef = rays;
                RayPayloadHolder* rayPtr = &rays;
                CallableData data;
                CallableData& dataRef = data;
                CallableDataHolder callables;
                CallableDataHolder* callablePtr = &callables;
                traceReferenceRelay(scene, ray, rayRef);
                tracePointerRelay(scene, ray, rayPtr);
                callableReferenceRelay(0, dataRef);
                callablePointerRelay(1, callablePtr);
            }
        }

        ray_miss {
            void main(RayPayload payload @ rayPayloadInEXT) { }
        }

        ray_callable {
            void main(CallableData data @ callableDataInEXT) { }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "void traceReferenceRelay("
        "RaytracingAccelerationStructure accelerationStructure, "
        "RayDesc ray, RayPayloadHolder& rays)" in generated_code
    )
    assert "traceLeaf(accelerationStructure, ray, rays.payloads[1]);" in generated_code
    assert (
        "void tracePointerRelay("
        "RaytracingAccelerationStructure accelerationStructure, "
        "RayDesc ray, RayPayloadHolder* rays)" in generated_code
    )
    assert "traceLeaf(accelerationStructure, ray, rays->primary);" in generated_code
    assert (
        "void callableReferenceRelay(uint shaderIndex, CallableData& data)"
        in generated_code
    )
    assert "callableLeaf(shaderIndex, data);" in generated_code
    assert (
        "void callablePointerRelay(uint shaderIndex, CallableDataHolder* callables)"
        in generated_code
    )
    assert "callableLeaf(shaderIndex + 1u, callables->items[0]);" in generated_code
    assert "traceReferenceRelay(scene, ray, rayRef);" in generated_code
    assert "tracePointerRelay(scene, ray, rayPtr);" in generated_code
    assert "callableReferenceRelay(0, dataRef);" in generated_code
    assert "callablePointerRelay(1, callablePtr);" in generated_code
    assert "RayTracingOpNode" not in generated_code


def test_slang_ray_query_methods_emit_native_calls_and_infer_results():
    code = """
    shader SlangRayQuery {
        RaytracingAccelerationStructure scene;

        compute {
            void main() {
                RayDesc ray;
                RayQuery<RAY_FLAG_NONE> rq;
                rq.TraceRayInline(scene, 0, 0xFF, ray);
                let advanced = rq.Proceed();
                RayQuery<RAY_FLAG_NONE> indexedQueries[2];
                let indexedAdvanced = indexedQueries[0].Proceed();
                let candidateType = rq.CandidateType();
                let origin = rq.CandidateObjectRayOrigin();
                let barycentrics = rq.CandidateTriangleBarycentrics();
                let objectToWorld = rq.CommittedObjectToWorld3x4();
                let committedT = rq.CommittedRayT();
                rq.CommitProceduralPrimitiveHit(1.0);
                rq.CommitNonOpaqueTriangleHit();
                rq.Abort();
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "[[vk::binding(0, 0)]] RaytracingAccelerationStructure scene "
        ": register(t0);" in generated_code
    )
    assert "RayQuery<RAY_FLAG_NONE> rq;" in generated_code
    assert "rq.TraceRayInline(scene, 0, 255, ray);" in generated_code
    assert "bool advanced = rq.Proceed();" in generated_code
    assert "RayQuery<RAY_FLAG_NONE> indexedQueries[2];" in generated_code
    assert "bool indexedAdvanced = indexedQueries[0].Proceed();" in generated_code
    assert "uint candidateType = rq.CandidateType();" in generated_code
    assert "float3 origin = rq.CandidateObjectRayOrigin();" in generated_code
    assert "float2 barycentrics = rq.CandidateTriangleBarycentrics();" in generated_code
    assert "float3x4 objectToWorld = rq.CommittedObjectToWorld3x4();" in generated_code
    assert "float committedT = rq.CommittedRayT();" in generated_code
    assert "rq.CommitProceduralPrimitiveHit(1.0);" in generated_code
    assert "rq.CommitNonOpaqueTriangleHit();" in generated_code
    assert "rq.Abort();" in generated_code
    assert "RayQueryOpNode" not in generated_code


def test_slang_ray_query_reference_and_member_receivers_emit_native_calls():
    code = """
    shader SlangRayQueryAliases {
        struct QueryHolder {
            RayQuery<RAY_FLAG_NONE> query;
            RayQuery<RAY_FLAG_NONE> queries[2];
        }
        RaytracingAccelerationStructure scene;

        compute {
            void main() {
                RayDesc ray;
                RayQuery<RAY_FLAG_NONE> rq;
                RayQuery<RAY_FLAG_NONE>& rqRef = rq;
                rqRef.TraceRayInline(scene, 0, 0xFF, ray);
                let refAdvanced = rqRef.Proceed();
                rqRef.CommitProceduralPrimitiveHit(1.0);
                QueryHolder holder;
                holder.query.TraceRayInline(scene, 0, 0xFF, ray);
                let memberAdvanced = holder.query.Proceed();
                let indexedCommittedT = holder.queries[1].CommittedRayT();
                QueryHolder* holderPtr = &holder;
                let pointerMemberType = holderPtr->query.CandidateType();
                holderPtr->query.CommitNonOpaqueTriangleHit();
                RayQuery<RAY_FLAG_NONE>& indexedRef = holder.queries[0];
                let indexedRefBarycentrics = indexedRef.CandidateTriangleBarycentrics();
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "RayQuery<RAY_FLAG_NONE>& rqRef = rq;" in generated_code
    assert "rqRef.TraceRayInline(scene, 0, 255, ray);" in generated_code
    assert "bool refAdvanced = rqRef.Proceed();" in generated_code
    assert "rqRef.CommitProceduralPrimitiveHit(1.0);" in generated_code
    assert "holder.query.TraceRayInline(scene, 0, 255, ray);" in generated_code
    assert "bool memberAdvanced = holder.query.Proceed();" in generated_code
    assert (
        "float indexedCommittedT = holder.queries[1].CommittedRayT();" in generated_code
    )
    assert "QueryHolder* holderPtr = &holder;" in generated_code
    assert (
        "uint pointerMemberType = holderPtr->query.CandidateType();" in generated_code
    )
    assert "holderPtr->query.CommitNonOpaqueTriangleHit();" in generated_code
    assert "RayQuery<RAY_FLAG_NONE>& indexedRef = holder.queries[0];" in generated_code
    assert (
        "float2 indexedRefBarycentrics = "
        "indexedRef.CandidateTriangleBarycentrics();" in generated_code
    )
    assert "RayQueryOpNode" not in generated_code


def test_slang_ray_query_inout_helper_receivers_emit_native_calls():
    code = """
    shader SlangRayQueryInoutHelper {
        RaytracingAccelerationStructure scene;

        void advanceQuery(
            inout RayQuery<RAY_FLAG_NONE> query,
            RaytracingAccelerationStructure accelerationStructure,
            RayDesc ray
        ) {
            query.TraceRayInline(accelerationStructure, 0, 0xFF, ray);
            let advanced = query.Proceed();
        }

        compute {
            void main() {
                RayDesc ray;
                RayQuery<RAY_FLAG_NONE> rq;
                advanceQuery(rq, scene, ray);
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "void advanceQuery(inout RayQuery<RAY_FLAG_NONE> query, "
        "RaytracingAccelerationStructure accelerationStructure, RayDesc ray)"
        in generated_code
    )
    assert "query.TraceRayInline(accelerationStructure, 0, 255, ray);" in generated_code
    assert "bool advanced = query.Proceed();" in generated_code
    assert "advanceQuery(rq, scene, ray);" in generated_code
    assert "RayQueryOpNode" not in generated_code


@pytest.mark.parametrize(
    ("stage_source", "message"),
    [
        (
            """
            ray_generation {
                void main(RayPayload payload @ payload) { }
            }
            """,
            "ray_generation stage cannot use payload parameter",
        ),
        (
            """
            ray_miss {
                void main(HitAttributes attributes @ hit_attribute) { }
            }
            """,
            "ray_miss stage cannot use hit_attribute parameter",
        ),
        (
            """
            ray_miss {
                void main(float payload @ payload) { }
            }
            """,
            "payload parameter 'payload' must use a user-defined struct type",
        ),
        (
            """
            ray_closest_hit {
                @rayPayloadInEXT RayPayload payload;
                void main(RayPayload payload @ payload) { }
            }
            """,
            "interface variable 'payload' duplicates an entry parameter",
        ),
        (
            """
            ray_closest_hit {
                void main(
                    RayPayload first @ payload,
                    RayPayload second @ rayPayloadInEXT,
                    HitAttributes attributes @ hit_attribute
                ) { }
            }
            """,
            "must declare at most one payload parameter",
        ),
    ],
)
def test_invalid_slang_ray_stage_semantic_parameters_raise(stage_source, message):
    code = f"""
    shader InvalidRayStageSemantics {{
        struct RayPayload {{
            vec3 color;
        }};

        struct HitAttributes {{
            vec2 barycentrics;
        }};

        {stage_source}
    }}
    """
    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(code)))


@pytest.mark.parametrize(
    ("stage_source", "message"),
    [
        (
            """
            compute {
                void main() {
                    TraceRay(1, 2, 3, 4, 5, 6, 7, 8);
                }
            }
            """,
            "compute stage cannot call TraceRay",
        ),
        (
            """
            ray_generation {
                void main() {
                    TraceRay(1, 2, 3);
                }
            }
            """,
            "TraceRay requires 8 or 11 argument",
        ),
        (
            """
            ray_any_hit {
                void main() {
                    ReportHit(1.0, 0, attributes);
                }
            }
            """,
            "ray_any_hit stage cannot call ReportHit",
        ),
        (
            """
            ray_intersection {
                void main() {
                    IgnoreHit();
                }
            }
            """,
            "ray_intersection stage cannot call IgnoreHit",
        ),
        (
            """
            ray_intersection {
                void main() {
                    ReportHit(1.0);
                }
            }
            """,
            "ReportHit requires 2 or 3 argument",
        ),
    ],
)
def test_invalid_slang_ray_tracing_intrinsic_stage_and_arity_raise(
    stage_source, message
):
    code = f"""
    shader InvalidSlangRayIntrinsicStage {{
        struct HitAttributes {{
            vec2 barycentrics;
        }};

        {stage_source}
    }}
    """
    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(code)))


@pytest.mark.parametrize(
    ("helper_source", "stage_source", "message"),
    [
        (
            """
            void traceScene(
                RaytracingAccelerationStructure scene,
                RayDesc ray,
                inout RayPayload payload
            ) {
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
            }
            """,
            """
            compute {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayDesc ray;
                    RayPayload payload;
                    traceScene(scene, ray, payload);
                }
            }
            """,
            "compute stage cannot call TraceRay",
        ),
        (
            """
            void invokeCallable(uint shaderIndex, inout CallableData data) {
                CallShader(shaderIndex, data);
            }
            """,
            """
            compute {
                void main() {
                    CallableData data;
                    invokeCallable(0, data);
                }
            }
            """,
            "compute stage cannot call CallShader",
        ),
        (
            """
            bool reportHit(HitAttributes attributes) {
                return ReportHit(1.0, 0, attributes);
            }
            """,
            """
            ray_any_hit {
                void main(
                    RayPayload payload @ payload,
                    HitAttributes attributes @ hit_attribute
                ) {
                    bool accepted = reportHit(attributes);
                }
            }
            """,
            "ray_any_hit stage cannot call ReportHit",
        ),
        (
            """
            void traceScene(uint scene, RayDesc ray, inout RayPayload payload) {
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
            }
            """,
            """
            ray_generation {
                void main() {
                    uint scene;
                    RayDesc ray;
                    RayPayload payload;
                    traceScene(scene, ray, payload);
                }
            }
            """,
            "TraceRay acceleration structure.*RaytracingAccelerationStructure",
        ),
    ],
)
def test_invalid_slang_ray_tracing_helper_intrinsics_raise(
    helper_source, stage_source, message
):
    code = f"""
    shader InvalidSlangRayIntrinsicHelper {{
        struct RayPayload {{
            vec3 color;
        }};

        struct CallableData {{
            uint value;
        }};

        struct HitAttributes {{
            vec2 barycentrics;
        }};

        {helper_source}

        {stage_source}
    }}
    """
    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(code)))


def test_slang_ray_hit_control_helpers_validate_through_deep_calls():
    code = """
    shader SlangRayHitControlHelperChain {
        struct RayPayload {
            uint value;
        };

        struct HitAttributes {
            vec2 barycentrics;
        };

        void acceptLeaf() {
            AcceptHitAndEndSearch();
        }

        void acceptRelay() {
            acceptLeaf();
        }

        void ignoreLeaf() {
            IgnoreHit();
        }

        void ignoreRelay() {
            ignoreLeaf();
        }

        ray_any_hit {
            void main(RayPayload payload @ payload, HitAttributes attributes @ hit_attribute) {
                if (payload.value != 0u) {
                    acceptRelay();
                } else {
                    ignoreRelay();
                }
            }
        }
    }
    """
    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "void acceptLeaf()" in generated_code
    assert "AcceptHitAndEndSearch();" in generated_code
    assert "void ignoreLeaf()" in generated_code
    assert "IgnoreHit();" in generated_code
    assert "acceptRelay();" in generated_code
    assert "ignoreRelay();" in generated_code
    assert "RayTracingOpNode" not in generated_code


@pytest.mark.parametrize(
    ("helper_source", "stage_source", "message"),
    [
        (
            """
            void traceScene(
                RaytracingAccelerationStructure scene,
                RayDesc ray,
                inout RayPayload payload
            ) {
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
            }
            """,
            """
            ray_generation {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayDesc ray;
                    OtherPayload payload;
                    traceScene(scene, ray, payload);
                }
            }
            """,
            "traceScene payload argument type OtherPayload must match parameter type RayPayload",
        ),
        (
            """
            RayPayload makePayload() {
                RayPayload payload;
                return payload;
            }

            void traceScene(
                RaytracingAccelerationStructure scene,
                RayDesc ray,
                inout RayPayload payload
            ) {
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
            }
            """,
            """
            ray_generation {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayDesc ray;
                    traceScene(scene, ray, makePayload());
                }
            }
            """,
            "traceScene payload argument.*lvalue",
        ),
        (
            """
            void traceScene(
                RaytracingAccelerationStructure scene,
                RayDesc ray,
                inout RayPayload payload
            ) {
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
            }
            """,
            """
            ray_generation {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayDesc ray;
                    traceScene(
                        scene,
                        ray,
                        RayPayload(vec3(0.0, 0.0, 0.0))
                    );
                }
            }
            """,
            "traceScene payload argument.*lvalue",
        ),
        (
            """
            RayPayload makePayload() {
                RayPayload payload;
                return payload;
            }

            void traceLeaf(
                RaytracingAccelerationStructure scene,
                RayDesc ray,
                inout RayPayload payload
            ) {
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
            }

            void traceReferenceRelay(
                RaytracingAccelerationStructure scene,
                RayDesc ray,
                RayPayload& payload
            ) {
                traceLeaf(scene, ray, payload);
            }
            """,
            """
            ray_generation {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayDesc ray;
                    traceReferenceRelay(scene, ray, makePayload());
                }
            }
            """,
            "traceReferenceRelay payload argument.*lvalue",
        ),
        (
            """
            void traceLeaf(
                RaytracingAccelerationStructure scene,
                RayDesc ray,
                inout RayPayload payload
            ) {
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
            }

            void traceReferenceRelay(
                RaytracingAccelerationStructure scene,
                RayDesc ray,
                RayPayload& payload
            ) {
                traceLeaf(scene, ray, payload);
            }
            """,
            """
            ray_generation {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayDesc ray;
                    traceReferenceRelay(
                        scene,
                        ray,
                        RayPayload(vec3(0.0, 0.0, 0.0))
                    );
                }
            }
            """,
            "traceReferenceRelay payload argument.*lvalue",
        ),
        (
            """
            RayPayloadHolder makeHolder() {
                RayPayloadHolder holder;
                return holder;
            }

            void traceLeaf(
                RaytracingAccelerationStructure scene,
                RayDesc ray,
                inout RayPayload payload
            ) {
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
            }

            void traceHolderRelay(
                RaytracingAccelerationStructure scene,
                RayDesc ray,
                inout RayPayloadHolder rays
            ) {
                traceLeaf(scene, ray, rays.primary);
            }
            """,
            """
            ray_generation {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayDesc ray;
                    traceHolderRelay(scene, ray, makeHolder());
                }
            }
            """,
            "traceHolderRelay payload argument.*lvalue",
        ),
        (
            """
            void invokeCallable(uint shaderIndex, inout CallableData data) {
                CallShader(shaderIndex, data);
            }
            """,
            """
            ray_generation {
                void main() {
                    OtherCallableData data;
                    invokeCallable(0, data);
                }
            }
            """,
            (
                "invokeCallable callable data argument type OtherCallableData "
                "must match parameter type CallableData"
            ),
        ),
        (
            """
            CallableData makeCallableData() {
                CallableData data;
                return data;
            }

            void invokeCallable(uint shaderIndex, inout CallableData data) {
                CallShader(shaderIndex, data);
            }
            """,
            """
            ray_generation {
                void main() {
                    invokeCallable(0, makeCallableData());
                }
            }
            """,
            "invokeCallable callable data argument.*lvalue",
        ),
        (
            """
            void invokeCallable(uint shaderIndex, inout CallableData data) {
                CallShader(shaderIndex, data);
            }
            """,
            """
            ray_generation {
                void main() {
                    invokeCallable(0, CallableData(1u));
                }
            }
            """,
            "invokeCallable callable data argument.*lvalue",
        ),
        (
            """
            CallableData makeCallableData() {
                CallableData data;
                return data;
            }

            void callableLeaf(uint shaderIndex, inout CallableData data) {
                CallShader(shaderIndex, data);
            }

            void callableReferenceRelay(
                uint shaderIndex,
                CallableData& data
            ) {
                callableLeaf(shaderIndex, data);
            }
            """,
            """
            ray_generation {
                void main() {
                    callableReferenceRelay(0, makeCallableData());
                }
            }
            """,
            "callableReferenceRelay callable data argument.*lvalue",
        ),
        (
            """
            void callableLeaf(uint shaderIndex, inout CallableData data) {
                CallShader(shaderIndex, data);
            }

            void callableReferenceRelay(
                uint shaderIndex,
                CallableData& data
            ) {
                callableLeaf(shaderIndex, data);
            }
            """,
            """
            ray_generation {
                void main() {
                    callableReferenceRelay(0, CallableData(1u));
                }
            }
            """,
            "callableReferenceRelay callable data argument.*lvalue",
        ),
        (
            """
            bool reportHit(HitAttributes attributes) {
                return ReportHit(1.0, 0, attributes);
            }
            """,
            """
            ray_intersection {
                void main() {
                    OtherHitAttributes attributes;
                    bool accepted = reportHit(attributes);
                }
            }
            """,
            (
                "reportHit hit attribute argument type OtherHitAttributes "
                "must match parameter type HitAttributes"
            ),
        ),
        (
            """
            void traceScene(
                RaytracingAccelerationStructure scene,
                RayDesc ray,
                inout RayPayload payload
            ) {
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
            }

            void relayTrace(
                RaytracingAccelerationStructure scene,
                RayDesc ray,
                inout OtherPayload payload
            ) {
                traceScene(scene, ray, payload);
            }
            """,
            """
            ray_generation {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayDesc ray;
                    OtherPayload payload;
                    relayTrace(scene, ray, payload);
                }
            }
            """,
            "traceScene payload argument type OtherPayload must match parameter type RayPayload",
        ),
        (
            """
            void traceScene(
                RaytracingAccelerationStructure scene,
                RayDesc ray,
                inout RayPayload payload
            ) {
                RayPayload& alias = payload;
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, alias);
            }
            """,
            """
            ray_generation {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayDesc ray;
                    OtherPayload payload;
                    traceScene(scene, ray, payload);
                }
            }
            """,
            "traceScene payload argument type OtherPayload must match parameter type RayPayload",
        ),
        (
            """
            void invokeCallable(uint shaderIndex, inout CallableData data) {
                CallableData& alias = data;
                CallShader(shaderIndex, alias);
            }
            """,
            """
            ray_generation {
                void main() {
                    OtherCallableData data;
                    invokeCallable(0, data);
                }
            }
            """,
            (
                "invokeCallable callable data argument type OtherCallableData "
                "must match parameter type CallableData"
            ),
        ),
        (
            """
            bool reportHit(HitAttributes attributes) {
                HitAttributes& alias = attributes;
                return ReportHit(1.0, 0, alias);
            }
            """,
            """
            ray_intersection {
                void main() {
                    OtherHitAttributes attributes;
                    bool accepted = reportHit(attributes);
                }
            }
            """,
            (
                "reportHit hit attribute argument type OtherHitAttributes "
                "must match parameter type HitAttributes"
            ),
        ),
        (
            """
            void traceScene(
                RaytracingAccelerationStructure scene,
                RayDesc ray,
                inout RayPayloadHolder rays
            ) {
                RayPayload& alias = rays.primary;
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, alias);
            }
            """,
            """
            ray_generation {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayDesc ray;
                    OtherPayloadHolder rays;
                    traceScene(scene, ray, rays);
                }
            }
            """,
            (
                "traceScene payload argument type OtherPayloadHolder "
                "must match parameter type RayPayloadHolder"
            ),
        ),
        (
            """
            void invokeCallable(uint shaderIndex, inout CallableDataHolder callables) {
                CallableData& alias = callables.items[0];
                CallShader(shaderIndex, alias);
            }
            """,
            """
            ray_generation {
                void main() {
                    OtherCallableDataHolder callables;
                    invokeCallable(0, callables);
                }
            }
            """,
            (
                "invokeCallable callable data argument type OtherCallableDataHolder "
                "must match parameter type CallableDataHolder"
            ),
        ),
        (
            """
            bool reportHit(HitAttributeHolder hits) {
                HitAttributes& alias = hits.items[1];
                return ReportHit(1.0, 0, alias);
            }
            """,
            """
            ray_intersection {
                void main() {
                    OtherHitAttributeHolder hits;
                    bool accepted = reportHit(hits);
                }
            }
            """,
            (
                "reportHit hit attribute argument type OtherHitAttributeHolder "
                "must match parameter type HitAttributeHolder"
            ),
        ),
        (
            """
            void traceHolder(
                RaytracingAccelerationStructure scene,
                RayDesc ray,
                RayPayloadHolder* rays
            ) {
                RayPayload& alias = rays->primary;
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, alias);
            }
            """,
            """
            ray_generation {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayDesc ray;
                    OtherPayloadHolder wrongs;
                    OtherPayloadHolder* rays = &wrongs;
                    traceHolder(scene, ray, rays);
                }
            }
            """,
            (
                "traceHolder payload argument type OtherPayloadHolder\\* "
                "must match parameter type RayPayloadHolder\\*"
            ),
        ),
        (
            """
            void tracePayload(
                RaytracingAccelerationStructure scene,
                RayDesc ray,
                inout RayPayload payload
            ) {
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
            }

            void relayTrace(
                RaytracingAccelerationStructure scene,
                RayDesc ray,
                RayPayloadHolder* rays
            ) {
                tracePayload(scene, ray, rays->primary);
            }
            """,
            """
            ray_generation {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayDesc ray;
                    OtherPayloadHolder wrongs;
                    OtherPayloadHolder* rays = &wrongs;
                    relayTrace(scene, ray, rays);
                }
            }
            """,
            (
                "relayTrace payload argument type OtherPayloadHolder\\* "
                "must match parameter type RayPayloadHolder\\*"
            ),
        ),
        (
            """
            void invokeHolder(uint shaderIndex, CallableDataHolder* callables) {
                CallableData& alias = callables->items[0];
                CallShader(shaderIndex, alias);
            }
            """,
            """
            ray_generation {
                void main() {
                    OtherCallableDataHolder wrongs;
                    OtherCallableDataHolder* callables = &wrongs;
                    invokeHolder(0, callables);
                }
            }
            """,
            (
                "invokeHolder callable data argument type OtherCallableDataHolder\\* "
                "must match parameter type CallableDataHolder\\*"
            ),
        ),
        (
            """
            bool reportHolder(HitAttributeHolder* hits) {
                HitAttributes& alias = hits->items[1];
                return ReportHit(1.0, 0, alias);
            }
            """,
            """
            ray_intersection {
                void main() {
                    OtherHitAttributeHolder wrongs;
                    OtherHitAttributeHolder* hits = &wrongs;
                    bool accepted = reportHolder(hits);
                }
            }
            """,
            (
                "reportHolder hit attribute argument type OtherHitAttributeHolder\\* "
                "must match parameter type HitAttributeHolder\\*"
            ),
        ),
        (
            """
            RayPayloadHolder makeHolder() {
                RayPayloadHolder holder;
                return holder;
            }

            void tracePayload(
                RaytracingAccelerationStructure scene,
                RayDesc ray,
                inout RayPayload payload
            ) {
                TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
            }

            void tracePointerRelay(
                RaytracingAccelerationStructure scene,
                RayDesc ray,
                RayPayloadHolder* rays
            ) {
                tracePayload(scene, ray, rays->primary);
            }
            """,
            """
            ray_generation {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayDesc ray;
                    tracePointerRelay(scene, ray, &makeHolder());
                }
            }
            """,
            "tracePointerRelay payload argument.*address of an lvalue",
        ),
        (
            """
            CallableDataHolder makeCallableHolder() {
                CallableDataHolder holder;
                return holder;
            }

            void callableLeaf(uint shaderIndex, inout CallableData data) {
                CallShader(shaderIndex, data);
            }

            void callablePointerRelay(
                uint shaderIndex,
                CallableDataHolder* callables
            ) {
                callableLeaf(shaderIndex, callables->items[0]);
            }
            """,
            """
            ray_generation {
                void main() {
                    callablePointerRelay(0, &makeCallableHolder());
                }
            }
            """,
            "callablePointerRelay callable data argument.*address of an lvalue",
        ),
        (
            """
            HitAttributeHolder makeHitHolder() {
                HitAttributeHolder holder;
                return holder;
            }

            bool reportPointerRelay(HitAttributeHolder* hits) {
                return ReportHit(1.0, 0, hits->items[1]);
            }
            """,
            """
            ray_intersection {
                void main() {
                    bool accepted = reportPointerRelay(&makeHitHolder());
                }
            }
            """,
            "reportPointerRelay hit attribute argument.*address of an lvalue",
        ),
    ],
)
def test_invalid_slang_ray_tracing_helper_interface_arguments_raise(
    helper_source, stage_source, message
):
    code = f"""
    shader InvalidSlangRayHelperInterfaceArgs {{
        struct RayPayload {{
            vec3 color;
        }};

        struct OtherPayload {{
            vec3 color;
        }};

        struct CallableData {{
            uint value;
        }};

        struct OtherCallableData {{
            uint value;
        }};

        struct HitAttributes {{
            vec2 barycentrics;
        }};

        struct OtherHitAttributes {{
            vec2 barycentrics;
        }};

        struct RayPayloadHolder {{
            RayPayload primary;
            RayPayload payloads[2];
        }};

        struct OtherPayloadHolder {{
            OtherPayload primary;
            OtherPayload payloads[2];
        }};

        struct CallableDataHolder {{
            CallableData primary;
            CallableData items[2];
        }};

        struct OtherCallableDataHolder {{
            OtherCallableData primary;
            OtherCallableData items[2];
        }};

        struct HitAttributeHolder {{
            HitAttributes primary;
            HitAttributes items[2];
        }};

        struct OtherHitAttributeHolder {{
            OtherHitAttributes primary;
            OtherHitAttributes items[2];
        }};

        {helper_source}

        {stage_source}
    }}
    """
    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(code)))


@pytest.mark.parametrize(
    ("stage_source", "message"),
    [
        (
            """
            ray_generation {
                void main() {
                    uint scene;
                    RayDesc ray;
                    RayPayload payload;
                    TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
                }
            }
            """,
            "TraceRay acceleration structure.*RaytracingAccelerationStructure",
        ),
        (
            """
            ray_generation {
                void main() {
                    RaytracingAccelerationStructure scene;
                    vec3 ray;
                    RayPayload payload;
                    TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
                }
            }
            """,
            "TraceRay ray descriptor.*RayDesc",
        ),
        (
            """
            ray_generation {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayDesc ray;
                    uint payload;
                    TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
                }
            }
            """,
            "TraceRay payload.*user-defined struct",
        ),
        (
            """
            ray_generation {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayDesc ray;
                    OtherRayPayload payload;
                    TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
                }
            }

            ray_miss {
                void main(RayPayload payload @ rayPayloadInEXT) { }
            }
            """,
            "TraceRay payload argument type OtherRayPayload "
            "must match ray payload parameter type RayPayload",
        ),
        (
            """
            RayPayload makePayload() {
                RayPayload payload;
                return payload;
            }

            ray_generation {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayDesc ray;
                    TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, makePayload());
                }
            }

            ray_miss {
                void main(RayPayload payload @ rayPayloadInEXT) { }
            }
            """,
            "TraceRay payload argument.*lvalue",
        ),
        (
            """
            ray_generation {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayDesc ray;
                    TraceRay(
                        scene,
                        0,
                        0xFF,
                        0,
                        1,
                        0,
                        ray,
                        RayPayload(vec3(0.0, 0.0, 0.0))
                    );
                }
            }

            ray_miss {
                void main(RayPayload payload @ rayPayloadInEXT) { }
            }
            """,
            "TraceRay payload argument.*lvalue",
        ),
        (
            """
            ray_generation {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayDesc ray;
                    OtherRayPayload payload;
                    TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, payload);
                }
            }

            ray_miss {
                @rayPayloadInEXT RayPayload payload;
                void main() { }
            }
            """,
            "TraceRay payload argument type OtherRayPayload "
            "must match ray payload parameter type RayPayload",
        ),
        (
            """
            ray_generation {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayPayload payload;
                    TraceRay(
                        scene,
                        0,
                        0xFF,
                        0,
                        1,
                        0,
                        vec2(0.0, 0.0),
                        0.0,
                        vec3(0.0, 0.0, 1.0),
                        100.0,
                        payload
                    );
                }
            }
            """,
            "TraceRay origin.*float3",
        ),
        (
            """
            ray_generation {
                void main() {
                    CallableData data;
                    CallShader(0.5, data);
                }
            }
            """,
            "CallShader shader index.*scalar int or uint",
        ),
        (
            """
            ray_generation {
                void main() {
                    OtherCallableData data;
                    CallShader(1, data);
                }
            }

            ray_callable {
                void main(CallableData data @ callableDataInEXT) { }
            }
            """,
            "CallShader callable data argument type OtherCallableData "
            "must match callable data parameter type CallableData",
        ),
        (
            """
            ray_generation {
                void main() {
                    OtherCallableData data;
                    CallShader(1, data);
                }
            }

            ray_callable {
                @callableDataInEXT CallableData data;
                void main() { }
            }
            """,
            "CallShader callable data argument type OtherCallableData "
            "must match callable data parameter type CallableData",
        ),
        (
            """
            CallableData makeCallableData() {
                CallableData data;
                return data;
            }

            ray_generation {
                void main() {
                    CallShader(1, makeCallableData());
                }
            }

            ray_callable {
                void main(CallableData data @ callableDataInEXT) { }
            }
            """,
            "CallShader callable data argument.*lvalue",
        ),
        (
            """
            ray_generation {
                void main() {
                    CallShader(1, CallableData(1u));
                }
            }

            ray_callable {
                void main(CallableData data @ callableDataInEXT) { }
            }
            """,
            "CallShader callable data argument.*lvalue",
        ),
        (
            """
            ray_intersection {
                void main() {
                    HitAttributes attributes;
                    uint distance;
                    ReportHit(distance, 0, attributes);
                }
            }
            """,
            "ReportHit hit distance.*scalar floating",
        ),
        (
            """
            ray_intersection {
                void main() {
                    uint attributes;
                    ReportHit(1.0, 0, attributes);
                }
            }
            """,
            "ReportHit hit attribute.*user-defined struct",
        ),
        (
            """
            ray_intersection {
                void main() {
                    OtherHitAttributes attributes;
                    ReportHit(1.0, 0, attributes);
                }
            }

            ray_closest_hit {
                void main(
                    RayPayload payload @ payload,
                    HitAttributes attributes @ hit_attribute
                ) { }
            }
            """,
            "ReportHit hit attribute argument type OtherHitAttributes "
            "must match hit attribute parameter type HitAttributes",
        ),
        (
            """
            ray_intersection {
                void main() {
                    OtherHitAttributes attributes;
                    ReportHit(1.0, 0, attributes);
                }
            }

            ray_closest_hit {
                @hitAttributeEXT HitAttributes attributes;
                void main(RayPayload payload @ payload) { }
            }
            """,
            "ReportHit hit attribute argument type OtherHitAttributes "
            "must match hit attribute parameter type HitAttributes",
        ),
        (
            """
            OtherHitAttributes makeOtherAttributes() {
                OtherHitAttributes attributes;
                return attributes;
            }

            ray_intersection {
                void main() {
                    ReportHit(1.0, 0, makeOtherAttributes());
                }
            }

            ray_closest_hit {
                void main(
                    RayPayload payload @ payload,
                    HitAttributes attributes @ hit_attribute
                ) { }
            }
            """,
            "ReportHit hit attribute argument type OtherHitAttributes "
            "must match hit attribute parameter type HitAttributes",
        ),
        (
            """
            OtherHitAttributes makeOtherAttributes() {
                OtherHitAttributes attributes;
                return attributes;
            }

            OtherHitAttributes relayAttributes() {
                return makeOtherAttributes();
            }

            ray_intersection {
                void main() {
                    ReportHit(1.0, 0, relayAttributes());
                }
            }

            ray_closest_hit {
                @hitAttributeEXT HitAttributes attributes;
                void main(RayPayload payload @ payload) { }
            }
            """,
            "ReportHit hit attribute argument type OtherHitAttributes "
            "must match hit attribute parameter type HitAttributes",
        ),
        (
            """
            uint makeAttributeScalar() {
                return 1u;
            }

            ray_intersection {
                void main() {
                    ReportHit(1.0, 0, makeAttributeScalar());
                }
            }
            """,
            "ReportHit hit attribute.*user-defined struct",
        ),
    ],
)
def test_invalid_slang_ray_tracing_intrinsic_argument_types_raise(
    stage_source, message
):
    code = f"""
    shader InvalidSlangRayIntrinsicArguments {{
        struct RayPayload {{
            vec3 color;
        }};

        struct OtherRayPayload {{
            vec3 color;
        }};

        struct CallableData {{
            uint value;
        }};

        struct OtherCallableData {{
            uint value;
        }};

        struct HitAttributes {{
            vec2 barycentrics;
        }};

        struct OtherHitAttributes {{
            vec2 barycentrics;
        }};

        {stage_source}
    }}
    """
    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(code)))


@pytest.mark.parametrize(
    ("stage_source", "message"),
    [
        (
            """
            ray_generation {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayDesc ray;
                    RayPayloadHolder rays;
                    TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, rays.wrong);
                }
            }

            ray_miss {
                void main(RayPayload payload @ rayPayloadInEXT) { }
            }
            """,
            "TraceRay payload argument type OtherRayPayload "
            "must match ray payload parameter type RayPayload",
        ),
        (
            """
            ray_generation {
                void main() {
                    CallableDataHolder callables;
                    CallShader(0, callables.wrong);
                }
            }

            ray_callable {
                void main(CallableData data @ callableDataInEXT) { }
            }
            """,
            "CallShader callable data argument type OtherCallableData "
            "must match callable data parameter type CallableData",
        ),
        (
            """
            ray_intersection {
                void main() {
                    HitAttributeHolder hits;
                    ReportHit(1.0, 0, hits.wrong);
                }
            }

            ray_closest_hit {
                void main(
                    RayPayload payload @ payload,
                    HitAttributes attributes @ hit_attribute
                ) { }
            }
            """,
            "ReportHit hit attribute argument type OtherHitAttributes "
            "must match hit attribute parameter type HitAttributes",
        ),
    ],
)
def test_invalid_slang_ray_tracing_member_lvalue_interface_types_raise(
    stage_source, message
):
    code = f"""
    shader InvalidSlangRayInterfaceLvalues {{
        struct RayPayload {{
            vec3 color;
        }};

        struct OtherRayPayload {{
            vec3 color;
        }};

        struct CallableData {{
            uint value;
        }};

        struct OtherCallableData {{
            uint value;
        }};

        struct HitAttributes {{
            vec2 barycentrics;
        }};

        struct OtherHitAttributes {{
            vec2 barycentrics;
        }};

        struct RayPayloadHolder {{
            RayPayload primary;
            RayPayload payloads[2];
            OtherRayPayload wrong;
        }};

        struct CallableDataHolder {{
            CallableData primary;
            CallableData items[2];
            OtherCallableData wrong;
        }};

        struct HitAttributeHolder {{
            HitAttributes primary;
            HitAttributes items[2];
            OtherHitAttributes wrong;
        }};

        {stage_source}
    }}
    """
    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(code)))


@pytest.mark.parametrize(
    ("stage_source", "message"),
    [
        (
            """
            ray_generation {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayDesc ray;
                    OtherRayPayload wrong;
                    OtherRayPayload& wrongRef = wrong;
                    TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, wrongRef);
                }
            }

            ray_miss {
                void main(RayPayload payload @ rayPayloadInEXT) { }
            }
            """,
            "TraceRay payload argument type OtherRayPayload "
            "must match ray payload parameter type RayPayload",
        ),
        (
            """
            ray_generation {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayDesc ray;
                    OtherRayPayloadHolder wrongs;
                    OtherRayPayloadHolder* wrongPtr = &wrongs;
                    TraceRay(scene, 0, 0xFF, 0, 1, 0, ray, wrongPtr->primary);
                }
            }

            ray_miss {
                void main(RayPayload payload @ rayPayloadInEXT) { }
            }
            """,
            "TraceRay payload argument type OtherRayPayload "
            "must match ray payload parameter type RayPayload",
        ),
        (
            """
            ray_generation {
                void main() {
                    OtherCallableData wrong;
                    OtherCallableData& wrongRef = wrong;
                    CallShader(0, wrongRef);
                }
            }

            ray_callable {
                void main(CallableData data @ callableDataInEXT) { }
            }
            """,
            "CallShader callable data argument type OtherCallableData "
            "must match callable data parameter type CallableData",
        ),
        (
            """
            ray_generation {
                void main() {
                    OtherCallableDataHolder wrongs;
                    OtherCallableDataHolder* wrongPtr = &wrongs;
                    CallShader(0, wrongPtr->primary);
                }
            }

            ray_callable {
                void main(CallableData data @ callableDataInEXT) { }
            }
            """,
            "CallShader callable data argument type OtherCallableData "
            "must match callable data parameter type CallableData",
        ),
        (
            """
            ray_intersection {
                void main() {
                    OtherHitAttributes wrong;
                    OtherHitAttributes& wrongRef = wrong;
                    ReportHit(1.0, 0, wrongRef);
                }
            }

            ray_closest_hit {
                void main(
                    RayPayload payload @ payload,
                    HitAttributes attributes @ hit_attribute
                ) { }
            }
            """,
            "ReportHit hit attribute argument type OtherHitAttributes "
            "must match hit attribute parameter type HitAttributes",
        ),
        (
            """
            ray_intersection {
                void main() {
                    OtherHitAttributeHolder wrongs;
                    OtherHitAttributeHolder* wrongPtr = &wrongs;
                    ReportHit(1.0, 0, wrongPtr->primary);
                }
            }

            ray_closest_hit {
                void main(
                    RayPayload payload @ payload,
                    HitAttributes attributes @ hit_attribute
                ) { }
            }
            """,
            "ReportHit hit attribute argument type OtherHitAttributes "
            "must match hit attribute parameter type HitAttributes",
        ),
    ],
)
def test_invalid_slang_ray_tracing_pointer_reference_interface_types_raise(
    stage_source, message
):
    code = f"""
    shader InvalidSlangRayPointerReferenceInterfaceTypes {{
        struct RayPayload {{
            vec3 color;
        }};

        struct OtherRayPayload {{
            vec3 color;
        }};

        struct CallableData {{
            uint value;
        }};

        struct OtherCallableData {{
            uint value;
        }};

        struct HitAttributes {{
            vec2 barycentrics;
        }};

        struct OtherHitAttributes {{
            vec2 barycentrics;
        }};

        struct OtherRayPayloadHolder {{
            OtherRayPayload primary;
        }};

        struct OtherCallableDataHolder {{
            OtherCallableData primary;
        }};

        struct OtherHitAttributeHolder {{
            OtherHitAttributes primary;
        }};

        {stage_source}
    }}
    """
    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(code)))


@pytest.mark.parametrize(
    ("stage_source", "message"),
    [
        (
            """
            compute {
                void main() {
                    uint rq;
                    rq.Proceed();
                }
            }
            """,
            "RayQuery.Proceed receiver.*RayQuery",
        ),
        (
            """
            compute {
                void main() {
                    uint queries[2];
                    queries[0].CandidateObjectRayOrigin();
                }
            }
            """,
            "RayQuery.CandidateObjectRayOrigin receiver.*RayQuery",
        ),
        (
            """
            compute {
                void main() {
                    vec3 origin;
                    origin.CommittedObjectToWorld3x4();
                }
            }
            """,
            "RayQuery.CommittedObjectToWorld3x4 receiver.*RayQuery",
        ),
        (
            """
            compute {
                void main() {
                    RayQuery<RAY_FLAG_NONE> rq;
                    RayQuery<RAY_FLAG_NONE>* rqPtr = &rq;
                    rqPtr->Proceed();
                }
            }
            """,
            "RayQuery.Proceed receiver.*RayQuery",
        ),
        (
            """
            RayQuery<RAY_FLAG_NONE> pick(RayQuery<RAY_FLAG_NONE> query) {
                return query;
            }
            compute {
                void main() {
                    RayQuery<RAY_FLAG_NONE> rq;
                    pick(rq).Proceed();
                }
            }
            """,
            "RayQuery.Proceed receiver.*RayQuery lvalue",
        ),
        (
            """
            struct QueryHolder {
                RayQuery<RAY_FLAG_NONE> query;
            }
            QueryHolder pick(QueryHolder holder) {
                return holder;
            }
            compute {
                void main() {
                    QueryHolder holder;
                    let t = pick(holder).query.CommittedRayT();
                }
            }
            """,
            "RayQuery.CommittedRayT receiver.*RayQuery lvalue",
        ),
        (
            """
            compute {
                void main() {
                    RaytracingAccelerationStructure scene;
                    RayQuery<RAY_FLAG_NONE> rq;
                    rq.TraceRayInline(scene, 0, 0xFF);
                }
            }
            """,
            "TraceRayInline requires 4 argument",
        ),
        (
            """
            compute {
                void main() {
                    uint scene;
                    RayDesc ray;
                    RayQuery<RAY_FLAG_NONE> rq;
                    rq.TraceRayInline(scene, 0, 0xFF, ray);
                }
            }
            """,
            "TraceRayInline acceleration structure.*RaytracingAccelerationStructure",
        ),
        (
            """
            compute {
                void main() {
                    RaytracingAccelerationStructure scene;
                    vec3 ray;
                    RayQuery<RAY_FLAG_NONE> rq;
                    rq.TraceRayInline(scene, 0, 0xFF, ray);
                }
            }
            """,
            "TraceRayInline ray descriptor.*RayDesc",
        ),
        (
            """
            compute {
                void main() {
                    RayQuery<RAY_FLAG_NONE> rq;
                    uint hitDistance;
                    rq.CommitProceduralPrimitiveHit(hitDistance);
                }
            }
            """,
            "CommitProceduralPrimitiveHit hit distance.*scalar floating",
        ),
        (
            """
            compute {
                void main() {
                    RayQuery<RAY_FLAG_NONE> rq;
                    rq.CommitNonOpaqueTriangleHit(1.0);
                }
            }
            """,
            "CommitNonOpaqueTriangleHit requires 0 argument",
        ),
    ],
)
def test_invalid_slang_ray_query_method_arguments_raise(stage_source, message):
    code = f"""
    shader InvalidSlangRayQuery {{
        {stage_source}
    }}
    """
    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(code)))


@pytest.mark.parametrize(
    ("helper_source", "message"),
    [
        (
            """
            void invalidReceiver(uint value) {
                value.Proceed();
            }
            """,
            "Slang function RayQuery.Proceed receiver.*RayQuery",
        ),
        (
            """
            RayQuery<RAY_FLAG_NONE> pick(RayQuery<RAY_FLAG_NONE> query) {
                return query;
            }
            void invalidTemporary(RayQuery<RAY_FLAG_NONE> query) {
                pick(query).Proceed();
            }
            """,
            "Slang function RayQuery.Proceed receiver.*RayQuery lvalue",
        ),
        (
            """
            void invalidTrace(
                RayQuery<RAY_FLAG_NONE> query,
                uint accelerationStructure,
                RayDesc ray
            ) {
                query.TraceRayInline(accelerationStructure, 0, 0xFF, ray);
            }
            """,
            "Slang function RayQuery.TraceRayInline acceleration structure"
            ".*RaytracingAccelerationStructure",
        ),
    ],
)
def test_invalid_slang_ray_query_helper_method_arguments_raise(helper_source, message):
    code = f"""
    shader InvalidSlangRayQueryHelper {{
        {helper_source}

        compute {{
            void main() {{ }}
        }}
    }}
    """
    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(code)))


@pytest.mark.parametrize(
    ("stage_source", "message"),
    [
        (
            "geometry { void main() @maxvertexcount(0) { } }",
            "maxvertexcount.*positive",
        ),
        (
            "tessellation_control { void main() @domain(patch) { } }",
            "domain 'patch'.*tri",
        ),
        (
            (
                "tessellation_control { void main() "
                "@outputtopology(triangles_cw) { } }"
            ),
            "outputtopology 'triangles_cw'",
        ),
        (
            "tessellation_control { void main() @partitioning(fractional) { } }",
            "partitioning 'fractional'",
        ),
        (
            (
                "tessellation_control { void main() "
                "@domain(tri) @outputtopology(line) { } }"
            ),
            "domain 'tri' requires outputtopology triangle_cw",
        ),
        (
            (
                "tessellation_control { void main() "
                "@domain(isoline) @outputtopology(triangle_cw) { } }"
            ),
            "domain 'isoline' requires outputtopology line",
        ),
        (
            "tessellation_control { void main() @outputcontrolpoints(0) { } }",
            "outputcontrolpoints.*positive",
        ),
        (
            "mesh { void main() @outputtopology(triangle_cw) { } }",
            "mesh stage outputtopology 'triangle_cw'",
        ),
        (
            "compute { void main() @maxvertexcount(3) { } }",
            "compute stage does not support maxvertexcount",
        ),
        (
            "compute { void main() @numthreads(8, 4, 2, 1) { } }",
            "numthreads requires at most three arguments",
        ),
        (
            "compute { void main() @numthreads(0, 1, 1) { } }",
            "numthreads values must be positive",
        ),
    ],
)
def test_invalid_slang_stage_attributes_raise(stage_source, message):
    code = f"""
    shader InvalidStageAttributes {{
        {stage_source}
    }}
    """
    with pytest.raises(ValueError, match=message):
        generate_code(parse_code(tokenize_code(code)))


def test_if_else_control_flow_emits_slang_blocks():
    code = """
    shader main {
        vertex {
            void main() {
                float y = 1.0;
                if (y > 0.0) {
                    y = y + 1.0;
                } else {
                    y = 0.0;
                }
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "if (y > 0.0)" in generated_code
    assert "y = y + 1.0;" in generated_code
    assert "else" in generated_code
    assert "y = 0.0;" in generated_code


def test_for_loop_control_flow_emits_slang_loop():
    code = """
    shader main {
        vertex {
            void main() {
                float y = 0.0;
                for (int i = 0; i < 4; i = i + 1) {
                    y = y + 1.0;
                }
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (int i = 0; i < 4; i = i + 1)" in generated_code
    assert "y = y + 1.0;" in generated_code


def test_optional_for_loop_clauses_emit_empty_slang_slots():
    code = """
    shader main {
        compute {
            void main() {
                int i = 0;
                for (; ; ) {
                    i = i + 1;
                    if (i > 3) {
                        break;
                    }
                }
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (; ; )" in generated_code
    assert "None" not in generated_code
    assert "break;" in generated_code


def test_for_in_statement_lowers_to_counted_slang_loops():
    code = """
    shader main {
        compute {
            void main() {
                int total = 0;
                for i in 4 {
                    total = total + i;
                }
                for j in 2..5 {
                    total = total + j;
                }
                for k in 1..=4 {
                    total = total + k;
                }
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "for (int i = 0; i < 4; ++i)" in generated_code
    assert "for (int j = 2; j < 5; ++j)" in generated_code
    assert "for (int k = 1; k <= 4; ++k)" in generated_code
    assert "total = total + i;" in generated_code
    assert "total = total + j;" in generated_code
    assert "total = total + k;" in generated_code
    assert "ForInNode" not in generated_code
    assert "RangeNode" not in generated_code


def test_while_loop_control_flow_emits_slang_loop():
    code = """
    shader main {
        vertex {
            void main() {
                int i = 0;
                while (i < 4) {
                    i = i + 1;
                }
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "while (i < 4)" in generated_code
    assert "i = i + 1;" in generated_code
    assert "WhileNode(" not in generated_code


def test_do_while_loop_control_flow_emits_slang_loop():
    code = """
    shader main {
        vertex {
            void main() {
                int i = 0;
                do {
                    i = i + 1;
                } while (i < 4);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "\n    do\n    {\n" in generated_code
    assert "i = i + 1;" in generated_code
    assert "\n    } while (i < 4);" in generated_code
    assert "DoWhileNode(" not in generated_code


def test_control_flow_blocks_are_indented_consistently():
    code = """
    shader main {
        vertex {
            void main() {
                float y = 0.0;
                while (y < 4.0) {
                    if (y > 1.0) {
                        y = y + 2.0;
                    } else {
                        y = y + 1.0;
                    }
                }
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "\n    while (y < 4.0)\n    {\n" in generated_code
    assert "\n        if (y > 1.0)\n        {\n" in generated_code
    assert "\n        else\n        {\n" in generated_code
    assert "\n            y = y + 2.0;\n" in generated_code


def test_switch_break_continue_and_void_return_emit_slang_syntax():
    code = """
    shader main {
        vertex {
            void main() {
                int i = 0;
                while (i < 4) {
                    switch (i) {
                        case 0:
                            i = i + 1;
                            continue;
                        default:
                            break;
                    }
                }
                return;
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "switch (i)" in generated_code
    assert "case 0:" in generated_code
    assert "default:" in generated_code
    assert "continue;" in generated_code
    assert "break;" in generated_code
    assert "return;" in generated_code
    assert "SwitchNode(" not in generated_code
    assert "BreakNode(" not in generated_code
    assert "ContinueNode(" not in generated_code
    assert "return None;" not in generated_code


def test_match_literal_and_wildcard_arms_lower_to_slang_switch():
    code = """
    shader main {
        compute {
            int main(int mode) {
                int value = 0;
                match mode {
                    0 => {
                        value = 1;
                    }
                    _ => {
                        value = 2;
                    }
                }
                return value;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "switch (mode)" in generated_code
    assert "case 0:" in generated_code
    assert "value = 1;" in generated_code
    assert "default:" in generated_code
    assert "value = 2;" in generated_code
    assert generated_code.count("break;") == 2
    assert "MatchNode" not in generated_code


def test_match_guarded_arm_rejected_for_slang_switch_lowering():
    code = """
    shader main {
        compute {
            int main(int mode) {
                int value = 0;
                match mode {
                    0 if mode > 0 => {
                        value = 1;
                    }
                    _ => {
                        value = 2;
                    }
                }
                return value;
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)

    with pytest.raises(ValueError, match="Unsupported match arm for Slang"):
        generate_code(ast)


def test_array_access_and_ternary_expressions_emit_slang_syntax():
    code = """
    shader main {
        vertex {
            void main() {
                float values[2];
                values[0] = 1.0;
                values[1] = 2.0;
                float chosen = values[0] > 0.0 ? values[0] : values[1];
                gl_Position = vec4(chosen, chosen, chosen, 1.0);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float values[2];" in generated_code
    assert "values[0] = 1.0;" in generated_code
    assert "values[1] = 2.0;" in generated_code
    assert "float chosen = (values[0] > 0.0 ? values[0] : values[1]);" in generated_code
    assert "ArrayType(" not in generated_code
    assert "ArrayAccessNode(" not in generated_code
    assert "TernaryOpNode(" not in generated_code


def test_matrix_and_non_float_vector_types_emit_slang_names():
    code = """
    shader main {
        struct Transform {
            mat4 model;
            mat3 normal;
            ivec2 tile;
            uvec4 mask;
            bvec3 flags;
        };

        vertex {
            vec4 transformPosition(mat4 m, vec4 p) {
                return m * p;
            }

            void main() {
                mat4 model = mat4(1.0);
                mat3 normal;
                mat2x3 nonSquare;
                ivec2 tile;
                uvec4 mask;
                bvec3 flags;
                bvec2 mask = bvec2(true, false);
                dvec2 preciseUV = dvec2(1.0, 2.0);
                dmat2 precise = dmat2(1.0, 0.0, 0.0, 1.0);
                dmat4x3 jacobian;
                vec4 p = transformPosition(model, vec4(1.0, 2.0, 3.0, 1.0));
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float4x4 model;" in generated_code
    assert "float3x3 normal;" in generated_code
    assert "int2 tile;" in generated_code
    assert "uint4 mask;" in generated_code
    assert "bool3 flags;" in generated_code
    assert "bool2 mask = bool2(true, false);" in generated_code
    assert "float4 transformPosition(float4x4 m, float4 p)" in generated_code
    assert "float4x4 model = float4x4(1.0);" in generated_code
    assert "float2x3 nonSquare;" in generated_code
    assert "double2 preciseUV = double2(1.0, 2.0);" in generated_code
    assert "double2x2 precise = double2x2(1.0, 0.0, 0.0, 1.0);" in generated_code
    assert "double4x3 jacobian;" in generated_code
    assert "dvec2(" not in generated_code
    assert "bvec2(" not in generated_code
    assert "dmat2(" not in generated_code
    assert "MatrixType(" not in generated_code


def test_generic_vector_constructors_emit_slang_names():
    code = """
    shader main {
        compute {
            void main() {
                vec2<f64> precise = vec2<f64>(1.0, 2.0);
                vec3<i32> index = vec3<i32>(1, 2, 3);
                vec4<u32> mask = vec4<u32>(1, 2, 3, 4);
                vec2<bool> flags = vec2<bool>(true, false);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "double2 precise = double2(1.0, 2.0);" in generated_code
    assert "int3 index = int3(1, 2, 3);" in generated_code
    assert "uint4 mask = uint4(1, 2, 3, 4);" in generated_code
    assert "bool2 flags = bool2(true, false);" in generated_code
    assert "vec2<" not in generated_code
    assert "vec3<" not in generated_code
    assert "vec4<" not in generated_code


def test_resource_types_emit_slang_texture_names():
    code = """
    shader Resources {
        sampler1d lineMap;
        sampler1darray lineArray;
        sampler2d colorMap;
        sampler2dms msTex;
        sampler2dmsarray msArray;
        image1D lineImage;
        image1DArray lineImages;
        iimage1D signedLine;
        uimage1DArray unsignedLines;
        image2D colorImage;
        image2DMS msColor;
        image2DMSArray msLayers;
        iimage2DMSArray signedLayers;
        uimage2DMS counters;

        compute {
            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Sampler1D<float4> lineMap : register(t0);" in generated_code
    assert "Sampler1DArray<float4> lineArray : register(t1);" in generated_code
    assert "Sampler2D<float4> colorMap : register(t2);" in generated_code
    assert "Sampler2DMS<float4> msTex : register(t3);" in generated_code
    assert "Sampler2DMSArray<float4> msArray : register(t4);" in generated_code
    assert "RWTexture1D<float4> lineImage : register(u0);" in generated_code
    assert "RWTexture1DArray<float4> lineImages : register(u1);" in generated_code
    assert "RWTexture1D<int> signedLine : register(u2);" in generated_code
    assert "RWTexture1DArray<uint> unsignedLines : register(u3);" in generated_code
    assert "RWTexture2D<float4> colorImage : register(u4);" in generated_code
    assert "RWTexture2DMS<float4> msColor : register(u5);" in generated_code
    assert "RWTexture2DMSArray<float4> msLayers : register(u6);" in generated_code
    assert "RWTexture2DMSArray<int> signedLayers : register(u7);" in generated_code
    assert "RWTexture2DMS<uint> counters : register(u8);" in generated_code
    assert "sampler2d" not in generated_code
    assert "image2DMS" not in generated_code
    assert "iimage2DMSArray" not in generated_code
    assert "uimage2DMS" not in generated_code


def test_resource_binding_metadata_emits_slang_vk_and_register_decorations(tmp_path):
    code = """
    shader ResourceBindings {
        cbuffer Camera : register(b1, space2) {
            mat4 viewProj;
        };

        layout(set = 2, binding = 7) uniform sampler2D colorMap;
        sampler linearSampler @register(s3, space2);
        uimage2D counters @r32ui @binding(8) @set(2);
        accelerationStructureEXT scene @binding(9) @set(2);
        sampler2d textureArray[4] @binding(12) @set(2);

        compute {
            vec4 sampleBound(
                sampler2D paramTex @binding(10) @set(2),
                sampler paramSampler @register(s11, space2),
                uimage2D paramImage @r32ui @binding(13) @set(2),
                vec2 uv
            ) {
                uint oldValue = imageAtomicAdd(paramImage, ivec2(0, 0), 1u);
                return texture(paramTex, uv) + vec4(float(oldValue));
            }

            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "[[vk::binding(1, 2)]] cbuffer Camera : register(b1, space2)" in generated_code
    )
    assert "float4x4 viewProj;" in generated_code
    assert (
        "[[vk::binding(7, 2)]] Sampler2D<float4> colorMap "
        ": register(t7, space2);" in generated_code
    )
    assert (
        "[[vk::binding(3, 2)]] SamplerState linearSampler "
        ": register(s3, space2);" in generated_code
    )
    assert (
        "[[vk::binding(8, 2)]] RWTexture2D<uint> counters "
        ": register(u8, space2);" in generated_code
    )
    assert (
        "[[vk::binding(9, 2)]] RaytracingAccelerationStructure scene "
        ": register(t9, space2);" in generated_code
    )
    assert (
        "[[vk::binding(12, 2)]] Sampler2D<float4> textureArray[4] "
        ": register(t12, space2);" in generated_code
    )
    assert (
        "[[vk::binding(10, 2)]] Sampler2D<float4> paramTex "
        ": register(t10, space2)" in generated_code
    )
    assert (
        "[[vk::binding(11, 2)]] SamplerState paramSampler "
        ": register(s11, space2)" in generated_code
    )
    assert (
        "[[vk::binding(13, 2)]] RWTexture2D<uint> paramImage "
        ": register(u13, space2)" in generated_code
    )
    assert ": binding" not in generated_code
    assert ": set" not in generated_code
    assert ": register(register" not in generated_code

    compile_generated_slang(generated_code, tmp_path, "compute")


@pytest.mark.parametrize(
    ("resource_source", "message"),
    [
        (
            "sampler2d colorMap @binding(4) @register(t5);",
            "binding 4 does not match register binding 5",
        ),
        (
            "sampler2d colorMap @set(1) @register(t4, space2);",
            "set 1 does not match register space2",
        ),
    ],
)
def test_conflicting_slang_resource_binding_metadata_raises(resource_source, message):
    code = f"""
    shader InvalidResourceBindings {{
        {resource_source}

        compute {{
            void main() {{}}
        }}
    }}
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    with pytest.raises(ValueError, match=message):
        generate_code(ast)


def test_slang_auto_resource_bindings_assign_ranges_and_skip_reserved_slots():
    code = """
    shader AutoResourceBindings {
        cbuffer Camera {
            mat4 viewProj;
        };

        sampler2d firstTexture;
        sampler2d textureArray[2];
        sampler2d reservedTexture @binding(2);
        sampler2d afterArray;
        sampler2d setOneTexture @set(1);
        sampler2d setOneAfter @set(1);
        sampler linearSampler;
        uimage2D counters @r32ui;

        compute {
            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "[[vk::binding(0, 0)]] cbuffer Camera : register(b0)" in generated_code
    assert (
        "[[vk::binding(0, 0)]] Sampler2D<float4> firstTexture "
        ": register(t0);" in generated_code
    )
    assert (
        "[[vk::binding(3, 0)]] Sampler2D<float4> textureArray[2] "
        ": register(t3);" in generated_code
    )
    assert (
        "[[vk::binding(2, 0)]] Sampler2D<float4> reservedTexture "
        ": register(t2);" in generated_code
    )
    assert (
        "[[vk::binding(5, 0)]] Sampler2D<float4> afterArray "
        ": register(t5);" in generated_code
    )
    assert (
        "[[vk::binding(0, 1)]] Sampler2D<float4> setOneTexture "
        ": register(t0, space1);" in generated_code
    )
    assert (
        "[[vk::binding(1, 1)]] Sampler2D<float4> setOneAfter "
        ": register(t1, space1);" in generated_code
    )
    assert (
        "[[vk::binding(0, 0)]] SamplerState linearSampler "
        ": register(s0);" in generated_code
    )
    assert (
        "[[vk::binding(0, 0)]] RWTexture2D<uint> counters "
        ": register(u0);" in generated_code
    )


def test_structured_buffers_emit_slang_load_store_dimensions_and_bindings():
    code = """
    shader SlangStructuredBuffers {
        struct Particle {
            vec4 position;
            float mass;
        };

        StructuredBuffer<Particle> particlesIn @binding(1);
        RWStructuredBuffer<Particle> particlesOut @binding(2);
        StructuredBuffer<vec4> vectors;
        RWStructuredBuffer<float> values[2];

        compute {
            uint countValues(RWStructuredBuffer<float> data) {
                return buffer_dimensions(data);
            }

            void main(uint3 tid @ gl_GlobalInvocationID) {
                uint len;
                Particle particle = buffer_load(particlesIn, tid.x);
                vec4 vectorValue = buffer_load(vectors, 0u);
                particle.mass = particle.mass + vectorValue.x;
                buffer_dimensions(particlesOut, len);
                buffer_store(particlesOut, tid.x, particle);
                buffer_store(values[1], 0u, particle.mass);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "[[vk::binding(1, 0)]] StructuredBuffer<Particle> particlesIn "
        ": register(t1);" in generated_code
    )
    assert (
        "[[vk::binding(2, 0)]] RWStructuredBuffer<Particle> particlesOut "
        ": register(u2);" in generated_code
    )
    assert (
        "[[vk::binding(2, 0)]] StructuredBuffer<float4> vectors "
        ": register(t2);" in generated_code
    )
    assert (
        "[[vk::binding(3, 0)]] RWStructuredBuffer<float> values[2] "
        ": register(u3);" in generated_code
    )
    assert (
        "uint cgl_bufferDimensions_RWStructuredBuffer_float("
        "RWStructuredBuffer<float> buffer)" in generated_code
    )
    assert (
        "return cgl_bufferDimensions_RWStructuredBuffer_float(data);" in generated_code
    )
    assert "Particle particle = particlesIn.Load(tid.x);" in generated_code
    assert "float4 vectorValue = vectors.Load(0u);" in generated_code
    assert "particlesOut.GetDimensions(len);" in generated_code
    assert "particlesOut.Store(tid.x, particle);" in generated_code
    assert "values[1].Store(0u, particle.mass);" in generated_code
    assert "buffer_load(" not in generated_code
    assert "buffer_store(" not in generated_code
    assert "buffer_dimensions(" not in generated_code
    assert "StructuredBuffer<vec4>" not in generated_code


def test_structured_buffer_store_to_readonly_buffer_emits_slang_diagnostic():
    code = """
    shader SlangReadonlyStructuredBuffer {
        StructuredBuffer<float> values;

        compute {
            void main() {
                buffer_store(values, 0u, 1.0);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "/* unsupported Slang structured buffer: buffer_store requires "
        "RWStructuredBuffer resource */;" in generated_code
    )
    assert "buffer_store(values" not in generated_code


def test_structured_buffer_access_qualifiers_emit_slang_kinds_and_diagnostics():
    code = """
    shader SlangStructuredBufferAccessQualifiers {
        readonly RWStructuredBuffer<float> readOnlyValues @binding(4);
        writeonly StructuredBuffer<float> writeOnlyValues @binding(5);
        readwrite StructuredBuffer<float> readWriteValues @binding(6);
        RWStructuredBuffer<float> attrReadOnly @access(read) @binding(7);
        StructuredBuffer<float> attrWriteOnly @access(write) @binding(8);

        compute {
            void main() {
                float blockedLoad = buffer_load(writeOnlyValues, 0u);
                float blockedAttrLoad = buffer_load(attrWriteOnly, 0u);
                buffer_store(readOnlyValues, 0u, 1.0);
                buffer_store(readWriteValues, 0u, 2.0);
                buffer_store(attrReadOnly, 0u, 3.0);
                buffer_store(attrWriteOnly, 0u, 4.0);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "[[vk::binding(4, 0)]] StructuredBuffer<float> readOnlyValues "
        ": register(t4);" in generated_code
    )
    assert (
        "[[vk::binding(5, 0)]] RWStructuredBuffer<float> writeOnlyValues "
        ": register(u5);" in generated_code
    )
    assert (
        "[[vk::binding(6, 0)]] RWStructuredBuffer<float> readWriteValues "
        ": register(u6);" in generated_code
    )
    assert (
        "[[vk::binding(7, 0)]] StructuredBuffer<float> attrReadOnly "
        ": register(t7);" in generated_code
    )
    assert (
        "[[vk::binding(8, 0)]] RWStructuredBuffer<float> attrWriteOnly "
        ": register(u8);" in generated_code
    )
    assert (
        "float blockedLoad = /* unsupported Slang structured buffer: "
        "buffer_load requires readable structured buffer resource */ 0;"
        in generated_code
    )
    assert (
        "float blockedAttrLoad = /* unsupported Slang structured buffer: "
        "buffer_load requires readable structured buffer resource */ 0;"
        in generated_code
    )
    assert (
        "/* unsupported Slang structured buffer: buffer_store requires "
        "writable structured buffer resource */;" in generated_code
    )
    assert "readWriteValues.Store(0u, 2.0);" in generated_code
    assert "attrWriteOnly.Store(0u, 4.0);" in generated_code
    assert "writeOnlyValues.Load" not in generated_code
    assert "attrWriteOnly.Load" not in generated_code
    assert "buffer_load(" not in generated_code
    assert "buffer_store(" not in generated_code


def test_structured_buffer_explicit_result_atomics_emit_slang_interlocked():
    code = """
    shader SlangStructuredBufferAtomics {
        RWStructuredBuffer<uint> counters @binding(24);
        RWStructuredBuffer<int> signedCounters @binding(25);
        StructuredBuffer<uint> readOnlyCounters @binding(26);
        RWStructuredBuffer<uint> attrReadOnlyCounters @access(read) @binding(27);
        RWStructuredBuffer<float> floatCounters @binding(28);

        uint fetchOld(uint3 tid) {
            return atomicExchange(counters[tid.x], 9u);
        }

        compute {
            void main(uint3 tid @gl_GlobalInvocationID) {
                uint oldValue;
                int oldSigned;
                uint cgl_atomicAdd_original = 0u;
                atomicAdd(counters[tid.x], 1u, oldValue);
                atomicCompareExchange(counters[tid.x], 2u, 3u, oldValue);
                atomicMax(signedCounters[tid.x], -1, oldSigned);
                uint returnedOld = atomicAdd(counters[tid.x], 1u);
                uint returnedCompare = atomicCompareExchange(counters[tid.x], 2u, 3u);
                uint combined = atomicOr(counters[tid.x], 4u) + 1u;
                int returnedSigned = atomicMin(signedCounters[tid.x], -4);
                oldValue = atomicExchange(counters[tid.x], 7u);
                atomicXor(counters[tid.x], 8u);
                if (atomicAnd(counters[tid.x], 1u) != 0u) {
                    oldValue = fetchOld(tid);
                }
                uint readOnlyBlocked = atomicAdd(readOnlyCounters[tid.x], 1u, oldValue);
                uint attrReadOnlyBlocked = atomicAdd(
                    attrReadOnlyCounters[tid.x],
                    1u,
                    oldValue
                );
                uint floatBlocked = atomicAdd(floatCounters[tid.x], 1.0, oldValue);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "[[vk::binding(24, 0)]] RWStructuredBuffer<uint> counters "
        ": register(u24);" in generated_code
    )
    assert (
        "[[vk::binding(25, 0)]] RWStructuredBuffer<int> signedCounters "
        ": register(u25);" in generated_code
    )
    assert (
        "[[vk::binding(26, 0)]] StructuredBuffer<uint> readOnlyCounters "
        ": register(t26);" in generated_code
    )
    assert (
        "[[vk::binding(27, 0)]] StructuredBuffer<uint> attrReadOnlyCounters "
        ": register(t27);" in generated_code
    )
    assert "InterlockedAdd(counters[tid.x], 1u, oldValue);" in generated_code
    assert (
        "InterlockedCompareExchange(counters[tid.x], 2u, 3u, oldValue);"
        in generated_code
    )
    assert "InterlockedMax(signedCounters[tid.x], -1, oldSigned);" in generated_code
    assert "uint fetchOld(uint3 tid)" in generated_code
    assert "uint cgl_atomicExchange_original;" in generated_code
    assert (
        "InterlockedExchange(counters[tid.x], 9u, "
        "cgl_atomicExchange_original);" in generated_code
    )
    assert "return cgl_atomicExchange_original;" in generated_code
    assert "uint cgl_atomicAdd_original = 0u;" in generated_code
    assert "uint cgl_atomicAdd_original_1;" in generated_code
    assert (
        "InterlockedAdd(counters[tid.x], 1u, cgl_atomicAdd_original_1);"
        in generated_code
    )
    assert "uint returnedOld = cgl_atomicAdd_original_1;" in generated_code
    assert "uint cgl_atomicCompareExchange_original;" in generated_code
    assert (
        "InterlockedCompareExchange(counters[tid.x], 2u, 3u, "
        "cgl_atomicCompareExchange_original);" in generated_code
    )
    assert (
        "uint returnedCompare = cgl_atomicCompareExchange_original;" in generated_code
    )
    assert "uint cgl_atomicOr_original;" in generated_code
    assert (
        "InterlockedOr(counters[tid.x], 4u, cgl_atomicOr_original);" in generated_code
    )
    assert "uint combined = cgl_atomicOr_original + 1u;" in generated_code
    assert "int cgl_atomicMin_original;" in generated_code
    assert (
        "InterlockedMin(signedCounters[tid.x], -4, cgl_atomicMin_original);"
        in generated_code
    )
    assert "int returnedSigned = cgl_atomicMin_original;" in generated_code
    assert generated_code.count("uint cgl_atomicExchange_original;") == 2
    assert "uint cgl_atomicExchange_original_1;" not in generated_code
    assert (
        "InterlockedExchange(counters[tid.x], 7u, "
        "cgl_atomicExchange_original);" in generated_code
    )
    assert "oldValue = cgl_atomicExchange_original;" in generated_code
    assert "uint cgl_atomicXor_original;" in generated_code
    assert (
        "InterlockedXor(counters[tid.x], 8u, cgl_atomicXor_original);" in generated_code
    )
    assert "cgl_atomicXor_original;" not in [
        line.strip() for line in generated_code.splitlines()
    ]
    assert "uint cgl_atomicAnd_original;" in generated_code
    assert (
        "InterlockedAnd(counters[tid.x], 1u, cgl_atomicAnd_original);" in generated_code
    )
    assert "if (cgl_atomicAnd_original != 0u)" in generated_code
    assert "oldValue = fetchOld(tid);" in generated_code
    assert (
        "uint readOnlyBlocked = /* unsupported Slang structured buffer: "
        "atomicAdd requires RWStructuredBuffer resource */ 0u;" in generated_code
    )
    assert (
        "uint attrReadOnlyBlocked = /* unsupported Slang structured buffer: "
        "atomicAdd requires writable structured buffer resource */ 0u;"
        in generated_code
    )
    assert (
        "uint floatBlocked = /* unsupported Slang structured buffer: atomicAdd "
        "requires scalar int or uint RWStructuredBuffer element */ 0u;"
        in generated_code
    )
    assert "atomicAdd(counters" not in generated_code
    assert "atomicCompareExchange(counters" not in generated_code
    assert "atomicMax(signedCounters" not in generated_code
    assert "atomicMin(signedCounters" not in generated_code
    assert "atomicOr(counters" not in generated_code
    assert "atomicExchange(counters" not in generated_code
    assert "atomicXor(counters" not in generated_code
    assert "atomicAnd(counters" not in generated_code


def test_structured_buffer_expression_atomics_in_loop_contexts_emit_diagnostics():
    code = """
    shader SlangStructuredBufferAtomicLoopContexts {
        RWStructuredBuffer<uint> counters @binding(29);

        compute {
            void main(uint3 tid @gl_GlobalInvocationID) {
                uint oldValue = 0u;
                for (
                    uint initOld = atomicAdd(counters[tid.x], 1u);
                    initOld < 2u;
                    initOld = initOld + 1u
                ) {
                    atomicAdd(counters[tid.x], 1u, oldValue);
                }
                for (
                    uint i = 0u;
                    atomicAdd(counters[tid.x], 1u) < 4u;
                    i = atomicExchange(counters[tid.x], i)
                ) {
                    break;
                }
                while (atomicOr(counters[tid.x], 1u) != 0u) {
                    break;
                }
                do {
                    break;
                } while (atomicAnd(counters[tid.x], 1u) != 0u);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "atomicAdd expression-valued atomic cannot be used in "
        "for-loop initializer context; use explicit original output argument"
        in generated_code
    )
    assert (
        "atomicAdd expression-valued atomic cannot be used in "
        "for-loop condition context; use explicit original output argument"
        in generated_code
    )
    assert (
        "atomicExchange expression-valued atomic cannot be used in "
        "for-loop update context; use explicit original output argument"
        in generated_code
    )
    assert (
        "atomicOr expression-valued atomic cannot be used in "
        "while-loop condition context; use explicit original output argument"
        in generated_code
    )
    assert (
        "atomicAnd expression-valued atomic cannot be used in "
        "do-while-loop condition context; use explicit original output argument"
        in generated_code
    )
    assert "InterlockedAdd(counters[tid.x], 1u, oldValue);" in generated_code
    assert "atomicAdd(counters" not in generated_code
    assert "atomicExchange(counters" not in generated_code
    assert "atomicOr(counters" not in generated_code
    assert "atomicAnd(counters" not in generated_code
    assert "cgl_atomic" not in generated_code


def test_append_consume_structured_buffers_emit_slang_methods_and_uav_bindings():
    code = """
    shader SlangAppendConsumeStructuredBuffers {
        struct Particle {
            vec4 position;
            float mass;
        };

        AppendStructuredBuffer<Particle> appended @binding(3);
        ConsumeStructuredBuffer<Particle> consumed @binding(4);
        AppendStructuredBuffer<vec4> vectors;
        ConsumeStructuredBuffer<uint> indices[2];

        compute {
            void main() {
                Particle particle = buffer_consume(consumed);
                buffer_append(appended, particle);
                buffer_append(vectors, vec4(particle.mass));
                uint index = buffer_consume(indices[1]);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "[[vk::binding(3, 0)]] AppendStructuredBuffer<Particle> appended "
        ": register(u3);" in generated_code
    )
    assert (
        "[[vk::binding(4, 0)]] ConsumeStructuredBuffer<Particle> consumed "
        ": register(u4);" in generated_code
    )
    assert (
        "[[vk::binding(5, 0)]] AppendStructuredBuffer<float4> vectors "
        ": register(u5);" in generated_code
    )
    assert (
        "[[vk::binding(6, 0)]] ConsumeStructuredBuffer<uint> indices[2] "
        ": register(u6);" in generated_code
    )
    assert "Particle particle = consumed.Consume();" in generated_code
    assert "appended.Append(particle);" in generated_code
    assert "vectors.Append(float4(particle.mass));" in generated_code
    assert "uint index = indices[1].Consume();" in generated_code
    assert "buffer_append(" not in generated_code
    assert "buffer_consume(" not in generated_code
    assert "AppendStructuredBuffer<vec4>" not in generated_code


def test_append_consume_wrong_structured_buffer_kinds_emit_slang_diagnostics():
    code = """
    shader SlangInvalidAppendConsumeStructuredBuffers {
        RWStructuredBuffer<int> writableValues;
        StructuredBuffer<int> readOnlyValues;

        compute {
            void main() {
                buffer_append(writableValues, 1);
                int value = buffer_consume(readOnlyValues);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "/* unsupported Slang structured buffer: buffer_append requires "
        "AppendStructuredBuffer resource */;" in generated_code
    )
    assert (
        "int value = /* unsupported Slang structured buffer: buffer_consume "
        "requires ConsumeStructuredBuffer resource */ 0;" in generated_code
    )
    assert "buffer_append(" not in generated_code
    assert "buffer_consume(" not in generated_code


def test_append_consume_member_calls_respect_access_qualifiers_and_kinds():
    code = """
    shader SlangAppendConsumeMemberAccessQualifiers {
        struct Particle {
            float mass;
        };

        AppendStructuredBuffer<Particle> appended @binding(14);
        ConsumeStructuredBuffer<Particle> consumed @binding(15);
        readonly AppendStructuredBuffer<Particle> readOnlyAppend @binding(16);
        writeonly ConsumeStructuredBuffer<Particle> writeOnlyConsume @binding(17);
        AppendStructuredBuffer<float> attrReadAppend @access(read) @binding(18);
        ConsumeStructuredBuffer<float> attrWriteConsume @access(write) @binding(19);
        RWStructuredBuffer<Particle> writableParticles;
        StructuredBuffer<Particle> readOnlyParticles;

        compute {
            void main() {
                Particle particle = consumed.Consume();
                appended.Append(particle);
                readOnlyAppend.Append(particle);
                Particle blocked = writeOnlyConsume.Consume();
                attrReadAppend.Append(1.0);
                float attrBlocked = attrWriteConsume.Consume();
                writableParticles.Append(particle);
                Particle wrong = readOnlyParticles.Consume();
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "[[vk::binding(16, 0)]] AppendStructuredBuffer<Particle> "
        "readOnlyAppend : register(u16);" in generated_code
    )
    assert (
        "[[vk::binding(17, 0)]] ConsumeStructuredBuffer<Particle> "
        "writeOnlyConsume : register(u17);" in generated_code
    )
    assert "Particle particle = consumed.Consume();" in generated_code
    assert "appended.Append(particle);" in generated_code
    assert (
        "/* unsupported Slang structured buffer: Append requires writable "
        "structured buffer receiver */;" in generated_code
    )
    assert (
        "Particle blocked = /* unsupported Slang structured buffer: Consume "
        "requires readable structured buffer receiver */ Particle();" in generated_code
    )
    assert (
        "float attrBlocked = /* unsupported Slang structured buffer: Consume "
        "requires readable structured buffer receiver */ 0;" in generated_code
    )
    assert (
        "/* unsupported Slang structured buffer: Append requires "
        "AppendStructuredBuffer receiver */;" in generated_code
    )
    assert (
        "Particle wrong = /* unsupported Slang structured buffer: Consume "
        "requires ConsumeStructuredBuffer receiver */ Particle();" in generated_code
    )
    assert "readOnlyAppend.Append" not in generated_code
    assert "writeOnlyConsume.Consume" not in generated_code
    assert "attrReadAppend.Append" not in generated_code
    assert "attrWriteConsume.Consume" not in generated_code
    assert "writableParticles.Append" not in generated_code
    assert "readOnlyParticles.Consume" not in generated_code


def test_byte_address_buffers_emit_slang_load_store_dimensions_and_bindings():
    code = """
    shader SlangByteAddressBuffers {
        ByteAddressBuffer rawBytes @binding(1);
        RWByteAddressBuffer rawOutput @binding(2);
        RWByteAddressBuffer rawArray[2];

        uint readRaw(ByteAddressBuffer input, uint offset) {
            return buffer_load(input, offset);
        }

        float readFloat(ByteAddressBuffer input, uint offset) {
            return asfloat(buffer_load(input, offset));
        }

        void writeRaw(
            RWByteAddressBuffer output,
            uint offset,
            uint value,
            float weight
        ) {
            buffer_store(output, offset, value);
            buffer_store(output, offset + uint(4), asuint(weight));
        }

        uint updateArray(uint slot, uint offset) {
            uint value = buffer_load(rawArray[slot], offset);
            buffer_store(rawArray[slot], offset, value + uint(1));
            return value;
        }

        uint queryRaw(ByteAddressBuffer input) {
            return buffer_dimensions(input);
        }

        compute {
            void main() {}
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "[[vk::binding(1, 0)]] ByteAddressBuffer rawBytes "
        ": register(t1);" in generated_code
    )
    assert (
        "[[vk::binding(2, 0)]] RWByteAddressBuffer rawOutput "
        ": register(u2);" in generated_code
    )
    assert (
        "[[vk::binding(3, 0)]] RWByteAddressBuffer rawArray[2] "
        ": register(u3);" in generated_code
    )
    assert (
        "uint cgl_bufferDimensions_ByteAddressBuffer(ByteAddressBuffer buffer)"
        in generated_code
    )
    assert "return input.Load(offset);" in generated_code
    assert "return asfloat(input.Load(offset));" in generated_code
    assert "output.Store(offset, value);" in generated_code
    assert "output.Store(offset + uint(4), asuint(weight));" in generated_code
    assert "uint value = rawArray[slot].Load(offset);" in generated_code
    assert "rawArray[slot].Store(offset, value + uint(1));" in generated_code
    assert "return cgl_bufferDimensions_ByteAddressBuffer(input);" in generated_code
    assert "buffer_load(" not in generated_code
    assert "buffer_store(" not in generated_code
    assert "buffer_dimensions(" not in generated_code


def test_byte_address_vector_methods_and_readonly_stores_emit_slang():
    code = """
    shader SlangByteAddressVectorMethods {
        ByteAddressBuffer rawBytes;
        RWByteAddressBuffer rawVectors;

        vec2 readPair(uint offset) {
            return asfloat(rawVectors.Load2(offset));
        }

        vec3 readTriple(uint offset) {
            return asfloat(rawVectors.Load3(offset));
        }

        vec4 readQuad(uint offset) {
            return asfloat(rawVectors.Load4(offset));
        }

        int readSigned(uint offset) {
            return asint(rawVectors.Load(offset));
        }

        void writeVectors(uint offset, vec2 pair, vec3 normal, vec4 color, uint value) {
            rawVectors.Store2(offset, asuint(pair));
            rawVectors.Store3(offset + uint(16), asuint(normal));
            rawVectors.Store4(offset + uint(32), asuint(color));
            rawBytes.Store(offset, value);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "ByteAddressBuffer rawBytes : register(t0);" in generated_code
    assert "RWByteAddressBuffer rawVectors : register(u0);" in generated_code
    assert "return asfloat(rawVectors.Load2(offset));" in generated_code
    assert "return asfloat(rawVectors.Load3(offset));" in generated_code
    assert "return asfloat(rawVectors.Load4(offset));" in generated_code
    assert "return asint(rawVectors.Load(offset));" in generated_code
    assert "rawVectors.Store2(offset, asuint(pair));" in generated_code
    assert "rawVectors.Store3(offset + uint(16), asuint(normal));" in generated_code
    assert "rawVectors.Store4(offset + uint(32), asuint(color));" in generated_code
    assert (
        "/* unsupported Slang byte-address buffer: Store requires "
        "RWByteAddressBuffer receiver */;" in generated_code
    )
    assert "None(" not in generated_code


def test_byte_address_buffer_store_to_readonly_helper_emits_slang_diagnostic():
    code = """
    shader SlangReadonlyByteAddressBuffer {
        ByteAddressBuffer rawBytes;

        void invalidRaw(uint offset, uint value) {
            buffer_store(rawBytes, offset, value);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "/* unsupported Slang byte-address buffer: buffer_store requires "
        "RWByteAddressBuffer resource */;" in generated_code
    )
    assert "buffer_store(" not in generated_code


def test_byte_address_buffer_access_qualifiers_emit_slang_kinds_and_diagnostics():
    code = """
    shader SlangByteAddressBufferAccessQualifiers {
        readonly RWByteAddressBuffer readOnlyRaw @binding(9);
        writeonly ByteAddressBuffer writeOnlyRaw @binding(10);
        readwrite ByteAddressBuffer readWriteRaw @binding(11);
        RWByteAddressBuffer attrReadOnlyRaw @access(read) @binding(12);
        ByteAddressBuffer attrWriteOnlyRaw @access(write) @binding(13);

        void main(uint offset) {
            uint blockedLoad = buffer_load(writeOnlyRaw, offset);
            uint blockedAttrLoad = attrWriteOnlyRaw.Load(offset);
            buffer_store(readOnlyRaw, offset, uint(1));
            attrReadOnlyRaw.Store(offset, uint(2));
            buffer_store(readWriteRaw, offset, uint(3));
            attrWriteOnlyRaw.Store(offset, uint(4));
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "[[vk::binding(9, 0)]] ByteAddressBuffer readOnlyRaw "
        ": register(t9);" in generated_code
    )
    assert (
        "[[vk::binding(10, 0)]] RWByteAddressBuffer writeOnlyRaw "
        ": register(u10);" in generated_code
    )
    assert (
        "[[vk::binding(11, 0)]] RWByteAddressBuffer readWriteRaw "
        ": register(u11);" in generated_code
    )
    assert (
        "[[vk::binding(12, 0)]] ByteAddressBuffer attrReadOnlyRaw "
        ": register(t12);" in generated_code
    )
    assert (
        "[[vk::binding(13, 0)]] RWByteAddressBuffer attrWriteOnlyRaw "
        ": register(u13);" in generated_code
    )
    assert (
        "uint blockedLoad = /* unsupported Slang byte-address buffer: "
        "buffer_load requires readable byte-address buffer resource */ 0u;"
        in generated_code
    )
    assert (
        "uint blockedAttrLoad = /* unsupported Slang byte-address buffer: "
        "Load requires readable byte-address buffer receiver */ 0u;" in generated_code
    )
    assert (
        "/* unsupported Slang byte-address buffer: buffer_store requires "
        "writable byte-address buffer resource */;" in generated_code
    )
    assert (
        "/* unsupported Slang byte-address buffer: Store requires "
        "writable byte-address buffer receiver */;" in generated_code
    )
    assert "readWriteRaw.Store(offset, uint(3));" in generated_code
    assert "attrWriteOnlyRaw.Store(offset, uint(4));" in generated_code
    assert "writeOnlyRaw.Load" not in generated_code
    assert "attrWriteOnlyRaw.Load" not in generated_code
    assert "buffer_load(" not in generated_code
    assert "buffer_store(" not in generated_code


def test_byte_address_interlocked_member_calls_respect_access_qualifiers():
    code = """
    shader SlangByteAddressInterlockedMembers {
        struct Helper {
            uint value;
        };

        RWByteAddressBuffer rawOutput @binding(20);
        ByteAddressBuffer readOnlyRaw @binding(21);
        RWByteAddressBuffer attrReadOnlyRaw @access(read) @binding(22);
        ByteAddressBuffer attrWriteOnlyRaw @access(write) @binding(23);
        Helper helper;

        void main(uint offset, uint value) {
            uint oldValue;
            rawOutput.InterlockedAdd(offset, value, oldValue);
            rawOutput.InterlockedCompareExchange(offset + 4u, 1u, value, oldValue);
            attrWriteOnlyRaw.InterlockedExchange(offset + 8u, value, oldValue);
            readOnlyRaw.InterlockedOr(offset, value, oldValue);
            attrReadOnlyRaw.InterlockedXor(offset, value, oldValue);
            helper.InterlockedAdd(value);
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "[[vk::binding(20, 0)]] RWByteAddressBuffer rawOutput "
        ": register(u20);" in generated_code
    )
    assert (
        "[[vk::binding(21, 0)]] ByteAddressBuffer readOnlyRaw "
        ": register(t21);" in generated_code
    )
    assert (
        "[[vk::binding(22, 0)]] ByteAddressBuffer attrReadOnlyRaw "
        ": register(t22);" in generated_code
    )
    assert (
        "[[vk::binding(23, 0)]] RWByteAddressBuffer attrWriteOnlyRaw "
        ": register(u23);" in generated_code
    )
    assert "rawOutput.InterlockedAdd(offset, value, oldValue);" in generated_code
    assert (
        "rawOutput.InterlockedCompareExchange(offset + 4u, 1u, value, oldValue);"
        in generated_code
    )
    assert (
        "attrWriteOnlyRaw.InterlockedExchange(offset + 8u, value, oldValue);"
        in generated_code
    )
    assert (
        "/* unsupported Slang byte-address buffer: InterlockedOr requires "
        "RWByteAddressBuffer receiver */;" in generated_code
    )
    assert (
        "/* unsupported Slang byte-address buffer: InterlockedXor requires "
        "writable byte-address buffer receiver */;" in generated_code
    )
    assert "helper.InterlockedAdd(value);" in generated_code
    assert "readOnlyRaw.InterlockedOr" not in generated_code
    assert "attrReadOnlyRaw.InterlockedXor" not in generated_code
    assert "None(" not in generated_code


@pytest.mark.parametrize(
    ("resource_source", "message"),
    [
        (
            """
            sampler2d textures[2] @binding(0);
            sampler2d overlap @binding(1);
            """,
            "t1 overlaps 'textures' t0-t1",
        ),
        (
            """
            sampler2d first @binding(0);
            sampler2d second @binding(0);
            """,
            "t0 overlaps 'first' t0",
        ),
    ],
)
def test_conflicting_slang_resource_binding_ranges_raise(resource_source, message):
    code = f"""
    shader InvalidResourceBindingRanges {{
        {resource_source}

        compute {{
            void main() {{}}
        }}
    }}
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    with pytest.raises(ValueError, match=message):
        generate_code(ast)


def test_one_dimensional_resources_emit_slang_methods_and_helpers():
    code = """
    shader OneDResources {
        sampler1d line;
        sampler1darray lineLayers;
        image1D values @r32f;
        image1DArray layerValues @rgba16f;
        uimage1D counters @r32ui;
        uimage1DArray layerCounters @r32ui;

        compute {
            vec4 sampleLine(
                sampler1D tex,
                sampler1DArray layers,
                float u,
                vec2 uvLayer,
                int pixel,
                ivec2 pixelLayer,
                int lod
            ) {
                vec4 filtered = texture(tex, u);
                vec4 layered = texture(layers, uvLayer);
                vec4 fetched = texelFetch(tex, pixel, lod);
                vec4 fetchedLayer = texelFetch(layers, pixelLayer, lod);
                return filtered + layered + fetched + fetchedLayer;
            }

            void main() {
                int lineSize = textureSize(line, 0);
                ivec2 layerSize = textureSize(lineLayers, 0);
                float oldValue = imageLoad(values, 3);
                imageStore(values, 4, oldValue + 1.0);
                vec4 oldLayer = imageLoad(layerValues, ivec2(2, 1));
                imageStore(layerValues, ivec2(3, 1), oldLayer);
                int valueSize = imageSize(values);
                ivec2 layerValueSize = imageSize(layerValues);
                uint previous = imageAtomicAdd(counters, 2, 1u);
                uint layerPrevious = imageAtomicExchange(
                    layerCounters,
                    ivec2(2, 1),
                    previous
                );
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Sampler1D<float4> line : register(t0);" in generated_code
    assert "Sampler1DArray<float4> lineLayers : register(t1);" in generated_code
    assert "RWTexture1D<float> values : register(u0);" in generated_code
    assert "RWTexture1DArray<float4> layerValues : register(u1);" in generated_code
    assert "RWTexture1D<uint> counters : register(u2);" in generated_code
    assert "RWTexture1DArray<uint> layerCounters : register(u3);" in generated_code
    assert (
        "float4 sampleLine(Sampler1D<float4> tex, "
        "Sampler1DArray<float4> layers, float u, float2 uvLayer, "
        "int pixel, int2 pixelLayer, int lod)" in generated_code
    )
    assert "float4 filtered = tex.Sample(u);" in generated_code
    assert "float4 layered = layers.Sample(uvLayer);" in generated_code
    assert "float4 fetched = tex.Load(int2(pixel, lod));" in generated_code
    assert "float4 fetchedLayer = layers.Load(int3(pixelLayer, lod));" in generated_code
    assert (
        "int cgl_textureSize_sampler1D(Sampler1D<float4> tex, uint mipLevel)"
        in generated_code
    )
    assert (
        "int2 cgl_textureSize_sampler1DArray("
        "Sampler1DArray<float4> tex, uint mipLevel)" in generated_code
    )
    assert (
        "int cgl_imageSize_image1D_RWTexture1D_float(RWTexture1D<float> tex)"
        in generated_code
    )
    assert (
        "int2 cgl_imageSize_image1DArray(RWTexture1DArray<float4> tex)"
        in generated_code
    )
    assert "float oldValue = values[3];" in generated_code
    assert "values[4] = oldValue + 1.0;" in generated_code
    assert "float4 oldLayer = layerValues[int2(2, 1)];" in generated_code
    assert "layerValues[int2(3, 1)] = oldLayer;" in generated_code
    assert (
        "int valueSize = cgl_imageSize_image1D_RWTexture1D_float(values);"
        in generated_code
    )
    assert (
        "int2 layerValueSize = cgl_imageSize_image1DArray(layerValues);"
        in generated_code
    )
    assert "uint cgl_imageAtomicAdd_original;" in generated_code
    assert (
        "InterlockedAdd(counters[2], 1u, cgl_imageAtomicAdd_original);"
        in generated_code
    )
    assert "uint previous = cgl_imageAtomicAdd_original;" in generated_code
    assert "uint cgl_imageAtomicExchange_original;" in generated_code
    assert (
        "InterlockedExchange(layerCounters[int2(2, 1)], previous, "
        "cgl_imageAtomicExchange_original);" in generated_code
    )
    assert "uint layerPrevious = cgl_imageAtomicExchange_original;" in generated_code
    assert "sampler1d" not in generated_code
    assert "imageAtomicAdd(counters" not in generated_code
    assert "imageAtomicExchange(layerCounters" not in generated_code
    assert "uint cgl_imageAtomicAdd_uimage1D(" not in generated_code
    assert "uint cgl_imageAtomicExchange_uimage1DArray(" not in generated_code


def test_non_resource_arrays_preserve_expression_sizes():
    code = """
    shader ArraySizes {
        vec3 colors[(2 + 1) * 2];

        float accumulate(float values[(2 + 1) * 2], vec3 normals[+6]) {
            float localWeights[(2 + 1) * 2];
            return values[0] + localWeights[1] + normals[0].x;
        }

        compute {
            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float3 colors[((2 + 1) * 2)];" in generated_code
    assert (
        "float accumulate(float values[((2 + 1) * 2)], float3 normals[+6])"
        in generated_code
    )
    assert "float localWeights[((2 + 1) * 2)];" in generated_code
    assert "2 + 1 * 2" not in generated_code


def test_sampled_texture_builtins_emit_slang_methods():
    code = """
    shader Resources {
        sampler2d colorMap;
        sampler2darray layerMap;
        sampler2dms msTex;
        sampler2dmsarray msArray;

        compute {
            vec4 sampleTexture(
                sampler2d tex,
                sampler2darray layers,
                sampler2dms ms,
                sampler2dmsarray msLayers,
                vec2 uv,
                vec3 uvLayer,
                ivec2 pixel,
                ivec3 pixelLayer,
                int sampleIndex
            ) {
                vec4 color = texture(tex, uv);
                vec4 biased = texture(tex, uv, 0.5);
                vec4 mip = textureLod(tex, uv, 2.0);
                vec4 grad = textureGrad(
                    tex,
                    uv,
                    vec2(0.1, 0.0),
                    vec2(0.0, 0.1)
                );
                vec4 fetched = texelFetch(tex, pixel, 1);
                vec4 fetchedLayer = texelFetch(layers, pixelLayer, 1);
                vec4 msColor = texelFetch(ms, pixel, sampleIndex);
                vec4 msLayer = texelFetch(msLayers, pixelLayer, sampleIndex);
                return color + biased + mip + grad + fetched
                    + fetchedLayer + msColor + msLayer;
            }

            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Sampler2D<float4> colorMap : register(t0);" in generated_code
    assert "Sampler2DArray<float4> layerMap : register(t1);" in generated_code
    assert "Sampler2DMS<float4> msTex : register(t2);" in generated_code
    assert "Sampler2DMSArray<float4> msArray : register(t3);" in generated_code
    assert "Sampler2D<float4> tex" in generated_code
    assert "Sampler2DArray<float4> layers" in generated_code
    assert "Sampler2DMS<float4> ms" in generated_code
    assert "Sampler2DMSArray<float4> msLayers" in generated_code
    assert "float4 color = tex.Sample(uv);" in generated_code
    assert "float4 biased = tex.SampleBias(uv, 0.5);" in generated_code
    assert "float4 mip = tex.SampleLevel(uv, 2.0);" in generated_code
    assert (
        "float4 grad = tex.SampleGrad(uv, float2(0.1, 0.0), float2(0.0, 0.1));"
        in generated_code
    )
    assert "float4 fetched = tex.Load(int3(pixel, 1));" in generated_code
    assert "float4 fetchedLayer = layers.Load(int4(pixelLayer, 1));" in generated_code
    assert "float4 msColor = ms[pixel, sampleIndex];" in generated_code
    assert "float4 msLayer = msLayers[pixelLayer, sampleIndex];" in generated_code
    assert "texture(" not in generated_code
    assert "textureLod(" not in generated_code
    assert "textureGrad(" not in generated_code
    assert "texelFetch(" not in generated_code


def test_explicit_sampler_texture_builtins_emit_combined_slang_methods():
    code = """
    shader Resources {
        sampler linearSampler;
        sampler2d colorMap;
        sampler2darray layerMap;
        sampler3d volumeMap;

        compute {
            vec4 sampleExplicit(
                sampler2d tex,
                sampler2darray layers,
                sampler3d volume,
                sampler sampleState,
                vec2 uv,
                vec3 uvw,
                vec2 ddx,
                vec2 ddy,
                vec3 ddx3,
                vec3 ddy3
            ) {
                vec4 color = texture(tex, sampleState, uv);
                vec4 biased = texture(tex, sampleState, uv, 0.5);
                vec4 mip = textureLod(tex, sampleState, uv, 2.0);
                vec4 grad = textureGrad(tex, sampleState, uv, ddx, ddy);
                vec4 layer = texture(layers, sampleState, uvw);
                vec4 volumeGrad = textureGrad(volume, sampleState, uvw, ddx3, ddy3);
                return color + biased + mip + grad + layer + volumeGrad;
            }

            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert "SamplerState sampleState" in generated_code
    assert "float4 color = tex.Sample(uv);" in generated_code
    assert "float4 biased = tex.SampleBias(uv, 0.5);" in generated_code
    assert "float4 mip = tex.SampleLevel(uv, 2.0);" in generated_code
    assert "float4 grad = tex.SampleGrad(uv, ddx, ddy);" in generated_code
    assert "float4 layer = layers.Sample(uvw);" in generated_code
    assert "float4 volumeGrad = volume.SampleGrad(uvw, ddx3, ddy3);" in generated_code
    assert "tex.Sample(sampleState" not in generated_code
    assert "tex.SampleBias(sampleState" not in generated_code
    assert "tex.SampleLevel(sampleState" not in generated_code
    assert "tex.SampleGrad(sampleState" not in generated_code
    assert "texture(" not in generated_code
    assert "textureLod(" not in generated_code
    assert "textureGrad(" not in generated_code


def test_malformed_sampled_texture_ops_emit_slang_diagnostics():
    code = """
    shader InvalidSampledTextureOps {
        sampler querySampler;
        sampler2d colorMap;
        sampler2dshadow shadowMap;
        sampler2dms msTex;
        samplercube cubeTex;
        image2D colorImage;

        compute {
            void main() {
                float scalarCoord = 0.25;
                vec2 uv = vec2(0.25, 0.5);
                vec3 badUv = vec3(0.25, 0.5, 0.75);
                vec2 ddx = vec2(0.1, 0.0);
                vec2 ddy = vec2(0.0, 0.1);
                ivec2 pixel = ivec2(2, 3);
                ivec3 badPixel = ivec3(2, 3, 4);
                vec2 badLod = vec2(0.0, 1.0);
                ivec2 badSampleIndex = ivec2(0, 1);
                vec4 missingTexture = texture();
                vec4 missingCoord = texture(colorMap);
                vec4 explicitMissingCoord = texture(colorMap, querySampler);
                vec4 extraBias = texture(colorMap, uv, 0.5, 1.0);
                vec4 samplerOnly = texture(querySampler, uv);
                vec4 imageSample = texture(colorImage, uv);
                vec4 shadowSample = texture(shadowMap, uv);
                vec4 msSample = texture(msTex, uv);
                vec4 missingLod = textureLod(colorMap, uv);
                vec4 extraLod = textureLod(colorMap, querySampler, uv, 1.0, 2.0);
                vec4 missingGrad = textureGrad(colorMap, uv, ddx);
                vec4 extraGrad = textureGrad(
                    colorMap,
                    querySampler,
                    uv,
                    ddx,
                    ddy,
                    ddx
                );
                vec4 badCoordRank = texture(colorMap, scalarCoord);
                vec4 badExplicitCoordRank = textureLod(
                    colorMap,
                    querySampler,
                    scalarCoord,
                    1.0
                );
                vec4 badBiasRank = texture(colorMap, uv, badUv);
                vec4 badLodCoordRank = textureLod(colorMap, badUv, 1.0);
                vec4 badLodRank = textureLod(colorMap, uv, badLod);
                vec4 badGradCoordRank = textureGrad(colorMap, scalarCoord, ddx, ddy);
                vec4 badGradRank = textureGrad(colorMap, uv, badUv, ddy);
                vec4 missingFetchLevel = texelFetch(colorMap, pixel);
                vec4 extraFetchLevel = texelFetch(
                    colorMap,
                    querySampler,
                    pixel,
                    0,
                    1
                );
                vec4 fetchSamplerOnly = texelFetch(querySampler, pixel, 0);
                vec4 fetchImage = texelFetch(colorImage, pixel, 0);
                vec4 fetchShadow = texelFetch(shadowMap, pixel, 0);
                vec4 fetchCube = texelFetch(cubeTex, pixel, 0);
                vec4 badFetchCoordRank = texelFetch(colorMap, badPixel, 0);
                vec4 badFetchIndexRank = texelFetch(msTex, pixel, badSampleIndex);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float4 missingTexture = /* unsupported Slang sampled texture: "
        "texture requires texture and coordinate arguments */ float4(0.0);"
        in generated_code
    )
    assert (
        "float4 missingCoord = /* unsupported Slang sampled texture: "
        "texture requires texture and coordinate arguments */ float4(0.0);"
        in generated_code
    )
    assert (
        "float4 explicitMissingCoord = /* unsupported Slang sampled texture: "
        "texture requires texture and coordinate arguments */ float4(0.0);"
        in generated_code
    )
    assert (
        "float4 extraBias = /* unsupported Slang sampled texture: "
        "texture accepts coordinate and optional bias arguments */ float4(0.0);"
        in generated_code
    )
    assert (
        "float4 samplerOnly = /* unsupported Slang sampled texture: "
        "texture requires a non-shadow non-multisampled sampled texture resource */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 imageSample = /* unsupported Slang sampled texture: "
        "texture requires a non-shadow non-multisampled sampled texture resource */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 shadowSample = /* unsupported Slang sampled texture: "
        "texture requires a non-shadow non-multisampled sampled texture resource */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 msSample = /* unsupported Slang sampled texture: "
        "texture requires a non-shadow non-multisampled sampled texture resource */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 missingLod = /* unsupported Slang sampled texture: "
        "textureLod requires one lod argument */ float4(0.0);" in generated_code
    )
    assert (
        "float4 extraLod = /* unsupported Slang sampled texture: "
        "textureLod requires one lod argument */ float4(0.0);" in generated_code
    )
    assert (
        "float4 missingGrad = /* unsupported Slang sampled texture: "
        "textureGrad requires gradient x and gradient y arguments */ float4(0.0);"
        in generated_code
    )
    assert (
        "float4 extraGrad = /* unsupported Slang sampled texture: "
        "textureGrad requires gradient x and gradient y arguments */ float4(0.0);"
        in generated_code
    )
    assert (
        "float4 badCoordRank = /* unsupported Slang sampled texture: "
        "texture requires a 2-component coordinate for sampler2D */ float4(0.0);"
        in generated_code
    )
    assert (
        "float4 badExplicitCoordRank = /* unsupported Slang sampled texture: "
        "textureLod requires a 2-component coordinate for sampler2D */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 badBiasRank = /* unsupported Slang sampled texture: "
        "texture requires a scalar bias argument */ float4(0.0);" in generated_code
    )
    assert (
        "float4 badLodCoordRank = /* unsupported Slang sampled texture: "
        "textureLod requires a 2-component coordinate for sampler2D */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 badLodRank = /* unsupported Slang sampled texture: "
        "textureLod requires a scalar lod argument */ float4(0.0);" in generated_code
    )
    assert (
        "float4 badGradCoordRank = /* unsupported Slang sampled texture: "
        "textureGrad requires a 2-component coordinate for sampler2D */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 badGradRank = /* unsupported Slang sampled texture: "
        "textureGrad requires a 2-component gradient for sampler2D */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 missingFetchLevel = /* unsupported Slang sampled texture: "
        "texelFetch requires one lod/sample argument */ float4(0.0);" in generated_code
    )
    assert (
        "float4 extraFetchLevel = /* unsupported Slang sampled texture: "
        "texelFetch requires one lod/sample argument */ float4(0.0);" in generated_code
    )
    assert (
        "float4 fetchSamplerOnly = /* unsupported Slang sampled texture: "
        "texelFetch requires a non-shadow texel-fetchable sampled texture resource */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 fetchImage = /* unsupported Slang sampled texture: "
        "texelFetch requires a non-shadow texel-fetchable sampled texture resource */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 fetchShadow = /* unsupported Slang sampled texture: "
        "texelFetch requires a non-shadow texel-fetchable sampled texture resource */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 fetchCube = /* unsupported Slang sampled texture: "
        "texelFetch requires a non-shadow texel-fetchable sampled texture resource */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 badFetchCoordRank = /* unsupported Slang sampled texture: "
        "texelFetch requires a 2-component coordinate for sampler2D */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 badFetchIndexRank = /* unsupported Slang sampled texture: "
        "texelFetch requires a scalar fetch index argument */ float4(0.0);"
        in generated_code
    )
    assert generated_code.count("unsupported Slang sampled texture") == 27
    assert "texture(" not in generated_code
    assert "textureLod(" not in generated_code
    assert "textureGrad(" not in generated_code
    assert "texelFetch(" not in generated_code
    assert "querySampler.Sample" not in generated_code
    assert "colorImage.Sample" not in generated_code


def test_texture_and_shadow_arrays_preserve_expression_sizes_and_group_indices():
    code = """
    shader TextureArrays {
        sampler2d textures[(2 + 1) * 2];
        sampler samplers[(2 + 1) * 2];
        sampler2d unaryTextures[+6];
        sampler2dshadow shadowMaps[(2 + 1) * 2];
        sampler shadowSamplers[(2 + 1) * 2];

        compute {
            vec4 sampleLayer(
                sampler2d textures[(2 + 1) * 2],
                sampler samplers[(2 + 1) * 2],
                sampler2d unaryTextures[+6],
                vec2 uv,
                vec2 ddx,
                vec2 ddy
            ) {
                vec4 color = texture(textures[1 + 2], samplers[1 + 2], uv);
                vec4 lodColor = textureLod(textures[2], samplers[2], uv, 1.0);
                vec4 gradColor = textureGrad(textures[2], samplers[2], uv, ddx, ddy);
                vec4 gathered = textureGather(textures[1 + 2], samplers[1 + 2], uv);
                vec4 unaryColor = texture(unaryTextures[2], uv);
                return color + lodColor + gradColor + gathered + unaryColor;
            }

            float shadowLayer(
                sampler2dshadow shadowMaps[(2 + 1) * 2],
                sampler shadowSamplers[(2 + 1) * 2],
                vec2 uv,
                float depth
            ) {
                float compared = textureCompare(
                    shadowMaps[1 + 2],
                    shadowSamplers[1 + 2],
                    uv,
                    depth
                );
                vec4 gathered = textureGatherCompare(
                    shadowMaps[2],
                    shadowSamplers[2],
                    uv,
                    depth
                );
                return compared + gathered.x;
            }

            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Sampler2D<float4> textures[((2 + 1) * 2)] : register(t0);" in generated_code
    assert "SamplerState samplers[((2 + 1) * 2)] : register(s0);" in generated_code
    assert "Sampler2D<float4> unaryTextures[+6] : register(t6);" in generated_code
    assert (
        "Sampler2DShadow shadowMaps[((2 + 1) * 2)] : register(t12);" in generated_code
    )
    assert (
        "SamplerState shadowSamplers[((2 + 1) * 2)] : register(s6);" in generated_code
    )
    assert (
        "float4 sampleLayer(Sampler2D<float4> textures[((2 + 1) * 2)], "
        "SamplerState samplers[((2 + 1) * 2)], "
        "Sampler2D<float4> unaryTextures[+6], float2 uv, float2 ddx, float2 ddy)"
        in generated_code
    )
    assert (
        "float shadowLayer(Sampler2DShadow shadowMaps[((2 + 1) * 2)], "
        "SamplerState shadowSamplers[((2 + 1) * 2)], float2 uv, float depth)"
        in generated_code
    )
    assert "float4 color = textures[(1 + 2)].Sample(uv);" in generated_code
    assert "float4 lodColor = textures[2].SampleLevel(uv, 1.0);" in generated_code
    assert "float4 gradColor = textures[2].SampleGrad(uv, ddx, ddy);" in generated_code
    assert "float4 gathered = textures[(1 + 2)].Gather(uv);" in generated_code
    assert "float4 unaryColor = unaryTextures[2].Sample(uv);" in generated_code
    assert (
        "float compared = shadowMaps[(1 + 2)].SampleCmp(uv, depth);" in generated_code
    )
    assert "float4 gathered = shadowMaps[2].GatherCmp(uv, depth);" in generated_code
    assert "1 + 2]." not in generated_code
    assert "2 + 1 * 2" not in generated_code
    assert ".Sample(samplers" not in generated_code
    assert ".SampleCmp(shadowSamplers" not in generated_code
    assert "texture(" not in generated_code
    assert "textureLod(" not in generated_code
    assert "textureGrad(" not in generated_code
    assert "textureGather(" not in generated_code
    assert "textureCompare(" not in generated_code


def test_texture_offset_builtins_emit_slang_offset_methods():
    code = """
    shader Resources {
        sampler linearSampler;
        sampler2d colorMap;
        sampler2darray layerMap;

        compute {
            vec4 offsetOps(
                sampler2d tex,
                sampler2darray layers,
                sampler sampleState,
                vec2 uv,
                vec3 uvw,
                vec2 ddx,
                vec2 ddy,
                ivec2 offset
            ) {
                vec4 offsetColor = textureOffset(tex, uv, offset);
                vec4 explicitOffset = textureOffset(
                    tex,
                    sampleState,
                    uv,
                    offset
                );
                vec4 arrayOffset = textureOffset(layers, sampleState, uvw, offset);
                vec4 lodOffset = textureLodOffset(tex, uv, 2.0, offset);
                vec4 explicitLodOffset = textureLodOffset(
                    tex,
                    sampleState,
                    uv,
                    3.0,
                    offset
                );
                vec4 gradOffset = textureGradOffset(tex, uv, ddx, ddy, offset);
                vec4 explicitGradOffset = textureGradOffset(
                    tex,
                    sampleState,
                    uv,
                    ddx,
                    ddy,
                    offset
                );
                return offsetColor
                    + explicitOffset
                    + arrayOffset
                    + lodOffset
                    + explicitLodOffset
                    + gradOffset
                    + explicitGradOffset;
            }

            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Sampler2D<float4> tex" in generated_code
    assert "Sampler2DArray<float4> layers" in generated_code
    assert "SamplerState sampleState" in generated_code
    assert "float4 offsetColor = tex.Sample(uv, offset);" in generated_code
    assert "float4 explicitOffset = tex.Sample(uv, offset);" in generated_code
    assert "float4 arrayOffset = layers.Sample(uvw, offset);" in generated_code
    assert "float4 lodOffset = tex.SampleLevel(uv, 2.0, offset);" in generated_code
    assert (
        "float4 explicitLodOffset = tex.SampleLevel(uv, 3.0, offset);" in generated_code
    )
    assert "float4 gradOffset = tex.SampleGrad(uv, ddx, ddy, offset);" in generated_code
    assert (
        "float4 explicitGradOffset = tex.SampleGrad(uv, ddx, ddy, offset);"
        in generated_code
    )
    assert "textureOffset(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code
    assert ".Sample(sampleState" not in generated_code
    assert ".SampleLevel(sampleState" not in generated_code
    assert ".SampleGrad(sampleState" not in generated_code


def test_texture_offset_invalid_slang_calls_emit_diagnostic_stubs():
    code = """
    shader Resources {
        sampler querySampler;
        sampler2d colorMap;
        sampler2dshadow shadowMap;
        sampler2dms msTex;
        image2D colorImage;

        compute {
            void main() {
                float scalarCoord = 0.25;
                vec2 uv = vec2(0.25, 0.75);
                vec3 badUv = vec3(0.25, 0.75, 0.5);
                vec2 ddx = vec2(0.1, 0.0);
                ivec2 offset = ivec2(1, 0);
                ivec3 badOffset = ivec3(1, 0, 0);
                vec4 missingOffset = textureOffset(colorMap, uv);
                vec4 missingLodOffset = textureLodOffset(colorMap, uv, offset);
                vec4 missingGradOffset = textureGradOffset(colorMap, uv, ddx, offset);
                vec4 samplerOffset = textureOffset(querySampler, uv, offset);
                vec4 imageOffset = textureLodOffset(colorImage, uv, 1.0, offset);
                vec4 shadowOffset = textureGradOffset(
                    shadowMap,
                    uv,
                    ddx,
                    ddx,
                    offset
                );
                vec4 multisampleOffset = textureOffset(msTex, uv, offset);
                vec4 badCoordRank = textureOffset(colorMap, scalarCoord, offset);
                vec4 badOffsetRank = textureLodOffset(colorMap, uv, 1.0, badOffset);
                vec4 badGradRank = textureGradOffset(
                    colorMap,
                    uv,
                    badUv,
                    ddx,
                    offset
                );
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float4 missingOffset = /* unsupported Slang texture offset: "
        "textureOffset requires one offset argument */ float4(0.0);" in generated_code
    )
    assert (
        "float4 missingLodOffset = /* unsupported Slang texture offset: "
        "textureLodOffset requires lod and offset arguments */ float4(0.0);"
        in generated_code
    )
    assert (
        "float4 missingGradOffset = /* unsupported Slang texture offset: "
        "textureGradOffset requires gradient x, gradient y, and offset arguments */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 samplerOffset = /* unsupported Slang texture offset: "
        "textureOffset requires a non-shadow non-multisampled sampled texture "
        "resource */ float4(0.0);" in generated_code
    )
    assert (
        "float4 imageOffset = /* unsupported Slang texture offset: "
        "textureLodOffset requires a non-shadow non-multisampled sampled texture "
        "resource */ float4(0.0);" in generated_code
    )
    assert (
        "float4 shadowOffset = /* unsupported Slang texture offset: "
        "textureGradOffset requires a non-shadow non-multisampled sampled texture "
        "resource */ float4(0.0);" in generated_code
    )
    assert (
        "float4 multisampleOffset = /* unsupported Slang texture offset: "
        "textureOffset requires a non-shadow non-multisampled sampled texture "
        "resource */ float4(0.0);" in generated_code
    )
    assert (
        "float4 badCoordRank = /* unsupported Slang texture offset: "
        "textureOffset requires a 2-component coordinate for sampler2D */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 badOffsetRank = /* unsupported Slang texture offset: "
        "textureLodOffset requires a 2-component offset for sampler2D */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 badGradRank = /* unsupported Slang texture offset: "
        "textureGradOffset requires a 2-component gradient for sampler2D */ "
        "float4(0.0);" in generated_code
    )
    assert "textureOffset(" not in generated_code
    assert "textureLodOffset(" not in generated_code
    assert "textureGradOffset(" not in generated_code
    assert "querySampler.Sample" not in generated_code
    assert "colorImage.Sample" not in generated_code
    assert "shadowMap.Sample" not in generated_code
    assert "msTex.Sample" not in generated_code


def test_projected_texture_builtins_emit_slang_projected_samples():
    code = """
    shader Resources {
        sampler linearSampler;
        sampler2d colorMap;
        sampler3d volumeMap;

        compute {
            vec4 projectedOps(
                sampler2d tex,
                sampler3d volume,
                sampler sampleState,
                vec3 uvq,
                vec4 uvqw,
                vec4 xyzq,
                vec2 ddx,
                vec2 ddy,
                ivec2 offset
            ) {
                vec4 projected = textureProj(tex, uvq);
                vec4 explicitProjected = textureProj(tex, sampleState, uvqw, 0.25);
                vec4 volumeProjected = textureProj(volume, xyzq);
                vec4 projectedOffset = textureProjOffset(tex, uvq, offset);
                vec4 projectedOffsetBias = textureProjOffset(
                    tex,
                    sampleState,
                    uvq,
                    offset,
                    0.5
                );
                vec4 projectedLod = textureProjLod(tex, sampleState, uvq, 2.0);
                vec4 projectedLodOffset = textureProjLodOffset(
                    tex,
                    uvq,
                    3.0,
                    offset
                );
                vec4 projectedGrad = textureProjGrad(tex, uvq, ddx, ddy);
                vec4 projectedGradOffset = textureProjGradOffset(
                    tex,
                    sampleState,
                    uvq,
                    ddx,
                    ddy,
                    offset
                );
                return projected
                    + explicitProjected
                    + volumeProjected
                    + projectedOffset
                    + projectedOffsetBias
                    + projectedLod
                    + projectedLodOffset
                    + projectedGrad
                    + projectedGradOffset;
            }

            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Sampler2D<float4> tex" in generated_code
    assert "Sampler3D<float4> volume" in generated_code
    assert "SamplerState sampleState" in generated_code
    assert "float4 projected = tex.Sample(uvq.xy / uvq.z);" in generated_code
    assert (
        "float4 explicitProjected = tex.SampleBias(uvqw.xy / uvqw.w, 0.25);"
        in generated_code
    )
    assert (
        "float4 volumeProjected = volume.Sample(xyzq.xyz / xyzq.w);" in generated_code
    )
    assert (
        "float4 projectedOffset = tex.Sample(uvq.xy / uvq.z, offset);" in generated_code
    )
    assert (
        "float4 projectedOffsetBias = tex.SampleBias("
        "uvq.xy / uvq.z, 0.5, offset);" in generated_code
    )
    assert (
        "float4 projectedLod = tex.SampleLevel(uvq.xy / uvq.z, 2.0);" in generated_code
    )
    assert (
        "float4 projectedLodOffset = tex.SampleLevel("
        "uvq.xy / uvq.z, 3.0, offset);" in generated_code
    )
    assert (
        "float4 projectedGrad = tex.SampleGrad(uvq.xy / uvq.z, ddx, ddy);"
        in generated_code
    )
    assert (
        "float4 projectedGradOffset = tex.SampleGrad("
        "uvq.xy / uvq.z, ddx, ddy, offset);" in generated_code
    )
    assert "textureProj(" not in generated_code
    assert "textureProjOffset(" not in generated_code
    assert "textureProjLod(" not in generated_code
    assert "textureProjLodOffset(" not in generated_code
    assert "textureProjGrad(" not in generated_code
    assert "textureProjGradOffset(" not in generated_code
    assert ".Sample(sampleState" not in generated_code
    assert ".SampleBias(sampleState" not in generated_code
    assert ".SampleLevel(sampleState" not in generated_code
    assert ".SampleGrad(sampleState" not in generated_code


def test_projected_texture_invalid_slang_calls_emit_diagnostic_stubs():
    code = """
    shader Resources {
        sampler querySampler;
        sampler2d colorMap;
        sampler2dshadow shadowMap;
        sampler2dms msTex;
        image2D colorImage;

        compute {
            void main() {
                vec2 uv = vec2(0.25, 0.75);
                vec3 uvq = vec3(0.25, 0.75, 1.0);
                vec2 ddx = vec2(0.1, 0.0);
                vec3 badDdx = vec3(0.1, 0.0, 0.0);
                ivec2 offset = ivec2(1, 0);
                ivec3 badOffset = ivec3(1, 0, 0);
                vec4 badCoord = textureProj(colorMap, uv);
                vec4 samplerProjected = textureProj(querySampler, uvq);
                vec4 imageProjected = textureProj(colorImage, uvq);
                vec4 shadowProjected = textureProj(shadowMap, uvq);
                vec4 multisampleProjected = textureProj(msTex, uvq);
                vec4 missingLod = textureProjLod(colorMap, uvq);
                vec4 missingGrad = textureProjGrad(colorMap, uvq, ddx);
                vec4 missingOffset = textureProjGradOffset(
                    colorMap,
                    uvq,
                    ddx,
                    ddx
                );
                vec4 badProjectedOffset = textureProjOffset(colorMap, uvq, badOffset);
                vec4 badProjectedGrad = textureProjGrad(
                    colorMap,
                    uvq,
                    badDdx,
                    ddx
                );
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float4 badCoord = /* unsupported Slang projected texture: "
        "textureProj requires sampler1D/2D/3D projection coordinates */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 samplerProjected = /* unsupported Slang projected texture: "
        "textureProj requires sampler1D/2D/3D texture resource */ float4(0.0);"
        in generated_code
    )
    assert (
        "float4 imageProjected = /* unsupported Slang projected texture: "
        "textureProj requires sampler1D/2D/3D texture resource */ float4(0.0);"
        in generated_code
    )
    assert (
        "float4 shadowProjected = /* unsupported Slang projected texture: "
        "textureProj requires sampler1D/2D/3D texture resource */ float4(0.0);"
        in generated_code
    )
    assert (
        "float4 multisampleProjected = /* unsupported Slang projected texture: "
        "textureProj requires sampler1D/2D/3D texture resource */ float4(0.0);"
        in generated_code
    )
    assert (
        "float4 missingLod = /* unsupported Slang projected texture: "
        "textureProjLod requires one lod argument */ float4(0.0);" in generated_code
    )
    assert (
        "float4 missingGrad = /* unsupported Slang projected texture: "
        "textureProjGrad requires gradient x and gradient y arguments */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 missingOffset = /* unsupported Slang projected texture: "
        "textureProjGradOffset requires gradient x, gradient y, and offset arguments */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 badProjectedOffset = /* unsupported Slang projected texture: "
        "textureProjOffset requires a 2-component offset for sampler2D */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 badProjectedGrad = /* unsupported Slang projected texture: "
        "textureProjGrad requires a 2-component gradient for sampler2D */ "
        "float4(0.0);" in generated_code
    )
    assert "textureProj(" not in generated_code
    assert "textureProjOffset(" not in generated_code
    assert "textureProjLod(" not in generated_code
    assert "textureProjGrad(" not in generated_code
    assert "textureProjGradOffset(" not in generated_code
    assert "querySampler.Sample" not in generated_code
    assert "colorImage.Sample" not in generated_code
    assert "shadowMap.Sample" not in generated_code
    assert "msTex.Sample" not in generated_code


def test_explicit_sampler_texel_fetch_emits_combined_slang_methods():
    code = """
    shader Resources {
        sampler linearSampler;
        sampler2d colorMap;
        sampler2darray layerMap;
        sampler3d volumeMap;
        sampler2dms msTex;
        sampler2dmsarray msArray;

        compute {
            vec4 fetchExplicit(
                sampler2d tex,
                sampler2darray layers,
                sampler3d volume,
                sampler2dms ms,
                sampler2dmsarray msLayers,
                sampler sampleState,
                ivec2 pixel,
                ivec3 pixelLayer,
                int lod,
                int sampleIndex
            ) {
                vec4 fetched = texelFetch(tex, sampleState, pixel, lod);
                vec4 fetchedLayer = texelFetch(
                    layers,
                    sampleState,
                    pixelLayer,
                    lod
                );
                vec4 fetchedVolume = texelFetch(volume, sampleState, pixelLayer, lod);
                vec4 msColor = texelFetch(ms, sampleState, pixel, sampleIndex);
                vec4 msLayer = texelFetch(
                    msLayers,
                    sampleState,
                    pixelLayer,
                    sampleIndex
                );
                return fetched + fetchedLayer + fetchedVolume + msColor + msLayer;
            }

            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "SamplerState linearSampler : register(s0);" in generated_code
    assert "SamplerState sampleState" in generated_code
    assert "float4 fetched = tex.Load(int3(pixel, lod));" in generated_code
    assert "float4 fetchedLayer = layers.Load(int4(pixelLayer, lod));" in generated_code
    assert (
        "float4 fetchedVolume = volume.Load(int4(pixelLayer, lod));" in generated_code
    )
    assert "float4 msColor = ms[pixel, sampleIndex];" in generated_code
    assert "float4 msLayer = msLayers[pixelLayer, sampleIndex];" in generated_code
    assert "tex.Load(int3(sampleState" not in generated_code
    assert "layers.Load(int4(sampleState" not in generated_code
    assert "volume.Load(int4(sampleState" not in generated_code
    assert "ms[sampleState" not in generated_code
    assert "msLayers[sampleState" not in generated_code
    assert "texelFetch(" not in generated_code


def test_texture_gather_builtins_emit_slang_gather_methods():
    code = """
    shader Resources {
        sampler linearSampler;
        sampler2d colorMap;
        sampler2darray layerMap;

        compute {
            vec4 gatherOps(
                sampler2d tex,
                sampler2darray layers,
                sampler sampleState,
                vec2 uv,
                vec3 uvLayer,
                ivec2 offset,
                ivec2 offsets[4],
                int component
            ) {
                vec4 gathered = textureGather(tex, uv);
                vec4 explicitGathered = textureGather(tex, sampleState, uv);
                vec4 greenGather = textureGather(tex, uv, 1);
                vec4 blueExplicitGather = textureGather(
                    tex,
                    sampleState,
                    uv,
                    2
                );
                vec4 offsetGather = textureGatherOffset(tex, uv, offset);
                vec4 alphaOffsetGather = textureGatherOffset(
                    tex,
                    sampleState,
                    uv,
                    offset,
                    3
                );
                vec4 offsetsGather = textureGatherOffsets(
                    layers,
                    sampleState,
                    uvLayer,
                    offsets
                );
                vec4 dynamicGather = textureGather(tex, uv, component);
                vec4 dynamicOffsetGather = textureGatherOffset(
                    tex,
                    sampleState,
                    uv,
                    offset,
                    component
                );
                vec4 dynamicOffsetsGather = textureGatherOffsets(
                    layers,
                    sampleState,
                    uvLayer,
                    offsets,
                    component
                );
                return gathered
                    + explicitGathered
                    + greenGather
                    + blueExplicitGather
                    + offsetGather
                    + alphaOffsetGather
                    + offsetsGather
                    + dynamicGather
                    + dynamicOffsetGather
                    + dynamicOffsetsGather;
            }

            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Sampler2D<float4> tex" in generated_code
    assert "Sampler2DArray<float4> layers" in generated_code
    assert "SamplerState sampleState" in generated_code
    assert "float4 gathered = tex.Gather(uv);" in generated_code
    assert "float4 explicitGathered = tex.Gather(uv);" in generated_code
    assert "float4 greenGather = tex.GatherGreen(uv);" in generated_code
    assert "float4 blueExplicitGather = tex.GatherBlue(uv);" in generated_code
    assert "float4 offsetGather = tex.Gather(uv, offset);" in generated_code
    assert "float4 alphaOffsetGather = tex.GatherAlpha(uv, offset);" in generated_code
    assert (
        "float4 offsetsGather = layers.Gather("
        "uvLayer, offsets[0], offsets[1], offsets[2], offsets[3]);" in generated_code
    )
    assert (
        "float4 dynamicGather = (component == 0 ? tex.GatherRed(uv) : "
        "component == 1 ? tex.GatherGreen(uv) : "
        "component == 2 ? tex.GatherBlue(uv) : tex.GatherAlpha(uv));" in generated_code
    )
    assert (
        "float4 dynamicOffsetGather = ("
        "component == 0 ? tex.GatherRed(uv, offset) : "
        "component == 1 ? tex.GatherGreen(uv, offset) : "
        "component == 2 ? tex.GatherBlue(uv, offset) : "
        "tex.GatherAlpha(uv, offset));" in generated_code
    )
    assert (
        "float4 dynamicOffsetsGather = ("
        "component == 0 ? layers.GatherRed("
        "uvLayer, offsets[0], offsets[1], offsets[2], offsets[3]) : "
        "component == 1 ? layers.GatherGreen("
        "uvLayer, offsets[0], offsets[1], offsets[2], offsets[3]) : "
        "component == 2 ? layers.GatherBlue("
        "uvLayer, offsets[0], offsets[1], offsets[2], offsets[3]) : "
        "layers.GatherAlpha("
        "uvLayer, offsets[0], offsets[1], offsets[2], offsets[3]));" in generated_code
    )
    assert "textureGather" not in generated_code
    assert ".Gather(sampleState" not in generated_code
    assert ".GatherBlue(sampleState" not in generated_code
    assert ".GatherAlpha(sampleState" not in generated_code


def test_texture_gather_invalid_slang_calls_emit_diagnostic_stubs():
    code = """
    shader Resources {
        sampler querySampler;
        sampler2d colorMap;
        sampler2dshadow shadowMap;
        sampler2dms msTex;
        image2D colorImage;

        compute {
            vec4 gatherDiagnostics(
                sampler2d tex,
                vec2 uv,
                ivec2 offset,
                ivec3 badOffset,
                ivec3 badOffsetArray[4]
            ) {
                float scalarCoord = 0.25;
                vec4 badComponent = textureGather(tex, uv, 4);
                vec4 missingOffset = textureGatherOffset(tex, uv);
                vec4 badOffsets = textureGatherOffsets(tex, uv, offset);
                vec4 samplerGather = textureGather(querySampler, uv);
                vec4 imageGather = textureGather(colorImage, uv);
                vec4 shadowGather = textureGather(shadowMap, uv);
                vec4 multisampleGather = textureGather(msTex, uv);
                vec4 badCoordRank = textureGather(tex, scalarCoord);
                vec4 badOffsetRank = textureGatherOffset(tex, uv, badOffset);
                vec4 badOffsetsRank = textureGatherOffsets(tex, uv, badOffsetArray);
                return badComponent + missingOffset + badOffsets;
            }

            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float4 badComponent = /* unsupported Slang texture gather: "
        "textureGather component literal must be 0, 1, 2, or 3 */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 missingOffset = /* unsupported Slang texture gather: "
        "textureGatherOffset requires offset and optional component arguments */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 badOffsets = /* unsupported Slang texture gather: "
        "textureGatherOffsets requires a typed offsets array or four offset arguments */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 samplerGather = /* unsupported Slang texture gather: "
        "textureGather requires a non-shadow non-multisampled sampled texture "
        "resource */ float4(0.0);" in generated_code
    )
    assert (
        "float4 imageGather = /* unsupported Slang texture gather: "
        "textureGather requires a non-shadow non-multisampled sampled texture "
        "resource */ float4(0.0);" in generated_code
    )
    assert (
        "float4 shadowGather = /* unsupported Slang texture gather: "
        "textureGather requires a non-shadow non-multisampled sampled texture "
        "resource */ float4(0.0);" in generated_code
    )
    assert (
        "float4 multisampleGather = /* unsupported Slang texture gather: "
        "textureGather requires a non-shadow non-multisampled sampled texture "
        "resource */ float4(0.0);" in generated_code
    )
    assert (
        "float4 badCoordRank = /* unsupported Slang texture gather: "
        "textureGather requires a 2-component coordinate for sampler2D */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 badOffsetRank = /* unsupported Slang texture gather: "
        "textureGatherOffset requires a 2-component offset for sampler2D */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 badOffsetsRank = /* unsupported Slang texture gather: "
        "textureGatherOffsets requires a 2-component offset for sampler2D */ "
        "float4(0.0);" in generated_code
    )
    assert "textureGather(" not in generated_code
    assert "textureGatherOffset(" not in generated_code
    assert "textureGatherOffsets(" not in generated_code
    assert "querySampler.Gather" not in generated_code
    assert "colorImage.Gather" not in generated_code
    assert "shadowMap.Gather" not in generated_code
    assert "msTex.Gather" not in generated_code


def test_shadow_compare_builtins_emit_slang_compare_methods():
    code = """
    shader Resources {
        sampler compareSampler;
        sampler2dshadow shadowMap;
        sampler2darrayshadow shadowLayers;
        samplercubeshadow cubeShadow;

        compute {
            float compareOps(
                sampler2DShadow tex,
                sampler2DArrayShadow layers,
                samplerCubeShadow cube,
                sampler compareState,
                vec2 uv,
                vec3 uvLayer,
                vec3 direction,
                float depth,
                vec2 ddx,
                vec2 ddy,
                ivec2 offset
            ) {
                float direct = textureCompare(tex, uv, depth);
                float explicitCompare = textureCompare(
                    tex,
                    compareState,
                    uv,
                    depth
                );
                float arrayCompare = textureCompare(
                    layers,
                    compareState,
                    uvLayer,
                    depth
                );
                float cubeCompare = textureCompare(cube, direction, depth);
                float lod = textureCompareLod(tex, compareState, uv, depth, 2.0);
                float grad = textureCompareGrad(
                    tex,
                    compareState,
                    uv,
                    depth,
                    ddx,
                    ddy
                );
                float offsetCompare = textureCompareOffset(tex, uv, depth, offset);
                vec4 gathered = textureGatherCompare(tex, compareState, uv, depth);
                vec4 gatheredOffset = textureGatherCompareOffset(
                    tex,
                    uv,
                    depth,
                    offset
                );
                return direct
                    + explicitCompare
                    + arrayCompare
                    + cubeCompare
                    + lod
                    + grad
                    + offsetCompare
                    + gathered.x
                    + gatheredOffset.x;
            }

            void main() {}
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Sampler2DShadow tex" in generated_code
    assert "Sampler2DArrayShadow layers" in generated_code
    assert "SamplerCubeShadow cube" in generated_code
    assert "SamplerState compareState" in generated_code
    assert "float direct = tex.SampleCmp(uv, depth);" in generated_code
    assert "float explicitCompare = tex.SampleCmp(uv, depth);" in generated_code
    assert "float arrayCompare = layers.SampleCmp(uvLayer, depth);" in generated_code
    assert "float cubeCompare = cube.SampleCmp(direction, depth);" in generated_code
    assert "float lod = tex.SampleCmpLevel(uv, depth, 2.0);" in generated_code
    assert "float grad = tex.SampleCmpGrad(uv, depth, ddx, ddy);" in generated_code
    assert "float offsetCompare = tex.SampleCmp(uv, depth, offset);" in generated_code
    assert "float4 gathered = tex.GatherCmp(uv, depth);" in generated_code
    assert "float4 gatheredOffset = tex.GatherCmp(uv, depth, offset);" in generated_code
    assert "textureCompare" not in generated_code
    assert "textureGatherCompare" not in generated_code
    assert ".SampleCmp(compareState" not in generated_code
    assert ".GatherCmp(compareState" not in generated_code


def test_shadow_compare_invalid_slang_calls_emit_diagnostic_stubs():
    code = """
    shader Resources {
        sampler querySampler;
        sampler2d colorMap;
        sampler2dshadow shadowMap;
        sampler2dms msTex;
        image2D colorImage;

        compute {
            void main() {
                float scalarCoord = 0.25;
                vec2 uv = vec2(0.25, 0.75);
                vec3 badUv = vec3(0.25, 0.75, 0.5);
                vec2 ddx = vec2(0.1, 0.0);
                vec3 badDdx = vec3(0.1, 0.0, 0.0);
                vec2 badDepth = vec2(0.5, 0.25);
                ivec3 badOffset = ivec3(1, 0, 0);
                float badResource = textureCompare(colorMap, uv, 0.5);
                float samplerCompare = textureCompare(querySampler, uv, 0.5);
                float imageCompare = textureCompare(colorImage, uv, 0.5);
                float multisampleCompare = textureCompare(msTex, uv, 0.5);
                float missingDepth = textureCompare(shadowMap, uv);
                float missingLod = textureCompareLod(shadowMap, uv, 0.5);
                float missingGrad = textureCompareGrad(shadowMap, uv, 0.5, ddx);
                vec4 badGatherResource = textureGatherCompare(colorMap, uv, 0.5);
                vec4 missingGatherOffset = textureGatherCompareOffset(
                    shadowMap,
                    uv,
                    0.5
                );
                float badCoordRank = textureCompare(shadowMap, scalarCoord, 0.5);
                float badCompareRank = textureCompare(shadowMap, uv, badDepth);
                float badOffsetRank = textureCompareOffset(
                    shadowMap,
                    uv,
                    0.5,
                    badOffset
                );
                float badGradRank = textureCompareGrad(
                    shadowMap,
                    uv,
                    0.5,
                    badDdx,
                    ddx
                );
                vec4 badGatherCoordRank = textureGatherCompare(
                    shadowMap,
                    badUv,
                    0.5
                );
                vec4 badGatherCompareRank = textureGatherCompare(
                    shadowMap,
                    uv,
                    badDepth
                );
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float badResource = /* unsupported Slang shadow compare: "
        "textureCompare requires a shadow sampler resource */ 0.0;" in generated_code
    )
    assert (
        "float samplerCompare = /* unsupported Slang shadow compare: "
        "textureCompare requires a shadow sampler resource */ 0.0;" in generated_code
    )
    assert (
        "float imageCompare = /* unsupported Slang shadow compare: "
        "textureCompare requires a shadow sampler resource */ 0.0;" in generated_code
    )
    assert (
        "float multisampleCompare = /* unsupported Slang shadow compare: "
        "textureCompare requires a shadow sampler resource */ 0.0;" in generated_code
    )
    assert (
        "float missingDepth = /* unsupported Slang shadow compare: "
        "textureCompare requires texture, coordinate, and compare arguments */ 0.0;"
        in generated_code
    )
    assert (
        "float missingLod = /* unsupported Slang shadow compare: "
        "textureCompareLod requires one lod argument */ 0.0;" in generated_code
    )
    assert (
        "float missingGrad = /* unsupported Slang shadow compare: "
        "textureCompareGrad requires gradient x and gradient y arguments */ 0.0;"
        in generated_code
    )
    assert (
        "float4 badGatherResource = /* unsupported Slang shadow gather compare: "
        "textureGatherCompare requires a shadow sampler resource */ float4(0.0);"
        in generated_code
    )
    assert (
        "float4 missingGatherOffset = /* unsupported Slang shadow gather compare: "
        "textureGatherCompareOffset requires one offset argument */ float4(0.0);"
        in generated_code
    )
    assert (
        "float badCoordRank = /* unsupported Slang shadow compare: "
        "textureCompare requires a 2-component coordinate for sampler2DShadow */ 0.0;"
        in generated_code
    )
    assert (
        "float badCompareRank = /* unsupported Slang shadow compare: "
        "textureCompare requires a scalar compare reference */ 0.0;" in generated_code
    )
    assert (
        "float badOffsetRank = /* unsupported Slang shadow compare: "
        "textureCompareOffset requires a 2-component offset for sampler2DShadow */ "
        "0.0;" in generated_code
    )
    assert (
        "float badGradRank = /* unsupported Slang shadow compare: "
        "textureCompareGrad requires a 2-component gradient for sampler2DShadow */ "
        "0.0;" in generated_code
    )
    assert (
        "float4 badGatherCoordRank = /* unsupported Slang shadow gather compare: "
        "textureGatherCompare requires a 2-component coordinate for sampler2DShadow */ "
        "float4(0.0);" in generated_code
    )
    assert (
        "float4 badGatherCompareRank = /* unsupported Slang shadow gather compare: "
        "textureGatherCompare requires a scalar compare reference */ float4(0.0);"
        in generated_code
    )
    assert "textureCompare(" not in generated_code
    assert "textureCompareOffset(" not in generated_code
    assert "textureCompareLod(" not in generated_code
    assert "textureCompareGrad(" not in generated_code
    assert "textureGatherCompare(" not in generated_code
    assert "textureGatherCompareOffset(" not in generated_code
    assert "querySampler.SampleCmp" not in generated_code
    assert "colorImage.SampleCmp" not in generated_code
    assert "msTex.SampleCmp" not in generated_code


def test_texture_query_lod_emits_slang_lod_methods():
    code = """
    shader Resources {
        sampler querySampler;
        sampler2d colorMap;
        sampler2darray layers;
        sampler3d volumeTex;
        samplercube cubeTex;

        compute {
            void main() {
                vec2 uv = vec2(0.25, 0.5);
                vec3 uvw = vec3(0.25, 0.5, 0.75);
                vec3 direction = vec3(1.0, 0.0, 0.0);
                vec2 lod = textureQueryLod(colorMap, uv);
                vec2 explicitLod = textureQueryLod(colorMap, querySampler, uv);
                vec2 arrayLod = textureQueryLod(layers, querySampler, uvw);
                vec2 volumeLod = textureQueryLod(volumeTex, uvw);
                vec2 cubeLod = textureQueryLod(cubeTex, direction);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float2 lod = float2("
        "colorMap.CalculateLevelOfDetailUnclamped(uv), "
        "colorMap.CalculateLevelOfDetail(uv));" in generated_code
    )
    assert (
        "float2 explicitLod = float2("
        "colorMap.CalculateLevelOfDetailUnclamped(uv), "
        "colorMap.CalculateLevelOfDetail(uv));" in generated_code
    )
    assert (
        "float2 arrayLod = float2("
        "layers.CalculateLevelOfDetailUnclamped(uvw), "
        "layers.CalculateLevelOfDetail(uvw));" in generated_code
    )
    assert (
        "float2 volumeLod = float2("
        "volumeTex.CalculateLevelOfDetailUnclamped(uvw), "
        "volumeTex.CalculateLevelOfDetail(uvw));" in generated_code
    )
    assert (
        "float2 cubeLod = float2("
        "cubeTex.CalculateLevelOfDetailUnclamped(direction), "
        "cubeTex.CalculateLevelOfDetail(direction));" in generated_code
    )
    assert "textureQueryLod(" not in generated_code
    assert "CalculateLevelOfDetail(querySampler" not in generated_code


def test_invalid_texture_query_lod_calls_emit_slang_diagnostics():
    code = """
    shader InvalidTextureQueryLod {
        sampler querySampler;
        sampler2d colorMap;
        sampler2dshadow shadowMap;
        sampler2dms msTex;
        image2D colorImage;

        compute {
            void main() {
                float scalarCoord = 0.25;
                vec2 uv = vec2(0.25, 0.5);
                vec3 wideCoord = vec3(0.25, 0.5, 0.75);
                vec2 missingTexture = textureQueryLod();
                vec2 missingCoord = textureQueryLod(colorMap);
                vec2 explicitMissingCoord = textureQueryLod(colorMap, querySampler);
                vec2 extraArg = textureQueryLod(colorMap, uv, 1);
                vec2 explicitExtraArg = textureQueryLod(
                    colorMap,
                    querySampler,
                    uv,
                    1
                );
                vec2 badImplicitCoord = textureQueryLod(colorMap, scalarCoord);
                vec2 badExplicitCoord = textureQueryLod(
                    colorMap,
                    querySampler,
                    wideCoord
                );
                vec2 samplerOnly = textureQueryLod(querySampler, uv);
                vec2 shadowLod = textureQueryLod(shadowMap, uv);
                vec2 msLod = textureQueryLod(msTex, uv);
                vec2 imageLod = textureQueryLod(colorImage, uv);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float2 missingTexture = /* unsupported Slang resource query: "
        "textureQueryLod requires texture and coordinate arguments */ "
        "float2(0.0, 0.0);" in generated_code
    )
    assert (
        "float2 missingCoord = /* unsupported Slang resource query: "
        "textureQueryLod requires texture and coordinate arguments */ "
        "float2(0.0, 0.0);" in generated_code
    )
    assert (
        "float2 explicitMissingCoord = /* unsupported Slang resource query: "
        "textureQueryLod requires texture and coordinate arguments */ "
        "float2(0.0, 0.0);" in generated_code
    )
    assert (
        "float2 extraArg = /* unsupported Slang resource query: "
        "textureQueryLod accepts only texture, optional sampler, and coordinate "
        "arguments */ float2(0.0, 0.0);" in generated_code
    )
    assert (
        "float2 explicitExtraArg = /* unsupported Slang resource query: "
        "textureQueryLod accepts only texture, optional sampler, and coordinate "
        "arguments */ float2(0.0, 0.0);" in generated_code
    )
    assert (
        "float2 badImplicitCoord = /* unsupported Slang resource query: "
        "textureQueryLod requires a 2-component coordinate for sampler2D */ "
        "float2(0.0, 0.0);" in generated_code
    )
    assert (
        "float2 badExplicitCoord = /* unsupported Slang resource query: "
        "textureQueryLod requires a 2-component coordinate for sampler2D */ "
        "float2(0.0, 0.0);" in generated_code
    )
    assert (
        "float2 samplerOnly = /* unsupported Slang resource query: "
        "textureQueryLod requires a non-shadow non-multisampled sampled texture "
        "resource */ float2(0.0, 0.0);" in generated_code
    )
    assert (
        "float2 shadowLod = /* unsupported Slang resource query: "
        "textureQueryLod requires a non-shadow non-multisampled sampled texture "
        "resource */ float2(0.0, 0.0);" in generated_code
    )
    assert (
        "float2 msLod = /* unsupported Slang resource query: "
        "textureQueryLod requires a non-shadow non-multisampled sampled texture "
        "resource */ float2(0.0, 0.0);" in generated_code
    )
    assert (
        "float2 imageLod = /* unsupported Slang resource query: "
        "textureQueryLod requires a non-shadow non-multisampled sampled texture "
        "resource */ float2(0.0, 0.0);" in generated_code
    )
    assert generated_code.count("unsupported Slang resource query") == 11
    assert "textureQueryLod(" not in generated_code
    assert "CalculateLevelOfDetail(querySampler" not in generated_code


def test_resource_queries_on_texture_arrays_emit_slang_helpers():
    code = """
    shader QueryArrays {
        sampler querySampler;
        sampler2d textures[(2 + 1) * 2];
        sampler2dms msTextures[(2 + 1) * 2];
        sampler2dmsarray msLayers[(2 + 1) * 2];
        sampler2dshadow shadowMaps[(2 + 1) * 2];

        compute {
            void main() {
                vec2 uv = vec2(0.25, 0.5);
                ivec3 pixelLayer = ivec3(1, 2, 3);
                ivec2 sizeValue = textureSize(textures[1 + 2], 0);
                int levelsValue = textureQueryLevels(textures[1 + 2]);
                vec2 lodValue = textureQueryLod(textures[1 + 2], uv);
                vec2 explicitLodValue = textureQueryLod(
                    textures[1 + 2],
                    querySampler,
                    uv
                );
                ivec2 msSize = textureSize(msTextures[1 + 2], 0);
                int msSamples = textureSamples(msTextures[1 + 2]);
                ivec3 msLayerSize = textureSize(msLayers[1 + 2], 0);
                int msLayerSamples = textureSamples(msLayers[1 + 2]);
                ivec2 shadowSize = textureSize(shadowMaps[1 + 2], 0);
                int shadowLevels = textureQueryLevels(shadowMaps[1 + 2]);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "int2 cgl_textureSize_sampler2D(Sampler2D<float4> tex, uint mipLevel)"
        in generated_code
    )
    assert (
        "int cgl_textureQueryLevels_sampler2D(Sampler2D<float4> tex)" in generated_code
    )
    assert "int2 cgl_textureSize_sampler2DMS(Sampler2DMS<float4> tex)" in generated_code
    assert (
        "int cgl_textureSamples_sampler2DMS(Sampler2DMS<float4> tex)" in generated_code
    )
    assert (
        "int3 cgl_textureSize_sampler2DMSArray(Sampler2DMSArray<float4> tex)"
        in generated_code
    )
    assert (
        "int cgl_textureSamples_sampler2DMSArray(Sampler2DMSArray<float4> tex)"
        in generated_code
    )
    assert (
        "int2 cgl_textureSize_sampler2DShadow(Sampler2DShadow tex, uint mipLevel)"
        in generated_code
    )
    assert (
        "int cgl_textureQueryLevels_sampler2DShadow(Sampler2DShadow tex)"
        in generated_code
    )
    assert (
        "int2 sizeValue = cgl_textureSize_sampler2D(textures[(1 + 2)], 0);"
        in generated_code
    )
    assert (
        "int levelsValue = cgl_textureQueryLevels_sampler2D(textures[(1 + 2)]);"
        in generated_code
    )
    assert (
        "float2 lodValue = float2("
        "textures[(1 + 2)].CalculateLevelOfDetailUnclamped(uv), "
        "textures[(1 + 2)].CalculateLevelOfDetail(uv));" in generated_code
    )
    assert (
        "float2 explicitLodValue = float2("
        "textures[(1 + 2)].CalculateLevelOfDetailUnclamped(uv), "
        "textures[(1 + 2)].CalculateLevelOfDetail(uv));" in generated_code
    )
    assert (
        "int2 msSize = cgl_textureSize_sampler2DMS(msTextures[(1 + 2)]);"
        in generated_code
    )
    assert (
        "int msSamples = cgl_textureSamples_sampler2DMS(msTextures[(1 + 2)]);"
        in generated_code
    )
    assert (
        "int3 msLayerSize = cgl_textureSize_sampler2DMSArray(msLayers[(1 + 2)]);"
        in generated_code
    )
    assert (
        "int msLayerSamples = cgl_textureSamples_sampler2DMSArray(msLayers[(1 + 2)]);"
        in generated_code
    )
    assert (
        "int2 shadowSize = cgl_textureSize_sampler2DShadow(shadowMaps[(1 + 2)], 0);"
        in generated_code
    )
    assert (
        "int shadowLevels = cgl_textureQueryLevels_sampler2DShadow("
        "shadowMaps[(1 + 2)]);" in generated_code
    )
    assert "1 + 2]." not in generated_code
    assert "textureSize(" not in generated_code
    assert "textureQueryLevels(" not in generated_code
    assert "textureSamples(" not in generated_code
    assert "textureQueryLod(" not in generated_code
    assert "CalculateLevelOfDetail(querySampler" not in generated_code


def test_nested_resource_arrays_emit_slang_queries_and_operations():
    code = """
    shader NestedResources {
        sampler querySampler;
        sampler2d textureGrid[2][3];
        sampler2dms msGrid[2][2];
        sampler2dshadow shadowGrid[2][3];
        image2D imageGrid @rgba16f[2][4];
        uimage2D counterGrid @r32ui[2][2];

        vec4 sampleGrid(sampler2d paramGrid[2][3], int layer, int slot, vec2 uv) {
            return texture(paramGrid[layer][slot], uv);
        }

        float compareGrid(
            sampler2dshadow paramShadow[2][3],
            int layer,
            int slot,
            vec2 uv,
            float depth
        ) {
            return textureCompare(paramShadow[layer][slot], uv, depth);
        }

        compute {
            void main() {
                int layer = 1;
                int slot = 1;
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                float depth = 0.5;
                ivec2 sizeValue = textureSize(textureGrid[layer][slot], 0);
                int levelsValue = textureQueryLevels(textureGrid[layer][slot]);
                vec2 lodValue = textureQueryLod(
                    textureGrid[layer][slot],
                    querySampler,
                    uv
                );
                int msSamples = textureSamples(msGrid[layer][slot]);
                ivec2 imageSizeValue = imageSize(imageGrid[layer][slot]);
                vec4 sampled = texture(textureGrid[layer][slot], uv);
                vec4 sampledParam = sampleGrid(textureGrid, layer, slot, uv);
                float compared = textureCompare(shadowGrid[layer][slot], uv, depth);
                float comparedParam = compareGrid(shadowGrid, layer, slot, uv, depth);
                vec4 gathered = textureGather(textureGrid[layer][slot], uv, 1);
                vec4 gatheredShadow = textureGatherCompare(
                    shadowGrid[layer][slot],
                    uv,
                    depth
                );
                vec4 fetched = texelFetch(textureGrid[layer][slot], pixel, 0);
                vec4 fetchedMs = texelFetch(msGrid[layer][slot], pixel, 1);
                vec4 color = imageLoad(imageGrid[layer][slot], pixel);
                imageStore(imageGrid[layer][slot], pixel, color);
                uint count = imageLoad(counterGrid[layer][slot], pixel);
                imageStore(counterGrid[layer][slot], pixel, count);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Sampler2D<float4> textureGrid[2][3] : register(t0);" in generated_code
    assert "Sampler2DMS<float4> msGrid[2][2] : register(t6);" in generated_code
    assert "Sampler2DShadow shadowGrid[2][3] : register(t10);" in generated_code
    assert "RWTexture2D<float4> imageGrid[2][4] : register(u0);" in generated_code
    assert "RWTexture2D<uint> counterGrid[2][2] : register(u8);" in generated_code
    assert (
        "float4 sampleGrid(Sampler2D<float4> paramGrid[2][3], "
        "int layer, int slot, float2 uv)" in generated_code
    )
    assert (
        "float compareGrid(Sampler2DShadow paramShadow[2][3], "
        "int layer, int slot, float2 uv, float depth)" in generated_code
    )
    assert "return paramGrid[layer][slot].Sample(uv);" in generated_code
    assert "return paramShadow[layer][slot].SampleCmp(uv, depth);" in generated_code
    assert (
        "int2 sizeValue = cgl_textureSize_sampler2D"
        "(textureGrid[layer][slot], 0);" in generated_code
    )
    assert (
        "int levelsValue = cgl_textureQueryLevels_sampler2D"
        "(textureGrid[layer][slot]);" in generated_code
    )
    assert (
        "float2 lodValue = float2("
        "textureGrid[layer][slot].CalculateLevelOfDetailUnclamped(uv), "
        "textureGrid[layer][slot].CalculateLevelOfDetail(uv));" in generated_code
    )
    assert (
        "int msSamples = cgl_textureSamples_sampler2DMS(msGrid[layer][slot]);"
        in generated_code
    )
    assert (
        "int2 imageSizeValue = cgl_imageSize_image2D(imageGrid[layer][slot]);"
        in generated_code
    )
    assert "float4 sampled = textureGrid[layer][slot].Sample(uv);" in generated_code
    assert (
        "float4 sampledParam = sampleGrid(textureGrid, layer, slot, uv);"
        in generated_code
    )
    assert (
        "float compared = shadowGrid[layer][slot].SampleCmp(uv, depth);"
        in generated_code
    )
    assert (
        "float comparedParam = compareGrid(shadowGrid, layer, slot, uv, depth);"
        in generated_code
    )
    assert (
        "float4 gathered = textureGrid[layer][slot].GatherGreen(uv);" in generated_code
    )
    assert (
        "float4 gatheredShadow = shadowGrid[layer][slot].GatherCmp(uv, depth);"
        in generated_code
    )
    assert (
        "float4 fetched = textureGrid[layer][slot].Load(int3(pixel, 0));"
        in generated_code
    )
    assert "float4 fetchedMs = msGrid[layer][slot][pixel, 1];" in generated_code
    assert "float4 color = imageGrid[layer][slot][pixel];" in generated_code
    assert "imageGrid[layer][slot][pixel] = color;" in generated_code
    assert "uint count = counterGrid[layer][slot][pixel];" in generated_code
    assert "counterGrid[layer][slot][pixel] = count;" in generated_code
    assert "textureSize(" not in generated_code
    assert "textureQueryLevels(" not in generated_code
    assert "textureQueryLod(" not in generated_code
    assert "textureSamples(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "textureGather(" not in generated_code
    assert "textureGatherCompare(" not in generated_code
    assert "texelFetch(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_dynamic_resource_array_params_emit_slang_queries_and_operations():
    code = """
    shader DynamicResources {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];
        uimage2D counterGrid @r32ui[2][2];

        vec4 sampleDynamic(
            sampler2d dynGrid[][3],
            int layer,
            int slot,
            vec2 uv
        ) {
            return texture(dynGrid[layer][slot], uv);
        }

        vec4 readDynamic(
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            ivec2 pixel
        ) {
            return imageLoad(dynImages[layer][slot], pixel);
        }

        uint readCounter(
            uimage2D dynCounters[][2] @r32ui,
            int layer,
            int slot,
            ivec2 pixel
        ) {
            return imageLoad(dynCounters[layer][slot], pixel);
        }

        void queryDynamic(
            sampler2d dynGrid[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot
        ) {
            ivec2 texSize = textureSize(dynGrid[layer][slot], 0);
            ivec2 imageSizeValue = imageSize(dynImages[layer][slot]);
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                vec4 sampled = sampleDynamic(textureGrid, 1, 2, uv);
                vec4 color = readDynamic(imageGrid, 1, 2, pixel);
                uint count = readCounter(counterGrid, 1, 1, pixel);
                queryDynamic(textureGrid, imageGrid, 1, 2);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "Sampler2D<float4> textureGrid[2][3] : register(t0);" in generated_code
    assert "RWTexture2D<float4> imageGrid[2][3] : register(u0);" in generated_code
    assert "RWTexture2D<uint> counterGrid[2][2] : register(u6);" in generated_code
    assert (
        "float4 sampleDynamic(Sampler2D<float4> dynGrid[][3], "
        "int layer, int slot, float2 uv)" in generated_code
    )
    assert (
        "float4 readDynamic(RWTexture2D<float4> dynImages[][3], "
        "int layer, int slot, int2 pixel)" in generated_code
    )
    assert (
        "uint readCounter(RWTexture2D<uint> dynCounters[][2], "
        "int layer, int slot, int2 pixel)" in generated_code
    )
    assert (
        "void queryDynamic(Sampler2D<float4> dynGrid[][3], "
        "RWTexture2D<float4> dynImages[][3], int layer, int slot)" in generated_code
    )
    assert "return dynGrid[layer][slot].Sample(uv);" in generated_code
    assert "return dynImages[layer][slot][pixel];" in generated_code
    assert "return dynCounters[layer][slot][pixel];" in generated_code
    assert (
        "int2 texSize = cgl_textureSize_sampler2D(dynGrid[layer][slot], 0);"
        in generated_code
    )
    assert (
        "int2 imageSizeValue = cgl_imageSize_image2D(dynImages[layer][slot]);"
        in generated_code
    )
    assert "float4 sampled = sampleDynamic(textureGrid, 1, 2, uv);" in generated_code
    assert "float4 color = readDynamic(imageGrid, 1, 2, pixel);" in generated_code
    assert "uint count = readCounter(counterGrid, 1, 1, pixel);" in generated_code
    assert "queryDynamic(textureGrid, imageGrid, 1, 2);" in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_forwarded_dynamic_resource_array_params_emit_slang_operations():
    code = """
    shader ForwardedDynamicResources {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];
        uimage2D counterGrid @r32ui[2][2];

        vec4 leafSample(sampler2d dynGrid[][3], int layer, int slot, vec2 uv) {
            return texture(dynGrid[layer][slot], uv);
        }

        vec4 forwardSample(sampler2d dynGrid[][3], int layer, int slot, vec2 uv) {
            return leafSample(dynGrid, layer, slot, uv);
        }

        vec4 leafImage(
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            ivec2 pixel
        ) {
            vec4 color = imageLoad(dynImages[layer][slot], pixel);
            imageStore(dynImages[layer][slot], pixel, color);
            return color;
        }

        vec4 forwardImage(
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            ivec2 pixel
        ) {
            return leafImage(dynImages, layer, slot, pixel);
        }

        uint leafCounter(
            uimage2D dynCounters[][2] @r32ui,
            int layer,
            int slot,
            ivec2 pixel
        ) {
            uint count = imageLoad(dynCounters[layer][slot], pixel);
            imageStore(dynCounters[layer][slot], pixel, count);
            return count;
        }

        uint forwardCounter(
            uimage2D dynCounters[][2] @r32ui,
            int layer,
            int slot,
            ivec2 pixel
        ) {
            return leafCounter(dynCounters, layer, slot, pixel);
        }

        void leafQuery(
            sampler2d dynGrid[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot
        ) {
            ivec2 texSize = textureSize(dynGrid[layer][slot], 0);
            ivec2 imageSizeValue = imageSize(dynImages[layer][slot]);
        }

        void forwardQuery(
            sampler2d dynGrid[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot
        ) {
            leafQuery(dynGrid, dynImages, layer, slot);
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                vec4 sampled = forwardSample(textureGrid, 1, 2, uv);
                vec4 color = forwardImage(imageGrid, 1, 2, pixel);
                uint count = forwardCounter(counterGrid, 1, 1, pixel);
                forwardQuery(textureGrid, imageGrid, 1, 2);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float4 leafSample(Sampler2D<float4> dynGrid[][3], "
        "int layer, int slot, float2 uv)" in generated_code
    )
    assert (
        "float4 forwardSample(Sampler2D<float4> dynGrid[][3], "
        "int layer, int slot, float2 uv)" in generated_code
    )
    assert "return leafSample(dynGrid, layer, slot, uv);" in generated_code
    assert (
        "float4 leafImage(RWTexture2D<float4> dynImages[][3], "
        "int layer, int slot, int2 pixel)" in generated_code
    )
    assert "return leafImage(dynImages, layer, slot, pixel);" in generated_code
    assert (
        "uint leafCounter(RWTexture2D<uint> dynCounters[][2], "
        "int layer, int slot, int2 pixel)" in generated_code
    )
    assert "return leafCounter(dynCounters, layer, slot, pixel);" in generated_code
    assert (
        "void leafQuery(Sampler2D<float4> dynGrid[][3], "
        "RWTexture2D<float4> dynImages[][3], int layer, int slot)" in generated_code
    )
    assert "leafQuery(dynGrid, dynImages, layer, slot);" in generated_code
    assert "float4 sampled = forwardSample(textureGrid, 1, 2, uv);" in generated_code
    assert "float4 color = forwardImage(imageGrid, 1, 2, pixel);" in generated_code
    assert "uint count = forwardCounter(counterGrid, 1, 1, pixel);" in generated_code
    assert "forwardQuery(textureGrid, imageGrid, 1, 2);" in generated_code
    assert "return dynGrid[layer][slot].Sample(uv);" in generated_code
    assert "float4 color = dynImages[layer][slot][pixel];" in generated_code
    assert "dynImages[layer][slot][pixel] = color;" in generated_code
    assert "uint count = dynCounters[layer][slot][pixel];" in generated_code
    assert "dynCounters[layer][slot][pixel] = count;" in generated_code
    assert (
        "int2 texSize = cgl_textureSize_sampler2D(dynGrid[layer][slot], 0);"
        in generated_code
    )
    assert (
        "int2 imageSizeValue = cgl_imageSize_image2D(dynImages[layer][slot]);"
        in generated_code
    )
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_mixed_fixed_dynamic_resource_array_params_emit_slang_operations():
    code = """
    shader MixedResourceForwarding {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];

        void leafMixed(
            sampler2d dynamicGrid[][3],
            sampler2d fixedRow[3],
            image2D dynamicImages[][3] @rgba16f,
            image2D fixedImages[3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            vec4 dynamicSample = texture(dynamicGrid[layer][slot], uv);
            vec4 fixedSample = texture(fixedRow[slot], uv);
            vec4 dynamicColor = imageLoad(dynamicImages[layer][slot], pixel);
            vec4 fixedColor = imageLoad(fixedImages[slot], pixel);
            imageStore(dynamicImages[layer][slot], pixel, dynamicColor);
            imageStore(fixedImages[slot], pixel, fixedColor);
            ivec2 dynamicSize = textureSize(dynamicGrid[layer][slot], 0);
            ivec2 fixedSize = textureSize(fixedRow[slot], 0);
            ivec2 dynamicImageSize = imageSize(dynamicImages[layer][slot]);
            ivec2 fixedImageSize = imageSize(fixedImages[slot]);
        }

        void forwardMixed(
            sampler2d dynamicGrid[][3],
            sampler2d fixedRow[3],
            image2D dynamicImages[][3] @rgba16f,
            image2D fixedImages[3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            leafMixed(
                dynamicGrid,
                fixedRow,
                dynamicImages,
                fixedImages,
                layer,
                slot,
                uv,
                pixel
            );
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                forwardMixed(
                    textureGrid,
                    textureGrid[1],
                    imageGrid,
                    imageGrid[1],
                    1,
                    2,
                    uv,
                    pixel
                );
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "void leafMixed(Sampler2D<float4> dynamicGrid[][3], "
        "Sampler2D<float4> fixedRow[3], "
        "RWTexture2D<float4> dynamicImages[][3], "
        "RWTexture2D<float4> fixedImages[3], "
        "int layer, int slot, float2 uv, int2 pixel)" in generated_code
    )
    assert (
        "void forwardMixed(Sampler2D<float4> dynamicGrid[][3], "
        "Sampler2D<float4> fixedRow[3], "
        "RWTexture2D<float4> dynamicImages[][3], "
        "RWTexture2D<float4> fixedImages[3], "
        "int layer, int slot, float2 uv, int2 pixel)" in generated_code
    )
    assert (
        "leafMixed(dynamicGrid, fixedRow, dynamicImages, fixedImages, "
        "layer, slot, uv, pixel);" in generated_code
    )
    assert (
        "forwardMixed(textureGrid, textureGrid[1], imageGrid, imageGrid[1], "
        "1, 2, uv, pixel);" in generated_code
    )
    assert (
        "float4 dynamicSample = dynamicGrid[layer][slot].Sample(uv);" in generated_code
    )
    assert "float4 fixedSample = fixedRow[slot].Sample(uv);" in generated_code
    assert "float4 dynamicColor = dynamicImages[layer][slot][pixel];" in generated_code
    assert "float4 fixedColor = fixedImages[slot][pixel];" in generated_code
    assert "dynamicImages[layer][slot][pixel] = dynamicColor;" in generated_code
    assert "fixedImages[slot][pixel] = fixedColor;" in generated_code
    assert (
        "int2 dynamicSize = cgl_textureSize_sampler2D"
        "(dynamicGrid[layer][slot], 0);" in generated_code
    )
    assert (
        "int2 fixedSize = cgl_textureSize_sampler2D(fixedRow[slot], 0);"
        in generated_code
    )
    assert (
        "int2 dynamicImageSize = cgl_imageSize_image2D"
        "(dynamicImages[layer][slot]);" in generated_code
    )
    assert (
        "int2 fixedImageSize = cgl_imageSize_image2D(fixedImages[slot]);"
        in generated_code
    )
    assert "texture(" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_multihop_reordered_resource_array_params_emit_slang_operations():
    code = """
    shader MultiHopResourceForwarding {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];

        void leafShuffle(
            sampler2d dynA[][3],
            image2D fixedImageRow[3] @rgba16f,
            sampler2d fixedTexRow[3],
            image2D dynImageGrid[][3] @rgba16f,
            sampler2d dynAlias[][3],
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            vec4 sampledA = texture(dynA[layer][slot], uv);
            vec4 sampledFixed = texture(fixedTexRow[slot], uv);
            vec4 sampledAlias = texture(dynAlias[layer][slot], uv);
            vec4 dynamicColor = imageLoad(dynImageGrid[layer][slot], pixel);
            vec4 fixedColor = imageLoad(fixedImageRow[slot], pixel);
            imageStore(dynImageGrid[layer][slot], pixel, dynamicColor);
            imageStore(fixedImageRow[slot], pixel, fixedColor);
            ivec2 dynSize = textureSize(dynA[layer][slot], 0);
            ivec2 fixedTexSize = textureSize(fixedTexRow[slot], 0);
            ivec2 aliasSize = textureSize(dynAlias[layer][slot], 0);
            ivec2 dynamicImageSize = imageSize(dynImageGrid[layer][slot]);
            ivec2 fixedImageSize = imageSize(fixedImageRow[slot]);
        }

        void midForward(
            sampler2d firstDyn[][3],
            sampler2d firstFixed[3],
            image2D firstDynImages[][3] @rgba16f,
            image2D firstFixedImages[3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            leafShuffle(
                firstDyn,
                firstFixedImages,
                firstFixed,
                firstDynImages,
                firstDyn,
                layer,
                slot,
                uv,
                pixel
            );
        }

        void topForward(
            sampler2d topFixedTex[3],
            image2D topDynImages[][3] @rgba16f,
            sampler2d topDynTex[][3],
            image2D topFixedImages[3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            midForward(
                topDynTex,
                topFixedTex,
                topDynImages,
                topFixedImages,
                layer,
                slot,
                uv,
                pixel
            );
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                topForward(
                    textureGrid[1],
                    imageGrid,
                    textureGrid,
                    imageGrid[1],
                    1,
                    2,
                    uv,
                    pixel
                );
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "void leafShuffle(Sampler2D<float4> dynA[][3], "
        "RWTexture2D<float4> fixedImageRow[3], "
        "Sampler2D<float4> fixedTexRow[3], "
        "RWTexture2D<float4> dynImageGrid[][3], "
        "Sampler2D<float4> dynAlias[][3], "
        "int layer, int slot, float2 uv, int2 pixel)" in generated_code
    )
    assert (
        "void midForward(Sampler2D<float4> firstDyn[][3], "
        "Sampler2D<float4> firstFixed[3], "
        "RWTexture2D<float4> firstDynImages[][3], "
        "RWTexture2D<float4> firstFixedImages[3], "
        "int layer, int slot, float2 uv, int2 pixel)" in generated_code
    )
    assert (
        "void topForward(Sampler2D<float4> topFixedTex[3], "
        "RWTexture2D<float4> topDynImages[][3], "
        "Sampler2D<float4> topDynTex[][3], "
        "RWTexture2D<float4> topFixedImages[3], "
        "int layer, int slot, float2 uv, int2 pixel)" in generated_code
    )
    assert (
        "leafShuffle(firstDyn, firstFixedImages, firstFixed, firstDynImages, "
        "firstDyn, layer, slot, uv, pixel);" in generated_code
    )
    assert (
        "midForward(topDynTex, topFixedTex, topDynImages, topFixedImages, "
        "layer, slot, uv, pixel);" in generated_code
    )
    assert (
        "topForward(textureGrid[1], imageGrid, textureGrid, imageGrid[1], "
        "1, 2, uv, pixel);" in generated_code
    )
    assert "float4 sampledA = dynA[layer][slot].Sample(uv);" in generated_code
    assert "float4 sampledFixed = fixedTexRow[slot].Sample(uv);" in generated_code
    assert "float4 sampledAlias = dynAlias[layer][slot].Sample(uv);" in generated_code
    assert "float4 dynamicColor = dynImageGrid[layer][slot][pixel];" in generated_code
    assert "float4 fixedColor = fixedImageRow[slot][pixel];" in generated_code
    assert "dynImageGrid[layer][slot][pixel] = dynamicColor;" in generated_code
    assert "fixedImageRow[slot][pixel] = fixedColor;" in generated_code
    assert (
        "int2 dynSize = cgl_textureSize_sampler2D(dynA[layer][slot], 0);"
        in generated_code
    )
    assert (
        "int2 fixedTexSize = cgl_textureSize_sampler2D(fixedTexRow[slot], 0);"
        in generated_code
    )
    assert (
        "int2 aliasSize = cgl_textureSize_sampler2D(dynAlias[layer][slot], 0);"
        in generated_code
    )
    assert (
        "int2 dynamicImageSize = cgl_imageSize_image2D(dynImageGrid[layer][slot]);"
        in generated_code
    )
    assert (
        "int2 fixedImageSize = cgl_imageSize_image2D(fixedImageRow[slot]);"
        in generated_code
    )
    assert "texture(" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_resource_calls_in_control_flow_emit_slang_operations():
    code = """
    shader ControlFlowResourceForwarding {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];

        vec4 sampleDynamic(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            vec4 sampled = texture(dynTex[layer][slot], uv);
            vec4 loaded = imageLoad(dynImages[layer][slot], pixel);
            ivec2 texSize = textureSize(dynTex[layer][slot], 0);
            ivec2 imageSizeValue = imageSize(dynImages[layer][slot]);
            return sampled + loaded;
        }

        vec4 sampleFixed(
            sampler2d fixedTex[3],
            image2D fixedImages[3] @rgba16f,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            vec4 sampled = texture(fixedTex[slot], uv);
            vec4 loaded = imageLoad(fixedImages[slot], pixel);
            ivec2 texSize = textureSize(fixedTex[slot], 0);
            ivec2 imageSizeValue = imageSize(fixedImages[slot]);
            return sampled + loaded;
        }

        vec4 chooseBranch(
            sampler2d dynTex[][3],
            sampler2d fixedTex[3],
            image2D dynImages[][3] @rgba16f,
            image2D fixedImages[3] @rgba16f,
            bool useFixed,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            vec4 result = sampleDynamic(dynTex, dynImages, layer, slot, uv, pixel);
            if (useFixed) {
                result = sampleFixed(fixedTex, fixedImages, slot, uv, pixel);
            }
            return result;
        }

        vec4 chooseReturn(
            sampler2d dynTex[][3],
            sampler2d fixedTex[3],
            image2D dynImages[][3] @rgba16f,
            image2D fixedImages[3] @rgba16f,
            bool useFixed,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            if (useFixed) {
                return sampleFixed(fixedTex, fixedImages, slot, uv, pixel);
            }
            return sampleDynamic(dynTex, dynImages, layer, slot, uv, pixel);
        }

        vec4 chooseTernary(
            sampler2d dynTex[][3],
            sampler2d fixedTex[3],
            image2D dynImages[][3] @rgba16f,
            image2D fixedImages[3] @rgba16f,
            bool useFixed,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            return useFixed
                ? sampleFixed(fixedTex, fixedImages, slot, uv, pixel)
                : sampleDynamic(dynTex, dynImages, layer, slot, uv, pixel);
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                vec4 branchValue = chooseBranch(
                    textureGrid,
                    textureGrid[1],
                    imageGrid,
                    imageGrid[1],
                    true,
                    1,
                    2,
                    uv,
                    pixel
                );
                vec4 returnValue = chooseReturn(
                    textureGrid,
                    textureGrid[1],
                    imageGrid,
                    imageGrid[1],
                    false,
                    1,
                    2,
                    uv,
                    pixel
                );
                vec4 ternaryValue = chooseTernary(
                    textureGrid,
                    textureGrid[1],
                    imageGrid,
                    imageGrid[1],
                    true,
                    1,
                    2,
                    uv,
                    pixel
                );
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float4 chooseBranch(Sampler2D<float4> dynTex[][3], "
        "Sampler2D<float4> fixedTex[3], "
        "RWTexture2D<float4> dynImages[][3], "
        "RWTexture2D<float4> fixedImages[3], "
        "bool useFixed, int layer, int slot, float2 uv, int2 pixel)" in generated_code
    )
    assert (
        "float4 result = sampleDynamic(dynTex, dynImages, layer, slot, uv, pixel);"
        in generated_code
    )
    assert (
        "result = sampleFixed(fixedTex, fixedImages, slot, uv, pixel);"
        in generated_code
    )
    assert (
        "return sampleFixed(fixedTex, fixedImages, slot, uv, pixel);" in generated_code
    )
    assert (
        "return sampleDynamic(dynTex, dynImages, layer, slot, uv, pixel);"
        in generated_code
    )
    assert (
        "return (useFixed ? sampleFixed(fixedTex, fixedImages, slot, uv, pixel) : "
        "sampleDynamic(dynTex, dynImages, layer, slot, uv, pixel));" in generated_code
    )
    assert (
        "chooseBranch(textureGrid, textureGrid[1], imageGrid, imageGrid[1], "
        "true, 1, 2, uv, pixel);" in generated_code
    )
    assert (
        "chooseTernary(textureGrid, textureGrid[1], imageGrid, imageGrid[1], "
        "true, 1, 2, uv, pixel);" in generated_code
    )
    assert "float4 sampled = dynTex[layer][slot].Sample(uv);" in generated_code
    assert "float4 loaded = dynImages[layer][slot][pixel];" in generated_code
    assert "float4 sampled = fixedTex[slot].Sample(uv);" in generated_code
    assert "float4 loaded = fixedImages[slot][pixel];" in generated_code
    assert (
        "int2 texSize = cgl_textureSize_sampler2D(dynTex[layer][slot], 0);"
        in generated_code
    )
    assert (
        "int2 imageSizeValue = cgl_imageSize_image2D(dynImages[layer][slot]);"
        in generated_code
    )
    assert (
        "int2 texSize = cgl_textureSize_sampler2D(fixedTex[slot], 0);" in generated_code
    )
    assert (
        "int2 imageSizeValue = cgl_imageSize_image2D(fixedImages[slot]);"
        in generated_code
    )
    assert "texture(" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_resource_calls_in_loops_emit_slang_operations():
    code = """
    shader LoopResourceForwarding {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];

        vec4 sampleLoop(
            sampler2d dynTex[][3],
            sampler2d fixedTex[3],
            image2D dynImages[][3] @rgba16f,
            image2D fixedImages[3] @rgba16f,
            int layer,
            vec2 uv,
            ivec2 pixel
        ) {
            vec4 accum = vec4(0.0);
            for (int slot = 0; slot < 3; slot++) {
                if (slot == 1) {
                    continue;
                }
                vec4 sampled = texture(dynTex[layer][slot], uv);
                vec4 loaded = imageLoad(dynImages[layer][slot], pixel);
                ivec2 texSize = textureSize(dynTex[layer][slot], 0);
                ivec2 imageSizeValue = imageSize(dynImages[layer][slot]);
                accum = accum + sampled + loaded;
                if (slot == 2) {
                    break;
                }
            }
            int fixedSlot = 0;
            while (fixedSlot < 3) {
                accum = accum + texture(fixedTex[fixedSlot], uv);
                accum = accum + imageLoad(fixedImages[fixedSlot], pixel);
                ivec2 fixedTexSize = textureSize(fixedTex[fixedSlot], 0);
                ivec2 fixedImageSize = imageSize(fixedImages[fixedSlot]);
                fixedSlot++;
            }
            return accum;
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                vec4 value = sampleLoop(
                    textureGrid,
                    textureGrid[1],
                    imageGrid,
                    imageGrid[1],
                    1,
                    uv,
                    pixel
                );
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float4 sampleLoop(Sampler2D<float4> dynTex[][3], "
        "Sampler2D<float4> fixedTex[3], "
        "RWTexture2D<float4> dynImages[][3], "
        "RWTexture2D<float4> fixedImages[3], int layer, float2 uv, int2 pixel)"
        in generated_code
    )
    assert "for (int slot = 0; slot < 3; slot++)" in generated_code
    assert "continue;" in generated_code
    assert "break;" in generated_code
    assert "while (fixedSlot < 3)" in generated_code
    assert "float4 sampled = dynTex[layer][slot].Sample(uv);" in generated_code
    assert "float4 loaded = dynImages[layer][slot][pixel];" in generated_code
    assert (
        "int2 texSize = cgl_textureSize_sampler2D(dynTex[layer][slot], 0);"
        in generated_code
    )
    assert (
        "int2 imageSizeValue = cgl_imageSize_image2D(dynImages[layer][slot]);"
        in generated_code
    )
    assert "accum = accum + fixedTex[fixedSlot].Sample(uv);" in generated_code
    assert "accum = accum + fixedImages[fixedSlot][pixel];" in generated_code
    assert (
        "int2 fixedTexSize = cgl_textureSize_sampler2D(fixedTex[fixedSlot], 0);"
        in generated_code
    )
    assert (
        "int2 fixedImageSize = cgl_imageSize_image2D(fixedImages[fixedSlot]);"
        in generated_code
    )
    assert (
        "sampleLoop(textureGrid, textureGrid[1], imageGrid, imageGrid[1], "
        "1, uv, pixel);" in generated_code
    )
    assert "texture(" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_nested_loop_resource_indices_emit_slang_operations():
    code = """
    shader NestedLoopResourceForwarding {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];

        vec4 sampleNestedLoops(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            vec2 uv,
            ivec2 pixel
        ) {
            vec4 accum = vec4(0.0);
            for (int layer = 0; layer < 2; layer++) {
                if (layer == 1) {
                    break;
                }
                for (int slot = 0; slot < 3; slot++) {
                    if (slot == 1) {
                        continue;
                    }
                    vec4 sampled = texture(dynTex[layer][slot], uv);
                    vec4 loaded = imageLoad(dynImages[layer][slot], pixel);
                    imageStore(dynImages[layer][slot], pixel, loaded);
                    ivec2 texSize = textureSize(dynTex[layer][slot], 0);
                    ivec2 imageSizeValue = imageSize(dynImages[layer][slot]);
                    accum = accum + sampled + loaded;
                }
            }
            return accum;
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                vec4 value = sampleNestedLoops(textureGrid, imageGrid, uv, pixel);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "float4 sampleNestedLoops(Sampler2D<float4> dynTex[][3], "
        "RWTexture2D<float4> dynImages[][3], float2 uv, int2 pixel)" in generated_code
    )
    assert "for (int layer = 0; layer < 2; layer++)" in generated_code
    assert "for (int slot = 0; slot < 3; slot++)" in generated_code
    assert "break;" in generated_code
    assert "continue;" in generated_code
    assert "float4 sampled = dynTex[layer][slot].Sample(uv);" in generated_code
    assert "float4 loaded = dynImages[layer][slot][pixel];" in generated_code
    assert "dynImages[layer][slot][pixel] = loaded;" in generated_code
    assert (
        "int2 texSize = cgl_textureSize_sampler2D(dynTex[layer][slot], 0);"
        in generated_code
    )
    assert (
        "int2 imageSizeValue = cgl_imageSize_image2D(dynImages[layer][slot]);"
        in generated_code
    )
    assert "sampleNestedLoops(textureGrid, imageGrid, uv, pixel);" in generated_code
    assert "texture(" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_struct_and_scalar_params_preserve_slang_resource_argument_order():
    code = """
    struct SampleParams {
        vec2 uv;
        ivec2 pixel;
        int layer;
        int slot;
    };

    shader PackedResourceForwarding {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];

        vec4 samplePacked(
            sampler2d dynTex[][3],
            SampleParams params,
            image2D dynImages[][3] @rgba16f,
            float weight,
            sampler2d fixedTex[3],
            image2D fixedImages[3] @rgba16f
        ) {
            vec4 dynamicSample = texture(
                dynTex[params.layer][params.slot],
                params.uv
            );
            vec4 fixedSample = texture(fixedTex[params.slot], params.uv);
            vec4 dynamicColor = imageLoad(
                dynImages[params.layer][params.slot],
                params.pixel
            );
            vec4 fixedColor = imageLoad(fixedImages[params.slot], params.pixel);
            ivec2 dynSize = textureSize(dynTex[params.layer][params.slot], 0);
            ivec2 fixedSize = textureSize(fixedTex[params.slot], 0);
            ivec2 dynImageSize = imageSize(dynImages[params.layer][params.slot]);
            ivec2 fixedImageSize = imageSize(fixedImages[params.slot]);
            vec4 combined = dynamicSample + fixedSample + dynamicColor + fixedColor;
            return combined * weight;
        }

        vec4 passPacked(
            SampleParams params,
            sampler2d firstDyn[][3],
            float weight,
            image2D firstImages[][3] @rgba16f,
            sampler2d fixedTex[3],
            image2D fixedImages[3] @rgba16f
        ) {
            return samplePacked(
                firstDyn,
                params,
                firstImages,
                weight,
                fixedTex,
                fixedImages
            );
        }

        compute {
            void main() {
                SampleParams params;
                params.uv = vec2(0.5, 0.25);
                params.pixel = ivec2(0, 0);
                params.layer = 1;
                params.slot = 2;
                vec4 value = passPacked(
                    params,
                    textureGrid,
                    0.5,
                    imageGrid,
                    textureGrid[1],
                    imageGrid[1]
                );
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct SampleParams" in generated_code
    assert (
        "float4 samplePacked(Sampler2D<float4> dynTex[][3], "
        "SampleParams params, RWTexture2D<float4> dynImages[][3], "
        "float weight, Sampler2D<float4> fixedTex[3], "
        "RWTexture2D<float4> fixedImages[3])" in generated_code
    )
    assert (
        "float4 passPacked(SampleParams params, "
        "Sampler2D<float4> firstDyn[][3], float weight, "
        "RWTexture2D<float4> firstImages[][3], "
        "Sampler2D<float4> fixedTex[3], "
        "RWTexture2D<float4> fixedImages[3])" in generated_code
    )
    assert (
        "return samplePacked(firstDyn, params, firstImages, weight, fixedTex, "
        "fixedImages);" in generated_code
    )
    assert (
        "passPacked(params, textureGrid, 0.5, imageGrid, textureGrid[1], "
        "imageGrid[1]);" in generated_code
    )
    assert (
        "float4 dynamicSample = dynTex[params.layer][params.slot].Sample(params.uv);"
        in generated_code
    )
    assert (
        "float4 fixedSample = fixedTex[params.slot].Sample(params.uv);"
        in generated_code
    )
    assert (
        "float4 dynamicColor = dynImages[params.layer][params.slot][params.pixel];"
        in generated_code
    )
    assert (
        "float4 fixedColor = fixedImages[params.slot][params.pixel];" in generated_code
    )
    assert (
        "int2 dynSize = cgl_textureSize_sampler2D"
        "(dynTex[params.layer][params.slot], 0);" in generated_code
    )
    assert (
        "int2 fixedSize = cgl_textureSize_sampler2D(fixedTex[params.slot], 0);"
        in generated_code
    )
    assert (
        "int2 dynImageSize = cgl_imageSize_image2D"
        "(dynImages[params.layer][params.slot]);" in generated_code
    )
    assert (
        "int2 fixedImageSize = cgl_imageSize_image2D(fixedImages[params.slot]);"
        in generated_code
    )
    assert "texture(" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_resource_values_in_struct_returns_emit_slang_operations():
    code = """
    struct SampleResult {
        vec4 sampled;
        vec4 loaded;
        ivec2 texSize;
        ivec2 imageSizeValue;
    };

    shader Resources {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];

        SampleResult buildAssigned(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            SampleResult result;
            result.sampled = texture(dynTex[layer][slot], uv);
            result.loaded = imageLoad(dynImages[layer][slot], pixel);
            result.texSize = textureSize(dynTex[layer][slot], 0);
            result.imageSizeValue = imageSize(dynImages[layer][slot]);
            return result;
        }

        SampleResult buildConstructed(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            return SampleResult(
                texture(dynTex[layer][slot], uv),
                imageLoad(dynImages[layer][slot], pixel),
                textureSize(dynTex[layer][slot], 0),
                imageSize(dynImages[layer][slot])
            );
        }

        vec4 consumeResults(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            SampleResult assigned = buildAssigned(
                dynTex,
                dynImages,
                layer,
                slot,
                uv,
                pixel
            );
            SampleResult constructed = buildConstructed(
                dynTex,
                dynImages,
                layer,
                slot,
                uv,
                pixel
            );
            return assigned.sampled + assigned.loaded +
                constructed.sampled + constructed.loaded;
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                vec4 value = consumeResults(textureGrid, imageGrid, 1, 2, uv, pixel);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct SampleResult" in generated_code
    assert (
        "SampleResult buildAssigned(Sampler2D<float4> dynTex[][3], "
        "RWTexture2D<float4> dynImages[][3], int layer, int slot, "
        "float2 uv, int2 pixel)" in generated_code
    )
    assert (
        "SampleResult buildConstructed(Sampler2D<float4> dynTex[][3], "
        "RWTexture2D<float4> dynImages[][3], int layer, int slot, "
        "float2 uv, int2 pixel)" in generated_code
    )
    assert (
        "float4 consumeResults(Sampler2D<float4> dynTex[][3], "
        "RWTexture2D<float4> dynImages[][3], int layer, int slot, "
        "float2 uv, int2 pixel)" in generated_code
    )
    assert "result.sampled = dynTex[layer][slot].Sample(uv);" in generated_code
    assert "result.loaded = dynImages[layer][slot][pixel];" in generated_code
    assert (
        "result.texSize = cgl_textureSize_sampler2D(dynTex[layer][slot], 0);"
        in generated_code
    )
    assert (
        "result.imageSizeValue = cgl_imageSize_image2D(dynImages[layer][slot]);"
        in generated_code
    )
    assert (
        "return SampleResult(dynTex[layer][slot].Sample(uv), "
        "dynImages[layer][slot][pixel], "
        "cgl_textureSize_sampler2D(dynTex[layer][slot], 0), "
        "cgl_imageSize_image2D(dynImages[layer][slot]));" in generated_code
    )
    assert (
        "SampleResult assigned = buildAssigned(dynTex, dynImages, layer, slot, "
        "uv, pixel);" in generated_code
    )
    assert (
        "SampleResult constructed = buildConstructed(dynTex, dynImages, layer, slot, "
        "uv, pixel);" in generated_code
    )
    assert "consumeResults(textureGrid, imageGrid, 1, 2, uv, pixel);" in generated_code
    assert "texture(" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_nested_struct_resource_assignments_emit_slang_operations():
    code = """
    struct SampleResult {
        vec4 sampled;
        vec4 loaded;
        ivec2 texSize;
        ivec2 imageSizeValue;
    };

    struct SampleEnvelope {
        SampleResult result;
        vec4 bias;
    };

    shader Resources {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];

        SampleEnvelope buildNestedAssigned(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            SampleEnvelope envelope;
            envelope.result.sampled = texture(dynTex[layer][slot], uv);
            envelope.result.loaded = imageLoad(dynImages[layer][slot], pixel);
            envelope.result.texSize = textureSize(dynTex[layer][slot], 0);
            envelope.result.imageSizeValue = imageSize(dynImages[layer][slot]);
            envelope.bias = texture(dynTex[layer][slot], uv) +
                imageLoad(dynImages[layer][slot], pixel);
            return envelope;
        }

        vec4 consumeNested(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            SampleEnvelope envelope = buildNestedAssigned(
                dynTex,
                dynImages,
                layer,
                slot,
                uv,
                pixel
            );
            return envelope.result.sampled + envelope.result.loaded + envelope.bias;
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                vec4 value = consumeNested(textureGrid, imageGrid, 1, 2, uv, pixel);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct SampleEnvelope" in generated_code
    assert (
        "SampleEnvelope buildNestedAssigned(Sampler2D<float4> dynTex[][3], "
        "RWTexture2D<float4> dynImages[][3], int layer, int slot, "
        "float2 uv, int2 pixel)" in generated_code
    )
    assert "envelope.result.sampled = dynTex[layer][slot].Sample(uv);" in generated_code
    assert "envelope.result.loaded = dynImages[layer][slot][pixel];" in generated_code
    assert (
        "envelope.result.texSize = "
        "cgl_textureSize_sampler2D(dynTex[layer][slot], 0);" in generated_code
    )
    assert (
        "envelope.result.imageSizeValue = "
        "cgl_imageSize_image2D(dynImages[layer][slot]);" in generated_code
    )
    assert (
        "envelope.bias = dynTex[layer][slot].Sample(uv) + "
        "dynImages[layer][slot][pixel];" in generated_code
    )
    assert (
        "SampleEnvelope envelope = buildNestedAssigned("
        "dynTex, dynImages, layer, slot, uv, pixel);" in generated_code
    )
    assert "consumeNested(textureGrid, imageGrid, 1, 2, uv, pixel);" in generated_code
    assert "texture(" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_struct_member_array_resource_assignments_emit_slang_operations():
    code = """
    struct SampleResult {
        vec4 sampled;
        vec4 loaded;
        ivec2 texSize;
        ivec2 imageSizeValue;
    };

    struct SampleEnvelope {
        SampleResult results[3];
        vec4 bias;
    };

    shader Resources {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];

        SampleEnvelope buildArrayEnvelope(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            SampleEnvelope envelope;
            envelope.results[slot].sampled = texture(dynTex[layer][slot], uv);
            envelope.results[slot].loaded = imageLoad(dynImages[layer][slot], pixel);
            envelope.results[slot].texSize = textureSize(dynTex[layer][slot], 0);
            envelope.results[slot].imageSizeValue = imageSize(dynImages[layer][slot]);
            envelope.bias = envelope.results[slot].sampled +
                envelope.results[slot].loaded;
            return envelope;
        }

        vec4 consumeArrayEnvelope(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            SampleEnvelope envelope = buildArrayEnvelope(
                dynTex,
                dynImages,
                layer,
                slot,
                uv,
                pixel
            );
            return envelope.results[slot].sampled +
                envelope.results[slot].loaded + envelope.bias;
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                vec4 value = consumeArrayEnvelope(
                    textureGrid,
                    imageGrid,
                    1,
                    2,
                    uv,
                    pixel
                );
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "SampleResult results[3];" in generated_code
    assert (
        "SampleEnvelope buildArrayEnvelope(Sampler2D<float4> dynTex[][3], "
        "RWTexture2D<float4> dynImages[][3], int layer, int slot, "
        "float2 uv, int2 pixel)" in generated_code
    )
    assert (
        "envelope.results[slot].sampled = dynTex[layer][slot].Sample(uv);"
        in generated_code
    )
    assert (
        "envelope.results[slot].loaded = dynImages[layer][slot][pixel];"
        in generated_code
    )
    assert (
        "envelope.results[slot].texSize = "
        "cgl_textureSize_sampler2D(dynTex[layer][slot], 0);" in generated_code
    )
    assert (
        "envelope.results[slot].imageSizeValue = "
        "cgl_imageSize_image2D(dynImages[layer][slot]);" in generated_code
    )
    assert (
        "SampleEnvelope envelope = buildArrayEnvelope("
        "dynTex, dynImages, layer, slot, uv, pixel);" in generated_code
    )
    assert (
        "consumeArrayEnvelope(textureGrid, imageGrid, 1, 2, uv, pixel);"
        in generated_code
    )
    assert "texture(" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_control_flow_struct_resource_assignments_emit_slang_operations():
    code = """
    struct SampleResult {
        vec4 sampled;
        vec4 loaded;
        ivec2 texSize;
        ivec2 imageSizeValue;
    };

    struct SampleEnvelope {
        SampleResult result;
        vec4 accum;
        int chosen;
    };

    shader Resources {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];

        SampleEnvelope fillControlled(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            bool useAlternate,
            vec2 uv,
            ivec2 pixel
        ) {
            SampleEnvelope envelope;
            envelope.accum = vec4(0.0);
            envelope.chosen = slot;

            if (useAlternate) {
                envelope.result.sampled = texture(dynTex[layer][slot], uv);
                envelope.result.loaded = imageLoad(dynImages[layer][slot], pixel);
            } else {
                int fallback = 0;
                envelope.result.sampled = texture(dynTex[fallback][slot], uv);
                envelope.result.loaded = imageLoad(dynImages[fallback][slot], pixel);
            }

            for (int i = 0; i < 3; i++) {
                if (i == 1) {
                    continue;
                }
                envelope.result.texSize = textureSize(dynTex[layer][i], 0);
                envelope.result.imageSizeValue = imageSize(dynImages[layer][i]);
                envelope.accum = envelope.accum + texture(dynTex[layer][i], uv);
                if (i == slot) {
                    break;
                }
            }

            return envelope;
        }

        vec4 consumeControlled(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            SampleEnvelope envelope = fillControlled(
                dynTex,
                dynImages,
                layer,
                slot,
                true,
                uv,
                pixel
            );
            return envelope.result.sampled + envelope.result.loaded +
                envelope.accum;
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                vec4 value = consumeControlled(
                    textureGrid,
                    imageGrid,
                    1,
                    2,
                    uv,
                    pixel
                );
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "SampleEnvelope fillControlled(Sampler2D<float4> dynTex[][3], "
        "RWTexture2D<float4> dynImages[][3], int layer, int slot, "
        "bool useAlternate, float2 uv, int2 pixel)" in generated_code
    )
    assert "if (useAlternate)" in generated_code
    assert "else" in generated_code
    assert "for (int i = 0; i < 3; i++)" in generated_code
    assert "continue;" in generated_code
    assert "break;" in generated_code
    assert "envelope.accum = float4(0.0);" in generated_code
    assert "envelope.chosen = slot;" in generated_code
    assert "envelope.result.sampled = dynTex[layer][slot].Sample(uv);" in generated_code
    assert "envelope.result.loaded = dynImages[layer][slot][pixel];" in generated_code
    assert (
        "envelope.result.sampled = dynTex[fallback][slot].Sample(uv);" in generated_code
    )
    assert (
        "envelope.result.loaded = dynImages[fallback][slot][pixel];" in generated_code
    )
    assert (
        "envelope.result.texSize = cgl_textureSize_sampler2D(dynTex[layer][i], 0);"
        in generated_code
    )
    assert (
        "envelope.result.imageSizeValue = cgl_imageSize_image2D(dynImages[layer][i]);"
        in generated_code
    )
    assert (
        "envelope.accum = envelope.accum + dynTex[layer][i].Sample(uv);"
        in generated_code
    )
    assert (
        "fillControlled(dynTex, dynImages, layer, slot, true, uv, pixel);"
        in generated_code
    )
    assert (
        "consumeControlled(textureGrid, imageGrid, 1, 2, uv, pixel);" in generated_code
    )
    assert "texture(" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_struct_parameter_resource_values_round_trip_slang_operations():
    code = """
    struct SampleResult {
        vec4 sampled;
        vec4 loaded;
        ivec2 texSize;
        ivec2 imageSizeValue;
    };

    shader Resources {
        sampler2d textureGrid[2][3];
        image2D imageGrid @rgba16f[2][3];

        SampleResult buildResourceResult(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            SampleResult result;
            result.sampled = texture(dynTex[layer][slot], uv);
            result.loaded = imageLoad(dynImages[layer][slot], pixel);
            result.texSize = textureSize(dynTex[layer][slot], 0);
            result.imageSizeValue = imageSize(dynImages[layer][slot]);
            return result;
        }

        SampleResult adjustResult(SampleResult payload, float weight) {
            payload.sampled = payload.sampled * weight;
            payload.loaded = payload.loaded + payload.sampled;
            return payload;
        }

        vec4 consumeAdjusted(
            sampler2d dynTex[][3],
            image2D dynImages[][3] @rgba16f,
            int layer,
            int slot,
            vec2 uv,
            ivec2 pixel
        ) {
            SampleResult raw = buildResourceResult(
                dynTex,
                dynImages,
                layer,
                slot,
                uv,
                pixel
            );
            SampleResult adjusted = adjustResult(raw, 0.5);
            return adjusted.sampled + adjusted.loaded;
        }

        compute {
            void main() {
                vec2 uv = vec2(0.5, 0.25);
                ivec2 pixel = ivec2(0, 0);
                vec4 value = consumeAdjusted(textureGrid, imageGrid, 1, 2, uv, pixel);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "SampleResult buildResourceResult(Sampler2D<float4> dynTex[][3], "
        "RWTexture2D<float4> dynImages[][3], int layer, int slot, "
        "float2 uv, int2 pixel)" in generated_code
    )
    assert (
        "SampleResult adjustResult(SampleResult payload, float weight)"
        in generated_code
    )
    assert (
        "float4 consumeAdjusted(Sampler2D<float4> dynTex[][3], "
        "RWTexture2D<float4> dynImages[][3], int layer, int slot, "
        "float2 uv, int2 pixel)" in generated_code
    )
    assert "payload.sampled = payload.sampled * weight;" in generated_code
    assert "payload.loaded = payload.loaded + payload.sampled;" in generated_code
    assert "return payload;" in generated_code
    assert (
        "SampleResult raw = buildResourceResult(dynTex, dynImages, layer, slot, "
        "uv, pixel);" in generated_code
    )
    assert "SampleResult adjusted = adjustResult(raw, 0.5);" in generated_code
    assert "consumeAdjusted(textureGrid, imageGrid, 1, 2, uv, pixel);" in generated_code
    assert "texture(" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "unsupported Slang" not in generated_code


def test_mutable_scalar_and_array_params_emit_slang_updates():
    code = """
    shader MutableParams {
        float bumpScalar(float weight) {
            weight = weight + 1.0;
            return weight * 2.0;
        }

        float bumpArray(float values[3], int slot) {
            values[slot] = values[slot] + 1.0;
            values[0] = values[slot] * 0.5;
            return values[slot] + values[0];
        }

        compute {
            void main() {
                float values[3];
                values[0] = 1.0;
                values[1] = 2.0;
                values[2] = 3.0;
                float a = bumpScalar(0.5);
                float b = bumpArray(values, 1);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float bumpScalar(float weight)" in generated_code
    assert "weight = weight + 1.0;" in generated_code
    assert "return weight * 2.0;" in generated_code
    assert "float bumpArray(float values[3], int slot)" in generated_code
    assert "values[slot] = values[slot] + 1.0;" in generated_code
    assert "values[0] = values[slot] * 0.5;" in generated_code
    assert "return values[slot] + values[0];" in generated_code
    assert "float values[3];" in generated_code
    assert "float a = bumpScalar(0.5);" in generated_code
    assert "float b = bumpArray(values, 1);" in generated_code
    assert "unsupported Slang" not in generated_code


def test_array_literals_emit_slang_brace_initializers():
    code = """
    float globalWeights[4] = {1.0, 2.0};

    float pickScalar(int index) {
        float values[4] = {1.0, 2.0, 3.0, 4.0};
        return values[index];
    }

    vec3 pickColor(int index) {
        vec3 colors[2] = {vec3(1.0, 2.0, 3.0), vec3(4.0, 5.0, 6.0)};
        return colors[index];
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float globalWeights[4] = {1.0, 2.0};" in generated_code
    assert "float values[4] = {1.0, 2.0, 3.0, 4.0};" in generated_code
    assert (
        "float3 colors[2] = {float3(1.0, 2.0, 3.0), " "float3(4.0, 5.0, 6.0)};"
    ) in generated_code
    assert "ArrayLiteralNode" not in generated_code


def test_nested_array_and_struct_member_params_emit_slang_updates():
    code = """
    struct Payload {
        float values[3];
        float bias;
    };

    shader NestedMutableParams {
        float bumpNested(float values[2][3], int row, int col) {
            values[row][col] = values[row][col] + 1.0;
            values[0][col] = values[row][col] * 0.5;
            return values[row][col] + values[0][col];
        }

        float bumpPayload(Payload payload, int slot) {
            payload.values[slot] = payload.values[slot] + payload.bias;
            payload.bias = payload.values[slot] * 0.5;
            return payload.values[slot] + payload.bias;
        }

        compute {
            void main() {
                float grid[2][3];
                grid[0][0] = 1.0;
                grid[1][2] = 3.0;
                Payload payload;
                payload.values[0] = 2.0;
                payload.values[1] = 4.0;
                payload.bias = 0.25;
                float a = bumpNested(grid, 1, 2);
                float b = bumpPayload(payload, 1);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct Payload" in generated_code
    assert "float values[3];" in generated_code
    assert "float bumpNested(float values[2][3], int row, int col)" in generated_code
    assert "values[row][col] = values[row][col] + 1.0;" in generated_code
    assert "values[0][col] = values[row][col] * 0.5;" in generated_code
    assert "return values[row][col] + values[0][col];" in generated_code
    assert "float bumpPayload(Payload payload, int slot)" in generated_code
    assert (
        "payload.values[slot] = payload.values[slot] + payload.bias;" in generated_code
    )
    assert "payload.bias = payload.values[slot] * 0.5;" in generated_code
    assert "return payload.values[slot] + payload.bias;" in generated_code
    assert "float grid[2][3];" in generated_code
    assert "float a = bumpNested(grid, 1, 2);" in generated_code
    assert "float b = bumpPayload(payload, 1);" in generated_code
    assert "unsupported Slang" not in generated_code


def test_returned_nested_struct_params_emit_slang_updates():
    code = """
    struct InnerPayload {
        float values[3];
        float bias;
    };

    struct OuterPayload {
        InnerPayload inner;
        float scale;
    };

    shader ReturnedParamPayloads {
        OuterPayload makeOuter(float first, float second, float bias, float scale) {
            OuterPayload outer;
            outer.inner.values[0] = first;
            outer.inner.values[1] = second;
            outer.inner.values[2] = first + second;
            outer.inner.bias = bias;
            outer.scale = scale;
            return outer;
        }

        OuterPayload adjustOuter(OuterPayload payload, int slot) {
            payload.inner.values[slot] = payload.inner.values[slot] + payload.inner.bias;
            payload.inner.values[0] = payload.inner.values[slot] * payload.scale;
            payload.scale = payload.inner.values[0] + payload.inner.bias;
            return payload;
        }

        float consumeAdjustedOuter(int slot) {
            OuterPayload adjusted = adjustOuter(makeOuter(1.0, 2.0, 0.25, 4.0), slot);
            return adjusted.inner.values[slot] + adjusted.inner.values[0] + adjusted.scale;
        }

        compute {
            void main() {
                float value = consumeAdjustedOuter(1);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "OuterPayload adjustOuter(OuterPayload payload, int slot)" in generated_code
    assert (
        "payload.inner.values[slot] = payload.inner.values[slot] + payload.inner.bias;"
        in generated_code
    )
    assert (
        "payload.inner.values[0] = payload.inner.values[slot] * payload.scale;"
        in generated_code
    )
    assert (
        "payload.scale = payload.inner.values[0] + payload.inner.bias;"
        in generated_code
    )
    assert "return payload;" in generated_code
    assert "float consumeAdjustedOuter(int slot)" in generated_code
    assert (
        "OuterPayload adjusted = adjustOuter(makeOuter(1.0, 2.0, 0.25, 4.0), slot);"
        in generated_code
    )
    assert (
        "return adjusted.inner.values[slot] + adjusted.inner.values[0] + "
        "adjusted.scale;" in generated_code
    )
    assert "float value = consumeAdjustedOuter(1);" in generated_code
    assert "unsupported Slang" not in generated_code


def test_conditional_returned_nested_structs_emit_slang_expressions():
    code = """
    struct InnerPayload {
        float values[3];
        float bias;
    };

    struct OuterPayload {
        InnerPayload inner;
        float scale;
    };

    shader ConditionalReturnedPayloads {
        OuterPayload makeOuter(float first, float second, float bias, float scale) {
            OuterPayload outer;
            outer.inner.values[0] = first;
            outer.inner.values[1] = second;
            outer.inner.values[2] = first + second;
            outer.inner.bias = bias;
            outer.scale = scale;
            return outer;
        }

        OuterPayload chooseBranch(bool useSecond) {
            if (useSecond) {
                return makeOuter(3.0, 4.0, 0.5, 5.0);
            }
            return makeOuter(1.0, 2.0, 0.25, 4.0);
        }

        OuterPayload chooseTernary(bool useSecond) {
            return useSecond
                ? makeOuter(3.0, 4.0, 0.5, 5.0)
                : makeOuter(1.0, 2.0, 0.25, 4.0);
        }

        float consumeConditional(int slot, bool useSecond) {
            OuterPayload branchPayload = chooseBranch(useSecond);
            OuterPayload ternaryPayload = chooseTernary(useSecond);
            ternaryPayload.inner.values[slot] =
                ternaryPayload.inner.values[slot] + ternaryPayload.inner.bias;
            branchPayload.scale =
                branchPayload.inner.values[slot] + ternaryPayload.scale;
            return branchPayload.scale + ternaryPayload.inner.values[slot];
        }

        compute {
            void main() {
                float value = consumeConditional(1, true);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "OuterPayload chooseBranch(bool useSecond)" in generated_code
    assert "return makeOuter(3.0, 4.0, 0.5, 5.0);" in generated_code
    assert "return makeOuter(1.0, 2.0, 0.25, 4.0);" in generated_code
    assert "OuterPayload chooseTernary(bool useSecond)" in generated_code
    assert (
        "return (useSecond ? makeOuter(3.0, 4.0, 0.5, 5.0) : "
        "makeOuter(1.0, 2.0, 0.25, 4.0));" in generated_code
    )
    assert "OuterPayload branchPayload = chooseBranch(useSecond);" in generated_code
    assert "OuterPayload ternaryPayload = chooseTernary(useSecond);" in generated_code
    assert (
        "ternaryPayload.inner.values[slot] = "
        "ternaryPayload.inner.values[slot] + ternaryPayload.inner.bias;"
        in generated_code
    )
    assert (
        "branchPayload.scale = branchPayload.inner.values[slot] + "
        "ternaryPayload.scale;" in generated_code
    )
    assert (
        "return branchPayload.scale + ternaryPayload.inner.values[slot];"
        in generated_code
    )
    assert "float value = consumeConditional(1, true);" in generated_code
    assert "unsupported Slang" not in generated_code


def test_temporary_struct_array_member_reads_emit_slang_expressions():
    code = """
    struct Payload {
        float values[3];
        float bias;
    };

    shader TemporaryPayloads {
        Payload makePayload(float first, float second, float bias) {
            Payload payload;
            payload.values[0] = first;
            payload.values[1] = second;
            payload.values[2] = first + second;
            payload.bias = bias;
            return payload;
        }

        float readTemporary(int slot) {
            float dynamicValue = makePayload(1.0, 2.0, 0.25).values[slot];
            float fixedValue = makePayload(3.0, 4.0, 0.5).values[0];
            float biasValue = makePayload(5.0, 6.0, 0.75).bias;
            return dynamicValue + fixedValue + biasValue;
        }

        compute {
            void main() {
                float value = readTemporary(1);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "Payload makePayload(float first, float second, float bias)" in generated_code
    )
    assert "payload.values[2] = first + second;" in generated_code
    assert "float readTemporary(int slot)" in generated_code
    assert (
        "float dynamicValue = makePayload(1.0, 2.0, 0.25).values[slot];"
        in generated_code
    )
    assert "float fixedValue = makePayload(3.0, 4.0, 0.5).values[0];" in generated_code
    assert "float biasValue = makePayload(5.0, 6.0, 0.75).bias;" in generated_code
    assert "return dynamicValue + fixedValue + biasValue;" in generated_code
    assert "float value = readTemporary(1);" in generated_code
    assert "unsupported Slang" not in generated_code


def test_nested_temporary_struct_array_member_reads_emit_slang_expressions():
    code = """
    struct InnerPayload {
        float values[3];
        float bias;
    };

    struct OuterPayload {
        InnerPayload inner;
        float scale;
    };

    shader NestedTemporaryPayloads {
        OuterPayload makeOuter(float first, float second, float bias, float scale) {
            OuterPayload outer;
            outer.inner.values[0] = first;
            outer.inner.values[1] = second;
            outer.inner.values[2] = first + second;
            outer.inner.bias = bias;
            outer.scale = scale;
            return outer;
        }

        float readNestedTemporary(int slot) {
            float dynamicValue = makeOuter(1.0, 2.0, 0.25, 4.0).inner.values[slot];
            float fixedValue = makeOuter(3.0, 4.0, 0.5, 5.0).inner.values[0];
            float biasValue = makeOuter(5.0, 6.0, 0.75, 6.0).inner.bias;
            float scaleValue = makeOuter(7.0, 8.0, 1.0, 9.0).scale;
            return dynamicValue + fixedValue + biasValue + scaleValue;
        }

        compute {
            void main() {
                float value = readNestedTemporary(1);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "struct InnerPayload" in generated_code
    assert "struct OuterPayload" in generated_code
    assert (
        "OuterPayload makeOuter(float first, float second, float bias, float scale)"
        in generated_code
    )
    assert "outer.inner.values[2] = first + second;" in generated_code
    assert "outer.inner.bias = bias;" in generated_code
    assert "outer.scale = scale;" in generated_code
    assert "float readNestedTemporary(int slot)" in generated_code
    assert (
        "float dynamicValue = "
        "makeOuter(1.0, 2.0, 0.25, 4.0).inner.values[slot];" in generated_code
    )
    assert (
        "float fixedValue = makeOuter(3.0, 4.0, 0.5, 5.0).inner.values[0];"
        in generated_code
    )
    assert (
        "float biasValue = makeOuter(5.0, 6.0, 0.75, 6.0).inner.bias;" in generated_code
    )
    assert "float scaleValue = makeOuter(7.0, 8.0, 1.0, 9.0).scale;" in generated_code
    assert (
        "return dynamicValue + fixedValue + biasValue + scaleValue;" in generated_code
    )
    assert "float value = readNestedTemporary(1);" in generated_code
    assert "unsupported Slang" not in generated_code


def test_returned_local_nested_struct_array_writes_emit_slang_expressions():
    code = """
    struct InnerPayload {
        float values[3];
        float bias;
    };

    struct OuterPayload {
        InnerPayload inner;
        float scale;
    };

    shader LocalReturnedPayloads {
        OuterPayload makeOuter(float first, float second, float bias, float scale) {
            OuterPayload outer;
            outer.inner.values[0] = first;
            outer.inner.values[1] = second;
            outer.inner.values[2] = first + second;
            outer.inner.bias = bias;
            outer.scale = scale;
            return outer;
        }

        float mutateReturnedLocal(int slot) {
            OuterPayload outer = makeOuter(1.0, 2.0, 0.25, 4.0);
            outer.inner.values[slot] = outer.inner.values[slot] + outer.inner.bias;
            outer.inner.values[0] = outer.inner.values[slot] * outer.scale;
            outer.scale = outer.inner.values[0] + outer.inner.bias;
            return outer.inner.values[slot] + outer.inner.values[0] + outer.scale;
        }

        compute {
            void main() {
                float value = mutateReturnedLocal(1);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "OuterPayload outer = makeOuter(1.0, 2.0, 0.25, 4.0);" in generated_code
    assert (
        "outer.inner.values[slot] = outer.inner.values[slot] + outer.inner.bias;"
        in generated_code
    )
    assert (
        "outer.inner.values[0] = outer.inner.values[slot] * outer.scale;"
        in generated_code
    )
    assert "outer.scale = outer.inner.values[0] + outer.inner.bias;" in generated_code
    assert (
        "return outer.inner.values[slot] + outer.inner.values[0] + outer.scale;"
        in generated_code
    )
    assert "float value = mutateReturnedLocal(1);" in generated_code
    assert "unsupported Slang" not in generated_code


def test_returned_local_nested_struct_array_writes_in_control_flow_emit_slang():
    code = """
    struct InnerPayload {
        float values[3];
        float bias;
    };

    struct OuterPayload {
        InnerPayload inner;
        float scale;
    };

    shader ControlFlowReturnedPayloads {
        OuterPayload makeOuter(float first, float second, float bias, float scale) {
            OuterPayload outer;
            outer.inner.values[0] = first;
            outer.inner.values[1] = second;
            outer.inner.values[2] = first + second;
            outer.inner.bias = bias;
            outer.scale = scale;
            return outer;
        }

        float mutateReturnedLocalControl(int slot) {
            OuterPayload outer = makeOuter(1.0, 2.0, 0.25, 4.0);
            for (int i = 0; i < 3; i++) {
                outer.inner.values[i] = outer.inner.values[i] + outer.inner.bias;
                if (i == slot) {
                    outer.inner.values[i] = outer.inner.values[i] * outer.scale;
                }
            }
            if (slot > 1) {
                outer.scale = outer.inner.values[slot] + outer.inner.bias;
            } else {
                outer.scale = outer.inner.values[0] - outer.inner.bias;
            }
            return outer.inner.values[slot] + outer.scale;
        }

        compute {
            void main() {
                float value = mutateReturnedLocalControl(1);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float mutateReturnedLocalControl(int slot)" in generated_code
    assert "OuterPayload outer = makeOuter(1.0, 2.0, 0.25, 4.0);" in generated_code
    assert "for (int i = 0; i < 3; i++)" in generated_code
    assert (
        "outer.inner.values[i] = outer.inner.values[i] + outer.inner.bias;"
        in generated_code
    )
    assert "if (i == slot)" in generated_code
    assert (
        "outer.inner.values[i] = outer.inner.values[i] * outer.scale;" in generated_code
    )
    assert "if (slot > 1)" in generated_code
    assert (
        "outer.scale = outer.inner.values[slot] + outer.inner.bias;" in generated_code
    )
    assert "outer.scale = outer.inner.values[0] - outer.inner.bias;" in generated_code
    assert "return outer.inner.values[slot] + outer.scale;" in generated_code
    assert "float value = mutateReturnedLocalControl(1);" in generated_code
    assert "unsupported Slang" not in generated_code


def test_resource_query_builtins_emit_slang_get_dimensions_helpers():
    code = """
    shader Resources {
        sampler2d colorMap;
        sampler2darray layers;
        sampler3d volumeTex;
        sampler2dshadow shadowMap;
        sampler2darrayshadow shadowLayers;
        samplercubeshadow cubeShadow;
        samplercubearrayshadow cubeShadowLayers;
        sampler2dms msTex;
        sampler2dmsarray msLayers;
        image2D colorImage;
        image3D volumeImage;
        image2DArray layerImage;
        image2DMS msImage;
        image2DMSArray msImageLayers;
        uimage2DMS counters;

        compute {
            void main() {
                ivec2 texSize = textureSize(colorMap, 2);
                ivec3 layerSize = textureSize(layers, 1);
                ivec3 volumeSize = textureSize(volumeTex, 0);
                ivec2 shadowSize = textureSize(shadowMap, 0);
                ivec3 shadowLayerSize = textureSize(shadowLayers, 0);
                ivec2 cubeShadowSize = textureSize(cubeShadow, 0);
                ivec3 cubeShadowLayerSize = textureSize(cubeShadowLayers, 0);
                ivec2 msSize = textureSize(msTex, 0);
                ivec3 msLayerSize = textureSize(msLayers, 0);
                int texLevels = textureQueryLevels(colorMap);
                int layerLevels = textureQueryLevels(layers);
                int volumeLevels = textureQueryLevels(volumeTex);
                int shadowLevels = textureQueryLevels(shadowMap);
                int shadowLayerLevels = textureQueryLevels(shadowLayers);
                int cubeShadowLevels = textureQueryLevels(cubeShadow);
                int cubeShadowLayerLevels = textureQueryLevels(cubeShadowLayers);
                ivec2 imageSize2d = imageSize(colorImage);
                ivec3 imageSize3d = imageSize(volumeImage);
                ivec3 imageSizeLayer = imageSize(layerImage);
                ivec2 msImageSize = imageSize(msImage);
                ivec3 msImageLayerSize = imageSize(msImageLayers);
                int texSamples = textureSamples(msTex);
                int texLayerSamples = textureSamples(msLayers);
                int imageSamplesValue = imageSamples(msImage);
                int counterSamples = imageSamples(counters);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "int2 cgl_textureSize_sampler2D(Sampler2D<float4> tex, uint mipLevel)"
        in generated_code
    )
    assert "tex.GetDimensions(mipLevel, width, height, levels);" in generated_code
    assert (
        "int3 cgl_textureSize_sampler2DArray("
        "Sampler2DArray<float4> tex, uint mipLevel)" in generated_code
    )
    assert (
        "tex.GetDimensions(mipLevel, width, height, elements, levels);"
        in generated_code
    )
    assert (
        "int3 cgl_textureSize_sampler3D(Sampler3D<float4> tex, uint mipLevel)"
        in generated_code
    )
    assert (
        "tex.GetDimensions(mipLevel, width, height, depth, levels);" in generated_code
    )
    assert (
        "int2 cgl_textureSize_sampler2DShadow(Sampler2DShadow tex, uint mipLevel)"
        in generated_code
    )
    assert (
        "int3 cgl_textureSize_sampler2DArrayShadow("
        "Sampler2DArrayShadow tex, uint mipLevel)" in generated_code
    )
    assert (
        "int2 cgl_textureSize_samplerCubeShadow(SamplerCubeShadow tex, uint mipLevel)"
        in generated_code
    )
    assert (
        "int3 cgl_textureSize_samplerCubeArrayShadow("
        "SamplerCubeArrayShadow tex, uint mipLevel)" in generated_code
    )
    assert (
        "int cgl_textureQueryLevels_sampler2D(Sampler2D<float4> tex)" in generated_code
    )
    assert "tex.GetDimensions(width, height, levels);" in generated_code
    assert (
        "int cgl_textureQueryLevels_sampler2DArray(Sampler2DArray<float4> tex)"
        in generated_code
    )
    assert "tex.GetDimensions(width, height, elements, levels);" in generated_code
    assert (
        "int cgl_textureQueryLevels_sampler3D(Sampler3D<float4> tex)" in generated_code
    )
    assert "tex.GetDimensions(width, height, depth, levels);" in generated_code
    assert (
        "int cgl_textureQueryLevels_sampler2DShadow(Sampler2DShadow tex)"
        in generated_code
    )
    assert (
        "int cgl_textureQueryLevels_sampler2DArrayShadow(Sampler2DArrayShadow tex)"
        in generated_code
    )
    assert (
        "int cgl_textureQueryLevels_samplerCubeShadow(SamplerCubeShadow tex)"
        in generated_code
    )
    assert (
        "int cgl_textureQueryLevels_samplerCubeArrayShadow("
        "SamplerCubeArrayShadow tex)" in generated_code
    )
    assert "int2 cgl_textureSize_sampler2DMS(Sampler2DMS<float4> tex)" in generated_code
    assert (
        "int3 cgl_textureSize_sampler2DMSArray(Sampler2DMSArray<float4> tex)"
        in generated_code
    )
    assert "tex.GetDimensions(width, height, samples);" in generated_code
    assert "tex.GetDimensions(width, height, elements, samples);" in generated_code
    assert "int2 cgl_imageSize_image2D(RWTexture2D<float4> tex)" in generated_code
    assert "int3 cgl_imageSize_image3D(RWTexture3D<float4> tex)" in generated_code
    assert (
        "int3 cgl_imageSize_image2DArray(RWTexture2DArray<float4> tex)"
        in generated_code
    )
    assert "int2 cgl_imageSize_image2DMS(RWTexture2DMS<float4> tex)" in generated_code
    assert (
        "int3 cgl_imageSize_image2DMSArray(RWTexture2DMSArray<float4> tex)"
        in generated_code
    )
    assert (
        "int cgl_textureSamples_sampler2DMS(Sampler2DMS<float4> tex)" in generated_code
    )
    assert (
        "int cgl_textureSamples_sampler2DMSArray(Sampler2DMSArray<float4> tex)"
        in generated_code
    )
    assert "int cgl_imageSamples_image2DMS(RWTexture2DMS<float4> tex)" in generated_code
    assert "int cgl_imageSamples_uimage2DMS(RWTexture2DMS<uint> tex)" in generated_code
    assert "return samples;" in generated_code
    assert "int2 texSize = cgl_textureSize_sampler2D(colorMap, 2);" in generated_code
    assert (
        "int3 layerSize = cgl_textureSize_sampler2DArray(layers, 1);" in generated_code
    )
    assert (
        "int3 volumeSize = cgl_textureSize_sampler3D(volumeTex, 0);" in generated_code
    )
    assert (
        "int2 shadowSize = cgl_textureSize_sampler2DShadow(shadowMap, 0);"
        in generated_code
    )
    assert (
        "int3 shadowLayerSize = cgl_textureSize_sampler2DArrayShadow("
        "shadowLayers, 0);" in generated_code
    )
    assert (
        "int2 cubeShadowSize = cgl_textureSize_samplerCubeShadow(cubeShadow, 0);"
        in generated_code
    )
    assert (
        "int3 cubeShadowLayerSize = cgl_textureSize_samplerCubeArrayShadow("
        "cubeShadowLayers, 0);" in generated_code
    )
    assert (
        "int texLevels = cgl_textureQueryLevels_sampler2D(colorMap);" in generated_code
    )
    assert (
        "int layerLevels = cgl_textureQueryLevels_sampler2DArray(layers);"
        in generated_code
    )
    assert (
        "int volumeLevels = cgl_textureQueryLevels_sampler3D(volumeTex);"
        in generated_code
    )
    assert (
        "int shadowLevels = cgl_textureQueryLevels_sampler2DShadow(shadowMap);"
        in generated_code
    )
    assert (
        "int shadowLayerLevels = cgl_textureQueryLevels_sampler2DArrayShadow("
        "shadowLayers);" in generated_code
    )
    assert (
        "int cubeShadowLevels = cgl_textureQueryLevels_samplerCubeShadow(cubeShadow);"
        in generated_code
    )
    assert (
        "int cubeShadowLayerLevels = cgl_textureQueryLevels_samplerCubeArrayShadow("
        "cubeShadowLayers);" in generated_code
    )
    assert "int2 msSize = cgl_textureSize_sampler2DMS(msTex);" in generated_code
    assert (
        "int3 msLayerSize = cgl_textureSize_sampler2DMSArray(msLayers);"
        in generated_code
    )
    assert "int2 imageSize2d = cgl_imageSize_image2D(colorImage);" in generated_code
    assert "int3 imageSize3d = cgl_imageSize_image3D(volumeImage);" in generated_code
    assert (
        "int3 imageSizeLayer = cgl_imageSize_image2DArray(layerImage);"
        in generated_code
    )
    assert "int2 msImageSize = cgl_imageSize_image2DMS(msImage);" in generated_code
    assert (
        "int3 msImageLayerSize = cgl_imageSize_image2DMSArray(msImageLayers);"
        in generated_code
    )
    assert "int texSamples = cgl_textureSamples_sampler2DMS(msTex);" in generated_code
    assert (
        "int texLayerSamples = cgl_textureSamples_sampler2DMSArray(msLayers);"
        in generated_code
    )
    assert (
        "int imageSamplesValue = cgl_imageSamples_image2DMS(msImage);" in generated_code
    )
    assert (
        "int counterSamples = cgl_imageSamples_uimage2DMS(counters);" in generated_code
    )
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "textureSamples(" not in generated_code
    assert "imageSamples(" not in generated_code
    assert "textureQueryLevels(" not in generated_code


def test_unsupported_resource_query_combinations_emit_slang_diagnostics():
    code = """
    struct QueryShapeResult {
        int scalarTextureSize;
        ivec2 vectorTextureSize;
        int scalarImageSize;
        ivec2 vectorImageSize;
    };

    struct QueryEnvelope {
        QueryShapeResult result;
    };

    shader UnsupportedResourceQueries {
        sampler querySampler;
        sampler1d line;
        sampler2d colorMap;
        sampler2dms msTex;
        sampler2dmsarray msLayers;
        image1D values;
        image2D colorImage;
        image2DMS msImage;

        compute {
            void acceptScalarTexture(int value) {
            }

            void acceptVectorTexture(ivec2 value) {
            }

            void acceptEnvelope(QueryEnvelope envelope) {
            }

            void main() {
                vec2 badMip = vec2(0.0, 1.0);
                int missingTextureSize = textureSize();
                int missingImageSize = imageSize();
                int textureSizeFromSampler = textureSize(querySampler, 0);
                int textureSizeFromImage = textureSize(colorImage, 0);
                int imageSizeFromTexture = imageSize(colorMap);
                int imageSizeFromSampler = imageSize(querySampler);
                int extraTextureSize = textureSize(colorMap, 0, 1);
                int badTextureMip = textureSize(colorMap, badMip);
                int badMultisampleTextureMip = textureSize(msTex, badMip);
                int extraImageSize = imageSize(colorImage, 0);
                int msLevels = textureQueryLevels(msTex);
                int msArrayLevels = textureQueryLevels(msLayers);
                int plainTextureSamples = textureSamples(colorMap);
                int plainImageSamples = imageSamples(colorImage);
                int imageSamplesFromSampler = imageSamples(msTex);
                int textureSamplesFromImage = textureSamples(msImage);
                int missingTextureSamples = textureSamples();
                int missingImageSamples = imageSamples();
                int missingLevels = textureQueryLevels();
                int extraTextureSamples = textureSamples(msTex, 0);
                int extraImageSamples = imageSamples(msImage, 0);
                int extraLevels = textureQueryLevels(colorMap, 0);
                ivec3 scalarTextureTarget = textureSize(line, 0);
                int vectorTextureTarget = textureSize(colorMap, 0);
                ivec2 scalarImageTarget = imageSize(values);
                int vectorImageTarget = imageSize(colorImage);
                ivec2 assignedTarget;
                assignedTarget = imageSize(values);
                QueryShapeResult fields;
                fields.vectorTextureSize = textureSize(line, 0);
                fields.scalarTextureSize = textureSize(colorMap, 0);
                fields.vectorImageSize = imageSize(values);
                fields.scalarImageSize = imageSize(colorImage);
                QueryShapeResult constructed = QueryShapeResult(
                    textureSize(colorMap, 0),
                    textureSize(line, 0),
                    imageSize(colorImage),
                    imageSize(values)
                );
                QueryEnvelope nestedConstructed = QueryEnvelope(QueryShapeResult(
                    textureSize(colorMap, 0),
                    textureSize(line, 0),
                    imageSize(colorImage),
                    imageSize(values)
                ));
                acceptScalarTexture(textureSize(colorMap, 0));
                acceptVectorTexture(textureSize(line, 0));
                acceptEnvelope(QueryEnvelope(QueryShapeResult(
                    textureSize(colorMap, 0),
                    textureSize(line, 0),
                    imageSize(colorImage),
                    imageSize(values)
                )));
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "int missingTextureSize = /* unsupported Slang resource query: "
        "textureSize requires a resource argument */ 0;" in generated_code
    )
    assert (
        "int missingImageSize = /* unsupported Slang resource query: "
        "imageSize requires a resource argument */ 0;" in generated_code
    )
    assert (
        "int textureSizeFromSampler = /* unsupported Slang resource query: "
        "textureSize requires a sampled texture resource */ 0;" in generated_code
    )
    assert (
        "int textureSizeFromImage = /* unsupported Slang resource query: "
        "textureSize requires a sampled texture resource */ 0;" in generated_code
    )
    assert (
        "int imageSizeFromTexture = /* unsupported Slang resource query: "
        "imageSize requires an image resource */ 0;" in generated_code
    )
    assert (
        "int imageSizeFromSampler = /* unsupported Slang resource query: "
        "imageSize requires an image resource */ 0;" in generated_code
    )
    assert (
        "int extraTextureSize = /* unsupported Slang resource query: "
        "textureSize accepts resource and optional mip argument */ 0;" in generated_code
    )
    assert (
        "int badTextureMip = /* unsupported Slang resource query: "
        "textureSize requires a scalar mip argument */ 0;" in generated_code
    )
    assert (
        "int badMultisampleTextureMip = /* unsupported Slang resource query: "
        "textureSize requires a scalar mip argument */ 0;" in generated_code
    )
    assert (
        "int extraImageSize = /* unsupported Slang resource query: "
        "imageSize accepts only a resource argument */ 0;" in generated_code
    )
    assert (
        "int msLevels = /* unsupported Slang resource query: "
        "textureQueryLevels requires a mipmapped sampled texture resource */ 0;"
        in generated_code
    )
    assert (
        "int msArrayLevels = /* unsupported Slang resource query: "
        "textureQueryLevels requires a mipmapped sampled texture resource */ 0;"
        in generated_code
    )
    assert (
        "int plainTextureSamples = /* unsupported Slang resource query: "
        "textureSamples requires a multisampled texture resource */ 0;"
        in generated_code
    )
    assert (
        "int plainImageSamples = /* unsupported Slang resource query: "
        "imageSamples requires a multisampled image resource */ 0;" in generated_code
    )
    assert (
        "int imageSamplesFromSampler = /* unsupported Slang resource query: "
        "imageSamples requires a multisampled image resource */ 0;" in generated_code
    )
    assert (
        "int textureSamplesFromImage = /* unsupported Slang resource query: "
        "textureSamples requires a multisampled texture resource */ 0;"
        in generated_code
    )
    assert (
        "int missingTextureSamples = /* unsupported Slang resource query: "
        "textureSamples requires a resource argument */ 0;" in generated_code
    )
    assert (
        "int missingImageSamples = /* unsupported Slang resource query: "
        "imageSamples requires a resource argument */ 0;" in generated_code
    )
    assert (
        "int missingLevels = /* unsupported Slang resource query: "
        "textureQueryLevels requires a resource argument */ 0;" in generated_code
    )
    assert (
        "int extraTextureSamples = /* unsupported Slang resource query: "
        "textureSamples accepts only a resource argument */ 0;" in generated_code
    )
    assert (
        "int extraImageSamples = /* unsupported Slang resource query: "
        "imageSamples accepts only a resource argument */ 0;" in generated_code
    )
    assert (
        "int extraLevels = /* unsupported Slang resource query: "
        "textureQueryLevels accepts only a resource argument */ 0;" in generated_code
    )
    assert (
        "int3 scalarTextureTarget = /* unsupported Slang resource query: "
        "textureSize returns int but target expects int3 */ int3(0);" in generated_code
    )
    assert (
        "int vectorTextureTarget = /* unsupported Slang resource query: "
        "textureSize returns int2 but target expects int */ 0;" in generated_code
    )
    assert (
        "int2 scalarImageTarget = /* unsupported Slang resource query: "
        "imageSize returns int but target expects int2 */ int2(0);" in generated_code
    )
    assert (
        "int vectorImageTarget = /* unsupported Slang resource query: "
        "imageSize returns int2 but target expects int */ 0;" in generated_code
    )
    assert (
        "assignedTarget = /* unsupported Slang resource query: "
        "imageSize returns int but target expects int2 */ int2(0);" in generated_code
    )
    assert (
        "fields.vectorTextureSize = /* unsupported Slang resource query: "
        "textureSize returns int but target expects int2 */ int2(0);" in generated_code
    )
    assert (
        "fields.scalarTextureSize = /* unsupported Slang resource query: "
        "textureSize returns int2 but target expects int */ 0;" in generated_code
    )
    assert (
        "fields.vectorImageSize = /* unsupported Slang resource query: "
        "imageSize returns int but target expects int2 */ int2(0);" in generated_code
    )
    assert (
        "fields.scalarImageSize = /* unsupported Slang resource query: "
        "imageSize returns int2 but target expects int */ 0;" in generated_code
    )
    assert (
        "QueryShapeResult constructed = QueryShapeResult("
        "/* unsupported Slang resource query: textureSize returns int2 but target "
        "expects int */ 0, "
        "/* unsupported Slang resource query: textureSize returns int but target "
        "expects int2 */ int2(0), "
        "/* unsupported Slang resource query: imageSize returns int2 but target "
        "expects int */ 0, "
        "/* unsupported Slang resource query: imageSize returns int but target "
        "expects int2 */ int2(0));" in generated_code
    )
    assert (
        "QueryEnvelope nestedConstructed = QueryEnvelope(QueryShapeResult("
        "/* unsupported Slang resource query: textureSize returns int2 but target "
        "expects int */ 0, "
        "/* unsupported Slang resource query: textureSize returns int but target "
        "expects int2 */ int2(0), "
        "/* unsupported Slang resource query: imageSize returns int2 but target "
        "expects int */ 0, "
        "/* unsupported Slang resource query: imageSize returns int but target "
        "expects int2 */ int2(0)));" in generated_code
    )
    assert (
        "acceptScalarTexture(/* unsupported Slang resource query: textureSize "
        "returns int2 but target expects int */ 0);" in generated_code
    )
    assert (
        "acceptVectorTexture(/* unsupported Slang resource query: textureSize "
        "returns int but target expects int2 */ int2(0));" in generated_code
    )
    assert (
        "acceptEnvelope(QueryEnvelope(QueryShapeResult("
        "/* unsupported Slang resource query: textureSize returns int2 but target "
        "expects int */ 0, "
        "/* unsupported Slang resource query: textureSize returns int but target "
        "expects int2 */ int2(0), "
        "/* unsupported Slang resource query: imageSize returns int2 but target "
        "expects int */ 0, "
        "/* unsupported Slang resource query: imageSize returns int but target "
        "expects int2 */ int2(0))));" in generated_code
    )
    assert generated_code.count("unsupported Slang resource query") == 45
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "textureQueryLevels(" not in generated_code
    assert "textureSamples(" not in generated_code
    assert "imageSamples(" not in generated_code


def test_resource_query_expected_types_flow_into_ternaries_and_array_literals():
    code = """
    struct QueryShapeResult {
        int scalarTextureSize;
        ivec2 vectorTextureSize;
        int scalarImageSize;
        ivec2 vectorImageSize;
    };

    shader ResourceQueryExpectedContexts {
        sampler1d line;
        sampler2d colorMap;
        image1D values;
        image2D colorImage;

        compute {
            int chooseScalar(bool take) {
                return take ? textureSize(colorMap, 0) : imageSize(colorImage);
            }

            ivec2 chooseVector(bool take) {
                return take ? textureSize(line, 0) : imageSize(values);
            }

            void acceptScalarTexture(int value) {
            }

            void acceptVectorTexture(ivec2 value) {
            }

            void acceptScalarArray(int values[2]) {
            }

            void acceptVectorArray(ivec2 values[2]) {
            }

            void main() {
                bool take = true;
                int scalarFromTernary =
                    take ? textureSize(colorMap, 0) : imageSize(colorImage);
                ivec2 vectorFromTernary =
                    take ? textureSize(line, 0) : imageSize(values);
                QueryShapeResult constructed = QueryShapeResult(
                    take ? textureSize(colorMap, 0) : imageSize(colorImage),
                    take ? textureSize(line, 0) : imageSize(values),
                    take ? imageSize(colorImage) : textureSize(colorMap, 0),
                    take ? imageSize(values) : textureSize(line, 0)
                );
                int scalarArray[2] = {
                    textureSize(colorMap, 0),
                    imageSize(colorImage)
                };
                ivec2 vectorArray[2] = {
                    textureSize(line, 0),
                    imageSize(values)
                };
                acceptScalarTexture(
                    take ? textureSize(colorMap, 0) : imageSize(colorImage)
                );
                acceptVectorTexture(
                    take ? textureSize(line, 0) : imageSize(values)
                );
                acceptScalarArray({
                    textureSize(colorMap, 0),
                    imageSize(colorImage)
                });
                acceptVectorArray({
                    textureSize(line, 0),
                    imageSize(values)
                });
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "return (take ? /* unsupported Slang resource query: textureSize returns "
        "int2 but target expects int */ 0 : /* unsupported Slang resource query: "
        "imageSize returns int2 but target expects int */ 0);" in generated_code
    )
    assert (
        "return (take ? /* unsupported Slang resource query: textureSize returns "
        "int but target expects int2 */ int2(0) : /* unsupported Slang resource "
        "query: imageSize returns int but target expects int2 */ int2(0));"
        in generated_code
    )
    assert (
        "int scalarFromTernary = (take ? /* unsupported Slang resource query: "
        "textureSize returns int2 but target expects int */ 0 : /* unsupported "
        "Slang resource query: imageSize returns int2 but target expects int */ 0);"
        in generated_code
    )
    assert (
        "int2 vectorFromTernary = (take ? /* unsupported Slang resource query: "
        "textureSize returns int but target expects int2 */ int2(0) : "
        "/* unsupported Slang resource query: imageSize returns int but target "
        "expects int2 */ int2(0));" in generated_code
    )
    assert (
        "QueryShapeResult constructed = QueryShapeResult((take ? "
        "/* unsupported Slang resource query: textureSize returns int2 but target "
        "expects int */ 0 : /* unsupported Slang resource query: imageSize returns "
        "int2 but target expects int */ 0), (take ? /* unsupported Slang resource "
        "query: textureSize returns int but target expects int2 */ int2(0) : "
        "/* unsupported Slang resource query: imageSize returns int but target "
        "expects int2 */ int2(0)), (take ? /* unsupported Slang resource query: "
        "imageSize returns int2 but target expects int */ 0 : /* unsupported Slang "
        "resource query: textureSize returns int2 but target expects int */ 0), "
        "(take ? /* unsupported Slang resource query: imageSize returns int but "
        "target expects int2 */ int2(0) : /* unsupported Slang resource query: "
        "textureSize returns int but target expects int2 */ int2(0)));"
        in generated_code
    )
    assert (
        "int scalarArray[2] = {/* unsupported Slang resource query: textureSize "
        "returns int2 but target expects int */ 0, /* unsupported Slang resource "
        "query: imageSize returns int2 but target expects int */ 0};" in generated_code
    )
    assert (
        "int2 vectorArray[2] = {/* unsupported Slang resource query: textureSize "
        "returns int but target expects int2 */ int2(0), /* unsupported Slang "
        "resource query: imageSize returns int but target expects int2 */ int2(0)};"
        in generated_code
    )
    assert (
        "acceptScalarTexture((take ? /* unsupported Slang resource query: "
        "textureSize returns int2 but target expects int */ 0 : /* unsupported "
        "Slang resource query: imageSize returns int2 but target expects int */ 0));"
        in generated_code
    )
    assert (
        "acceptVectorTexture((take ? /* unsupported Slang resource query: "
        "textureSize returns int but target expects int2 */ int2(0) : "
        "/* unsupported Slang resource query: imageSize returns int but target "
        "expects int2 */ int2(0)));" in generated_code
    )
    assert (
        "acceptScalarArray({/* unsupported Slang resource query: textureSize "
        "returns int2 but target expects int */ 0, /* unsupported Slang resource "
        "query: imageSize returns int2 but target expects int */ 0});" in generated_code
    )
    assert (
        "acceptVectorArray({/* unsupported Slang resource query: textureSize "
        "returns int but target expects int2 */ int2(0), /* unsupported Slang "
        "resource query: imageSize returns int but target expects int2 */ int2(0)});"
        in generated_code
    )
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code


def test_resource_query_expected_types_flow_through_nested_array_return_helpers():
    code = """
    shader ResourceQueryNestedArrayContexts {
        sampler1d line;
        sampler2d colorMap;
        image1D values;
        image2D colorImage;

        compute {
            int[2] makeScalarRow() {
                return {
                    textureSize(colorMap, 0),
                    imageSize(colorImage)
                };
            }

            ivec2[2] makeVectorRow() {
                return {
                    textureSize(line, 0),
                    imageSize(values)
                };
            }

            void acceptScalarRows(int rows[2][2]) {
            }

            void acceptVectorRows(ivec2 rows[2][2]) {
            }

            void main() {
                int scalarRows[2][2] = {
                    makeScalarRow(),
                    {
                        textureSize(colorMap, 0),
                        imageSize(colorImage)
                    }
                };
                ivec2 vectorRows[2][2] = {
                    makeVectorRow(),
                    {
                        textureSize(line, 0),
                        imageSize(values)
                    }
                };
                acceptScalarRows({
                    makeScalarRow(),
                    {
                        textureSize(colorMap, 0),
                        imageSize(colorImage)
                    }
                });
                acceptVectorRows({
                    makeVectorRow(),
                    {
                        textureSize(line, 0),
                        imageSize(values)
                    }
                });
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "int[2] makeScalarRow()" in generated_code
    assert "int2[2] makeVectorRow()" in generated_code
    assert "ivec2[2] makeVectorRow()" not in generated_code
    assert (
        "return {/* unsupported Slang resource query: textureSize returns int2 "
        "but target expects int */ 0, /* unsupported Slang resource query: "
        "imageSize returns int2 but target expects int */ 0};" in generated_code
    )
    assert (
        "return {/* unsupported Slang resource query: textureSize returns int "
        "but target expects int2 */ int2(0), /* unsupported Slang resource query: "
        "imageSize returns int but target expects int2 */ int2(0)};" in generated_code
    )
    assert (
        "int scalarRows[2][2] = {makeScalarRow(), "
        "{/* unsupported Slang resource query: textureSize returns int2 but target "
        "expects int */ 0, /* unsupported Slang resource query: imageSize returns "
        "int2 but target expects int */ 0}};" in generated_code
    )
    assert (
        "int2 vectorRows[2][2] = {makeVectorRow(), "
        "{/* unsupported Slang resource query: textureSize returns int but target "
        "expects int2 */ int2(0), /* unsupported Slang resource query: imageSize "
        "returns int but target expects int2 */ int2(0)}};" in generated_code
    )
    assert (
        "acceptScalarRows({makeScalarRow(), {/* unsupported Slang resource query: "
        "textureSize returns int2 but target expects int */ 0, /* unsupported "
        "Slang resource query: imageSize returns int2 but target expects int */ 0}});"
        in generated_code
    )
    assert (
        "acceptVectorRows({makeVectorRow(), {/* unsupported Slang resource query: "
        "textureSize returns int but target expects int2 */ int2(0), /* unsupported "
        "Slang resource query: imageSize returns int but target expects int2 */ "
        "int2(0)}});" in generated_code
    )
    assert generated_code.count("unsupported Slang resource query") == 12
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code


def test_resource_query_struct_array_rows_preserve_remaining_element_types():
    code = """
    struct QueryRows {
        int scalarRows[2][2];
        ivec2 vectorRows[2][2];
    };

    shader ResourceQueryStructArrayContexts {
        sampler1d line;
        sampler2d colorMap;
        image1D values;
        image2D colorImage;

        compute {
            int[2] passScalarRow(int row[2]) {
                return row;
            }

            ivec2[2] passVectorRow(ivec2 row[2]) {
                return row;
            }

            QueryRows makeRows() {
                return QueryRows(
                    {
                        passScalarRow({
                            textureSize(colorMap, 0),
                            imageSize(colorImage)
                        }),
                        {
                            textureSize(colorMap, 0),
                            imageSize(colorImage)
                        }
                    },
                    {
                        passVectorRow({
                            textureSize(line, 0),
                            imageSize(values)
                        }),
                        {
                            textureSize(line, 0),
                            imageSize(values)
                        }
                    }
                );
            }

            void main() {
                QueryRows rows = makeRows();
                rows.scalarRows[0] = {
                    textureSize(colorMap, 0),
                    imageSize(colorImage)
                };
                rows.vectorRows[0] = {
                    textureSize(line, 0),
                    imageSize(values)
                };
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    codegen = SlangCodeGen()
    zero = LiteralNode(0, PrimitiveType("int"))
    codegen.variable_types["nestedRows"] = "int[2][3][4]"
    row = ArrayAccessNode(IdentifierNode("nestedRows"), zero)
    plane = ArrayAccessNode(row, zero)
    assert codegen.expression_result_type(row) == "int[3][4]"
    assert codegen.expression_result_type(plane) == "int[4]"
    assert codegen.expression_result_type(ArrayAccessNode(plane, zero)) == "int"

    codegen.variable_types["rows"] = "QueryRows"
    codegen.struct_member_types["QueryRows"] = {"vectorRows": "ivec2[2][3]"}
    member_row = ArrayAccessNode(
        MemberAccessNode(IdentifierNode("rows"), "vectorRows"),
        zero,
    )
    assert codegen.expression_result_type(member_row) == "ivec2[3]"

    assert "int2 vectorRows[2][2];" in generated_code
    assert "int2[2] passVectorRow(int2 row[2])" in generated_code
    assert (
        "return QueryRows({passScalarRow({/* unsupported Slang resource query: "
        "textureSize returns int2 but target expects int */ 0, /* unsupported "
        "Slang resource query: imageSize returns int2 but target expects int */ "
        "0}), {/* unsupported Slang resource query: textureSize returns int2 "
        "but target expects int */ 0, /* unsupported Slang resource query: "
        "imageSize returns int2 but target expects int */ 0}}, "
        "{passVectorRow({/* unsupported Slang resource query: textureSize "
        "returns int but target expects int2 */ int2(0), /* unsupported Slang "
        "resource query: imageSize returns int but target expects int2 */ "
        "int2(0)}), {/* unsupported Slang resource query: textureSize returns "
        "int but target expects int2 */ int2(0), /* unsupported Slang resource "
        "query: imageSize returns int but target expects int2 */ int2(0)}});"
        in generated_code
    )
    assert (
        "rows.scalarRows[0] = {/* unsupported Slang resource query: textureSize "
        "returns int2 but target expects int */ 0, /* unsupported Slang "
        "resource query: imageSize returns int2 but target expects int */ 0};"
        in generated_code
    )
    assert (
        "rows.vectorRows[0] = {/* unsupported Slang resource query: textureSize "
        "returns int but target expects int2 */ int2(0), /* unsupported Slang "
        "resource query: imageSize returns int but target expects int2 */ "
        "int2(0)};" in generated_code
    )
    assert generated_code.count("unsupported Slang resource query") == 12
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code


def test_resource_query_expected_types_flow_through_tail_expression_returns():
    code = """
    shader ResourceQueryTailExpressionContexts {
        sampler1d line;
        sampler2d colorMap;
        image1D values;
        image2D colorImage;

        compute {
            int[2] chooseScalarRow(bool take, int mode) {
                if (take) {
                    ({
                        textureSize(colorMap, 0),
                        imageSize(colorImage)
                    })
                }
                match mode {
                    0 => ({
                        textureSize(colorMap, 0),
                        imageSize(colorImage)
                    }),
                    _ => ({
                        textureSize(colorMap, 0),
                        imageSize(colorImage)
                    })
                }
            }

            ivec2[2] chooseVectorRow(bool take, int mode) {
                if (take) {
                    ({
                        textureSize(line, 0),
                        imageSize(values)
                    })
                }
                match mode {
                    0 => ({
                        textureSize(line, 0),
                        imageSize(values)
                    }),
                    _ => ({
                        textureSize(line, 0),
                        imageSize(values)
                    })
                }
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "int[2] chooseScalarRow(bool take, int mode)" in generated_code
    assert "int2[2] chooseVectorRow(bool take, int mode)" in generated_code
    assert "ivec2[2] chooseVectorRow" not in generated_code
    assert (
        "if (take)\n"
        "    {\n"
        "        return {/* unsupported Slang resource query: textureSize returns "
        "int2 but target expects int */ 0, /* unsupported Slang resource query: "
        "imageSize returns int2 but target expects int */ 0};\n"
        "    }" in generated_code
    )
    assert (
        "case 0:\n"
        "            return {/* unsupported Slang resource query: textureSize "
        "returns int2 but target expects int */ 0, /* unsupported Slang resource "
        "query: imageSize returns int2 but target expects int */ 0};" in generated_code
    )
    assert (
        "default:\n"
        "            return {/* unsupported Slang resource query: textureSize "
        "returns int2 but target expects int */ 0, /* unsupported Slang resource "
        "query: imageSize returns int2 but target expects int */ 0};" in generated_code
    )
    assert (
        "return {/* unsupported Slang resource query: textureSize returns int "
        "but target expects int2 */ int2(0), /* unsupported Slang resource query: "
        "imageSize returns int but target expects int2 */ int2(0)};" in generated_code
    )
    assert generated_code.count("unsupported Slang resource query") == 12
    assert "break;" not in generated_code
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code


def test_resource_query_expected_types_flow_through_match_expressions():
    code = """
    shader ResourceQueryMatchExpressionContexts {
        sampler1d line;
        sampler2d colorMap;
        image1D values;
        image2D colorImage;

        compute {
            int[2] chooseScalarRow(int mode) {
                return match mode {
                    0 => ({
                        textureSize(colorMap, 0),
                        imageSize(colorImage)
                    }),
                    _ => ({
                        textureSize(colorMap, 0),
                        imageSize(colorImage)
                    })
                };
            }

            ivec2[2] chooseVectorRow(int mode) {
                return match mode {
                    0 => ({
                        textureSize(line, 0),
                        imageSize(values)
                    }),
                    _ => ({
                        textureSize(line, 0),
                        imageSize(values)
                    })
                };
            }

            void acceptScalarRow(int row[2]) {
            }

            void acceptVectorRow(ivec2 row[2]) {
            }

            void main() {
                int mode = 0;
                int scalarRow[2] = match mode {
                    0 => ({
                        textureSize(colorMap, 0),
                        imageSize(colorImage)
                    }),
                    _ => ({
                        textureSize(colorMap, 0),
                        imageSize(colorImage)
                    })
                };
                ivec2 vectorRow[2];
                vectorRow = match mode {
                    0 => ({
                        textureSize(line, 0),
                        imageSize(values)
                    }),
                    _ => ({
                        textureSize(line, 0),
                        imageSize(values)
                    })
                };
                acceptScalarRow(match mode {
                    0 => ({
                        textureSize(colorMap, 0),
                        imageSize(colorImage)
                    }),
                    _ => ({
                        textureSize(colorMap, 0),
                        imageSize(colorImage)
                    })
                });
                acceptVectorRow(match mode {
                    0 => ({
                        textureSize(line, 0),
                        imageSize(values)
                    }),
                    _ => ({
                        textureSize(line, 0),
                        imageSize(values)
                    })
                });
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "MatchNode(" not in generated_code
    assert "int2[2] chooseVectorRow(int mode)" in generated_code
    assert "ivec2[2] chooseVectorRow" not in generated_code
    assert generated_code.count("switch (mode)") == 6
    assert "int cgl_match_value[2];" in generated_code
    assert "int2 cgl_match_value[2];" in generated_code
    assert "return cgl_match_value;" in generated_code
    assert "return {/* unsupported Slang resource query" not in generated_code
    assert (
        "int scalarRow[2] = cgl_match_value;" in generated_code
        or "int scalarRow[2] = cgl_match_value_1;" in generated_code
    )
    assert (
        "vectorRow = cgl_match_value;" in generated_code
        or "vectorRow = cgl_match_value_1;" in generated_code
    )
    assert "acceptScalarRow(cgl_match_value" in generated_code
    assert "acceptVectorRow(cgl_match_value" in generated_code
    assert (
        "cgl_match_value = {/* unsupported Slang resource query: textureSize "
        "returns int2 but target expects int */ 0, /* unsupported Slang resource "
        "query: imageSize returns int2 but target expects int */ 0};" in generated_code
    )
    assert (
        "cgl_match_value = {/* unsupported Slang resource query: textureSize "
        "returns int but target expects int2 */ int2(0), /* unsupported Slang "
        "resource query: imageSize returns int but target expects int2 */ int2(0)};"
        in generated_code
    )
    assert generated_code.count("unsupported Slang resource query") == 24
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code


def test_match_expression_defaults_assign_typed_values_for_missing_value_paths():
    code = """
    shader SlangMatchExpressionDefaults {
        compute {
            int chooseScalar(int mode) {
                return match mode {
                    0 => 7,
                };
            }

            ivec2[2] chooseVectorRow(int mode) {
                return match mode {
                    0 => ({
                        ivec2(1, 2),
                        ivec2(3, 4)
                    }),
                    1 => {
                        int scratch = mode;
                    },
                };
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "MatchNode(" not in generated_code
    assert "int chooseScalar(int mode)" in generated_code
    assert "int2[2] chooseVectorRow(int mode)" in generated_code
    assert generated_code.count("default:") == 2
    assert (
        "cgl_match_value = /* unsupported Slang match expression: no wildcard "
        "arm handles remaining cases */ 0;" in generated_code
    )
    assert (
        "cgl_match_value = /* unsupported Slang match expression: arm does not "
        "produce a value */ {int2(0), int2(0)};" in generated_code
    )
    assert (
        "cgl_match_value = /* unsupported Slang match expression: no wildcard "
        "arm handles remaining cases */ {int2(0), int2(0)};" in generated_code
    )
    assert "cgl_match_value = scratch;" not in generated_code
    assert generated_code.count("unsupported Slang match expression") == 3


def test_match_expression_rejects_guarded_and_range_patterns_with_typed_values():
    code = """
    shader SlangMatchExpressionPatternDiagnostics {
        compute {
            int chooseGuarded(int mode) {
                return match mode {
                    0 if mode > 0 => 7,
                    _ => 1,
                };
            }

            ivec2[2] chooseRangeStyle(int mode) {
                return match mode {
                    .. => ({
                        ivec2(1, 2),
                        ivec2(3, 4)
                    }),
                };
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "MatchNode(" not in generated_code
    assert "if mode > 0" not in generated_code
    assert "case .." not in generated_code
    assert (
        "return /* unsupported Slang match expression: guarded arms cannot be "
        "lowered to switch */ 0;" in generated_code
    )
    assert (
        "return /* unsupported Slang match expression: range-style patterns "
        "cannot be lowered to switch */ {int2(0), int2(0)};" in generated_code
    )


def test_match_expression_reports_wildcard_and_destructuring_pattern_diagnostics():
    code = """
    shader SlangMatchExpressionPatternDetailDiagnostics {
        compute {
            int chooseDuplicateWildcard(int mode) {
                return match mode {
                    _ => 1,
                    _ => 2,
                };
            }

            int chooseNonFinalWildcard(int mode) {
                return match mode {
                    _ => 1,
                    0 => 2,
                };
            }

            int chooseBinding(int mode) {
                return match mode {
                    candidate => candidate,
                    _ => 0,
                };
            }

            int chooseConstructor(int mode) {
                return match mode {
                    Option::Some(value) => value,
                    _ => 0,
                };
            }

            int chooseStructDestructure(int mode) {
                return match mode {
                    Pair { left, right } => left + right,
                    _ => 0,
                };
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "MatchNode(" not in generated_code
    assert "case candidate" not in generated_code
    assert "case Option::Some" not in generated_code
    assert "case Pair" not in generated_code
    assert "only unguarded literal patterns" not in generated_code
    assert (
        "return /* unsupported Slang match expression: multiple wildcard arms "
        "cannot be lowered to switch */ 0;" in generated_code
    )
    assert (
        "return /* unsupported Slang match expression: wildcard arm must be "
        "final */ 0;" in generated_code
    )
    assert (
        "return /* unsupported Slang match expression: identifier binding "
        "patterns cannot be lowered to switch */ 0;" in generated_code
    )
    assert (
        "return /* unsupported Slang match expression: constructor patterns "
        "cannot be lowered to switch */ 0;" in generated_code
    )
    assert (
        "return /* unsupported Slang match expression: struct destructuring "
        "patterns cannot be lowered to switch */ 0;" in generated_code
    )


def test_slangc_smoke_compiles_generated_resource_query_helpers_if_available(
    tmp_path,
):
    code = """
    shader SlangResourceQueryCompileSmoke {
        sampler2D colorMap @register(t0);
        sampler3D volumeMap @register(t1);
        sampler2DMS msColor @register(t2);
        image2D colorImage @rgba32f @register(u0);
        image2DMS msImage @rgba32f @register(u1);
        uimage2DMS counters @r32ui @register(u2);

        compute {
            @numthreads(1, 1, 1)
            void main() {
                ivec2 texSize = textureSize(colorMap, 0);
                ivec3 volumeSize = textureSize(volumeMap, 0);
                ivec2 msTexSize = textureSize(msColor, 0);
                int texLevels = textureQueryLevels(colorMap);
                int volumeLevels = textureQueryLevels(volumeMap);
                ivec2 imageSizeValue = imageSize(colorImage);
                ivec2 msImageSizeValue = imageSize(msImage);
                int msImageSamples = imageSamples(msImage);
                int counterSamples = imageSamples(counters);
                imageStore(
                    colorImage,
                    ivec2(0, 0),
                    vec4(
                        float(
                            texSize.x
                            + volumeSize.x
                            + msTexSize.x
                            + imageSizeValue.x
                        ),
                        float(msImageSizeValue.x),
                        float(
                            texLevels
                            + volumeLevels
                            + msImageSamples
                            + counterSamples
                        ),
                        1.0
                    )
                );
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "int2 cgl_textureSize_sampler2D(Sampler2D<float4> tex, uint mipLevel)"
        in generated_code
    )
    assert (
        "int3 cgl_textureSize_sampler3D(Sampler3D<float4> tex, uint mipLevel)"
        in generated_code
    )
    assert "int2 cgl_textureSize_sampler2DMS(Sampler2DMS<float4> tex)" in (
        generated_code
    )
    assert (
        "int cgl_textureQueryLevels_sampler2D(Sampler2D<float4> tex)" in generated_code
    )
    assert (
        "int cgl_textureQueryLevels_sampler3D(Sampler3D<float4> tex)" in generated_code
    )
    assert "int2 cgl_imageSize_image2D(RWTexture2D<float4> tex)" in generated_code
    assert "int2 cgl_imageSize_image2DMS(RWTexture2DMS<float4> tex)" in (generated_code)
    assert "int cgl_imageSamples_image2DMS(RWTexture2DMS<float4> tex)" in (
        generated_code
    )
    assert "int cgl_imageSamples_uimage2DMS(RWTexture2DMS<uint> tex)" in (
        generated_code
    )
    assert "int2 texSize = cgl_textureSize_sampler2D(colorMap, 0);" in generated_code
    assert (
        "int3 volumeSize = cgl_textureSize_sampler3D(volumeMap, 0);" in generated_code
    )
    assert "int2 msTexSize = cgl_textureSize_sampler2DMS(msColor);" in (generated_code)
    assert (
        "int texLevels = cgl_textureQueryLevels_sampler2D(colorMap);" in generated_code
    )
    assert (
        "int volumeLevels = cgl_textureQueryLevels_sampler3D(volumeMap);"
        in generated_code
    )
    assert "int2 imageSizeValue = cgl_imageSize_image2D(colorImage);" in generated_code
    assert "int2 msImageSizeValue = cgl_imageSize_image2DMS(msImage);" in generated_code
    assert "int msImageSamples = cgl_imageSamples_image2DMS(msImage);" in generated_code
    assert (
        "int counterSamples = cgl_imageSamples_uimage2DMS(counters);" in generated_code
    )
    assert "textureSize(" not in generated_code
    assert "imageSize(" not in generated_code
    assert "imageSamples(" not in generated_code
    assert "textureQueryLevels(" not in generated_code

    compile_generated_slang(generated_code, tmp_path, "compute")


def test_storage_image_load_store_emit_slang_subscript_access():
    code = """
    shader Resources {
        image2D colorImage;
        image2DMS msColor;
        image2DMSArray msLayers;
        uimage2DMS counters;
        iimage2DMSArray signedLayers;

        compute {
            void main() {
                int sampleIndex = 2;
                ivec2 pixel = ivec2(4, 8);
                ivec3 pixelLayer = ivec3(4, 8, 1);
                vec4 regular = imageLoad(colorImage, pixel);
                imageStore(colorImage, pixel, regular);
                vec4 color = imageLoad(msColor, pixel, sampleIndex);
                imageStore(msColor, pixel, sampleIndex, color);
                vec4 layer = imageLoad(msLayers, pixelLayer, sampleIndex);
                imageStore(msLayers, pixelLayer, sampleIndex, layer);
                uint count = imageLoad(counters, pixel, sampleIndex);
                imageStore(counters, pixel, sampleIndex, count);
                int signedValue = imageLoad(signedLayers, pixelLayer, sampleIndex);
                imageStore(signedLayers, pixelLayer, sampleIndex, signedValue);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float4 regular = colorImage[pixel];" in generated_code
    assert "colorImage[pixel] = regular;" in generated_code
    assert "float4 color = msColor[pixel, sampleIndex];" in generated_code
    assert "msColor[pixel, sampleIndex] = color;" in generated_code
    assert "float4 layer = msLayers[pixelLayer, sampleIndex];" in generated_code
    assert "msLayers[pixelLayer, sampleIndex] = layer;" in generated_code
    assert "uint count = counters[pixel, sampleIndex];" in generated_code
    assert "counters[pixel, sampleIndex] = count;" in generated_code
    assert "int signedValue = signedLayers[pixelLayer, sampleIndex];" in generated_code
    assert "signedLayers[pixelLayer, sampleIndex] = signedValue;" in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_storage_image_access_qualifiers_emit_slang_diagnostics():
    code = """
    shader ImageAccessQualifiers {
        readonly image2D source @rgba32f;
        writeonly image2D target @rgba32f;
        readwrite uimage2D counters @r32ui;
        readonly uimage2D readOnlyCounters @r32ui;
        writeonly uimage2D writeOnlyCounters @r32ui;

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 0);
                vec4 color = imageLoad(source, pixel);
                imageStore(source, pixel, color);
                vec4 blocked = imageLoad(target, pixel);
                imageStore(target, pixel, color);
                uint count = imageAtomicAdd(counters, pixel, 1u);
                uint readOnlyAtomic = imageAtomicAdd(readOnlyCounters, pixel, 1u);
                uint writeOnlyAtomic = imageAtomicAdd(writeOnlyCounters, pixel, 1u);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float4 color = source[pixel];" in generated_code
    assert "imageStore requires writable image resource */;" in generated_code
    assert (
        "float4 blocked = /* unsupported Slang image access: imageLoad "
        "requires readable image resource */ float4(0.0);" in generated_code
    )
    assert "target[pixel] = color;" in generated_code
    assert "uint cgl_imageAtomicAdd_original;" in generated_code
    assert (
        "InterlockedAdd(counters[pixel], 1u, cgl_imageAtomicAdd_original);"
        in generated_code
    )
    assert "uint count = cgl_imageAtomicAdd_original;" in generated_code
    assert (
        "uint readOnlyAtomic = /* unsupported Slang image atomic: imageAtomicAdd "
        "requires readwrite image resource */ 0u;" in generated_code
    )
    assert (
        "uint writeOnlyAtomic = /* unsupported Slang image atomic: imageAtomicAdd "
        "requires readwrite image resource */ 0u;" in generated_code
    )
    assert "source[pixel] = color;" not in generated_code
    assert "float4 blocked = target[pixel];" not in generated_code
    assert "cgl_imageAtomicAdd_uimage2D(readOnlyCounters" not in generated_code
    assert "cgl_imageAtomicAdd_uimage2D(writeOnlyCounters" not in generated_code
    assert "uint cgl_imageAtomicAdd_uimage2D(" not in generated_code


def test_explicit_image_formats_emit_slang_storage_element_types():
    code = """
    shader ExplicitImageFormats {
        image2D scalarFloat @r32f;
        image2D rgFloat @rg32f;
        uimage2D rgUnsigned @format(rg32ui);
        image3D rgbaVolume @rgba16f;
        uimage2DMS msCounter @r32ui;
        image2DMSArray msLayers @rgba8;

        float scalarLoad(image2D image @rg32f, ivec2 pixel, float value) {
            float oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        vec2 vectorLoad(image2D image @rg32f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        uint unsignedLoad(uimage2D image @rg32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(image, pixel);
            imageStore(image, pixel, oldValue + value);
            return oldValue;
        }

        vec4 rgbaLoad(image3D image @format(rgba16f), ivec3 voxel, vec4 value) {
            vec4 oldValue = imageLoad(image, voxel);
            imageStore(image, voxel, oldValue + value);
            return oldValue;
        }

        uint sampleLoad(uimage2DMS image @r32ui, ivec2 pixel, int sampleIndex, uint value) {
            uint oldValue = imageLoad(image, pixel, sampleIndex);
            imageStore(image, pixel, sampleIndex, oldValue + value);
            return oldValue;
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "RWTexture2D<float> scalarFloat : register(u0);" in generated_code
    assert "RWTexture2D<float2> rgFloat : register(u1);" in generated_code
    assert "RWTexture2D<uint2> rgUnsigned : register(u2);" in generated_code
    assert "RWTexture3D<float4> rgbaVolume : register(u3);" in generated_code
    assert "RWTexture2DMS<uint> msCounter : register(u4);" in generated_code
    assert "RWTexture2DMSArray<float4> msLayers : register(u5);" in generated_code
    assert (
        "float scalarLoad(RWTexture2D<float2> image, int2 pixel, float value)"
        in generated_code
    )
    assert (
        "float2 vectorLoad(RWTexture2D<float2> image, int2 pixel, float2 value)"
        in generated_code
    )
    assert (
        "uint unsignedLoad(RWTexture2D<uint2> image, int2 pixel, uint value)"
        in generated_code
    )
    assert (
        "float4 rgbaLoad(RWTexture3D<float4> image, int3 voxel, float4 value)"
        in generated_code
    )
    assert (
        "uint sampleLoad(RWTexture2DMS<uint> image, int2 pixel, int sampleIndex, uint value)"
        in generated_code
    )
    assert "float oldValue = image[pixel].x;" in generated_code
    assert "image[pixel] = float2(oldValue + value, 0.0);" in generated_code
    assert "float2 oldValue = image[pixel];" in generated_code
    assert "image[pixel] = oldValue + value;" in generated_code
    assert "uint oldValue = image[pixel].x;" in generated_code
    assert "image[pixel] = uint2(oldValue + value, 0u);" in generated_code
    assert "float4 oldValue = image[voxel];" in generated_code
    assert "image[voxel] = oldValue + value;" in generated_code
    assert "uint oldValue = image[pixel, sampleIndex];" in generated_code
    assert "image[pixel, sampleIndex] = oldValue + value;" in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_direct_atomic_op_nodes_emit_slang_image_atomic_helpers():
    codegen = SlangCodeGen()
    codegen.image_resource_types["counters"] = "RWTexture2D<uint>"
    codegen.image_resource_types["signedCounters"] = "RWTexture2D<int>"

    pixel = IdentifierNode("pixel")
    add_node = AtomicOpNode(
        "atomicAdd",
        IdentifierNode("counters"),
        [pixel, LiteralNode(1, PrimitiveType("uint"))],
    )
    compare_node = AtomicOpNode(
        "atomicCompareExchange",
        IdentifierNode("counters"),
        [
            pixel,
            LiteralNode(2, PrimitiveType("uint")),
            LiteralNode(3, PrimitiveType("uint")),
        ],
    )
    signed_node = AtomicOpNode(
        "Add",
        IdentifierNode("signedCounters"),
        [pixel, LiteralNode(-1, PrimitiveType("int"))],
    )

    add_call = codegen.generate_expression(add_node)
    compare_call = codegen.generate_expression(compare_node)
    signed_call = codegen.generate_expression(signed_node)
    helper_code = codegen.emit_helper_functions()

    assert add_call == "cgl_imageAtomicAdd_uimage2D(counters, pixel, 1u)"
    assert compare_call == "cgl_imageAtomicCompSwap_uimage2D(counters, pixel, 2u, 3u)"
    assert signed_call == "cgl_imageAtomicAdd_iimage2D(signedCounters, pixel, -1)"
    assert codegen.expression_result_type(add_node) == "uint"
    assert codegen.expression_result_type(signed_node) == "int"
    assert (
        "uint cgl_imageAtomicAdd_uimage2D(RWTexture2D<uint> image, "
        "int2 coord, uint value)" in helper_code
    )
    assert (
        "uint cgl_imageAtomicCompSwap_uimage2D(RWTexture2D<uint> image, "
        "int2 coord, uint compareValue, uint value)" in helper_code
    )
    assert (
        "int cgl_imageAtomicAdd_iimage2D(RWTexture2D<int> image, "
        "int2 coord, int value)" in helper_code
    )
    assert "AtomicOpNode(" not in add_call
    assert "atomicCompareExchange" not in compare_call


def test_integer_image_atomics_emit_slang_interlocked_helpers():
    code = """
    shader AtomicImages {
        image2D unsignedCounters @r32ui;
        image2D signedCounters @r32i;
        image3D unsignedVolume @format(r32ui);
        image2DArray signedLayers @format(r32i);

        uint unsignedOps(image2D image @r32ui, ivec2 pixel, uint value) {
            uint added = imageAtomicAdd(image, pixel, value);
            uint minValue = imageAtomicMin(image, pixel, value);
            uint maxValue = imageAtomicMax(image, pixel, value);
            uint andValue = imageAtomicAnd(image, pixel, value);
            uint orValue = imageAtomicOr(image, pixel, value);
            uint xorValue = imageAtomicXor(image, pixel, value);
            uint exchanged = imageAtomicExchange(image, pixel, value);
            return added + minValue + maxValue + andValue + orValue + xorValue + exchanged;
        }

        int signedOps(image2D image @r32i, ivec2 pixel, int value) {
            int minValue = imageAtomicMin(image, pixel, value);
            int exchanged = imageAtomicExchange(image, pixel, value);
            return minValue + exchanged;
        }

        uint swapVolume(image3D image @r32ui, ivec3 voxel, uint expected, uint value) {
            return imageAtomicCompSwap(image, voxel, expected, value);
        }

        int exchangeLayer(image2DArray image @r32i, ivec3 pixelLayer, int value) {
            return imageAtomicExchange(image, pixelLayer, value);
        }

        compute {
            void main() {
                uint a = unsignedOps(unsignedCounters, ivec2(0, 0), 1u);
                int b = signedOps(signedCounters, ivec2(1, 0), -1);
                uint c = swapVolume(unsignedVolume, ivec3(0, 1, 2), 3u, 4u);
                int d = exchangeLayer(signedLayers, ivec3(2, 3, 4), 5);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "RWTexture2D<uint> unsignedCounters : register(u0);" in generated_code
    assert "RWTexture2D<int> signedCounters : register(u1);" in generated_code
    assert "RWTexture3D<uint> unsignedVolume : register(u2);" in generated_code
    assert "RWTexture2DArray<int> signedLayers : register(u3);" in generated_code
    assert "uint cgl_imageAtomicAdd_original;" in generated_code
    assert (
        "InterlockedAdd(image[pixel], value, cgl_imageAtomicAdd_original);"
        in generated_code
    )
    assert "uint added = cgl_imageAtomicAdd_original;" in generated_code
    assert "uint cgl_imageAtomicMin_original;" in generated_code
    assert (
        "InterlockedMin(image[pixel], value, cgl_imageAtomicMin_original);"
        in generated_code
    )
    assert "uint minValue = cgl_imageAtomicMin_original;" in generated_code
    assert "uint cgl_imageAtomicMax_original;" in generated_code
    assert (
        "InterlockedMax(image[pixel], value, cgl_imageAtomicMax_original);"
        in generated_code
    )
    assert "uint maxValue = cgl_imageAtomicMax_original;" in generated_code
    assert "uint cgl_imageAtomicAnd_original;" in generated_code
    assert (
        "InterlockedAnd(image[pixel], value, cgl_imageAtomicAnd_original);"
        in generated_code
    )
    assert "uint andValue = cgl_imageAtomicAnd_original;" in generated_code
    assert "uint cgl_imageAtomicOr_original;" in generated_code
    assert (
        "InterlockedOr(image[pixel], value, cgl_imageAtomicOr_original);"
        in generated_code
    )
    assert "uint orValue = cgl_imageAtomicOr_original;" in generated_code
    assert "uint cgl_imageAtomicXor_original;" in generated_code
    assert (
        "InterlockedXor(image[pixel], value, cgl_imageAtomicXor_original);"
        in generated_code
    )
    assert "uint xorValue = cgl_imageAtomicXor_original;" in generated_code
    assert "uint cgl_imageAtomicExchange_original;" in generated_code
    assert (
        "InterlockedExchange(image[pixel], value, cgl_imageAtomicExchange_original);"
        in generated_code
    )
    assert "uint exchanged = cgl_imageAtomicExchange_original;" in generated_code
    assert "int cgl_imageAtomicMin_original;" in generated_code
    assert (
        "InterlockedMin(image[pixel], value, cgl_imageAtomicMin_original);"
        in generated_code
    )
    assert "int minValue = cgl_imageAtomicMin_original;" in generated_code
    assert "int cgl_imageAtomicExchange_original;" in generated_code
    assert (
        "InterlockedExchange(image[pixel], value, cgl_imageAtomicExchange_original);"
        in generated_code
    )
    assert "int exchanged = cgl_imageAtomicExchange_original;" in generated_code
    assert "uint cgl_imageAtomicCompSwap_original;" in generated_code
    assert (
        "InterlockedCompareExchange(image[voxel], expected, value, "
        "cgl_imageAtomicCompSwap_original);" in generated_code
    )
    assert "return cgl_imageAtomicCompSwap_original;" in generated_code
    assert "int cgl_imageAtomicExchange_original;" in generated_code
    assert (
        "InterlockedExchange(image[pixelLayer], value, "
        "cgl_imageAtomicExchange_original);" in generated_code
    )
    assert "return cgl_imageAtomicExchange_original;" in generated_code
    assert "uint cgl_imageAtomicAdd_uimage2D(" not in generated_code
    assert "int cgl_imageAtomicMin_iimage2D(" not in generated_code
    assert "uint cgl_imageAtomicCompSwap_uimage3D(" not in generated_code
    assert "int cgl_imageAtomicExchange_iimage2DArray(" not in generated_code
    for operation in [
        "imageAtomicAdd",
        "imageAtomicMin",
        "imageAtomicMax",
        "imageAtomicAnd",
        "imageAtomicOr",
        "imageAtomicXor",
        "imageAtomicExchange",
        "imageAtomicCompSwap",
    ]:
        assert f"{operation}(image" not in generated_code


def test_slangc_smoke_compiles_generated_image_atomics_if_available(tmp_path):
    code = """
    shader SlangImageAtomicCompileSmoke {
        uimage2D counters @r32ui;
        image2D signedCounters @r32i;

        compute {
            @numthreads(1, 1, 1)
            void main(uvec3 tid @gl_GlobalInvocationID) {
                ivec2 pixel = ivec2(int(tid.x), 0);
                uint oldAdd = imageAtomicAdd(counters, pixel, 1u);
                uint oldExchange = imageAtomicExchange(counters, pixel, oldAdd + 1u);
                uint oldSwap = imageAtomicCompSwap(counters, pixel, oldAdd, oldExchange);
                int oldMin = imageAtomicMin(signedCounters, pixel, -1);
                if (imageAtomicOr(counters, pixel, oldSwap) != 0u) {
                    imageAtomicXor(counters, pixel, 3u);
                }
                imageStore(
                    counters,
                    pixel,
                    oldAdd + oldExchange + oldSwap + uint(oldMin)
                );
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert "RWTexture2D<uint> counters : register(u0);" in generated_code
    assert "RWTexture2D<int> signedCounters : register(u1);" in generated_code
    assert "uint cgl_imageAtomicAdd_original;" in generated_code
    assert (
        "InterlockedAdd(counters[pixel], 1u, cgl_imageAtomicAdd_original);"
        in generated_code
    )
    assert (
        "InterlockedExchange(counters[pixel], oldAdd + 1u, "
        "cgl_imageAtomicExchange_original);" in generated_code
    )
    assert (
        "InterlockedCompareExchange(counters[pixel], oldAdd, oldExchange, "
        "cgl_imageAtomicCompSwap_original);" in generated_code
    )
    assert (
        "InterlockedMin(signedCounters[pixel], -1, cgl_imageAtomicMin_original);"
        in generated_code
    )
    assert (
        "InterlockedOr(counters[pixel], oldSwap, cgl_imageAtomicOr_original);"
        in generated_code
    )
    assert "if (cgl_imageAtomicOr_original != 0u)" in generated_code
    assert (
        "InterlockedXor(counters[pixel], 3u, cgl_imageAtomicXor_original);"
        in generated_code
    )
    assert "imageAtomicAdd(counters" not in generated_code
    assert "imageAtomicExchange(counters" not in generated_code
    assert "imageAtomicCompSwap(counters" not in generated_code
    assert "imageAtomicMin(signedCounters" not in generated_code
    assert "imageAtomicOr(counters" not in generated_code
    assert "imageAtomicXor(counters" not in generated_code

    compile_generated_slang(generated_code, tmp_path, "compute")


def test_integer_image_array_atomics_preserve_expression_indices():
    code = """
    shader AtomicImageArrays {
        image2D counterImages @r32ui[(2 + 1) * 2];
        image2DArray layerImages @r32i[(2 + 1) * 2];

        uint addArray(image2D images[(2 + 1) * 2] @r32ui, ivec2 pixel, uint value) {
            return imageAtomicAdd(images[1 + 2], pixel, value);
        }

        int compareLayer(
            image2DArray images[(2 + 1) * 2] @r32i,
            ivec3 pixelLayer,
            int expected,
            int value
        ) {
            return imageAtomicCompSwap(images[1 + 2], pixelLayer, expected, value);
        }

        compute {
            void main() {
                uint a = addArray(counterImages, ivec2(0, 0), 1u);
                int b = compareLayer(layerImages, ivec3(1, 2, 3), 4, 5);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "RWTexture2D<uint> counterImages[((2 + 1) * 2)] : register(u0);"
        in generated_code
    )
    assert (
        "RWTexture2DArray<int> layerImages[((2 + 1) * 2)] : register(u6);"
        in generated_code
    )
    assert (
        "uint addArray(RWTexture2D<uint> images[((2 + 1) * 2)], "
        "int2 pixel, uint value)" in generated_code
    )
    assert (
        "int compareLayer(RWTexture2DArray<int> images[((2 + 1) * 2)], "
        "int3 pixelLayer, int expected, int value)" in generated_code
    )
    assert "uint cgl_imageAtomicAdd_original;" in generated_code
    assert (
        "InterlockedAdd(images[(1 + 2)][pixel], value, "
        "cgl_imageAtomicAdd_original);" in generated_code
    )
    assert "return cgl_imageAtomicAdd_original;" in generated_code
    assert "int cgl_imageAtomicCompSwap_original;" in generated_code
    assert (
        "InterlockedCompareExchange(images[(1 + 2)][pixelLayer], "
        "expected, value, cgl_imageAtomicCompSwap_original);" in generated_code
    )
    assert "return cgl_imageAtomicCompSwap_original;" in generated_code
    assert "1 + 2]," not in generated_code
    assert "2 + 1 * 2" not in generated_code
    assert "imageAtomicAdd(images" not in generated_code
    assert "imageAtomicCompSwap(images" not in generated_code
    assert "cgl_imageAtomicAdd_uimage2D(" not in generated_code
    assert "cgl_imageAtomicCompSwap_iimage2DArray(" not in generated_code


def test_image_expression_atomics_in_loop_contexts_emit_diagnostics():
    code = """
    shader AtomicImageLoopContexts {
        uimage2D counters @r32ui;

        compute {
            void main() {
                ivec2 pixel = ivec2(0, 0);
                uint oldValue = 0u;
                if (imageAtomicAdd(counters, pixel, 1u) != 0u) {
                    oldValue = 2u;
                }
                for (
                    uint initOld = imageAtomicAdd(counters, pixel, 1u);
                    initOld < 2u;
                    initOld = initOld + 1u
                ) {
                    imageAtomicAdd(counters, pixel, 1u);
                }
                for (
                    uint i = 0u;
                    imageAtomicAdd(counters, pixel, 1u) < 4u;
                    i = imageAtomicExchange(counters, pixel, i)
                ) {
                    break;
                }
                while (imageAtomicOr(counters, pixel, 1u) != 0u) {
                    break;
                }
                do {
                    break;
                } while (imageAtomicAnd(counters, pixel, 1u) != 0u);
            }
        }
    }
    """

    generated_code = generate_code(parse_code(tokenize_code(code)))

    assert (
        "imageAtomicAdd expression-valued atomic cannot be used in "
        "for-loop initializer context; assign the atomic result before the loop"
        in generated_code
    )
    assert (
        "imageAtomicAdd expression-valued atomic cannot be used in "
        "for-loop condition context; assign the atomic result before the loop"
        in generated_code
    )
    assert (
        "imageAtomicExchange expression-valued atomic cannot be used in "
        "for-loop update context; assign the atomic result before the loop"
        in generated_code
    )
    assert (
        "imageAtomicOr expression-valued atomic cannot be used in "
        "while-loop condition context; assign the atomic result before the loop"
        in generated_code
    )
    assert (
        "imageAtomicAnd expression-valued atomic cannot be used in "
        "do-while-loop condition context; assign the atomic result before the loop"
        in generated_code
    )
    assert "uint cgl_imageAtomicAdd_original;" in generated_code
    assert (
        "InterlockedAdd(counters[pixel], 1u, cgl_imageAtomicAdd_original);"
        in generated_code
    )
    assert "if (cgl_imageAtomicAdd_original != 0u)" in generated_code
    assert "uint cgl_imageAtomicAdd_original_1;" in generated_code
    assert (
        "InterlockedAdd(counters[pixel], 1u, cgl_imageAtomicAdd_original_1);"
        in generated_code
    )
    assert "cgl_imageAtomicAdd_original_1;" not in [
        line.strip() for line in generated_code.splitlines()
    ]
    assert "imageAtomicAdd(counters" not in generated_code
    assert "imageAtomicExchange(counters" not in generated_code
    assert "imageAtomicOr(counters" not in generated_code
    assert "imageAtomicAnd(counters" not in generated_code
    assert "cgl_imageAtomicAdd_uimage2D(" not in generated_code


def test_unsupported_image_atomics_emit_slang_diagnostic_stubs():
    code = """
    shader UnsupportedAtomicImages {
        image2DMS msCounters @r32ui;
        image2DMSArray msLayers @r32i;
        image2D vectorCounters @rg32ui;

        uint addMs(image2DMS image @r32ui, ivec2 pixel, int sampleIndex, uint value) {
            return imageAtomicAdd(image, pixel, sampleIndex, value);
        }

        int compareMsLayer(
            image2DMSArray image @r32i,
            ivec3 pixelLayer,
            int sampleIndex,
            int expected,
            int value
        ) {
            return imageAtomicCompSwap(image, pixelLayer, sampleIndex, expected, value);
        }

        uint addVector(image2D image @rg32ui, ivec2 pixel, uint value) {
            return imageAtomicAdd(image, pixel, value);
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "RWTexture2DMS<uint> msCounters : register(u0);" in generated_code
    assert "RWTexture2DMSArray<int> msLayers : register(u1);" in generated_code
    assert "RWTexture2D<uint2> vectorCounters : register(u2);" in generated_code
    assert (
        "return /* unsupported Slang image atomic: imageAtomicAdd "
        "requires scalar int or uint image1D/image1DArray/image2D/image3D/image2DArray "
        "resource */ 0u;" in generated_code
    )
    assert (
        "return /* unsupported Slang image atomic: imageAtomicCompSwap "
        "requires scalar int or uint image1D/image1DArray/image2D/image3D/image2DArray "
        "resource */ 0;" in generated_code
    )
    assert "imageAtomicAdd(image" not in generated_code
    assert "imageAtomicCompSwap(image" not in generated_code
    assert "cgl_imageAtomicAdd_uimage2DMS" not in generated_code
    assert "cgl_imageAtomicCompSwap_iimage2DMSArray" not in generated_code


def test_formatted_image_arrays_preserve_expression_sizes():
    code = """
    shader ExprFormattedImageArrays {
        image2D counters @r32ui[(1 + 1) * 2];
        image2D rgPairs @rg16f[+3];
        image2D afterCounters @r32ui;

        uint touchCounters(image2D images[(1 + 1) * 2] @r32ui, ivec2 pixel, uint value) {
            uint oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        vec2 touchPairs(image2D images[+3] @rg16f, ivec2 pixel, vec2 value) {
            vec2 oldValue = imageLoad(images[2], pixel);
            imageStore(images[1], pixel, oldValue + value);
            return oldValue;
        }

        compute {
            void main() {
                uint a = touchCounters(counters, ivec2(1, 2), 3u);
                vec2 b = touchPairs(rgPairs, ivec2(2, 3), vec2(0.5));
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "RWTexture2D<uint> counters[((1 + 1) * 2)] : register(u0);" in generated_code
    assert "RWTexture2D<float2> rgPairs[+3] : register(u4);" in generated_code
    assert "RWTexture2D<uint> afterCounters : register(u7);" in generated_code
    assert (
        "uint touchCounters(RWTexture2D<uint> images[((1 + 1) * 2)], "
        "int2 pixel, uint value)" in generated_code
    )
    assert (
        "float2 touchPairs(RWTexture2D<float2> images[+3], int2 pixel, float2 value)"
        in generated_code
    )
    assert "uint oldValue = images[2][pixel];" in generated_code
    assert "images[1][pixel] = oldValue + value;" in generated_code
    assert "float2 oldValue = images[2][pixel];" in generated_code
    assert "images[1][pixel] = oldValue + value;" in generated_code
    assert "uint a = touchCounters(counters, int2(1, 2), 3u);" in generated_code
    assert "float2 b = touchPairs(rgPairs, int2(2, 3), float2(0.5));" in generated_code
    assert "1 + 1 * 2" not in generated_code
    assert "imageLoad(" not in generated_code
    assert "imageStore(" not in generated_code


def test_formatted_image_queries_emit_typed_slang_helpers():
    code = """
    shader FormattedImageQueries {
        image2D rgFloat @rg32f;
        image2DMS msRg @format(rg32f);

        compute {
            void main() {
                ivec2 imageSizeValue = imageSize(rgFloat);
                ivec2 msSizeValue = imageSize(msRg);
                int msSamples = imageSamples(msRg);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert (
        "int2 cgl_imageSize_image2D_RWTexture2D_float2(RWTexture2D<float2> tex)"
        in generated_code
    )
    assert (
        "int2 cgl_imageSize_image2DMS_RWTexture2DMS_float2("
        "RWTexture2DMS<float2> tex)" in generated_code
    )
    assert (
        "int cgl_imageSamples_image2DMS_RWTexture2DMS_float2("
        "RWTexture2DMS<float2> tex)" in generated_code
    )
    assert (
        "int2 imageSizeValue = "
        "cgl_imageSize_image2D_RWTexture2D_float2(rgFloat);" in generated_code
    )
    assert (
        "int2 msSizeValue = "
        "cgl_imageSize_image2DMS_RWTexture2DMS_float2(msRg);" in generated_code
    )
    assert (
        "int msSamples = "
        "cgl_imageSamples_image2DMS_RWTexture2DMS_float2(msRg);" in generated_code
    )
    assert "imageSize(" not in generated_code
    assert "imageSamples(" not in generated_code


def test_formatted_image_query_helpers_avoid_user_symbol_collisions():
    code = """
    shader FormattedImageQueryCollisions {
        int cgl_imageSize_image2D_RWTexture2D_float2;
        int cgl_imageSize_image2DMS_RWTexture2DMS_float2;
        int cgl_imageSamples_image2DMS_RWTexture2DMS_float2;
        image2D rgFloat @rg32f;
        image2DMS msRg @format(rg32f);

        compute {
            void main() {
                ivec2 imageSizeValue = imageSize(rgFloat);
                ivec2 msSizeValue = imageSize(msRg);
                int msSamples = imageSamples(msRg);
            }
        }
    }
    """

    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "int cgl_imageSize_image2D_RWTexture2D_float2;" in generated_code
    assert "int cgl_imageSize_image2DMS_RWTexture2DMS_float2;" in generated_code
    assert "int cgl_imageSamples_image2DMS_RWTexture2DMS_float2;" in generated_code
    assert (
        "int2 cgl_imageSize_image2D_RWTexture2D_float2_1("
        "RWTexture2D<float2> tex)" in generated_code
    )
    assert (
        "int2 cgl_imageSize_image2DMS_RWTexture2DMS_float2_1("
        "RWTexture2DMS<float2> tex)" in generated_code
    )
    assert (
        "int cgl_imageSamples_image2DMS_RWTexture2DMS_float2_1("
        "RWTexture2DMS<float2> tex)" in generated_code
    )
    assert (
        "int2 imageSizeValue = "
        "cgl_imageSize_image2D_RWTexture2D_float2_1(rgFloat);" in generated_code
    )
    assert (
        "int2 msSizeValue = "
        "cgl_imageSize_image2DMS_RWTexture2DMS_float2_1(msRg);" in generated_code
    )
    assert (
        "int msSamples = "
        "cgl_imageSamples_image2DMS_RWTexture2DMS_float2_1(msRg);" in generated_code
    )
    assert "int2 cgl_imageSize_image2D_RWTexture2D_float2(" not in generated_code
    assert "int2 cgl_imageSize_image2DMS_RWTexture2DMS_float2(" not in generated_code
    assert "int cgl_imageSamples_image2DMS_RWTexture2DMS_float2(" not in generated_code


def test_bool_string_and_char_literals_emit_slang_syntax():
    code = """
    shader main {
        vertex {
            void main() {
                bool enabled = true;
                bool disabled = false;
                string label = "debug";
                char marker = 'x';
                if (enabled && !disabled) {
                    label = "active";
                    marker = 'y';
                }
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    entry_point = next(iter(ast.stages.values())).entry_point

    assert entry_point.body.statements[0].initial_value.value is True
    assert entry_point.body.statements[1].initial_value.value is False

    generated_code = generate_code(ast)

    assert "bool enabled = true;" in generated_code
    assert "bool disabled = false;" in generated_code
    assert 'string label = "debug";' in generated_code
    assert "char marker = 'x';" in generated_code
    assert 'label = "active";' in generated_code
    assert "marker = 'y';" in generated_code
    assert "True" not in generated_code
    assert "False" not in generated_code


def test_inferred_let_declarations_emit_inferred_slang_types():
    code = """
    shader main {
        compute {
            void main() {
                let scalar = 1.0;
                let flag = true;
                let label = "debug";
                let marker = 'x';
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float scalar = 1.0;" in generated_code
    assert "bool flag = true;" in generated_code
    assert 'string label = "debug";' in generated_code
    assert "char marker = 'x';" in generated_code
    assert "None scalar" not in generated_code
    assert "None flag" not in generated_code


def test_direct_literal_nodes_emit_slang_escaping():
    codegen = SlangCodeGen()

    assert (
        codegen.generate_expression(LiteralNode(True, PrimitiveType("bool"))) == "true"
    )
    assert (
        codegen.generate_expression(LiteralNode('debug"name', PrimitiveType("string")))
        == '"debug\\"name"'
    )
    assert (
        codegen.generate_expression(LiteralNode("'", PrimitiveType("char"))) == "'\\''"
    )


def test_binary_expression_precedence_preserves_grouping_in_slang():
    code = """
    shader Precedence {
        compute {
            void main() {
                float a = 1.0;
                float b = 2.0;
                float c = 3.0;
                float grouped = (a + b) * c;
                float divisor = a / (b * c);
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    generated_code = generate_code(ast)

    assert "float grouped = (a + b) * c;" in generated_code
    assert "float grouped = a + b * c;" not in generated_code
    assert "float divisor = a / (b * c);" in generated_code
    assert "float divisor = a / b * c;" not in generated_code


def test_prefix_and_postfix_unary_operators_preserve_position():
    code = """
    shader main {
        vertex {
            void main() {
                int i = 0;
                i++;
                ++i;
                i--;
                --i;
                for (int j = 0; j < 2; j++) {
                    i++;
                }
            }
        }
    }
    """
    tokens = tokenize_code(code)
    ast = parse_code(tokens)
    entry_point = next(iter(ast.stages.values())).entry_point
    first_postfix = entry_point.body.statements[1].expression
    first_prefix = entry_point.body.statements[2].expression
    for_loop = entry_point.body.statements[5]

    assert first_postfix.is_postfix is True
    assert first_prefix.is_postfix is False
    assert for_loop.update.is_postfix is True

    generated_code = generate_code(ast)

    assert "\n    i++;\n" in generated_code
    assert "\n    ++i;\n" in generated_code
    assert "\n    i--;\n" in generated_code
    assert "\n    --i;\n" in generated_code
    assert "for (int j = 0; j < 2; j++)" in generated_code
    assert "++j" not in generated_code


if __name__ == "__main__":
    pytest.main()
