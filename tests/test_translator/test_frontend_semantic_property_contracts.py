from hypothesis import given, settings
from hypothesis import strategies as st

from crosstl.translator.ast import IdentifierNode, LiteralNode, ShaderStage
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)
INTERPOLATION_QUALIFIERS = (
    "flat",
    "smooth",
    "noperspective",
    "centroid",
    "sample",
    "nointerpolation",
    "linear",
    "linear_centroid",
    "linear_noperspective",
    "linear_noperspective_centroid",
    "linear_sample",
)
TESSELLATION_DOMAINS = ("triangles", "quads", "isolines")
TESSELLATION_SPACING = ("equal_spacing", "fractional_even_spacing")
TESSELLATION_WINDING = ("cw", "ccw")


def parse_code(code):
    return Parser(Lexer(code).tokens).parse()


def attribute_names(attributes):
    return [attribute.name for attribute in attributes]


def single_literal_value(attribute):
    assert len(attribute.arguments) == 1
    argument = attribute.arguments[0]
    assert isinstance(argument, LiteralNode)
    return argument.value


def single_identifier_name(attribute):
    assert len(attribute.arguments) == 1
    argument = attribute.arguments[0]
    assert isinstance(argument, IdentifierNode)
    return argument.name


@settings(max_examples=30, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    member_interpolation=st.sampled_from(INTERPOLATION_QUALIFIERS),
    parameter_interpolation=st.sampled_from(INTERPOLATION_QUALIFIERS),
    semantic_slot=st.integers(min_value=0, max_value=15),
    location=st.integers(min_value=0, max_value=31),
)
def test_generated_semantic_and_interpolation_metadata_preserves_all_surfaces(
    suffix,
    member_interpolation,
    parameter_interpolation,
    semantic_slot,
    location,
):
    code = f"""
    shader SemanticMetadata_{suffix} {{
        struct StagePayload_{suffix} {{
            {member_interpolation} vec4 color_{suffix}
                : COLOR{semantic_slot}
                @location({location})
                [[user(userColor)]];
        }};

        fragment {{
            [earlydepthstencil]
            vec4 main(
                {parameter_interpolation} StagePayload_{suffix} input_{suffix}
                    : TEXCOORD{semantic_slot}
                    @builtin(position)
                    [[stage_in]]
            ) : SV_Target{semantic_slot} @location({location}) {{
                return vec4(0.0, 0.0, 0.0, 1.0);
            }}
        }}
    }}
    """

    ast = parse_code(code)
    member = ast.structs[0].members[0]
    fragment_entry = ast.stages[ShaderStage.FRAGMENT].entry_point
    parameter = fragment_entry.parameters[0]

    assert member.name == f"color_{suffix}"
    assert attribute_names(member.attributes) == [
        member_interpolation,
        f"COLOR{semantic_slot}",
        "location",
        "user",
    ]
    assert single_literal_value(member.attributes[2]) == location
    assert single_identifier_name(member.attributes[3]) == "userColor"

    assert parameter.name == f"input_{suffix}"
    assert parameter.qualifiers == [parameter_interpolation]
    assert attribute_names(parameter.attributes) == [
        f"TEXCOORD{semantic_slot}",
        "builtin",
        "stage_in",
    ]
    assert single_identifier_name(parameter.attributes[1]) == "position"

    assert attribute_names(fragment_entry.attributes) == [
        "earlydepthstencil",
        f"SV_Target{semantic_slot}",
        "location",
    ]
    assert single_literal_value(fragment_entry.attributes[2]) == location


@settings(max_examples=30, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    patch_vertices=st.integers(min_value=1, max_value=32),
    mesh_x=st.integers(min_value=1, max_value=128),
    mesh_y=st.integers(min_value=1, max_value=16),
    mesh_z=st.integers(min_value=1, max_value=8),
    max_vertices=st.integers(min_value=1, max_value=256),
    max_primitives=st.integers(min_value=1, max_value=256),
    domain=st.sampled_from(TESSELLATION_DOMAINS),
    spacing=st.sampled_from(TESSELLATION_SPACING),
    winding=st.sampled_from(TESSELLATION_WINDING),
)
def test_generated_mesh_and_tessellation_layout_metadata_preserves_entries(
    suffix,
    patch_vertices,
    mesh_x,
    mesh_y,
    mesh_z,
    max_vertices,
    max_primitives,
    domain,
    spacing,
    winding,
):
    code = f"""
    shader StageLayoutMetadata_{suffix} {{
        tessellation_control {{
            layout(vertices = {patch_vertices}) out;
            void main() {{
            }}
        }}

        tessellation_evaluation {{
            layout({domain}, {spacing}, {winding}) in;
            void main() {{
            }}
        }}

        mesh {{
            layout(
                local_size_x = {mesh_x},
                local_size_y = {mesh_y},
                local_size_z = {mesh_z}
            ) in;
            layout(
                triangles,
                max_vertices = {max_vertices},
                max_primitives = {max_primitives}
            ) out;
            void main() {{
            }}
        }}
    }}
    """

    ast = parse_code(code)
    tess_control = ast.stages[ShaderStage.TESSELLATION_CONTROL]
    tess_eval = ast.stages[ShaderStage.TESSELLATION_EVALUATION]
    mesh = ast.stages[ShaderStage.MESH]

    tess_control_layout = tess_control.layout_qualifiers[0]
    assert tess_control_layout.direction == "out"
    assert attribute_names(tess_control_layout.entries) == ["vertices"]
    assert single_literal_value(tess_control_layout.entries[0]) == patch_vertices

    tess_eval_layout = tess_eval.layout_qualifiers[0]
    assert tess_eval_layout.direction == "in"
    assert attribute_names(tess_eval_layout.entries) == [domain, spacing, winding]
    assert [entry.arguments for entry in tess_eval_layout.entries] == [[], [], []]

    assert mesh.execution_config == {
        "local_size_x": str(mesh_x),
        "local_size_y": str(mesh_y),
        "local_size_z": str(mesh_z),
    }
    mesh_input_layout, mesh_output_layout = mesh.layout_qualifiers
    assert mesh_input_layout.direction == "in"
    assert attribute_names(mesh_input_layout.entries) == [
        "local_size_x",
        "local_size_y",
        "local_size_z",
    ]
    assert [single_literal_value(entry) for entry in mesh_input_layout.entries] == [
        mesh_x,
        mesh_y,
        mesh_z,
    ]

    assert mesh_output_layout.direction == "out"
    assert attribute_names(mesh_output_layout.entries) == [
        "triangles",
        "max_vertices",
        "max_primitives",
    ]
    assert mesh_output_layout.entries[0].arguments == []
    assert single_literal_value(mesh_output_layout.entries[1]) == max_vertices
    assert single_literal_value(mesh_output_layout.entries[2]) == max_primitives
