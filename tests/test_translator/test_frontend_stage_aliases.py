import pytest

from crosstl.translator.ast import ShaderStage
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser, shader_stage_from_name
from crosstl.translator.stage_utils import (
    shader_stage_from_name as shared_shader_stage_from_name,
)


def parse_code(code):
    return Parser(Lexer(code).tokens).parse()


@pytest.mark.parametrize(
    ("source_stage", "expected_stage", "expected_entry_name"),
    [
        ("vs VertexMain", ShaderStage.VERTEX, "VertexMain"),
        ("pixel", ShaderStage.FRAGMENT, "main"),
        ("pixel PixelMain", ShaderStage.FRAGMENT, "PixelMain"),
        ("pixel_shader PixelMain", ShaderStage.FRAGMENT, "PixelMain"),
        ("ps PixelMain", ShaderStage.FRAGMENT, "PixelMain"),
        ("cs ComputeMain", ShaderStage.COMPUTE, "ComputeMain"),
        ("kernel KernelMain", ShaderStage.COMPUTE, "KernelMain"),
        ("gs GeometryMain", ShaderStage.GEOMETRY, "GeometryMain"),
        ("hs HullMain", ShaderStage.TESSELLATION_CONTROL, "HullMain"),
        ("ds DomainMain", ShaderStage.TESSELLATION_EVALUATION, "DomainMain"),
        ("as AmplificationMain", ShaderStage.AMPLIFICATION, "AmplificationMain"),
        ("ms MeshMain", ShaderStage.MESH, "MeshMain"),
        ("raygeneration RayGenMain", ShaderStage.RAY_GENERATION, "RayGenMain"),
        ("raygen", ShaderStage.RAY_GENERATION, "main"),
        ("rgen RayGenMain", ShaderStage.RAY_GENERATION, "RayGenMain"),
        ("rchit ClosestHitMain", ShaderStage.RAY_CLOSEST_HIT, "ClosestHitMain"),
    ],
)
def test_backend_stage_alias_blocks_parse_to_canonical_stages(
    source_stage, expected_stage, expected_entry_name
):
    ast = parse_code(f"""
        shader StageAliases {{
            {source_stage} {{
                void main() {{ return; }}
            }}
        }}
        """)

    assert expected_stage in ast.stages
    if expected_stage is not ShaderStage.VERTEX:
        assert ShaderStage.VERTEX not in ast.stages
    assert ast.stages[expected_stage].entry_point.name == expected_entry_name


@pytest.mark.parametrize(
    ("alias", "expected_stage"),
    [
        ("vs", ShaderStage.VERTEX),
        ("vertex-shader", ShaderStage.VERTEX),
        ("pixel", ShaderStage.FRAGMENT),
        ("ps", ShaderStage.FRAGMENT),
        ("frag", ShaderStage.FRAGMENT),
        ("fragment-shader", ShaderStage.FRAGMENT),
        ("pixel-shader", ShaderStage.FRAGMENT),
        ("cs", ShaderStage.COMPUTE),
        ("compute-shader", ShaderStage.COMPUTE),
        ("kernel", ShaderStage.COMPUTE),
        ("gs", ShaderStage.GEOMETRY),
        ("geometry-shader", ShaderStage.GEOMETRY),
        ("hs", ShaderStage.TESSELLATION_CONTROL),
        ("hull-shader", ShaderStage.TESSELLATION_CONTROL),
        ("ds", ShaderStage.TESSELLATION_EVALUATION),
        ("domain-shader", ShaderStage.TESSELLATION_EVALUATION),
        ("as", ShaderStage.AMPLIFICATION),
        ("amplification-shader", ShaderStage.AMPLIFICATION),
        ("ms", ShaderStage.MESH),
        ("mesh-shader", ShaderStage.MESH),
        ("raygeneration", ShaderStage.RAY_GENERATION),
        ("raygen", ShaderStage.RAY_GENERATION),
        ("rgen", ShaderStage.RAY_GENERATION),
        ("raygeneration-shader", ShaderStage.RAY_GENERATION),
        ("ray-generation-shader", ShaderStage.RAY_GENERATION),
        ("closesthit", ShaderStage.RAY_CLOSEST_HIT),
        ("closest-hit", ShaderStage.RAY_CLOSEST_HIT),
        ("rchit", ShaderStage.RAY_CLOSEST_HIT),
        ("any-hit", ShaderStage.RAY_ANY_HIT),
        ("rahit", ShaderStage.RAY_ANY_HIT),
        ("intersection-shader", ShaderStage.RAY_INTERSECTION),
        ("rint", ShaderStage.RAY_INTERSECTION),
        ("miss-shader", ShaderStage.RAY_MISS),
        ("rmiss", ShaderStage.RAY_MISS),
        ("callable-shader", ShaderStage.RAY_CALLABLE),
        ("rcall", ShaderStage.RAY_CALLABLE),
        ("tesscontrol", ShaderStage.TESSELLATION_CONTROL),
        ("tesseval", ShaderStage.TESSELLATION_EVALUATION),
    ],
)
def test_shader_stage_from_name_canonicalizes_backend_aliases(alias, expected_stage):
    assert shader_stage_from_name(alias) is expected_stage


def test_parser_stage_alias_lookup_uses_shared_frontend_helper():
    assert shader_stage_from_name is shared_shader_stage_from_name


def test_shader_stage_attribute_aliases_validate_through_shared_normalizer():
    ast = parse_code("""
        shader AttributeAliases {
            fragment {
                [shader("pixel-shader")]
                void main() { return; }
            }

            ray_generation {
                void main() @shader(raygen) { return; }
            }
        }
        """)

    assert ast.stages[ShaderStage.FRAGMENT].entry_point.attributes[0].name == "shader"
    assert (
        ast.stages[ShaderStage.RAY_GENERATION].entry_point.attributes[0].name
        == "shader"
    )


def test_backend_stage_alias_names_remain_available_as_type_names():
    ast = parse_code("""
        shader AliasTypeNames {
            struct pixel {
                float value;
            }

            pixel make_pixel() {
                return pixel{value: 1.0};
            }
        }
        """)

    assert ast.structs[0].name == "pixel"
    assert ast.functions[0].name == "make_pixel"
    assert ast.stages == {}
