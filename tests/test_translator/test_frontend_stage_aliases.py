import pytest

from crosstl.translator.ast import ShaderStage
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser, shader_stage_from_name


def parse_code(code):
    return Parser(Lexer(code).tokens).parse()


@pytest.mark.parametrize(
    ("source_stage", "expected_stage", "expected_entry_name"),
    [
        ("pixel", ShaderStage.FRAGMENT, "main"),
        ("pixel PixelMain", ShaderStage.FRAGMENT, "PixelMain"),
        ("pixel_shader PixelMain", ShaderStage.FRAGMENT, "PixelMain"),
        ("raygeneration RayGenMain", ShaderStage.RAY_GENERATION, "RayGenMain"),
        ("raygen", ShaderStage.RAY_GENERATION, "main"),
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
    assert ShaderStage.VERTEX not in ast.stages
    assert ast.stages[expected_stage].entry_point.name == expected_entry_name


@pytest.mark.parametrize(
    ("alias", "expected_stage"),
    [
        ("pixel", ShaderStage.FRAGMENT),
        ("pixel-shader", ShaderStage.FRAGMENT),
        ("raygeneration", ShaderStage.RAY_GENERATION),
        ("raygen", ShaderStage.RAY_GENERATION),
        ("closesthit", ShaderStage.RAY_CLOSEST_HIT),
        ("tesscontrol", ShaderStage.TESSELLATION_CONTROL),
        ("tesseval", ShaderStage.TESSELLATION_EVALUATION),
    ],
)
def test_shader_stage_from_name_canonicalizes_backend_aliases(alias, expected_stage):
    assert shader_stage_from_name(alias) is expected_stage


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
