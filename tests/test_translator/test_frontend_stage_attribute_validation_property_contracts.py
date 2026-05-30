from typing import NamedTuple, Tuple

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from crosstl.translator.ast import ShaderStage
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


class ShaderStageAttributeCase(NamedTuple):
    canonical: str
    stage: ShaderStage
    block_names: Tuple[str, ...]
    aliases: Tuple[str, ...]


STAGE_ATTRIBUTE_CASES = (
    ShaderStageAttributeCase(
        "vertex",
        ShaderStage.VERTEX,
        ("vertex", "vs"),
        ("vertex", "vertex_shader", "vs"),
    ),
    ShaderStageAttributeCase(
        "fragment",
        ShaderStage.FRAGMENT,
        ("fragment", "pixel"),
        ("fragment", "pixel", "pixel-shader", "ps"),
    ),
    ShaderStageAttributeCase(
        "compute",
        ShaderStage.COMPUTE,
        ("compute", "kernel"),
        ("compute", "compute_shader", "kernel", "cs"),
    ),
    ShaderStageAttributeCase(
        "geometry",
        ShaderStage.GEOMETRY,
        ("geometry", "gs"),
        ("geometry", "geometry_shader", "gs"),
    ),
    ShaderStageAttributeCase(
        "tessellation_control",
        ShaderStage.TESSELLATION_CONTROL,
        ("hull", "tesscontrol"),
        ("hull", "hull_shader", "tesscontrol", "hs"),
    ),
    ShaderStageAttributeCase(
        "tessellation_evaluation",
        ShaderStage.TESSELLATION_EVALUATION,
        ("domain", "tesseval"),
        ("domain", "domain_shader", "tesseval", "ds"),
    ),
    ShaderStageAttributeCase(
        "task",
        ShaderStage.TASK,
        ("task",),
        ("task", "task_shader"),
    ),
    ShaderStageAttributeCase(
        "mesh",
        ShaderStage.MESH,
        ("mesh", "ms"),
        ("mesh", "mesh_shader", "ms"),
    ),
    ShaderStageAttributeCase(
        "amplification",
        ShaderStage.AMPLIFICATION,
        ("amplification", "as"),
        ("amplification", "amplification_shader", "as"),
    ),
    ShaderStageAttributeCase(
        "ray_generation",
        ShaderStage.RAY_GENERATION,
        ("ray_generation", "raygen"),
        ("ray_generation", "raygeneration", "raygen", "rgen"),
    ),
    ShaderStageAttributeCase(
        "ray_intersection",
        ShaderStage.RAY_INTERSECTION,
        ("intersection", "rint"),
        ("ray_intersection", "intersection", "rint"),
    ),
    ShaderStageAttributeCase(
        "ray_closest_hit",
        ShaderStage.RAY_CLOSEST_HIT,
        ("closesthit", "rchit"),
        ("ray_closest_hit", "closesthit", "closest_hit", "rchit"),
    ),
    ShaderStageAttributeCase(
        "ray_any_hit",
        ShaderStage.RAY_ANY_HIT,
        ("anyhit", "rahit"),
        ("ray_any_hit", "anyhit", "any_hit", "rahit"),
    ),
    ShaderStageAttributeCase(
        "ray_miss",
        ShaderStage.RAY_MISS,
        ("miss", "rmiss"),
        ("ray_miss", "miss", "rmiss"),
    ),
    ShaderStageAttributeCase(
        "ray_callable",
        ShaderStage.RAY_CALLABLE,
        ("callable", "rcall"),
        ("ray_callable", "callable", "rcall"),
    ),
)

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)


def parse_code(code):
    return Parser(Lexer(code).tokens).parse()


@settings(max_examples=60, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    case=st.sampled_from(STAGE_ATTRIBUTE_CASES),
    block_name_index=st.integers(min_value=0, max_value=1),
    alias_index=st.integers(min_value=0, max_value=3),
)
def test_generated_shader_stage_attribute_aliases_match_explicit_stage_blocks(
    suffix,
    case,
    block_name_index,
    alias_index,
):
    block_name = case.block_names[block_name_index % len(case.block_names)]
    alias = case.aliases[alias_index % len(case.aliases)]
    code = f"""
    shader StageAttributeAliases_{suffix} {{
        {block_name} {{
            [shader("{alias}")]
            void main() {{
                return;
            }}
        }}
    }}
    """

    ast = parse_code(code)

    stage = ast.stages[case.stage]
    assert stage.entry_point.name == "main"
    assert stage.entry_point.attributes[0].name == "shader"
    assert stage.entry_point.attributes[0].arguments[0].value == alias


@settings(max_examples=60, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    block_case=st.sampled_from(STAGE_ATTRIBUTE_CASES),
    alias_case=st.sampled_from(STAGE_ATTRIBUTE_CASES),
    alias_index=st.integers(min_value=0, max_value=3),
)
def test_generated_shader_stage_attribute_aliases_reject_mismatched_blocks(
    suffix,
    block_case,
    alias_case,
    alias_index,
):
    assume(block_case.canonical != alias_case.canonical)
    block_name = block_case.block_names[0]
    alias = alias_case.aliases[alias_index % len(alias_case.aliases)]
    code = f"""
    shader MismatchedStageAttribute_{suffix} {{
        {block_name} {{
            [shader("{alias}")]
            void main() {{
                return;
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=f"Function metadata '@shader\\({alias}\\)'.*"
        f"conflicts with {block_case.canonical} stage",
    ):
        parse_code(code)


@settings(max_examples=30, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    case=st.sampled_from(STAGE_ATTRIBUTE_CASES),
    alias_index=st.integers(min_value=0, max_value=3),
)
def test_generated_top_level_shader_stage_attributes_preserve_native_entrypoints(
    suffix,
    case,
    alias_index,
):
    alias = case.aliases[alias_index % len(case.aliases)]
    code = f"""
    shader TopLevelNativeEntrypoint_{suffix} {{
        [shader("{alias}")]
        void native_entry_{suffix}() {{
            return;
        }}

        compute {{
            void main() {{
                return;
            }}
        }}
    }}
    """

    ast = parse_code(code)

    native_entry = ast.functions[0]
    assert native_entry.name == f"native_entry_{suffix}"
    assert native_entry.attributes[0].name == "shader"
    assert native_entry.attributes[0].arguments[0].value == alias


@settings(max_examples=20, deadline=None)
@given(suffix=IDENTIFIER_SUFFIXES)
def test_generated_stage_shader_attribute_requires_single_known_stage(suffix):
    invalid_cases = (
        (
            '[shader("vertex", "fragment")]',
            "requires exactly one shader stage value",
        ),
        ('[shader("not_a_stage")]', "unknown shader stage 'not_a_stage'"),
        ("@shader", "requires exactly one shader stage value"),
    )

    for attribute_source, message in invalid_cases:
        code = f"""
        shader InvalidStageAttribute_{suffix} {{
            vertex {{
                {attribute_source}
                void main() {{
                    return;
                }}
            }}
        }}
        """

        with pytest.raises(ValueError, match=message):
            parse_code(code)
