from typing import NamedTuple, Tuple

import pytest
from hypothesis import assume, given, settings, strategies as st

from crosstl.translator.ast import LiteralNode, NamedType, ShaderStage
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


class TessellationFlagCase(NamedTuple):
    attribute: str
    layout_entry: str
    aliases: Tuple[str, ...]


class TessellationTopologyCase(NamedTuple):
    attribute: str
    layout_entries: Tuple[str, ...]
    aliases: Tuple[str, ...]


DOMAIN_CASES = (
    TessellationFlagCase("domain", "triangles", ("tri", "triangle", "triangles")),
    TessellationFlagCase("domain", "quads", ("quad", "quads")),
    TessellationFlagCase("domain", "isolines", ("isoline", "isolines")),
)

PARTITIONING_CASES = (
    TessellationFlagCase(
        "partitioning",
        "equal_spacing",
        ("integer", "equal", "equal_spacing"),
    ),
    TessellationFlagCase(
        "partitioning",
        "fractional_even_spacing",
        ("fractional_even", "fractional_even_spacing"),
    ),
    TessellationFlagCase(
        "partitioning",
        "fractional_odd_spacing",
        ("fractional_odd", "fractional_odd_spacing"),
    ),
)

TOPOLOGY_CASES = (
    TessellationTopologyCase("outputtopology", ("points",), ("point", "points")),
    TessellationTopologyCase("outputtopology", ("lines",), ("line", "lines")),
    TessellationTopologyCase("outputtopology", ("line_strip",), ("line_strip",)),
    TessellationTopologyCase(
        "outputtopology",
        ("triangles",),
        ("triangle", "triangles"),
    ),
    TessellationTopologyCase(
        "outputtopology",
        ("triangles", "cw"),
        ("triangle_cw",),
    ),
    TessellationTopologyCase(
        "outputtopology",
        ("triangles", "ccw"),
        ("triangle_ccw",),
    ),
)

NON_TESSELLATION_STAGES = (
    "vertex",
    "fragment",
    "compute",
    "geometry",
    "mesh",
    "ray_generation",
)

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)


def parse_code(code):
    return Parser(Lexer(code).tokens).parse()


def assert_named_type(node, name):
    assert isinstance(node, NamedType)
    assert node.name == name


def assert_literal_argument(type_node, index, value):
    argument = type_node.generic_args[index]
    assert isinstance(argument, LiteralNode)
    assert argument.value == value


@settings(max_examples=60, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    domain_case=st.sampled_from(DOMAIN_CASES),
    partition_case=st.sampled_from(PARTITIONING_CASES),
    domain_alias_index=st.integers(min_value=0, max_value=2),
    partition_alias_index=st.integers(min_value=0, max_value=2),
)
def test_generated_tessellation_domain_and_partition_aliases_match_layouts(
    suffix,
    domain_case,
    partition_case,
    domain_alias_index,
    partition_alias_index,
):
    domain_alias = domain_case.aliases[domain_alias_index % len(domain_case.aliases)]
    partition_alias = partition_case.aliases[
        partition_alias_index % len(partition_case.aliases)
    ]
    code = f"""
    shader TessellationFlags_{suffix} {{
        tessellation_evaluation {{
            layout({domain_case.layout_entry}, {partition_case.layout_entry}) in;
            [domain("{domain_alias}"), partitioning("{partition_alias}")]
            void main() {{
            }}
        }}
    }}
    """

    ast = parse_code(code)

    evaluation = ast.stages[ShaderStage.TESSELLATION_EVALUATION]
    assert [entry.name for entry in evaluation.layout_qualifiers[0].entries] == [
        domain_case.layout_entry,
        partition_case.layout_entry,
    ]
    assert [attr.name for attr in evaluation.entry_point.attributes] == [
        domain_case.attribute,
        partition_case.attribute,
    ]


@settings(max_examples=40, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    topology_case=st.sampled_from(TOPOLOGY_CASES),
    alias_index=st.integers(min_value=0, max_value=1),
)
def test_generated_tessellation_output_topology_aliases_match_layouts(
    suffix,
    topology_case,
    alias_index,
):
    topology_alias = topology_case.aliases[alias_index % len(topology_case.aliases)]
    layout_entries = ", ".join(topology_case.layout_entries)
    code = f"""
    shader TessellationTopology_{suffix} {{
        tessellation_evaluation {{
            layout({layout_entries}) in;
            [outputtopology("{topology_alias}")]
            void main() {{
            }}
        }}
    }}
    """

    ast = parse_code(code)

    evaluation = ast.stages[ShaderStage.TESSELLATION_EVALUATION]
    assert [entry.name for entry in evaluation.layout_qualifiers[0].entries] == list(
        topology_case.layout_entries
    )
    assert evaluation.entry_point.attributes[0].name == topology_case.attribute


@settings(max_examples=30, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    control_points=st.integers(min_value=1, max_value=32),
)
def test_generated_tessellation_control_patch_counts_match_layout_and_function(
    suffix,
    control_points,
):
    code = f"""
    shader TessellationControlCounts_{suffix} {{
        struct HSOut_{suffix} {{
            vec4 position;
        }};

        tessellation_control {{
            layout(vertices = {control_points}) out;
            void main(OutputPatch<HSOut_{suffix}, {control_points}> outputPatch)
                @outputcontrolpoints({control_points})
                @patchconstantfunc(HSConst_{suffix})
            {{
            }}
        }}
    }}
    """

    ast = parse_code(code)

    control = ast.stages[ShaderStage.TESSELLATION_CONTROL]
    output_patch = control.entry_point.parameters[0]
    assert_named_type(output_patch.param_type, "OutputPatch")
    assert_named_type(output_patch.param_type.generic_args[0], f"HSOut_{suffix}")
    assert_literal_argument(output_patch.param_type, 1, control_points)
    assert [attr.name for attr in control.entry_point.attributes] == [
        "outputcontrolpoints",
        "patchconstantfunc",
    ]


@settings(max_examples=40, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    layout_points=st.integers(min_value=1, max_value=32),
    function_points=st.integers(min_value=1, max_value=32),
)
def test_generated_tessellation_control_count_mismatches_are_rejected(
    suffix,
    layout_points,
    function_points,
):
    assume(layout_points != function_points)
    code = f"""
    shader InvalidTessellationControlCounts_{suffix} {{
        tessellation_control {{
            layout(vertices = {layout_points}) out;
            [outputcontrolpoints({function_points})]
            void main() {{
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match="Conflicting stage/function metadata.*layout vertices="
        f"{layout_points}.*outputcontrolpoints={function_points}",
    ):
        parse_code(code)


@settings(max_examples=40, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    control_points=st.integers(min_value=1, max_value=32),
    evaluation_points=st.integers(min_value=1, max_value=32),
)
def test_generated_cross_stage_output_patch_count_mismatches_are_rejected(
    suffix,
    control_points,
    evaluation_points,
):
    assume(control_points != evaluation_points)
    code = f"""
    shader InvalidCrossStagePatchCounts_{suffix} {{
        struct HSOut_{suffix} {{
            vec4 position;
        }};

        tessellation_control {{
            HSOut_{suffix} main() @outputcontrolpoints({control_points}) {{
                HSOut_{suffix} output;
                return output;
            }}
        }}

        tessellation_evaluation {{
            void main(OutputPatch<HSOut_{suffix}, {evaluation_points}> patch)
                @domain(tri)
            {{
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match="Conflicting tessellation patch control-point metadata.*"
        f"OutputPatch<HSOut_{suffix}, {evaluation_points}>.*"
        f"outputcontrolpoints\\({control_points}\\).*"
        f"OutputPatch control points={evaluation_points}",
    ):
        parse_code(code)


@settings(max_examples=30, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    stage_name=st.sampled_from(NON_TESSELLATION_STAGES),
    patch_type=st.sampled_from(("InputPatch", "OutputPatch")),
    control_points=st.integers(min_value=1, max_value=32),
)
def test_generated_patch_parameters_require_tessellation_stages(
    suffix,
    stage_name,
    patch_type,
    control_points,
):
    code = f"""
    shader InvalidPatchStage_{suffix} {{
        struct PatchPayload_{suffix} {{
            vec4 position;
        }};

        {stage_name} {{
            void main({patch_type}<PatchPayload_{suffix}, {control_points}> patch) {{
            }}
        }}
    }}
    """

    with pytest.raises(
        ValueError,
        match=f"Patch type '{patch_type}'.*requires tessellation_control or "
        "tessellation_evaluation stage",
    ):
        parse_code(code)
