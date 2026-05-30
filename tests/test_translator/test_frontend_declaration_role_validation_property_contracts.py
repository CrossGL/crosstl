from typing import NamedTuple, Tuple

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from crosstl.translator.ast import ArrayType, ShaderStage, StructNode
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)


class ParameterRoleCase(NamedTuple):
    canonical: str
    attributes: Tuple[str, ...]
    stage: ShaderStage
    stage_source: str
    parameter_source: str


class StructRoleCase(NamedTuple):
    canonical: str
    attributes: Tuple[str, ...]
    type_prefix: str


PARAMETER_ROLE_CASES = (
    ParameterRoleCase(
        "payload",
        ("payload", "ray_payload"),
        ShaderStage.RAY_GENERATION,
        "ray_generation",
        "Payload_{suffix} data",
    ),
    ParameterRoleCase(
        "payload",
        ("payload", "raypayload"),
        ShaderStage.RAY_GENERATION,
        "ray_generation",
        "Payload_{suffix} data",
    ),
    ParameterRoleCase(
        "hit_attribute",
        ("hit_attribute", "hitattribute"),
        ShaderStage.RAY_MISS,
        "ray_miss",
        "HitAttributes_{suffix} data",
    ),
    ParameterRoleCase(
        "callable_data",
        ("callable_data", "callabledata"),
        ShaderStage.RAY_CALLABLE,
        "ray_callable",
        "CallableData_{suffix} data",
    ),
    ParameterRoleCase(
        "mesh_payload",
        ("mesh_payload", "meshpayload"),
        ShaderStage.MESH,
        "mesh",
        "MeshPayload_{suffix} data",
    ),
    ParameterRoleCase(
        "mesh_vertices",
        ("vertices",),
        ShaderStage.MESH,
        "mesh",
        "MeshVertex_{suffix} values[]",
    ),
    ParameterRoleCase(
        "mesh_indices",
        ("indices",),
        ShaderStage.MESH,
        "mesh",
        "uint values[]",
    ),
    ParameterRoleCase(
        "mesh_primitives",
        ("primitives",),
        ShaderStage.MESH,
        "mesh",
        "MeshPrimitive_{suffix} values[]",
    ),
)

STRUCT_ROLE_CASES = (
    StructRoleCase("payload", ("payload", "ray_payload", "raypayload"), "Payload"),
    StructRoleCase(
        "hit_attribute",
        ("hit_attribute", "hitattribute"),
        "HitAttributes",
    ),
    StructRoleCase(
        "callable_data",
        ("callable_data", "callabledata"),
        "CallableData",
    ),
    StructRoleCase(
        "mesh_payload",
        ("mesh_payload", "meshpayload"),
        "MeshPayload",
    ),
)

MESH_ONLY_ROLE_DECLARATIONS = (
    ("vertices", "struct MeshVertex @vertices { vec4 position; };"),
    ("indices", "struct MeshIndex @indices { uint value; };"),
    ("primitives", "struct MeshPrimitive @primitives { uint material; };"),
    ("vertices", "MeshVertex values[] @vertices;"),
    ("indices", "uint values[] @indices;"),
    ("primitives", "MeshPrimitive values[] @primitives;"),
    ("indices", "struct Container { uint values[] @indices; };"),
    ("primitives", "struct Container { MeshPrimitive values[] @primitives; };"),
)

ROLE_ALIAS_GROUPS = {
    "payload": ("payload", "ray_payload", "raypayload"),
    "hit_attribute": ("hit_attribute", "hitattribute"),
    "callable_data": ("callable_data", "callabledata"),
    "mesh_payload": ("mesh_payload", "meshpayload"),
    "mesh_vertices": ("vertices",),
    "mesh_indices": ("indices",),
    "mesh_primitives": ("primitives",),
}

CONFLICTING_ROLE_ALIAS_PAIRS = tuple(
    (left_alias, right_alias)
    for left_role, left_aliases in ROLE_ALIAS_GROUPS.items()
    for right_role, right_aliases in ROLE_ALIAS_GROUPS.items()
    if left_role < right_role
    for left_alias in left_aliases
    for right_alias in right_aliases
)


def parse_code(code):
    return Parser(Lexer(code).tokens).parse()


def role_support_declarations(suffix):
    return f"""
    struct Payload_{suffix} {{
        float value;
    }};
    struct HitAttributes_{suffix} {{
        vec2 barycentrics;
    }};
    struct CallableData_{suffix} {{
        uint shaderIndex;
    }};
    struct MeshPayload_{suffix} {{
        uint meshlet;
    }};
    struct MeshVertex_{suffix} {{
        vec4 position;
    }};
    struct MeshPrimitive_{suffix} {{
        uint material;
    }};
    """


@settings(max_examples=40, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    case=st.sampled_from(PARAMETER_ROLE_CASES),
)
def test_generated_declaration_role_aliases_are_allowed_on_function_parameters(
    suffix,
    case,
):
    parameter_source = case.parameter_source.format(suffix=suffix)
    attributes = " ".join(f"@{name}" for name in case.attributes)
    code = f"""
    shader ParameterRoleAliases_{suffix} {{
        {role_support_declarations(suffix)}

        {case.stage_source} {{
            void consume_{suffix}({parameter_source} {attributes}) {{
            }}
        }}
    }}
    """

    ast = parse_code(code)

    stage = ast.stages[case.stage]
    function = stage.local_functions[0]
    parameter = function.parameters[0]
    assert [attr.name for attr in parameter.attributes] == list(case.attributes)
    if case.canonical.startswith("mesh_") and case.canonical != "mesh_payload":
        assert isinstance(parameter.param_type, ArrayType)


@settings(max_examples=30, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    case=st.sampled_from(STRUCT_ROLE_CASES),
)
def test_generated_declaration_role_aliases_are_allowed_on_role_structs(
    suffix,
    case,
):
    attributes = " ".join(f"@{name}" for name in case.attributes)
    code = f"""
    shader StructRoleAliases_{suffix} {{
        struct {case.type_prefix}_{suffix} {attributes} {{
            float value;
        }};
    }}
    """

    ast = parse_code(code)

    role_struct = ast.structs[0]
    assert isinstance(role_struct, StructNode)
    assert role_struct.name == f"{case.type_prefix}_{suffix}"
    assert [attr.name for attr in role_struct.attributes] == list(case.attributes)


@settings(max_examples=30, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    role_declaration=st.sampled_from(MESH_ONLY_ROLE_DECLARATIONS),
)
def test_generated_mesh_output_roles_require_function_parameters(
    suffix,
    role_declaration,
):
    role_name, declaration = role_declaration
    code = f"""
    shader InvalidMeshRoleOwner_{suffix} {{
        struct MeshVertex {{
            vec4 position;
        }};
        struct MeshPrimitive {{
            uint material;
        }};

        {declaration}
    }}
    """

    with pytest.raises(
        ValueError,
        match=f"Declaration role metadata.*@{role_name}.*requires function parameter",
    ):
        parse_code(code)


@settings(max_examples=50, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    role_pair=st.sampled_from(CONFLICTING_ROLE_ALIAS_PAIRS),
)
def test_generated_distinct_declaration_role_aliases_conflict(suffix, role_pair):
    left_role, right_role = role_pair
    code = f"""
    shader ConflictingRoleAliases_{suffix} {{
        struct Payload {{
            float value;
        }};

        mesh {{
            void consume_{suffix}(Payload data @{left_role} @{right_role}) {{
            }}
        }}
    }}
    """

    with pytest.raises(ValueError, match="Conflicting declaration role metadata"):
        parse_code(code)
