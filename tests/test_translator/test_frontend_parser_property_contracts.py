from __future__ import annotations

from dataclasses import dataclass

from hypothesis import given, settings, strategies as st

from crosstl.translator.ast import NamedType, PointerType, ShaderStage
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)


@dataclass(frozen=True)
class StorageImageCase:
    access: str
    image_type: str
    format_source: str
    format_name: str
    format_value: str | None = None


STORAGE_IMAGE_CASES = (
    StorageImageCase("readonly", "image2D", "format(rgba32f)", "format", "rgba32f"),
    StorageImageCase("writeonly", "image3D", "rgba16f", "rgba16f"),
    StorageImageCase("readwrite", "uimage2D", "r32ui", "r32ui"),
)

ADDRESS_SPACE_QUALIFIERS = (
    "device",
    "constant",
    "thread",
    "threadgroup",
    "workgroup",
    "global",
    "local",
    "private",
    "function",
    "shared",
    "groupshared",
)


def parse_code(code):
    return Parser(Lexer(code).tokens).parse()


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    local_x=st.integers(min_value=1, max_value=128),
    local_y=st.integers(min_value=1, max_value=64),
    local_z=st.integers(min_value=1, max_value=16),
)
def test_generated_compute_layout_and_numthreads_metadata_match(
    suffix,
    local_x,
    local_y,
    local_z,
):
    code = f"""
    shader ComputeMetadata_{suffix} {{
        compute {{
            layout(
                local_size_x = {local_x},
                local_size_y = {local_y},
                local_size_z = {local_z}
            ) in;

            [numthreads({local_x}, {local_y}, {local_z})]
            void main() {{
            }}
        }}
    }}
    """

    ast = parse_code(code)
    compute_stage = ast.stages[ShaderStage.COMPUTE]
    layout = compute_stage.layout_qualifiers[0]
    numthreads = compute_stage.entry_point.attributes[0]

    assert compute_stage.execution_config == {
        "local_size_x": str(local_x),
        "local_size_y": str(local_y),
        "local_size_z": str(local_z),
        "numthreads": (str(local_x), str(local_y), str(local_z)),
    }
    assert layout.direction == "in"
    assert [entry.name for entry in layout.entries] == [
        "local_size_x",
        "local_size_y",
        "local_size_z",
    ]
    assert [entry.arguments[0].value for entry in layout.entries] == [
        local_x,
        local_y,
        local_z,
    ]
    assert numthreads.name == "numthreads"
    assert [argument.value for argument in numthreads.arguments] == [
        local_x,
        local_y,
        local_z,
    ]


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    set_index=st.integers(min_value=0, max_value=7),
    binding=st.integers(min_value=0, max_value=31),
    resource_case=st.sampled_from(STORAGE_IMAGE_CASES),
)
def test_generated_storage_image_metadata_preserves_layout_access_and_format(
    suffix,
    set_index,
    binding,
    resource_case,
):
    code = f"""
    shader ResourceMetadata_{suffix} {{
        layout(set = {set_index}, binding = {binding})
        {resource_case.access} uniform {resource_case.image_type}
            image_{suffix} @{resource_case.format_source};
    }}
    """

    ast = parse_code(code)
    resource = ast.global_variables[0]

    assert resource.name == f"image_{suffix}"
    assert resource.qualifiers == [resource_case.access, "uniform"]
    assert isinstance(resource.var_type, NamedType)
    assert resource.var_type.name == resource_case.image_type
    assert [attribute.name for attribute in resource.attributes] == [
        "set",
        "binding",
        resource_case.format_name,
    ]
    assert resource.attributes[0].arguments[0].value == set_index
    assert resource.attributes[1].arguments[0].value == binding
    if resource_case.format_value is not None:
        assert resource.attributes[2].arguments[0].name == resource_case.format_value
    else:
        assert resource.attributes[2].arguments == []


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    qualifier=st.sampled_from(ADDRESS_SPACE_QUALIFIERS),
)
def test_generated_parameter_address_space_qualifiers_preserve_pointer_ir(
    suffix,
    qualifier,
):
    code = f"""
    shader AddressSpaceMetadata_{suffix} {{
        struct Payload_{suffix} {{
            float value;
        }};

        void consume_{suffix}({qualifier} Payload_{suffix}* payload_{suffix}) {{
        }}
    }}
    """

    ast = parse_code(code)
    parameter = ast.functions[0].parameters[0]

    assert parameter.name == f"payload_{suffix}"
    assert parameter.qualifiers == [qualifier]
    assert isinstance(parameter.param_type, PointerType)
    assert isinstance(parameter.param_type.pointee_type, NamedType)
    assert parameter.param_type.pointee_type.name == f"Payload_{suffix}"
