from hypothesis import given, settings
from hypothesis import strategies as st

from crosstl.translator.ast import ArrayType, NamedType, PrimitiveType, ShaderStage
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,8}", fullmatch=True)


def parse_code(code):
    return Parser(Lexer(code).tokens).parse()


def attribute_values(node):
    return {
        attribute.name: [argument.value for argument in attribute.arguments]
        for attribute in node.attributes
    }


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    set_index=st.integers(min_value=0, max_value=7),
    texture_binding=st.integers(min_value=0, max_value=31),
    texture_slot=st.integers(min_value=0, max_value=31),
    sampler_slot=st.integers(min_value=0, max_value=31),
    uav_slot=st.integers(min_value=0, max_value=31),
)
def test_generated_global_descriptor_role_metadata_preserves_resource_ir(
    suffix,
    set_index,
    texture_binding,
    texture_slot,
    sampler_slot,
    uav_slot,
):
    code = f"""
    shader DescriptorRoles_{suffix} {{
        Texture2D<float4> colorTex_{suffix}
            @texture({texture_slot}) @set({set_index}) @binding({texture_binding});
        sampler linearSampler_{suffix} @sampler({sampler_slot});
        RWTexture2D<float4> outColor_{suffix} @uav({uav_slot});
    }}
    """

    ast = parse_code(code)
    color_texture, sampler, storage_texture = ast.global_variables

    assert color_texture.name == f"colorTex_{suffix}"
    assert isinstance(color_texture.var_type, NamedType)
    assert color_texture.var_type.name == "Texture2D"
    assert len(color_texture.var_type.generic_args) == 1
    assert color_texture.var_type.generic_args[0].name == "float4"
    assert attribute_values(color_texture) == {
        "texture": [texture_slot],
        "set": [set_index],
        "binding": [texture_binding],
    }

    assert sampler.name == f"linearSampler_{suffix}"
    assert isinstance(sampler.var_type, NamedType)
    assert sampler.var_type.name == "sampler"
    assert attribute_values(sampler) == {"sampler": [sampler_slot]}

    assert storage_texture.name == f"outColor_{suffix}"
    assert isinstance(storage_texture.var_type, NamedType)
    assert storage_texture.var_type.name == "RWTexture2D"
    assert storage_texture.var_type.generic_args[0].name == "float4"
    assert attribute_values(storage_texture) == {"uav": [uav_slot]}


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    local_x=st.integers(min_value=1, max_value=128),
    set_index=st.integers(min_value=0, max_value=7),
    texture_binding=st.integers(min_value=0, max_value=31),
    sampler_binding=st.integers(min_value=0, max_value=31),
    buffer_binding=st.integers(min_value=0, max_value=31),
)
def test_generated_stage_local_resource_layouts_do_not_merge_with_stage_layouts(
    suffix,
    local_x,
    set_index,
    texture_binding,
    sampler_binding,
    buffer_binding,
):
    code = f"""
    shader StageResourceLayouts_{suffix} {{
        compute {{
            layout(local_size_x = {local_x}) in;
            layout(set = {set_index}, binding = {texture_binding})
            uniform sampler2D sourceTexture_{suffix};
            layout(set = {set_index}, binding = {sampler_binding})
            sampler sourceSampler_{suffix};
            layout(set = {set_index}, binding = {buffer_binding})
            buffer float values_{suffix}[];

            void main() {{
            }}
        }}
    }}
    """

    ast = parse_code(code)
    compute_stage = ast.stages[ShaderStage.COMPUTE]

    assert compute_stage.execution_config == {"local_size_x": str(local_x)}
    assert len(compute_stage.layout_qualifiers) == 1
    stage_layout = compute_stage.layout_qualifiers[0]
    assert stage_layout.direction == "in"
    assert [entry.name for entry in stage_layout.entries] == ["local_size_x"]
    assert stage_layout.entries[0].arguments[0].value == local_x

    source_texture, source_sampler, values = compute_stage.local_variables
    assert source_texture.name == f"sourceTexture_{suffix}"
    assert source_texture.qualifiers == ["uniform"]
    assert isinstance(source_texture.var_type, NamedType)
    assert source_texture.var_type.name == "sampler2D"
    assert attribute_values(source_texture) == {
        "set": [set_index],
        "binding": [texture_binding],
    }

    assert source_sampler.name == f"sourceSampler_{suffix}"
    assert isinstance(source_sampler.var_type, NamedType)
    assert source_sampler.var_type.name == "sampler"
    assert attribute_values(source_sampler) == {
        "set": [set_index],
        "binding": [sampler_binding],
    }

    assert values.name == f"values_{suffix}"
    assert values.qualifiers == ["buffer"]
    assert isinstance(values.var_type, ArrayType)
    assert isinstance(values.var_type.element_type, PrimitiveType)
    assert values.var_type.element_type.name == "float"
    assert values.var_type.size is None
    assert attribute_values(values) == {
        "set": [set_index],
        "binding": [buffer_binding],
    }


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    set_index=st.integers(min_value=0, max_value=7),
    binding=st.integers(min_value=0, max_value=31),
    array_size=st.integers(min_value=1, max_value=16),
)
def test_generated_resource_arrays_preserve_metadata_and_element_type(
    suffix,
    set_index,
    binding,
    array_size,
):
    code = f"""
    shader ResourceArrays_{suffix} {{
        sampler2D textures_{suffix}[{array_size}] @set({set_index}) @binding({binding});
    }}
    """

    ast = parse_code(code)
    textures = ast.global_variables[0]

    assert textures.name == f"textures_{suffix}"
    assert isinstance(textures.var_type, ArrayType)
    assert isinstance(textures.var_type.element_type, NamedType)
    assert textures.var_type.element_type.name == "sampler2D"
    assert textures.var_type.size.value == array_size
    assert attribute_values(textures) == {
        "set": [set_index],
        "binding": [binding],
    }
