from __future__ import annotations

import re
from dataclasses import dataclass

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import crosstl.translator
import crosstl.translator.codegen as codegen
from crosstl.translator.ast import PointerType, ShaderStage
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.codegen.mojo_codegen import MojoCodeGen

IDENTIFIER_SUFFIXES = st.from_regex(r"[a-z][a-z0-9_]{0,10}", fullmatch=True)


@dataclass(frozen=True)
class NumericCase:
    source_type: str
    local_initializer: str
    update_expression: str


NUMERIC_CASES = (
    NumericCase("float", "a", "tmp + b"),
    NumericCase("int", "a", "tmp + b"),
    NumericCase("uint", "a", "tmp + b"),
    NumericCase("vec2", "a", "tmp + b"),
    NumericCase("vec3", "a", "tmp + b"),
    NumericCase("vec4", "a", "tmp + b"),
    NumericCase("ivec2", "a", "tmp + b"),
    NumericCase("uvec2", "a", "tmp + b"),
)


@dataclass(frozen=True)
class ResourceMemoryQualifierCase:
    attribute: str
    hlsl_prefix: str
    glsl_qualifier: str
    metal_access: str


RESOURCE_MEMORY_QUALIFIER_CASES = (
    ResourceMemoryQualifierCase(
        "coherent", "globallycoherent ", "coherent", "read_write"
    ),
    ResourceMemoryQualifierCase(
        "globallycoherent", "globallycoherent ", "coherent", "read_write"
    ),
    ResourceMemoryQualifierCase("volatile", "", "volatile", "read_write"),
    ResourceMemoryQualifierCase("restrict", "", "restrict", "read_write"),
    ResourceMemoryQualifierCase("readonly", "", "readonly", "read"),
    ResourceMemoryQualifierCase("writeonly", "", "writeonly", "write"),
)


@dataclass(frozen=True)
class StructuredBufferCase:
    source_type: str
    hlsl_type: str
    glsl_type: str
    metal_type: str


@dataclass(frozen=True)
class FrontendResourceMetadataCase:
    image_type: str
    access_qualifier: str
    format_source: str
    format_name: str
    format_value: str | None = None


STRUCTURED_BUFFER_CASES = (
    StructuredBufferCase("float", "float", "float", "float"),
    StructuredBufferCase("int", "int", "int", "int"),
    StructuredBufferCase("uint", "uint", "uint", "uint"),
)

FRONTEND_RESOURCE_METADATA_CASES = (
    FrontendResourceMetadataCase("image2D", "readonly", "rgba32f", "rgba32f"),
    FrontendResourceMetadataCase("image3D", "readwrite", "r32f", "r32f"),
    FrontendResourceMetadataCase(
        "uimage2D", "writeonly", "format(r32ui)", "format", "r32ui"
    ),
)


def _assert_codegen_output_is_usable(backend, generated):
    assert isinstance(generated, str)
    assert generated.strip(), f"{backend} generated empty output"
    assert "Traceback" not in generated
    assert "<crosstl." not in generated
    assert "NotImplemented" not in generated


@settings(max_examples=25, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    set_index=st.integers(min_value=0, max_value=7),
    binding=st.integers(min_value=0, max_value=31),
    location=st.integers(min_value=0, max_value=15),
    component=st.integers(min_value=0, max_value=3),
    interpolation=st.sampled_from(("flat", "smooth", "noperspective")),
    sample_qualifier=st.sampled_from(("centroid", "sample")),
    address_space=st.sampled_from(("device", "constant", "threadgroup")),
    matrix_layout=st.sampled_from(("row_major", "column_major")),
    pack_offset=st.integers(min_value=0, max_value=15),
    resource_case=st.sampled_from(FRONTEND_RESOURCE_METADATA_CASES),
)
def test_frontend_metadata_ir_preserves_valid_backend_neutral_combinations(
    suffix,
    set_index,
    binding,
    location,
    component,
    interpolation,
    sample_qualifier,
    address_space,
    matrix_layout,
    pack_offset,
    resource_case,
):
    shader = f"""
    shader FrontendMetadata_{suffix} {{
        cbuffer Camera_{suffix} : register(b0, space{set_index}) {{
            {matrix_layout} mat4 viewProj_{suffix} : packoffset(c{pack_offset});
            layout(offset = {pack_offset * 16}) vec4 tint_{suffix};
        }}

        struct Payload {{
            float value;
        }};

        struct VertexOutput {{
            layout(location = {location}) vec3 position @{interpolation};
        }};

        layout(set = {set_index}, binding = {binding})
        {resource_case.access_qualifier} uniform {resource_case.image_type}
            image_{suffix} @{resource_case.format_source};

        void consume(
            {address_space} Payload* payload_{suffix} @payload,
            {resource_case.access_qualifier} {resource_case.image_type}
                param_image_{suffix} @{resource_case.format_source}
        ) {{
        }}

        vertex {{
            layout(std140, binding = {binding})
            [[buffer({binding})]]
            uniform StageCamera_{suffix} {{
                vec4 stageTint_{suffix};
            }};

            layout(location = {location}, component = {component})
            {interpolation} {sample_qualifier} out vec4 color_{suffix};

            void main() {{
            }}
        }}
    }}
    """

    ast = crosstl.translator.parse(shader)

    cbuffer = ast.cbuffers[0]
    assert cbuffer.name == f"Camera_{suffix}"
    assert [attribute.name for attribute in cbuffer.attributes] == ["register"]
    assert [arg.name for arg in cbuffer.attributes[0].arguments] == [
        "b0",
        f"space{set_index}",
    ]
    view_proj, tint = cbuffer.members
    assert [attribute.name for attribute in view_proj.attributes] == [
        matrix_layout,
        "packoffset",
    ]
    assert view_proj.attributes[1].arguments[0].name == f"c{pack_offset}"
    assert [attribute.name for attribute in tint.attributes] == ["offset"]
    assert tint.attributes[0].arguments[0].value == pack_offset * 16

    payload_struct, output_struct = ast.structs
    assert payload_struct.name == "Payload"
    output_member = output_struct.members[0]
    assert [attribute.name for attribute in output_member.attributes] == [
        "location",
        interpolation,
    ]
    assert output_member.attributes[0].arguments[0].value == location

    resource = ast.global_variables[0]
    assert resource.name == f"image_{suffix}"
    assert resource.qualifiers == [resource_case.access_qualifier, "uniform"]
    assert [attribute.name for attribute in resource.attributes] == [
        "set",
        "binding",
        resource_case.format_name,
    ]
    assert resource.attributes[0].arguments[0].value == set_index
    assert resource.attributes[1].arguments[0].value == binding
    if resource_case.format_value is not None:
        assert resource.attributes[2].arguments[0].name == resource_case.format_value

    pointer_param, image_param = ast.functions[0].parameters
    assert pointer_param.name == f"payload_{suffix}"
    assert pointer_param.qualifiers == [address_space]
    assert isinstance(pointer_param.param_type, PointerType)
    assert pointer_param.param_type.pointee_type.name == "Payload"
    assert [attribute.name for attribute in pointer_param.attributes] == ["payload"]

    assert image_param.name == f"param_image_{suffix}"
    assert image_param.qualifiers == [resource_case.access_qualifier]
    assert [attribute.name for attribute in image_param.attributes] == [
        resource_case.format_name
    ]

    vertex_stage = ast.stages[ShaderStage.VERTEX]
    stage_cbuffer = vertex_stage.local_cbuffers[0]
    assert stage_cbuffer.name == f"StageCamera_{suffix}"
    assert [attribute.name for attribute in stage_cbuffer.attributes] == [
        "std140",
        "binding",
        "buffer",
    ]
    assert stage_cbuffer.attributes[1].arguments[0].value == binding
    assert stage_cbuffer.attributes[2].arguments[0].value == binding

    color = vertex_stage.local_variables[0]
    assert color.name == f"color_{suffix}"
    assert color.qualifiers == [interpolation, sample_qualifier, "out"]
    assert [attribute.name for attribute in color.attributes] == [
        "location",
        "component",
    ]
    assert color.attributes[0].arguments[0].value == location
    assert color.attributes[1].arguments[0].value == component


@settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    numeric_case=st.sampled_from(NUMERIC_CASES),
)
def test_numeric_helper_codegen_is_total_and_deterministic_for_all_backends(
    suffix, numeric_case
):
    function_name = f"fn_{suffix}"
    shader = f"""
    shader Property_{suffix} {{
        {numeric_case.source_type} {function_name}(
            {numeric_case.source_type} a,
            {numeric_case.source_type} b
        ) {{
            {numeric_case.source_type} tmp = {numeric_case.local_initializer};
            tmp = {numeric_case.update_expression};
            return tmp;
        }}
    }}
    """

    ast = crosstl.translator.parse(shader)

    for backend in codegen.backend_names():
        first = codegen.get_codegen(backend).generate(ast)
        second = codegen.get_codegen(backend).generate(ast)
        assert first == second, f"{backend} code generation is not deterministic"
        _assert_codegen_output_is_usable(backend, first)


@settings(max_examples=20, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    loop_bound=st.integers(min_value=1, max_value=16),
    threshold=st.floats(
        min_value=-8.0,
        max_value=8.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    ),
)
def test_control_flow_codegen_is_total_for_all_backends(suffix, loop_bound, threshold):
    threshold_literal = f"{threshold:.6f}"
    shader = f"""
    shader ControlFlow_{suffix} {{
        float accumulate_{suffix}(float seed, float step) {{
            float acc = seed;
            for (int i = 0; i < {loop_bound}; i++) {{
                acc = acc + step;
            }}
            if (acc > {threshold_literal}) {{
                return acc;
            }} else {{
                return step;
            }}
        }}
    }}
    """

    ast = crosstl.translator.parse(shader)

    for backend in codegen.backend_names():
        generated = codegen.get_codegen(backend).generate(ast)
        _assert_codegen_output_is_usable(backend, generated)


def _texture_sampling_shader(suffix, arrayed, helper):
    array_suffix = "[4]" if arrayed else ""
    declarations = f"""
        sampler2D tex_{suffix}{array_suffix};
        sampler smp_{suffix}{array_suffix};
    """

    if helper and arrayed:
        helper_source = f"""
        vec4 sample_{suffix}(sampler2D textures[], sampler samplers[], vec2 uv) {{
            return texture(textures[2], samplers[2], uv);
        }}
        """
        sample_expression = f"sample_{suffix}(tex_{suffix}, smp_{suffix}, input.uv)"
    elif helper:
        helper_source = f"""
        vec4 sample_{suffix}(sampler2D tex, sampler smp, vec2 uv) {{
            return texture(tex, smp, uv);
        }}
        """
        sample_expression = f"sample_{suffix}(tex_{suffix}, smp_{suffix}, input.uv)"
    else:
        index = "[1]" if arrayed else ""
        helper_source = ""
        sample_expression = (
            f"texture(tex_{suffix}{index}, smp_{suffix}{index}, input.uv)"
        )

    return f"""
    shader TextureProperty_{suffix} {{
        {declarations}

        struct FSInput {{
            vec2 uv @ TEXCOORD0;
        }};

        {helper_source}

        fragment {{
            vec4 main(FSInput input) @ gl_FragColor {{
                return {sample_expression};
            }}
        }}
    }}
    """


def _assert_no_source_sampler_leaks_to_glsl(glsl, suffix):
    assert f"smp_{suffix}" not in glsl
    assert "SamplerState" not in glsl
    assert "[[sampler" not in glsl


@settings(max_examples=20, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    arrayed=st.booleans(),
    helper=st.booleans(),
)
def test_primary_graphics_texture_sampling_contracts_are_preserved(
    suffix, arrayed, helper
):
    shader = _texture_sampling_shader(suffix, arrayed, helper)
    ast = crosstl.translator.parse(shader)

    hlsl = HLSLCodeGen().generate_stage(ast, "fragment")
    glsl = GLSLCodeGen().generate_stage(ast, "fragment")
    metal = MetalCodeGen().generate_stage(ast, "fragment")

    for backend, generated in (
        ("directx", hlsl),
        ("opengl", glsl),
        ("metal", metal),
    ):
        _assert_codegen_output_is_usable(backend, generated)

    assert ".Sample(" in hlsl
    assert "SamplerState" in hlsl
    assert not re.search(r"\btexture\s*\(", hlsl)

    assert re.search(r"\btexture\s*\(", glsl)
    assert f"sampler2D tex_{suffix}" in glsl
    _assert_no_source_sampler_leaks_to_glsl(glsl, suffix)

    assert ".sample(" in metal
    assert re.search(r"\bsampler\b", metal)
    assert not re.search(r"(?<!\[)\btexture\s*\(", metal)


def _shadow_texture_sampling_shader(suffix, arrayed, helper):
    array_suffix = "[4]" if arrayed else ""
    declarations = f"""
        sampler2DShadow shadow_{suffix}{array_suffix};
        sampler cmp_{suffix}{array_suffix};
    """

    if helper and arrayed:
        helper_source = f"""
        float sample_shadow_{suffix}(
            sampler2DShadow shadows[],
            sampler samplers[],
            vec2 uv,
            float depth
        ) {{
            return textureCompare(shadows[2], samplers[2], uv, depth);
        }}
        """
        sample_expression = (
            f"sample_shadow_{suffix}(shadow_{suffix}, cmp_{suffix}, "
            "input.uv, input.depth)"
        )
    elif helper:
        helper_source = f"""
        float sample_shadow_{suffix}(
            sampler2DShadow shadowMap,
            sampler cmpSampler,
            vec2 uv,
            float depth
        ) {{
            return textureCompare(shadowMap, cmpSampler, uv, depth);
        }}
        """
        sample_expression = (
            f"sample_shadow_{suffix}(shadow_{suffix}, cmp_{suffix}, "
            "input.uv, input.depth)"
        )
    else:
        index = "[1]" if arrayed else ""
        helper_source = ""
        sample_expression = (
            f"textureCompare(shadow_{suffix}{index}, cmp_{suffix}{index}, "
            "input.uv, input.depth)"
        )

    return f"""
    shader ShadowTextureProperty_{suffix} {{
        {declarations}

        struct FSInput {{
            vec2 uv @ TEXCOORD0;
            float depth;
        }};

        {helper_source}

        fragment {{
            float main(FSInput input) @ gl_FragDepth {{
                return {sample_expression};
            }}
        }}
    }}
    """


@settings(max_examples=20, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    arrayed=st.booleans(),
    helper=st.booleans(),
)
def test_primary_graphics_shadow_texture_sampler_contracts_are_preserved(
    suffix, arrayed, helper
):
    shader = _shadow_texture_sampling_shader(suffix, arrayed, helper)
    ast = crosstl.translator.parse(shader)

    hlsl = HLSLCodeGen().generate_stage(ast, "fragment")
    glsl = GLSLCodeGen().generate_stage(ast, "fragment")
    metal = MetalCodeGen().generate_stage(ast, "fragment")

    for backend, generated in (
        ("directx", hlsl),
        ("opengl", glsl),
        ("metal", metal),
    ):
        _assert_codegen_output_is_usable(backend, generated)

    assert "SamplerComparisonState" in hlsl
    assert ".SampleCmp(" in hlsl
    assert not re.search(r"\btextureCompare\s*\(", hlsl)

    assert f"sampler2DShadow shadow_{suffix}" in glsl
    assert f"cmp_{suffix}" not in glsl
    assert re.search(r"\btexture\s*\(", glsl)

    assert "depth2d<float>" in metal
    assert ".sample_compare(" in metal
    assert re.search(r"\bsampler\b", metal)


def test_comparison_sampler_array_is_an_explicit_sampler_resource():
    shader = """
    shader ComparisonSamplerArrayRole {
        compute {
            layout(set = 0, binding = 2) uniform sampler2DShadow shadowMap;
            layout(set = 0, binding = 4) comparison_sampler shadowCompareSamplers[2];

            void main() {
                float visibility = textureCompare(
                    shadowMap,
                    shadowCompareSamplers[0],
                    vec2(0.5, 0.5),
                    0.25
                );
                float lodVisibility = textureCompareLod(
                    shadowMap,
                    shadowCompareSamplers[1],
                    vec2(0.25, 0.75),
                    0.33,
                    1.0
                );
            }
        }
    }
    """
    ast = crosstl.translator.parse(shader)

    hlsl = HLSLCodeGen().generate(ast)
    glsl = GLSLCodeGen().generate(ast)
    metal = MetalCodeGen().generate(ast)
    mojo = MojoCodeGen().generate(ast)

    for backend, generated in (
        ("directx", hlsl),
        ("opengl", glsl),
        ("metal", metal),
        ("mojo", mojo),
    ):
        _assert_codegen_output_is_usable(backend, generated)

    assert "SamplerComparisonState shadowCompareSamplers[2]" in hlsl
    assert (
        "shadowMap.SampleCmp(shadowCompareSamplers[0], "
        "float2(0.5, 0.5), 0.25)" in hlsl
    )
    assert (
        "shadowMap.SampleCmpLevel(shadowCompareSamplers[1], "
        "float2(0.25, 0.75), 0.33, 1.0)" in hlsl
    )

    assert "comparison_sampler" not in glsl
    assert "shadowCompareSamplers" not in glsl
    assert "texture(shadowMap, vec3(vec2(0.5, 0.5), 0.25))" in glsl
    assert "textureLod(shadowMap, vec3(vec2(0.25, 0.75), 0.33), 1.0)" in glsl

    assert "array<sampler, 2> shadowCompareSamplers [[sampler(4)]]" in metal
    assert (
        "shadowMap.sample_compare(shadowCompareSamplers[0], "
        "float2(0.5, 0.5), 0.25)" in metal
    )
    assert (
        "shadowMap.sample_compare(shadowCompareSamplers[1], "
        "float2(0.25, 0.75), 0.33, level(1.0))" in metal
    )

    assert "name=shadowCompareSamplers kind=sampler" in mojo
    assert "InlineArray[Sampler, 2](Sampler(), Sampler())" in mojo
    assert "_crossgl_texture_compare_sampler_Texture2DShadow_Sampler" in mojo
    assert "_crossgl_texture_compare_lod_sampler_Texture2DShadow_Sampler" in mojo


@settings(max_examples=20, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    qualifier_case=st.sampled_from(RESOURCE_MEMORY_QUALIFIER_CASES),
)
def test_primary_graphics_resource_memory_qualifiers_are_preserved_where_supported(
    suffix, qualifier_case
):
    image_name = f"image_{suffix}"
    shader = f"""
    shader ResourceMemoryQualifier_{suffix} {{
        uimage2D {image_name} @r32ui @{qualifier_case.attribute};

        compute {{
            void main() {{
            }}
        }}
    }}
    """

    ast = crosstl.translator.parse(shader)

    hlsl = HLSLCodeGen().generate(ast)
    glsl = GLSLCodeGen().generate(ast)
    metal = MetalCodeGen().generate(ast)

    for backend, generated in (
        ("directx", hlsl),
        ("opengl", glsl),
        ("metal", metal),
    ):
        _assert_codegen_output_is_usable(backend, generated)

    assert (
        f"{qualifier_case.hlsl_prefix}RWTexture2D<uint> {image_name} : register(u0);"
        in hlsl
    )
    assert f": {qualifier_case.attribute}" not in hlsl
    assert f"@{qualifier_case.attribute}" not in hlsl

    assert (
        "layout(r32ui, binding = 0) "
        f"{qualifier_case.glsl_qualifier} uniform uimage2D {image_name};" in glsl
    )

    assert (
        f"texture2d<uint, access::{qualifier_case.metal_access}> "
        f"{image_name} [[texture(0)]]" in metal
    )
    assert f"[[{qualifier_case.attribute}]]" not in metal


@settings(max_examples=20, deadline=None)
@given(
    suffix=IDENTIFIER_SUFFIXES,
    buffer_case=st.sampled_from(STRUCTURED_BUFFER_CASES),
)
def test_primary_graphics_structured_buffer_access_contracts_are_preserved(
    suffix, buffer_case
):
    source_name = f"source_{suffix}"
    target_name = f"target_{suffix}"
    shader = f"""
    shader StructuredBufferProperty_{suffix} {{
        StructuredBuffer<{buffer_case.source_type}> {source_name} @binding(1);
        RWStructuredBuffer<{buffer_case.source_type}> {target_name} @binding(2);

        compute {{
            void main(uint3 tid @ gl_GlobalInvocationID) {{
                uint len;
                {buffer_case.source_type} value = buffer_load({source_name}, tid.x);
                buffer_dimensions({target_name}, len);
                buffer_store({target_name}, tid.x, value);
            }}
        }}
    }}
    """

    ast = crosstl.translator.parse(shader)

    hlsl = HLSLCodeGen().generate(ast)
    glsl = GLSLCodeGen().generate(ast)
    metal = MetalCodeGen().generate(ast)

    for backend, generated in (
        ("directx", hlsl),
        ("opengl", glsl),
        ("metal", metal),
    ):
        _assert_codegen_output_is_usable(backend, generated)

    assert (
        f"StructuredBuffer<{buffer_case.hlsl_type}> {source_name} : register(t1);"
        in hlsl
    )
    assert (
        f"RWStructuredBuffer<{buffer_case.hlsl_type}> {target_name} : register(u2);"
        in hlsl
    )
    assert f"{buffer_case.hlsl_type} value = {source_name}.Load(tid.x);" in hlsl
    assert f"{target_name}.GetDimensions(len);" in hlsl
    assert f"{target_name}.Store(tid.x, value);" in hlsl
    assert "buffer_load" not in hlsl
    assert "buffer_store" not in hlsl
    assert "buffer_dimensions" not in hlsl

    assert (
        f"layout(std430, binding = 1) readonly buffer {source_name}Buffer "
        f"{{ {buffer_case.glsl_type} {source_name}[]; }};" in glsl
    )
    assert (
        f"layout(std430, binding = 2) buffer {target_name}Buffer "
        f"{{ {buffer_case.glsl_type} {target_name}[]; }};" in glsl
    )
    assert (
        f"{buffer_case.glsl_type} value = {source_name}[gl_GlobalInvocationID.x];"
        in glsl
    )
    assert f"len = {target_name}.length();" in glsl
    assert f"{target_name}[gl_GlobalInvocationID.x] = value;" in glsl

    assert f"const device {buffer_case.metal_type}* {source_name}" in metal
    assert f"device {buffer_case.metal_type}* {target_name}" in metal
    assert f"{buffer_case.metal_type} value = {source_name}[tid.x];" in metal
    assert f"constant uint* {target_name}Length" in metal
    assert f"len = {target_name}Length[0];" in metal
    assert f"{target_name}[tid.x] = value;" in metal


def test_primary_compute_builtin_parameters_map_to_backend_native_aliases():
    shader = """
    shader ComputeBuiltinAliasConsistency {
        compute {
            void main(
                uvec3 dispatchId @ gl_GlobalInvocationID,
                uvec3 groupId @ gl_WorkGroupID,
                uvec3 groupThreadId @ gl_LocalInvocationID,
                uint groupIndex @ gl_LocalInvocationIndex
            ) {
                uint folded = dispatchId.x + groupId.y + groupThreadId.z
                    + groupIndex;
            }
        }
    }
    """
    ast = crosstl.translator.parse(shader)

    hlsl = HLSLCodeGen().generate_stage(ast, "compute")
    glsl = GLSLCodeGen().generate_stage(ast, "compute")
    metal = MetalCodeGen().generate_stage(ast, "compute")

    for backend, generated in (
        ("directx", hlsl),
        ("opengl", glsl),
        ("metal", metal),
    ):
        _assert_codegen_output_is_usable(backend, generated)

    assert "uint3 dispatchId : SV_DispatchThreadID" in hlsl
    assert "uint3 groupId : SV_GroupID" in hlsl
    assert "uint3 groupThreadId : SV_GroupThreadID" in hlsl
    assert "uint groupIndex : SV_GroupIndex" in hlsl
    assert "gl_GlobalInvocationID" not in hlsl

    assert "void main()" in glsl
    assert "gl_GlobalInvocationID.x" in glsl
    assert "gl_WorkGroupID.y" in glsl
    assert "gl_LocalInvocationID.z" in glsl
    assert "gl_LocalInvocationIndex" in glsl
    for source_name in ("dispatchId", "groupId", "groupThreadId", "groupIndex"):
        assert source_name not in glsl

    assert "uint3 dispatchId [[thread_position_in_grid]]" in metal
    assert "uint3 groupId [[threadgroup_position_in_grid]]" in metal
    assert "uint3 groupThreadId [[thread_position_in_threadgroup]]" in metal
    assert "uint groupIndex [[thread_index_in_threadgroup]]" in metal
    assert "gl_GlobalInvocationID" not in metal


def test_primary_graphics_vector_result_types_are_preserved():
    shader = """
    shader VectorResultTypeConsistency {
        vec3 bend(vec3 normal, vec3 tangent, float scale) {
            vec3 n = normalize(normal);
            vec3 t = normalize(tangent);
            vec3 b = cross(n, t);
            return reflect(b * scale, n);
        }

        fragment {
            vec4 main(vec3 normal @ NORMAL, vec3 tangent @ TANGENT)
                @ gl_FragColor {
                vec3 lit = bend(normal, tangent, 0.5);
                return vec4(lit, 1.0);
            }
        }
    }
    """
    ast = crosstl.translator.parse(shader)

    hlsl = HLSLCodeGen().generate_stage(ast, "fragment")
    glsl = GLSLCodeGen().generate_stage(ast, "fragment")
    metal = MetalCodeGen().generate_stage(ast, "fragment")

    for backend, generated in (
        ("directx", hlsl),
        ("opengl", glsl),
        ("metal", metal),
    ):
        _assert_codegen_output_is_usable(backend, generated)

    assert "float3 bend(float3 normal, float3 tangent, float scale)" in hlsl
    assert "float3 n = normalize(normal);" in hlsl
    assert "float3 b = cross(n, t);" in hlsl
    assert "float3 lit = bend(normal, tangent, 0.5);" in hlsl

    assert "vec3 bend(vec3 normal, vec3 tangent, float scale)" in glsl
    assert "vec3 n = normalize(normal);" in glsl
    assert "vec3 b = cross(n, t);" in glsl
    assert "vec3 lit = bend(normal, tangent, 0.5);" in glsl

    assert "float3 bend(float3 normal, float3 tangent, float scale)" in metal
    assert "float3 n = normalize(normal);" in metal
    assert "float3 b = cross(n, t);" in metal
    assert "float3 lit = bend(normal, tangent, 0.5);" in metal


def test_primary_graphics_texture_helper_calls_survive_shadowed_identifier_names():
    shader = """
    shader TextureHelperShadowedIdentifiers {
        sampler2D textures[4];
        sampler samplers[4];

        struct FSInput {
            vec2 uv @ TEXCOORD0;
            int layer @ TEXCOORD1;
        };

        vec4 sampleLayer(
            sampler2D textures[4],
            sampler samplers[4],
            int layer,
            vec2 uv
        ) {
            sampler2D texture = textures[layer];
            sampler sampler = samplers[layer];
            return texture(texture, sampler, uv);
        }

        fragment {
            vec4 main(FSInput input) @ gl_FragColor {
                return sampleLayer(textures, samplers, input.layer, input.uv);
            }
        }
    }
    """
    ast = crosstl.translator.parse(shader)

    hlsl = HLSLCodeGen().generate_stage(ast, "fragment")
    glsl = GLSLCodeGen().generate_stage(ast, "fragment")
    metal = MetalCodeGen().generate_stage(ast, "fragment")

    for backend, generated in (
        ("directx", hlsl),
        ("opengl", glsl),
        ("metal", metal),
    ):
        _assert_codegen_output_is_usable(backend, generated)

    assert "Texture2D texture = textures[layer];" in hlsl
    assert "SamplerState sampler = samplers[layer];" in hlsl
    assert "return texture.Sample(sampler, uv);" in hlsl
    assert "texture(texture" not in hlsl

    assert "vec4 sampleLayer(sampler2D textures[4], int layer, vec2 uv)" in glsl
    assert "return texture(textures[layer], uv);" in glsl
    assert "samplers" not in glsl
    assert "sampler sampler" not in glsl

    assert "texture2d<float> texture = textures[layer];" in metal
    assert "sampler sampler = samplers[layer];" in metal
    assert "return texture.sample(sampler, uv);" in metal
    assert "texture(texture" not in metal


def test_hlsl_lerp_alias_does_not_call_shadowed_mix_target():
    shader = """
    shader LerpAliasTargetShadow {
        float shade(float a, float b, float t) {
            float mix = t;
            return lerp(a, b, t) + mix;
        }
    }
    """
    ast = crosstl.translator.parse(shader)

    hlsl = HLSLCodeGen().generate(ast)
    glsl = GLSLCodeGen().generate(ast)
    metal = MetalCodeGen().generate(ast)

    for backend, generated in (
        ("directx", hlsl),
        ("opengl", glsl),
        ("metal", metal),
    ):
        _assert_codegen_output_is_usable(backend, generated)

    assert "return (lerp(a, b, t) + mix);" in hlsl

    assert "float mix_ = t;" in glsl
    assert "return (mix(a, b, t) + mix_);" in glsl
    assert "return (mix(a, b, t) + mix);" not in glsl

    assert "float mix = t;" in metal
    assert "return metal::mix(a, b, t) + mix;" in metal
    assert "return mix(a, b, t) + mix;" not in metal
