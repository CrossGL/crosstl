"""CUDA and HIP refuse a built-in type of another language instead of writing it unchanged.

A type name the CUDA and HIP type tables had no entry for was written as it was, so ``min16float v;``,
``packed_float3 v;`` or ``atomic_uint v;`` reached the output, where neither toolkit defines it. The name is now
recorded when it is a built-in type of some language that neither the program nor the toolkit declares, and the
rendered program is refused if it still names it. Program types, the toolkits' own vector and half types and the
HLSL resource aliases are unaffected.
"""

import pytest

from crosstl.translator.codegen.cuda_codegen import CudaCodeGen
from crosstl.translator.codegen.cuda_refusals import CudaLoweringRefused
from crosstl.translator.codegen.hip_codegen import HipCodeGen
from crosstl.translator.codegen.hip_refusals import HipLoweringRefused
from crosstl.translator.codegen.written_type_names import (
    CUDA_TYPE_NAMES,
    HIP_TYPE_NAMES,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

GENERATORS = [
    pytest.param(CudaCodeGen, CudaLoweringRefused, "CUDA", id="cuda"),
    pytest.param(HipCodeGen, HipLoweringRefused, "HIP", id="hip"),
]


def _program(type_name):
    return f"""
shader S {{
    struct Particle {{
        float mass;
    }};
    compute {{
        layout(local_size_x = 1) in;
        void main() {{
            {type_name} value;
        }}
    }}
}}
"""


def _generate(generator, type_name):
    return generator().generate(Parser(Lexer(_program(type_name)).get_tokens()).parse())


@pytest.mark.parametrize("generator, refusal, label", GENERATORS)
@pytest.mark.parametrize(
    "type_name",
    ["min10float", "packed_float3", "atomic_uint", "Float16", "SIMD", "depth2d"],
)
def test_foreign_builtin_type_is_refused(generator, refusal, label, type_name):
    with pytest.raises(refusal) as raised:
        _generate(generator, type_name)
    assert f"the built-in type '{type_name}' has no {label} spelling" in str(
        raised.value
    )


@pytest.mark.parametrize("generator, refusal, label", GENERATORS)
@pytest.mark.parametrize(
    "type_name", ["Particle", "float4", "int2", "uint3", "half2", "double4"]
)
def test_program_and_toolkit_types_are_written(generator, refusal, label, type_name):
    output = _generate(generator, type_name)
    assert f"{type_name} value;" in output


@pytest.mark.parametrize(
    "generator, sampled, storage",
    [
        pytest.param(
            CudaCodeGen, "cudaTextureObject_t", "cudaSurfaceObject_t", id="cuda"
        ),
        pytest.param(HipCodeGen, "hipTextureObject_t", "hipSurfaceObject_t", id="hip"),
    ],
)
@pytest.mark.parametrize(
    "type_name, kind",
    [
        ("Texture2D<float4>", "sampled"),
        ("TextureCubeArray<float4>", "sampled"),
        ("RWTexture2D<float4>", "storage"),
        ("RWTexture3D<int>", "storage"),
        ("RWTexture2DArray<uint>", "storage"),
    ],
)
def test_resource_aliases_map_to_the_toolkit_handles(
    generator, sampled, storage, type_name, kind
):
    output = _generate(generator, type_name)
    assert f"{sampled if kind == 'sampled' else storage} value;" in output


@pytest.mark.parametrize("generator, refusal, label", GENERATORS)
@pytest.mark.parametrize(
    "type_name, written",
    [
        ("int8", "char"),
        ("uint8", "unsigned char"),
        ("uchar", "unsigned char"),
        ("int16", "short"),
        ("uint16", "unsigned short"),
        ("int32", "int"),
        ("uint32", "unsigned int"),
        ("ulong", "unsigned long long"),
        ("float16_t", "half"),
        ("min16float", "float"),
        ("min16uint", "unsigned int"),
    ],
)
def test_sized_scalar_spellings_keep_their_width(
    generator, refusal, label, type_name, written
):
    assert f"    {written} value;" in _generate(generator, type_name)


def test_toolkit_type_names_differ_only_in_the_toolkit_handles():
    assert "cudaTextureObject_t" in CUDA_TYPE_NAMES
    assert "cudaTextureObject_t" not in HIP_TYPE_NAMES
    assert "hipTextureObject_t" in HIP_TYPE_NAMES
    assert {"float4", "longlong2", "ulong1", "dim3", "__half2"} <= (
        CUDA_TYPE_NAMES & HIP_TYPE_NAMES
    )
