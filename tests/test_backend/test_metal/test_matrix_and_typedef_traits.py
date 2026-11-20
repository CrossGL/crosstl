"""Type traits over matrix types and typedef'd structures fold to constants.

ggml's flash-attention kernels branch on ``is_same<kd4_t, half4x4>::value`` where ``kd4_t`` is a block
structure declared as ``typedef struct { ... } block_q4_0;``. Neither a matrix type nor a typedef'd structure
counted as a concrete trait operand, so every instantiation stopped with "operand unresolved". Both are now
concrete, and ``matrix<half, 4, 4>`` canonicalises to ``half4x4`` so the two spellings compare equal.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.backend.Metal.trait_operand_types import (
    canonical_matrix_type,
    typedef_aggregate_names,
)
from tests.stage_entry_helpers import translate_each_entry

KERNEL = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    typedef struct {
        half d;
        uchar qs[16];
    } block_q;

    template <typename block_t, typename tile_t>
    kernel void classify(device const block_t *src [[buffer(0)]],
                         device int *dst [[buffer(1)]],
                         uint gid [[thread_position_in_grid]])
    {
        dst[gid] = int(is_same<block_t, tile_t>::value)
            + 2 * int(is_same<tile_t, matrix<half, 4, 4>>::value)
            + 4 * int(is_same_v<block_t, block_q>);
    }

    template [[host_name("classify_block_half")]] kernel void classify<block_q, half4x4>(
        device const block_q *, device int *, uint);
    template [[host_name("classify_float_half")]] kernel void classify<float4x4, half4x4>(
        device const float4x4 *, device int *, uint);
    template [[host_name("classify_block_float")]] kernel void classify<block_q, float4x4>(
        device const block_q *, device int *, uint);
    """)

# The SPIR-V emitter has no 16-bit floating-point storage, so the half matrix buffer is checked in CrossGL only.
HALF_MATRIX_INSTANTIATION = textwrap.dedent("""
    template [[host_name("classify_half_half")]] kernel void classify<half4x4, half4x4>(
        device const half4x4 *, device int *, uint);
    """)


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("half4x4", "half4x4"),
        ("float2x3", "float2x3"),
        ("matrix<half, 4, 4>", "half4x4"),
        ("metal::matrix<float,3,2>", "float3x2"),
        ("::metal::matrix<bfloat, 4, 4>", "bfloat4x4"),
        ("half4", None),
        ("matrix<int, 4, 4>", None),
        ("float5x5", None),
    ],
)
def test_canonical_matrix_type(spelling, expected):
    assert canonical_matrix_type(spelling) == expected


def test_typedef_aggregate_names():
    code = (
        "typedef struct { half d; } block_a;\n"
        "typedef struct tag { float y; } block_b, *block_b_ptr, block_b_array[2];\n"
        "typedef union { uint u; float f; } bits_t;\n"
        "typedef float4 not_an_aggregate;\n"
    )
    preprocessor = MetalPreprocessor()
    names = typedef_aggregate_names(
        code,
        lambda text, start: preprocessor._find_matching_delimiter(
            text, start, "{", "}"
        ),
    )
    assert names == {"block_a", "block_b", "bits_t"}


def _classifications(code):
    values = {}
    for entry in (
        "classify_block_half",
        "classify_float_half",
        "classify_half_half",
        "classify_block_float",
    ):
        body = code[code.index(f"void {entry}(") :]
        values[entry] = body[: body.index(";")]
    return values


def test_instantiations_fold_each_trait(tmp_path):
    source = tmp_path / "classify.metal"
    source.write_text(KERNEL + HALF_MATRIX_INSTANTIATION, encoding="utf-8")
    values = _classifications(crosstl.translate(str(source), backend="cgl"))
    assert values["classify_block_half"].endswith(
        "int(false) + 2 * int(true) + 4 * int(true))"
    )
    assert values["classify_float_half"].endswith(
        "int(false) + 2 * int(true) + 4 * int(false))"
    )
    assert values["classify_half_half"].endswith(
        "int(true) + 2 * int(true) + 4 * int(false))"
    )
    assert values["classify_block_float"].endswith(
        "int(false) + 2 * int(false) + 4 * int(true))"
    )


_VALIDATORS = {
    "opengl": (
        "classify.comp",
        [["glslangValidator", "-S", "comp", "classify.comp"]],
    ),
    "vulkan": (
        "classify.spvasm",
        [
            ["spirv-as", "--target-env", "vulkan1.1", "classify.spvasm", "-o", "c.spv"],
            ["spirv-val", "--target-env", "vulkan1.1", "c.spv"],
        ],
    ),
    "directx": (
        "classify.hlsl",
        [
            ["dxc", "-T", "cs_6_2", "-enable-16bit-types", "-E", entry, "classify.hlsl"]
            for entry in ("CSMain", "CSMain_2", "CSMain_3")
        ],
    ),
    "metal": (
        "classify.metal",
        [["xcrun", "-sdk", "macosx", "metal", "-c", "classify.metal", "-o", "c.air"]],
    ),
}


@pytest.mark.parametrize("backend", sorted(_VALIDATORS))
def test_instantiations_reach_every_target(tmp_path, backend):
    file_name, commands = _VALIDATORS[backend]
    source = tmp_path / "source.metal"
    source.write_text(KERNEL, encoding="utf-8")
    artifacts = translate_each_entry(source, backend)
    # GLSL holds one compute entry per unit, so each instantiation is its own artifact.
    assert len(artifacts) == (3 if backend == "opengl" else 1)
    missing = [command[0] for command in commands if shutil.which(command[0]) is None]
    if missing:
        pytest.skip(f"{', '.join(missing)} not installed")
    for code in artifacts.values():
        (tmp_path / file_name).write_text(code, encoding="utf-8")
        for command in commands:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=300, cwd=tmp_path
            )
            assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
