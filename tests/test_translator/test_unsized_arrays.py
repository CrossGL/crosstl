"""Unsized arrays reach every target as a storage buffer, a sized array or a typed refusal.

``buffer T name[]`` became a ``static`` 1024-element private array in HLSL, a ``uniform`` array in GLSL and a
runtime array outside any block in SPIR-V; unsized arrays of values elsewhere got 1024 elements or were printed
unchanged. See :mod:`crosstl.translator.semantics.unsized_arrays`.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics import UnsizedArrayUnsupportedError
from tests.conformance.validators import validate

FLAT = """
shader FlatBuffers {
    compute {
        layout(std430, binding = 1) readonly buffer float values[];
        layout(std430, binding = 2) buffer float outv[];
        buffer ivec2 pairs[];

        void main() {
            uint i = gl_GlobalInvocationID.x;
            outv[i] = buffer_load(values, i) + float(pairs[i].y);
            buffer_store(outv, i + 1u, values[i]);
        }
    }
}
"""

BLOCK = """
shader BlockBuffers {
    compute {
        layout(std430, binding = 0) buffer vec3 dirs[];
        layout(std430, binding = 1) readonly buffer mat2 bases[];

        void main() {
            uint i = gl_GlobalInvocationID.x;
            vec3 d = buffer_load(dirs, i);
            buffer_store(dirs, i, vec3(bases[i][1], 1.0));
            dirs[i + 1u] = d.zyx;
        }
    }
}
"""

STD140 = """
shader Std140Buffers {
    compute {
        layout(std140, binding = 2) readonly buffer float weights[];
        layout(std430, binding = 3) buffer float outv[];

        void main() {
            uint i = gl_GlobalInvocationID.x;
            outv[i] = weights[i] + buffer_load(weights, i + 1u);
        }
    }
}
"""

INITIALIZED = """
shader Initialized {
    layout(std430, binding = 0) buffer float outv[];

    compute {
        void main() {
            float a[] = {1.0, 2.0, 3.0};
            float c[] = float[](4.0, 5.0);
            outv[0] = a[2] + c[1];
        }
    }
}
"""

REFUSED = {
    "private_global": (
        """
shader S {
    float g[];
    compute { void main() { float b = g[2]; } }
}"""
    ),
    "uniform_struct": (
        """
shader S {
    struct Globals { float weights[]; };
    uniform Globals globals;
    compute { void main() { float b = globals.weights[0]; } }
}"""
    ),
    "not_last_member": (
        """
shader S {
    struct Block { float a[]; float b; };
    Block block @glsl_buffer_block(std430) @binding(0);
    compute { void main() { block.b = block.a[0]; } }
}"""
    ),
    "local": (
        """
shader S {
    compute { void main() { float l[]; l[0] = 1.0; } }
}"""
    ),
    "parameter": (
        """
shader S {
    float first(float v[]) { return v[0]; }
    compute { void main() { float a[2]; a[0] = 1.0; a[1] = 2.0; float r = first(a); } }
}"""
    ),
    "stage_uniform": (
        """
shader S {
    fragment {
        uniform vec4 colors[];
        vec4 main() @ gl_FragColor { return colors[1]; }
    }
}"""
    ),
}

KEPT = """
shader Kept {
    struct Particle { vec4 position; float mass; };
    struct ParticleBlock { uint count; Particle particles[]; };
    ParticleBlock particleBlock @glsl_buffer_block(std430) @binding(0);
    RWStructuredBuffer<float> sb @binding(1);

    compute {
        void main() {
            particleBlock.particles[1].mass = sb[0] + float(particleBlock.count);
        }
    }
}
"""

BACKEND = {"hlsl": "directx", "glsl": "opengl", "spirv": "vulkan", "metal": "metal"}
SUFFIX = {"hlsl": ".hlsl", "glsl": ".comp", "spirv": ".spvasm", "metal": ".metal"}
HLSL_ENTRY = re.compile(r"\[numthreads\([^)]*\)\]\s*void\s+(\w+)")
TARGETS = ["hlsl", "glsl", "spirv", "wgsl", "metal"]


def translate(tmp_path, source, target):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=BACKEND.get(target, target))


def check_natively(tmp_path, target, code):
    """Run the target's checker when it is installed; a missing tool is not a failure here."""
    if target == "wgsl":
        if not shutil.which("naga"):
            return
        (tmp_path / "out.wgsl").write_text(code)
        checked = subprocess.run(
            ["naga", "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
        return
    path = tmp_path / f"out{SUFFIX[target]}"
    path.write_text(code)
    entry = HLSL_ENTRY.search(code).group(1) if target == "hlsl" else "main"
    results, _ = validate(path, target, tmp_path / "out.bin", entry)
    for result in results:
        assert result.status in ("accepted", "unavailable"), result.log


@pytest.mark.parametrize(
    "source",
    [FLAT, BLOCK, STD140, INITIALIZED, KEPT],
    ids=["flat", "block", "std140", "init", "kept"],
)
@pytest.mark.parametrize("target", TARGETS)
def test_storage_and_initialized_arrays_translate_to_checked_output(
    tmp_path, source, target
):
    if source is STD140 and target == "wgsl":
        # WGSL storage arrays have the std430 stride (WGSL section 14.4); the emitter refuses other layouts.
        with pytest.raises(ValueError, match="only supports std430"):
            translate(tmp_path, source, target)
        return
    code = translate(tmp_path, source, target)
    assert "[1024]" not in code
    assert "1024>" not in code
    assert "unsupported" not in code
    check_natively(tmp_path, target, code)


def test_flat_buffer_arrays_become_structured_buffers(tmp_path):
    hlsl = translate(tmp_path, FLAT, "hlsl")
    assert "StructuredBuffer<float> values : register(t1);" in hlsl
    assert "RWStructuredBuffer<float> outv : register(u2);" in hlsl
    assert re.search(r"RWStructuredBuffer<int2> pairs : register\(u\d\);", hlsl)
    assert "static" not in hlsl
    glsl = translate(tmp_path, FLAT, "glsl")
    assert "readonly buffer valuesBuffer { float values[]; };" in glsl
    assert "uniform float" not in glsl


def test_other_buffer_arrays_become_buffer_blocks(tmp_path):
    hlsl = translate(tmp_path, BLOCK, "hlsl")
    assert re.search(r"\bRWByteAddressBuffer dirsStorage : register\(u0\);", hlsl)
    assert re.search(r"\bByteAddressBuffer basesStorage : register\(t1\);", hlsl)
    assert "StructuredBuffer" not in hlsl
    glsl = translate(tmp_path, BLOCK, "glsl")
    assert "layout(std430, binding = 0) buffer DirsBlock { vec3 dirs[]; }" in glsl
    assert "vec3 d = dirsStorage.dirs[i];" in glsl
    assert "dirsStorage.dirs[i] = vec3(basesStorage.bases[i][1], 1.0);" in glsl
    assert "dirsStorage.dirs[(i + 1u)] = d.zyx;" in glsl


def test_std140_buffer_arrays_keep_the_std140_stride(tmp_path):
    # std140 rounds a float array's stride up to 16 bytes (GLSL 4.60 section 7.6.2.2, rule 4).
    hlsl = translate(tmp_path, STD140, "hlsl")
    assert "weightsStorage.Load((i * 16))" in hlsl
    glsl = translate(tmp_path, STD140, "glsl")
    assert re.search(
        r"layout\(std140, binding = 2\) readonly buffer WeightsBlock", glsl
    )
    spirv = translate(tmp_path, STD140, "spirv")
    assert re.search(r"OpDecorate %_runtimearr_float\w* ArrayStride 16", spirv)


def test_initializers_size_unsized_arrays(tmp_path):
    assert "float a[3] = {1.0, 2.0, 3.0};" in translate(tmp_path, INITIALIZED, "hlsl")
    assert "float c[2] = float[2](4.0, 5.0);" in translate(
        tmp_path, INITIALIZED, "glsl"
    )
    assert "float c[2] = {4.0, 5.0};" in translate(tmp_path, INITIALIZED, "hlsl")


@pytest.mark.parametrize("case", sorted(REFUSED))
@pytest.mark.parametrize(
    "backend",
    [
        "directx",
        "metal",
        "opengl",
        "vulkan",
        "wgsl",
        "slang",
        "cuda",
        "hip",
        "rust",
        "mojo",
    ],
)
def test_unsized_value_arrays_are_refused(tmp_path, case, backend):
    with pytest.raises(UnsizedArrayUnsupportedError) as raised:
        translate(tmp_path, REFUSED[case], backend)
    assert raised.value.code == f"project.translate.{backend}-unsized-array-unsupported"
    assert raised.value.missing_capabilities == (f"{backend}.unsized-value-array",)


FLEXIBLE_MEMBER_METAL = {
    # A thread-local struct ending in a flexible array member, read through a byte view or copied from a buffer.
    "private_view": (
        """
#include <metal_stdlib>
using namespace metal;
struct WordBlock { uint words[]; };
inline uint inspect(const thread uint8_t* values) { return values[0]; }
kernel void k(device uint* output [[buffer(0)]]) {
    thread WordBlock block;
    output[0] = inspect((const thread uint8_t*)&block);
}
"""
    ),
    "storage_copy": (
        """
#include <metal_stdlib>
using namespace metal;
struct WordBlock { uint words[]; };
kernel void k(const device uint* input [[buffer(0)]], device uint* output [[buffer(1)]]) {
    thread WordBlock local;
    local = *((const device WordBlock*)input);
    output[0] = uint(local.words[0]);
}
"""
    ),
}


@pytest.mark.parametrize("case", sorted(FLEXIBLE_MEMBER_METAL))
@pytest.mark.parametrize("backend", ["directx", "opengl"])
def test_metal_flexible_array_member_locals_are_refused(tmp_path, case, backend):
    # Before any pointer reinterpretation is considered: no target has a thread-local of unknown size.
    path = tmp_path / f"{case}.metal"
    path.write_text(FLEXIBLE_MEMBER_METAL[case])
    with pytest.raises(UnsizedArrayUnsupportedError) as raised:
        crosstl.translate(str(path), backend=backend, source_backend="metal")
    assert "WordBlock.words" in str(raised.value)


EMITTER_STRUCT = """
shader S {
    struct Globals { vec4 tint; float weights[]; };
    uniform Globals globals;
    fragment { vec4 main() @ gl_FragColor { return globals.tint * globals.weights[0]; } }
}
"""


@pytest.mark.parametrize(
    ("module", "name", "target"),
    [
        ("directx_codegen", "HLSLCodeGen", "directx"),
        ("metal_codegen", "MetalCodeGen", "metal"),
        ("GLSL_codegen", "GLSLCodeGen", "opengl"),
        ("slang_codegen", "SlangCodeGen", "slang"),
    ],
)
def test_emitters_refuse_unsized_value_members_without_the_hub_pass(
    module, name, target
):
    # The emitters gave such members 1024 elements when called directly.
    import importlib

    codegen = getattr(
        importlib.import_module(f"crosstl.translator.codegen.{module}"), name
    )
    with pytest.raises(UnsizedArrayUnsupportedError) as raised:
        codegen().generate(crosstl.translator.parse(EMITTER_STRUCT))
    assert raised.value.code == f"project.translate.{target}-unsized-array-unsupported"


def test_unused_undecorated_runtime_array_structs_are_dropped():
    from crosstl.translator.codegen.spirv_dead_types import (
        drop_unused_runtime_array_structs,
    )

    declarations = [
        "%1 = OpTypeFloat 32",
        "%2 = OpTypeRuntimeArray %1",
        "%3 = OpTypeStruct %2",  # the std430 declaration nothing uses
        "%4 = OpTypeRuntimeArray %1",
        "%5 = OpTypeStruct %4",  # the std140 block
        "%6 = OpTypeStruct %2",  # undecorated but used
        "%7 = OpTypePointer Function %6",
    ]
    names = ['OpName %3 "WB"', 'OpMemberName %3 0 "w"', 'OpName %5 "WB_std140"']
    annotations = [
        "OpDecorate %5 BufferBlock",
        "OpMemberDecorate %5 0 Offset 0",
        "OpDecorate %2 ArrayStride 4",
    ]
    kept_names, kept_annotations, kept = drop_unused_runtime_array_structs(
        names, annotations, declarations, ["%8 = OpVariable %7 Function"]
    )
    assert "%3 = OpTypeStruct %2" not in kept
    assert {
        "%5 = OpTypeStruct %4",
        "%6 = OpTypeStruct %2",
        "%2 = OpTypeRuntimeArray %1",
    } <= set(kept)
    assert kept_names == ['OpName %5 "WB_std140"']
    assert kept_annotations == annotations
