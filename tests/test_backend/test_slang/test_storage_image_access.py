"""Slang storage-texture subscripts and ``Interlocked*`` atomics import as CrossGL image operations.

``image[coord]`` reads become ``imageLoad`` narrowed to the declared element
type, writes become ``imageStore`` of the value widened to four components,
compound writes, increments and component writes load the texel first, and
``Interlocked*`` becomes ``imageAtomic*`` on a texel and ``atomic*`` on any
other location. Every target compiles the result; the writes CrossGL cannot
express are refused.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.import_refusals import SlangImportRefused

TEXEL_ACCESS = """[format("r32ui")] RWTexture2D<uint> counters;
[format("rgba16f")] RWTexture2D<float4> colors;
[format("rg32f")] RWTexture2D<float2> pairs;
[format("rg32i")] RWTexture2DArray<int2> layers;
RWStructuredBuffer<uint> totals;
groupshared uint sharedCount;

void bump(RWTexture2D<uint> image, uint2 p) { InterlockedAdd(image[p], 1u); }

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    float4 c = colors[id.xy];
    colors[id.xy] = c * 0.5;
    colors[id.xy] += float4(1.0, 0.0, 0.0, 0.0);
    colors[id.xy].x = 2.0;
    colors[id.xy].zw *= 0.25;
    float2 p = pairs[id.xy];
    pairs[id.xy] = p.yx + pairs[id.xy].x;
    float2 q = pairs.Load(int2(id.xy));
    pairs.Store(int2(id.xy), q.x);
    layers[id] = layers[id] + int2(1, 2);
    layers[id].y += 3;
    counters[id.xy]++;
    uint original;
    InterlockedAdd(counters[id.xy], 1u, original);
    InterlockedMax(totals[0], original);
    InterlockedOr(sharedCount, 2u);
    uint prior;
    InterlockedCompareExchange(counters[id.xy], 0u, 5u, prior);
    InterlockedExchange(totals[1], prior, original);
    bump(counters, id.xy);
}
"""

EXPECTED_STATEMENTS = (
    "imageAtomicAdd(image, p, 1u);",
    "vec4 c = imageLoad(colors, id.xy);",
    "imageStore(colors, id.xy, c * 0.5);",
    "imageStore(colors, id.xy, imageLoad(colors, id.xy) + vec4(1.0, 0.0, 0.0, 0.0));",
    "vec4 _texel0 = imageLoad(colors, id.xy);",
    "_texel0.x = 2.0;",
    "imageStore(colors, id.xy, _texel0);",
    "_texel1.zw *= 0.25;",
    "vec2 p = imageLoad(pairs, id.xy).xy;",
    "imageStore(pairs, id.xy, vec4(p.yx + imageLoad(pairs, id.xy).x, 0.0, 1.0));",
    "vec2 q = imageLoad(pairs, ivec2(id.xy)).xy;",
    "imageStore(pairs, ivec2(id.xy), vec4(vec2(q.x), 0.0, 1.0));",
    "imageStore(layers, id, ivec4(ivec2(imageLoad(layers, id).xy + ivec2(1, 2)), 0, 1));",
    "_texel2.y += 3;",
    "imageStore(counters, id.xy, uvec4(imageLoad(counters, id.xy).x + 1u, 0u, 0u, 1u));",
    "original = imageAtomicAdd(counters, id.xy, 1u);",
    "atomicMax(totals[0], original);",
    "atomicOr(sharedCount, 2u);",
    "prior = imageAtomicCompSwap(counters, id.xy, 0u, 5u);",
    "original = atomicExchange(totals[1], prior);",
)

COMPUTE_PREFIX = """[format("r32ui")] RWTexture2D<uint> counters;
[format("rgba16f")] RWTexture2D<float4> colors;
RWTexture2D<float2x2> matrices;
RWStructuredBuffer<uint> totals;
[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
"""


def _write(tmp_path, source, suffix=".slang"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source, encoding="utf-8")
    return str(path)


def _translate(tmp_path, source, backend, suffix=".slang"):
    return crosstl.translate(
        _write(tmp_path, source, suffix), backend=backend, format_output=False
    )


def _run(tmp_path, command):
    result = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, check=False, timeout=180
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]


def _slangc():
    return shutil.which("slangc")


def _compile(tmp_path, backend, code):
    """Compile translated ``code`` with the target's toolchain; skip when it is missing."""
    if backend == "opengl":
        tool = shutil.which("glslangValidator")
        if tool is None:
            pytest.skip("glslangValidator is not installed")
        (tmp_path / "out.comp").write_text(code, encoding="utf-8")
        _run(tmp_path, [tool, "out.comp"])
    elif backend == "vulkan":
        assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
        if assembler is None or validator is None:
            pytest.skip("SPIRV-Tools is not installed")
        (tmp_path / "out.spvasm").write_text(code, encoding="utf-8")
        _run(
            tmp_path,
            [assembler, "--target-env", "vulkan1.0", "out.spvasm", "-o", "out.spv"],
        )
        _run(tmp_path, [validator, "--target-env", "vulkan1.0", "out.spv"])
    elif backend == "metal":
        if shutil.which("xcrun") is None:
            pytest.skip("the Metal toolchain is not installed")
        (tmp_path / "out.metal").write_text(code, encoding="utf-8")
        _run(
            tmp_path,
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-Werror",
                "-c",
                "out.metal",
                "-o",
                "out.air",
            ],
        )
    elif backend == "directx":
        dxc = shutil.which("dxc")
        if dxc is None:
            pytest.skip("dxc is not installed")
        (tmp_path / "out.hlsl").write_text(code, encoding="utf-8")
        _run(tmp_path, [dxc, "-T", "cs_6_0", "-E", "CSMain", "out.hlsl"])
    else:
        slangc = _slangc()
        if slangc is None:
            pytest.skip("slangc is not installed")
        (tmp_path / "out.slang").write_text(code, encoding="utf-8")
        _run(
            tmp_path,
            [
                slangc,
                "out.slang",
                "-target",
                "spirv",
                "-entry",
                "CSMain",
                "-stage",
                "compute",
            ]
            + ["-o", "out.spv"],
        )


def test_texel_subscripts_and_interlocked_calls_import_as_image_operations(tmp_path):
    crossgl = _translate(tmp_path, TEXEL_ACCESS, "cgl")
    statements = {line.strip() for line in crossgl.splitlines()}

    missing = [
        statement for statement in EXPECTED_STATEMENTS if statement not in statements
    ]
    assert not missing, (missing, crossgl)
    assert "Interlocked" not in crossgl
    assert "]++" not in crossgl


def test_the_source_compiles_with_slangc(tmp_path):
    slangc = _slangc()
    if slangc is None:
        pytest.skip("slangc is not installed")
    _write(tmp_path, TEXEL_ACCESS)
    _run(
        tmp_path,
        [
            slangc,
            "shader.slang",
            "-target",
            "spirv",
            "-entry",
            "CSMain",
            "-stage",
            "compute",
        ]
        + ["-o", "source.spv"],
    )


@pytest.mark.parametrize("backend", ["opengl", "vulkan", "metal", "directx", "slang"])
def test_imported_texel_access_compiles_for_every_target(tmp_path, backend):
    _compile(tmp_path, backend, _translate(tmp_path, TEXEL_ACCESS, backend))


def test_byte_address_interlocked_methods_import_as_offset_atomics(tmp_path):
    source = """RWByteAddressBuffer raw;
[numthreads(1, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint prior;
    raw.InterlockedAdd(4, 1u, prior);
    raw.InterlockedMax(8, prior);
    raw.InterlockedCompareExchange(0, 1u, 2u, prior);
}
"""
    crossgl = _translate(tmp_path, source, "cgl")
    statements = {line.strip() for line in crossgl.splitlines()}

    for statement in (
        "prior = atomicAdd(raw, 4, 1u);",
        "atomicMax(raw, 8, prior);",
        "prior = atomicCompareExchange(raw, 0, 1u, 2u);",
    ):
        assert statement in statements, crossgl


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("uint x = (counters[id.xy] = 1u);", "a texel written inside an expression"),
        ("uint y = counters[id.xy]++;", "a texel written inside an expression"),
        (
            "InterlockedAdd(colors[id.xy], 1.0);",
            "needs a texture of int or uint scalars",
        ),
        ("InterlockedAdd(counters[id.xy]);", "takes 2 or 3 arguments, not 1"),
        ("uint i = 0; counters[uint2(i++, 0)] += 1u;", "must have no side effects"),
        ("uint i = 0; counters[uint2(i++, 0)]++;", "must have no side effects"),
        ("matrices[id.xy] = float2x2(1, 0, 0, 1);", "no texels of type"),
        (
            "InterlockedCompareStoreFloatBitwise(totals[0], 1.0, 2.0);",
            "has no CrossGL atomic operation",
        ),
    ],
)
def test_texel_writes_crossgl_cannot_express_are_refused(tmp_path, body, reason):
    with pytest.raises(SlangImportRefused, match=reason):
        _translate(tmp_path, COMPUTE_PREFIX + f"    {body}\n}}\n", "cgl")
