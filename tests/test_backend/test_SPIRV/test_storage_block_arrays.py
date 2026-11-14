"""Arrays of SPIR-V storage blocks keep their declaration through the importer.

GLSL's ``buffer B { ... } bufs[N]`` compiles to one SPIR-V variable whose pointee is an array of the
``Block``-decorated struct (SPIR-V 1.6 section 2.16.2). The importer looked for the struct directly behind the
pointer, so it dropped the declaration and wrote ``bufs[k].data[i]`` against an undeclared name. It now declares
``RWStructuredBuffer<B> bufs[N]``, accesses element ``k`` as ``bufs[k][0]``, and the SPIR-V storage block pass
rewrites that into the GLSL importer's forms. Arrays of uniform blocks are refused instead of dropped.

The fixtures are glslang output; each starts with the GLSL it was compiled from.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import crosstl
from crosstl.backend.SPIRV.block_arrays import SpirvBlockArrayUnsupportedError
from tests.conformance.validators import validate

FIXTURES = Path(__file__).resolve().parent / "storage_block_arrays"
CONSTANT = FIXTURES / "constant_indexes.spvasm"  # Uniform + BufferBlock
DYNAMIC = FIXTURES / "dynamic_indexes.spvasm"  # StorageBuffer + Block
HEADER = FIXTURES / "header_blocks.spvasm"  # { uint count; float data[]; } bufs[2]
UNIFORM = FIXTURES / "uniform_blocks.spvasm"

HLSL_ENTRY = re.compile(r"\[numthreads\([^)]*\)\]\s*void\s+(\w+)")
SUFFIX = {"directx": ".hlsl", "vulkan": ".spvasm", "opengl": ".comp"}
CHECKER = {"directx": "hlsl", "vulkan": "spirv", "opengl": "glsl"}


def translate(source, backend):
    return crosstl.translate(str(source), backend=backend, format_output=False)


def check_natively(tmp_path, backend, code):
    """Run the target's checker when it is installed; a missing tool is not a failure here."""
    path = tmp_path / f"out{SUFFIX[backend]}"
    path.write_text(code, encoding="utf-8")
    entry = HLSL_ENTRY.search(code).group(1) if backend == "directx" else "main"
    results, _ = validate(path, CHECKER[backend], tmp_path / "out.bin", entry)
    for result in results:
        assert result.status in ("accepted", "unavailable"), result.log


def test_constant_indexes_declare_the_block_array():
    crossgl = translate(CONSTANT, "cgl")
    assert "RWStructuredBuffer<Block> bufs[3] @set(0) @binding(0);" in crossgl
    assert "bufs[0][0].data[i]" in crossgl
    assert "bufs[2][0].data[i]" in crossgl
    assert "spirvArrayLength(bufs[1], 0)" in crossgl
    assert "bufs[0].data" not in crossgl


def test_dynamic_indexes_declare_read_only_and_writable_arrays():
    crossgl = translate(DYNAMIC, "cgl")
    assert "RWStructuredBuffer<Block> bufs[4] @set(0) @binding(0);" in crossgl
    assert "StructuredBuffer<In> ins[2] @set(0) @binding(4);" in crossgl
    assert "atomicAdd(bufs[k][0].data[0]" in crossgl
    assert "ins[(i & 1)][0].values[i]" in crossgl
    assert "spirvArrayLength(bufs[(3 - k)], 0)" in crossgl


def test_single_member_block_arrays_become_structured_buffer_arrays():
    hlsl = translate(DYNAMIC, "directx")
    assert "RWStructuredBuffer<uint> bufs[4] : register(u0);" in hlsl
    assert "StructuredBuffer<float> ins[2] : register(t4);" in hlsl
    assert "InterlockedAdd(bufs[k][0]," in hlsl
    assert "bufs[k][i] = (bufs[k][i] + " in hlsl
    assert "ins[(i & 1)][i]" in hlsl
    assert "cgl_buffer_count(bufs[(3 - k)])" in hlsl
    assert "struct Block" not in hlsl
    assert "spirvArrayLength" not in hlsl


def test_header_block_arrays_become_buffer_block_arrays():
    hlsl = translate(HEADER, "directx")
    assert "RWByteAddressBuffer bufs[2] : register(u0);" in hlsl
    assert "bufs[1].Load(0)" in hlsl
    glsl = translate(HEADER, "opengl")
    assert "} bufs[2];" in glsl
    assert "bufs[(i & 1)].data[i]" in glsl
    assert "bufs[1].count" in glsl
    assert "bufs[0].data.length()" in glsl


@pytest.mark.parametrize("backend", ["directx", "vulkan"])
@pytest.mark.parametrize("source", [CONSTANT, DYNAMIC, HEADER])
def test_block_arrays_translate_to_checked_output(tmp_path, source, backend):
    code = translate(source, backend)
    assert "spirvArrayLength" not in code
    check_natively(tmp_path, backend, code)


def test_header_block_arrays_translate_to_checked_glsl(tmp_path):
    check_natively(tmp_path, "opengl", translate(HEADER, "opengl"))


def test_uniform_block_arrays_are_refused():
    with pytest.raises(SpirvBlockArrayUnsupportedError, match="'ubos'.*Uniform blocks"):
        translate(UNIFORM, "cgl")
