"""SPIR-V storage blocks and runtime-array lengths reach every target in a form its checker accepts.

The SPIR-V importer writes a storage block as a one-element structured buffer of the block struct, accessed through
``v[0]``, and ``OpArrayLength`` as ``spirvArrayLength(v, k)``. The DirectX emitter gave the runtime array 1024
elements, which DXC rejects, and no emitter knew ``spirvArrayLength``. Separately, HLSL, Metal, WGSL and SPIR-V
printed GLSL's ``block.member.length()`` or the hub's ``buffer_dimensions(b)`` unchanged. Execution evidence is in
the ``g_buffer_length``, ``g_buffer_count``, ``s_buffer_length`` and ``s_buffer_count`` conformance cases.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.codegen.buffer_length import BufferLengthUnsupportedError
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.semantics import (
    SpirvArrayLengthUnsupportedError,
    normalize_spirv_storage_blocks,
)
from tests.conformance.validators import validate

CASES = Path(__file__).resolve().parents[1] / "conformance" / "cases"
COUNT = CASES / "s_buffer_count" / "src.spvasm"  # block { int a[]; }
LENGTH = (
    CASES / "s_buffer_length" / "src.spvasm"
)  # block { int head; ivec2 pad; int a[]; }
GLSL_LENGTH = CASES / "g_buffer_length" / "src.comp"
GLSL_COUNT = CASES / "g_buffer_count" / "src.comp"
WRAP = CASES / "s_add_wrap" / "src.spvasm"  # blocks without a length query

HLSL_ENTRY = re.compile(r"\[numthreads\([^)]*\)\]\s*void\s+(\w+)")
SUFFIX = {"hlsl": ".hlsl", "glsl": ".comp", "spirv": ".spvasm", "metal": ".metal"}
BACKEND = {"hlsl": "directx", "glsl": "opengl", "spirv": "vulkan", "metal": "metal"}


def translate(source, target):
    return crosstl.translate(str(source), backend=BACKEND.get(target, target))


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


@pytest.mark.parametrize("source", [COUNT, LENGTH, WRAP])
@pytest.mark.parametrize("target", ["hlsl", "glsl", "spirv", "wgsl"])
def test_spirv_storage_blocks_translate_to_checked_output(tmp_path, source, target):
    code = translate(source, target)
    assert "spirvArrayLength" not in code
    assert "[1024]" not in code
    check_natively(tmp_path, target, code)


def test_single_array_block_becomes_a_structured_buffer_of_its_element():
    hlsl = translate(COUNT, "hlsl")
    assert re.search(r"\bStructuredBuffer<int> inp\b", hlsl)
    assert "inp[0]" not in hlsl
    assert "uint cgl_buffer_count(StructuredBuffer<int> b)" in hlsl
    assert "struct Inp" not in hlsl


def test_block_with_header_becomes_a_std430_byte_address_buffer():
    hlsl = translate(LENGTH, "hlsl")
    assert re.search(r"\bByteAddressBuffer inp\b", hlsl)
    # The runtime array starts after int head (0) and ivec2 pad (8): byte 16, stride 4.
    assert "int((cgl_buffer_bytes(inp) - 16u) / 4u)" in hlsl
    assert "inp.Load((16 + " in hlsl


def test_glsl_block_lengths_reach_hlsl_spirv_and_wgsl(tmp_path):
    for source in (GLSL_LENGTH, GLSL_COUNT):
        for target in ("hlsl", "spirv", "wgsl"):
            code = translate(source, target)
            assert ".length()" not in code
            check_natively(tmp_path, target, code)
    # Unformatted output keeps numeric ids whether or not SPIRV-Tools is installed to add friendly names.
    raw = crosstl.translate(str(GLSL_LENGTH), backend="vulkan", format_output=False)
    inp = re.search(r'^OpName %(\w+) "inp"$', raw, re.M).group(1)
    uint = re.search(r"^%(\w+) = OpTypeInt 32 0$", raw, re.M).group(1)
    assert re.search(rf"= OpArrayLength %{uint} %{inp} 2$", raw, re.M)
    assert "i32(arrayLength(&inp.a))" in translate(GLSL_LENGTH, "wgsl")


@pytest.mark.parametrize("source", [LENGTH, GLSL_LENGTH])
def test_metal_refuses_runtime_array_length(source):
    # MSL buffer arguments are pointers and carry no size (MSL 3.2 section 2.8).
    with pytest.raises(BufferLengthUnsupportedError) as raised:
        translate(source, "metal")
    assert raised.value.code == "project.translate.metal-buffer-length-unsupported"


def test_metal_translates_blocks_without_length_queries(tmp_path):
    check_natively(tmp_path, "metal", translate(WRAP, "metal"))


def spirv_like(body, declaration="RWStructuredBuffer<Block> v @set(0) @binding(0);"):
    return parse(f"""
        shader S {{
            struct Block {{ uint head; int a[]; }};
            {declaration}
            compute {{
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                void main() {{ {body} }}
            }}
        }}
        """)


def test_other_uses_leave_the_variable_unchanged():
    program = spirv_like("v[1].head = 0u;")
    assert not normalize_spirv_storage_blocks(program, "vulkan", "directx")
    assert program.global_variables[0].var_type.name == "RWStructuredBuffer"


def test_only_spirv_sources_are_rewritten():
    program = spirv_like("v[0].head = 0u;")
    assert not normalize_spirv_storage_blocks(program, "opengl", "directx")
    assert normalize_spirv_storage_blocks(program, "vulkan", "directx")
    variable = program.global_variables[0]
    assert variable.var_type.name == "Block"
    assert [a.name for a in variable.attributes][0] == "glsl_buffer_block"


def test_array_length_of_anything_else_is_refused():
    program = spirv_like(
        "uint n = spirvArrayLength(w, 0);", "RWStructuredBuffer<int> w @binding(0);"
    )
    with pytest.raises(SpirvArrayLengthUnsupportedError) as raised:
        normalize_spirv_storage_blocks(program, "vulkan", "metal")
    assert raised.value.code == "project.translate.metal-spirv-array-length-unsupported"


def test_hlsl_buffer_dimensions_of_a_byte_address_buffer_is_refused():
    program = parse("""
        shader S {
            ByteAddressBuffer raw @binding(0);
            RWStructuredBuffer<uint> o @binding(1);
            compute {
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                void main() { o[0] = buffer_dimensions(raw); }
            }
        }
        """)
    with pytest.raises(BufferLengthUnsupportedError, match="element count"):
        HLSLCodeGen().generate(program)
