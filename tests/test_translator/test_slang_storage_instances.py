"""CrossGL ``buffer S name`` storage instances and structure-member atomics in Slang output.

The Slang generator wrote ``buffer S name`` as ``S name;``, a uniform parameter, and refused ``atomicAdd(name.m, v)``
as a call to another language's built-in function. The declaration is now the std430 block the GLSL generator
writes for it, lowered to a byte-address buffer, so its members sit at their std430 offsets (GLSL 4.60 section
7.6.2.2). An atomic on an ``int`` or ``uint`` member of a structured-buffer element is now typed by the member, not
refused as an atomic on the element structure.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.slang_codegen import SlangCodeGen
from crosstl.translator.codegen.slang_refusals import SlangLoweringRefused
from crosstl.translator.codegen.slang_storage_instances import (
    lower_slang_storage_instances,
)
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

INSTANCES = """
shader Instances {
    struct Params { vec3 offset; float scale; int count; }
    struct Totals { int hits; vec4 color; uint misses; }
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        layout(std430, binding = 3) readonly buffer Params params;
        buffer Totals totals;
        void main() {
            int previous = atomicAdd(totals.hits, params.count);
            atomicMax(totals.misses, uint(previous));
            totals.color = vec4(params.offset * params.scale, 1.0);
        }
    }
}
"""

LARGE = """
shader Large {
    struct Cell { vec4 value; }
    struct Grid { Cell cells[1024]; int used; }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer Grid grid;
        void main() {
            grid.cells[3].value = vec4(1.0);
            atomicAdd(grid.used, 1);
        }
    }
}
"""

MEMBER_ATOMICS = """
shader MemberAtomics {
    struct Bin { int count; uint mask; float weight; }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer Bin bins[];
        void main() {
            int before = atomicAdd(bins[2].count, 1);
            atomicOr(bins[before].mask, 4u);
        }
    }
}
"""


def _translate(tmp_path, name, source):
    path = tmp_path / f"{name}.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="slang")


def _compact(text):
    return "".join(text.split())


def _slangc(tmp_path, name, code):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    shader = tmp_path / f"{name}.slang"
    shader.write_text(code)
    result = subprocess.run(
        [
            slangc,
            str(shader),
            "-target",
            "spirv",
            "-entry",
            "main",
            "-stage",
            "compute",
            "-o",
            str(tmp_path / f"{name}.spv"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_instances_are_byte_address_buffers_with_their_bindings(tmp_path):
    code = _translate(tmp_path, "instances", INSTANCES)
    compact = _compact(code)

    assert "[[vk::binding(3,0)]]ByteAddressBufferparams:register(t3);" in compact
    assert "RWByteAddressBuffertotals" in compact
    assert "Paramsparams;" not in compact
    assert "Totalstotals;" not in compact


def test_instance_members_sit_at_their_std430_offsets(tmp_path):
    compact = _compact(_translate(tmp_path, "instances", INSTANCES))

    # Params: vec3 offset at 0, float scale at 12, int count at 16.
    # Totals: int hits at 0, vec4 color at 16, uint misses at 32.
    assert (
        "__crossgl_slang_byteaddress_atomic_add_int(totals,0,asint(params.Load(16)))"
        in (compact)
    )
    assert "__crossgl_slang_byteaddress_atomic_max_uint(totals,32," in compact
    assert "totals.Store4(16," in compact
    assert "asfloat(params.Load(12))" in compact


def test_atomic_calls_are_not_written_verbatim(tmp_path):
    code = _translate(tmp_path, "instances", INSTANCES)

    assert "atomicAdd(" not in code
    assert "atomicMax(" not in code


def test_an_instance_larger_than_a_structured_buffer_element(tmp_path):
    # D3D limits a structured-buffer element to 2048 bytes; Grid is 16 KiB and 4 bytes.
    compact = _compact(_translate(tmp_path, "large", LARGE))

    assert "RWByteAddressBuffergrid" in compact
    assert "RWStructuredBuffer<Grid>" not in compact
    assert "__crossgl_slang_byteaddress_atomic_add_int(grid,16384,1)" in compact


def test_the_lowering_leaves_the_input_program_unchanged():
    program = Parser(Lexer(INSTANCES).get_tokens()).parse()
    stage = next(iter(program.stages.values()))
    before = [
        (list(node.qualifiers), [a.name for a in node.attributes])
        for node in stage.local_variables
    ]

    lowered = lower_slang_storage_instances(program)

    assert lowered is not program
    assert [
        (list(node.qualifiers), [a.name for a in node.attributes])
        for node in stage.local_variables
    ] == before
    lowered_stage = next(iter(lowered.stages.values()))
    assert [node.attributes[0].name for node in lowered_stage.local_variables] == [
        "glsl_buffer_block",
        "glsl_buffer_block",
    ]
    assert all(
        "buffer" not in node.qualifiers for node in lowered_stage.local_variables
    )
    SlangCodeGen().generate(program)


def test_an_element_array_stays_a_structured_buffer(tmp_path):
    compact = _compact(_translate(tmp_path, "member_atomics", MEMBER_ATOMICS))

    assert "RWStructuredBuffer<Bin>bins" in compact


def test_member_atomics_of_structured_buffer_elements(tmp_path):
    compact = _compact(_translate(tmp_path, "member_atomics", MEMBER_ATOMICS))

    assert (
        "intcgl_atomicAdd_original;InterlockedAdd(bins[2].count,1,cgl_atomicAdd_original);"
        in (compact)
    )
    assert "InterlockedOr(bins[before].mask,4u," in compact


def test_a_float_member_atomic_is_still_refused(tmp_path):
    source = MEMBER_ATOMICS.replace(
        "atomicOr(bins[before].mask, 4u);", "atomicAdd(bins[before].weight, 1.0);"
    )

    with pytest.raises(SlangLoweringRefused, match="int or uint"):
        _translate(tmp_path, "float_atomic", source)


@pytest.mark.parametrize(
    "name,source",
    [("instances", INSTANCES), ("large", LARGE), ("member_atomics", MEMBER_ATOMICS)],
)
def test_slangc_compiles_the_output(tmp_path, name, source):
    _slangc(tmp_path, name, _translate(tmp_path, name, source))
