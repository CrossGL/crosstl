"""Metal helper functions reach threadgroup storage through ``threadgroup`` reference parameters.

Metal declares threadgroup memory only in a kernel, object or mesh function (Metal Shading Language Specification
4.4), so a stage-block ``shared`` variable or an HLSL program-scope ``groupshared`` variable that a helper reads is
declared in the entry and passed to every helper that reaches it, directly or through calls.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

STAGE_BLOCK = """shader S {
    compute {
        layout(std430, binding = 0) buffer Seeds { int seed[]; } seeds;
        layout(std430, binding = 1) buffer Out { int values[]; } outBuf;
        layout(local_size_x = 1) in;
        shared int tile[4];
        shared int total;
        int poke(int k) { tile[1] = tile[1] + 1000; return k; }
        int pair(int k, inout int t) { t = t + k; return t; }
        int resetTile(inout int t) { tile[2] = 77; t = t + 1; return t; }
        void addTotal(int k) { total = total + k; }
        int nested(int k) { addTotal(k); return total; }
        int step(inout int idx, inout int slot) { idx = idx + 1; slot = slot + 100; return idx; }
        void main() {
            int base = seeds.seed[0];
            tile[0] = base;
            tile[1] = base;
            tile[2] = 2 * base;
            tile[3] = 0;
            total = 10;
            int r2 = pair(poke(5), tile[1]);
            int r4 = resetTile(tile[2]);
            int r5 = nested(7);
            int q = 0;
            int r6 = step(q, tile[q]);
            outBuf.values[0] = r2;
            outBuf.values[1] = tile[1];
            outBuf.values[2] = r4;
            outBuf.values[3] = tile[2];
            outBuf.values[4] = r5;
            outBuf.values[5] = total;
            outBuf.values[6] = q * 1000 + r6;
            outBuf.values[7] = tile[0];
        }
    }
}
"""

PROGRAM_SCOPE = """StructuredBuffer<int> seeds : register(t0);
RWStructuredBuffer<int> outBuf : register(u0);
groupshared int tile[4];
groupshared int total;
void bump(int k) { tile[k] = tile[k] + 1; }
int sumTile() { int s = 0; for (int i = 0; i < 4; i++) { s += tile[i]; } return s; }
int addTotal(int k) { total += k; return sumTile() + total; }
[numthreads(4, 1, 1)]
void main(uint3 tid : SV_GroupThreadID) {
    tile[tid.x] = seeds[0] * (int)tid.x;
    if (tid.x == 0) { total = 100; }
    GroupMemoryBarrierWithGroupSync();
    if (tid.x == 0) { bump(2); }
    GroupMemoryBarrierWithGroupSync();
    int r = 0;
    if (tid.x == 0) { r = addTotal(5); }
    GroupMemoryBarrierWithGroupSync();
    outBuf[tid.x] = tile[tid.x] * 1000 + r;
}
"""

ATOMIC = """RWStructuredBuffer<uint> outBuf : register(u0);
groupshared uint counter;
uint add(uint k) { uint old; InterlockedAdd(counter, k, old); return old; }
[numthreads(4, 1, 1)]
void main(uint3 tid : SV_GroupThreadID) {
    if (tid.x == 0) { counter = 0; }
    GroupMemoryBarrierWithGroupSync();
    add(tid.x + 1);
    GroupMemoryBarrierWithGroupSync();
    outBuf[tid.x] = counter;
}
"""

SHADOWED = """shader S {
    compute {
        layout(std430, binding = 0) buffer Out { int values[]; } outBuf;
        layout(local_size_x = 1) in;
        shared int tile[4];
        int poke(int k) { tile[1] = 5; return k; }
        int outer(int k) { int tile = 3; return poke(k) + tile; }
        void main() { tile[1] = 0; outBuf.values[0] = outer(1) + tile[1]; }
    }
}
"""

UNRELATED_LOCAL = """shader S {
    compute {
        layout(std430, binding = 0) buffer Out { int values[]; } outBuf;
        layout(local_size_x = 1) in;
        shared int tile[4];
        int peek(int k) { int tile = 3; return tile + k; }
        int poke(int k) { tile[1] = 5; return peek(k); }
        void main() { outBuf.values[0] = poke(1); }
    }
}
"""

SUFFIXES = {"cgl": "shader.cgl", "hlsl": "shader.hlsl"}


def _translate(tmp_path, source, language="cgl"):
    path = tmp_path / SUFFIXES[language]
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _kernel(code):
    return code[code.index("kernel void") :]


def test_stage_block_shared_storage_is_a_threadgroup_reference_parameter(tmp_path):
    code = _translate(tmp_path, STAGE_BLOCK)
    assert "int poke(int k, threadgroup int (&tile)[4])" in code
    assert "int resetTile(thread int& t, threadgroup int (&tile)[4])" in code
    assert "void addTotal(int k, threadgroup int& total)" in code
    # A helper that reaches the storage only through a call receives it as well.
    assert "int nested(int k, threadgroup int& total)" in code
    assert "addTotal(k, total);" in code
    kernel = _kernel(code)
    assert "threadgroup int tile[4];" in kernel
    assert "threadgroup int total;" in kernel
    assert "nested(7, total)" in kernel
    # Nothing outside the kernel declares threadgroup storage.
    assert not re.search(r"^threadgroup ", code, re.MULTILINE)


def test_program_scope_groupshared_is_declared_in_the_kernel(tmp_path):
    code = _translate(tmp_path, PROGRAM_SCOPE, "hlsl")
    assert "void bump(int k, threadgroup int (&tile)[4])" in code
    assert "int sumTile(threadgroup int (&tile)[4])" in code
    assert (
        "int addTotal(int k, threadgroup int (&tile)[4], threadgroup int& total)"
        in code
    )
    assert "sumTile(tile)" in code
    kernel = _kernel(code)
    assert "threadgroup int tile[4];" in kernel
    assert "bump(2, tile);" in kernel
    assert "addTotal(5, tile, total)" in kernel
    assert not re.search(r"^threadgroup ", code, re.MULTILINE)


def test_atomic_on_forwarded_storage_keeps_the_threadgroup_address_space(tmp_path):
    code = _translate(tmp_path, ATOMIC, "hlsl")
    assert "uint add(uint k, threadgroup uint& counter)" in code
    assert "reinterpret_cast<threadgroup atomic_uint*>(&counter)" in code


def test_forwarded_reference_argument_is_copied_after_earlier_arguments(tmp_path):
    kernel = _kernel(_translate(tmp_path, STAGE_BLOCK))
    # poke() writes tile[1] before pair() copies tile[1] in (left-to-right evaluation).
    hoisted = kernel.index("int __crossgl_arg1 = poke(5, tile);")
    copy_in = kernel.index("int __crossgl_ref2 = tile[1];")
    call = kernel.index("int r2 = pair(__crossgl_arg1, __crossgl_ref2);")
    copy_out = kernel.index("tile[1] = __crossgl_ref2;")
    assert hoisted < copy_in < call < copy_out
    # The copy-out of an inout argument follows the callee's own write of the same element.
    assert (
        "int r4 = resetTile(__crossgl_ref3, tile);\n    tile[2] = __crossgl_ref3;"
        in kernel
    )


def test_index_written_by_the_call_is_evaluated_once(tmp_path):
    kernel = _kernel(_translate(tmp_path, STAGE_BLOCK))
    assert "int __crossgl_index4 = q;" in kernel
    assert "int __crossgl_ref5 = tile[__crossgl_index4];" in kernel
    assert "int r6 = step_fn(q, __crossgl_ref5);" in kernel
    assert "tile[__crossgl_index4] = __crossgl_ref5;" in kernel


COMPONENT_INDICES = """shader S {
    compute {
        layout(std430, binding = 0) buffer Out { int values[]; } outBuf;
        layout(local_size_x = 1) in;
        shared int tile[4];
        int stepv(inout ivec2 c, inout int t) { c.y = c.y + 1; t = t + 10; return t; }
        int addTo(inout int t, int k) { t = t + k; return t; }
        void main() {
            ivec2 c = ivec2(0, 1);
            ivec2 d = ivec2(2, 3);
            int r = stepv(c, tile[c.y]);
            int s = addTo(tile[d.x], c.y);
            outBuf.values[0] = r + s;
        }
    }
}
"""


def test_component_index_is_captured_only_when_the_call_writes_its_vector(tmp_path):
    kernel = _kernel(_translate(tmp_path, COMPONENT_INDICES))
    # stepv() writes c, so tile[c.y] is copied back to the element chosen at the call.
    assert "int __crossgl_index1 = c.y;" in kernel
    assert "tile[__crossgl_index1] = __crossgl_ref2;" in kernel
    # Nothing in the second call writes d, so d.x is read where it stands.
    assert "int __crossgl_ref3 = tile[d.x];" in kernel
    assert "tile[d.x] = __crossgl_ref3;" in kernel


def test_helper_local_of_the_same_name_that_does_not_reach_the_storage(tmp_path):
    code = _translate(tmp_path, UNRELATED_LOCAL)
    assert "int peek(int k) {" in code
    assert "int poke(int k, threadgroup int (&tile)[4])" in code


def test_helper_that_shadows_forwarded_storage_is_refused(tmp_path):
    with pytest.raises(Exception, match="declares its own 'tile'"):
        _translate(tmp_path, SHADOWED)


def test_copy_after_an_effectful_argument_inside_an_expression_is_refused(tmp_path):
    source = STAGE_BLOCK.replace(
        "int r2 = pair(poke(5), tile[1]);", "int r2 = 1 + pair(poke(5), tile[1]);"
    )
    with pytest.raises(ValueError, match="after an argument with side effects"):
        _translate(tmp_path, source)


def _metal_compiler():
    if shutil.which("xcrun") is None:
        pytest.skip("the Metal toolchain is not installed")
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    if found.returncode != 0:
        pytest.skip("the Metal toolchain is not installed")


@pytest.mark.parametrize(
    "source, language",
    [
        (STAGE_BLOCK, "cgl"),
        (PROGRAM_SCOPE, "hlsl"),
        (ATOMIC, "hlsl"),
        (UNRELATED_LOCAL, "cgl"),
    ],
    ids=["stage-block", "program-scope", "atomic", "unrelated-local"],
)
def test_forwarded_storage_compiles(tmp_path, source, language):
    code = _translate(tmp_path, source, language)
    _metal_compiler()
    artifact = tmp_path / "out.metal"
    artifact.write_text(code)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-Werror", "-c", str(artifact)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
