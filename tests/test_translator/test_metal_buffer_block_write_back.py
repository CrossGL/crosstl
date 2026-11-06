"""Metal stores to elements of a GLSL buffer block that the generator holds as bytes.

A buffer block whose layout MSL cannot reproduce becomes a ``device uchar*``; reads are ``const device`` scalar
loads and writes are ``device`` scalar stores. An ``inout`` or ``out`` argument in such a block is copied into a
thread temporary and stored back after the call, and the store-back has to take the block's store form, not the load
(MSL section 2.9: a ``const`` lvalue is not assignable). A component write to an element of an array of vectors
(``block.v[i].y = x``) stores the component's scalar, like a component write to a vector member.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import crosstl
from crosstl.translator.codegen.buffer_block_diagnostics import (
    BufferBlockUnsupportedError,
)
from tests.runtime_helpers import compile_metal

ROOT = Path(__file__).resolve().parents[2]
METAL_REQUIRE_ENV = "CROSTL_REQUIRE_METAL_BUFFER_WRITE_BACK"

WRITE_BACK = """shader S {
    struct P { int a; ivec2 b; };
    compute {
        layout(std430, binding = 0) buffer Seeds { int seed[]; } seeds;
        layout(std430, binding = 1) buffer Vecs { ivec4 v[2]; P p[]; } vecBuf;
        layout(std430, binding = 2) buffer Out { int values[]; } outBuf;
        layout(local_size_x = 1) in;
        int bump(inout int a) { int old = a; a = a + 1; return old * 10; }
        void reset(out ivec2 r) { r = ivec2(0, 9); }
        void main() {
            outBuf.values[7] = seeds.seed[0];
            vecBuf.v[1] = ivec4(1, 2, 3, 4);
            vecBuf.p[0].a = 40;
            vecBuf.p[0].b = ivec2(6, 8);
            int r0 = bump(outBuf.values[7]);
            int r1 = bump(vecBuf.v[1].y);
            int r2 = bump(vecBuf.p[0].a);
            int r3 = bump(vecBuf.p[0].b.x);
            reset(vecBuf.p[0].b);
            outBuf.values[0] = r0;
            outBuf.values[1] = vecBuf.v[1].y;
            outBuf.values[2] = r1;
            outBuf.values[3] = vecBuf.p[0].a;
            outBuf.values[4] = r2 + r3;
            outBuf.values[5] = vecBuf.p[0].b.x + vecBuf.p[0].b.y * 100;
            outBuf.values[6] = vecBuf.v[1].x + vecBuf.v[1].z + vecBuf.v[1].w;
        }
    }
}
"""
# bump(values[7]) with values[7] = 20; bump(v[1].y) with 2; bump(p[0].a) with 40; bump(p[0].b.x) with 6.
WRITE_BACK_RESULTS = [200, 3, 20, 41, 400 + 60, 900, 8, 21]


def translate(tmp_path, source):
    path = tmp_path / "kernel.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal", format_output=False)


def test_inout_buffer_element_is_stored_back_through_a_device_pointer(tmp_path):
    metal = translate(tmp_path, WRITE_BACK)
    assert "device uchar* outBuf" in metal or "device uchar *outBuf" in metal
    # Every store-back assigns through the block's writable form; none assigns to a const load.
    assert "const device int*>(outBuf + 28)) = " not in metal.replace(" *>", "*>")
    for line in metal.splitlines():
        if "__crossgl_ref" in line and line.strip().startswith("(*reinterpret_cast"):
            assert "const device" not in line, line
    assert "int r0 = bump(__crossgl_ref" in metal


def test_component_write_to_an_array_element_stores_the_scalar(tmp_path):
    source = """shader S {
    compute {
        layout(std430, binding = 0) buffer Vecs { ivec4 v[2]; int rest[]; } vecBuf;
        layout(local_size_x = 1) in;
        void main() {
            vecBuf.v[1].y = 5;
            vecBuf.v[0].zw += ivec2(2, 3);
        }
    }
}
"""
    metal = translate(tmp_path, source).replace(" *>", "*>")
    # v[1].y lies at 16 + 4 bytes; v[0].z and v[0].w at 8 and 12.
    assert "(*reinterpret_cast<device int*>(vecBuf + 20)) = (5)" in metal
    assert "(*reinterpret_cast<device int*>(vecBuf + 8)) = " in metal
    assert "(*reinterpret_cast<device int*>(vecBuf + 12)) = " in metal
    assert ").y = " not in metal


def test_inout_element_of_a_readonly_block_is_refused(tmp_path):
    source = """shader S {
    compute {
        layout(std430, binding = 0) readonly buffer In { int values[]; } inBuf;
        layout(std430, binding = 1) buffer Out { int values[]; } outBuf;
        layout(local_size_x = 1) in;
        int bump(inout int a) { int old = a; a = a + 1; return old; }
        void main() {
            outBuf.values[0] = bump(inBuf.values[0]);
        }
    }
}
"""
    with pytest.raises(BufferBlockUnsupportedError, match="readonly"):
        translate(tmp_path, source)


def _metal_library(tmp_path):
    if sys.platform != "darwin" or shutil.which("xcrun") is None:
        pytest.skip("the Metal compiler requires macOS with xcrun")
    artifact = tmp_path / "kernel.metal"
    artifact.write_text(translate(tmp_path, WRITE_BACK), encoding="utf-8")
    return compile_metal(artifact, ROOT, tmp_path / "metal")


def test_write_back_compiles_for_metal(tmp_path):
    assert _metal_library(tmp_path).is_file()


def test_write_back_reaches_the_buffers_on_metal(tmp_path):
    if os.environ.get(METAL_REQUIRE_ENV) != "1":
        pytest.skip(f"set {METAL_REQUIRE_ENV}=1 to require native Metal execution")
    library = _metal_library(tmp_path)
    runner = tmp_path / "readback"
    fixture = ROOT / "tests/fixtures/runtime_verification/metal_uint32_buffers.swift"
    subprocess.run(["swiftc", str(fixture), "-o", str(runner)], check=True, timeout=300)
    request = tmp_path / "request.json"
    # The runner binds the inputs in order and the output buffer last: seeds, vecBuf, then outBuf.
    request.write_text(
        json.dumps(
            {
                "inputs": [[20], [0] * 64],
                "outputCount": len(WRITE_BACK_RESULTS),
                "threadCount": 1,
            }
        )
    )
    readback = subprocess.run(
        [str(runner), str(library), "kernel_main", str(request)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    assert json.loads(readback)["values"] == WRITE_BACK_RESULTS
