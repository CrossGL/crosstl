"""HLSL output of ``buffer_load`` and ``buffer_store`` through storage pointer parameters.

A helper's ``device T* data`` parameter becomes an HLSL resource plus an element offset. The CrossGL buffer
operations on it mean ``data[i]`` and ``data[i] = v``, so they must index the resource like the subscript forms do;
written as resource methods they came out as ``data.Store(i, v)``, which no typed or structured HLSL buffer has.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.directx_pointer_alias_errors import (
    DirectXResourcePointerAliasError,
)

HELPERS = """
shader main {
    void put(device float* data, uint i, float v) {
        float old = buffer_load(data, i);
        buffer_store(data, i, old + v);
    }
    void putm(device mat2* data, uint i) {
        mat2 m = buffer_load(data, i);
        m[1] = m[0] * 2.0;
        buffer_store(data, i, m);
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main(RWStructuredBuffer<float> fin @buffer(0), RWStructuredBuffer<mat2> outp @buffer(1),
                  uvec3 id @gl_GlobalInvocationID) {
            uint i = id.x;
            put(fin, i, 1.0);
            putm(outp, i);
            buffer_store(fin, i + 1u, buffer_load(fin, i));
        }
    }
}
"""

ALIAS = """
shader main {
    void put(device float* data, uint i, float v) {
        device float* row = data + 4u;
        buffer_store(row, i, buffer_load(row, i + 1u) + v);
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main(RWStructuredBuffer<float> fin @buffer(0), uvec3 id @gl_GlobalInvocationID) {
            put(fin, id.x, 1.0);
        }
    }
}
"""

NARROW = """
shader main {
    void put(device float16* data, uint i, float16 v) {
        buffer_store(data, i, buffer_load(data, i + 1u) + v);
    }
    void putu(device uint8* data, uint i, uint8 v) {
        buffer_store(data, i, buffer_load(data, i + 1u) + v);
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main(RWStructuredBuffer<float16> h @buffer(0), RWStructuredBuffer<uint8> b @buffer(1),
                  uvec3 id @gl_GlobalInvocationID) {
            put(h, id.x, float16(1.0));
            putu(b, id.x, uint8(3));
        }
    }
}
"""

READ_ONLY = """
shader main {
    float get(const device float* data, uint i) {
        buffer_store(data, i, 1.0);
        return buffer_load(data, i);
    }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main(StructuredBuffer<float> fin @buffer(0), RWStructuredBuffer<float> o @buffer(1),
                  uvec3 id @gl_GlobalInvocationID) {
            buffer_store(o, id.x, get(fin, id.x));
        }
    }
}
"""

# A Metal compound assignment into a component of a matrix in a buffer: the Metal importer writes a helper that
# loads the element through the pointer, updates it and stores it back with buffer_load and buffer_store.
METAL_MATRIX_BUFFER = """
#include <metal_stdlib>
using namespace metal;

kernel void probe(device float* fin [[buffer(0)]], device float2x2* outp [[buffer(1)]],
                  uint3 id [[thread_position_in_grid]]) {
    uint i = id.x;
    outp[i] = float2x2(fin[i], 1.0, 2.0, fin[i] * 0.5);
    outp[i][0][1] += 0.25;
}
"""


def _translate(tmp_path, source, suffix=".cgl"):
    path = tmp_path / f"probe{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="directx")


def _dxc(tmp_path, code):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not on PATH")
    path = tmp_path / "probe.hlsl"
    path.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [dxc, "-T", "cs_6_2", "-enable-16bit-types", "-E", "CSMain", str(path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_helper_pointer_buffer_calls_index_the_resource(tmp_path):
    code = _translate(tmp_path, HELPERS)
    assert "data.Store(" not in code
    assert "data.Load(" not in code
    assert "float old = data[uint((data_offset + i))];" in code
    assert "data[uint((data_offset + i))] = (old + v);" in code
    assert "float2x2 m = data[uint((data_offset + i))];" in code
    assert "data[uint((data_offset + i))] = m;" in code
    _dxc(tmp_path, code)


def test_buffer_calls_on_resources_keep_their_forms(tmp_path):
    code = _translate(tmp_path, HELPERS)
    assert "fin[(i + 1u)] = fin.Load(i);" in code


def test_pointer_alias_buffer_calls_add_the_alias_offset(tmp_path):
    code = _translate(tmp_path, ALIAS)
    assert "int64_t row_offset = int64_t((data_offset + 4u));" in code
    assert (
        "data[uint((row_offset + i))] = (data[uint((row_offset + (i + 1u)))] + v);"
        in code
    )
    assert ".Store(" not in code
    _dxc(tmp_path, code)


def test_narrow_pointer_buffer_calls_match_the_subscript_forms(tmp_path):
    code = _translate(tmp_path, NARROW)
    subscript = _translate(
        tmp_path,
        NARROW.replace(
            "buffer_store(data, i, buffer_load(data, i + 1u) + v);",
            "data[i] = data[i + 1u] + v;",
        ),
    )
    assert code == subscript
    assert "data.Store(" not in code
    assert "& 255u);" in code
    _dxc(tmp_path, code)


def test_store_through_read_only_pointer_is_refused(tmp_path):
    with pytest.raises(DirectXResourcePointerAliasError, match="writable storage"):
        _translate(tmp_path, READ_ONLY)


def test_metal_matrix_component_update_in_buffer_compiles(tmp_path):
    code = _translate(tmp_path, METAL_MATRIX_BUFFER, suffix=".metal")
    assert ".Store(" not in code
    _dxc(tmp_path, code)
