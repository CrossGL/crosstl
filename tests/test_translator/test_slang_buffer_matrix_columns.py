"""Slang reads and writes a storage-buffer matrix column by column.

A storage block lowered to a ``RWByteAddressBuffer`` holds each matrix as consecutive columns (GLSL 4.60 section
7.6.2.2, std430). The Slang generator declares CrossGL ``matCxR`` as ``floatRxC`` with the same rows, as the HLSL
generator does, so ``m[i]`` is a row. It built a loaded matrix as ``floatRxC(column0, column1, ...)``, but a Slang
matrix constructor fills rows: a square matrix came back transposed, and a non-square one did not compile (slangc:
"expected an expression of type 'matrix<float,2,2>'"). It stored ``value[i]``, a row, into column ``i``. Loads now
build ``transpose(floatCxR(columns...))`` and stores select each column with a ``_mRC`` swizzle, matching the HLSL
generator, whose output DXC runs correctly.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

SHADER = """
shader BufferMatrixColumns {
    struct Item {
        MATRIX m;
        vec4 tail;
    }
    @ binding(0)
    @ buffer_layout(std430)
    buffer Item items[];
    @ binding(1)
    @ buffer_layout(std430)
    buffer float values[];
    compute {
        @ numthreads(1, 1, 1)
        void main(uvec3 id @ gl_GlobalInvocationID) {
            MATRIX a = items[0].m;
            values[0] = a[1][0];
            values[1] = a[0][1];
            items[1].m = a * 2.0;
        }
    }
}
"""


def _translate(tmp_path, matrix, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(SHADER.replace("MATRIX", matrix), encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _column_load(load):
    if load.startswith("load3("):
        return f"__crossgl_load3_uint(itemsStorage, {load[len('load3('):-1]})"
    return f"itemsStorage.{load}"


def _method_call(match):
    return f"{match.group(2)}.{match.group(1).capitalize()}3("


def _tool(name):
    path = shutil.which(name)
    if path is None:
        pytest.skip(f"{name} is not installed")
    return path


@pytest.mark.parametrize(
    ("matrix", "columns_type", "loads"),
    [
        # Three-row columns go through the helper that works around slangc's Load3 (slang_byte_address_words).
        ("mat3", "float3x3", ("load3(0)", "load3(16)", "load3(32)")),
        ("mat2x3", "float2x3", ("load3(0)", "load3(16)")),
        ("mat3x2", "float3x2", ("Load2(0)", "Load2(8)", "Load2(16)")),
    ],
)
def test_slang_matrix_load_transposes_the_columns(
    tmp_path, matrix, columns_type, loads
):
    code = _flat(_translate(tmp_path, matrix, "slang"))

    columns = ", ".join(f"asfloat({_column_load(load)})" for load in loads)
    assert f"transpose({columns_type}({columns}))" in code


def test_slang_matrix_store_writes_columns(tmp_path):
    code = _flat(_translate(tmp_path, "mat2x3", "slang"))

    assert (
        "__crossgl_store3_uint(itemsStorage, 48, asuint((a * 2.0)._m00_m10_m20))"
        in code
    )
    assert (
        "__crossgl_store3_uint(itemsStorage, 64, asuint((a * 2.0)._m01_m11_m21))"
        in code
    )
    assert "Store3(" not in code
    assert "(a * 2.0)[0]" not in code


@pytest.mark.parametrize("matrix", ["mat3", "mat2x3", "mat4x2"])
def test_slang_matrix_accesses_match_hlsl(tmp_path, matrix):
    def body(backend):
        code = _flat(_translate(tmp_path, matrix, backend))
        # Slang moves three words through helpers (slang_byte_address_words); HLSL calls the methods.
        code = re.sub(r"__crossgl_(load|store)3_uint\((\w+), ", _method_call, code)
        statements = re.findall(r"(?:itemsStorage|values\[)[^;]*;", code)
        return [
            statement.replace("((a * 2.0))", "(a * 2.0)") for statement in statements
        ]

    slang = body("slang")
    assert slang
    assert slang == body("directx")


@pytest.mark.parametrize("matrix", ["mat3", "mat2x3", "mat3x2"])
def test_slang_matrix_output_compiles_with_slangc(tmp_path, matrix):
    slangc = _tool("slangc")
    source = tmp_path / "shader.slang"
    source.write_text(_translate(tmp_path, matrix, "slang"), encoding="utf-8")
    completed = subprocess.run(
        [
            slangc,
            str(source),
            "-target",
            "hlsl",
            "-entry",
            "main",
            "-stage",
            "compute",
            "-o",
            str(tmp_path / "shader.hlsl"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-3000:]
