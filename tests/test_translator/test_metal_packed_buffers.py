"""Metal buffers of packed vectors keep their element layout on every target.

``packed_float3`` is 12 bytes with 4-byte alignment; CrossGL spells it ``vec3``, which std430 and Metal's own
``float3`` lay out in 16. The Metal importer marks such buffers ``@buffer_layout(hlsl_structured)``, the HLSL
structured-buffer layout being exactly the packed one, so the Metal generator declares packed storage again and
OpenGL and SPIR-V address the buffer in 32-bit words. Execution is covered by the ``m_packed_float3`` and
``m_packed_struct`` conformance cases.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.import_refusals import MetalImportRefused
from crosstl.translator.semantics.structured_layouts import (
    StructuredLayoutUnsupportedError,
)

PRELUDE = "#include <metal_stdlib>\nusing namespace metal;\n"

PACKED = PRELUDE + """
struct P { float a; packed_float3 b; float c; packed_float2 d; };
kernel void k(device const packed_float3* v [[buffer(0)]], device P* p [[buffer(1)]],
              device float* o [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    float3 x = v[i];
    o[i] = x.y + p[i].b.z;
    p[i].b = x.zyx;
    p[i].d.y += 1.0;
    P q = p[i + 1u];
    p[i] = q;
}
"""


def _translate(tmp_path, backend, source, suffix="metal", format_output=False):
    path = tmp_path / f"shader.{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=format_output)


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def _kernel(element, prelude=""):
    return (
        PRELUDE
        + prelude
        + f"kernel void k(device {element}* b [[buffer(0)]], uint i [[thread_position_in_grid]]) {{ b[i + 1u] = b[i]; }}\n"
    )


@pytest.mark.parametrize(
    ("element", "prelude", "marked"),
    (
        ("packed_float3", "", True),
        ("packed_int2", "", True),
        ("P", "struct P { float a; packed_float3 b; };", True),
        # packed_float3 then float3: Metal puts b at 16, as std430 does, so no mark is needed.
        ("Q", "struct Q { packed_float3 a; float3 b; };", False),
        ("float3", "", False),
        ("S", "struct S { float a; float b; };", False),
    ),
)
def test_import_marks_packed_elements_with_the_hlsl_structured_layout(
    tmp_path, element, prelude, marked
):
    code = _flat(_translate(tmp_path, "cgl", _kernel(element, prelude)))
    assert ("@buffer_layout(hlsl_structured) RWStructuredBuffer" in code) is marked


@pytest.mark.parametrize(
    ("prelude", "reason"),
    (
        # Metal: b at 16, 24 bytes; HLSL structured: b at 12; std430: 32 bytes.
        ("struct R { packed_float3 a; float2 b; };", "matches neither"),
        ("struct M { packed_float3 a; float3x3 m; };", "packed vectors and matrices"),
    ),
)
def test_import_refuses_packed_elements_no_layout_reproduces(tmp_path, prelude, reason):
    element = prelude.split()[1]
    with pytest.raises(MetalImportRefused, match=re.escape(reason)):
        _translate(tmp_path, "cgl", _kernel(element, prelude))


def test_import_writes_through_buffer_elements(tmp_path):
    source = PRELUDE + """
struct P { float a; float4 b; int c; };
void set(device float& x) { x = 5.0; }
kernel void k(device P* d [[buffer(0)]], device float* f [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    d[i].a = 3.0;
    d[i].b.y += 2.0;
    d[i].c++;
    --d[i].c;
    set(d[i].a);
    set(f[i]);
    f[i] = d[i].b.x;
}
"""
    code = _flat(_translate(tmp_path, "cgl", source))
    for statement in (
        "d[i].a = 3.0;",
        "d[i].b.y += 2.0;",
        "d[i].c++;",
        "--d[i].c",
        "set(d[i].a);",
        "set(f[i]);",
        "buffer_store(f, i, buffer_load(d, i).b.x);",
    ):
        assert statement in code, statement
    assert not re.search(r"buffer_load\([^;]*\)\.[\w.]+ *[-+]?=", code)


def test_spirv_refuses_assignments_to_values_without_storage(tmp_path):
    source = """
shader main {
    struct P { float a; float b; }
    compute {
        @ stage_entry
        void k(RWStructuredBuffer<P> data @buffer(0), uint id @gl_GlobalInvocationID) {
            buffer_load(data, id).b = 3.0;
        }
    }
}
"""
    with pytest.raises(Exception, match="does not name storage"):
        _translate(tmp_path, "vulkan", source, suffix="cgl")


def test_metal_declares_packed_storage_for_marked_parameters(tmp_path):
    code = _flat(_translate(tmp_path, "metal", PACKED))
    assert "const device packed_float3* v [[buffer(0)]]" in code
    assert "struct P { float a; packed_float3 b; float c; packed_float2 d; };" in code
    assert "buffer_layout" not in code


def test_hlsl_keeps_the_structured_element(tmp_path):
    code = _flat(_translate(tmp_path, "directx", PACKED))
    assert "StructuredBuffer<float3> v" in code
    assert "RWStructuredBuffer<P> p" in code


def test_opengl_addresses_packed_parameters_in_words(tmp_path):
    code = _flat(_translate(tmp_path, "opengl", PACKED))
    assert re.search(r"buffer \w+ { uint v\[\]; }", code)
    assert re.search(r"buffer \w+ { uint p\[\]; }", code)
    # p[i].b = x.zyx: the index and the value are evaluated once, into block-scoped temporaries.
    assert "uint cgl_base = ((uint(i) * 7u) + 1u); vec3 cgl_value = x.zyx;" in code
    assert "p[(cgl_base + 2u)] = floatBitsToUint(cgl_value[2]);" in code
    assert "uintBitsToFloat(v[((uint(i) * 3u) + 1u)])" in code
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not on PATH")
    path = tmp_path / "shader.comp"
    path.write_text(_translate(tmp_path, "opengl", PACKED), encoding="utf-8")
    _run(["glslangValidator", "-S", "comp", str(path)])


def test_spirv_addresses_packed_parameters_in_words(tmp_path):
    assembly = _translate(tmp_path, "vulkan", PACKED, format_output=True)
    assert not re.search(r"OpTypeRuntimeArray %(v3float|P)\b", assembly)
    assert "OpTypeRuntimeArray %uint" in assembly
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    path = tmp_path / "shader.spvasm"
    path.write_text(assembly, encoding="utf-8")
    binary = tmp_path / "shader.spv"
    _run(["spirv-as", "--target-env", "vulkan1.1", str(path), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


PARAMETER = """
shader main {{
    uint next() {{ return 1u; }}
    compute {{
        @ stage_entry
        void k(@buffer_layout(hlsl_structured) RWStructuredBuffer<vec3> d @buffer(0), uint id @gl_GlobalInvocationID) {{
            {body}
        }}
    }}
}}
"""


@pytest.mark.parametrize("backend", ("opengl", "vulkan"))
def test_parameter_stores_evaluate_an_effectful_index_once(tmp_path, backend):
    source = PARAMETER.format(body="d[next()] = vec3(1.0, 2.0, 3.0);")
    code = _flat(_translate(tmp_path, backend, source, suffix="cgl"))
    if backend == "opengl":
        assert code.count("next()") == 3  # the prototype, the definition and one call
        assert "uint cgl_base = (uint(next()) * 3u);" in code


@pytest.mark.parametrize(
    ("body", "reason"),
    (
        ("vec3 v = d[next()];", "index with side effects"),
        ("for (uint j = 0u; j < 2u; d[j] = vec3(1.0)) { j++; }", "loop header"),
    ),
)
def test_parameter_accesses_that_cannot_be_expanded_are_refused(tmp_path, body, reason):
    source = PARAMETER.format(body=body)
    for backend in ("opengl", "vulkan"):
        with pytest.raises(StructuredLayoutUnsupportedError, match=re.escape(reason)):
            _translate(tmp_path, backend, source, suffix="cgl")
