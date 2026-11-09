"""Metal storage views whose source is another view, or whose constness the cast drops.

``(device float*)((device char*)dst + offset)`` reinterprets the result of a nested view. The outer view's
source address space is the one the nested view names, and the nested view proves that space against its own
source when it is lowered. Neither the nested view nor the offset makes the source read-only: an offset named
by a ``const`` local is not storage.

A C-style cast such as ``(device float*)readonly_bytes`` drops pointee constness. Metal can reproduce that
spelling only when nothing writes through the resulting pointer, so a view whose pointer is only read
(bound to a ``const`` pointee local, consumed by a ``const`` outer view, or dereferenced as a value) is
emitted as a ``const`` view. A view that is written through, or bound to a reference, is still refused.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.pointer_reinterpret import PointerReinterpretationError

PREFIX = "#include <metal_stdlib>\nusing namespace metal;\n"


def _translate(tmp_path, source, backend="metal"):
    path = tmp_path / "kernel.metal"
    path.write_text(PREFIX + source)
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_backend="metal"
    )


def _metal_compiler_available():
    if shutil.which("xcrun") is None:
        return False
    probe = subprocess.run(
        ["xcrun", "-f", "metal"], capture_output=True, text=True, timeout=60
    )
    return probe.returncode == 0


def _assert_compiles(tmp_path, metal):
    if not _metal_compiler_available():
        return
    source = tmp_path / "generated.metal"
    source.write_text(metal)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(source), "-o", "/dev/null"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


NESTED_VIEW_STORE = """
kernel void k(device char* dst [[buffer(0)]],
              constant uint& stride [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
    device float* row = (device float*)((device char*)dst + i * stride);
    row[0] = 1.0f;
}
"""

CONST_OFFSET_STORE = """
kernel void k(device float* dst [[buffer(0)]],
              constant uint& stride [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
    const int offset = int(stride);
    device float* row = (device float*)((device char*)dst + i * stride + offset);
    row[0] = 2.0f;
}
"""

CONST_LOCAL_READ = """
kernel void k(device const char* src [[buffer(0)]],
              device char* dst [[buffer(1)]],
              constant uint& stride [[buffer(2)]],
              uint i [[thread_position_in_grid]]) {
    device const float* const row = (device float*)(src + i * stride);
    device float* out = (device float*)(dst + i * stride);
    out[0] = row[0] + row[1];
}
"""

SUBSCRIPT_READ = """
kernel void k(device const char* src [[buffer(0)]],
              device float* dst [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
    float value = ((device float*)src)[i];
    dst[i] = value * 2.0f;
}
"""

NESTED_CONST_OUTER_READ = """
kernel void k(device const char* src [[buffer(0)]],
              device float* dst [[buffer(1)]],
              constant uint& stride [[buffer(2)]],
              uint i [[thread_position_in_grid]]) {
    device const float* row = (device const float*)((device char*)src + i * stride);
    dst[i] = row[0];
}
"""

DEREFERENCE_READ = """
kernel void k(device const char* src [[buffer(0)]],
              device char* dst [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
    device const char* src_ptr = src + i * 4;
    device char* dst_ptr = dst + i * 4;
    *((device float*)dst_ptr) = *((device float*)src_ptr);
}
"""


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            NESTED_VIEW_STORE,
            "reinterpret_cast<device float*>(reinterpret_cast<device char*>(dst) + i * stride)",
        ),
        (
            CONST_OFFSET_STORE,
            "reinterpret_cast<device float*>(reinterpret_cast<device char*>(dst) + i * stride + offset)",
        ),
        (
            CONST_LOCAL_READ,
            "const device float* row = reinterpret_cast<const device float*>(src + i * stride);",
        ),
        (
            SUBSCRIPT_READ,
            "float value = (reinterpret_cast<const device float*>(src))[i];",
        ),
        (
            NESTED_CONST_OUTER_READ,
            "reinterpret_cast<const device float*>(reinterpret_cast<const device char*>(src) + i * stride)",
        ),
        (
            DEREFERENCE_READ,
            "= *reinterpret_cast<const device float*>(src_ptr);",
        ),
    ],
    ids=(
        "nested-view-store",
        "const-offset-store",
        "const-local-read",
        "subscript-read",
        "nested-const-outer-read",
        "dereference-read",
    ),
)
def test_storage_views_round_trip_through_metal(tmp_path, source, expected):
    metal = _translate(tmp_path, source)
    assert expected in metal
    assert "reinterpret_cast<device float*>(src" not in metal
    _assert_compiles(tmp_path, metal)


def _translate_crossgl(tmp_path, body):
    path = tmp_path / "views.cgl"
    path.write_text("shader main {\n" + body + "\n}\n")
    return crosstl.translate(str(path), backend="metal", format_output=False)


def test_const_removing_view_read_as_value_is_a_const_view(tmp_path):
    metal = _translate_crossgl(
        tmp_path,
        "float read(const device float* src, int i) {"
        " float v = ((device float*)src)[i]; return v; }",
    )
    assert "float v = (reinterpret_cast<const device float*>(src))[i];" in metal


@pytest.mark.parametrize(
    "body",
    [
        "void store(const device float* src, int i) {"
        " ((device float*)src)[i] = 1.0; }",
        "void store(const device float* src, int i) {"
        " device float* view = (device float*)src; view[i] = 1.0; }",
        "void store(const device float* src, int i) {"
        " device float& slot = ((device float*)src)[i]; slot = 1.0; }",
        "void store(const device float* src, int i) {"
        " ((device float*)src)[i] += 1.0; }",
        "void store(const device float* src, int i) {" " ((device float*)src)[i]++; }",
    ],
    ids=(
        "subscript-write",
        "mutable-local",
        "reference-binding",
        "compound-write",
        "increment",
    ),
)
def test_const_removing_views_that_may_write_stay_refused(tmp_path, body):
    with pytest.raises(PointerReinterpretationError) as exc_info:
        _translate_crossgl(tmp_path, body)
    assert exc_info.value.reason == "const-removal"
    assert exc_info.value.target_backend == "metal"
