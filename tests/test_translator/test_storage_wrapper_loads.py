"""Storage wrapper value loads preserve logical layout and constructor behavior."""

import os
import shutil
import sys
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.translator.codegen.pointer_reinterpret import PointerReinterpretationError
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_SCALAR_ALIASES"
TYPES = (
    ("uchar", "uint8", 255),
    ("ushort", "uint16", 65535),
    ("uint", "uint32", 0xFFFFFFFF),
)


def source(scalar, mutable=False):
    qualifier = "" if mutable else "const "
    return f"""#include <metal_stdlib>
using namespace metal;
using Scalar = {scalar};
struct Encoded {{
    Scalar payload;
    Encoded(uint value) : payload(Scalar(value ^ 90u)) {{}}
    operator uint() const {{ return uint(payload); }}
}};
kernel void load_values(const device Scalar* values [[buffer(0)]],
                        device uint* results [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {{
    using Wrapper = Encoded;
    uint cursor = i + 3u;
    auto indexed = (({qualifier}device Wrapper*)values)[cursor++];
    auto offset = (({qualifier}device Wrapper*)(values + 3u))[i];
    auto addressed = *({qualifier}device Wrapper*)(&values[i + 3u]);
    Scalar byte = values[i + 3u];
    auto local = *(const thread Wrapper*)(&byte);
    Encoded constructed = Encoded(uint(byte));
    uint base = 4u + 6u * i;
    results[base] = uint(indexed);
    results[base + 1u] = uint(offset);
    results[base + 2u] = uint(addressed);
    results[base + 3u] = uint(local);
    results[base + 4u] = uint(constructed);
    results[base + 5u] = cursor;
}}
"""


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
@pytest.mark.parametrize("scalar,dtype,mask", TYPES)
def test_storage_wrapper_value_loads_compile(tmp_path, target, scalar, dtype, mask):
    path = tmp_path / "source.metal"
    path.write_text(source(scalar))
    generated = translate(str(path), backend=target, format_output=False)
    tool = {"metal": "xcrun", "opengl": "glslangValidator", "directx": "dxc"}[target]
    if shutil.which(tool):
        _compile(
            generated, target, tmp_path, directx_compile_flags=("-enable-16bit-types",)
        )


@pytest.mark.parametrize("mutable", (False, True))
@pytest.mark.parametrize("scalar,dtype,mask", TYPES)
def test_storage_wrapper_value_loads_execute(tmp_path, scalar, dtype, mask, mutable):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required wrapper execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    original = source(scalar, mutable)
    _, descriptor, package = _package(
        tmp_path, target, scalar, (1, 1, 1), source=original, software_subgroups=False
    )
    values = [((i * 0x01010101) ^ 0x80010000) & mask for i in range(256)]
    guard = 0xDEADBEEF
    expected = [guard] * 4
    for i, value in enumerate(values):
        expected.extend([value, value, value, value, (value ^ 90) & mask, i + 4])
    expected.extend([guard] * 4)
    input_binding = next(
        binding
        for binding in descriptor["bindings"]
        if binding.get("scalarLayout", {})
        .get("memberName", binding["name"])
        .rstrip("_")
        == "values"
    )
    transport_dtype = input_binding["scalarLayout"]["elementType"]
    inputs = {
        "values": {
            "dtype": transport_dtype,
            "shape": [262],
            "values": [17, 23, 31] + values + [47, 53, 61],
        },
        "results": {
            "dtype": "uint32",
            "shape": [len(expected)],
            "values": [guard] * len(expected),
        },
    }
    outputs = {"results": {**inputs["results"], "values": expected}}
    request = _request(descriptor, package, inputs, outputs, len(values))

    def validate(artifact, output, target):
        return _compile(
            artifact.read_text(),
            target,
            output,
            directx_compile_flags=("-enable-16bit-types",),
        )[1]

    _execute(
        request,
        _bound_values(descriptor, outputs),
        tmp_path,
        original_source=original,
        original_entry="load_values",
        validate=validate,
    )


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
@pytest.mark.parametrize("members", ("uchar a; uchar b;", "ushort a;", "uchar a[2];"))
def test_unproven_wrapper_layouts_remain_diagnostics(tmp_path, target, members):
    path = tmp_path / "source.metal"
    path.write_text(f"""#include <metal_stdlib>
using namespace metal;
struct Wrapper {{ {members} }};
kernel void load_values(const device uchar* values [[buffer(0)]],
                        device uint* results [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {{
    auto loaded = ((const device Wrapper*)values)[i];
    results[i] = i;
}}
""")
    with pytest.raises(PointerReinterpretationError):
        translate(str(path), backend=target, format_output=False)


def test_wrapper_execution_reuses_required_alias_gate():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate scalar alias resolution"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_storage_wrapper_loads.py" in step
    assert "--timeout-seconds 120" in step and "-n auto" in step
    assert "continue-on-error" not in step and "if:" not in step
