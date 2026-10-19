"""Byte storage retains signedness and physical width through native packages."""

import base64
import copy
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.project import MetalComputeRuntime, NativeLoaderDispatchError
from crosstl.project.native_runtime_drivers import (
    _normalize_dtype,
    _pack_values,
    _unpack_values,
)
from crosstl.project.runtime_verification import (
    RuntimeAllocationView,
    RuntimeExecutorUnavailable,
    RuntimeValue,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_native_runtime import _native_request
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_BYTE_BUFFER_RUNTIME"
FORMS = (
    "scalar",
    "constant",
    "reference",
    "vector2",
    "vector4",
    "alias",
    "struct",
    "offset",
)


def _case(root, dtype, form="scalar", count=7):
    root.mkdir(parents=True, exist_ok=True)
    scalar = "char" if dtype == "int8" else "uchar"
    width = 2 if form in {"vector2", "struct"} else 4 if form == "vector4" else 1
    payload = scalar + str(width) if form.startswith("vector") else scalar
    declarations = ""
    if form == "alias":
        declarations = f"using Byte = {dtype}_t;"
        payload = "Byte"
    elif form == "struct":
        declarations = f"struct Cell {{ {scalar} first; {scalar} second; }};"
        payload = "Cell"
    qualifier = "constant" if form in {"constant", "reference"} else "device"
    pointer = "&" if form == "reference" else "*"
    expression = "values" if form == "reference" else "values[i]"
    source = f"""#include <metal_stdlib>
using namespace metal;
{declarations}
kernel void byte_storage({qualifier} {payload}{pointer} values [[buffer(0)]],
                         device {payload}* results [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {{
    results[i + 1u] = {expression};
}}
"""
    _, descriptor, package = _package(root, "metal", "uint", (1, 1, 1), source=source)
    patterns = (
        [-128, -127, -1, 0, 1, 126, 127]
        if dtype == "int8"
        else [0, 1, 127, 128, 129, 254, 255]
    )
    values = [patterns[i % len(patterns)] for i in range(count * width)]
    if form == "reference":
        values = values[:1]
    shape = [count + 2, width] if width > 1 else [count + 2]
    initial = [85] * ((count + 2) * width)
    expected = (
        initial[:width]
        + (values * count if form == "reference" else values)
        + initial[-width:]
    )
    inputs = {
        "values": {"dtype": dtype, "shape": [len(values)], "values": values},
        "results": {"dtype": dtype, "shape": shape, "values": initial},
    }
    outputs = {
        "results": {"dtype": dtype, "shape": shape, "values": expected},
    }
    if qualifier == "device":
        outputs["values"] = inputs["values"]
    if form == "offset":
        for collection in (inputs, outputs):
            for name, value in list(collection.items()):
                size = len(value["values"])
                collection[name] = RuntimeValue(
                    name=name,
                    dtype=dtype,
                    shape=tuple(value["shape"]),
                    values=value["values"],
                    allocation=RuntimeAllocationView(name, 3, size, size + 8),
                )
    request = _request(descriptor, package, inputs, outputs, count)
    expected = _bound_values(
        descriptor,
        {
            name: (
                {
                    "dtype": value.dtype,
                    "shape": list(value.shape),
                    "values": value.values,
                }
                if isinstance(value, RuntimeValue)
                else value
            )
            for name, value in outputs.items()
        },
    )
    return source, descriptor, package, inputs, outputs, request, expected


@pytest.mark.parametrize("dtype", ("int8", "uint8"))
@pytest.mark.parametrize("form", FORMS)
def test_byte_packages_keep_physical_width(tmp_path, dtype, form):
    _, descriptor, _, inputs, _, request, _ = _case(tmp_path, dtype, form)
    for binding in descriptor["bindings"]:
        layout = binding["scalarLayout"]
        assert layout["elementType"] == dtype
        assert layout["elementSizeBytes"] == layout["elementStrideBytes"]
        assert layout["elementSizeBytes"] == (
            layout.get("vectorWidth", layout.get("componentCount", 1))
        )
    _, native = _native_request(request)
    payload, readbacks = MetalComputeRuntime()._prepare_request(native)
    allocations = {item["id"]: item for item in payload["allocations"]}
    for binding in payload["buffers"]:
        name = descriptor["bindings"][binding["index"]]["name"]
        value = inputs[name]
        values = value.values if isinstance(value, RuntimeValue) else value["values"]
        allocation = allocations[binding["allocation"]]
        assert allocation["length"] == len(values) + (8 if form == "offset" else 0)
        assert binding["offset"] == (3 if form == "offset" else 0)
        assert binding["length"] == len(values)
        assert allocation["uploads"] == [
            {
                "offset": binding["offset"],
                "data": (
                    base64.b64encode(bytes(item & 255 for item in values)).decode(
                        "ascii"
                    )
                ),
            }
        ]
    assert all(value[0] == dtype for value in readbacks.values())


@pytest.mark.parametrize(
    "dtype,aliases",
    (
        ("int8", ("int8", "int8_t", "char", "i8")),
        ("uint8", ("uint8", "uint8_t", "uchar", "u8")),
    ),
)
def test_byte_packing_covers_every_payload_without_widening(dtype, aliases):
    values = list(range(-128, 128)) if dtype == "int8" else list(range(256))
    payload = bytes(value & 255 for value in values)
    for alias in aliases:
        assert _normalize_dtype(alias, target="Metal") == dtype
        for target in ("OpenGL", "DirectX", "Vulkan"):
            with pytest.raises(RuntimeExecutorUnavailable):
                _normalize_dtype(alias, target=target)
    assert _pack_values(values, dtype, expected_count=256, target="Metal") == payload
    assert _unpack_values(payload, dtype, target="Metal") == values


@pytest.mark.parametrize("dtype", ("int8", "uint8"))
@pytest.mark.parametrize("value", (True, False, 1.5, "1", None, -129, 256))
def test_byte_values_do_not_coerce_or_wrap(tmp_path, dtype, value):
    _, descriptor, package, inputs, outputs, _, _ = _case(tmp_path, dtype)
    inputs["values"]["values"][0] = value
    with pytest.raises(
        NativeLoaderDispatchError,
        match="value-size-mismatch" if value is None else "value-data-invalid",
    ):
        _request(descriptor, package, inputs, outputs, 7)
    with pytest.raises(RuntimeExecutorUnavailable):
        _pack_values([value], dtype, expected_count=1, target="Metal")


@pytest.mark.parametrize("dtype,value", (("int8", 128), ("uint8", -1)))
def test_byte_values_reject_wrong_signed_range(tmp_path, dtype, value):
    test_byte_values_do_not_coerce_or_wrap(tmp_path, dtype, value)


@pytest.mark.parametrize("dtype", ("int8", "uint8"))
@pytest.mark.parametrize(
    "field,value",
    (
        ("elementSizeBytes", 4),
        ("elementStrideBytes", 4),
        ("alignmentBytes", 2),
        ("physicalType", "int"),
        ("memberOffsetBytes", 1),
        ("runtimeSized", False),
    ),
)
def test_byte_loader_rejects_incompatible_layout(tmp_path, dtype, field, value):
    _, descriptor, package, inputs, outputs, _, _ = _case(tmp_path, dtype)
    descriptor = copy.deepcopy(descriptor)
    for binding in descriptor["bindings"]:
        if binding["name"] == "results":
            binding["scalarLayout"][field] = value
    for binding in descriptor["scalarLayout"]["bindings"]:
        if binding["binding"] == "results":
            binding["layout"][field] = value
    with pytest.raises(NativeLoaderDispatchError):
        _request(descriptor, package, inputs, outputs, 7)


@pytest.mark.parametrize("dtype", ("int8", "uint8"))
@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_byte_payload_cannot_bind_widened_target_storage(tmp_path, dtype, target):
    source, _, _, inputs, outputs, _, _ = _case(tmp_path / "metal", dtype)
    (tmp_path / target).mkdir()
    _, descriptor, package = _package(
        tmp_path / target,
        target,
        "uint",
        (1, 1, 1),
        source=source,
        software_subgroups=False,
    )
    with pytest.raises(NativeLoaderDispatchError, match="resource-layout-mismatch"):
        _request(descriptor, package, inputs, outputs, 7)


@pytest.mark.parametrize("dtype", ("int8", "uint8"))
@pytest.mark.parametrize("offset,length,backing", ((-1, 7, 15), (3, 6, 15), (3, 7, 9)))
def test_byte_loader_rejects_invalid_allocation_views(
    tmp_path, dtype, offset, length, backing
):
    _, descriptor, package, inputs, outputs, _, _ = _case(tmp_path, dtype, "offset")
    inputs["values"] = replace(
        inputs["values"],
        allocation=RuntimeAllocationView("values", offset, length, backing),
    )
    with pytest.raises(NativeLoaderDispatchError):
        _request(descriptor, package, inputs, outputs, 7)


@pytest.mark.parametrize("dtype", ("int8", "uint8"))
@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("count", (1, 7))
def test_byte_buffers_execute_original_and_generated(tmp_path, dtype, form, count):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native byte storage")
    assert sys.platform == "darwin", "the required byte storage gate needs Metal"
    source, _, _, _, _, request, expected = _case(tmp_path, dtype, form, count)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="byte_storage",
    )


def test_byte_storage_is_required_in_metal_ci():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate Metal byte and vector storage"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_byte_buffer_runtime.py" in step
    assert "tests/test_translator/test_metal_narrow_aggregate_runtime.py" in step
    assert "-n auto" in step and "continue-on-error" not in step and "if:" not in step
