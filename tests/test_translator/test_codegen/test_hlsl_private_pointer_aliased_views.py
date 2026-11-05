"""DirectX calls that pass two private pointer views of one local array.

HLSL passes a private pointer view as a copy of its backing array plus a base
offset. Two views of one array in one call are independent copies: ``inout``
copies are written back one after another, so one view's writes overwrite the
other's, and a read-only copy never sees writes made through a writable view.
Such calls must raise a typed diagnostic instead of emitting HLSL that DXC
accepts but that computes a different result than Metal. Views that are all
read-only still share their backing array.
"""

import pytest

from crosstl._crosstl import translate
from crosstl.translator.codegen.directx_codegen import (
    DirectXPrivatePointerParameterError,
)
from crosstl.translator.codegen.private_pointer_alias_views import (
    hlsl_aliased_writable_view,
)
from tests.test_translator.test_codegen.test_directx_codegen import (
    assert_directx_warnings_clean_if_available,
)

PRELUDE = """#include <metal_stdlib>
using namespace metal;

void two(thread float* a, thread float* b) {
    a[0] = 1.0f;
    b[0] = 2.0f;
}

void accumulate(thread float* destination, thread const float* source) {
    destination[1] = 5.0f;
    destination[0] += source[0];
}

float total(thread const float* a, thread const float* b) {
    return a[0] + b[0];
}
"""

FORWARD = """
void forward(thread float* p) {
    two(p, p + 1);
}
"""


def _kernel(body, helpers=""):
    return (
        PRELUDE
        + helpers
        + "kernel void k(device float* out [[buffer(0)]],\n"
        + "              uint tid [[thread_position_in_grid]]) {\n"
        + "    float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};\n"
        + "    float y[4] = {0.0f, 0.0f, 0.0f, 0.0f};\n"
        + f"    {body}\n"
        + "    out[tid * 2] = x[0] + y[0];\n"
        + "    out[tid * 2 + 1] = x[1] + y[1];\n"
        + "}\n"
    )


def _translate(tmp_path, body, helpers=""):
    source = tmp_path / "aliased_views.metal"
    source.write_text(_kernel(body, helpers), encoding="utf-8")
    return translate(
        str(source), backend="directx", source_backend="metal", format_output=False
    )


@pytest.mark.parametrize(
    ("body", "helpers", "first", "second", "root"),
    (
        ("two(x, x + 1);", "", "a", "b", "x"),
        ("two(x, x);", "", "a", "b", "x"),
        ("two(&x[1], x + 2);", "", "a", "b", "x"),
        ("two(x + 2, &x[0]);", "", "a", "b", "x"),
        # A read-only view misses the writes made through the writable one.
        ("accumulate(x + 1, x);", "", "destination", "source", "x"),
        # Both views are rooted at the caller's parameter, not at its local.
        ("forward(x);", FORWARD, "a", "b", "p"),
    ),
    ids=("offset", "same", "address-and-offset", "reversed", "read-write", "forwarded"),
)
def test_aliased_writable_private_views_fail_closed(
    tmp_path, body, helpers, first, second, root
):
    with pytest.raises(DirectXPrivatePointerParameterError) as error:
        _translate(tmp_path, body, helpers)
    assert error.value.reason == "aliased-view-backing"
    message = str(error.value)
    assert f"parameters '{first}' and '{second}'" in message
    assert f"backing object '{root}'" in message


@pytest.mark.parametrize(
    "body",
    (
        "two(x, y + 1);",
        "accumulate(x, y);",
        "out[tid] = total(x, x + 1);",
        "out[tid] = total(&x[1], x);",
    ),
    ids=("distinct-arrays", "distinct-read-write", "read-only", "read-only-address"),
)
def test_unaliased_or_read_only_private_views_still_translate(tmp_path, body):
    generated = _translate(tmp_path, body)
    assert "CSMain" in generated
    assert_directx_warnings_clean_if_available(generated, tmp_path)


def test_read_only_views_share_one_backing_array(tmp_path):
    generated = _translate(tmp_path, "out[tid] = total(x, x + 1);")
    compact = " ".join(generated.split())
    assert "total(x, 0, x, 1)" in compact
    assert "const float a[4]" in compact and "const float b[4]" in compact


def test_alias_search_is_deterministic_and_skips_read_only_groups():
    read_only = {
        "callee": "f",
        "bindings": {
            "a": {"root_kind": "local", "root": "x"},
            "b": {"root_kind": "local", "root": "x"},
        },
    }
    writable = {
        "callee": "g",
        "bindings": {
            "c": {"root_kind": "local", "root": "y"},
            "d": {"root_kind": "parameter", "root": "y"},
            "e": {"root_kind": "local", "root": "y"},
        },
    }
    calls = {"main": [read_only, writable]}
    found = hlsl_aliased_writable_view(calls, {"f": frozenset(), "g": {"e"}})
    # ``d`` names the caller parameter ``y``, a different object than local ``y``.
    assert found == (writable, "c", "e", "y")
    assert hlsl_aliased_writable_view(calls, {"f": frozenset(), "g": set()}) is None
