"""DirectX private pointer helpers keep only the backings their calls reach.

A Metal helper taking ``const thread uint8_t*`` becomes an HLSL array
parameter, emitted once per backing its calls bind. A kernel that passes a
byte view of a local record and a plain local array for the other pointer::

    accumulate((thread uint8_t*)&record, 1.0f, result);

reaches only the view's variant. The helper's default backing, a plain
``uint w[4]``, has the same HLSL type, and ``in`` and ``inout`` do not tell
HLSL overloads apart, so emitting both would make every call ambiguous.
"""

import pytest

import crosstl
from crosstl.translator.codegen import hlsl_private_pointer_variants as variants
from crosstl.translator.codegen.directx_codegen import (
    DirectXPrivatePointerParameterError,
)
from tests.test_translator.test_codegen.test_directx_codegen import (
    assert_directx_warnings_clean_if_available,
)

ACCUMULATE = """
inline void accumulate(const thread uint8_t* w, float x, thread float* result) {
  for (int i = 0; i < 4; i++) {
    result[i] += x * float(w[i]);
  }
}
"""
# A helper nothing else calls is emitted with its default backing, which
# then reaches accumulate, so only kernels that call it define it.
FORWARD = """
inline void forward(const thread uint8_t* w, thread float* result) {
  accumulate(w, 2.0f, result);
}
"""


def _source(body, record_bytes=4):
    return f"""#include <metal_stdlib>
using namespace metal;
{ACCUMULATE}{FORWARD if "forward(" in body else ""}
kernel void k(
    const device uint8_t* w [[buffer(0)]],
    device float* y [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {{
  typedef struct {{
    uint8_t wi[{record_bytes}];
  }} vec_w;
  thread vec_w w_local;
  thread uint8_t plain[4] = {{1, 2, 3, 4}};
  thread float result[4] = {{0}};
  for (int j = 0; j < 2; j++) {{
    w_local = *((const device vec_w*)(w + {record_bytes} * j));
{body}
  }}
  for (int i = 0; i < 4; i++) {{
    y[tid * 4 + i] = result[i] + float(plain[i]);
  }}
}}
"""


def _translate(tmp_path, body, record_bytes=4):
    path = tmp_path / "kernel.metal"
    path.write_text(_source(body, record_bytes), encoding="utf-8")
    return crosstl.translate(str(path), backend="directx")


def _definitions(generated, name):
    return [
        line
        for line in generated.splitlines()
        if line.startswith(f"void {name}(") and not line.endswith(";")
    ]


@pytest.mark.parametrize(
    "body",
    [
        "    accumulate((thread uint8_t*)&w_local, 1.0f, result);",
        "    accumulate((thread uint8_t*)&w_local, 1.0f, result);\n"
        "    accumulate((const thread uint8_t*)&w_local, 3.0f, result);",
        "    forward((thread uint8_t*)&w_local, result);",
    ],
    ids=["one-call", "two-calls", "forwarded"],
)
def test_record_views_emit_one_helper_variant(tmp_path, body):
    generated = _translate(tmp_path, body)
    for name in ("accumulate", "forward") if "forward" in body else ("accumulate",):
        (definition,) = _definitions(generated, name)
        assert "uint w[4], int w_base" in definition
    # Bytes of the view are masked to their low byte.
    assert "& 255u" in generated
    assert_directx_warnings_clean_if_available(generated, tmp_path, profile="cs_6_2")


def test_unrecorded_call_keeps_the_default_backing(tmp_path):
    # The analysis skips the dead call, which still needs the helper's own
    # four-element backing beside the eight-byte view.
    body = (
        "    if (false) { accumulate(plain, 1.0f, result); }\n"
        "    accumulate((thread uint8_t*)&w_local, 1.0f, result);"
    )
    generated = _translate(tmp_path, body, record_bytes=8)
    definitions = _definitions(generated, "accumulate")
    assert len(definitions) == 2
    assert sum("uint w[4], int w_base" in line for line in definitions) == 1
    assert sum("uint w[8], int w_base" in line for line in definitions) == 1
    assert_directx_warnings_clean_if_available(generated, tmp_path, profile="cs_6_2")


def test_indistinguishable_backings_are_refused(tmp_path):
    body = (
        "    accumulate(plain, 1.0f, result);\n"
        "    accumulate((thread uint8_t*)&w_local, 1.0f, result);"
    )
    with pytest.raises(DirectXPrivatePointerParameterError) as error:
        _translate(tmp_path, body)
    assert error.value.reason == "indistinguishable-backing-variants"
    assert error.value.function_name == "accumulate"
    assert error.value.parameter_name == "w"


def _call(caller, callee, **roots):
    bindings = {
        name: (
            {"root_kind": "parameter", "root": root}
            if isinstance(root, str)
            else {"root_kind": "local", "extent": root}
        )
        for name, root in roots.items()
    }
    return {"caller": caller, "callee": callee, "bindings": bindings}


def _callee_variant(parameters):
    def bind(caller_variant, call):
        values = []
        for name in parameters[call["callee"]]:
            binding = call["bindings"][name]
            if binding["root_kind"] == "local":
                values.append(binding["extent"])
            else:
                values.append(
                    caller_variant[parameters[call["caller"]].index(binding["root"])]
                )
        return tuple(values)

    return bind


def test_reached_variants_follow_call_chains():
    parameters = {"inner": ["p"], "outer": ["q"], "unused": ["r"]}
    candidates = {
        "inner": {(4,), (8,), (16,)},
        "outer": {(4,), (8,)},
        "unused": {(2,)},
    }
    calls = {
        "main": [_call("main", "outer", q=8)],
        "outer": [_call("outer", "inner", p="q")],
    }
    reached = variants.reached_variants(
        candidates, calls, parameters, _callee_variant(parameters)
    )
    # outer's default (4,) is reached by no call, nor the inner variant it binds.
    assert reached == {"inner": {(8,)}, "outer": {(8,)}, "unused": {(2,)}}


def test_unreached_helpers_keep_their_candidates():
    parameters = {"inner": ["p"], "outer": ["q"]}
    candidates = {"inner": {(4,)}, "outer": {(4,)}}
    # outer calls inner but nothing reaches outer except itself being uncalled.
    calls = {"outer": [_call("outer", "inner", p="q")]}
    reached = variants.reached_variants(
        candidates, calls, parameters, lambda caller_variant, call: None
    )
    assert reached == {"inner": {(4,)}, "outer": {(4,)}}


VIEW = ("word_view", "uint8", "uint8", 4)
SIGNATURES = {(4,): "uint[4]", (VIEW,): "uint[4]", (8,): "uint[8]", (2,): "uint[2]"}


def test_emitted_variants_drop_unreached_overloads_of_reached_signatures():
    emitted, collisions = variants.emitted_variants(
        {(4,), (VIEW,), (8,)}, {(VIEW,)}, SIGNATURES.get
    )
    # The unreached default shares the view's signature; (8,) has its own.
    assert emitted == {(VIEW,), (8,)}
    assert collisions == []


def test_emitted_variants_keep_one_unreached_overload_per_signature():
    emitted, collisions = variants.emitted_variants(
        {(4,), (VIEW,), (2,)}, {(2,)}, SIGNATURES.get
    )
    # Of the two unreached uint[4] candidates, the first by repr is kept.
    assert emitted == {(2,), (VIEW,)}
    assert collisions == []


def test_emitted_variants_report_reached_collisions():
    emitted, collisions = variants.emitted_variants(
        {(4,), (VIEW,)}, {(4,), (VIEW,)}, SIGNATURES.get
    )
    assert emitted == {(4,), (VIEW,)}
    assert collisions == [((VIEW,), (4,))]
