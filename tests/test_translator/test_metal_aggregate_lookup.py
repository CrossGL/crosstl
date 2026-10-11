"""Inferred aggregate locals retain type identity when value names hide tags."""

import os
import sys
from pathlib import Path

import pytest

from tests.test_translator.test_boolean_buffer_runtime import _bound_values, _request
from tests.test_translator.test_byte_buffer_runtime import REQUIRE_ENV
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_software_subgroup_product import _package

KINDS = ("struct", "union")
CONTEXTS = ("entry", "local", "parameter", "helper")


def _case(root, kind, context):
    entry = "Payload" if context == "entry" else "aggregate_shadow"
    declaration = "uint Payload = 5u;" if context == "local" else ""
    helper = {
        "parameter": (
            "uint read_payload(uint Payload) { auto value = make_payload(); "
            "return value.word + Payload; }"
        ),
        "helper": "uint Payload(uint x) { return x; }",
    }.get(context, "")
    result = {
        "local": "value.word + Payload",
        "parameter": "value.word + read_payload(5u)",
        "helper": "value.word + Payload(5u)",
    }.get(context, "value.word")
    value = {"local": 47, "parameter": 89, "helper": 47}.get(context, 42)
    source = f"""#include <metal_stdlib>
using namespace metal;
{kind} Payload {{ uint word; }};
Payload make_payload() {{ Payload value; value.word = 42u; return value; }}
{helper}
kernel void {entry}(device uint* results [[buffer(0)]]) {{
    {declaration}
    auto value = make_payload();
    results[1] = {result};
}}
"""
    _, descriptor, package = _package(
        root, "metal", "uint", (1, 1, 1), source=source, software_subgroups=False
    )
    assert descriptor["entryPoint"]["name"] == entry
    inputs = {"results": {"dtype": "uint32", "shape": [3], "values": [99] * 3}}
    outputs = {"results": {"dtype": "uint32", "shape": [3], "values": [99, value, 99]}}
    return (
        source,
        entry,
        _request(descriptor, package, inputs, outputs, 1),
        _bound_values(descriptor, outputs),
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("context", CONTEXTS)
def test_metal_aggregate_lookup_retains_tag_and_entry(tmp_path, kind, context):
    _, entry, request, _ = _case(tmp_path, kind, context)
    translated = request.artifact_path.read_text()
    assert f"{kind} Payload value = make_payload()" in translated
    assert f"kernel void {entry}(" in translated


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("context", CONTEXTS)
def test_metal_aggregate_lookup_executes_original_and_generated(
    tmp_path, kind, context
):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native aggregate lookup")
    assert sys.platform == "darwin", "the required aggregate lookup gate needs Metal"
    source, entry, request, expected = _case(tmp_path, kind, context)
    _execute(request, expected, tmp_path, original_source=source, original_entry=entry)


def test_metal_aggregate_lookup_is_required_in_native_ci():
    workflow = (
        Path(__file__).parents[2] / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow.split("- name: Validate Metal byte and vector storage\n", 1)[
        1
    ].split("      - name:", 1)[0]
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_metal_aggregate_lookup.py" in step
