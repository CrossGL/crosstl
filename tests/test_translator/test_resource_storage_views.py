"""Aggregates may hold narrowing byte and halfword views of word storage.

Kernels reinterpret a ``uint32_t`` buffer as bytes and keep the byte pointer in a
loader struct that advances it. Resource-aggregate lowering represents the view as
a handle of the narrower element whose identity still names the word buffer, and a
read extracts the element from its little-endian word. Casting ``const`` away from
a view keeps it read-only, so only a later write through it fails, and a discarded
``++`` or ``--`` steps the view by one element. Casting a byte view of words back
to words restores the word view, at the word its byte offset names. Writes through
a narrowing view, signed views, widening views of storage that holds no whole
wider elements and views of workgroup memory fail closed.
"""

import os
import struct
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_RESOURCE_STORAGE_VIEWS"
ENTRY = "gather_views"
THREADS = 32
WORKGROUP = (THREADS, 1, 1)
STRIDE = 5
RESTORED_BYTES = 12
WORDS = [(index * 2654435761 + 0x9E3779B9) % (1 << 32) for index in range(64)]
GUARD = [0x6D2B79F5, 0x0BADF00D, 0x1234567, 0xA5A5A5A5]


def _source(byte_view="const device uint8_t*", writer=""):
    return f"""#include <metal_stdlib>
using namespace metal;

uint gather(const device uint8_t* bytes, int count) {{
  uint total = 0;
  for (int i = 0; i < count; i++) {{
    total |= uint(bytes[i]) << (8 * i);
  }}
  return total;
}}

struct ByteReader {{
  {byte_view} src;
  const int stride;
  ByteReader({byte_view} src_, int stride_, uint lane)
      : src(src_ + 3 * lane), stride(stride_) {{}}
  uint read() const {{
    // Casting const away keeps the view read-only.
    return gather((device uint8_t*)(src + 1), 3);
  }}
  void next() {{
    src += stride;
  }}
}};

struct HalfReader {{
  const device uint16_t* src;
  HalfReader(const device uint16_t* src_, uint lane) : src(src_ + lane + 1) {{}}
  uint read(int i) const {{
    return uint(src[i]);
  }}
  void advance() {{
    // Discarded increments and decrements step the view by elements.
    ++src;
    src++;
    src--;
  }}
}};

struct WordReader {{
  const device uint32_t* src;
  WordReader(const device uint32_t* src_, uint lane) : src(src_ + lane) {{}}
  uint read() const {{
    // Views the restored words as bytes again.
    auto bytes = (const device uint8_t*)src;
    return src[1] ^ uint(bytes[2]);
  }}
}};
{writer}
[[kernel]] void {ENTRY}(
    const device uint32_t* words [[buffer(0)]],
    device uint* results [[buffer(1)]],
    uint lane [[thread_index_in_threadgroup]]) {{
  auto bytes = ({byte_view})words;
  ByteReader reader(bytes + 2, {STRIDE}, lane);
  uint first = reader.read();
  reader.next();
  HalfReader halves((const device uint16_t*)words, lane);
  halves.advance();
  // A byte offset of whole words restores the word view.
  auto skipped = bytes + {RESTORED_BYTES};
  WordReader restored((const device uint32_t*)skipped, lane);
  results[lane] = (first ^ (reader.read() << 1)) + halves.read(2) + restored.read();
}}
"""


def _expected():
    data = b"".join(struct.pack("<I", word) for word in WORDS)
    halves = struct.unpack(f"<{len(WORDS) * 2}H", data)
    results = []
    for lane in range(THREADS):
        start = 2 + 3 * lane + 1
        first = int.from_bytes(data[start : start + 3], "little")
        start += STRIDE
        second = int.from_bytes(data[start : start + 3], "little")
        word = RESTORED_BYTES // 4 + lane
        restored = WORDS[word + 1] ^ data[4 * word + 2]
        value = (first ^ (second << 1)) + halves[lane + 1 + 1 + 2] + restored
        results.append(value % (1 << 32))
    return results + GUARD


def _request(root, target, source=None):
    # No index range assertions: every subscript must be proven.
    source, descriptor, package = _package(
        root,
        target,
        "uint",
        WORKGROUP,
        source=source or _source(),
        software_subgroups=False,
    )
    expected = _expected()
    initial = [0x21524111] * THREADS + GUARD
    inputs = _bound_values(
        descriptor,
        {
            "words": {"dtype": "uint32", "shape": [len(WORDS)], "values": WORDS},
            "results": {"dtype": "uint32", "shape": [len(initial)], "values": initial},
        },
    )
    outputs = _bound_values(
        descriptor,
        {"results": {"dtype": "uint32", "shape": [len(expected)], "values": expected}},
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [1, 1, 1], "workgroupSize": list(WORKGROUP)},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_storage_views_compile(tmp_path, target):
    _, request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text()
    assert "crosstl_resource_view_uint_as_uint8" in generated
    assert "crosstl_resource_view_uint_as_uint16" in generated
    assert "crosstl_resource_view_uint8_as_uint" in generated
    _compile(generated, target, tmp_path)


def test_storage_views_compile_on_metal(tmp_path):
    _, request, _ = _request(tmp_path, "metal")
    _compile(
        request.artifact_path.read_text(),
        "metal",
        tmp_path,
        metal_compile_flags=("-Wall", "-Wextra"),
    )


WRITER = """
struct ByteWriter {
  device uint8_t* dst;
  ByteWriter(device uint8_t* dst_, uint lane) : dst(dst_ + lane) {}
  void put(uint value) {
    dst[0] = uint8_t(value);
  }
};

[[kernel]] void write_bytes(
    device uint32_t* words [[buffer(0)]],
    uint lane [[thread_index_in_threadgroup]]) {
  ByteWriter writer((device uint8_t*)words, lane);
  writer.put(lane);
}
"""


WIDENED = """
struct WordReader {{
  const device uint32_t* src;
  WordReader(const device uint32_t* src_, uint lane) : src(src_ + lane) {{}}
  uint read() const {{
    return src[0];
  }}
}};

[[kernel]] void gather_views(
    const device {binding}* bytes [[buffer(0)]],
    device uint* results [[buffer(1)]],
    uint lane [[thread_index_in_threadgroup]]) {{
  WordReader reader((const device uint32_t*)({view}), lane);
  results[lane] = reader.read();
}}
"""

# Element-level views: storage a later view adds to byte handles reaches the
# word handles an earlier view widened them to.
LATE_NARROW = """
struct WordReader {
  const device uint32_t* src;
  WordReader(const device uint32_t* src_, uint lane) : src(src_ + lane) {}
  uint read() const {
    return src[0];
  }
};

[[kernel]] void gather_views(
    const device uint32_t* words [[buffer(0)]],
    const device uint16_t* halves [[buffer(1)]],
    device uint* results [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]]) {
  auto bytes = (const device uint8_t*)words;
  WordReader reader((const device uint32_t*)bytes, lane);
  auto narrow = (const device uint8_t*)halves;
  results[lane] = reader.read() + uint(narrow[lane]);
}
"""


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize(
    "source",
    (
        # A narrower store would race with its word's other elements.
        pytest.param(WRITER.replace("write_bytes", ENTRY), id="byte-store"),
        # Sign extension of a signed view is not modelled.
        pytest.param(_source("const device int8_t*"), id="signed-view"),
        # Words of a byte binding would combine four bindings' elements.
        pytest.param(
            WIDENED.format(binding="uint8_t", view="bytes"), id="widened-bytes"
        ),
        # Halfword storage holds no whole words.
        pytest.param(
            WIDENED.format(binding="uint16_t", view="(const device uint8_t*)bytes"),
            id="widened-halves",
        ),
    ),
)
def test_unsupported_storage_views_fail_closed(tmp_path, target, source):
    if "ByteWriter" in source:
        source = "#include <metal_stdlib>\nusing namespace metal;\n" + source
    with pytest.raises(AssertionError, match="pointer-contract-mismatch"):
        _request(tmp_path, target, source)


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_late_views_of_widened_handles_fail_closed(tmp_path, target):
    source = "#include <metal_stdlib>\nusing namespace metal;\n" + LATE_NARROW
    with pytest.raises(AssertionError, match="widened-narrow-storage"):
        _request(tmp_path, target, source)


def test_storage_views_execute(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native storage views")
    from tests.test_translator.test_loop_updates import _execute

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source, request, expected = _request(tmp_path, target)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry=ENTRY,
        metal_compile_flags=("-Wall", "-Wextra"),
        compare_with_original=target == "metal",
    )


def test_storage_views_are_required_on_all_native_platforms():
    from pathlib import Path

    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_resource_storage_views.py" in step
    assert "if:" not in step and "continue-on-error" not in step
