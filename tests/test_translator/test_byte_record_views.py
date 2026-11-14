"""Records of one fixed byte array keep their bytes through pointer views.

A kernel copies a 12-byte record out of a byte view of word storage at an
arbitrary byte offset, then reads it through private ``uint8_t`` and
``int8_t`` views::

    cursor += tid * 13 + 1;
    record = *((const device Record*)cursor);
    results[2 * tid] = weigh((const thread uint8_t*)&record);

Function-local records carry their native alignment as layout metadata,
which neither view may treat as noncanonical. HLSL keeps one element per
byte, so both views must address bytes, not words.
"""

import os
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_BYTE_RECORD_VIEWS"
TARGETS = ("metal", "directx", "opengl")
RECORD_BYTES = 12
STRIDE = 13
THREADS = 5
GUARD = 0xDEADBEEF

GLOBAL_RECORD = """struct Record {
  uint8_t bytes[12];
};
"""
LOCAL_RECORD = """  typedef struct {
    uint8_t bytes[12];
  } Record;
"""


def _source(scope):
    return f"""#include <metal_stdlib>
using namespace metal;

{GLOBAL_RECORD if scope == "global" else ""}
inline uint weigh(const thread uint8_t* bytes) {{
  uint total = 0;
  for (int i = 0; i < {RECORD_BYTES}; i++) {{
    total += uint(bytes[i]) * uint(i + 1);
  }}
  return total;
}}

inline int fold(const thread int8_t* bytes) {{
  int total = 0;
  for (int i = 0; i < {RECORD_BYTES}; i++) {{
    total = total * 3 + int(bytes[i]);
  }}
  return total;
}}

kernel void records(
    const device uint32_t* words [[buffer(0)]],
    device uint* results [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {{
{LOCAL_RECORD if scope == "local" else ""}
  const device uint8_t* cursor = (const device uint8_t*)words;
  cursor += tid * {STRIDE} + 1;
  thread Record record;
  record = *((const device Record*)cursor);
  results[2 * tid] = weigh((const thread uint8_t*)&record);
  results[2 * tid + 1] = uint(fold((const thread int8_t*)&record));
}}
"""


def _words():
    count = -(-(THREADS * STRIDE + 1) // 4)
    # Every byte value, signed bytes included, appears across the records.
    data = bytes((index * 167 + 89) % 256 for index in range(count * 4))
    words = [int.from_bytes(data[i : i + 4], "little") for i in range(0, len(data), 4)]
    return data, words


def _reference():
    data, words = _words()
    results = []
    for tid in range(THREADS):
        record = data[tid * STRIDE + 1 : tid * STRIDE + 1 + RECORD_BYTES]
        weighed = sum(value * (index + 1) for index, value in enumerate(record))
        folded = 0
        for value in record:
            folded = folded * 3 + (value - 256 if value >= 128 else value)
        results += [weighed, folded & 0xFFFFFFFF]
    return words, results


def test_reference_reads_signed_and_unaligned_bytes():
    data, _ = _words()
    assert any(value >= 128 for value in data) and any(value < 128 for value in data)
    # Records start on every byte position within a word.
    assert {(tid * STRIDE + 1) % 4 for tid in range(THREADS)} == {0, 1, 2, 3}
    _, results = _reference()
    assert any(value > 0x7FFFFFFF for value in results[1::2])


def _request(root, target, scope):
    source, descriptor, package = _package(
        root,
        target,
        "uint",
        (1, 1, 1),
        source=_source(scope),
        software_subgroups=False,
        preserve_resource_origins=True,
    )
    words, results = _reference()

    def typed(values):
        return {"dtype": "uint32", "shape": [len(values)], "values": values}

    expected = _bound_values(descriptor, {"results": typed(results + [GUARD] * 4)})
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(
            descriptor,
            {
                "words": typed(words),
                "results": typed([GUARD] * (len(results) + 4)),
            },
        ),
        expected,
        {"workgroupCount": [THREADS, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, expected


@pytest.mark.parametrize("scope", ["local", "global"])
@pytest.mark.parametrize("target", TARGETS)
def test_byte_record_views_translate_and_compile(tmp_path, target, scope):
    _, request, _ = _request(tmp_path, target, scope)
    generated = request.artifact_path.read_text()
    if target == "directx":
        # One element per byte: the record copies byte by byte and the views
        # read element i, signed bytes sign-extended.
        assert "uint bytes[12]" in generated
        assert "const uint bytes[12], int bytes_base" in generated
        assert "& 255u" in generated and ">> 24" in generated
        assert "PointerReinterpretNode" not in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("scope", ["local", "global"])
def test_byte_record_views_execute(tmp_path, scope):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required byte record view execution")
    target = {"darwin": "metal", "win32": "directx"}.get(sys.platform, "opengl")
    source, request, expected = _request(tmp_path, target, scope)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="records",
    )


def _diagnostics(tmp_path, source):
    from crosstl.project import ProjectConfig, translate_project

    (tmp_path / "records.metal").write_text(source, encoding="utf-8")
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=("records.metal",),
            targets=("directx",),
            output_dir="out",
            source_options={"metal": {"preserve_resource_origins": True}},
        ),
        format_output=False,
    ).to_json()
    return [
        item["details"]["pointerReinterpretation"] for item in report["diagnostics"]
    ]


@pytest.mark.parametrize(
    ("record", "view", "reason"),
    [
        pytest.param(
            "uint8_t bytes[12]; uint tag;",
            "results[0] = weigh((const thread uint8_t*)&record);",
            "heterogeneous-private-view-layout",
            id="bytes-beside-words",
        ),
        pytest.param(
            "uint words[3];",
            "record = *((const device Record*)cursor); results[0] = record.words[0];",
            "unsupported-storage-view-target-layout",
            id="bytes-into-words",
        ),
    ],
)
def test_byte_record_views_reject_mixed_layouts(tmp_path, record, view, reason):
    source = f"""#include <metal_stdlib>
using namespace metal;

inline uint weigh(const thread uint8_t* bytes) {{
  return uint(bytes[0]) + uint(bytes[11]);
}}

kernel void records(
    const device uint32_t* words [[buffer(0)]],
    device uint* results [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {{
  typedef struct {{
    {record}
  }} Record;
  const device uint8_t* cursor = (const device uint8_t*)words;
  cursor += tid;
  thread Record record;
  {view}
}}
"""
    (diagnostic,) = _diagnostics(tmp_path, source)
    assert diagnostic["reason"] == reason
    assert diagnostic["targetBackend"] == "directx"


def test_byte_record_views_are_required_on_every_native_target():
    from pathlib import Path

    from tests import workflow_text

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    step = workflow_text.workflow_step_section(
        workflow, "Validate collective helper arguments"
    )
    assert "tests/test_translator/test_byte_record_views.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "continue-on-error" not in step
