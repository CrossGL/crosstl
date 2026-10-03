"""Preserve bfloat vector representation through native Metal round trips."""

import os
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_mlx_current_gather import _validate
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_BFLOAT_VECTORS"
FORMS = (
    "named",
    "generic",
    "qualified",
    "scalar-alias",
    "chain",
    "alias",
    "conversion",
)
WORDS = (
    0x00000000,
    0x80000000,
    0x3F800000,
    0x3F807FFF,
    0x3F808000,
    0x3F808001,
    0x3F817FFF,
    0x3F818000,
    0x3F818001,
    0xBF808000,
    0xBF818000,
    0x00800000,
    0x7F7F0000,
    0x7F7F7FFF,
    0x7F7F8000,
    0x7F800000,
    0xFF800000,
    0x3F800001,
    0x3F807F80,
    0x3F807FC1,
    0x3F807F81,
    0x3F80FF81,
    0x3F807FFF,
)
GUARD = 0xDEADBEEF


def _source(width, form):
    vector = {
        "named": f"bfloat{width}",
        "generic": f"vec<bfloat, {width}>",
        "qualified": f"metal::vec<bfloat, {width}>",
        "scalar-alias": f"vec<bfloat16_t, {width}>",
        "chain": f"vec<Scalar, {width}>",
        "alias": "Lanes",
        "conversion": f"vec<bfloat16_t, {width}>",
    }[form]
    values = ", ".join(["base"] + [f"bfloat({lane}.0f)" for lane in range(1, width)])
    helper = f"{vector} make_lanes(bfloat base) {{ return {vector}({values}); }}"
    if form == "conversion":
        helper = f"""template <typename T>
struct Components {{
    T base;
    operator vec<T, {width}>() const {{ return vec<T, {width}>({values}); }}
}};
{vector} make_lanes(bfloat base) {{
    Components<bfloat16_t> item{{base}};
    return static_cast<{vector}>(item);
}}"""
    swizzle = "xyzw"[:width][::-1]
    return f"""#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
using Scalar = bfloat16_t;
using Lanes = vec<bfloat, {width}>;
{helper}
kernel void bfloat_vectors(const device uint* src [[buffer(0)]],
                           device uint* results [[buffer(1)]],
                           uint tid [[thread_position_in_grid]]) {{
    uint cursor = tid;
    bfloat base = bfloat(as_type<float>(src[cursor++]));
    {vector} lanes = make_lanes(base);
    {vector} copied = {vector}(lanes);
    {vector} reversed = copied.{swizzle};
    bfloat raw = as_type<bfloat>(ushort(src[tid] & 0xffffu));
    {vector} raw_lanes = {vector}(raw);
    uint offset = 4 + tid * {width * 4 + 3};
    for (uint lane = 0; lane < {width}; ++lane) {{
        results[offset + lane] = uint(as_type<ushort>(lanes[lane]));
        results[offset + {width} + lane] = uint(as_type<ushort>(copied[lane]));
        results[offset + {width * 2} + lane] = uint(as_type<ushort>(reversed[lane]));
        results[offset + {width * 3} + lane] = uint(as_type<ushort>(raw_lanes[lane]));
    }}
    results[offset + {width * 4}] = uint(as_type<ushort>(copied.x));
    results[offset + {width * 4 + 1}] = uint(sizeof({vector}));
    results[offset + {width * 4 + 2}] = cursor;
}}
"""


def _rounded(word):
    return ((word + 0x7FFF + ((word >> 16) & 1)) >> 16) & 0xFFFF


def _request(root, width, form):
    source, descriptor, package = _package(
        root,
        "metal",
        "uint",
        (1, 1, 1),
        source=_source(width, form),
        software_subgroups=False,
    )
    expected = [GUARD] * 4
    for tid, word in enumerate(WORDS):
        lanes = [_rounded(word)] + [0x3F80, 0x4000, 0x4040][: width - 1]
        expected.extend(lanes + lanes + list(reversed(lanes)))
        expected.extend([word & 0xFFFF] * width)
        expected.extend([lanes[0], 2 * (4 if width == 3 else width), tid + 1])
    expected += [GUARD] * 4

    def values(items):
        return {"dtype": "uint32", "shape": [len(items)], "values": list(items)}

    outputs = _bound_values(descriptor, {"results": values(expected)})
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(
            descriptor,
            {"src": values(WORDS), "results": values([GUARD] * len(expected))},
        ),
        outputs,
        {"workgroupCount": [len(WORDS), 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target="metal",
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("form", FORMS)
def test_bfloat_vectors_preserve_native_types(tmp_path, width, form):
    _, request, _ = _request(tmp_path, width, form)
    generated = request.artifact_path.read_text()
    assert f"bfloat{width} make_lanes(bfloat base)" in generated
    assert f"bfloat{width} copied" in generated
    assert "as_type<ushort>(copied.x)" in generated
    assert "vec<" not in generated
    if sys.platform == "darwin":
        assert _validate(request.artifact_path, tmp_path, "metal").is_file()


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("form", FORMS)
def test_bfloat_vectors_execute_natively(tmp_path, width, form):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required bfloat-vector execution")
    assert sys.platform == "darwin", "the bfloat round-trip gate requires native Metal"
    source, request, outputs = _request(tmp_path, width, form)
    _execute(
        request,
        outputs,
        tmp_path,
        original_source=source,
        original_entry="bfloat_vectors",
        metal_compile_flags=("-fno-fast-math",),
        validate=_validate,
    )


def test_bfloat_vector_gate_requires_original_and_generated_execution():
    from pathlib import Path

    from tools import ci_coverage

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/mlx-metal-host.yml").read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate bfloat vector round trips"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_metal_bfloat_vectors.py" in step
    assert "--timeout-seconds 180" in step and "--junitxml=" in step
    assert "--basetemp=" in step and "-n auto" in step
    assert "continue-on-error" not in step and "if:" not in step
    for event in ("pull_request", "push"):
        assert (
            "tests/test_translator/test_metal_bfloat_vectors.py"
            in ci_coverage.workflow_event_path_filters(workflow, event)
        )
