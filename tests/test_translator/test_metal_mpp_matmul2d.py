"""MetalPerformancePrimitives' matmul2d on cooperative tensors (crosstl/backend/Metal/mpp_matmul2d.py).

The kernels multiply integer-valued matrices, so every product and partial sum is exact in float and any
summation order gives the CPU reference bit for bit.
"""

import hashlib
import itertools
import json
import os
import re
import sys
from pathlib import Path

import pytest

from crosstl.backend.Metal.mpp_matmul2d import (
    LAYOUTS,
    MetalMatmul2dError,
    _Variant,
    lower_matmul2d_operations,
)
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.project import (
    ProjectConfig,
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    translate_project,
)
from tests.ci_helpers import assert_native_step_prerequisites, assert_paths_covered
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_MPP_MATMUL2D_RUNTIME"
TARGETS = ("directx", "opengl", "metal")
SHAPES = (
    (16, 16, 32),
    (16, 32, 16),
    (16, 32, 32),
    (32, 16, 16),
    (32, 16, 32),
    (32, 32, 16),
    (32, 32, 32),
)
# The layouts an Apple M2 Max reported through get_multidimensional_index (macOS 27.2 SDK, -std=metal4.0) for
# the left, right and destination tensors of every shape above and every transpose pair, serialised as
# _measured_layout_digest does. They did not depend on the element types (half, bfloat or float inputs; half or
# float destinations), and every element was valid.
MEASURED_SIMDGROUP_TILES_DIGEST = (
    "20e19dbdbfbc9adfeb8c2b1fe6c25acf58ab9705635745810daad0bb60b1fd1c"
)
HEADER = """#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
"""
MODES = {
    0: "mpp::tensor_ops::matmul2d_descriptor::mode::multiply",
    1: "mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate",
}


def _measured_layout_digest(layout):
    digest = hashlib.sha256()
    for (m, n, k), left, right in itertools.product(SHAPES, (0, 1), (0, 1)):
        digest.update(f"{m}x{n}x{k}:{left}{right}\n".encode())
        for rows, columns, transposed in ((m, k, left), (k, n, right), (m, n, 0)):
            lanes = _Variant(layout, rows, columns, bool(transposed)).stored()
            digest.update(
                ";".join(
                    ",".join(f"{r}:{c}" for r, c in lane) for lane in lanes
                ).encode()
                + b"\n"
            )
    return digest.hexdigest()


def test_default_layout_is_the_measured_apple_layout():
    assert (
        _measured_layout_digest("simdgroup_tiles_8x8")
        == MEASURED_SIMDGROUP_TILES_DIGEST
    )
    # Lane 0 of a 16x16 left operand holds rows 0 and 8, columns 0, 1, 8 and 9 (row, column).
    lane = _Variant("simdgroup_tiles_8x8", 16, 16, False).stored()[0]
    assert lane == ((0, 0), (0, 1), (0, 8), (0, 9), (8, 0), (8, 1), (8, 8), (8, 9))


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("shape", [(16, 16), (16, 32), (32, 16), (32, 32)])
def test_every_layout_places_each_element_once(layout, transposed, shape):
    variant = _Variant(layout, *shape, transposed)
    if layout == "fragments_16x16" and shape == (32, 32):
        with pytest.raises(MetalMatmul2dError, match="one or two 16x16 fragments"):
            variant.owners()
        return
    owners = variant.owners()
    assert len(owners) == shape[0] * shape[1]
    assert all(len(lane) == variant.capacity for lane in variant.stored())


def test_fragment_layout_keeps_four_adjacent_columns_in_stored_order():
    stored = _Variant("fragments_16x16", 32, 16, True).stored()
    assert stored[0][:4] == ((0, 0), (0, 1), (0, 2), (0, 3))
    assert stored[0][4:8] == ((8, 0), (8, 1), (8, 2), (8, 3))
    # The second fragment of a stored 16x32 operand holds its columns 16 to 31.
    assert stored[0][8] == (0, 16)


def _portable_kernel(m, n, k, transpose_left, transpose_right, mode):
    """A kernel that places every element through get_multidimensional_index, as MPP documents."""
    left_columns = m if transpose_left else k
    right_columns = k if transpose_right else n
    flags = f"{str(bool(transpose_left)).lower()}, {str(bool(transpose_right)).lower()}"
    return HEADER + f"""
kernel void tile_mma(device const float* a [[buffer(0)]],
                     device const float* b [[buffer(1)]],
                     device float* c [[buffer(2)]]) {{
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor({m}, {n}, {k}, {flags}, true,
      {MODES[mode]});
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto ct_a = op.template get_left_input_cooperative_tensor<half, half, float>();
  auto ct_b = op.template get_right_input_cooperative_tensor<half, half, float>();
  auto ct_c = op.template get_destination_cooperative_tensor<
      metal::remove_addrspace_t<decltype(ct_a)>,
      metal::remove_addrspace_t<decltype(ct_b)>, float>();
  for (ushort i = 0; i < ct_a.get_capacity(); i++) {{
    auto ix = ct_a.get_multidimensional_index(i);  // {{column, row}}
    ct_a[i] = half(a[ix[1] * {left_columns} + ix[0]]);
  }}
  for (ushort i = 0; i < ct_b.get_capacity(); i++) {{
    auto ix = ct_b.get_multidimensional_index(i);
    ct_b[i] = half(b[ix[1] * {right_columns} + ix[0]]);
  }}
  for (ushort i = 0; i < ct_c.get_capacity(); i++) {{
    if (ct_c.is_valid_element(i)) {{
      auto ix = ct_c.get_multidimensional_index(i);
      ct_c[i] = c[ix[1] * {n} + ix[0]];
    }}
  }}
  op.run(ct_a, ct_b, ct_c);
  for (ushort i = 0; i < ct_c.get_capacity(); i++) {{
    auto ix = ct_c.get_multidimensional_index(i);
    c[ix[1] * {n} + ix[0]] = ct_c[i];
  }}
}}
"""


def _fragment_kernel(transpose_left, transpose_right, mode):
    """A templated kernel that indexes elements by the 16x16-fragment lane layout, as NAX kernels do."""
    instantiation = (
        f"tile_mma<half, {str(bool(transpose_left)).lower()}, "
        f"{str(bool(transpose_right)).lower()}, {MODES[mode]}>"
    )
    return HEADER + f"""
// The {{column, row}} of element idx of a 16x16 fragment: rows r and r + 8, four columns from c.
inline short2 fragment_coordinate(ushort lane, short idx) {{
  const short q = lane >> 2;
  return short2(((q & 2) | (lane & 1)) * 4 + idx % 4, ((q & 4) | ((lane >> 1) & 3)) + (idx >> 2) * 8);
}}

template <typename T, bool transpose_a, bool transpose_b,
          mpp::tensor_ops::matmul2d_descriptor::mode Mode>
kernel void tile_mma(device const float* a [[buffer(0)]],
                     device const float* b [[buffer(1)]],
                     device float* c [[buffer(2)]],
                     ushort lane [[thread_index_in_simdgroup]]) {{
  // M = 16, N = 16, K = 32. A is stored 32x16 when transposed, B 16x32.
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      16, 16, 32, transpose_a, transpose_b, true, Mode);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;
  using ct_a_t = decltype(gemm_op.template get_left_input_cooperative_tensor<T, T, float>());
  using ct_b_t = decltype(gemm_op.template get_right_input_cooperative_tensor<T, T, float>());
  auto ct_a = gemm_op.template get_left_input_cooperative_tensor<T, T, float>();
  auto ct_b = gemm_op.template get_right_input_cooperative_tensor<T, T, float>();
  auto ct_c = gemm_op.template get_destination_cooperative_tensor<ct_a_t, ct_b_t, float>();
  constexpr short a_columns = transpose_a ? 16 : 32;
  constexpr short b_columns = transpose_b ? 32 : 16;
  for (short i = 0; i < 16; i++) {{
    const short2 p = fragment_coordinate(lane, i % 8);
    const short f = i / 8;
    ct_a[i] = T(a[((f / (a_columns / 16)) * 16 + p.y) * a_columns + (f % (a_columns / 16)) * 16 + p.x]);
    ct_b[i] = T(b[((f / (b_columns / 16)) * 16 + p.y) * b_columns + (f % (b_columns / 16)) * 16 + p.x]);
  }}
  for (short i = 0; i < 8; i++) {{
    const short2 p = fragment_coordinate(lane, i);
    ct_c[i] = c[p.y * 16 + p.x];
  }}
  gemm_op.run(ct_a, ct_b, ct_c);
  for (short i = 0; i < 8; i++) {{
    const short2 p = fragment_coordinate(lane, i);
    c[p.y * 16 + p.x] = ct_c[i];
  }}
}}

template [[host_name("tile_mma")]] [[kernel]] decltype({instantiation}) {instantiation};
"""


def _inputs(m, n, k):
    a = [float((7 * i) % 11 - 5) for i in range(m * k)]
    b = [float((5 * i) % 13 - 6) for i in range(k * n)]
    c = [float(i % 3) for i in range(m * n)]
    return a, b, c


def _expected(m, n, k, transpose_left, transpose_right, mode):
    a, b, c = _inputs(m, n, k)

    def left(r, kk):
        return a[kk * m + r] if transpose_left else a[r * k + kk]

    def right(kk, col):
        return b[col * k + kk] if transpose_right else b[kk * n + col]

    return [
        (c[r * n + col] if mode else 0.0)
        + sum(left(r, kk) * right(kk, col) for kk in range(k))
        for r in range(m)
        for col in range(n)
    ]


def _package(root, target, source, layout):
    (root / "tile.metal").write_text(source, encoding="utf-8")
    metal = {}
    if target != "metal":
        target_options = {"software_subgroup_width": 32}
        if target == "directx":
            target_options["relative_wave_shuffle_out_of_range"] = "self"
        metal["target_options"] = {target: target_options}
    if layout is not None:
        metal["cooperative_tensor_layout"] = layout
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("tile.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=(32, 1, 1),
            source_options={"metal": metal},
        ),
        format_output=False,
    )
    report.write_json(root / "report.json")
    assert report.to_json()["summary"]["failedCount"] == 0, json.dumps(report.to_json())
    manifest = build_runtime_artifact_manifest(root / "report.json")
    assert manifest["success"], manifest
    (root / "artifacts.json").write_text(json.dumps(manifest))
    package = root / "package"
    assert build_runtime_package(root / "artifacts.json", package)["success"]
    loader = build_runtime_loader_manifest(package / "runtime-package.json")
    assert loader["success"] and len(loader["loadUnits"]) == 1, loader
    descriptor = build_native_loader_abi_descriptor(
        loader, load_unit_id=loader["loadUnits"][0]["id"]
    )
    return descriptor, package


def _request(root, target, source, case, operand_type, layout=None):
    """The dispatch of ``source`` for ``case`` = (shape, transpose_left, transpose_right, mode)."""
    shape = case[0]
    descriptor, package = _package(root, target, source, layout)
    a, b, c = _inputs(*shape)
    inputs = {
        "a": {"dtype": operand_type, "shape": [len(a)], "values": a},
        "b": {"dtype": operand_type, "shape": [len(b)], "values": b},
        "c": {"dtype": "float32", "shape": [len(c)], "values": c},
    }
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        _outputs(descriptor, _expected(*shape, *case[1:])),
        {"workgroupCount": [1, 1, 1], "workgroupSize": [32, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return request, _outputs(descriptor, _expected(*shape, *case[1:]))


def _outputs(descriptor, values):
    return _bound_values(
        descriptor,
        {"c": {"dtype": "float32", "shape": [len(values)], "values": values}},
    )


PORTABLE_CASES = [
    ((16, 32, 16), 0, 0, 1),
    ((16, 16, 32), 1, 0, 1),
    ((32, 32, 32), 1, 1, 0),
    ((32, 16, 16), 0, 1, 1),
]
FRAGMENT_CASES = list(itertools.product((0, 1), (0, 1), (0, 1)))


def _ids(cases):
    return ["-".join(str(part) for part in case) for case in cases]


@pytest.mark.parametrize("target", TARGETS)
def test_portable_kernel_translates_and_compiles(tmp_path, target):
    case = PORTABLE_CASES[1]
    request, _ = _request(
        tmp_path, target, _portable_kernel(*case[0], *case[1:]), case, "float32"
    )
    generated = request.artifact_path.read_text()
    assert "MetalPerformancePrimitives" not in generated
    assert not re.search(r"\bmatmul2d|cooperative_tensor\b|\bmpp\b", generated)
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", TARGETS)
def test_fragment_kernel_translates_and_compiles(tmp_path, target):
    request, _ = _request(
        tmp_path,
        target,
        _fragment_kernel(1, 1, 1),
        ((16, 16, 32), 1, 1, 1),
        "float32",
        layout="fragments_16x16",
    )
    _compile(request.artifact_path.read_text(), target, tmp_path)


def _native_target():
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required matmul2d execution")
    return {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]


@pytest.mark.parametrize("case", PORTABLE_CASES, ids=_ids(PORTABLE_CASES))
def test_portable_kernel_matches_the_reference_natively(tmp_path, case):
    target = _native_target()
    shape, left, right, mode = case
    source = _portable_kernel(*shape, left, right, mode)
    request, expected = _request(tmp_path, target, source, case, "float32")
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="tile_mma",
        metal_language_version="4.0" if target == "metal" else None,
    )


@pytest.mark.parametrize("case", FRAGMENT_CASES, ids=_ids(FRAGMENT_CASES))
def test_fragment_kernel_matches_the_reference_natively(tmp_path, case):
    target = _native_target()
    left, right, mode = case
    request, expected = _request(
        tmp_path,
        target,
        _fragment_kernel(left, right, mode),
        ((16, 16, 32), left, right, mode),
        "float32",
        layout="fragments_16x16",
    )
    _execute(request, expected, tmp_path)


def _lower(source, layout=None):
    preprocessor = MetalPreprocessor(cooperative_tensor_layout=layout)
    return lower_matmul2d_operations(preprocessor, source)


def test_lowered_source_names_no_library_entity():
    lowered = _lower(_portable_kernel(16, 32, 16, 0, 0, 1))
    assert "MetalPerformancePrimitives" not in lowered
    assert "matmul2d" not in lowered and "get_capacity" not in lowered
    assert "half ct_a[8] = {};" in lowered and "float ct_c[16] = {};" in lowered


def test_uniform_layouts_shuffle_once_per_operand_element():
    lowered = _lower(_portable_kernel(16, 32, 16, 1, 1, 1))
    assert lowered.count("simd_shuffle(") == 2
    assert "_candidates" not in lowered


def test_transposed_fragment_operands_select_among_candidate_elements():
    lowered = _lower(_fragment_kernel(1, 0, 1), layout="fragments_16x16")
    assert "_candidates" in lowered
    assert "(transpose_a ? " in lowered and "(transpose_b ? " in lowered
    assert "int Mode = " not in lowered and "int Mode>" in lowered


REFUSALS = {
    "symbolic-size": (
        ("(16, 32, 16,", "(M, 32, 16,"),
        "M must be an integer literal",
    ),
    "eight": (("(16, 32, 16,", "(8, 32, 16,"), "M, N and K of 16 or 32"),
    "no-32": (("(16, 32, 16,", "(16, 16, 16,"), "at least one of them 32"),
    "simdgroups": (
        ("metal::execution_simdgroup>", "metal::execution_simdgroups<4>>"),
        "one SIMD-group",
    ),
    "hardcoded": (
        (
            "auto ix = ct_a.get_multidimensional_index(i);  // {column, row}\n"
            "    ct_a[i] = half(a[ix[1] * 16 + ix[0]]);",
            "ct_a[i] = a[i];",
        ),
        "fixed lane layout",
    ),
    "passed": (
        ("op.run(ct_a, ct_b, ct_c);", "op.run(ct_a, ct_b, ct_c); use(ct_c);"),
        "used other than",
    ),
    "argument-order": (
        ("op.run(ct_a, ct_b, ct_c);", "op.run(ct_b, ct_a, ct_c);"),
        "in that order",
    ),
    "half-destination": (
        ("ct_b)>, float>();", "ct_b)>, half>();"),
        "destination cooperative tensor's element type must be",
    ),
    "descriptor-use": (
        ("op.run(ct_a, ct_b, ct_c);", "op.run(ct_a, ct_b, ct_c); c[0] = desc.m;"),
        "used other than to declare",
    ),
    "operation-use": (
        (
            "op.run(ct_a, ct_b, ct_c);",
            "op.run(ct_a, ct_b, ct_c); op.run(ct_a, ct_b, ct_c, 1);",
        ),
        "in that order",
    ),
    "reserved": (
        ("kernel void", "constant int crosstl_mpp_x = 0;\nkernel void"),
        "reserved",
    ),
}


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_unprovable_operations_are_refused(name):
    (old, new), message = REFUSALS[name]
    source = _portable_kernel(16, 32, 16, 0, 0, 1)
    assert old in source
    with pytest.raises(MetalMatmul2dError, match=re.escape(message)):
        _lower(source.replace(old, new, 1))


def test_half_accumulation_is_refused():
    source = _portable_kernel(16, 32, 16, 0, 0, 1).replace("float>()", "half>()")
    with pytest.raises(MetalMatmul2dError, match="float destinations"):
        _lower(source)


def test_symbolic_destination_type_is_checked_after_materialization():
    source = _fragment_kernel(0, 0, 1).replace("T, T, float>", "T, T, T>")
    lowered = _lower(
        source.replace("ct_b_t, float>", "ct_b_t, T>"), layout="fragments_16x16"
    )
    assert "static_assert(metal::is_same_v<T, float>" in lowered


def test_unknown_layout_is_refused():
    with pytest.raises(
        MetalMatmul2dError, match="cooperative_tensor_layout must be one of"
    ):
        MetalPreprocessor(cooperative_tensor_layout="row_major")


def test_unlowered_library_operations_are_refused():
    source = HEADER + """
kernel void k(tensor<device half, dextents<int32_t, 2>> a, tensor<device float, dextents<int32_t, 2>> c) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  op.run(a, a, c);
}
"""
    with pytest.raises(MetalMatmul2dError, match="in that order"):
        _lower(source)


def test_native_dispatch_gate_is_required():
    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_job_step_section(
        workflow,
        "portable-host",
        "Validate MetalPerformancePrimitives matmul2d lowering",
    )
    assert "-k natively" in step and "test_metal_mpp_matmul2d.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    # Twelve native cases ran in 96 s on six local workers (translation about 7 s each).
    assert "--timeout-seconds 150" in step
    assert_native_step_prerequisites(step, job="portable-host")
    assert "continue-on-error" not in step
    for event in ("push", "pull_request"):
        for test_file in (
            "test_metal_mpp_matmul2d.py",
            "test_metal_simdgroup_lane_queries.py",
        ):
            assert f"tests/test_translator/{test_file}" in step
            assert_paths_covered(
                workflow_text.workflow_event_path_filters(workflow, event),
                f"tests/test_translator/{test_file}",
            )
