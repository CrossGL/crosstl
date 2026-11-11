"""OpenGL subscripts bounded through affine relations between integer locals.

A loop that walks rows from the end keeps the forward row in ``check`` and
reverses ``row`` in place: ``uint check = row; row = size - 1 - row;``. Only
``check < size`` guards the access ``source[row * stride + i]``. Interval
facts bound ``check`` but lose ``row == size - 1 - check (mod 2**32)``; the
relation bounds ``row`` exactly when ``size - 1 - check`` cannot wrap.
"""

import pytest

import crosstl.translator
from crosstl.project import ProjectConfig, translate_project
from crosstl.translator.codegen import GLSL_codegen, glsl_affine_relations
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen, OpenGLIndexTypeError
from tests.test_translator.test_metal_builtin_ownership import _compile

SHADER = """
shader ReversedRows {
    compute {
        layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

        void main(
            StructuredBuffer<float> source @buffer(0),
            RWStructuredBuffer<float> result @buffer(1),
            uvec3 lid @gl_LocalInvocationID,
            uvec3 gid @gl_WorkGroupID
        ) {
            uint64 size = uint64(37);
            uint64 stride = uint64(5);
            PROLOGUE
            for (uint j = 0u; j < size; j += 32u) {
                uint row = j + lid.x;
                uint check = row;
                row = size - 1u - row;
                STEP
                if (GUARD) {
                    result[gid.x] = source[row * stride + gid.x % 5u];
                }
            }
        }
    }
}
"""


def _generate(guard="check < size", step="", prologue=""):
    shader = (
        SHADER.replace("GUARD", guard)
        .replace("STEP", step)
        .replace("PROLOGUE", prologue)
    )
    return GLSLCodeGen().generate(crosstl.translator.parse(shader))


def _subscript(generated):
    (line,) = (line for line in generated.splitlines() if "* stride" in line)
    return line.strip()


@pytest.mark.parametrize(
    "step",
    [
        "",
        # Paired steps keep row + check constant.
        "row -= 1u; check += 1u;",
        # check + 1 < size still bounds row = size - check.
        "check += 1u;",
        # row = j - check / 2 + 36 needs j's own interval as well.
        "check = check * 2u;",
    ],
)
def test_guard_on_a_related_local_bounds_the_reversed_row(tmp_path, step):
    generated = _generate(step=step)
    assert "source[uint(((uint64_t(row) * stride)" in _subscript(generated)
    _compile(generated, "opengl", tmp_path)


def test_reversed_row_is_unproven_without_the_relation(monkeypatch):
    monkeypatch.setattr(
        GLSL_codegen, "related_local_interval", lambda *args, **kwargs: None
    )
    with pytest.raises(OpenGLIndexTypeError, match="index range out of target range"):
        _generate()


@pytest.mark.parametrize(
    "guard, step, prologue",
    [
        # No guard: row wraps for j + lid.x > 36.
        ("true", "", ""),
        # check == size makes size - 1 - check wrap to 0xffffffff.
        ("check <= size", "", ""),
        # A new value for check breaks the relation; check < size always holds.
        ("check < size", "check = lid.x;", ""),
        # The relation holds on one branch only.
        ("check < size", "if (lid.x > 3u) { check = 0u; }", ""),
    ],
)
def test_relations_that_do_not_bound_the_row_are_refused(guard, step, prologue):
    with pytest.raises(OpenGLIndexTypeError, match="index range out of target range"):
        _generate(guard=guard, step=step, prologue=prologue)


LOOP_CARRIED = """
shader LoopCarriedRows {
    compute {
        layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

        void main(
            StructuredBuffer<float> source @buffer(0),
            RWStructuredBuffer<float> result @buffer(1),
            uvec3 lid @gl_LocalInvocationID
        ) {
            uint64 size = uint64(37);
            uint64 stride = uint64(5);
            uint row = lid.x;
            uint check = row;
            row = size - 1u - row;
            for (uint k = 0u; k < 2u; k++) {
                if (check < size) {
                    result[lid.x] = source[row * stride];
                }
                row -= 40u;
            }
        }
    }
}
"""


def test_relations_do_not_survive_a_loop_that_writes_them():
    # The second iteration reads row - 40, which wraps.
    with pytest.raises(OpenGLIndexTypeError, match="index range out of target range"):
        GLSLCodeGen().generate(crosstl.translator.parse(LOOP_CARRIED))


METAL_KERNEL = """#include <metal_stdlib>
using namespace metal;

template <int N>
[[kernel]] void reversed_columns(
    const device float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    const constant size_t& rows [[buffer(2)]],
    const constant size_t& stride [[buffer(3)]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  size_t offset = lid.y * rows * stride;
  in += offset;
  out += offset;
  for (uint j = 0; j < rows; j += 32 / N) {
    uint row = j + lid.x / N;
    uint check = row;
    row = rows - 1 - row;
    if (GUARD) {
      for (int i = 0; i < N; i++) {
        out[row * stride + i] = in[row * stride + i];
      }
    }
  }
}

template [[host_name("reversed_columns_4")]] [[kernel]] void reversed_columns<4>(
    const device float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    const constant size_t& rows [[buffer(2)]],
    const constant size_t& stride [[buffer(3)]],
    uint3 lid [[thread_position_in_threadgroup]]);
"""


def _translate_metal(tmp_path, guard):
    source = "reversed_columns.metal"
    (tmp_path / source).write_text(METAL_KERNEL.replace("GUARD", guard), "utf-8")
    config = ProjectConfig(
        root=tmp_path,
        include_patterns=(source,),
        targets=("opengl",),
        output_dir="out",
        entry_points={source: ("reversed_columns_4",)},
        workgroup_size_rules={source: (32, 2, 1)},
        metadata_preconditions=[
            {
                "source": source,
                "entryPoint": "reversed_columns_4",
                "parameter": name,
                "minimum": value,
                "maximum": value,
            }
            for name, value in (("rows", 37), ("stride", 5))
        ],
        source_options={"metal": {"preserve_resource_origins": True}},
    )
    payload = translate_project(config, format_output=False).to_json()
    return payload, sorted((tmp_path / "out").rglob("*.glsl"))


def test_rebased_metal_pointers_use_the_relation(tmp_path):
    payload, artifacts = _translate_metal(tmp_path, "check < rows")
    assert payload["summary"]["failedCount"] == 0, payload["diagnostics"]
    (artifact,) = artifacts
    _compile(artifact.read_text(encoding="utf-8"), "opengl", tmp_path)


def test_rebased_metal_pointers_refuse_a_wrapping_row(tmp_path):
    payload, artifacts = _translate_metal(tmp_path, "check <= rows")
    assert artifacts == []
    codes = {diagnostic["code"] for diagnostic in payload["diagnostics"]}
    assert "project.translate.opengl-index-type-unsupported" in codes


# With ``rows`` known only to lie in [1, 4096], intervals give
# ``rows - 1 - check`` the range [-4095, 4095]; only the difference the guard
# states, ``check - rows + 1 <= 0``, keeps it non-negative.
RANGED_ROWS = """#include <metal_stdlib>
using namespace metal;

template <int N>
[[kernel]] void ranged_rows(
    const device float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    const constant size_t& rows [[buffer(2)]],
    const constant size_t& stride [[buffer(3)]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  size_t offset = lid.y * rows * stride;
  in += offset;
  out += offset;
  for (uint j = 0; j < rows; j += 32 / N) {
    uint row = j + lid.x / N;
    uint check = row;
    row = rows - 1 - row;
    if (GUARD) {
      STEP
      if (INNER) {
        for (int i = 0; i < N; i++) {
          out[row * stride + i] = in[row * stride + i];
        }
      }
    }
  }
}

template [[host_name("ranged_rows_4")]] [[kernel]] void ranged_rows<4>(
    const device float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    const constant size_t& rows [[buffer(2)]],
    const constant size_t& stride [[buffer(3)]],
    uint3 lid [[thread_position_in_threadgroup]]);
"""

SIGNED_GUARD = """#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void guarded_difference(
    const device float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    const constant size_t& rows [[buffer(2)]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  T c = T(lid.x) OFFSET;
  uint d = c - rows;
  if (c >= rows) {
    out[lid.x] = in[d * rows];
  }
}

template [[host_name("guarded_difference_t")]] [[kernel]] void
guarded_difference<TYPE>(
    const device float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    const constant size_t& rows [[buffer(2)]],
    uint3 lid [[thread_position_in_threadgroup]]);
"""


def _translate_ranged(tmp_path, kernel, entry, parameters):
    source = "ranged.metal"
    (tmp_path / source).write_text(kernel, "utf-8")
    config = ProjectConfig(
        root=tmp_path,
        include_patterns=(source,),
        targets=("opengl",),
        output_dir="out",
        entry_points={source: (entry,)},
        workgroup_size_rules={source: (32, 2, 1)},
        metadata_preconditions=[
            {
                "source": source,
                "entryPoint": entry,
                "parameter": name,
                "minimum": 1,
                "maximum": 4096,
            }
            for name in parameters
        ],
        source_options={"metal": {"preserve_resource_origins": True}},
    )
    payload = translate_project(config, format_output=False).to_json()
    return payload, sorted((tmp_path / "out").rglob("*.glsl"))


def _ranged_rows(tmp_path, guard, step="", inner="true"):
    kernel = (
        RANGED_ROWS.replace("GUARD", guard)
        .replace("STEP", step)
        .replace("INNER", inner)
    )
    return _translate_ranged(tmp_path, kernel, "ranged_rows_4", ("rows", "stride"))


def _assert_refused(payload, artifacts):
    assert artifacts == []
    codes = {diagnostic["code"] for diagnostic in payload["diagnostics"]}
    assert "project.translate.opengl-index-type-unsupported" in codes


@pytest.mark.parametrize(
    "guard, step, inner",
    [
        ("check < rows", "", "true"),
        ("rows > check", "", "true"),
        ("check < rows && check < 64", "", "true"),
        ("true", "", "check < rows"),
        # Stepping both locals keeps row + check; the new guard restates the
        # fact for the new check.
        ("check < rows", "row -= 1; check += 1;", "check < rows"),
    ],
)
def test_guard_differences_bound_rows_of_a_ranged_size(tmp_path, guard, step, inner):
    payload, artifacts = _ranged_rows(tmp_path, guard, step, inner)
    assert payload["summary"]["failedCount"] == 0, payload["diagnostics"]
    (artifact,) = artifacts
    _compile(artifact.read_text(encoding="utf-8"), "opengl", tmp_path)


def test_ranged_rows_are_unproven_without_guard_facts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        glsl_affine_relations._Relations, "_guard_facts", lambda *args: []
    )
    _assert_refused(*_ranged_rows(tmp_path, "check < rows"))


@pytest.mark.parametrize(
    "guard, step",
    [
        ("true", ""),
        # check == rows wraps row to 0xffffffff.
        ("check <= rows", ""),
        # The guard bounds a difference the subscript does not use.
        ("check < stride", ""),
        # A disjunction states neither comparison.
        ("check < rows || check < 2", ""),
        # The step changes check, so the guard no longer describes it: for
        # check == rows - 1 at the guard, row wraps.
        ("check < rows", "row -= 1; check += 1;"),
    ],
)
def test_guards_that_do_not_bound_the_ranged_row_are_refused(tmp_path, guard, step):
    _assert_refused(*_ranged_rows(tmp_path, guard, step))


def _guarded_difference(tmp_path, type_name, offset):
    kernel = SIGNED_GUARD.replace("TYPE", type_name).replace("OFFSET", offset)
    return _translate_ranged(tmp_path, kernel, "guarded_difference_t", ("rows",))


def test_an_exact_unsigned_guard_bounds_the_difference(tmp_path):
    payload, artifacts = _guarded_difference(tmp_path, "uint", "+ 16")
    assert payload["summary"]["failedCount"] == 0, payload["diagnostics"]
    (artifact,) = artifacts
    _compile(artifact.read_text(encoding="utf-8"), "opengl", tmp_path)


def test_a_signed_operand_converted_to_unsigned_states_no_fact(tmp_path, monkeypatch):
    # c < 0 converts to a value above every rows, so c >= rows holds while
    # d = c - rows wraps.
    _assert_refused(*_guarded_difference(tmp_path, "int", "- 16"))
    # Treating the comparison as exact would accept it.
    monkeypatch.setattr(glsl_affine_relations, "_exact_comparison", lambda *args: True)
    (tmp_path / "exact").mkdir()
    payload, _ = _guarded_difference(tmp_path / "exact", "int", "- 16")
    assert payload["summary"]["failedCount"] == 0, payload["diagnostics"]
