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
from crosstl.translator.codegen import GLSL_codegen
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
