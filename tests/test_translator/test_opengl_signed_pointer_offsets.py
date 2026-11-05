"""OpenGL logical pointer offsets that may be negative before guarded access.

MLX's reverse scans pass ``in + axis_size - offset - N_READS`` to a guarded
loader. The pointer can precede the row (and the buffer) when the block is
partial; only accesses with ``start + N - i - 1 < total`` read through it.
"""

import re

import pytest

import crosstl.translator
from crosstl.project import ProjectConfig, translate_project
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen, OpenGLIndexTypeError
from tests.test_translator.test_metal_builtin_ownership import _compile

SHADER = """
shader ReverseGuardedLoads {
    void load_safe(inout float[4] values, const device float* input, int start, int total) {
        for (int i = 0; i < 4; i++) {
            values[4 - i - 1] = start + 4 - i - 1 < total ? input[i] : 0.0;
        }
    }

    compute {
        layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

        void main(
            StructuredBuffer<float> source @buffer(0),
            RWStructuredBuffer<float> result @buffer(1),
            uvec3 lid @gl_LocalInvocationID
        ) {
            uint64 total = TOTAL;
            uint offset = lid.x * 4u;
            float[4] values;
            load_safe(values, source + total - offset - 4, int(offset), int(total));
            result[lid.x] = values[0] + values[3];
        }
    }
}
"""
ACCESS = {"expression": "input + i", "minimum": 0, "maximum": 2**31 - 1}


def _generate(total="uint64(6)", assertions=()):
    return (
        GLSLCodeGen()
        .set_index_range_assertions(list(assertions))
        .generate(crosstl.translator.parse(SHADER.replace("TOTAL", total)))
    )


def test_negative_logical_offset_is_computed_exactly_in_int(tmp_path):
    generated = _generate(assertions=[ACCESS])
    # 6 - 124 - 4 < 0: each delta fits int and so does every partial sum, so the
    # termwise int arithmetic is the exact logical offset (no uint64 wrap).
    call = re.search(r"load_safe\w*\(values, [^;]*\);", generated)
    assert call is not None, generated
    assert "- int(offset)) - 4)" in call.group(0), call.group(0)
    assert "uint64_t" not in call.group(0), call.group(0)
    assert "input_offset + i" in generated
    _compile(generated, "opengl", tmp_path)


GUARD = "start + 4 - i - 1 < total ? input[i] : 0.0"
CALL = "load_safe(values, source + total - offset - 4, int(offset), int(total));"


def _guarded(guard=GUARD, call=CALL, prologue="", total="uint64(6)"):
    shader = SHADER.replace("TOTAL", total).replace(GUARD, guard).replace(CALL, call)
    shader = shader.replace(
        "        for (int i = 0; i < 4; i++) {",
        prologue + "        for (int i = 0; i < 4; i++) {",
        1,
    )
    return GLSLCodeGen().generate(crosstl.translator.parse(shader))


@pytest.mark.parametrize(
    "guard",
    (
        GUARD,
        "!(start + 4 - i - 1 >= total) ? input[i] : 0.0",
        "i >= 0 && total > start + 3 - i ? input[i] : 0.0",
    ),
)
def test_guarded_accesses_through_negative_offsets_are_proved(tmp_path, guard):
    # The call's base is ``total - start - 4`` in the helper's own scalars, and
    # the dominating guard ``start + 4 - i - 1 < total`` gives
    # ``base + i >= 0``; the base is at most ``total``, so the subscript fits int.
    generated = _guarded(guard)
    assert "source[(input_offset + i)]" in generated
    _compile(generated, "opengl", tmp_path)


@pytest.mark.parametrize(
    "total",
    ("uint64(2000000)", "uint64(asuint(source[1]) & 0xffffu)"),
)
def test_guard_proofs_hold_for_bounded_unknown_totals(tmp_path, total):
    generated = _guarded(total=total)
    assert "source[(input_offset + i)]" in generated
    _compile(generated, "opengl", tmp_path)


@pytest.mark.parametrize(
    "guard, call, prologue",
    (
        # One element short: ``base + i >= -1`` admits a read before the buffer.
        ("start + 4 - i - 2 < total ? input[i] : 0.0", CALL, ""),
        ("start + 4 - i - 1 <= total ? input[i] : 0.0", CALL, ""),
        # No guard at all.
        ("input[i]", CALL, ""),
        # The guard compares other values than the base was built from: with
        # ``start = offset - 1`` it admits ``base + i == -1``.
        (
            GUARD,
            "load_safe(values, source + total - offset - 4, int(offset) - 1,"
            " int(total));",
            "",
        ),
        (
            GUARD,
            "load_safe(values, source + total - offset - 4, int(total),"
            " int(offset));",
            "",
        ),
        # The helper rebinds a parameter the relation names.
        (GUARD, CALL, "        total = total + 1;\n"),
        (GUARD, CALL, "        start = start - 1;\n"),
    ),
)
def test_guards_that_do_not_bound_the_base_are_refused(guard, call, prologue):
    with pytest.raises(OpenGLIndexTypeError) as raised:
        _guarded(guard, call, prologue)
    assert raised.value.reason == "composed-pointer-index-range-unproven"
    assert raised.value.indexed_value == "input"


def test_unbounded_totals_cannot_prove_the_upper_bound():
    # Without a range for ``total`` neither the base nor its upper bound fits
    # int, so the relation is not recorded.
    with pytest.raises(OpenGLIndexTypeError):
        _guarded(total="uint64(asuint(source[1]))")


@pytest.mark.parametrize(
    "total",
    (
        # An unbounded wide delta cannot be shown to fit int.
        "uint64(asuint(source[0])) << 32",
        # 2^31 alone does not fit int, so int arithmetic could overflow.
        "uint64(2147483648l)",
    ),
)
def test_logical_offsets_require_every_delta_to_fit_int(total):
    with pytest.raises(OpenGLIndexTypeError):
        _generate(total=total, assertions=[ACCESS])


BRANCHED_KERNEL = """#include <metal_stdlib>
using namespace metal;

template <int N>
inline void load_unguarded(thread float values[N], const device float* input) {
  for (int i = 0; i < N; i++) {
    values[N - i - 1] = input[i];
  }
}

template <int N>
inline void load_reversed(thread float values[N], const device float* input, int start, int total) {
  for (int i = 0; i < N; i++) {
    values[N - i - 1] = (start + N - i - 1 < total) ? input[i] : 0.0f;
  }
}

template <int N>
[[kernel]] void reversed_tail(
    const device float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    const constant size_t& size [[buffer(2)]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  float values[N];
  uint offset = lid.x * N;
  if (offset + N < size) {
    load_unguarded<N>(values, in + size - offset - N);
  } else {
    load_reversed<N>(values, in + size - offset - N, offset, size);
  }
  for (int i = 0; i < N; i++) {
    if (offset + i < size) {
      out[offset + i] = values[i];
    }
  }
}

template [[host_name("reversed_tail_4")]] [[kernel]] void reversed_tail<4>(
    const device float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    const constant size_t& size [[buffer(2)]],
    uint3 lid [[thread_position_in_threadgroup]]);
"""


def _translate_branched(tmp_path, minimum, maximum):
    source = "reversed_tail.metal"
    (tmp_path / source).write_text(BRANCHED_KERNEL, encoding="utf-8")
    config = ProjectConfig(
        root=tmp_path,
        include_patterns=(source,),
        targets=("opengl",),
        output_dir="out",
        entry_points={source: ("reversed_tail_4",)},
        workgroup_size_rules={source: (16, 1, 1)},
        metadata_preconditions=[
            {
                "source": source,
                "entryPoint": "reversed_tail_4",
                "parameter": "size",
                "minimum": minimum,
                "maximum": maximum,
            }
        ],
        source_options={"metal": {"preserve_resource_origins": True}},
    )
    payload = translate_project(config, format_output=False).to_json()
    return payload, sorted((tmp_path / "out").rglob("*.glsl"))


def test_helpers_called_only_from_refuted_branches_are_not_specialized(tmp_path):
    # With ``size <= 4`` no thread has ``offset + 4 < size``; the unguarded
    # helper is never called, so its unprovable subscript is not emitted.
    payload, artifacts = _translate_branched(tmp_path, 1, 4)
    assert payload["summary"]["failedCount"] == 0, payload["diagnostics"]
    (artifact,) = artifacts
    generated = artifact.read_text(encoding="utf-8")
    assert "load_unguarded" not in generated
    assert "in_[(input_offset + i)]" in generated
    _compile(generated, "opengl", tmp_path)


def test_guards_with_constant_addends_bound_their_operand(tmp_path):
    # With ``size == 37`` the branch condition ``offset + 4 < size`` bounds
    # ``offset`` by 32, so the unguarded helper's base ``37 - offset - 4`` is
    # at least 1. Its operands cannot wrap, so the addend moves to the bound.
    payload, artifacts = _translate_branched(tmp_path, 37, 37)
    assert payload["summary"]["failedCount"] == 0, payload["diagnostics"]
    (artifact,) = artifacts
    generated = artifact.read_text(encoding="utf-8")
    assert "load_unguarded" in generated
    _compile(generated, "opengl", tmp_path)


def test_reachable_unguarded_helpers_are_still_refused(tmp_path):
    # With ``size`` up to 64 the unguarded helper is reachable, and the base
    # interval alone cannot bound its subscript from below.
    payload, artifacts = _translate_branched(tmp_path, 1, 64)
    assert payload["summary"]["failedCount"] == 1
    assert not artifacts
    assert {item["code"] for item in payload["diagnostics"]} == {
        "project.translate.opengl-index-type-unsupported"
    }
