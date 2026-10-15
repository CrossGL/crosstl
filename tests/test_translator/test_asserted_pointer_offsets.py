"""Explicit complete-offset assertions for rebased OpenGL storage pointers.

OpenGL lowers a Metal storage pointer to a buffer plus a signed ``int``
logical offset. A wide (64-bit) pointer update ``P += D`` and a subscript
``P[I]`` through a rebased pointer therefore need the complete offset to fit
the signed target domain. These tests cover the caller-trusted route: an
explicit project assertion on the delta or on the complete pointer expression
``P + D`` for the update, and on the complete pointer expression ``P + I`` for
the subscript. Assertions on an individual operand never establish the sum.
"""

import shutil
import subprocess

import pytest

from crosstl.project import ProjectConfig, translate_project

SOURCE = """#include <metal_stdlib>
using namespace metal;
kernel void read_value(const device uint* values [[buffer(0)]],
                       constant ulong& index [[buffer(1)]],
                       constant ulong& offset [[buffer(2)]],
                       device uint* result [[buffer(3)]]) {
    BODY
}
"""
INT_MAX = 2**31 - 1
LOOP_READ = "for (int i = 0; i < 4; i++) { result[i] = values[i]; }"


def _translate(tmp_path, body, assertions):
    (tmp_path / "kernel.metal").write_text(SOURCE.replace("BODY", body))
    config = ProjectConfig(
        root=tmp_path,
        include_patterns=("kernel.metal",),
        targets=("opengl",),
        workgroup_size=(1, 1, 1),
        entry_points={"kernel.metal": ("read_value",)},
        index_range_assertions=[
            {"expression": expression, "minimum": minimum, "maximum": maximum}
            for expression, minimum, maximum in assertions
        ],
    )
    return translate_project(config, format_output=False).to_json()


def _generated(tmp_path, payload):
    assert payload["summary"]["translatedCount"] == 1, payload["diagnostics"]
    assert payload["diagnostics"] == []
    return (tmp_path / payload["artifacts"][0]["path"]).read_text(encoding="utf-8")


def _rejection(payload):
    assert payload["summary"]["translatedCount"] == 0, payload
    conversions = [
        item["details"]["indexConversion"]
        for item in payload["diagnostics"]
        if item["code"] == "project.translate.opengl-index-type-unsupported"
    ]
    assert conversions, payload["diagnostics"]
    return conversions[0]


def _validate_spirv(source, tmp_path):
    compiler = shutil.which("glslangValidator")
    validator = shutil.which("spirv-val")
    if compiler is None or validator is None:
        return
    source_path = tmp_path / "asserted.comp"
    binary_path = tmp_path / "asserted.spv"
    source_path.write_text(source, encoding="utf-8")
    command = [compiler, "--target-env", "opengl", "--target-env", "spirv1.3"]
    command += ["-S", "comp", str(source_path), "-o", str(binary_path)]
    compiled = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=60
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    validated = subprocess.run(
        [validator, "--target-env", "spv1.3", str(binary_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert validated.returncode == 0, validated.stdout + validated.stderr


@pytest.mark.parametrize(
    "update_assertion",
    [("offset", 0, 4), ("values + offset", 0, 4)],
    ids=["delta", "complete-pointer"],
)
def test_asserted_wide_pointer_update_keeps_int_offset(tmp_path, update_assertion):
    # The update assertion does not describe later subscripts; each read keeps
    # its own complete-offset obligation (see the missing-complete control).
    payload = _translate(
        tmp_path,
        f"values += offset; {LOOP_READ}",
        [update_assertion, ("values + i", 0, 7)],
    )
    generated = _generated(tmp_path, payload)
    assert "values_offset += int(offset);" in generated
    assert "result[i] = values[(values_offset + int(i))];" in generated
    _validate_spirv(generated, tmp_path)


@pytest.mark.parametrize(
    "assertions",
    [
        [("values - offset", 0, 8), ("values + i", 0, 11)],
        [("offset", 0, 8), ("values + i", 0, 11)],
    ],
    ids=["complete-pointer", "delta"],
)
def test_asserted_wide_pointer_decrement_keeps_int_offset(tmp_path, assertions):
    payload = _translate(
        tmp_path, f"values += 8; values -= offset; {LOOP_READ}", assertions
    )
    generated = _generated(tmp_path, payload)
    assert "values_offset -= int(offset);" in generated
    assert "result[i] = values[(values_offset + int(i))];" in generated
    _validate_spirv(generated, tmp_path)


@pytest.mark.parametrize(
    "assertions",
    [
        [],
        [("index", 0, 3)],
        [("offset", 0, INT_MAX + 1)],
        [("values + offset", 0, INT_MAX + 1)],
        [("offset", -(2**31) - 1, 0)],
        [("values", 0, 4)],
    ],
    ids=[
        "missing",
        "unrelated",
        "delta-too-wide",
        "complete-too-wide",
        "delta-below-int",
        "base-only",
    ],
)
def test_wide_pointer_update_requires_an_int_representable_assertion(
    tmp_path, assertions
):
    payload = _translate(tmp_path, f"values += offset; {LOOP_READ}", assertions)
    conversion = _rejection(payload)
    assert conversion["reason"] == "pointer-offset-range-unproven"
    assert conversion["indexExpression"] == "offset"


def test_complete_pointer_assertion_covers_a_rebased_subscript(tmp_path):
    payload = _translate(
        tmp_path,
        "values += offset; result[0] = values[index];",
        [("values + offset", 0, 4), ("index", 0, 3), ("values + index", 0, 7)],
    )
    generated = _generated(tmp_path, payload)
    assert "values_offset += int(offset);" in generated
    assert "values[(values_offset + " in generated
    _validate_spirv(generated, tmp_path)


def test_complete_pointer_assertion_covers_a_rebased_dereference(tmp_path):
    payload = _translate(
        tmp_path,
        "values += offset; result[0] = *values;",
        [("values + offset", 0, 4), ("values + 0", 0, 4)],
    )
    generated = _generated(tmp_path, payload)
    assert "result[0] = values[values_offset];" in generated
    _validate_spirv(generated, tmp_path)


def test_complete_pointer_assertion_covers_a_proved_int_loop_index(tmp_path):
    payload = _translate(
        tmp_path,
        f"values += offset; {LOOP_READ}",
        [("values + offset", 0, 4), ("values + i", 0, 7)],
    )
    generated = _generated(tmp_path, payload)
    assert "result[i] = values[(values_offset + int(i))];" in generated
    _validate_spirv(generated, tmp_path)


@pytest.mark.parametrize(
    "body,assertions,index",
    [
        (
            f"values += offset; {LOOP_READ}",
            [("values + offset", 0, 4)],
            "i",
        ),
        (
            f"values += offset; {LOOP_READ}",
            [("values + offset", 0, 4), ("values + i", 0, INT_MAX + 1)],
            "i",
        ),
        (
            f"values += offset; {LOOP_READ}",
            [("values + offset", 0, 4), ("values + i", -1, 7)],
            "i",
        ),
        (
            "values += offset; result[0] = values[index];",
            [("values + offset", 0, 4), ("index", 0, 3)],
            "index",
        ),
        (
            "values += offset; result[0] = values[index];",
            [("values + offset", 0, 4), ("values + index", 0, 7)],
            "index",
        ),
        (
            "values += offset; result[0] = *values;",
            [("values + offset", 0, 4)],
            "0",
        ),
    ],
    ids=[
        "missing-complete",
        "complete-too-wide",
        "complete-negative",
        "index-only",
        "wide-relative-index",
        "dereference-missing-complete",
    ],
)
def test_rebased_subscript_requires_the_complete_offset_assertion(
    tmp_path, body, assertions, index
):
    conversion = _rejection(_translate(tmp_path, body, assertions))
    assert conversion["reason"] == "composed-pointer-index-range-unproven"
    assert conversion["indexExpression"] == index
    assert conversion["rangeStatus"] == "unproven"


# Inside a loop the current offset of an advancing pointer has no proved
# interval, so the delta alone cannot bound the update. The complete pointer
# assertion bounds every stored offset and the bounded delta keeps the int
# conversion exact; both are required (the arg-reduce stride walk shape).
LOOP_ADVANCE = (
    "for (int i = 0; i < 4; i++) { result[i] = values[0]; values += offset; }"
)


def test_in_loop_wide_advance_accepts_complete_and_delta_assertions(tmp_path):
    payload = _translate(
        tmp_path,
        LOOP_ADVANCE,
        [("offset", 0, 4), ("values + offset", 0, 16), ("values + 0", 0, 16)],
    )
    generated = _generated(tmp_path, payload)
    assert "values_offset += int(offset);" in generated
    assert "result[i] = values[" in generated
    assert "values_offset" in generated.split("result[i] = values[", 1)[1][:40]
    _validate_spirv(generated, tmp_path)


@pytest.mark.parametrize(
    "assertions",
    [
        [("values + offset", 0, 16), ("values + 0", 0, 16)],
        [("offset", 0, 4), ("values + 0", 0, 16)],
        [("offset", 0, 4), ("values + offset", 0, INT_MAX + 1), ("values + 0", 0, 16)],
        [("offset", 0, INT_MAX + 1), ("values + offset", 0, 16), ("values + 0", 0, 16)],
        [("offset", 0, 4), ("values + offset", -1, 16), ("values + 0", 0, 16)],
    ],
    ids=[
        "complete-without-delta",
        "delta-without-complete",
        "complete-too-wide",
        "delta-too-wide",
        "complete-negative",
    ],
)
def test_in_loop_wide_advance_requires_both_bounded_assertions(tmp_path, assertions):
    conversion = _rejection(_translate(tmp_path, LOOP_ADVANCE, assertions))
    assert conversion["reason"] == "pointer-offset-range-unproven"
    assert conversion["indexExpression"] == "offset"
