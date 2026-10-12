"""Unsigned shift proofs preserve source width before index narrowing."""

import itertools

import pytest

from crosstl.project import translate_project
from crosstl.translator.ast import BinaryOpNode, IdentifierNode
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from tests.test_translator.test_metadata_index_bounds import (
    CONDITIONS,
    SOURCE,
    _index_config,
)
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_project_metadata_entry_scope import _packages

SHIFT_EXPRESSIONS = (
    "index << offset",
    "index >> offset",
    "ulong(uint(index) << uint(offset))",
    "ulong(uint(index) >> uint(offset))",
)
SHIFT_CONDITIONS = (CONDITIONS[0], {**CONDITIONS[1], "maximum": 2})


def _shift_source(expression):
    return SOURCE.replace(
        "result[0] = values[offset + index];",
        f"ulong at = {expression}; values += at; result[2] = values[0];",
    )


def _shift_config(root, target, expression):
    return _index_config(
        root,
        target=target,
        source=_shift_source(expression),
        conditions=SHIFT_CONDITIONS,
    )


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
@pytest.mark.parametrize("expression", SHIFT_EXPRESSIONS)
def test_checked_shifts_translate_and_compile(tmp_path, target, expression):
    _, package, (descriptor,) = _packages(_shift_config(tmp_path, target, expression))
    generated = (package / descriptor["artifact"]["packagePath"]).read_text()
    assert ("<<" if "<<" in expression else ">>") in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("operator", ["<<", ">>"])
@pytest.mark.parametrize("kind", ["uint", "uint64_t"])
def test_shift_proofs_cover_all_values_in_small_unsigned_intervals(operator, kind):
    generator = GLSLCodeGen()
    generator.local_variable_types = {"value": kind, "count": "uint"}
    expression = BinaryOpNode(
        IdentifierNode("value"), operator, IdentifierNode("count")
    )
    values = itertools.combinations_with_replacement(range(8), 2)
    counts = tuple(itertools.combinations_with_replacement(range(5), 2))
    for left, right in itertools.product(values, counts):
        facts = {"value": left, "count": right}
        observed = [
            value << count if operator == "<<" else value >> count
            for value in range(left[0], left[1] + 1)
            for count in range(right[0], right[1] + 1)
        ]
        expected = min(observed), max(observed)
        assert (
            generator.glsl_private_pointer_interval(expression, facts, {}) == expected
        )
        result = generator.glsl_index_value_range(expression, intervals=facts)
        assert result is not None
        assert (result.minimum, result.maximum) == expected


@pytest.mark.parametrize("operator", ["<<", ">>"])
@pytest.mark.parametrize(
    "kind,values,counts",
    [
        ("uint", (0, 1), (-1, 2)),
        ("uint", (0, 1), (0, 32)),
        ("uint64_t", (0, 1), (0, 64)),
        ("uint", (-1, 1), (0, 1)),
        ("uint", (0, 2**32), (0, 1)),
        ("int64_t", (-1, 1), (0, 1)),
        ("uint16_t", (0, 1), (0, 1)),
    ],
)
def test_shift_proofs_reject_unproved_source_semantics(operator, kind, values, counts):
    generator = GLSLCodeGen()
    generator.local_variable_types = {"value": kind, "count": "int"}
    expression = BinaryOpNode(
        IdentifierNode("value"), operator, IdentifierNode("count")
    )
    assert (
        generator.glsl_private_pointer_interval(
            expression, {"value": values, "count": counts}, {}
        )
        is None
    )


@pytest.mark.parametrize("width,kind", [(32, "uint"), (64, "uint64_t")])
def test_shift_proofs_reject_unsigned_wrap_but_allow_large_right_shifts(width, kind):
    generator = GLSLCodeGen()
    generator.local_variable_types = {"value": kind, "count": "uint"}
    facts = {"value": (0, 2**width - 1), "count": (width - 1, width - 1)}
    left = BinaryOpNode(IdentifierNode("value"), "<<", IdentifierNode("count"))
    right = BinaryOpNode(IdentifierNode("value"), ">>", IdentifierNode("count"))
    assert generator.glsl_private_pointer_interval(left, facts, {}) is None
    assert generator.glsl_private_pointer_interval(right, facts, {}) == (0, 1)


@pytest.mark.parametrize(
    "expression",
    ["index << 63ul", "index << 64ul", "index >> 64ul", "index << -1", "index << 32ul"],
)
def test_invalid_shift_cannot_authorize_pointer_narrowing(tmp_path, expression):
    payload = translate_project(
        _shift_config(tmp_path, "opengl", expression), format_output=False
    ).to_json()
    assert payload["summary"]["translatedCount"] == 0, payload
    assert any(item["severity"] == "error" for item in payload["diagnostics"])


def test_shift_without_checked_operand_does_not_acquire_bounds(tmp_path):
    payload = translate_project(
        _index_config(tmp_path, source=_shift_source("index << 2ul"), conditions=()),
        format_output=False,
    ).to_json()
    assert payload["summary"]["translatedCount"] == 0, payload


@pytest.mark.parametrize(
    "body",
    [
        "ulong at = 0ul; at = index << offset; values += at; result[2] = values[0];",
        "ulong at = index << offset; result[2] = values[at];",
        "values += index << offset; result[2] = values[0];",
    ],
)
def test_checked_shift_bounds_survive_assignment_and_direct_use(tmp_path, body):
    config = _index_config(
        tmp_path,
        source=SOURCE.replace("result[0] = values[offset + index];", body),
        conditions=SHIFT_CONDITIONS,
    )
    _, package, (descriptor,) = _packages(config)
    _compile(
        (package / descriptor["artifact"]["packagePath"]).read_text(),
        "opengl",
        tmp_path,
    )


@pytest.mark.parametrize(
    "body",
    [
        "ulong at = index << offset; at += 4294967296ul; values += at; result[2] = values[0];",
        "ulong count = offset; count = 64ul; ulong at = index << count; values += at; result[2] = values[0];",
        "uint narrow = uint(index); ulong at = ulong(narrow << 31); values += at; result[2] = values[0];",
        "long signed_value = -1; ulong at = ulong(signed_value >> offset); values += at; result[2] = values[0];",
        "{ uint index = 3u; ulong at = ulong(index << 31); values += at; result[2] = values[0]; }",
    ],
)
def test_shift_proofs_cannot_hide_mutation_overflow_or_shadowing(tmp_path, body):
    payload = translate_project(
        _index_config(
            tmp_path,
            source=SOURCE.replace("result[0] = values[offset + index];", body),
            conditions=SHIFT_CONDITIONS,
        ),
        format_output=False,
    ).to_json()
    assert payload["summary"]["translatedCount"] == 0, payload
