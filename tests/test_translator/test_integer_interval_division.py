"""Integer range proofs must not round through floating-point arithmetic."""

import itertools

import pytest

from crosstl.project import build_native_loader_dispatch_request, translate_project
from crosstl.translator.ast import BinaryOpNode, IdentifierNode
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.index_range_contracts import (
    nonnegative_remainder_interval,
    truncating_division_interval,
)
from tests.test_translator.test_metadata_index_bounds import (
    GUARD,
    SOURCE,
    _index_config,
)

DIVISION_SOURCE = SOURCE.replace(
    "result[0] = values[offset + index];",
    "ulong at = (index / offset) % 16ul; values += at; result[2] = values[0];",
)
DIVISION_CONDITIONS = (
    dict(entryPoint="read_value", parameter="index", minimum=2**53, maximum=2**64 - 1),
    dict(entryPoint="read_value", parameter="offset", minimum=1, maximum=8),
)
DIVISION_CASES = (
    (2**53, 1),
    (2**53 + 1, 1),
    (2**53 + 3, 1),
    (2**53 + 1, 3),
    (2**63 - 1, 2),
    (2**63, 7),
    (2**64 - 1, 1),
    (2**64 - 1, 3),
    (2**64 - 1, 8),
)


def _division_config(root, target):
    return _index_config(
        root, source=DIVISION_SOURCE, conditions=DIVISION_CONDITIONS, target=target
    )


def _division_request(descriptor, package, numerator, divisor):
    values = [17 + position * 13 for position in range(16)]
    result = [GUARD] * 8
    result[2] = values[(numerator // divisor) % 16]
    inputs, outputs, observed = {}, {}, []
    for binding in descriptor["bindings"]:
        origin = binding.get("provenance", {}).get("sourceResource")
        if origin is None:
            continue
        name, parameter = binding["name"], origin["parameter"]
        words = {
            "values": values,
            "index": [numerator],
            "offset": [divisor],
            "result": [GUARD] * 8,
        }[parameter]
        inputs[name] = dict(
            dtype=binding["scalarLayout"]["elementType"],
            shape=[len(words)],
            values=words,
        )
        if parameter == "result":
            outputs[name] = dict(dtype="uint32", shape=[8], values=result)
        else:
            observed.append(name)
    return (
        build_native_loader_dispatch_request(
            descriptor,
            package,
            inputs,
            outputs,
            {"workgroupCount": [1, 1, 1], "workgroupSize": [1, 1, 1]},
            observe_inputs=observed,
        ),
        outputs,
    )


@pytest.mark.parametrize(
    "numerator,denominator,expected",
    [
        (2**53 + 1, 1, 2**53 + 1),
        (2**53 + 3, 1, 2**53 + 3),
        (2**64 - 1, 1, 2**64 - 1),
        (2**63 - 1, 2, 2**62 - 1),
        (-(2**63) + 1, 2, -(2**62) + 1),
        (2**64 - 1, 3, 6148914691236517205),
        (7, -3, -2),
        (-7, -3, 2),
        (-7, 3, -2),
        (0, -3, 0),
    ],
)
def test_exact_integer_quotients(numerator, denominator, expected):
    assert truncating_division_interval(
        (numerator, numerator), (denominator, denominator)
    ) == (expected, expected)


def test_division_bounds_cover_exhaustive_small_integer_intervals():
    intervals = list(itertools.combinations_with_replacement(range(-6, 7), 2))
    for left, right in itertools.product(intervals, repeat=2):
        actual = truncating_division_interval(left, right)
        if right[0] <= 0 <= right[1]:
            assert actual is None
            continue
        results = []
        for numerator in range(left[0], left[1] + 1):
            for denominator in range(right[0], right[1] + 1):
                quotient, remainder = divmod(numerator, denominator)
                if quotient < 0 and remainder:
                    quotient += 1
                results.append(quotient)
        assert actual == (min(results), max(results))


def test_remainder_bounds_cover_exhaustive_nonnegative_intervals():
    intervals = list(itertools.combinations_with_replacement(range(9), 2))
    for left, right in itertools.product(intervals, repeat=2):
        bounds = nonnegative_remainder_interval(left, right)
        if right[0] == 0:
            assert bounds is None
            continue
        assert bounds is not None
        for numerator in range(left[0], left[1] + 1):
            for denominator in range(right[0], right[1] + 1):
                assert bounds[0] <= numerator % denominator <= bounds[1]


@pytest.mark.parametrize("left,right", [((-1, 7), (1, 3)), ((0, 7), (-3, -1))])
def test_remainder_does_not_claim_a_nonnegative_result(left, right):
    assert nonnegative_remainder_interval(left, right) is None


@pytest.mark.parametrize("target", ["opengl", "directx"])
@pytest.mark.parametrize(
    "dividend,divisor,expected",
    [
        ((2**53 + 1, 2**53 + 3), 1, (2**53 + 1, 2**53 + 3)),
        ((2**64 - 2, 2**64 - 1), 3, (6148914691236517204, 6148914691236517205)),
        ((-(2**63) + 1, -(2**63) + 3), 2, (-(2**62) + 1, -(2**62) + 2)),
        ((-7, 7), 3, (-2, 2)),
    ],
)
def test_backend_pointer_intervals_preserve_exact_integer_division(
    target, dividend, divisor, expected
):
    generator = GLSLCodeGen() if target == "opengl" else HLSLCodeGen()
    interval = getattr(
        generator,
        f"{'glsl' if target == 'opengl' else 'hlsl'}_private_pointer_interval",
    )
    expression = BinaryOpNode(
        IdentifierNode("numerator"), "/", IdentifierNode("divisor")
    )
    assert (
        interval(expression, {"numerator": dividend, "divisor": (divisor, divisor)}, {})
        == expected
    )


@pytest.mark.parametrize(
    "body",
    [
        "ulong at = index / 1ul - 9007197107257345ul; values += at; result[0] = values[0];",
        "ulong at = index / 1ul - 9007194959773697ul; result[0] = values[at];",
        "ulong at = index / 1ul - 9007199254740992ul; uint local[1]; result[0] = local[at];",
    ],
)
def test_rounded_division_cannot_authorize_an_invalid_index(tmp_path, body):
    conditions = (
        dict(
            entryPoint="read_value",
            parameter="index",
            minimum=2**53 + 1,
            maximum=2**53 + 1,
        ),
    )
    report = translate_project(
        _index_config(
            tmp_path,
            source=SOURCE.replace("result[0] = values[offset + index];", body),
            conditions=conditions,
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0, report
    assert any(
        item["code"] == "project.translate.opengl-index-type-unsupported"
        for item in report["diagnostics"]
    )


@pytest.mark.parametrize("operator", ["/", "%"])
@pytest.mark.parametrize("mode", ["inline", "local", "pointer", "helper"])
def test_checked_runtime_divisor_proves_bounded_indices(tmp_path, operator, mode):
    expression = f"index {operator} offset"
    bodies = {
        "inline": f"result[0] = values[{expression}];",
        "local": f"ulong at = {expression}; result[0] = values[at];",
        "pointer": f"values += {expression}; result[0] = values[0];",
        "helper": f"result[0] = read_at(values, {expression});",
    }
    source = SOURCE.replace("result[0] = values[offset + index];", bodies[mode])
    if mode == "helper":
        source = source.replace(
            "kernel void",
            "uint read_at(const device uint* data, ulong at) { return data[at]; }\nkernel void",
        )
    conditions = (
        dict(entryPoint="read_value", parameter="index", minimum=0, maximum=31),
        dict(entryPoint="read_value", parameter="offset", minimum=1, maximum=8),
    )
    report = translate_project(
        _index_config(tmp_path, source=source, conditions=conditions),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 1, report["diagnostics"]


@pytest.mark.parametrize("operator", ["/", "%"])
@pytest.mark.parametrize("mode", ["inline", "local", "pointer"])
def test_divisor_range_containing_zero_does_not_prove_an_index(
    tmp_path, operator, mode
):
    expression = f"index {operator} offset"
    body = {
        "inline": f"result[0] = values[{expression}];",
        "local": f"ulong at = {expression}; result[0] = values[at];",
        "pointer": f"values += {expression}; result[0] = values[0];",
    }[mode]
    report = translate_project(
        _index_config(
            tmp_path, source=SOURCE.replace("result[0] = values[offset + index];", body)
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0, report


@pytest.mark.parametrize("operator", ["/", "%"])
def test_division_does_not_hide_source_overflow(tmp_path, operator):
    source = SOURCE.replace(
        "result[0] = values[offset + index];",
        f"ulong at = (index + 1ul) {operator} offset; result[0] = values[at];",
    )
    conditions = (
        dict(
            entryPoint="read_value",
            parameter="index",
            minimum=2**64 - 1,
            maximum=2**64 - 1,
        ),
        dict(entryPoint="read_value", parameter="offset", minimum=1, maximum=8),
    )
    report = translate_project(
        _index_config(tmp_path, source=source, conditions=conditions),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0, report
