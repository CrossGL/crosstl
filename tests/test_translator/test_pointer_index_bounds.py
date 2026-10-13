"""Checked pointer bases permit signed relative indices without losing bounds."""

import os
import sys
from dataclasses import replace

import pytest

from crosstl.project import translate_project
from crosstl.translator.codegen import GLSL_codegen
from tests.runtime_helpers import _validate
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metadata_index_bounds import (
    GUARD,
    SOURCE,
    _index_config,
    _index_request,
)
from tests.test_translator.test_project_metadata_entry_scope import _packages

CONDITIONS = (
    dict(entryPoint="read_value", parameter="index", minimum=-2, maximum=-1),
    dict(entryPoint="read_value", parameter="offset", minimum=2, maximum=4),
)
BODIES = {
    "load": "values += offset; result[2] = values[index];",
    "alias": (
        "const device uint* shifted = values + offset; result[2] = shifted[index];"
    ),
    "narrow": "values += offset; result[2] = values[int(index)];",
    "member": (
        "values += offset; long2 at = long2(index, index); result[2] = values[at.y];"
    ),
    "store": "result += offset; result[index] = values[offset + index];",
    "constant": "values += 4; result[2] = values[-1];",
}


def _source(case):
    return SOURCE.replace("ulong", "long").replace(
        "result[0] = values[offset + index];", BODIES[case]
    )


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
@pytest.mark.parametrize("case", BODIES)
def test_checked_base_keeps_signed_relative_indices(tmp_path, target, case):
    _, package, (descriptor,) = _packages(
        _index_config(
            tmp_path, source=_source(case), conditions=CONDITIONS, target=target
        )
    )
    assert (package / descriptor["artifact"]["packagePath"]).is_file()
    for index in (-2, -1):
        for offset in (2, 4):
            request, _ = _index_request(descriptor, package, index, offset)
            assert not request.execution_plan.diagnostics


@pytest.mark.parametrize(
    "bounds,accepted",
    [((0, 2), True), ((-1, -1), False), ((2147483648, 2147483648), False)],
)
def test_proved_zero_base_preserves_absolute_index_assertion_status(
    tmp_path, monkeypatch, bounds, accepted
):
    decisions = []
    original = GLSL_codegen.decide_index_narrowing

    def record(**kwargs):
        decision = original(**kwargs)
        value_range = kwargs["value_range"]
        if value_range is not None and value_range.status == "asserted":
            decisions.append((kwargs, decision))
        return decision

    monkeypatch.setattr(GLSL_codegen, "decide_index_narrowing", record)
    config = replace(
        _index_config(
            tmp_path,
            source=_source("load"),
            conditions=({**CONDITIONS[1], "minimum": 0, "maximum": 0},),
        ),
        index_range_assertions=[
            dict(expression="index", minimum=bounds[0], maximum=bounds[1])
        ],
    )
    report = translate_project(config, format_output=False).to_json()
    assert report["summary"]["translatedCount"] == int(accepted), report
    assert decisions
    assert all(arguments["base_range"] is None for arguments, _ in decisions)
    if accepted:
        assert all(decision.range_status == "asserted" for _, decision in decisions)
    else:
        assert all(not decision.accepted for _, decision in decisions)


@pytest.mark.parametrize(
    "offset,reason",
    [
        (0, "index-range-unproven"),
        (2, "composed-pointer-index-range-unproven"),
        (8, "composed-pointer-index-range-unproven"),
    ],
)
def test_checked_pointer_base_does_not_prove_an_unknown_index(tmp_path, offset, reason):
    report = translate_project(
        _index_config(
            tmp_path,
            source=_source("load"),
            conditions=({**CONDITIONS[1], "minimum": offset, "maximum": offset},),
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    diagnostic = report["diagnostics"][0]
    assert diagnostic["code"] == "project.translate.opengl-index-type-unsupported"
    assert diagnostic["details"]["indexConversion"]["reason"] == reason
    assert diagnostic["details"]["indexConversion"]["rangeStatus"] == "unproven"


@pytest.mark.parametrize(
    "fault",
    [
        "missing-base",
        "missing-index",
        "negative-sum",
        "overflow",
        "mutation",
        "shadow",
        "escape",
        "assertion",
        "positive-assertion",
        "derived-assertion",
        "local-assertion",
    ],
)
def test_relative_index_requires_a_complete_checked_address(tmp_path, fault):
    conditions = list(CONDITIONS)
    source = _source("load")
    if fault == "missing-base":
        conditions = conditions[:1]
    elif fault == "missing-index" or "assertion" in fault:
        conditions = conditions[1:]
        if fault == "derived-assertion":
            source = source.replace("values[index]", "values[index + 1]")
        elif fault == "local-assertion":
            source = source.replace(
                "values += offset;", "values += offset; long at = index + 1;"
            ).replace("values[index]", "values[at]")
    elif fault == "negative-sum":
        conditions[1] = {**conditions[1], "minimum": 0, "maximum": 1}
    elif fault == "overflow":
        conditions[0] = {**conditions[0], "minimum": 2147483647, "maximum": 2147483647}
    elif fault == "shadow":
        source = source.replace(
            BODIES["load"], "{ long index = -99; " + BODIES["load"] + " }"
        )
    elif fault == "escape":
        source = source.replace(
            "kernel void", "void overwrite(thread long& at) { at = -99; }\nkernel void"
        )
        source = source.replace(
            "values += offset;", "values += offset; long at = index; overwrite(at);"
        ).replace("values[index]", "values[at]")
    else:
        source = source.replace(
            "values += offset;", "values += offset; long at = index; at = -99;"
        ).replace("values[index]", "values[at]")
    config = _index_config(tmp_path, source=source, conditions=conditions)
    if "assertion" in fault:
        minimum, maximum = (-2, -1) if fault == "assertion" else (0, 0)
        config = replace(
            config,
            index_range_assertions=[
                dict(expression="index", minimum=minimum, maximum=maximum)
            ],
        )
    report = translate_project(config, format_output=False).to_json()
    assert report["summary"]["translatedCount"] == 0
    diagnostic = report["diagnostics"][0]
    assert diagnostic["code"] == "project.translate.opengl-index-type-unsupported"
    assert diagnostic["details"]["indexConversion"]["reason"] in {
        "composed-pointer-index-range-unproven",
        "pointer-offset-range-unproven",
    }


@pytest.mark.parametrize("kind", ["int", "long"])
def test_negative_absolute_subscript_is_still_rejected(tmp_path, kind):
    source = SOURCE.replace(
        "result[0] = values[offset + index];", f"result[2] = values[{kind}(-1)];"
    )
    report = translate_project(
        _index_config(tmp_path, source=source), format_output=False
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    assert (
        report["diagnostics"][0]["details"]["indexConversion"]["reason"]
        == "negative-index"
    )


@pytest.mark.parametrize(
    "bounds,reason,status",
    [
        ((-1, -1), "negative-index", "out-of-range"),
        ((-3, -1), "negative-index", "out-of-range"),
        ((-1, 0), "index-range-may-be-negative", "unproven"),
        ((-1, 5), "index-range-may-be-negative", "unproven"),
    ],
)
def test_index_diagnostic_distinguishes_possible_and_proven_negative_values(
    tmp_path, bounds, reason, status
):
    source = _source("load").replace("values += offset; ", "")
    conditions = ({**CONDITIONS[0], "minimum": bounds[0], "maximum": bounds[1]},)
    report = translate_project(
        _index_config(tmp_path, source=source, conditions=conditions),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    diagnostic = report["diagnostics"][0]["details"]["indexConversion"]
    assert diagnostic["reason"] == reason
    assert diagnostic["rangeStatus"] == status
    assert (
        diagnostic["sourceRange"]["minimum"],
        diagnostic["sourceRange"]["maximum"],
    ) == bounds


@pytest.mark.parametrize("case", ["load", "alias", "member", "store"])
@pytest.mark.parametrize("index", [-2, -1])
@pytest.mark.parametrize("offset", [2, 4])
def test_signed_relative_indices_execute_natively(tmp_path, case, index, offset):
    if os.environ.get("CROSTL_REQUIRE_METADATA_PRECONDITIONS") != "1":
        pytest.skip(
            "set CROSTL_REQUIRE_METADATA_PRECONDITIONS=1 for native index execution"
        )
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source = _source(case)
    _, package, (descriptor,) = _packages(
        _index_config(tmp_path, source=source, conditions=CONDITIONS, target=target)
    )
    expected = [GUARD] * 8
    expected[index + offset if case == "store" else 2] = 17 + 13 * (index + offset)
    request, outputs = _index_request(
        descriptor, package, index, offset, expected_words=expected
    )
    _execute(
        request,
        outputs,
        tmp_path,
        validate=_validate,
        original_source=source,
        original_entry="read_value",
    )
