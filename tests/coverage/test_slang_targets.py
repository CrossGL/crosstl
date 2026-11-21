"""How Slang probes are checked: slangc's SPIR-V first, then the HLSL or CUDA source slangc writes, compiled by DXC
or nvcc, and how slangc's error logs become causes."""

import pytest

from tools.coverage import lowering, runner, validators
from tools.coverage.model import Feature, Param, Signature, Type

# slangc 2026.9 writes an OpAtomicCompareExchange on floats for this call, which its SPIR-V validator rejects.
SPIRV_ONLY_REJECTION = (
    "error: AtomicCompareExchange: expected Result Type to be integer scalar type"
)


def _slang_compiler(monkeypatch, accepts: set[str]) -> list[tuple[str, str]]:
    """Stand in for slangc and DXC: accept every source for the targets in ``accepts`` ("" is SPIR-V)."""
    seen = []

    def validated(language, source, stage, entry="", target=""):
        seen.append((source, target))
        if target in accepts:
            return validators.Verdict(True)
        if target:
            return validators.Verdict(False, "error: rejected for HLSL")
        return validators.Verdict(False, SPIRV_ONLY_REJECTION)

    monkeypatch.setattr(validators, "available", lambda language, target="": True)
    monkeypatch.setattr(runner, "_validated", validated)
    monkeypatch.setattr(runner, "parse_crossgl", lambda source: None)
    monkeypatch.setattr(runner, "translate", lambda *args: "translated slang")
    monkeypatch.setattr(
        runner,
        "lower",
        lambda feature: ("" if feature.target else "OpAtomicCompareExchange", ""),
    )
    return seen


SLANG_SIN = Feature(
    "slang",
    "function",
    "sin",
    Signature(Type("float"), (Param(Type("float")),)),
    crossgl="sin",
)


@pytest.mark.parametrize("direction", ["input", "output"])
def test_a_slang_probe_slangc_cannot_write_as_spirv_is_probed_for_hlsl(
    monkeypatch, direction
):
    seen = _slang_compiler(monkeypatch, {"hlsl"})
    run = runner.run_input if direction == "input" else runner.run_output
    result = run(SLANG_SIN, "slang")
    assert result.outcome == "supported", result.detail
    assert result.feature.target == "hlsl"
    # The native probe is tried for SPIR-V first, then HLSL; the translation is checked for HLSL only.
    assert [target for _, target in seen] == ["", "hlsl", "hlsl"]
    assert seen[-1][0] == "translated slang"
    if direction == "output":
        # The semantic id is the one the SPIR-V probe lowered to.
        assert result.semantic == "OpAtomicCompareExchange"


@pytest.mark.parametrize("direction", ["input", "output"])
def test_a_slang_probe_both_targets_reject_keeps_the_spirv_reason(
    monkeypatch, direction
):
    seen = _slang_compiler(monkeypatch, set())
    run = runner.run_input if direction == "input" else runner.run_output
    result = run(SLANG_SIN, "slang")
    assert result.outcome == "untested"
    assert result.detail == f"the probe itself is rejected: {SPIRV_ONLY_REJECTION}"
    assert result.feature.target == ""
    assert [target for _, target in seen] == ["", "hlsl"]


SLANG_UNSPELLED = Feature(
    "slang",
    "function",
    "sin",
    Signature(Type("float"), (Param(Type("float")),)),
    note="no CrossGL built-in computes it",
)


def test_an_unspelled_feature_the_target_rejects_is_untested_in_the_output_row(
    monkeypatch,
):
    seen = _slang_compiler(monkeypatch, set())
    result = runner.run_output(SLANG_UNSPELLED, "slang")
    assert result.outcome == "untested"
    assert result.detail == f"the probe itself is rejected: {SPIRV_ONLY_REJECTION}"
    assert result.validator == ""
    assert [target for _, target in seen] == ["", "hlsl"]


def test_an_unspelled_feature_the_target_accepts_is_refused_in_the_output_row(
    monkeypatch,
):
    seen = _slang_compiler(monkeypatch, {""})
    result = runner.run_output(SLANG_UNSPELLED, "slang")
    assert result.outcome == "refused"
    assert result.detail == (
        "CrossGL has no spelling for this operation: no CrossGL built-in computes it"
    )
    # Only the native probe is checked; there is no translation to validate.
    assert [target for _, target in seen] == [""]


def test_a_probe_its_usual_target_accepts_is_not_retried(monkeypatch):
    seen = _slang_compiler(monkeypatch, {""})
    result = runner.run_input(SLANG_SIN, "slang")
    assert (result.outcome, result.feature.target) == ("supported", "")
    assert [target for _, target in seen] == ["", ""]


def test_only_slang_falls_back_to_another_target(monkeypatch):
    seen = _slang_compiler(monkeypatch, {"hlsl"})
    glsl = Feature("glsl", "function", "sin", SLANG_SIN.signature, crossgl="sin")
    result = runner.run_input(glsl, "glsl")
    assert result.outcome == "untested"
    assert [target for _, target in seen] == [""]


# The shape of slangc 2026.9's log for an undefined name: the header names the kind of error, the label the name.
SLANGC_LOG = """error[E30015]: undefined identifier
  --> probe.slang:10:15
   |
10 | uint2 r = {name}(a0);
   |           ^^^^^^^^ undefined identifier '{name}'.
---'
error[E30019]: type mismatch in expression
  --> probe.slang:11:5
   |
11 | outp[i] = r;
   |     ^ expected an expression of type 'float', got 'uint2'
---'"""


@pytest.mark.parametrize(
    ("language", "log", "expected"),
    [
        (
            "slang",
            SLANGC_LOG.format(name="bitCount"),
            "error[E30015]: undefined identifier 'bitCount'",
        ),
        (
            "slang",
            "error[E30011]: left of '=' is not an l-value\n --> probe.slang:9:5\n"
            "  |\n9 | f(a0) = r;\n  | ^ left of '=' is not an l-value.",
            "error[E30011]: left of '=' is not an l-value",
        ),
        (
            "slang",
            SLANGC_LOG.split("---'\n", 1)[1],
            "error[E30019]: type mismatch in expression: expected an expression of type"
            " 'float', got 'uint2'",
        ),
        (
            "slang",
            "error[E30015]: undefined identifier",
            "error[E30015]: undefined identifier",
        ),
        # Other tools' first error line already carries the particulars.
        ("rust", SLANGC_LOG.format(name="x"), "error[E30015]: undefined identifier"),
        (
            "glsl",
            "note: x\nERROR: 0:4: 'f' : no matching function",
            "ERROR: 0:4: 'f' : no matching function",
        ),
        ("glsl", "nothing to report", "nothing to report"),
    ],
)
def test_first_error_joins_slangc_headers_with_their_label(language, log, expected):
    assert validators.first_error(log, language) == expected


def test_slang_probes_failing_on_different_names_have_different_causes(monkeypatch):
    from tools.coverage import causes

    monkeypatch.setattr(
        validators,
        "validate",
        lambda language, source, *args: validators.Verdict(
            False, SLANGC_LOG.format(name=source)
        ),
    )
    names = ("bitCount", "findMSB", "crossgl_NumWorkGroups")
    details = [runner._validated("slang", name, "compute").log for name in names]
    features = [
        {"name": feature, "crossgl": crossgl, "detail": detail}
        for (feature, crossgl), detail in zip(
            [
                ("countbits", "bitCount"),
                ("firstbithigh", "findMSB"),
                ("num-workgroups", None),
            ],
            details,
        )
    ]
    found = [causes.cause(feature, "slang") for feature in features]
    assert found[0] == found[1] == "error[E30015]: undefined identifier '<feature>'"
    assert found[2] == "error[E30015]: undefined identifier 'crossgl_NumWorkGroups'"


@pytest.mark.parametrize(
    ("status", "accepted"), [(0, True), (2, False)], ids=["accepts", "rejects"]
)
def test_a_slang_probe_for_cuda_is_written_by_slangc_and_compiled_by_nvcc(
    tmp_path, monkeypatch, status, accepted
):
    calls = tmp_path / "calls"
    slangc = tmp_path / "slangc"
    slangc.write_text(
        "#!/bin/sh\n"
        f'echo "slangc $*" >> "{calls}"\n'
        'while [ $# -gt 0 ]; do [ "$1" = -o ] && out="$2"; shift; done\n'
        'echo "__global__ void main_0() {}" > "$out"\n'
    )
    nvcc = tmp_path / "nvcc"
    nvcc.write_text(
        "#!/bin/sh\n"
        f'echo "nvcc $*" >> "{calls}"\n'
        f'cat probe.cu >> "{calls}"\n'
        f'[ {status} = 0 ] || echo "probe.cu(1): error: rejected" >&2\n'
        f"exit {status}\n"
    )
    for tool in (slangc, nvcc):
        tool.chmod(0o755)
    tools = {"slangc": slangc, "nvcc": nvcc}
    monkeypatch.setattr(validators, "command", lambda name: (str(tools[name]),))
    verdict = validators.validate("slang", "probe", "compute", "main", "cuda")
    assert verdict.accepted is accepted
    assert verdict.log == ("" if accepted else "probe.cu(1): error: rejected")
    assert calls.read_text().splitlines() == [
        "slangc probe.slang -entry main -stage compute -target cuda -o probe.cu",
        f"nvcc -arch={validators.CUDA_ARCH} -std=c++17 -c probe.cu -o probe.o",
        "__global__ void main_0() {}",
    ]


def test_strict_run_names_the_compiler_a_slang_target_needs(monkeypatch, capsys):
    from tools.coverage import __main__ as cli

    monkeypatch.setattr(validators, "available", lambda language, target="": True)
    monkeypatch.setattr(lowering, "can_lower", lambda language: True)
    monkeypatch.setattr(
        validators, "command", lambda tool: () if tool == "nvcc" else (tool,)
    )
    assert cli.main(["run", "--strict", "--backend", "slang"]) == 2
    assert capsys.readouterr().err.strip() == (
        "error: missing tools for slang for cuda (nvcc)"
    )
