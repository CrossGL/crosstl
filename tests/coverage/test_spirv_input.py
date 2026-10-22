import pytest

from tools.coverage import runner, spirv_input, validators
from tools.coverage.dialect import ProbeError
from tools.coverage.model import Feature, Param, Signature, Type

VEC3 = Type("float", (3,))
FABS = Feature(
    "spirv", "function", "FAbs", Signature(VEC3, (Param(VEC3),)), crossgl="abs"
)
NMIN = Feature(
    "spirv",
    "function",
    "NMin",
    Signature(VEC3, (Param(VEC3), Param(VEC3))),
    note="NaN-aware min has no GLSL built-in",
)
needs_glslang = pytest.mark.skipif(
    not (validators.can_lower("glsl") and validators.available("spirv")),
    reason="glslangValidator or SPIRV-Tools is missing",
)


@needs_glslang
def test_the_probe_is_the_assembly_glslang_compiles_the_glsl_probe_to():
    probe = spirv_input.build(FABS)
    assert (probe.language, probe.suffix) == ("spirv", ".spvasm")
    assert " FAbs " in probe.source
    # Disassembled with names, as a project would keep its SPIR-V assembly.
    assert "OpName %main" in probe.source
    assert validators.validate("spirv", probe.source).accepted


@needs_glslang
def test_a_nan_aware_instruction_is_renamed_from_the_one_with_its_operands():
    probe = spirv_input.build(NMIN)
    assert " NMin " in probe.source and " FMin " not in probe.source
    assert validators.validate("spirv", probe.source).accepted


@needs_glslang
def test_a_probe_must_call_the_features_own_instruction():
    modf = Feature(
        "spirv",
        "function",
        "Modf",
        Signature(VEC3, (Param(VEC3), Param(VEC3, "out"))),
        crossgl="modf",
    )
    with pytest.raises(ProbeError, match="glslang lowers modf to ModfStruct, not Modf"):
        spirv_input.build(modf)


@needs_glslang
def test_a_struct_form_is_probed_with_the_glsl_call_glslang_lowers_to_it():
    # The feature list gives the struct forms no signature (tools/coverage/features/spirv.py).
    frexp = Feature("spirv", "function", "FrexpStruct", None, note="alternative")
    probe = spirv_input.build(frexp)
    assert " FrexpStruct " in probe.source
    assert validators.validate("spirv", probe.source).accepted


@needs_glslang
def test_input_probe_end_to_end():
    result = runner.run_input(FABS, "spirv")
    assert (result.outcome, result.validator) == (
        "supported",
        "spirv-val",
    ), result.detail


def test_a_round_trip_that_drops_the_instruction_is_wrong():
    lowered = "%5 = OpExtInst %v3float %1 FMin %3 %4\n"
    assert spirv_input.dropped(NMIN, lowered) == "the round trip no longer calls NMin"
    assert spirv_input.dropped(NMIN, lowered.replace("FMin", "NMin")) == ""
    assert spirv_input.dropped(NMIN, "") == "the round trip no longer calls NMin"
    # Either encoding of modf keeps the call.
    modf = Feature("spirv", "function", "ModfStruct", None)
    assert spirv_input.dropped(modf, "%5 = OpExtInst %v3float %1 Modf %3 %4\n") == ""
    assert (
        spirv_input.dropped(modf, lowered)
        == "the round trip no longer calls ModfStruct"
    )
    loop = Feature("spirv", "control-flow", "for-loop", None)
    assert spirv_input.dropped(loop, "") == ""


def test_without_glslang_the_input_probes_are_untested(monkeypatch):
    monkeypatch.setattr(validators, "can_lower", lambda language: False)
    result = runner.run_input(FABS, "spirv")
    assert (result.outcome, result.detail) == (
        "untested",
        "glslangValidator or spirv-dis is not installed",
    )


def test_strict_run_names_glslang_for_spirv_input(monkeypatch, capsys):
    from tools.coverage import __main__ as cli

    monkeypatch.setattr(validators, "available", lambda language: True)
    monkeypatch.setattr(validators, "can_lower", lambda language: False)
    monkeypatch.setattr(
        validators,
        "command",
        lambda tool: () if tool == "glslangValidator" else (tool,),
    )
    argv = ["run", "--strict", "--backend", "spirv", "--direction", "input"]
    assert cli.main(argv) == 2
    assert capsys.readouterr().err.strip() == (
        "error: missing tools for spirv input (glslangValidator)"
    )
