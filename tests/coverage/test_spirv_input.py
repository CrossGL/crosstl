import dataclasses

import pytest

from tools.coverage import runner, spirv_input, validators
from tools.coverage.dialect import ProbeError
from tools.coverage.model import VOID, Feature, Param, Signature, Type

VEC3 = Type("float", (3,))
FABS = Feature(
    "spirv", "function", "FAbs", Signature(VEC3, (Param(VEC3),)), crossgl="abs"
)
NMIN = Feature(
    "spirv",
    "function",
    "NMin",
    Signature(VEC3, (Param(VEC3), Param(VEC3))),
    crossgl="fmin",
)
NCLAMP = Feature(
    "spirv",
    "function",
    "NClamp",
    Signature(VEC3, (Param(VEC3), Param(VEC3), Param(VEC3))),
    crossgl="fmin(fmax({}, {}), {})",
)
UVEC3 = Type("uint", (3,))
ULESS = Feature(
    "spirv",
    "function",
    "OpULessThan",
    Signature(Type("bool", (3,)), (Param(UVEC3), Param(UVEC3))),
    group="core-relational-and-logical",
    crossgl="lessThan",
    spelling="lessThan",
)
STREAM = Feature(
    "spirv",
    "function",
    "OpEmitStreamVertex",
    Signature(VOID, (Param(Type("int", ()), "const"),)),
    group="core-primitive",
    crossgl="EmitStreamVertex",
    stage="geometry",
    spelling="EmitStreamVertex",
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
    # A spelling whose GLSL call glslang lowers to another instruction.
    inverse = Feature(
        "spirv",
        "function",
        "InverseSqrt",
        Signature(VEC3, (Param(VEC3),)),
        crossgl="sqrt",
    )
    with pytest.raises(
        ProbeError, match="glslang lowers sqrt to Sqrt, not InverseSqrt"
    ):
        spirv_input.build(inverse)


@needs_glslang
@pytest.mark.parametrize("name, out", [("Frexp", Type("int", (3,))), ("Modf", VEC3)])
def test_a_pointer_form_is_the_struct_forms_module_writing_the_out_variable(name, out):
    feature = Feature(
        "spirv",
        "function",
        name,
        Signature(VEC3, (Param(VEC3), Param(out, "out"))),
        crossgl=name.lower(),
    )
    probe = spirv_input.build(feature)
    assert f" {name}Struct " not in probe.source
    assert "OpCompositeExtract" not in probe.source
    (call,) = (line for line in probe.source.splitlines() if f" {name} " in line)
    assert call.split()[-1] == "%a1"
    assert validators.validate("spirv", probe.source).accepted


STRUCT_CALL = """\
         %48 = OpExtInst %ResType %1 FrexpStruct %43
         %49 = OpCompositeExtract %v3int %48 1
               OpStore %a1 %49
         %50 = OpCompositeExtract %v3float %48 0
               OpStore %r %50
"""


def test_the_pointer_form_rewrite_replaces_the_call_extracts_and_store():
    assert spirv_input._pointer_form(STRUCT_CALL, "Frexp") == (
        "         %50 = OpExtInst %v3float %1 Frexp %43 %a1\n"
        "               OpStore %r %50\n"
    )


@pytest.mark.parametrize(
    "assembly",
    [
        STRUCT_CALL + STRUCT_CALL.replace("%48", "%58"),
        STRUCT_CALL + "         %51 = OpCopyObject %ResType %48\n",
        STRUCT_CALL + "               OpStore %e %49\n",
        STRUCT_CALL.replace("OpStore %a1 %49", "OpNop"),
    ],
    ids=["two calls", "struct used again", "member 1 used again", "no store"],
)
def test_the_pointer_form_rewrite_refuses_any_other_shape(assembly):
    with pytest.raises(ProbeError, match="FrexpStruct call has an unexpected shape"):
        spirv_input._pointer_form(assembly, "Frexp")


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


def test_a_round_trip_may_write_nclamp_as_the_calls_it_is_defined_as():
    nested = (
        "%5 = OpExtInst %v3float %1 NMax %2 %3\n"
        "%6 = OpExtInst %v3float %1 NMin %5 %4\n"
    )
    assert spirv_input.dropped(NCLAMP, nested) == ""
    assert spirv_input.dropped(NCLAMP, nested.replace("NMax", "FMax")) == (
        "the round trip no longer calls NClamp"
    )


def test_a_translation_of_a_nan_aware_instruction_must_handle_nan():
    ignored = "%5 = OpExtInst %v3float %1 FMin %3 %4\n"
    assert spirv_input.ignores_nan(NMIN, ignored) == (
        "the translation calls no NMin and tests no operand for NaN"
    )
    assert spirv_input.ignores_nan(NMIN, ignored.replace("FMin", "NMin")) == ""
    tested = "%6 = OpIsNan %v3bool %3\n" + ignored
    assert spirv_input.ignores_nan(NMIN, tested) == ""
    # FMin is the instruction of FMin itself.
    fmin = Feature("spirv", "function", "FMin", NMIN.signature, crossgl="min")
    assert spirv_input.ignores_nan(fmin, ignored) == ""


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


@needs_glslang
def test_a_core_instruction_is_probed_with_the_glsl_call_that_compiles_to_it():
    probe = spirv_input.build(ULESS)
    assert " OpULessThan " in probe.source
    assert validators.validate("spirv", probe.source).accepted
    signed = dataclasses.replace(ULESS, name="OpSLessThan")
    with pytest.raises(
        ProbeError, match="glslang does not lower lessThan to OpSLessThan"
    ):
        spirv_input.build(signed)


@needs_glslang
def test_a_vertex_stream_probe_declares_the_geometry_streams_capability():
    probe = spirv_input.build(STREAM)
    assert "OpCapability GeometryStreams" in probe.source
    assert validators.validate("spirv", probe.source).accepted


def test_a_core_round_trip_must_keep_the_instruction():
    body = (
        "%main = OpFunction %void None %3\n"
        "%5 = OpULessThan %v3bool %1 %2\n"
        "OpControlBarrier %uint_2 %uint_2 %uint_264\n"
        "OpFunctionEnd\n"
    )
    assert spirv_input.dropped(ULESS, body) == ""
    assert (
        spirv_input.dropped(ULESS, body.replace("OpULessThan", "OpSLessThan"))
        == "the round trip no longer calls OpULessThan"
    )
    # Outside a function body the instruction is not called.
    assert spirv_input.dropped(ULESS, "%5 = OpULessThan %v3bool %1 %2\n")
    barrier = dataclasses.replace(ULESS, name="OpControlBarrier", group="core-barrier")
    assert spirv_input.dropped(barrier, body) == ""


def test_a_dropped_core_instruction_the_generator_never_writes_is_left_to_execution(
    monkeypatch,
):
    written = {"text": ""}
    monkeypatch.setattr(runner, "translate", lambda *args: written["text"])
    written["text"] = (
        "%main = OpFunction %void None %3\n%5 = OpULessThan %v3bool %1 %2\n"
    )
    assert not runner._encodes_otherwise(ULESS)
    written["text"] = (
        "%main = OpFunction %void None %3\n%5 = OpSLessThan %v3bool %1 %2\n"
    )
    assert runner._encodes_otherwise(ULESS)
    # Extended instructions are always judged by the call.
    assert not runner._encodes_otherwise(FABS)
