"""The buffers, constants and inputs a translation must keep for its probe to run."""

from dataclasses import replace

import numpy as np
import pytest

from tests.conformance import runner as harness
from tools.coverage import (
    execution,
    issues,
    lowering,
    probes,
    publish,
    report,
    runner,
    validators,
)
from tools.coverage.model import Feature, Param, Result, Signature, Type

from .execution_cases import (
    CALLS,
    NATIVE,
    SIN,
    constants_given,
    fake_harness,
    hlsl_result,
    metal_runs,
    supported,
)

COUNTED = """\
               OpEntryPoint GLCompute %main "main" %id %counter_var_outp
               OpName %type_ACSBuffer_counter "type.ACSBuffer.counter"
               OpName %counter_var_outp "counter.var.outp"
               OpDecorate %outp Binding 1
               OpDecorate %counter_var_outp DescriptorSet 0
               OpDecorate %counter_var_outp Binding 2
%type_ACSBuffer_counter = OpTypeStruct %int
%_ptr_Uniform_type_ACSBuffer_counter = OpTypePointer Uniform %type_ACSBuffer_counter
%counter_var_outp = OpVariable %_ptr_Uniform_type_ACSBuffer_counter Uniform
       %outp = OpVariable %_ptr_Uniform_float Uniform
"""


def test_a_counter_buffer_nothing_uses_is_left_out_of_hlsl_spirv():
    # DXC declares one for a RWStructuredBuffer a function takes; the host binds no such buffer.
    kept = execution.without_unused_counters(COUNTED)
    assert "counter_var_outp" not in kept
    assert '"main" %id\n' in kept and "%outp Binding 1" in kept
    assert "%type_ACSBuffer_counter = OpTypeStruct" in kept
    used = COUNTED + "%7 = OpAccessChain %_ptr_Uniform_int %counter_var_outp %uint_0\n"
    assert execution.without_unused_counters(used) == used
    plain = COUNTED.replace("type.ACSBuffer.counter", "type.other")
    assert execution.without_unused_counters(plain) == plain


HELPER = """\
RWStructuredBuffer<float> fin : register(u0);
RWStructuredBuffer<float> outp : register(u1);
void store(RWStructuredBuffer<float> buffer, uint index, float value) {
    buffer[index] = value;%s
}
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    store(outp, id.x, fin[id.x]);
}
"""


def test_a_buffer_passed_to_a_function_keeps_the_hlsl_interface(tmp_path):
    if not lowering.can_lower("hlsl"):
        pytest.skip(f"{validators.TOOLS['hlsl']} or spirv-dis is not installed")
    from tests.conformance import interface

    path = tmp_path / "probe.spvasm"
    path.write_text(execution.as_spirv("hlsl", HELPER % ""))
    assert interface.interface(path, "spirv", 1)["bindings"] == (0, 1)
    # A translation that uses the counter needs a buffer the native probe does not bind.
    path.write_text(execution.as_spirv("hlsl", HELPER % " buffer.IncrementCounter();"))
    with pytest.raises(interface.InterfaceChanged, match=r"\[0, 1, 2\]"):
        interface.interface(path, "spirv", 1)


def test_nothing_runs_without_a_runtime(monkeypatch):
    monkeypatch.setattr(
        harness, "select_runtimes", lambda spec, targets: {t: [] for t in targets}
    )
    results = [supported()]
    assert execution.run(results) == (results, [])
    assert execution.idle(results) == ["metal"]
    # One Metal probe that ran is enough.
    assert execution.idle([*results, replace(supported(), runtime="metal-apple")]) == []
    assert execution.idle([Result(SIN, "output", "refused")]) == []


def test_the_harness_takes_coverage_probes_as_cases(tmp_path):
    outputs = {}
    path = tmp_path / "probe.metal"
    path.write_text(NATIVE)
    case = execution._case(
        "00000-native", path, "metal", execution.inputs("fin"), None, outputs
    )
    [record] = harness.run_suite(
        [case], ["metal"], ["strict"], {"metal": []}, tmp_path / "run", reference=True
    )
    assert record["case"] == "00000-native"
    assert record["runtime"].startswith("compile-only@")


@pytest.mark.skipif(not metal_runs(), reason="no Metal runtime")
def test_metal_runs_the_native_probe_against_the_translation():
    same = supported()
    other = supported(NATIVE.replace("sin(a0)", "cos(a0)"))
    outside = supported(NATIVE.replace("outp[i] = r;", "outp[i + 1024u] = r;"))
    (same, other, outside), used = execution.run([same, other, outside])
    assert used == ["metal-apple"]
    assert (same.outcome, same.runtime) == ("supported", "metal-apple")
    assert other.outcome == "wrong" and execution.DIVERGED in other.detail
    assert outside.outcome == "wrong" and "outside its output buffer" in outside.detail


def test_supported_metal_compute_results_keep_both_sources(monkeypatch):
    monkeypatch.setattr(validators, "available", lambda language, target="": True)
    monkeypatch.setattr(
        runner, "_native_rejection", lambda feature, language: (feature, "")
    )
    monkeypatch.setattr(runner, "translate", lambda *args: "translated metal")
    monkeypatch.setattr(
        runner, "_validated", lambda *args, **kwargs: validators.Verdict(True)
    )
    assert runner.run_output(SIN, "metal").sources == (NATIVE, "translated metal")
    assert runner.run_input(SIN, "metal").sources == (NATIVE, "translated metal")
    fragment = Feature(
        "metal", "function", "sin", SIN.signature, crossgl="sin", stage="fragment"
    )
    assert runner.run_output(fragment, "metal").sources == ()
    hlsl = hlsl_result().feature
    assert runner.run_output(hlsl, "hlsl").sources == (
        hlsl_result().sources[0],
        "translated metal",
    )


def test_the_report_names_the_runtime_and_coverage_md_says_what_ran():
    ran = Result(SIN, "output", "supported", runtime="metal-apple")
    data = report.build([ran], ["metal"], ["output"])
    [row] = [b for b in data["backends"] if b["language"] == "metal"]
    assert (row["runtime"], row["executed"]) == ("metal-apple", 1)
    assert "Executed: Metal output, 1 probes on metal-apple." in publish.markdown(data)
    plain = report.build([Result(SIN, "output", "supported")], ["metal"], ["output"])
    assert "Executed:" not in publish.markdown(plain)


def test_an_execution_bug_is_reproduced_with_execute():
    detail = (
        f"on metal-apple, the translation {execution.DIVERGED}\n1 output words differ."
    )
    feature = {
        "name": "sin",
        "category": "function",
        "outcome": "wrong",
        "detail": detail,
    }
    row = {
        "language": "metal",
        "direction": "output",
        "offered": True,
        "score": 0.0,
        "counts": {"supported": 0, "refused": 0, "wrong": 1, "untested": 0},
        "features": [feature],
    }
    body = issues.bug_body(row, "cause", [feature], "CrossGL/crosstl")
    assert "--feature sin --execute" in body
    assert "but run, it does not compute what the native Metal probe computes" in body
    feature["detail"] = f"on metal-apple, the translation {execution.REBOUND}\nExpected"
    body = issues.bug_body(row, "cause", [feature], "CrossGL/crosstl")
    assert "--feature sin --execute" in body
    assert "does not bind the buffers the native Metal probe binds" in body
    feature["detail"] = f"on metal-apple, the translation {execution.REJECTED}\nerror"
    body = issues.bug_body(row, "cause", [feature], "CrossGL/crosstl")
    assert "--feature sin --execute" in body
    assert "but the runtime that runs the native Metal probe rejects it." in body
    assert not execution.found("probe.metal:3:1: error: no matching function")


@pytest.mark.parametrize(
    ("language", "via_spirv"),
    [
        ("metal", False),
        ("glsl", False),
        ("glsl", True),
        ("hlsl", True),
        ("slang", True),
        ("wgsl", True),
        ("spirv", False),
    ],
)
@pytest.mark.parametrize("names", [("fin", "iin"), ("uin", "fin", "iin")])
def test_execution_passes_one_typed_array_per_declared_input(
    monkeypatch, tmp_path, language, via_spirv, names
):
    # A non-alphabetic order catches sorting the buffers instead of preserving declarations.
    source = " ".join(names) + "\n" + " ".join(reversed(names))
    feature = Feature(language, "function", "mixed", SIN.signature)
    result = Result(feature, "output", "supported", sources=(source, source))
    assert execution.executable(result)
    assert execution.input_buffers(source) == names
    if via_spirv:
        monkeypatch.setattr(
            execution,
            "as_spirv",
            lambda language, text: "; lowered without input names",
        )
    captured = []

    def run_suite(cases, *args, **kwargs):
        captured.extend(cases)
        return []

    monkeypatch.setattr(harness, "run_suite", run_suite)
    execution._execute(language, [(0, result)], {}, tmp_path, via_spirv)
    assert len(captured) == 2
    for case in captured:
        assert len(case.input_buffers) == len(names)
        for name, data in zip(names, case.input_buffers):
            np.testing.assert_array_equal(data, execution.inputs(name))
            assert data.dtype == execution.inputs(name).dtype


@pytest.mark.parametrize("language", ["metal", "glsl", "hlsl", "slang", "wgsl"])
def test_generated_mixed_scalar_probe_is_executable(language):
    feature = Feature(
        language,
        "function",
        "ldexp",
        Signature(Type("float"), (Param(Type("float")), Param(Type("int")))),
    )
    source = probes.build(feature, language).source
    assert execution.input_buffers(source) == ("fin", "iin")
    assert execution.executable(
        Result(feature, "output", "supported", sources=(source, source))
    )


def _mixed_interface(language, input_count, constants):
    """Discovery fixtures; SPIR-V needs only entry-point, binding, block and storage-buffer type declarations."""
    storage = [("float", "fin"), ("int", "iin"), ("uint", "uin")][:input_count]
    storage.append(("float", "outp"))
    if language == "metal":
        parameters = [
            f"device {kind}* {name} [[buffer({slot})]]"
            for slot, (kind, name) in enumerate(storage)
        ]
        if constants:
            parameters.append(f"constant float& scale [[buffer({input_count + 1})]]")
        return "kernel void probe(" + ", ".join(parameters) + ") {}\n"
    if language == "glsl":
        source = "#version 450\nlayout(local_size_x = 1) in;\n"
        source += "".join(
            f"layout(std430, binding = {slot}) buffer B{slot} {{ {kind} {name}[]; }};\n"
            for slot, (kind, name) in enumerate(storage)
        )
        if constants:
            source += f"layout(std140, binding = {input_count + 1}) uniform Params {{ float scale; }};\n"
        return source + "void main() {}\n"
    source = (
        'OpEntryPoint GLCompute %main "main"\nOpExecutionMode %main LocalSize 1 1 1\n'
    )
    source += "".join(
        f"OpDecorate %{name} DescriptorSet 0\nOpDecorate %{name} Binding {slot}\n"
        for slot, (_, name) in enumerate(storage)
    )
    scalars = {
        "float": "OpTypeFloat 32",
        "int": "OpTypeInt 32 1",
        "uint": "OpTypeInt 32 0",
    }
    source += "".join(
        f"%{name}_scalar = {scalars[kind]}\n%{name}_array = OpTypeRuntimeArray %{name}_scalar\n"
        f"%{name}_block = OpTypeStruct %{name}_array\n"
        f"%{name}_pointer = OpTypePointer StorageBuffer %{name}_block\n"
        f"%{name} = OpVariable %{name}_pointer StorageBuffer\n"
        for kind, name in storage
    )
    if constants:
        source += (
            f"OpDecorate %params DescriptorSet 0\nOpDecorate %params Binding {input_count + 1}\n"
            "OpDecorate %Params Block\n%ptr = OpTypePointer Uniform %Params\n"
            "%params = OpVariable %ptr Uniform\n"
        )
    return source


@pytest.mark.parametrize("language", ["metal", "glsl", "spirv"])
@pytest.mark.parametrize("input_count", [2, 3])
@pytest.mark.parametrize("constants", [False, True])
def test_multiple_inputs_keep_binding_and_constants_checks(
    tmp_path, language, input_count, constants
):
    source = _mixed_interface(language, input_count, constants)
    native, translated = (tmp_path / name for name in ("native", "translated"))
    native.write_text(source)
    assert (
        execution.declares_constants(language, native, input_count=input_count)
        == constants
    )
    assert (
        execution.rebinding(
            language, native, native, constants, input_count=input_count
        )
        == ""
    )
    binding = {"metal": "buffer({})", "glsl": "binding = {}", "spirv": "Binding {}"}[
        language
    ]
    # Duplicate or moved input/output slots must fail, not silently keep the validation result.
    for old, new in ((1, 0), (1, input_count + 2), (input_count, input_count + 2)):
        translated.write_text(source.replace(binding.format(old), binding.format(new)))
        detail = execution.rebinding(
            language, native, translated, constants, input_count=input_count
        )
        assert "slots" in detail
        assert str(list(range(input_count + 1))) in detail
    native.write_text(translated.read_text())
    assert (
        execution.rebinding(
            language, native, translated, constants, input_count=input_count
        )
        == ""
    )


@pytest.mark.parametrize("language", ["metal", "glsl", "spirv"])
@pytest.mark.parametrize("input_count", [2, 3])
@pytest.mark.parametrize(
    "change", ["same", "binding", "drop-constants", "add-constants"]
)
def test_execution_checks_multiple_inputs_against_the_native_interface(
    monkeypatch, language, input_count, change
):
    runtime = {
        "metal": "metal-apple",
        "glsl": "opengl-llvmpipe",
        "spirv": "vulkan-lavapipe",
    }[language]
    fake_harness(
        monkeypatch, "preserved" if change == "same" else "harness-error", 0, runtime
    )
    given = constants_given(monkeypatch)
    has_constants = change != "add-constants"
    source = _mixed_interface(language, input_count, has_constants)
    translated = source
    if change == "binding":
        binding = {
            "metal": "buffer({})",
            "glsl": "binding = {}",
            "spirv": "Binding {}",
        }[language]
        translated = source.replace(
            binding.format(input_count), binding.format(input_count + 2)
        )
    elif change != "same":
        translated = _mixed_interface(language, input_count, not has_constants)
    probe = Result(
        Feature(language, "function", "mixed", SIN.signature),
        "output",
        "supported",
        sources=(source, translated),
    )
    [result], used = execution.run([probe])
    assert used == [runtime]
    assert given == dict.fromkeys(
        ("00000-native", "00000-translated"),
        execution.CONSTANTS if has_constants else None,
    )
    assert result.outcome == ("supported" if change == "same" else "wrong")
    assert result.runtime == runtime
    if change != "same":
        assert execution.REBOUND in result.detail


@pytest.mark.parametrize("language", sorted(CALLS))
def test_no_argument_calls_keep_their_declared_unused_input(language):
    name, result = CALLS[language]
    feature = Feature(language, "function", name, Signature(result, ()))
    source = probes.build(feature, language).source
    assert execution.input_buffers(source) == ("fin",)
    assert source.count("fin") == 1
    assert execution.executable(
        Result(feature, "output", "supported", sources=(source, source))
    )
