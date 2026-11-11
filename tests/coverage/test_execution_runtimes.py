"""Which runtime each language's probes run on: GLSL on OpenGL or Vulkan, the others as SPIR-V."""

import numpy as np
import pytest

from tests.conformance import runner as harness
from tests.conformance import runtimes
from tools.coverage import execution, lowering, probes, validators
from tools.coverage.model import Feature, Result, Signature, Type

from .execution_cases import CALLS, NATIVE, SIN, fake_harness, hlsl_result


def test_glsl_runs_as_glsl_450_because_llvmpipe_implements_opengl_45():
    assert execution.runnable("glsl", "#version 460\nvoid main() {}\n") == (
        "#version 450 core\nvoid main() {}\n"
    )
    assert execution.runnable("glsl", "#version 450 core\n") == "#version 450 core\n"
    assert execution.runnable("metal", NATIVE) == NATIVE


def test_glsl_460_group_votes_run_through_arb_shader_group_vote():
    source = "#version 460\nvoid main() { bool r = allInvocationsEqual (b) || anyInvocation(c); }\n"
    assert execution.runnable("glsl", source) == (
        "#version 450 core\n#extension GL_ARB_shader_group_vote : require\n"
        "void main() { bool r = allInvocationsEqualARB (b) || anyInvocationARB(c); }\n"
    )
    # GLSL 4.50 has no unsuffixed vote functions, so a source that declares 4.50 runs as written.
    written = "#version 450 core\nvoid main() { bool r = allInvocations(b); }\n"
    assert execution.runnable("glsl", written) == written


def _glsl(declared="#version 460"):
    feature = Feature("glsl", "function", "sin", SIN.signature, crossgl="sin")
    native = probes.build(feature, "glsl").source
    assert native.startswith("#version 460\n")
    translated = native.replace("#version 460", declared, 1)
    return Result(feature, "input", "supported", sources=(native, translated))


def _opengl_then_vulkan(monkeypatch, outcomes, written=0):
    """Run GLSL on llvmpipe and SPIR-V on lavapipe, giving each probe ``outcomes[(target, role)]``, or "preserved".

    A translation that runs as SPIR-V writes ``written``. Returns the targets of the runs, in order.
    """

    class Provided:
        available = True

    on = {
        "glsl": runtimes.RUNTIMES["opengl-llvmpipe"],
        "spirv": runtimes.RUNTIMES["vulkan-lavapipe"],
    }
    monkeypatch.setattr(
        harness,
        "select_runtimes",
        lambda spec, targets: {t: [(on[t], Provided())] for t in targets},
    )
    monkeypatch.setattr(lowering, "can_lower", lambda language: True)
    monkeypatch.setattr(
        execution, "as_spirv", lambda language, source: f"; {language} as SPIR-V\n"
    )
    ran = []

    def run_suite(
        cases, targets, configs, selection, root, *, reference=False, environment
    ):
        [target] = targets
        ran.append(target)
        records = []
        for case in cases:
            role = case.id.split("-")[1]
            outcome = outcomes.get((target, role), "preserved")
            words = np.zeros(case.out_count, dtype=np.uint32)
            if role == "translated" and target == "spirv":
                words[:] = written
            if outcome == "preserved":
                case.check(case.inputs, words)
            records.append(
                {"case": case.id, "runtime": on[target].id, "outcome": outcome}
            )
        return records

    monkeypatch.setattr(harness, "run_suite", run_suite)
    return ran


@pytest.mark.parametrize(
    ("declared", "ran", "expected"),
    [
        # llvmpipe compiles a 4.60 translation as 4.50, so its rejection decides nothing and both run on Vulkan.
        ("#version 460", ["glsl", "spirv"], "supported"),
        ("#version 450 core", ["glsl"], "wrong"),
    ],
)
def test_a_rejected_glsl_translation_is_wrong_unless_it_declares_460(
    monkeypatch, declared, ran, expected
):
    targets = _opengl_then_vulkan(
        monkeypatch, {("glsl", "translated"): "driver-rejected"}
    )
    [result], used = execution.run([_glsl(declared)])
    assert targets == ran
    assert result.outcome == expected
    assert used == ["opengl-llvmpipe", "vulkan-lavapipe"][: len(ran)]


@pytest.mark.parametrize("rejection", ["driver-rejected", "invalid"])
@pytest.mark.parametrize(("written", "expected"), [(0, "supported"), (1, "wrong")])
def test_glsl_probes_llvmpipe_rejects_run_as_spirv_on_vulkan(
    monkeypatch, rejection, written, expected
):
    targets = _opengl_then_vulkan(monkeypatch, {("glsl", "native"): rejection}, written)
    [result], used = execution.run([_glsl()])
    assert targets == ["glsl", "spirv"]
    assert used == ["opengl-llvmpipe", "vulkan-lavapipe"]
    assert (result.outcome, result.runtime) == (expected, "vulkan-lavapipe")
    detail = f"on vulkan-lavapipe as SPIR-V from glslangValidator, the translation {execution.DIVERGED}"
    assert result.detail.startswith(detail) == (expected == "wrong")
    assert execution.found(result.detail) == (expected == "wrong")


def test_a_glsl_probe_glslang_cannot_lower_keeps_its_validation_result(monkeypatch):
    targets = _opengl_then_vulkan(monkeypatch, {("glsl", "native"): "driver-rejected"})
    monkeypatch.setattr(execution, "as_spirv", lambda language, source: "")
    [result], used = execution.run([_glsl()])
    assert targets == ["glsl"]
    assert used == ["opengl-llvmpipe"]
    assert result.outcome == "supported" and not result.runtime


def test_a_glsl_probe_llvmpipe_runs_does_not_run_again(monkeypatch):
    targets = _opengl_then_vulkan(monkeypatch, {}, written=1)
    [result], used = execution.run([_glsl()])
    assert targets == ["glsl"] and used == ["opengl-llvmpipe"]
    assert (result.outcome, result.runtime) == ("supported", "opengl-llvmpipe")


@pytest.mark.parametrize(
    ("translated_outcome", "written", "expected", "detail"),
    [
        ("preserved", 0, "supported", ""),
        (
            "preserved",
            1,
            "wrong",
            f"on vulkan-lavapipe as SPIR-V from dxc, the translation {execution.DIVERGED}",
        ),
        ("harness-error", 0, "supported", ""),
    ],
)
def test_hlsl_slang_and_wgsl_probes_run_as_spirv_on_vulkan(
    monkeypatch, translated_outcome, written, expected, detail
):
    fake_harness(monkeypatch, translated_outcome, written, "vulkan-lavapipe")
    monkeypatch.setattr(lowering, "can_lower", lambda language: True)
    lowered = []

    def as_spirv(language, source):
        lowered.append(language)
        return f"; {language} as SPIR-V\n"

    monkeypatch.setattr(execution, "as_spirv", as_spirv)
    [result], used = execution.run([hlsl_result()])
    assert lowered == ["hlsl", "hlsl"]
    assert used == ["vulkan-lavapipe"]
    assert result.outcome == expected
    assert result.detail.startswith(detail)
    assert execution.found(result.detail) == (expected == "wrong")
    # A translation that did not run leaves HLSL without a probe that ran.
    assert execution.idle([result]) == ([] if result.runtime else ["hlsl"])


@pytest.mark.parametrize(
    ("version", "expected"),
    [("", "vulkan1.1"), ("1.0", "vulkan1.1"), ("1.3", "vulkan1.1")]
    + [("1.4", "vulkan1.2"), ("1.5", "vulkan1.2"), ("1.6", "vulkan1.3")],
)
def test_spirv_runs_in_the_oldest_environment_that_takes_its_version(version, expected):
    header = f"; SPIR-V\n; Version: {version}\n" if version else ""
    assert execution.environment([f"{header}OpCapability Shader\n"]) == expected
    older = "; SPIR-V\n; Version: 1.0\n"
    assert execution.environment([older, header]) == expected


def test_slang_probes_run_in_the_environment_of_the_spirv_slangc_writes(
    monkeypatch,
):
    # slangc writes SPIR-V 1.5, which spirv-val rejects for Vulkan 1.1, so no Slang probe would ever run there.
    environments = []
    fake_harness(monkeypatch, "preserved", 0, "vulkan-lavapipe", environments)
    monkeypatch.setattr(lowering, "can_lower", lambda language: True)
    versions = {"slang": "1.5", "hlsl": "1.3"}
    monkeypatch.setattr(
        execution,
        "as_spirv",
        lambda language, source: f"; SPIR-V\n; Version: {versions[language]}\n",
    )
    feature = Feature("slang", "function", "sin", SIN.signature, crossgl="sin")
    native = probes.build(feature, "slang").source
    slang = Result(feature, "input", "supported", sources=(native, native))
    [result, hlsl], used = execution.run([slang, hlsl_result()], workers=1)
    assert sorted(environments) == ["vulkan1.1", "vulkan1.2"]
    assert result.runtime == hlsl.runtime == "vulkan-lavapipe"


# Rust's sine is a method of the Float trait; the other languages call a function.
_SINE = {"rust": ("Float::sin", ".sin")}


@pytest.mark.parametrize("language", sorted(execution.VIA_SPIRV))
def test_the_native_probe_runs_as_the_spirv_its_compiler_writes(language):
    """A native compute probe, lowered by its reference compiler, runs on a Vulkan runtime and computes, and a
    translation that computes the cosine instead is wrong."""
    if not lowering.can_lower(language):
        pytest.skip(f"{validators.TOOLS[language]} or spirv-dis is not installed")
    name, spelling = _SINE.get(language, ("sin", ""))
    feature = Feature(
        language, "function", name, SIN.signature, crossgl="sin", spelling=spelling
    )
    native = probes.build(feature, language).source
    cosine = native.replace("sin(", "cos(")
    assert cosine != native
    probe = Result(feature, "output", "supported", sources=(native, native))
    other = Result(feature, "output", "supported", sources=(native, cosine))
    [result, wrong], used = execution.run([probe, other], workers=1)
    if not used:
        pytest.skip("no Vulkan runtime")
    assert (result.outcome, result.runtime) == ("supported", used[0])
    assert (wrong.outcome, wrong.runtime) == ("wrong", used[0])
    assert execution.found(wrong.detail)


def test_a_probe_its_compiler_cannot_lower_keeps_its_validation_result(monkeypatch):
    fake_harness(monkeypatch, "preserved", 1, "vulkan-lavapipe")
    monkeypatch.setattr(lowering, "can_lower", lambda language: True)
    monkeypatch.setattr(
        execution,
        "as_spirv",
        lambda language, source: "" if "CSMain" in source else ";",
    )
    probe = hlsl_result(hlsl_result().sources[0].replace("void main", "void CSMain"))
    assert execution.run([probe])[0] == [probe]
    monkeypatch.setattr(lowering, "can_lower", lambda language: False)
    assert execution.run([hlsl_result()]) == ([hlsl_result()], [])
    assert execution.idle([hlsl_result()]) == ["hlsl"]


@pytest.mark.parametrize("language", ["hlsl", "spirv"])
def test_a_probe_that_rounds_through_16_bit_floats_keeps_its_validation_result(
    monkeypatch, language
):
    # lavapipe drops a float-to-half-to-float round trip, so such a native probe is no reference.
    fake_harness(monkeypatch, "preserved", 1, "vulkan-lavapipe")
    monkeypatch.setattr(lowering, "can_lower", lambda language: True)
    half = "%half = OpTypeFloat 16\n; reads fin\n"
    single = "%float = OpTypeFloat 32\n; reads fin\n"
    feature = Feature(language, "function", "asfloat16", SIN.signature)
    for source, outcome in ((half, "supported"), (single, "wrong")):
        monkeypatch.setattr(execution, "as_spirv", lambda language, text: source)
        probe = Result(feature, "output", "supported", sources=(source, source))
        assert execution.run([probe])[0][0].outcome == outcome


@pytest.mark.parametrize("instruction", ["OpReadClockKHR", "OpTypeRayQueryKHR"])
def test_a_probe_that_reads_a_clock_or_a_ray_query_keeps_its_validation_result(
    monkeypatch, instruction
):
    # Neither a clock nor a ray query the harness never initializes gives two runs the same result.
    fake_harness(monkeypatch, "preserved", 1, "vulkan-lavapipe")
    monkeypatch.setattr(lowering, "can_lower", lambda language: True)
    feature = Feature("slang", "function", "clockARB", Signature(Type("uint"), ()))
    for source, outcome in ((f"%1 = {instruction}\n", "supported"), ("", "wrong")):
        source += "; declares fin\n"
        monkeypatch.setattr(execution, "as_spirv", lambda language, text: source)
        probe = Result(feature, "input", "supported", sources=(source, source))
        assert execution.run([probe])[0][0].outcome == outcome


@pytest.mark.parametrize("language", sorted(execution.VIA_SPIRV))
def test_the_reference_compiler_keeps_the_harness_interface(tmp_path, language):
    if not lowering.can_lower(language):
        pytest.skip(f"{validators.TOOLS[language]} or spirv-dis is not installed")
    from tests.conformance import interface

    feature = Feature(language, "function", "sin", SIN.signature, crossgl="sin")
    name, result = CALLS[language]
    call = Feature(language, "function", name, Signature(result, ()))
    path = tmp_path / "probe.spvasm"
    for probe in (feature, call):
        source = probes.build(probe, language).source
        path.write_text(execution.as_spirv(language, source))
        found = interface.interface(path, "spirv", 1)
        assert found["bindings"] == (0, 1) and found["group"] == (1, 1, 1)
    source = probes.build(feature, language).source
    if language == "hlsl":
        renamed = source.replace("void main", "void CSMain")
        path.write_text(execution.as_spirv(language, renamed))
        assert interface.interface(path, "spirv", 1)["entry"] == "CSMain"
    assert execution.as_spirv(language, "not a shader") == ""
