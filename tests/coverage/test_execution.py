from dataclasses import replace

import numpy as np
import pytest

from tests.conformance import runner as harness
from tests.conformance import runtimes
from tools.coverage import (
    causes,
    execution,
    issues,
    probes,
    publish,
    report,
    runner,
    validators,
)
from tools.coverage.model import VOID, Feature, Param, Result, Signature, Type

SIN = Feature(
    "metal",
    "function",
    "sin",
    Signature(Type("float"), (Param(Type("float")),)),
    crossgl="sin",
)
NATIVE = probes.build(SIN, "metal").source


def supported(translated=NATIVE, direction="output"):
    return Result(SIN, direction, "supported", sources=(NATIVE, translated))


def metal_runs():
    try:
        return runtimes.provision(runtimes.RUNTIMES["metal-apple"]).available
    except Exception:  # noqa: BLE001 - an unprovisionable runtime is an unavailable one
        return False


def test_only_supported_probes_with_matching_ordered_input_buffers_run():
    assert execution.executable(supported())
    assert not execution.executable(Result(SIN, "output", "supported"))
    assert not execution.executable(
        Result(SIN, "output", "wrong", sources=(NATIVE, NATIVE))
    )
    ldexp = Feature(
        "metal",
        "function",
        "ldexp",
        Signature(Type("float"), (Param(Type("float")), Param(Type("int")))),
    )
    two_buffers = probes.build(ldexp, "metal").source
    assert not execution.executable(supported(two_buffers))
    assert execution.executable(
        Result(SIN, "output", "supported", sources=(two_buffers, two_buffers))
    )
    reordered = (
        two_buffers.replace("fin", "temporary")
        .replace("iin", "fin")
        .replace("temporary", "iin")
    )
    assert not execution.executable(
        Result(SIN, "output", "supported", sources=(two_buffers, reordered))
    )
    assert not execution.executable(
        Result(SIN, "output", "supported", sources=("void main() {}", "void main() {}"))
    )


def test_every_run_gets_the_same_inputs_and_small_integers():
    assert np.array_equal(execution.inputs("fin"), execution.inputs("fin"))
    for buffer, dtype in (("iin", np.int32), ("uin", np.uint32)):
        data = execution.inputs(buffer)
        assert data.dtype == dtype
        assert data.min() >= 0 and data.max() < 16
    assert len(execution.inputs("fin")) == execution.THREADS + execution.SPAN


def test_results_match_word_for_word_and_nan_matches_nan():
    words = np.array([0x3F800000, 0x7FC00000, 0xA5A5A5A5], dtype=np.uint32)
    other_nan = np.array([0x3F800000, 0x7FC00001, 0xA5A5A5A5], dtype=np.uint32)
    assert execution.difference(words, other_nan) == ""
    changed = np.array([0x3F800001, 0x7FC00000, 0], dtype=np.uint32)
    found = execution.difference(words, changed)
    assert found.startswith(execution.DIVERGED + "\n")
    assert "2 output words differ. The first, word 0, is 0x3f800001" in found
    assert "the native probe writes 0x3f800000" in found


def _fake_harness(
    monkeypatch,
    translated_outcome,
    written,
    runtime="metal-apple",
    environments=None,
    message="",
):
    """Make run_suite report the native probe as run and the translation as ``translated_outcome``.

    ``environments``, when given, collects the Vulkan environment of each run. ``message`` is what the runtime says
    about the translation.
    """

    class Provided:
        available = True

    runtime = runtimes.RUNTIMES[runtime]
    monkeypatch.setattr(
        harness,
        "select_runtimes",
        lambda spec, targets: {t: [(runtime, Provided())] for t in targets},
    )

    def run_suite(
        cases, targets, configs, selection, root, *, reference=False, environment
    ):
        assert reference and targets == [runtime.target] and configs == ["strict"]
        if environments is not None:
            environments.append(environment)
        records = []
        for case in cases:
            assert "#version 460" not in case.src.read_text()
            role = case.id.split("-")[1]
            outcome = "preserved" if role == "native" else translated_outcome
            words = np.zeros(case.out_count, dtype=np.uint32)
            if role == "translated":
                words[:] = written
            if outcome in ("preserved", "memory-corruption"):
                case.check(case.inputs, words)
            record = {"case": case.id, "runtime": runtime.id, "outcome": outcome}
            if role == "translated" and message:
                record["execution"] = {"status": outcome, "message": message}
            records.append(record)
        return records

    monkeypatch.setattr(harness, "run_suite", run_suite)


@pytest.mark.parametrize(
    ("outcome", "written", "expected", "detail"),
    [
        ("preserved", 0, "supported", ""),
        (
            "preserved",
            1,
            "wrong",
            f"on metal-apple, the translation {execution.DIVERGED}",
        ),
        (
            "memory-corruption",
            0,
            "wrong",
            "on metal-apple, the translation writes outside its output buffer",
        ),
        ("driver-rejected", 0, "wrong", "on metal-apple, the translation is rejected"),
        # No evidence either way: the validation result stands.
        ("timeout", 0, "supported", ""),
        ("interface-changed", 0, "supported", ""),
    ],
)
def test_a_translation_that_computes_otherwise_is_wrong(
    monkeypatch, outcome, written, expected, detail
):
    _fake_harness(monkeypatch, outcome, written)
    untouched = Result(SIN, "input", "refused", "a diagnostic")
    [result, other], used = execution.run([supported(), untouched], workers=2)
    assert used == ["metal-apple"]
    assert other == untouched
    assert result.outcome == expected
    assert result.detail.startswith(detail)
    ran = outcome in ("preserved", "memory-corruption", "driver-rejected")
    assert result.runtime == ("metal-apple" if ran else "")
    assert execution.found(result.detail) == (expected == "wrong")


def test_a_rejected_translation_quotes_the_runtime(monkeypatch):
    errors = "".join(
        f"0:{line}(1): error: '{line}' : syntax error\n" for line in range(2, 12)
    )
    message = "GLSL Compiler failed\n\ncompute_shader\n==============\n" + errors
    _fake_harness(monkeypatch, "driver-rejected", 0, message=message)
    [result], _ = execution.run([supported()])
    first, _, quoted = result.detail.partition("\n")
    assert first == f"on metal-apple, the translation {execution.REJECTED}"
    assert quoted.splitlines() == [
        "GLSL Compiler failed",
        "compute_shader",
        "==============",
        "0:2(1): error: '2' : syntax error",
        "0:3(1): error: '3' : syntax error",
        "0:4(1): error: '4' : syntax error",
    ]
    # The quote follows the first line, so it does not change the result's cause or its issue.
    feature = {"name": "sin", "outcome": "wrong", "detail": result.detail}
    assert causes.diagnostic(feature) == first
    _fake_harness(monkeypatch, "driver-rejected", 0, message="x" * 2000)
    [result], _ = execution.run([supported()])
    quoted = result.detail.partition("\n")[2]
    assert len(quoted) == 600 and quoted.endswith("…")


# The native Metal probe with its buffers moved from slots 0 and 1, as a host binding the native probe's buffers
# would not find them.
REBOUND = NATIVE.replace("[[buffer(0)]]", "[[buffer(2)]]").replace(
    "[[buffer(1)]]", "[[buffer(3)]]"
)


def test_rebinding_names_what_the_harness_cannot_bind(tmp_path):
    native, rebound, broken = (tmp_path / f"{name}.metal" for name in "nrb")
    native.write_text(NATIVE)
    rebound.write_text(REBOUND)
    broken.write_text("kernel void a() {}\nkernel void b() {}\n")
    assert execution.rebinding("metal", native, native, False) == ""
    assert (
        execution.rebinding("metal", native, rebound, False)
        == "Expected Metal buffer slots [0, 1], found [2, 3]"
    )
    # A native probe the harness cannot bind either is no evidence against the translation.
    assert execution.rebinding("metal", broken, rebound, False) == ""


@pytest.mark.parametrize("outcome", ["harness-error", "preserved"])
def test_a_translation_that_binds_other_resources_is_wrong(monkeypatch, outcome):
    _fake_harness(monkeypatch, outcome, 0)
    [result], used = execution.run([supported(REBOUND)])
    assert used == ["metal-apple"]
    assert (result.outcome, result.runtime) == ("wrong", "metal-apple")
    assert result.detail == (
        f"on metal-apple, the translation {execution.REBOUND}\n"
        "Expected Metal buffer slots [0, 1], found [2, 3]"
    )
    assert execution.found(result.detail)


def test_a_rebound_translation_keeps_its_result_when_the_native_probe_does_not_run(
    monkeypatch,
):
    _fake_harness(monkeypatch, "harness-error", 0)
    original = harness.run_suite

    def native_fails(*args, **kwargs):
        records = original(*args, **kwargs)
        return [dict(record, outcome="timeout") for record in records]

    monkeypatch.setattr(harness, "run_suite", native_fails)
    [result], _ = execution.run([supported(REBOUND)])
    assert (result.outcome, result.runtime) == ("supported", "")


# A Metal probe that scales its input by the first float of a constants block, as the uniform-buffer probe does.
SCALED = """#include <metal_stdlib>
using namespace metal;
struct Params {
    float scale;
};
kernel void probe(device float* fin [[buffer(0)]], device float* outp [[buffer(1)]],
                  uint3 id [[thread_position_in_grid]], constant Params& params [[buffer(3)]]) {
    uint i = id.x;
    outp[i] = fin[i] * params.scale;
}
"""


def _constants_given(monkeypatch):
    """Collect the constants block of every case the fake harness runs, by case id."""
    given = {}
    original = harness.run_suite

    def run_suite(cases, *args, **kwargs):
        given.update((case.id, case.constants) for case in cases)
        return original(cases, *args, **kwargs)

    monkeypatch.setattr(harness, "run_suite", run_suite)
    return given


def test_a_probe_that_declares_a_constants_block_gets_one(monkeypatch, tmp_path):
    _fake_harness(monkeypatch, "preserved", 0)
    given = _constants_given(monkeypatch)
    [scaled, plain], _ = execution.run(
        [
            Result(SIN, "output", "supported", sources=(SCALED, SCALED)),
            supported(),
        ]
    )
    assert scaled.outcome == plain.outcome == "supported"
    assert given == {
        "00000-native": execution.CONSTANTS,
        "00000-translated": execution.CONSTANTS,
        "00001-native": None,
        "00001-translated": None,
    }
    assert np.frombuffer(execution.CONSTANTS, np.float32).tolist() == [
        2.5,
        -1.75,
        3.25,
        0.5,
    ]
    path = tmp_path / "scaled.metal"
    path.write_text(SCALED)
    assert execution.declares_constants("metal", path)
    path.write_text(NATIVE)
    assert not execution.declares_constants("metal", path)


def test_a_translation_without_the_native_constants_block_is_wrong(monkeypatch):
    _fake_harness(monkeypatch, "preserved", 0)
    [result], _ = execution.run(
        [Result(SIN, "output", "supported", sources=(SCALED, NATIVE))]
    )
    assert result.outcome == "wrong"
    assert result.detail.startswith(
        f"on metal-apple, the translation {execution.REBOUND}\n"
    )
    assert "Expected one constants block" in result.detail


@pytest.mark.skipif(not metal_runs(), reason="no Metal runtime")
def test_metal_reads_the_constants_block_at_the_translation_offsets():
    shifted = SCALED.replace("float scale;", "float pad;\n    float scale;")
    (same, other), used = execution.run(
        [
            Result(SIN, "output", "supported", sources=(SCALED, SCALED)),
            Result(SIN, "output", "supported", sources=(SCALED, shifted)),
        ]
    )
    assert used == ["metal-apple"]
    assert (same.outcome, same.runtime) == ("supported", "metal-apple")
    assert other.outcome == "wrong" and execution.DIVERGED in other.detail


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
    monkeypatch.setattr(validators, "can_lower", lambda language: True)
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


def _hlsl(translated=None):
    feature = Feature("hlsl", "function", "sin", SIN.signature, crossgl="sin")
    native = probes.build(feature, "hlsl").source
    return Result(
        feature, "output", "supported", sources=(native, translated or native)
    )


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
    _fake_harness(monkeypatch, translated_outcome, written, "vulkan-lavapipe")
    monkeypatch.setattr(validators, "can_lower", lambda language: True)
    lowered = []

    def as_spirv(language, source):
        lowered.append(language)
        return f"; {language} as SPIR-V\n"

    monkeypatch.setattr(execution, "as_spirv", as_spirv)
    [result], used = execution.run([_hlsl()])
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
    _fake_harness(monkeypatch, "preserved", 0, "vulkan-lavapipe", environments)
    monkeypatch.setattr(validators, "can_lower", lambda language: True)
    versions = {"slang": "1.5", "hlsl": "1.3"}
    monkeypatch.setattr(
        execution,
        "as_spirv",
        lambda language, source: f"; SPIR-V\n; Version: {versions[language]}\n",
    )
    feature = Feature("slang", "function", "sin", SIN.signature, crossgl="sin")
    native = probes.build(feature, "slang").source
    slang = Result(feature, "input", "supported", sources=(native, native))
    [result, hlsl], used = execution.run([slang, _hlsl()], workers=1)
    assert sorted(environments) == ["vulkan1.1", "vulkan1.2"]
    assert result.runtime == hlsl.runtime == "vulkan-lavapipe"


@pytest.mark.parametrize("language", sorted(execution.VIA_SPIRV))
def test_the_native_probe_runs_as_the_spirv_its_compiler_writes(language):
    """A native compute probe, lowered by its reference compiler, runs on a Vulkan runtime and computes."""
    if not validators.can_lower(language):
        pytest.skip(f"{validators.TOOLS[language]} or spirv-dis is not installed")
    feature = Feature(language, "function", "sin", SIN.signature, crossgl="sin")
    native = probes.build(feature, language).source
    probe = Result(feature, "output", "supported", sources=(native, native))
    [result], used = execution.run([probe], workers=1)
    if not used:
        pytest.skip("no Vulkan runtime")
    assert (result.outcome, result.runtime) == ("supported", used[0])


def test_a_probe_its_compiler_cannot_lower_keeps_its_validation_result(monkeypatch):
    _fake_harness(monkeypatch, "preserved", 1, "vulkan-lavapipe")
    monkeypatch.setattr(validators, "can_lower", lambda language: True)
    monkeypatch.setattr(
        execution,
        "as_spirv",
        lambda language, source: "" if "CSMain" in source else ";",
    )
    probe = _hlsl(_hlsl().sources[0].replace("void main", "void CSMain"))
    assert execution.run([probe])[0] == [probe]
    monkeypatch.setattr(validators, "can_lower", lambda language: False)
    assert execution.run([_hlsl()]) == ([_hlsl()], [])
    assert execution.idle([_hlsl()]) == ["hlsl"]


@pytest.mark.parametrize("language", ["hlsl", "spirv"])
def test_a_probe_that_rounds_through_16_bit_floats_keeps_its_validation_result(
    monkeypatch, language
):
    # lavapipe drops a float-to-half-to-float round trip, so such a native probe is no reference.
    _fake_harness(monkeypatch, "preserved", 1, "vulkan-lavapipe")
    monkeypatch.setattr(validators, "can_lower", lambda language: True)
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
    _fake_harness(monkeypatch, "preserved", 1, "vulkan-lavapipe")
    monkeypatch.setattr(validators, "can_lower", lambda language: True)
    feature = Feature("slang", "function", "clockARB", Signature(Type("uint"), ()))
    for source, outcome in ((f"%1 = {instruction}\n", "supported"), ("", "wrong")):
        source += "; declares fin\n"
        monkeypatch.setattr(execution, "as_spirv", lambda language, text: source)
        probe = Result(feature, "input", "supported", sources=(source, source))
        assert execution.run([probe])[0][0].outcome == outcome


# A call that takes no argument, whose probe declares an input it never reads.
CALLS = {
    "hlsl": ("WaveGetLaneIndex", Type("uint")),
    "slang": ("WaveGetLaneIndex", Type("uint")),
    "wgsl": ("workgroupBarrier", VOID),
}


@pytest.mark.parametrize("language", sorted(execution.VIA_SPIRV))
def test_the_reference_compiler_keeps_the_harness_interface(tmp_path, language):
    if not validators.can_lower(language):
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
    if not validators.can_lower("hlsl"):
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
    hlsl = _hlsl().feature
    assert runner.run_output(hlsl, "hlsl").sources == (
        _hlsl().sources[0],
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
    """Discovery fixtures; SPIR-V needs only entry-point, binding and block declarations."""
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
    _fake_harness(
        monkeypatch, "preserved" if change == "same" else "harness-error", 0, runtime
    )
    given = _constants_given(monkeypatch)
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
