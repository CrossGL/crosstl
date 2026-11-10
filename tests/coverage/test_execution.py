import numpy as np
import pytest

from tests.conformance import runner as harness
from tools.coverage import (
    causes,
    execution,
    probes,
)
from tools.coverage.model import Feature, Param, Result, Signature, Type

from .execution_cases import (
    NATIVE,
    REBOUND,
    SIN,
    constants_given,
    fake_harness,
    metal_runs,
    supported,
)


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


def _module(inputs: tuple[str, ...], named: bool) -> str:
    """A SPIR-V compute module binding one block of a runtime array per input, then the output and a uniform block."""
    scalars = {
        "fin": "OpTypeFloat 32",
        "iin": "OpTypeInt 32 1",
        "half": "OpTypeFloat 16",
    }
    lines = [
        'OpEntryPoint GLCompute %main "main"',
        "OpExecutionMode %main LocalSize 1 1 1",
    ]
    for slot, name in enumerate((*inputs, "fin")):
        if named:
            lines.append(
                f'OpMemberName %block{slot} 0 "{name if slot < len(inputs) else "outp"}"'
            )
        lines += [
            f"OpDecorate %var{slot} Binding {slot}",
            f"OpDecorate %var{slot} DescriptorSet 0",
        ]
    lines.append(f"OpDecorate %params Binding {len(inputs) + 1}")
    for slot, name in enumerate((*inputs, "fin")):
        lines += [
            f"%scalar{slot} = {scalars[name]}",
            f"%array{slot} = OpTypeRuntimeArray %scalar{slot}",
            f"%block{slot} = OpTypeStruct %array{slot}",
            f"%pointer{slot} = OpTypePointer StorageBuffer %block{slot}",
            f"%var{slot} = OpVariable %pointer{slot} StorageBuffer",
        ]
    lines += [
        "%v4float = OpTypeVector %scalar0 4",
        "%Params = OpTypeStruct %v4float",
        "%params_pointer = OpTypePointer Uniform %Params",
        "%params = OpVariable %params_pointer Uniform",
    ]
    return "\n".join(lines) + "\n"


def test_spirv_inputs_are_the_storage_buffers_by_binding_and_scalar_type():
    # A SPIR-V producer may drop debug names, so a round trip without them still runs in the native probe's place.
    native = _module(("fin", "iin"), named=True)
    assert execution.input_buffers(native) == ("fin", "iin")
    assert execution.input_buffers(_module(("fin", "iin"), named=False)) == (
        "fin",
        "iin",
    )
    spirv = Feature("spirv", "function", "Ldexp", SIN.signature)

    def runs(translated: str) -> bool:
        return execution.executable(
            Result(spirv, "input", "supported", sources=(native, translated))
        )

    assert runs(_module(("fin", "iin"), named=False))
    # Swapped bindings, or a 16-bit float buffer the harness has no input for, keep the validation result.
    assert not runs(_module(("iin", "fin"), named=True))
    assert not runs(_module(("fin", "half"), named=True))
    assert not execution.executable(
        Result(spirv, "input", "supported", sources=(_module(("half",), True),) * 2)
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
    fake_harness(monkeypatch, outcome, written)
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
    fake_harness(monkeypatch, "driver-rejected", 0, message=message)
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
    fake_harness(monkeypatch, "driver-rejected", 0, message="x" * 2000)
    [result], _ = execution.run([supported()])
    quoted = result.detail.partition("\n")[2]
    assert len(quoted) == 600 and quoted.endswith("…")


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
    fake_harness(monkeypatch, outcome, 0)
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
    fake_harness(monkeypatch, "harness-error", 0)
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


def test_a_probe_that_declares_a_constants_block_gets_one(monkeypatch, tmp_path):
    fake_harness(monkeypatch, "preserved", 0)
    given = constants_given(monkeypatch)
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
    # Four std140 rows whose words all differ, so a translation that reads another offset reads another value.
    words = np.frombuffer(execution.CONSTANTS, np.float32).tolist()
    assert words[:4] == [2.5, -1.75, 3.25, 0.5]
    assert len(words) == 16 and len(set(words)) == 16
    path = tmp_path / "scaled.metal"
    path.write_text(SCALED)
    assert execution.declares_constants("metal", path)
    path.write_text(NATIVE)
    assert not execution.declares_constants("metal", path)


def test_a_translation_without_the_native_constants_block_is_wrong(monkeypatch):
    fake_harness(monkeypatch, "preserved", 0)
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
