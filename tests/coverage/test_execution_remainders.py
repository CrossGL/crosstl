"""The execution comparison of an fmod probe whose native SPIR-V uses OpFRem, which Vulkan lets round."""

import numpy as np
import pytest

from tests.conformance import runner as harness
from tests.conformance import runtimes
from tools.coverage import execution, lowering, probes
from tools.coverage.model import Feature, Param, Result, Signature, Type

FLOAT2 = Type("float", (2,))


def fmod(kind=FLOAT2, *, divisor=None, crossgl="fmod", language="hlsl"):
    divisor = divisor or kind
    signature = Signature(kind, (Param(kind), Param(divisor)))
    return Feature(language, "function", "fmod", signature, crossgl=crossgl)


def rounded(width: int):
    """The words lavapipe's OpFRem writes for a ``float<width>`` probe: x - y * trunc(x / y) in 32-bit floats."""
    data = execution.inputs("fin")
    elements = np.arange(execution.THREADS)[:, None] + np.arange(width)[None, :]
    x, y = data[elements], data[elements + width]
    quotient = np.trunc(x / y).astype(np.float32)
    return (x - (y * quotient).astype(np.float32)).astype(np.float32).reshape(-1)


def output(words):
    """``words`` at the start of a probe's output buffer, which is otherwise zero."""
    found = np.zeros(execution.THREADS * execution.WORDS, dtype=np.uint32)
    found[: len(words)] = np.asarray(words).view(np.uint32)
    return found


def test_the_remainder_reference_reads_the_arguments_the_probe_reads():
    source = probes.build(fmod(), "hlsl").source
    assert "float2 a0 = float2(fin[i + 0u], fin[i + 1u]);" in source
    assert "float2 a1 = float2(fin[i + 2u], fin[i + 3u]);" in source
    assert "outp[i] = r;" in source
    data = execution.inputs("fin")
    exact = execution.remainders(fmod()).view(np.float32)
    assert len(exact) == 2 * execution.THREADS
    for i in (0, 14, 63):
        for k in (0, 1):
            assert exact[2 * i + k] == np.fmod(data[i + k], data[i + 2 + k])
    scalar_source = probes.build(fmod(Type("float")), "hlsl").source
    assert "float a0 = fin[i + 0u];" in scalar_source
    assert "float a1 = fin[i + 1u];" in scalar_source
    scalar = execution.remainders(fmod(Type("float"))).view(np.float32)
    assert len(scalar) == execution.THREADS
    assert scalar[5] == np.fmod(data[5], data[6])


@pytest.mark.parametrize(
    "feature",
    [
        fmod(crossgl="mod"),
        fmod(Type("int", (2,))),
        fmod(divisor=Type("float")),
        fmod(Type("float", (2, 2))),
        Feature("hlsl", "function", "fmod", None, crossgl="fmod"),
    ],
)
def test_only_an_elementwise_float_fmod_has_a_remainder_reference(feature):
    assert execution.remainders(feature) is None


def test_an_exact_remainder_matches_where_the_native_probe_rounds():
    native = output(rounded(2))
    exact = execution.remainders(fmod())
    differing = np.nonzero(native[: len(exact)] != exact)[0]
    # The inputs give OpFRem's expression a rounding error in some words, as lavapipe's run of the HLSL probe shows.
    assert len(differing) > 0
    assert execution.difference(native, output(exact)).startswith(execution.DIVERGED)
    assert execution.difference(native, output(exact), exact) == ""
    assert execution.difference(native, native, exact) == ""
    other = output(exact)
    other[int(differing[0])] ^= 1
    assert "1 output words differ" in execution.difference(native, other, exact)
    past = output(exact)
    past[len(exact)] = 1
    assert "1 output words differ" in execution.difference(native, past, exact)


def _harness(monkeypatch, native_spirv: str, translated_words):
    """Run HLSL as the given SPIR-V on a stand-in for lavapipe that writes OpFRem's rounding for the native probe."""

    class Provided:
        available = True

    runtime = runtimes.RUNTIMES["vulkan-lavapipe"]
    monkeypatch.setattr(lowering, "can_lower", lambda language: True)
    monkeypatch.setattr(
        execution,
        "as_spirv",
        lambda language, source: (
            "OpFDiv" if "crosstl_fmod" in source else native_spirv
        ),
    )
    monkeypatch.setattr(execution, "rebinding", lambda *args, **kwargs: "")
    monkeypatch.setattr(execution, "declares_constants", lambda *a, **k: False)
    monkeypatch.setattr(
        harness,
        "select_runtimes",
        lambda spec, targets: {t: [(runtime, Provided())] for t in targets},
    )

    def run_suite(cases, targets, *args, **kwargs):
        records = []
        for case in cases:
            role = case.id.split("-")[1]
            words = output(rounded(2)) if role == "native" else translated_words
            case.check(case.inputs, words)
            records.append(
                {"case": case.id, "runtime": runtime.id, "outcome": "preserved"}
            )
        return records

    monkeypatch.setattr(harness, "run_suite", run_suite)


@pytest.mark.parametrize(
    ("native_spirv", "flipped", "expected"),
    [
        ("%r = OpFRem %v2float %a0 %a1", False, "supported"),
        ("%r = OpFRem %v2float %a0 %a1", True, "wrong"),
        # Without OpFRem the native probe's words are the reference, as for every other probe.
        ("%r = OpExtInst %v2float %glsl Fma %a0 %a1 %a1", False, "wrong"),
    ],
)
def test_a_translation_of_an_opfrem_probe_may_write_the_exact_remainder(
    monkeypatch, native_spirv, flipped, expected
):
    exact = execution.remainders(fmod())
    translated = output(exact)
    if flipped:
        translated[int(np.nonzero(output(rounded(2)) != translated)[0][0])] ^= 1
    _harness(monkeypatch, native_spirv, translated)
    native = probes.build(fmod(), "hlsl").source
    stand_in = native.replace("fmod(a0, a1)", "crosstl_fmod(a0, a1)")
    [result], used = execution.run(
        [Result(fmod(), "output", "supported", sources=(native, stand_in))]
    )
    assert used == ["vulkan-lavapipe"]
    assert result.outcome == expected
    if expected == "wrong":
        assert execution.DIVERGED in result.detail
