"""Exact binary32 multiply-add, including native integer-word implementations."""

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from crosstl import translate
from crosstl.project import MetalComputeRuntime
from crosstl.project.host_reflection import reflect_target_host_interface
from crosstl.project.native_runtime_drivers import (
    DirectXComputeRuntime,
    OpenGLComputeRuntime,
)
from crosstl.project.runtime_verification import (
    NativeRuntimeBufferBinding,
    NativeRuntimeDispatchRequest,
    RuntimeDispatchGeometry,
    RuntimeResourceBinding,
)
from crosstl.translator.fused_math import FMA_HELPER_KEYS, binary32_fma_support
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_FUSED_MATH"


def _source():
    names = {key: "fused_" + key for key in FMA_HELPER_KEYS}
    return "shader FusedReference {\n" + binary32_fma_support(names) + """
    StructuredBuffer<uint> values @binding(0);
    RWStructuredBuffer<uint> results @binding(1);
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void computeMain(uvec3 tid @ gl_GlobalInvocationID) @ stage_entry {
            uint i = tid.x;
            uint a = values[3u * i];
            uint b = values[3u * i + 1u];
            uint c = values[3u * i + 2u];
            results[2u * i] = fused_bits(a, b, c, false);
            results[2u * i + 1u] = fused_bits(a, b, c, true);
        }
    }
}
"""


def _translate(tmp_path, target):
    source = tmp_path / "fused.cgl"
    source.write_text(_source())
    return translate(str(source), backend=target, format_output=False)


@pytest.mark.parametrize("target", ["directx", "opengl", "metal"])
def test_integer_fma_helper_compiles(tmp_path, target):
    generated = _translate(tmp_path, target)
    assert "Berkeley SoftFloat" in generated
    assert "University of California" in generated
    assert "fused_bits(" in generated
    assert "uint64" not in generated and "double" not in generated
    _compile(generated, target, tmp_path)


def _flush(bits):
    return bits & 0x80000000 if bits & 0x7F800000 == 0 else bits


def _finite_parts(bits):
    exponent = (bits >> 23) & 255
    fraction = bits & 0x7FFFFF
    significand = fraction | (0x800000 if exponent else 0)
    return (-significand if bits >> 31 else significand), (
        exponent - 150 if exponent else -149
    )


def _oracle(a, b, c, flush=False):
    if flush:
        a, b, c = (_flush(value) for value in (a, b, c))
    magnitudes = [value & 0x7FFFFFFF for value in (a, b, c)]
    if any(value > 0x7F800000 for value in magnitudes):
        return 0x7FC00000
    product_sign = (a ^ b) & 0x80000000
    if magnitudes[0] == 0x7F800000 or magnitudes[1] == 0x7F800000:
        if 0 in magnitudes[:2] or (
            magnitudes[2] == 0x7F800000 and product_sign != (c & 0x80000000)
        ):
            return 0x7FC00000
        return product_sign | 0x7F800000
    if magnitudes[2] == 0x7F800000:
        return c
    ma, ea = _finite_parts(a)
    mb, eb = _finite_parts(b)
    mc, ec = _finite_parts(c)
    base = min(ea + eb, ec)
    exact = (ma * mb << (ea + eb - base)) + (mc << (ec - base))
    if exact == 0:
        return product_sign & c if ma * mb == mc == 0 else 0
    sign = 0x80000000 if exact < 0 else 0
    magnitude = abs(exact)
    exponent = magnitude.bit_length() - 1 + base
    if flush and exponent < -126:
        return sign
    quantum = max(exponent - 23, -149)
    shift = quantum - base
    if shift > 0:
        rounded, remainder = divmod(magnitude, 1 << shift)
        half = 1 << (shift - 1)
        rounded += remainder > half or (remainder == half and rounded & 1)
    else:
        rounded = magnitude << -shift
    if rounded == 0:
        return sign
    exponent = rounded.bit_length() - 1 + quantum
    if exponent > 127:
        return sign | 0x7F800000
    if exponent < -126:
        return sign if flush else sign | rounded
    if rounded >= 1 << 24:
        rounded >>= 1
    return sign | ((exponent + 127) << 23) | (rounded & 0x7FFFFF)


def _triples():
    edges = (
        0,
        0x80000000,
        1,
        0x80000001,
        0x7FFFFF,
        0x807FFFFF,
        0x800000,
        0x80800000,
        0x3F000000,
        0x3F7FFFFF,
        0x3F800000,
        0x3F800001,
        0xBF800000,
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7FC00000,
        0x7F800001,
    )
    triples = [(a, b, c) for a in edges for b in edges for c in edges]
    triples.append((0x3F800001, 0x3F7FFFFE, 0xBF800000))
    random_source = random.Random(1962)
    triples.extend(
        tuple(random_source.getrandbits(32) for _ in range(3)) for _ in range(8192)
    )
    return triples


def test_fma_oracle_exact_cancellation_and_signed_zero():
    assert _oracle(0x3F800001, 0x3F7FFFFE, 0xBF800000) == 0xA8800000
    assert _oracle(0x80000000, 0x3F800000, 0x80000000) == 0x80000000
    assert _oracle(0x3F800000, 0x3F800000, 0xBF800000) == 0
    assert _oracle(0x800000, 0x3F000000, 0) == 0x400000
    assert _oracle(0x800000, 0x3F000000, 0, flush=True) == 0
    assert _oracle(0x800000, 0x3F7FFFFF, 0) == 0x800000
    assert _oracle(0x800000, 0x3F7FFFFF, 0, flush=True) == 0
    assert _oracle(0x800000, 0x800000, 0x80800000, flush=True) == 0x80000000


@pytest.mark.parametrize(
    "a,b,c,expected",
    [
        (0x3F800000, 0x3F800000, 0x33800000, 0x3F800000),
        (0x3F800001, 0x3F800000, 0x33800000, 0x3F800002),
        (0x7F7FFFFF, 0x40000000, 0, 0x7F800000),
        (0xFF7FFFFF, 0x40000000, 0, 0xFF800000),
        (0x7F800000, 0, 0, 0x7FC00000),
        (0x7F800000, 0x3F800000, 0xFF800000, 0x7FC00000),
        (0x7F800001, 0x3F800000, 0, 0x7FC00000),
        (1, 0x3F800000, 0, 1),
        (1, 0x3F000000, 0, 0),
        (3, 0x3F000000, 0, 2),
    ],
)
def test_fma_oracle_rounding_boundaries(a, b, c, expected):
    assert _oracle(a, b, c) == expected


@pytest.mark.parametrize("job", ["mlx-metal-porting", "portable-host"])
def test_ci_requires_native_fma_on_all_three_platforms(job):
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, job, "Validate binary32 arithmetic"
    )
    assert "if:" not in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "tests/test_translator/test_fused_math.py" in step
    assert "tests/test_translator/test_metal_fma.py" in step
    assert "mkdir -p" in step
    assert "--basetemp=" in step
    assert "--junitxml=" in step
    assert "--timeout-seconds 120" in step
    assert "pytest -q -n auto" in step


def test_integer_fma_helper_executes(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native fused rounding")
    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    generated = _translate(tmp_path, target)
    triples = _triples()
    expected = [
        _oracle(*triple, flush=flush) for triple in triples for flush in (False, True)
    ]
    (tmp_path / "expected.json").write_text(json.dumps(expected))
    actual, evidence = _dispatch(tmp_path, target, generated, triples, len(expected))
    mismatches = [
        {
            "index": index,
            "inputBits": triples[index // 2],
            "expected": want,
            "actual": got,
        }
        for index, (want, got) in enumerate(zip(expected, actual))
        if want != got
    ]
    evidence.update(mismatchCount=len(mismatches), mismatches=mismatches)
    (tmp_path / "evidence.json").write_text(json.dumps(evidence, indent=2))
    assert len(actual) == len(expected)
    assert mismatches == []


def test_original_metal_fma_flush_profile(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for native fused rounding")
    if sys.platform != "darwin":
        pytest.skip("The original Metal control requires macOS")
    source = """#include <metal_stdlib>
using namespace metal;
kernel void fused_control(device const uint* values [[buffer(0)]],
                          device uint* results [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    float a = as_type<float>(values[3 * i]);
    float b = as_type<float>(values[3 * i + 1]);
    float c = as_type<float>(values[3 * i + 2]);
    results[2 * i] = as_type<uint>(metal::fma(a, b, c));
    results[2 * i + 1] = as_type<uint>(metal::precise::fma(a, b, c));
}
"""
    triples = _triples()
    expected = [_oracle(*triple, flush=True) for triple in triples for _ in range(2)]
    (tmp_path / "expected.json").write_text(json.dumps(expected))
    actual, evidence = _dispatch(
        tmp_path, "metal", source, triples, len(expected), entry="fused_control"
    )
    mismatches = []
    for index, (want, got) in enumerate(zip(expected, actual)):
        both_nan = (want & 0x7FFFFFFF) > 0x7F800000 and (got & 0x7FFFFFFF) > 0x7F800000
        if got != want and not both_nan:
            mismatches.append(
                {
                    "index": index,
                    "inputBits": triples[index // 2],
                    "expected": want,
                    "actual": got,
                }
            )
    evidence.update(
        profile="rne-flush-before-rounding",
        mismatchCount=len(mismatches),
        mismatches=mismatches,
        nanComparison="classification-only",
    )
    (tmp_path / "evidence.json").write_text(json.dumps(evidence, indent=2))
    assert len(actual) == len(expected)
    assert mismatches == []


def _dispatch(
    tmp_path,
    target,
    generated,
    triples,
    output_count,
    entry=None,
    initial_output=None,
    output_dtype="uint32",
):
    if initial_output is not None:
        assert len(initial_output) == output_count
    artifact, module = _compile(generated, target, tmp_path)
    assert module.is_file(), "A native compiler must produce the module"
    values = [value for triple in triples for value in triple]
    layouts = {}
    if target == "metal":
        layouts = {
            item["name"]: item["scalarLayout"]
            for item in reflect_target_host_interface(artifact, target=target)[
                "resources"
            ]
        }
    buffers = {}
    for name, slot, data, count in (
        ("values", 0, values, len(values)),
        ("results", 1, initial_output, output_count),
    ):
        output = name == "results"
        floating_output = output and output_dtype == "float32"
        buffers[name] = NativeRuntimeBufferBinding(
            name=name,
            binding=RuntimeResourceBinding(
                name=name,
                kind="buffer",
                set=0,
                binding=slot,
                type_name=("RW" if output else "")
                + f"StructuredBuffer<{'float' if floating_output else 'uint'}>",
                access="read_write" if output else "read",
                metadata={"scalarLayout": layouts[name]} if layouts else {},
            ),
            source="expectedOutput" if output else "input",
            dtype=output_dtype if output else "uint32",
            encoding="ieee754-binary32" if floating_output else None,
            shape=(count,),
            value=data,
        )
    entry = (
        entry or {"directx": "CSMain", "opengl": "main", "metal": "computeMain"}[target]
    )
    request = NativeRuntimeDispatchRequest(
        target=target,
        artifact={"target": target},
        artifact_path=artifact,
        module_path=module,
        loaded_artifact=generated if target == "opengl" else module.read_bytes(),
        buffers=buffers,
        constants={},
        entry_point=entry,
        dispatch=RuntimeDispatchGeometry(
            entry_point=entry,
            workgroup_size=(1, 1, 1),
            workgroup_count=(len(triples), 1, 1),
        ),
    )
    runtime = {
        "metal": MetalComputeRuntime,
        "directx": DirectXComputeRuntime,
        "opengl": lambda: OpenGLComputeRuntime(context_backends=("egl",)),
    }[target]()
    state = SimpleNamespace(details={})
    (tmp_path / "inputs.json").write_text(json.dumps(triples))
    if initial_output is not None:
        (tmp_path / "initial-output.json").write_text(json.dumps(initial_output))
    try:
        actual = runtime.dispatch(None, state, request)["results"]["values"]
    except Exception as error:
        (tmp_path / "runtime-error.json").write_text(
            json.dumps(
                {
                    "message": str(error),
                    "details": getattr(error, "details", {}),
                    "runtime": state.details,
                },
                indent=2,
            )
        )
        raise
    (tmp_path / "readback.json").write_text(json.dumps(actual))
    evidence = {
        "target": target,
        "inputCount": len(triples),
        "outputCount": len(actual),
        "artifactSha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "moduleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        "runtime": state.details,
    }
    return actual, evidence
