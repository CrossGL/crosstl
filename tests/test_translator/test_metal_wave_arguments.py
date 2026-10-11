"""Source builtin parameter conversions survive native wave lowering."""

import hashlib
import json
import os
import struct
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalSourceOverloadResolutionError
from crosstl.project import build_native_loader_dispatch_request
from tests.ci_helpers import assert_paths_covered
from tests.test_backend.test_metal.test_codegen import convert, parse_crossgl
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_native_runtime import _native_request
from tests.test_translator.test_native_loader_dispatch_integration import _executor
from tests.test_translator.test_software_subgroup_product import (
    _check_outputs,
    _package,
)

REQUIRE_ENV = "CROSTL_REQUIRE_WAVE_ARGUMENT_READBACKS"
OFFSETS = (0, 1, 31, 32, 65535, 65536, 65537, 0xFFFFFFFF)


@pytest.mark.parametrize(
    "operation,arguments",
    [
        ("simd_broadcast", "value, offset"),
        ("simd_shuffle", "value, offset"),
        ("simd_shuffle_down", "value, offset"),
        ("simd_shuffle_up", "value, offset"),
        ("simd_shuffle_xor", "value, offset"),
        ("simd_shuffle_and_fill_up", "value, value, offset, modulo"),
    ],
)
@pytest.mark.parametrize("prefix", ["", "metal::"])
def test_wave_builtin_parameters_retain_source_width(operation, arguments, prefix):
    code = convert(
        f"float f(float value, uint offset, uint modulo) {{ return {prefix}{operation}({arguments}); }}"
    )
    assert "(uint(offset) & 65535u)" in code
    assert ("(uint(modulo) & 65535u)" in code) == ("modulo" in arguments)
    assert parse_crossgl(code) is not None


def test_source_wave_override_keeps_its_own_parameter_contract():
    code = convert("""
    uint simd_shuffle_down(uint value, uint delta) { return value ^ delta; }
    uint user_call(uint value, uint delta) { return simd_shuffle_down(value, delta); }
    uint native_call(uint value, uint delta) { return metal::simd_shuffle_down(value, delta); }
    """)
    assert "simd_shuffle_down(value, delta)" in code
    assert "WaveShuffleDown(value, (uint(delta) & 65535u))" in code


def test_canonical_wave_arguments_are_not_narrowed_by_source_contracts():
    code = convert(
        "uint f(uint value, uint delta) { return WaveShuffleDown(value, delta); }"
    )
    assert "WaveShuffleDown(value, delta)" in code
    assert "65535" not in code


def test_unrelated_namespace_does_not_hide_the_wave_builtin():
    code = convert("""
    using namespace metal;
    namespace custom {
        uint simd_shuffle_down(uint value, uint delta) { return value ^ delta; }
    }
    uint native_call(uint value, uint delta) { return simd_shuffle_down(value, delta); }
    uint user_call(uint value, uint delta) { return custom::simd_shuffle_down(value, delta); }
    """)
    assert "WaveShuffleDown(value, (uint(delta) & 65535u))" in code
    assert "return simd_shuffle_down(value, delta);" in code


@pytest.mark.parametrize(
    "declarations",
    [
        "namespace custom { uint f(uint value, uint delta) { return simd_shuffle_down(value, delta); } }",
        "namespace custom { namespace inner { uint f(uint value, uint delta) { return simd_shuffle_down(value, delta); } } }",
        "using namespace custom; uint f(uint value, uint delta) { return simd_shuffle_down(value, delta); }",
        "uint f(uint value, uint delta) { using namespace custom; return simd_shuffle_down(value, delta); }",
    ],
)
def test_visible_source_wave_overloads_keep_their_parameter_width(declarations):
    code = convert(
        "namespace custom { uint simd_shuffle_down(uint value, uint delta) { return value ^ delta; } }"
        + declarations
    )
    assert "return simd_shuffle_down(value, delta);" in code
    assert "65535" not in code and "WaveShuffleDown" not in code


@pytest.mark.parametrize(
    "kind", ["bool", "short", "ushort", "int", "uint", "long", "ulong", "half", "float"]
)
def test_wave_lane_scalar_conversions_are_explicit(kind):
    code = convert(
        f"float f(float value, {kind} delta) {{ return metal::simd_shuffle_down(value, delta); }}"
    )
    assert "WaveShuffleDown(value, (uint(delta) & 65535u))" in code


@pytest.mark.parametrize("kind", ["uint2", "float2", "device uint*"])
def test_wave_lane_non_scalar_conversions_are_rejected(kind):
    with pytest.raises(
        MetalSourceOverloadResolutionError, match="scalar conversion to ushort"
    ):
        convert(
            f"float f(float value, {kind} delta) {{ return metal::simd_shuffle_down(value, delta); }}"
        )


def _source(size, kind, mode):
    read = (
        "inputWords[index]" if kind == "uint" else f"as_type<{kind}>(inputWords[index])"
    )
    call = "wrapped_shuffle" if mode == "helper" else "simd_shuffle_down"
    return f"""#include <metal_stdlib>
using namespace metal;
namespace custom {{
uint simd_shuffle_down(uint value, uint offset) {{ return value ^ offset; }}
}}
{kind} wrapped_shuffle({kind} value, uint delta) {{ return metal::simd_shuffle_down(value, delta); }}
kernel void products(device uint* inputWords [[buffer(0)]],
                     device uint* outputWords [[buffer(1)]],
                     uint invocation [[thread_index_in_threadgroup]],
                     uint3 gid [[threadgroup_position_in_grid]]) {{
    uint index = gid.x * {size}u + invocation;
    uint offset = inputWords[{size * 3}u + gid.x];
    uint counter = 0u;
    {kind} value = {read};
    {kind} first = {call}(value, offset + counter++);
    {kind} second = {call}(value, offset + counter++);
    outputWords[index * 5u] = as_type<uint>(first);
    outputWords[index * 5u + 1u] = as_type<uint>(second);
    outputWords[index * 5u + 2u] = counter;
    outputWords[index * 5u + 3u] = inputWords[index];
    outputWords[index * 5u + 4u] = custom::simd_shuffle_down(inputWords[index], offset);
}}
"""


def _reference(size, kind, offset):
    words, wanted = [], []
    for group in range(3):
        values = [
            group * 41 + index - (0 if kind == "uint" else 17) for index in range(size)
        ]
        values = (
            [struct.unpack("<I", struct.pack("<f", value / 4))[0] for value in values]
            if kind == "float"
            else [value & 0xFFFFFFFF for value in values]
        )
        words.extend(values)
        for index, value in enumerate(values):
            results = []
            for counter in (0, 1):
                delta = (offset + counter) % 65536
                results.append(
                    values[index + delta] if index % 32 + delta < 32 else value
                )
            wanted.extend([*results, 2, value, value ^ offset])
    return words + [offset] * 3, wanted


@pytest.mark.parametrize("size", [32, 64])
@pytest.mark.parametrize("kind", ["int", "uint", "float"])
@pytest.mark.parametrize("mode", ["direct", "helper"])
@pytest.mark.parametrize("offset", OFFSETS)
def test_wave_argument_conversion_executes(tmp_path, size, kind, mode, offset):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required source-argument readbacks")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, descriptor, package = _package(
        tmp_path, target, kind, (size, 1, 1), source=_source(size, kind, mode)
    )
    words, wanted = _reference(size, kind, offset)
    guards = [0xBAD00000 + index for index in range(17)]

    def payload(values):
        return {"dtype": "uint32", "shape": [len(values)], "values": values}

    inputs = {
        "inputWords": payload(words),
        "outputWords": payload([0xDEADBEEF] * len(wanted) + guards),
    }
    expected = _bound_values(
        descriptor,
        {"inputWords": payload(words), "outputWords": payload(wanted + guards)},
    )
    names = {
        binding["scalarLayout"].get("memberName", binding["name"]): binding["name"]
        for binding in descriptor["bindings"]
    }
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        {
            name: {key: value for key, value in data.items() if key != "values"}
            for name, data in expected.items()
        },
        {"workgroupCount": [3, 1, 1], "workgroupSize": [size, 1, 1]},
        expected_target=target,
    )
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    _, module = _compile(request.artifact_path.read_text(), target, compiled)
    executor = _executor(target)
    records = {}
    try:
        assert executor.is_available(request).available
        result = executor.run(request)
        assert result.status == "ok", result
        records["generated"] = {"outputs": result.outputs, "details": result.details}
        if target == "metal":
            original = tmp_path / "original"
            original.mkdir()
            original_source, original_module = _compile(source, target, original)
            state, native = _native_request(request)
            actual = executor.runtime_adapter.runtime.dispatch(
                None,
                state,
                replace(
                    native,
                    artifact_path=original_source,
                    module_path=original_module,
                    entry_point="products",
                ),
            )
            records["originalMetal"] = {
                "outputs": actual,
                "moduleSha256": (
                    hashlib.sha256(original_module.read_bytes()).hexdigest()
                ),
            }
        (tmp_path / "evidence.json").write_text(
            json.dumps(
                {
                    "target": target,
                    "kind": kind,
                    "size": size,
                    "mode": mode,
                    "offset": offset,
                    "inputs": inputs,
                    "expected": expected,
                    "descriptor": descriptor,
                    "records": records,
                    "sourceSha256": (
                        hashlib.sha256(request.artifact_path.read_bytes()).hexdigest()
                    ),
                    "validationModuleSha256": (
                        hashlib.sha256(module.read_bytes()).hexdigest()
                    ),
                },
                indent=2,
            )
        )
        for record in records.values():
            _check_outputs(record["outputs"], expected, names, "uint")
    finally:
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if close:
            close()


def test_source_argument_native_gate_is_required_on_every_target():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate source wave argument conversions"
    )
    assert "if:" not in step and "continue-on-error" not in step
    assert f'{REQUIRE_ENV}: "1"' in step and "test_metal_wave_arguments.py" in step
    assert "-n auto" in step and "--basetemp=" in step and "--junitxml=" in step
    for event in ("pull_request", "push"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_metal_wave_arguments.py",
        )
