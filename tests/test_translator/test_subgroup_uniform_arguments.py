"""Collective helpers retain only argument facts proved at every caller."""

import hashlib
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import DirectXSoftwareSubgroupError
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_native_runtime import _native_request
from tests.test_translator.test_native_loader_dispatch_integration import _executor
from tests.test_translator.test_software_subgroup_product import (
    _check_outputs,
    _package,
)
from tests.test_translator.test_software_subgroup_votes import _canonical, _codegen

REQUIRE_ENV = "CROSTL_REQUIRE_SUBGROUP_UNIFORM_ARGUMENTS"


def _source(body, helpers):
    return _canonical(body, helpers).replace(
        "uint invocation @gl_LocalInvocationIndex",
        "uint invocation @gl_LocalInvocationIndex, uint groups @gl_NumSubgroups",
    )


def _helper(body=None, parameters="float value, uint count"):
    body = body or (
        "float result = WaveActiveSum(value); "
        "if (count > 1u) { result = WaveActiveSum(result); } return result;"
    )
    return f"float reduce_twice({parameters}) {{ {body} }}"


def _generate(body, helpers):
    return _codegen("directx").generate_stage(parse(_source(body, helpers)), "compute")


@pytest.mark.parametrize("argument", ["groups", "2u", "min(groups, 2u)"])
@pytest.mark.parametrize("depth", [0, 2])
def test_uniform_helper_arguments_compile(tmp_path, argument, depth):
    helpers = _helper()
    callee = "reduce_twice"
    for index in range(depth):
        name = f"wrapper{index}"
        helpers += f"float {name}(float value, uint count) {{ return {callee}(value, count); }}"
        callee = name
    source = _source(
        f"uint bound = {argument}; float result = {callee}(float(invocation), bound); results[invocation] = asuint(result);",
        helpers,
    )
    generator = _codegen("directx")
    ast = parse(source)
    generated = generator.generate_stage(ast, "compute")
    assert generated == generator.generate_stage(ast, "compute")
    assert "WaveActiveSum" not in generated
    assert "if (count > 1u)" in generated
    _compile(generated, "directx", tmp_path)


def test_uniform_helper_loop_arguments_compile(tmp_path):
    generated = _generate(
        "float result = 0.0; for (uint i = 0u; i < groups; ++i) { result += reduce_twice(float(invocation), i); } results[invocation] = asuint(result);",
        _helper(),
    )
    _compile(generated, "directx", tmp_path)


@pytest.mark.parametrize("second", ["groups + 1u", "1u"])
def test_every_call_can_supply_different_uniform_values(tmp_path, second):
    generated = _generate(
        f"float first = reduce_twice(float(invocation), groups); float second = reduce_twice(first, {second}); results[invocation] = asuint(second);",
        _helper(),
    )
    _compile(generated, "directx", tmp_path)


@pytest.mark.parametrize(
    "body,helpers",
    [
        ("float x = reduce_twice(float(invocation), invocation);", _helper()),
        (
            "float x = reduce_twice(float(invocation), groups); float y = reduce_twice(x, invocation);",
            _helper(),
        ),
        ("groups = invocation; float x = reduce_twice(1.0, groups);", _helper()),
        (
            "uint bound = groups; ++bound; float x = reduce_twice(1.0, bound);",
            _helper(),
        ),
        (
            "uint bound = groups; mutate(bound, invocation); float x = reduce_twice(1.0, bound);",
            "void mutate(inout uint x, uint lane) { x = lane; }" + _helper(),
        ),
        (
            "uint bound = groups; { uint bound = invocation; float x = reduce_twice(1.0, bound); }",
            _helper(),
        ),
        (
            "float x = reduce_twice(float(invocation), groups);",
            _helper(
                "count = uint(value); if (count > 1u) { value = WaveActiveSum(value); } return value;"
            ),
        ),
        (
            "float x = reduce_twice(float(invocation), groups);",
            _helper(
                "if (count > 1u) { value = WaveActiveSum(value); } return value;",
                "float value, inout uint count",
            ),
        ),
        (
            "bool vote = WaveActiveAnyTrue(invocation == 0u); float x = reduce_twice(1.0, uint(vote));",
            _helper(),
        ),
        (
            "float x = reduce_twice(float(invocation), invocation);",
            _helper(parameters="float value, const uint count"),
        ),
        (
            "float x = reduce_twice(float(invocation), invocation);",
            _helper(parameters="float value, uint count @gl_NumSubgroups"),
        ),
        ("if (invocation == 0u) { float x = reduce_twice(1.0, groups); }", _helper()),
        (
            "float x = outer(float(invocation), groups);",
            _helper()
            + "float outer(float value, uint count) { float x = reduce_twice(value, count); return reduce_twice(x, uint(value)); }",
        ),
    ],
)
def test_unproven_helper_arguments_remain_diagnostic(body, helpers):
    with pytest.raises(DirectXSoftwareSubgroupError):
        _generate(body, helpers)


@pytest.mark.parametrize("mutual", [False, True])
def test_recursive_collective_arguments_are_rejected(mutual):
    helpers = _helper("float x = WaveActiveSum(value); return recurse(x, count);")
    helpers += (
        "float recurse(float value, uint count) { return reduce_twice(value, count); }"
        if mutual
        else "float recurse(float value, uint count) { float x = WaveActiveSum(value); return recurse(x, count); }"
    )
    with pytest.raises(DirectXSoftwareSubgroupError) as raised:
        _generate("float x = reduce_twice(float(invocation), groups);", helpers)
    assert raised.value.reason == "helper-call-recursive"


def _native_source(size, depth, mode):
    helpers = """uint reduce_twice(uint value, uint count) {
        uint result = simd_sum(value);
        if (count > 1u) { result = simd_sum(result); }
        return result;
    }"""
    callee = "reduce_twice"
    for index in range(depth):
        name = f"wrapper{index}"
        helpers += (
            f"uint {name}(uint value, uint count) {{ return {callee}(value, count); }}"
        )
        callee = name
    count = "groups" if mode == "groups" else "gid.x % 2u + 1u"
    return f"""#include <metal_stdlib>
using namespace metal;
{helpers}
kernel void products(device uint* inputWords [[buffer(0)]],
                     device uint* outputWords [[buffer(1)]],
                     uint tid [[thread_index_in_threadgroup]],
                     uint3 gid [[threadgroup_position_in_grid]],
                     uint groups [[simdgroups_per_threadgroup]]) {{
    uint index = gid.x * {size}u + tid;
    uint value = inputWords[index];
    uint counter = 0u;
    uint first = {callee}(value + counter++, 1u);
    uint second = {callee}(value + counter++, {count});
    outputWords[index * 3u] = first;
    outputWords[index * 3u + 1u] = second;
    outputWords[index * 3u + 2u] = counter;
}}
"""


@pytest.mark.parametrize("shape", [(32, 1, 1), (32, 4, 1)])
@pytest.mark.parametrize("depth", [0, 2])
@pytest.mark.parametrize("mode", ["groups", "parity"])
def test_uniform_arguments_execute_across_helper_calls(tmp_path, shape, depth, mode):
    if sys.platform not in {"darwin", "win32"}:
        pytest.skip("OpenGL helper argument propagation remains unsupported")
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native helper argument checks")
    target = "metal" if sys.platform == "darwin" else "directx"
    size = math.prod(shape)
    source, descriptor, package = _package(
        tmp_path, target, "uint", shape, source=_native_source(size, depth, mode)
    )
    words = [index % 17 for index in range(512)]
    wanted = []
    for start in range(0, len(words), 32):
        first = sum(words[start : start + 32])
        count = size // 32 if mode == "groups" else (start // size) % 2 + 1
        second = (first + 32) * (32 if count > 1 else 1)
        wanted.extend([first, second, 2] * 32)
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
        {"workgroupCount": [len(words) // size, 1, 1], "workgroupSize": list(shape)},
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
            native = replace(
                native,
                artifact_path=original_source,
                module_path=original_module,
                entry_point="products",
            )
            actual = executor.runtime_adapter.runtime.dispatch(None, state, native)
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
                    "shape": shape,
                    "depth": depth,
                    "mode": mode,
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


def test_uniform_argument_native_gate_is_required_on_windows_and_metal():
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/mlx-portable-host.yml").read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate collective helper arguments"
    )
    assert "if: runner.os != 'Linux'" in step
    assert "test_subgroup_uniform_arguments.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "-n auto" in step and "continue-on-error" not in step
    assert "--timeout-seconds 180" in step
    assert "--basetemp=" in step and "--junitxml=" in step
    for event in ("pull_request", "push"):
        assert (
            "tests/test_translator/test_subgroup_uniform_arguments.py"
            in ci_coverage.workflow_event_path_filters(workflow, event)
        )
