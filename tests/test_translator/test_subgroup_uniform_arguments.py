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
from crosstl.translator.codegen.GLSL_codegen import (
    OpenGLPrivatePointerParameterError,
    OpenGLSoftwareSubgroupError,
)
from tests.ci_helpers import assert_paths_covered
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


@pytest.fixture(params=["directx", "opengl"])
def target(request):
    return request.param


def _generate(body, helpers, target):
    return _codegen(target).generate_stage(parse(_source(body, helpers)), "compute")


@pytest.mark.parametrize("argument", ["groups", "2u", "min(groups, 2u)"])
@pytest.mark.parametrize("depth", [0, 2])
def test_uniform_helper_arguments_compile(tmp_path, argument, depth, target):
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
    generator = _codegen(target)
    ast = parse(source)
    generated = generator.generate_stage(ast, "compute")
    assert generated == generator.generate_stage(ast, "compute")
    assert "WaveActiveSum" not in generated
    assert "count > 1u" in generated
    _compile(generated, target, tmp_path)


def test_uniform_helper_loop_arguments_compile(tmp_path, target):
    generated = _generate(
        "float result = 0.0; for (uint i = 0u; i < groups; ++i) { result += reduce_twice(float(invocation), i); } results[invocation] = asuint(result);",
        _helper(),
        target,
    )
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ["opengl", "directx"])
@pytest.mark.parametrize("unused_helper", [False, True])
def test_direct_collective_loop_branches_compile(tmp_path, target, unused_helper):
    helpers = (
        "uint unused_collective(uint value) { return WaveActiveSum(value); }"
        if unused_helper
        else ""
    )
    body = """uint result = 0u;
        for (uint i = 0u; i < groups; ++i) {
            if (i % 2u == 0u) { result += WaveActiveSum(invocation); }
            else {
                for (uint j = 1u; j < 4u; ++j) {
                    result += WaveShuffleDown(invocation, j);
                }
            }
        }
        results[invocation] = result;"""
    generated = _generate(body, helpers, target)
    assert "WaveActiveSum" not in generated and "WaveShuffleDown" not in generated
    assert "if (" in generated
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ["opengl", "directx"])
@pytest.mark.parametrize(
    "prefix,condition,helpers",
    [
        ("", "invocation == 0u", ""),
        ("uint count = groups; count = invocation;", "count > 0u", ""),
        (
            "uint count = groups; uint& alias = count; alias = invocation;",
            "count > 0u",
            "",
        ),
        ("uint count = groups; unknown(count);", "count > 0u", ""),
        (
            "uint count = groups; mutate(count, invocation);",
            "count > 0u",
            "void mutate(inout uint value, uint lane) { value = lane; }",
        ),
        ("uint i = invocation;", "i > 0u", ""),
        ("if (invocation == 0u) { return; }", "i > 0u", ""),
    ],
)
def test_direct_collective_branches_reject_unproven_control(
    target, prefix, condition, helpers
):
    error = (
        (OpenGLSoftwareSubgroupError, OpenGLPrivatePointerParameterError)
        if target == "opengl"
        else DirectXSoftwareSubgroupError
    )
    with pytest.raises(error):
        _generate(
            "for (uint i = 0u; i < groups; ++i) { "
            + prefix
            + f" if ({condition}) {{ results[invocation] = WaveShuffleDown(invocation, 1u); }} }}",
            helpers,
            target,
        )


@pytest.mark.parametrize(
    "kind,arguments",
    [
        ("uint", "bound"),
        ("int16", "bound"),
        ("uint8", "bound"),
        ("float16", "bound"),
        ("double", "bound"),
        ("vec2", "bound"),
        ("ivec3", "bound, invocation, 1"),
        ("uvec4", "uvec2(bound, invocation), 0u, 1u"),
        ("i16vec2", "bound, invocation"),
        ("u8vec3", "bound, invocation, 1u"),
        ("f16vec4", "bound"),
        ("i64vec2", "bound, invocation"),
        ("u64vec4", "bound"),
        ("dvec3", "bound, invocation, 1.0"),
        ("bvec2", "bound > 0u, invocation > 0u"),
        ("float2", "bound, invocation"),
        ("int2", "bound, invocation"),
        ("uint2", "uint(int16(bound)), invocation"),
    ],
)
def test_opengl_constructor_arguments_preserve_uniform_bounds(
    tmp_path, kind, arguments
):
    generator = _codegen("opengl")
    member = ".x" if generator.glsl_value_type_info(kind)["width"] > 1 else ""
    source = _source(
        f"uint bound = groups; {kind} coordinate = {kind}({arguments}); "
        "float result = 0.0; for (uint i = 0u; i < bound; ++i) { "
        "result = reduce_twice(float(invocation), i); } "
        f"results[invocation] = asuint(result + float(coordinate{member}));",
        _helper(),
    )
    ast = parse(source)
    generated = generator.generate_stage(ast, "compute")
    assert generated == generator.generate_stage(ast, "compute")
    _compile(generated, "opengl", tmp_path)


@pytest.mark.parametrize(
    "body,helpers",
    [
        ("uvec2 coordinate = uvec2(bound++, invocation);", ""),
        ("uvec2 coordinate = uvec2(++bound, invocation);", ""),
        (
            "uvec2 coordinate = uvec2(mutate(bound, invocation), invocation);",
            "uint mutate(inout uint x, uint lane) { x = lane; return x; }",
        ),
        (
            "uint coordinate = uvec2(bound, invocation);",
            "uint uvec2(inout uint x, uint lane) { x = lane; return x; }",
        ),
        (
            "uint coordinate = uint2(bound, invocation);",
            "uint uint2(inout uint x, uint lane) { x = lane; return x; }",
        ),
        (
            "uint& alias = bound; uvec2 coordinate = uvec2(alias, invocation);",
            "",
        ),
        (
            "uvec2 coordinate = uvec2(bound, invocation); bound = coordinate.y;",
            "",
        ),
        (
            "uvec2 coordinate = uvec2(bound, invocation); if (coordinate.y > 0u) { return; }",
            "",
        ),
        (
            "uvec2 coordinate = uvec2(bound, invocation); if (coordinate.y > 0u) { float x = reduce_twice(1.0, bound); }",
            "",
        ),
    ],
)
def test_opengl_constructors_do_not_hide_mutation_or_divergence(body, helpers):
    with pytest.raises(OpenGLSoftwareSubgroupError):
        _generate(
            "uint bound = groups; "
            + body
            + " for (uint i = 0u; i < bound; ++i) { float result = reduce_twice(1.0, i); }",
            helpers + _helper(),
            "opengl",
        )


@pytest.mark.parametrize(
    "body",
    [
        "{ result = reduce_twice(float(invocation), groups); }",
        "{ { result = reduce_twice(float(invocation), groups); } }",
        "if (groups > 0u) { { result = reduce_twice(float(invocation), groups); } }",
        "{ for (uint i = 0u; i < groups; ++i) { { result = reduce_twice(float(invocation), i); } } }",
        "{ uint bound = groups; { result = reduce_twice(float(invocation), bound); } }",
    ],
)
def test_uniform_lexical_blocks_compile(tmp_path, body, target):
    generated = _generate(
        "float result = 0.0; " + body + " results[invocation] = asuint(result);",
        _helper(),
        target,
    )
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize(
    "body",
    [
        "if (invocation == 0u) { { result = reduce_twice(1.0, groups); } }",
        "{ if (invocation == 0u) { result = reduce_twice(1.0, groups); } }",
        "for (uint i = 0u; i < invocation; ++i) { { result = reduce_twice(1.0, groups); } }",
        "{ if (invocation == 0u) { return; } } { result = reduce_twice(1.0, groups); }",
        "{ uint bound = groups; { uint bound = invocation; result = reduce_twice(1.0, bound); } }",
        "{ uint bound = groups; mutate(bound, invocation); result = reduce_twice(1.0, bound); }",
        "{ uint bound = groups; uint& alias = bound; alias = invocation; result = reduce_twice(1.0, bound); }",
        "{ result = invocation > 0u ? reduce_twice(1.0, groups) : 0.0; }",
        "{ bool condition = invocation > 0u && reduce_twice(1.0, groups) > 0.0; }",
    ],
)
def test_lexical_blocks_do_not_hide_divergence(body, target):
    error = (
        DirectXSoftwareSubgroupError
        if target == "directx"
        else OpenGLSoftwareSubgroupError
    )
    with pytest.raises(error):
        _generate(
            "float result = 0.0; " + body,
            "void mutate(inout uint x, uint lane) { x = lane; }" + _helper(),
            target,
        )


@pytest.mark.parametrize("second", ["groups + 1u", "1u"])
def test_every_call_can_supply_different_uniform_values(tmp_path, second, target):
    generated = _generate(
        f"float first = reduce_twice(float(invocation), groups); float second = reduce_twice(first, {second}); results[invocation] = asuint(second);",
        _helper(),
        target,
    )
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize(
    "body",
    [
        "if (count > 1u) { return reduce_twice(value, count); } return reduce_twice(value, 1u);",
        "for (uint i = 0u; i < count; ++i) { value = reduce_twice(value, i); } return value;",
        "if (count == 0u) { return value; } return reduce_twice(value, count);",
    ],
)
def test_uniform_wrapper_branches_loops_and_exits_compile(tmp_path, body, target):
    generated = _generate(
        "float result = outer(float(invocation), groups); results[invocation] = asuint(result);",
        _helper() + f"float outer(float value, uint count) {{ {body} }}",
        target,
    )
    _compile(generated, target, tmp_path)


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
        (
            "uint bound = groups; unknown(bound); float x = reduce_twice(1.0, bound);",
            _helper(),
        ),
        (
            "uint bound = groups; uint& alias = bound; float x = reduce_twice(1.0, bound);",
            _helper(),
        ),
        (
            "float x = outer(float(invocation), groups);",
            _helper()
            + "float outer(float value, uint count) { return count > 0u ? reduce_twice(value, count) : value; }",
        ),
        (
            "float x = outer(float(invocation), groups);",
            _helper()
            + "float outer(float value, uint count) { return count > 0u && reduce_twice(value, count) > 0.0 ? value : 0.0; }",
        ),
        (
            "float x = reduce_twice(float(invocation), min(groups, 2u));",
            "uint min(uint x, uint y) { return gl_LocalInvocationIndex; }" + _helper(),
        ),
    ],
)
def test_unproven_helper_arguments_remain_diagnostic(body, helpers, target):
    error = (
        DirectXSoftwareSubgroupError
        if target == "directx"
        else OpenGLSoftwareSubgroupError
    )
    with pytest.raises(error):
        _generate(body, helpers, target)


@pytest.mark.parametrize("mutual", [False, True])
def test_recursive_collective_arguments_are_rejected(mutual, target):
    helpers = _helper("float x = WaveActiveSum(value); return recurse(x, count);")
    helpers += (
        "float recurse(float value, uint count) { return reduce_twice(value, count); }"
        if mutual
        else "float recurse(float value, uint count) { float x = WaveActiveSum(value); return recurse(x, count); }"
    )
    error = (
        DirectXSoftwareSubgroupError
        if target == "directx"
        else OpenGLSoftwareSubgroupError
    )
    with pytest.raises(error) as raised:
        _generate("float x = reduce_twice(float(invocation), groups);", helpers, target)
    assert raised.value.reason == "helper-call-recursive"


@pytest.mark.parametrize(
    "body",
    [
        "mutate(i, invocation);",
        "uint& alias = i; alias = invocation;",
        "uint i = invocation;",
        "mutate(groups, invocation);",
    ],
)
def test_opengl_helper_loops_reject_mutable_controls(body):
    with pytest.raises(OpenGLSoftwareSubgroupError):
        _generate(
            "for (uint i = 0u; i < groups; ++i) { "
            + body
            + " float value = reduce_twice(1.0, i); }",
            "void mutate(inout uint value, uint lane) { value = lane; }" + _helper(),
            "opengl",
        )


def _native_source(size, depth, mode):
    helpers = """uint reduce_twice(uint value, uint count) {
        uint result = simd_sum(value);
        if (count > 1u) { result = simd_sum(result); }
        return result;
    }"""
    if mode in {"loop", "block-loop", "vector-loop"}:
        helpers = """uint reduce_twice(uint value, uint count) {
            for (uint i = 0u; i < count; ++i) { value = simd_sum(value); }
            return value;
        }"""
    callee = "reduce_twice"
    for index in range(depth):
        name = f"wrapper{index}"
        helpers += (
            f"uint {name}(uint value, uint count) {{ "
            f"if (count > 0u) {{ return {callee}(value, count); }} return value; }}"
        )
        callee = name
    count = "gid.x % 2u + 1u" if mode == "parity" else "groups"
    setup = ""
    first_output = "first"
    if mode in {"vector", "vector-loop"}:
        setup = "uint bound = groups; uint2 coordinate = uint2(bound, tid);"
        count = "bound"
        first_output = "first + coordinate.x + coordinate.y"
    calls = (
        f"uint first = {callee}(value + counter++, 1u);\n"
        f"uint second = {callee}(value + counter++, {count});"
    )
    if mode == "direct":
        helpers = ""
        calls = """uint first = simd_sum(value + counter++);
        uint second = value + counter++;
        for (uint i = 0u; i < groups; ++i) {
            if (i % 2u == 0u) { second = simd_sum(second); }
            else { second = simd_sum(second + 1u); }
        }"""
    if mode in {"block", "block-loop"}:
        calls = (
            "uint first = 0u; uint second = 0u;\n"
            f"{{ first = {callee}(value + counter++, 1u);\n"
            f"  {{ second = {callee}(value + counter++, {count}); }} }}"
        )
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
    {setup}
    {calls}
    outputWords[index * 3u] = {first_output};
    outputWords[index * 3u + 1u] = second;
    outputWords[index * 3u + 2u] = counter;
}}
"""


@pytest.mark.parametrize("shape", [(32, 1, 1), (64, 1, 1), (32, 4, 1)])
@pytest.mark.parametrize(
    "depth,mode",
    [
        (depth, mode)
        for depth in (0, 2)
        for mode in (
            "groups",
            "parity",
            "loop",
            "block",
            "block-loop",
            "vector",
            "vector-loop",
        )
    ]
    + [(0, "direct")],
)
def test_uniform_arguments_execute_across_helper_calls(tmp_path, shape, depth, mode):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native helper argument checks")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    size = math.prod(shape)
    source, descriptor, package = _package(
        tmp_path, target, "uint", shape, source=_native_source(size, depth, mode)
    )
    words = [index % 17 for index in range(512)]
    wanted = []
    for start in range(0, len(words), 32):
        first = sum(words[start : start + 32])
        count = (start // size) % 2 + 1 if mode == "parity" else size // 32
        repeats = (
            count - 1
            if mode in {"loop", "block-loop", "vector-loop"}
            else int(count > 1)
        )
        second = (first + 32) * 32**repeats
        if mode == "direct":
            second = first + 32
            for iteration in range(1, count):
                second = (second + (iteration % 2)) * 32
        for lane in range(start, start + 32):
            value = (
                first + size // 32 + lane % size
                if mode in {"vector", "vector-loop"}
                else first
            )
            wanted.extend([value, second, 2])
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


def test_uniform_argument_native_gate_is_required_on_every_target():
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate collective helper arguments"
    )
    assert "if:" not in step
    assert "test_subgroup_uniform_arguments.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "EGL_PLATFORM: surfaceless" in step
    assert 'LIBGL_ALWAYS_SOFTWARE: "1"' in step
    assert "PYOPENGL_PLATFORM: egl" in step
    assert "-n auto" in step and "continue-on-error" not in step
    assert "--timeout-seconds 360" in step
    assert "--basetemp=" in step and "--junitxml=" in step
    for event in ("pull_request", "push"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_subgroup_uniform_arguments.py",
        )
