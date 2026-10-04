"""Subgroup-varying early returns retain converged software collectives."""

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
from crosstl.translator.ast import IfNode, ReturnNode
from crosstl.translator.codegen.subgroup_control_flow import (
    converge_subgroup_guarded_returns,
)
from tests.ci_helpers import assert_paths_covered
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_native_runtime import _native_request
from tests.test_translator.test_native_loader_dispatch_integration import _executor
from tests.test_translator.test_software_subgroup_product import (
    _check_outputs,
    _float,
    _package,
    _word,
)
from tests.test_translator.test_software_subgroup_votes import _canonical, _codegen

REQUIRE_ENV = "CROSTL_REQUIRE_SUBGROUP_GUARDED_RETURNS"


def _program(body, parameters="float value", argument="float(invocation)"):
    return _canonical(
        f"float result = guarded({argument}); results[invocation] = asuint(result);",
        f"float guarded({parameters}) {{ {body} }}",
    )


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("operation", ["Min", "Max", "Sum", "Product"])
@pytest.mark.parametrize("form", ["direct", "alias", "inverted", "else"])
def test_guarded_returns_compile_without_changing_source_ast(
    tmp_path, target, operation, form
):
    vote = "WaveActiveAnyTrue(value != value)"
    prefix = ""
    if form == "alias":
        prefix = f"bool bad = {vote}; "
        vote = "bad"
    if form == "inverted":
        vote = "!WaveActiveAllTrue(value == value)"
    tail = f"return WaveActive{operation}(value);"
    if form == "else":
        tail = "else { " + tail + " }"
    source = _program(prefix + f"if ({vote}) {{ return NAN; }} " + tail)
    ast = parse(source)
    generator = _codegen(target)
    original_returns = sum(
        isinstance(node, ReturnNode) for node in generator.walk_ast(ast)
    )
    generated = generator.generate_stage(ast, "compute")
    assert "subgroup_guard" in generated and "subgroup_result" in generated
    assert "WaveActive" not in generated
    assert generator.generate_stage(ast, "compute") == generated
    assert (
        sum(isinstance(node, ReturnNode) for node in generator.walk_ast(ast))
        == original_returns
    )
    assert any(isinstance(node, IfNode) for node in generator.walk_ast(ast))
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize(
    "body",
    [
        "if (value != value) { return NAN; } return WaveActiveMin(value);",
        "if (WaveActiveAnyTrue(value != value) && value > 0.0) { return NAN; } return WaveActiveMin(value);",
        "if (value > 0.0 && WaveActiveAnyTrue(value != value)) { return NAN; } return WaveActiveMin(value);",
        "bool bad = WaveActiveAnyTrue(value != value); bad = value > 0.0; if (bad) { return NAN; } return WaveActiveMin(value);",
        "if (WaveActiveAnyTrue(value++ > 0.0)) { return NAN; } return WaveActiveMin(value);",
        "if (WaveActiveAnyTrue(value != value)) { return value++; } return WaveActiveMin(value);",
        "if (WaveActiveAnyTrue(value != value)) { return NAN; } return WaveActiveMin(value++);",
        "if (WaveActiveAnyTrue(value != value)) { return NAN; } return WaveActiveMin(1.0 / value);",
        "if (WaveActiveAnyTrue(value != value)) { return NAN; } return WaveActiveMin(int(value));",
        "if (WaveActiveAnyTrue(value != value)) { return NAN; } return WaveActiveMin(results[uint(value)]);",
        "if (WaveActiveAnyTrue(value != value)) { results[0] = 1u; return NAN; } return WaveActiveMin(value);",
        "if (WaveActiveAnyTrue(value != value)) { return WaveActiveMax(value); } return WaveActiveMin(value);",
    ],
)
def test_unsafe_guarded_returns_remain_diagnostic(target, body):
    with pytest.raises(ValueError):
        _codegen(target).generate_stage(parse(_program(body)), "compute")


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_guarded_return_names_are_collision_safe(target, tmp_path):
    source = _program(
        "if (WaveActiveAnyTrue(value != value)) { return NAN; } return WaveActiveMin(value);",
        "float value, float __crossgl_subgroup_guard, float __crossgl_subgroup_result",
        "float(invocation), 1.0, 2.0",
    )
    generated = _codegen(target).generate_stage(parse(source), "compute")
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_native_guarded_returns_are_not_rewritten(target):
    ast = parse(
        _program(
            "if (WaveActiveAnyTrue(value != value)) { return NAN; } return WaveActiveMin(value);"
        )
    )
    generated = _codegen(target, software=False).generate_stage(ast, "compute")
    assert "subgroup_guard" not in generated and "subgroup_result" not in generated
    assert "if (" in generated


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_unused_method_receiver_does_not_prevent_convergence(tmp_path, target):
    source = _canonical(
        "Ignored self; self.field = 7; float result = guarded(self, float(invocation)); results[invocation] = asuint(result);",
        "struct Ignored { int field; }; float guarded(inout Ignored self, float value) { if (WaveActiveAnyTrue(value != value)) { return float(NAN); } return WaveActiveMin(value); }",
    )
    generated = _codegen(target).generate_stage(parse(source), "compute")
    assert "subgroup_result" in generated
    _compile(generated, target, tmp_path)


def test_declared_intrinsics_and_unsafe_parameter_types_are_not_speculated():
    generator = _codegen("directx")
    for parameters in ("float* value", "inout float value", "volatile float value"):
        ast = parse(
            _program(
                "if (WaveActiveAnyTrue(value != value)) { return NAN; } return WaveActiveMin(value);",
                parameters,
            )
        )
        assert (
            converge_subgroup_guarded_returns(
                ast, generator.walk_ast, generator.map_operator
            )
            is ast
        )
    source = _program(
        "if (WaveActiveAnyTrue(value != value)) { return NAN; } return WaveActiveMin(value);"
    )
    source = source.replace(
        "compute {", "bool WaveActiveAnyTrue(bool value) { return value; } compute {"
    )
    ast = parse(source)
    assert (
        converge_subgroup_guarded_returns(
            ast, generator.walk_ast, generator.map_operator
        )
        is ast
    )


def test_guarded_return_gate_is_required_on_every_native_target():
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/demo-project-testing.yml").read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate subgroup-guarded returns"
    )
    assert "test_subgroup_guarded_returns.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "-n auto" in step and "continue-on-error" not in step and "if:" not in step
    assert (
        "--timeout-seconds 180" in step
        and "--basetemp=" in step
        and "--junitxml=" in step
    )
    for event in ("pull_request", "push"):
        assert_paths_covered(
            ci_coverage.workflow_event_path_filters(workflow, event),
            "tests/test_translator/test_subgroup_guarded_returns.py",
        )


def _native_source(operation, form, size):
    condition = "simd_any(value != value)"
    prefix = ""
    if form == "alias":
        prefix = "bool good = simd_all(value == value);"
        condition = "!good"
    return f"""#include <metal_stdlib>
using namespace metal;
float guarded(float value) {{
    {prefix}
    if ({condition}) {{ return NAN; }}
    return simd_{operation}(value);
}}
float wrapped(float value) {{ return guarded(value); }}
kernel void products(device uint* inputWords [[buffer(0)]],
                     device uint* outputWords [[buffer(1)]],
                     uint invocation [[thread_index_in_threadgroup]],
                     uint3 group [[threadgroup_position_in_grid]]) {{
    uint index = group.x * {size}u + invocation;
    float value = as_type<float>(inputWords[index]);
    uint counter = 0u;
    float first = wrapped(counter++ == 0u ? value : 123.0f);
    float second = wrapped(counter++ == 1u ? -value : 123.0f);
    outputWords[index * 3u] = as_type<uint>(first);
    outputWords[index * 3u + 1u] = as_type<uint>(second);
    outputWords[index * 3u + 2u] = counter;
}}
"""


def _input_words():
    finite = [lane * 0.125 - 2.0 for lane in range(32)]
    patterns = [
        [math.nan] + finite[1:],
        finite[:-1] + [math.nan],
        [math.nan] * 32,
        finite,
        [math.inf] * 32,
        [-math.inf] * 32,
        [0.0] * 32,
        [-0.0] * 32,
        [-2.0 - lane for lane in range(32)],
        [-math.inf] + finite[1:],
        [math.inf] + finite[1:],
        [-5.0, 5.0] * 16,
        finite[:16] + [math.nan] + finite[17:],
        [(lane + 1) * 2.0**-100 for lane in range(32)],
        finite,
        [float(lane) for lane in range(32)],
    ]
    words = [_word(value) for pattern in patterns for value in pattern]
    words[14 * 32 + 5] = 0xFFC12345
    return words


@pytest.mark.parametrize("shape", [(32, 1, 1), (32, 4, 1)])
@pytest.mark.parametrize("form", ["direct", "alias"])
@pytest.mark.parametrize("operation", ["min", "max"])
def test_guarded_reduction_executes_with_independent_subgroups(
    tmp_path, shape, form, operation
):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native guarded reductions")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, descriptor, package = _package(
        tmp_path,
        target,
        "float",
        shape,
        source=_native_source(operation, form, math.prod(shape)),
    )
    words = _input_words()
    wanted = []
    reduction = min if operation == "min" else max
    for start in range(0, len(words), 32):
        values = [_float(word) for word in words[start : start + 32]]
        if any(math.isnan(value) for value in values):
            first = second = _word(math.nan)
        else:
            first = _word(reduction(values))
            second = _word(reduction([-value for value in values]))
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
        {
            "workgroupCount": [len(words) // math.prod(shape), 1, 1],
            "workgroupSize": list(shape),
        },
        expected_target=target,
    )
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    _, module = _compile(
        request.artifact_path.read_text(),
        target,
        compiled,
        metal_compile_flags=("-fno-fast-math",),
    )
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
            original_source, original_module = _compile(
                source, target, original, metal_compile_flags=("-fno-fast-math",)
            )
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
                    "operation": operation,
                    "form": form,
                    "shape": shape,
                    "inputs": inputs,
                    "expected": expected,
                    "descriptor": descriptor,
                    "records": records,
                    "metalCompileFlags": (
                        ["-fno-fast-math"] if target == "metal" else []
                    ),
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
            _check_outputs(record["outputs"], expected, names, "float")
    finally:
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if close:
            close()
