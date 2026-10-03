"""Boolean votes preserve logical subgroup boundaries and operand evaluation."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_abi_descriptor,
    build_native_loader_dispatch_request,
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    translate_project,
)
from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import (
    DirectXSoftwareSubgroupError,
    HLSLCodeGen,
)
from crosstl.translator.codegen.GLSL_codegen import (
    GLSLCodeGen,
    OpenGLSoftwareSubgroupError,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_metal_native_runtime import _native_request
from tests.test_translator.test_native_loader_dispatch_integration import _executor

REQUIRE_ENV = "CROSTL_REQUIRE_SOFTWARE_SUBGROUP_VOTES"


def _codegen(target, software=True):
    options = {"software_subgroup_width": 32} if software else {}
    if target == "directx":
        return HLSLCodeGen(relative_wave_shuffle_out_of_range="self", **options)
    return GLSLCodeGen(**options)


def _canonical(body, helpers=""):
    return f"""shader Votes {{
        RWStructuredBuffer<uint> results @register(u0);
        {helpers}
        compute {{
            @numthreads(32, 4, 1)
            void main(uint invocation @gl_LocalInvocationIndex) {{
                {body}
            }}
        }}
    }}"""


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("operation", ["AllTrue", "AnyTrue"])
def test_software_votes_compile_with_single_operand_evaluation(
    tmp_path, target, operation
):
    source = _canonical(
        "uint counter = invocation; bool result = vote(counter++ != 0u); results[invocation] = uint(result);",
        f"bool vote(bool value) {{ return WaveActive{operation}(value); }}",
    )
    generator = _codegen(target)
    generated = generator.generate_stage(parse(source), "compute")
    assert generated.count("counter++") == 1
    assert "WaveActive" not in generated and "subgroupAll(" not in generated
    assert "subgroupAny(" not in generated
    assert "shared bool " in generated
    _compile(generated, target, tmp_path)
    assert generator.generate_stage(parse(source), "compute") == generated


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("operation", ["AllTrue", "AnyTrue"])
@pytest.mark.parametrize(
    "body",
    [
        "if (invocation < 16u) { results[invocation] = uint(VOTE); }",
        "if (invocation == 0u) { return; } results[invocation] = uint(VOTE);",
        "results[invocation] = invocation == 0u ? uint(VOTE) : 0u;",
        "bool selected = invocation == 0u && VOTE; results[invocation] = uint(selected);",
        "for (uint i = 0u; i < invocation; ++i) { results[invocation] = uint(VOTE); }",
    ],
)
def test_software_votes_reject_divergent_barriers(target, operation, body):
    expression = f"WaveActive{operation}(invocation != 0u)"
    error = (
        DirectXSoftwareSubgroupError
        if target == "directx"
        else OpenGLSoftwareSubgroupError
    )
    with pytest.raises(error):
        _codegen(target).generate_stage(
            parse(_canonical(body.replace("VOTE", expression))), "compute"
        )


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("value_type", ["uint", "float", "bvec2"])
def test_software_votes_reject_non_scalar_boolean_payloads(target, value_type):
    source = _canonical(
        f"{value_type} value = {value_type}(1); results[invocation] = uint(WaveActiveAnyTrue(value));"
    )
    error = ValueError if target == "directx" else OpenGLSoftwareSubgroupError
    with pytest.raises(error, match="bool"):
        _codegen(target).generate_stage(parse(source), "compute")


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("operation", ["Sum", "Min", "Max"])
def test_boolean_votes_do_not_enable_boolean_arithmetic(target, operation):
    source = _canonical(
        f"bool value = WaveActive{operation}(true); results[invocation] = uint(value);"
    )
    with pytest.raises(ValueError, match="bool"):
        _codegen(target).generate_stage(parse(source), "compute")


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_native_votes_remain_available(tmp_path, target):
    source = _canonical(
        "bool a = WaveActiveAllTrue(invocation != 0u); "
        "bool b = WaveActiveAnyTrue(invocation != 0u); "
        "results[invocation] = uint(a) + uint(b);"
    )
    generated = _codegen(target, software=False).generate_stage(
        parse(source), "compute"
    )
    assert "software_subgroup" not in generated and "SoftwareSubgroup" not in generated
    assert (
        "WaveActiveAllTrue(" if target == "directx" else "subgroupAll("
    ) in generated
    if target == "directx":
        _compile(generated, target, tmp_path)
    elif shutil.which("glslangValidator"):
        source_path = tmp_path / "native.comp"
        source_path.write_text(generated, encoding="utf-8")
        result = subprocess.run(
            [
                "glslangValidator",
                "--target-env",
                "opengl",
                "--target-env",
                "spirv1.3",
                "-S",
                "comp",
                str(source_path),
                "-o",
                str(tmp_path / "native.spv"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "operation", ["WaveActiveSum", "WaveActiveAllTrue", "WaveActiveAnyTrue"]
)
@pytest.mark.parametrize("nested", [False, True])
def test_opengl_collective_helpers_reject_divergent_early_returns(operation, nested):
    value_type = "float" if operation == "WaveActiveSum" else "bool"
    helper = f"""{value_type} reduce_value({value_type} value) {{
        if (value == {value_type}(0)) {{ return value; }}
        return {operation}(value);
    }}"""
    callee = "reduce_value"
    if nested:
        helper += f"""{value_type} outer({value_type} value) {{
            {value_type} initial = {operation}(value);
            return reduce_value(initial);
        }}"""
        callee = "outer"
    source = _canonical(
        f"{value_type} value = {callee}({value_type}(invocation)); results[invocation] = uint(value);",
        helper,
    )
    with pytest.raises(OpenGLSoftwareSubgroupError, match="divergent"):
        _codegen("opengl").generate_stage(parse(source), "compute")


def test_opengl_votes_allow_uniform_exit_and_final_lane_dependent_return(tmp_path):
    source = _canonical("""
        if (gl_WorkGroupID.x == 0u) { return; }
        bool result = WaveActiveAnyTrue(invocation != 0u);
        results[invocation] = uint(result);
        if (invocation == 0u) { return; }
    """)
    generated = _codegen("opengl").generate_stage(parse(source), "compute")
    _compile(generated, "opengl", tmp_path)


def test_software_votes_are_required_in_native_workflow():
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/mlx-portable-host.yml").read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate software subgroup votes"
    )
    assert "test_software_subgroup_votes.py" in step
    assert "test_opengl_subgroup_wrappers.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "-n auto" in step and "continue-on-error" not in step
    assert "--timeout-seconds" in step and "--junitxml" in step
    assert "if:" not in step
    for event in ("pull_request", "push"):
        assert (
            "tests/test_translator/test_software_subgroup_votes.py"
            in ci_coverage.workflow_event_path_filters(workflow, event)
        )
        assert (
            "tests/test_translator/test_opengl_subgroup_wrappers.py"
            in ci_coverage.workflow_event_path_filters(workflow, event)
        )


def _metal_source(size, wrapper_depth=0):
    helpers = "bool some_value(bool value) { return simd_any(value); }\n"
    callee = "some_value"
    for depth in range(wrapper_depth):
        name = f"wrapped_vote_{depth}"
        helpers += f"bool {name}(bool value) {{ return {callee}(value); }}\n"
        callee = name
    some = f"bool some = {callee}(predicate);"
    if wrapper_depth:
        some = f"""bool some = false;
    if (group.x % 2u == 0u) {{
        some = {callee}(predicate);
    }} else {{
        some = {callee}(!predicate);
    }}"""
    return f"""#include <metal_stdlib>
using namespace metal;
{helpers}
kernel void votes(device uint* values [[buffer(0)]],
                  device uint* results [[buffer(1)]],
                  uint invocation [[thread_index_in_threadgroup]],
                  uint3 group [[threadgroup_position_in_grid]]) {{
    uint index = group.x * {size}u + invocation;
    bool predicate = values[index] != 0u;
    uint counter = 0u;
    bool every = simd_all((counter++ == 0u) && predicate);
    {some}
    bool none = simd_all(!predicate);
    bool not_every = simd_any(!predicate);
    bool repeated = simd_all(some);
    results[index * 6u] = uint(every);
    results[index * 6u + 1u] = uint(some);
    results[index * 6u + 2u] = uint(none);
    results[index * 6u + 3u] = uint(not_every);
    results[index * 6u + 4u] = uint(repeated);
    results[index * 6u + 5u] = counter;
}}
"""


def _package(root, target, shape, wrapper_depth=0):
    source = _metal_source(shape[0] * shape[1] * shape[2], wrapper_depth)
    (root / "votes.metal").write_text(source, encoding="utf-8")
    options = {}
    if target != "metal":
        target_options = {"software_subgroup_width": 32}
        if target == "directx":
            target_options["relative_wave_shuffle_out_of_range"] = "self"
        options = {"metal": {"target_options": {target: target_options}}}
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("votes.metal",),
            targets=(target,),
            output_dir="out",
            workgroup_size=shape,
            source_options=options,
        ),
        format_output=False,
    )
    report.write_json(root / "report.json")
    assert report.to_json()["summary"]["failedCount"] == 0, report.to_json()
    manifest = build_runtime_artifact_manifest(root / "report.json")
    assert manifest["success"], manifest
    (root / "artifacts.json").write_text(json.dumps(manifest), encoding="utf-8")
    package = root / "package"
    assert build_runtime_package(root / "artifacts.json", package)["success"]
    loader = build_runtime_loader_manifest(package / "runtime-package.json")
    assert loader["success"], loader
    assert len(loader["loadUnits"]) == 1
    descriptor = build_native_loader_abi_descriptor(
        loader, load_unit_id=loader["loadUnits"][0]["id"]
    )
    return source, descriptor, package


def _values(count, workgroup_size=None):
    patterns = [
        [0] * 32,
        [7] * 32,
        [0] * 31 + [0x80000000],
        [0] + [0xFFFFFFFF] * 31,
        [9] + [0] * 31,
        [11] * 31 + [0],
        [0, 5] * 16,
        [0xFFFFFFFF, 0] * 16,
    ]
    values, expected = [], []
    for i in range(count // 32):
        lanes = patterns[i % len(patterns)]
        every, some = all(lanes), any(lanes)
        if workgroup_size is not None and (i * 32 // workgroup_size) % 2:
            some = not every
        values.extend(lanes)
        expected.extend(
            [int(every), int(some), int(not any(lanes)), int(not every), int(some), 1]
            * 32
        )
    guards = [0xBAD00000 + i for i in range(17)]

    def payload(items):
        return {"dtype": "uint32", "shape": [len(items)], "values": items}

    inputs = {
        "values": payload(values),
        "results": payload([0xDEADBEEF] * (count * 6) + guards),
    }
    outputs = {"values": payload(values), "results": payload(expected + guards)}
    return inputs, outputs


@pytest.mark.parametrize("target", ["metal", "opengl", "directx"])
def test_metal_votes_translate_through_project_packages(tmp_path, target):
    _, descriptor, package = _package(tmp_path, target, (32, 4, 1))
    inputs, outputs = _values(1024)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        _bound_values(descriptor, outputs),
        {"workgroupCount": [8, 1, 1], "workgroupSize": [32, 4, 1]},
        expected_target=target,
    )
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("shape", [(32, 1, 1), (32, 4, 1), (64, 2, 1)])
@pytest.mark.parametrize("wrapper_depth", [0, 3])
def test_software_votes_execute_on_device(tmp_path, shape, wrapper_depth):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native votes")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, descriptor, package = _package(tmp_path, target, shape, wrapper_depth)
    count = 8 * shape[0] * shape[1] * shape[2]
    inputs, outputs = _values(count, count // 8 if wrapper_depth else None)
    expected = _bound_values(descriptor, outputs)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        expected,
        {"workgroupCount": [8, 1, 1], "workgroupSize": list(shape)},
        expected_target=target,
    )
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    _, module = _compile(request.artifact_path.read_text(), target, compiled)
    assert (
        module.is_file() and module.stat().st_size
    ), "native gate requires the target compiler"
    executor = _executor(target)
    records = {}
    try:
        availability = executor.is_available(request)
        assert availability.available, availability
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
                entry_point="votes",
            )
            actual = executor.runtime_adapter.runtime.dispatch(None, state, native)
            assert actual == expected
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
                    "wrapperDepth": wrapper_depth,
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
            ),
            encoding="utf-8",
        )
        assert result.outputs == expected
    finally:
        close = getattr(executor.runtime_adapter.runtime, "close", None)
        if close:
            close()
