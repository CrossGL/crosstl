"""Source parameter usage survives specialization without suppressing real warnings."""

import os
import shutil
import sys
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_PARAMETER_USAGE"
TARGETS = ("metal", "directx", "opengl")
STRICT_METAL = ("-std=metal3.2", "-Wall", "-Wextra", "-fno-fast-math")
GUARD = 0x5A1B2C3D
INPUTS = [0, 11, 0xFFFFFFFF, 0x80000000, 0x89ABCDEF]
TAG = "struct Tag { static constant constexpr uint bias = 7u; uint field; };"
OP = "struct Op { static uint apply(uint value) { return value + 7u; } };"
CASES = {
    "receiver": (
        "struct Offset { uint value() const thread { return 7u; } };",
        "Offset offset; uint value = cursor++ + offset.value();",
        None,
    ),
    "static": (
        OP + "uint run(const thread Op& op, uint value) { return op.apply(value); }",
        "Op op; uint value = run(op, cursor++);",
        None,
    ),
    "tag": (
        TAG + "template<typename T> uint run(T tag, uint value) { "
        "return value + decltype(tag)::bias; }",
        "Tag tag; uint value = run<Tag>(tag, cursor++);",
        None,
    ),
    "template-method": (
        TAG + "struct Runner { template<typename T> uint run(T tag, uint value) "
        "const thread { return value + decltype(tag)::bias; } };",
        "Tag tag; Runner runner; uint value = runner.run(tag, cursor++);",
        None,
    ),
    "member-static": (
        OP + "struct Runner { uint run(const thread Op& op, uint value) "
        "const thread { return op.apply(value); } };",
        "Op op; Runner runner; uint value = runner.run(op, cursor++);",
        None,
    ),
    "sizeof": (
        "uint run(uint value) { return uint(sizeof(value)) + 7u; }",
        "uint value = run(cursor++);",
        11,
    ),
    "argument-effect": (
        TAG + "Tag make_tag(thread uint& cursor) { "
        "Tag tag; tag.field = cursor++; return tag; } "
        "template<typename T> uint run(T tag) { return decltype(tag)::bias; }",
        "uint value = run<Tag>(make_tag(cursor));",
        7,
    ),
    "overload": (
        OP + "uint run(const thread Op& op, uint value) { return op.apply(value); } "
        "uint run(float value) { return uint(value) + 9u; }",
        "Op op; uint value = run(op, cursor++);",
        None,
    ),
    "used": (
        "uint run(uint value) { return value + 7u; }",
        "uint value = run(cursor++);",
        None,
    ),
    "discard": (
        "uint run(uint value) { (void)value; return 7u; }",
        "uint value = run(cursor++);",
        7,
    ),
    "discard-effect": (
        "uint run(thread uint& value) { (void)value++; return 7u; }",
        "uint value = run(cursor);",
        7,
    ),
    "binding": (
        "",
        "(void)lid; uint value = cursor++ + 7u;",
        None,
    ),
    "type-alias": (
        TAG + "template<typename T> uint run(T tag, uint value) { "
        "using Selected = decltype(tag); return value + Selected::bias; }",
        "Tag tag; uint value = run<Tag>(tag, cursor++);",
        None,
    ),
}


def _source(case):
    declarations, body, _ = CASES[case]
    extra = ", uint lid [[thread_position_in_threadgroup]]" if case == "binding" else ""
    return f"""#include <metal_stdlib>
using namespace metal;
{declarations}
kernel void parameters(const device uint* inputs [[buffer(0)]],
                       device uint* results [[buffer(1)]],
                       uint tid [[thread_position_in_grid]]{extra}) {{
    uint cursor = inputs[tid];
    {body}
    results[4 + 2 * tid] = value;
    results[5 + 2 * tid] = cursor;
}}
"""


def _request(root, target, case):
    source, descriptor, package = _package(
        root,
        target,
        "uint",
        (1, 1, 1),
        source=_source(case),
        software_subgroups=False,
    )
    output = [GUARD] * 4
    for word in INPUTS:
        constant = CASES[case][2]
        output.extend(
            [
                (word + 7 if constant is None else constant) & 0xFFFFFFFF,
                (word + 1) & 0xFFFFFFFF,
            ]
        )
    output.extend([GUARD] * 4)
    inputs = _bound_values(
        descriptor,
        {
            "inputs": {"dtype": "uint32", "shape": [len(INPUTS)], "values": INPUTS},
            "results": {
                "dtype": "uint32",
                "shape": [len(output)],
                "values": [GUARD] * 4 + [0xDEADBEEF] * (2 * len(INPUTS)) + [GUARD] * 4,
            },
        },
    )
    expected = _bound_values(
        descriptor,
        {
            "results": {"dtype": "uint32", "shape": [len(output)], "values": output},
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        expected,
        {"workgroupCount": [len(INPUTS), 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, expected


def _validate(artifact, work, target):
    return _compile(
        artifact.read_text(), target, work, metal_compile_flags=STRICT_METAL
    )[1]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("target", TARGETS)
def test_parameter_usage_project_compile(tmp_path, case, target):
    _, request, _ = _request(tmp_path, target, case)
    generated = request.artifact_path.read_text()
    if target == "metal":
        assert ("__attribute__((unused))" in generated) == (
            case not in {"used", "discard-effect"}
        )
    else:
        assert "maybe_unused" not in generated
    _validate(request.artifact_path, tmp_path, target)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("target", TARGETS)
def test_parameter_usage_saved_intermediate(tmp_path, case, target):
    source = tmp_path / "source.metal"
    source.write_text(_source(case))
    intermediate = tmp_path / "source.cgl"
    code = translate(str(source), backend="crossgl", format_output=False)
    assert ("@maybe_unused" in code) == (case not in {"used", "discard-effect"})
    intermediate.write_text(code)
    generated = translate(str(intermediate), backend=target, format_output=False)
    _compile(generated, target, tmp_path, metal_compile_flags=STRICT_METAL)


@pytest.mark.parametrize(
    "body",
    [
        "return 7u;",
        "/* value */ return 7u;",
        "struct Box { uint value; }; Box box; box.value = 7u; return box.value;",
        "{ uint value = 7u; return value; }",
        "{ Tag value; return decltype(value)::bias; }",
    ],
)
def test_genuine_unused_parameter_is_not_annotated(tmp_path, body):
    source = tmp_path / "source.metal"
    source.write_text(TAG + f"uint run(uint value) {{ {body} }}")
    intermediate = translate(str(source), backend="crossgl", format_output=False)
    assert "@maybe_unused" not in intermediate
    if sys.platform == "darwin" and shutil.which("xcrun"):
        generated = translate(str(source), backend="metal", format_output=False)
        with pytest.raises(AssertionError, match="unused parameter"):
            _compile(generated, "metal", tmp_path, metal_compile_flags=STRICT_METAL)


def test_preserved_usage_keeps_default_argument():
    preprocessor = MetalPreprocessor()
    parameters = preprocessor._preserve_parameter_usage(
        "uint tag = 7u, uint value = 3u",
        "return decltype(tag)::bias + value;",
        "return 7u + value;",
    )
    assert parameters == "uint tag [[maybe_unused]] = 7u, uint value = 3u"
    from crosstl.backend.Metal.MetalLexer import MetalLexer
    from crosstl.backend.Metal.MetalParser import MetalParser

    ast = MetalParser(
        MetalLexer(f"uint run({parameters}) {{ return value; }}").tokenize()
    ).parse()
    parameter = ast.functions[0].params[0]
    assert any(attribute.name == "maybe_unused" for attribute in parameter.attributes)
    assert parameter.default_value is not None


@pytest.mark.parametrize("case", CASES)
def test_parameter_usage_execute(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required parameter usage execution")
    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="parameters",
        metal_compile_flags=STRICT_METAL,
        validate=_validate,
    )


def test_parameter_usage_execution_is_required():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_parameter_usage.py" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step
