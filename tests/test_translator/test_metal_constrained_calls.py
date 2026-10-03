"""Reachable constrained helper calls resolve before templates are removed."""

import os
import re
import shutil
import sys

import pytest

from crosstl.backend.Metal.preprocessor import (
    MetalPreprocessor,
    MetalTemplateSpecializationError,
)
from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_atomic_load_runtime import GUARD
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_CONSTRAINED_RUNTIME"
CONSTRAINT = (
    "template <typename T, metal::enable_if_t<metal::is_integral_v<T>, bool> = true>"
)


def _source(*, reverse=False, namespace=False, specialization=False):
    declarations = f"""{CONSTRAINT}
T inner(T value) {{ return value + T(1); }}
{CONSTRAINT}
T outer(T value) {{ return inner(value) * T(2); }}
{CONSTRAINT}
T top(T value) {{ T intermediate = outer(value); return inner(intermediate); }}
"""
    prefix = ""
    if namespace:
        declarations = f"""namespace other {{ {CONSTRAINT}
        T inner(T value) {{ return value + T(99); }} }}
        namespace selected {{ {declarations} }}"""
        prefix = "selected::"
    if specialization:
        declarations += "template <> int inner<int>(int value) { return value + 4; }\n"
    calls = [f"results[0] = {prefix}outer(3);", f"results[1] = {prefix}inner(1);"]
    if reverse:
        calls.reverse()
    return f"""#include <metal_stdlib>
using namespace metal;
{declarations}
kernel void compute_values(device int* results [[buffer(0)]]) {{
    {' '.join(calls)}
    results[2] = {prefix}top(3);
    results[3] = {prefix}outer(3);
}}
"""


@pytest.mark.parametrize("reverse", (False, True))
def test_transitive_specializations_are_reused(reverse):
    output = MetalPreprocessor().preprocess(_source(reverse=reverse))
    for name in ("inner", "outer", "top"):
        assert len(re.findall(rf"int {name}_int\(int value\)", output)) == 1
        assert not re.search(rf"\b{name}\(", output)
    assert "return inner_int(value) * int(2);" in output
    assert "return inner_int(intermediate);" in output


def test_nested_calls_keep_declaration_namespace():
    output = MetalPreprocessor().preprocess(_source(namespace=True))
    assert "return inner_int(value) * int(2);" in output
    assert "int inner_int(int value) { return value + int(1); }" in output


def test_nested_calls_exclude_later_competing_declarations():
    source = (
        _source()
        .replace(
            "kernel void compute_values",
            "int inner(int value);\nkernel void compute_values",
        )
        .replace("results[1] = inner(1);", "results[1] = 2;")
    )
    output = MetalPreprocessor().preprocess(source)
    assert "return inner_int(value) * int(2);" in output


def test_nested_calls_reject_visible_competing_declarations():
    source = _source().replace(
        f"{CONSTRAINT}\nT outer", f"int inner(int value);\n{CONSTRAINT}\nT outer"
    )
    with pytest.raises(
        MetalTemplateSpecializationError, match="competing free-function overload"
    ):
        MetalPreprocessor().preprocess(
            source.replace("results[1] = inner(1);", "results[1] = 2;")
        )


def test_nested_calls_reject_unknown_constraints():
    source = (
        _source()
        .replace(
            f"{CONSTRAINT}\nT inner",
            "template <typename T, metal::enable_if_t<unknown_v<T>, bool> = true>\nT inner",
        )
        .replace("results[1] = inner(1);", "results[1] = 2;")
    )
    with pytest.raises(
        MetalTemplateSpecializationError, match="constraint is not recognized"
    ) as error:
        MetalPreprocessor().preprocess(source)
    assert error.value.callee_template == "inner"


def test_nested_specializations_are_bounded():
    with pytest.raises(
        MetalTemplateSpecializationError, match="specialization limit exceeded"
    ) as error:
        MetalPreprocessor(max_template_specializations=2).preprocess(_source())
    assert error.value.limit == 2
    assert error.value.unique_specialization_count == 3
    assert error.value.requested_signature == "top<int>"
    output = MetalPreprocessor(max_template_specializations=3).preprocess(_source())
    assert "return inner_int(value) * int(2);" in output


def test_transitively_discovered_specializations_consume_the_budget():
    source = (
        _source()
        .replace("results[1] = inner(1);", "")
        .replace("results[2] = top(3);", "")
    )
    with pytest.raises(
        MetalTemplateSpecializationError, match="specialization limit exceeded"
    ) as error:
        MetalPreprocessor(max_template_specializations=1).preprocess(source)
    assert error.value.unique_specialization_count == 2
    assert error.value.requested_signature == "inner<int>"


def test_recursive_constrained_specialization_is_diagnosed():
    source = f"""{CONSTRAINT}
    T recurse(T value) {{ return value > T(0) ? recurse(value - T(1)) : value; }}
    kernel void run(device int* results [[buffer(0)]]) {{ results[0] = recurse(3); }}
    """
    with pytest.raises(
        MetalTemplateSpecializationError, match="recursive constrained helper"
    ) as error:
        MetalPreprocessor().preprocess(source)
    assert error.value.callee_template == "recurse"


def _late_source(*, namespace=False, policy=False):
    declarations = f"""template <typename T> struct Box {{ T value; }};
    {CONSTRAINT}
    T outer(const device Box<T>* values, T factor) {{ return inner(values, factor); }}
    {CONSTRAINT}
    T inner(const device Box<T>* values, T factor) {{ return values[0].value * factor; }}
    template <typename T> struct Policy {{
        T apply(const device Box<T>* values, T factor) thread {{ return outer(values, factor); }}
    }};
    """
    prefix = ""
    if namespace:
        declarations = f"namespace selected {{ {declarations} }}"
        prefix = "selected::"
    body = f"results[0] = {prefix}outer(values, 3);"
    if policy:
        body = (
            f"{prefix}Policy<int> operation; results[0] = operation.apply(values, 3);"
        )
    return f"""#include <metal_stdlib>
    using namespace metal;
    {declarations}
    kernel void compute_values(const device {prefix}Box<int>* values [[buffer(0)]],
        device int* results [[buffer(1)]]) {{ {body} }}
    """


@pytest.mark.parametrize("namespace", (False, True))
@pytest.mark.parametrize("policy", (False, True))
def test_dependent_class_lookup_translates_and_executes(tmp_path, namespace, policy):
    native = os.environ.get(REQUIRE_ENV) == "1"
    native_target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[
        sys.platform
    ]
    source = _late_source(namespace=namespace, policy=policy)
    for target in ("metal", "directx", "opengl"):
        root = tmp_path / target
        root.mkdir()
        original, descriptor, package = _package(
            root, target, "int", (1, 1, 1), source=source, software_subgroups=False
        )
        initial = {
            "values": {"dtype": "int32", "shape": [1, 1], "values": [7]},
            "results": {"dtype": "int32", "shape": [4], "values": [GUARD] * 4},
        }
        expected = _bound_values(
            descriptor,
            {"results": {**initial["results"], "values": [21, GUARD, GUARD, GUARD]}},
        )
        request = build_native_loader_dispatch_request(
            descriptor,
            package,
            _bound_values(descriptor, initial),
            expected,
            {"workgroupCount": [1, 1, 1], "workgroupSize": [1, 1, 1]},
            expected_target=target,
        )
        assert not request.execution_plan.diagnostics
        generated = request.artifact_path.read_text()
        assert not re.search(r"\b(?:inner|outer)\(", generated)
        tool = {"metal": "xcrun", "directx": "dxc", "opengl": "glslangValidator"}[
            target
        ]
        if shutil.which(tool):
            _, module = _compile(generated, target, root)
            assert module.is_file() and module.stat().st_size
        if native and target == native_target:
            _execute(
                request,
                expected,
                root,
                original_source=original,
                original_entry="compute_values",
            )


def test_late_primitive_overload_does_not_gain_dependent_visibility():
    source = f"""{CONSTRAINT}
    T outer(T value) {{ return inner(value); }}
    {CONSTRAINT}
    T inner(T value) {{ return value + T(1); }}
    kernel void run(device int* results [[buffer(0)]]) {{ results[0] = outer(3); }}
    """
    output = (
        MetalPreprocessor()._materialize_inferred_constrained_template_function_calls(
            source
        )
    )
    assert "return inner(value);" in output
    assert "inner_int" not in output


def test_late_associated_competing_overload_is_rejected():
    source = _late_source().replace(
        "template <typename T> struct Policy",
        "int inner(const device Box<int>* values, int factor);\n"
        "template <typename T> struct Policy",
    )
    with pytest.raises(
        MetalTemplateSpecializationError, match="competing free-function overload"
    ):
        MetalPreprocessor().preprocess(source)


def test_late_unrelated_namespace_is_not_associated():
    source = (
        _late_source()
        .replace(
            f"{CONSTRAINT}\n    T inner",
            f"namespace unrelated {{ {CONSTRAINT}\n    T inner",
        )
        .replace("values[0].value * factor; }", "values[0].value * factor; } }")
    )
    output = (
        MetalPreprocessor()._materialize_inferred_constrained_template_function_calls(
            source
        )
    )
    assert "return inner(values, factor);" in output
    assert "inner_int" not in output


@pytest.mark.parametrize("qualified", ("::inner", "(inner)"))
def test_qualified_or_parenthesized_call_does_not_gain_dependent_lookup(qualified):
    source = _late_source().replace(
        "return inner(values, factor);", f"return {qualified}(values, factor);"
    )
    output = (
        MetalPreprocessor()._materialize_inferred_constrained_template_function_calls(
            source
        )
    )
    assert f"return {qualified}(values, factor);" in output
    assert "inner_int" not in output


def test_non_dependent_class_call_keeps_definition_context():
    source = _late_source().replace(
        "T outer(const device Box<T>* values, T factor) { return inner(values, factor); }",
        "T outer(const device Box<int>* values, T factor) { return inner(values, 3); }",
    )
    output = (
        MetalPreprocessor()._materialize_inferred_constrained_template_function_calls(
            source
        )
    )
    assert "return inner(values, 3);" in output
    assert "inner_int" not in output


def test_late_declaration_after_instantiation_is_not_visible():
    source = _late_source()
    begin = source.index(f"{CONSTRAINT}\n    T inner")
    end = source.index("template <typename T> struct Policy", begin)
    declaration = source[begin:end]
    source = source[:begin] + source[end:] + declaration
    output = (
        MetalPreprocessor()._materialize_inferred_constrained_template_function_calls(
            source
        )
    )
    assert "return inner(values, factor);" in output
    assert "inner_int" not in output


@pytest.mark.parametrize("case", ("default", "reverse", "namespace", "specialization"))
def test_constrained_call_graph_translates_and_executes(tmp_path, case):
    native = os.environ.get(REQUIRE_ENV) == "1"
    native_target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[
        sys.platform
    ]
    source = _source(**({case: True} if case != "default" else {}))
    values = [14, 5, 18, 14] if case == "specialization" else [8, 2, 9, 8]
    for target in ("metal", "directx", "opengl"):
        root = tmp_path / target
        root.mkdir()
        original, descriptor, package = _package(
            root, target, "int", (1, 1, 1), source=source, software_subgroups=False
        )
        initial = {"dtype": "int32", "shape": [7], "values": [GUARD] * 7}
        expected = _bound_values(
            descriptor, {"results": {**initial, "values": [*values, *([GUARD] * 3)]}}
        )
        request = build_native_loader_dispatch_request(
            descriptor,
            package,
            _bound_values(descriptor, {"results": initial}),
            expected,
            {"workgroupCount": [1, 1, 1], "workgroupSize": [1, 1, 1]},
            expected_target=target,
        )
        assert not request.execution_plan.diagnostics
        generated = request.artifact_path.read_text()
        assert not re.search(r"\b(?:inner|outer|top)\(", generated)
        tool = {"metal": "xcrun", "directx": "dxc", "opengl": "glslangValidator"}[
            target
        ]
        if shutil.which(tool):
            _, module = _compile(generated, target, root)
            assert module.is_file() and module.stat().st_size
        if native and target == native_target:
            _execute(
                request,
                expected,
                root,
                original_source=original,
                original_entry="compute_values",
            )
