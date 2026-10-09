"""Keep aggregate context through recursive template-member lowering."""

import json
import os
import sys
from functools import partial
from pathlib import Path

import pytest

from crosstl import translate
from crosstl.backend.Metal.preprocessor import MetalPreprocessor, MetalStructMethodError
from tests.test_translator.test_metal_builtin_ownership import _compile

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_NESTED_MEMBERS"
GUARD = [0x58A5B6C7] * 4


def test_native_member_context_is_required_on_all_platforms():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_metal_nested_member_context.py" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step
    assert "--timeout-seconds 360" in step


def _source(route="explicit", alias="using item_t = Payload;"):
    if route == "operator":
        methods = """
        template <typename T>
        void operator()(thread item_t& item, T delta) { item.update(uint(delta)); }
        void run(thread item_t& item, uint delta) { operator()(item, delta); }
        """
        call = "engine.run(item, values[i]);"
        setup = "Engine engine;"
    else:
        leaf_call = (
            "leaf<T>(item, delta)" if route == "explicit" else "leaf(item, delta)"
        )
        methods = """
        template <typename T>
        static void leaf(thread item_t& item, T delta) { item.update(uint(delta)); }
        template <typename T>
        static void middle(thread item_t& item, T delta) { LEAF_CALL; }
        static void run(thread item_t& item, uint delta) { middle<uint>(item, delta); }
        """.replace("LEAF_CALL", leaf_call)
        call = "Engine::run(item, values[i]);"
        setup = ""
    return (
        """#include <metal_stdlib>
    using namespace metal;
    struct Payload {
        uint value;
        void update(uint delta) { value = value * 7u + delta; }
        uint read() const { return value; }
    };
    struct Engine {
        ALIAS
        METHODS
    };
    kernel void nested_members(const device uint* values [[buffer(0)]],
                              device uint* results [[buffer(1)]],
                              uint i [[thread_position_in_grid]]) {
        Payload item;
        item.value = i;
        SETUP
        CALL
        CALL
        results[i + 4u] = item.read();
    }
    """.replace("ALIAS", alias)
        .replace("METHODS", methods)
        .replace("SETUP", setup)
        .replace("CALL", call)
    )


@pytest.mark.parametrize("route", ("explicit", "inferred", "operator"))
@pytest.mark.parametrize(
    "alias",
    (
        "using item_t = Payload;",
        "typedef Payload item_t;",
        "using base_t = Payload; using item_t = base_t;",
    ),
)
def test_recursive_member_context_resolves_owner_aliases(route, alias):
    output = MetalPreprocessor().preprocess(_source(route, alias))
    assert "Payload__update(item, uint(delta))" in output
    assert "item.update(" not in output
    assert "thread Payload& item" in output
    assert output.count("static inline void Payload__update(") == 1
    if route != "operator":
        assert output.count("static inline void Engine__leaf__uint(") == 1


@pytest.mark.parametrize("route", ("explicit", "inferred", "operator"))
def test_recursive_member_context_retains_readonly_receivers(route):
    source = _source(route).replace("thread item_t&", "const thread item_t&")
    with pytest.raises(MetalStructMethodError) as error:
        MetalPreprocessor().preprocess(source)
    assert "update" in str(error.value)
    assert "concrete struct in its lexical scope" not in str(error.value)


def test_recursive_member_context_keeps_local_receiver_shadowing():
    source = _source().replace(
        "item.update(uint(delta));",
        "{ uint item = 0u; item.update(uint(delta)); }",
    )
    with pytest.raises(
        MetalStructMethodError, match="receiver declaration 'uint item'"
    ):
        MetalPreprocessor().preprocess(source)


def test_recursive_member_context_keeps_local_alias_shadowing():
    source = _source().replace(
        "item.update(uint(delta));",
        "using item_t = uint; item_t local = 0u; local.update(uint(delta));",
    )
    with pytest.raises(MetalStructMethodError, match="local.update"):
        MetalPreprocessor().preprocess(source)


def _translate(tmp_path, route, target):
    path = tmp_path / "nested.metal"
    path.write_text(_source(route))
    return translate(str(path), backend=target, format_output=False)


@pytest.mark.parametrize("route", ("explicit", "inferred", "operator"))
@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_recursive_member_context_compiles(tmp_path, route, target):
    generated = _translate(tmp_path, route, target)
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("route", ("explicit", "inferred", "operator"))
def test_recursive_member_context_executes(tmp_path, monkeypatch, route):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native member lowering")
    from tests.test_translator import test_fused_math as native

    target = {"win32": "directx", "linux": "opengl", "darwin": "metal"}[sys.platform]
    monkeypatch.setattr(
        native, "_compile", partial(_compile, metal_compile_flags=("-fno-fast-math",))
    )
    words = [0, 1, 3, 7, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xDEADBEEF] * 4
    expected = (
        GUARD
        + [(i * 49 + word * 8) & 0xFFFFFFFF for i, word in enumerate(words)]
        + GUARD
    )
    variants = [("generated", _translate(tmp_path, route, target))]
    if target == "metal":
        variants.append(("original", _source(route)))
    records = {}
    for label, source in variants:
        work = tmp_path / label
        work.mkdir()
        actual, evidence = native._dispatch(
            work,
            target,
            source,
            [(word,) for word in words],
            len(expected),
            entry="nested_members" if target == "metal" else None,
            initial_output=GUARD + [0xDEADBEEF] * len(words) + GUARD,
        )
        assert actual == expected
        records[label] = evidence
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "target": target,
                "route": route,
                "inputs": words,
                "expected": expected,
                "records": records,
            },
            indent=2,
        )
    )


FIELD_CASES = (
    "implicit",
    "this",
    "dereference",
    "explicit-template",
    "nested",
    "parameter-shadow",
    "local-shadow",
    "sibling",
    "operator",
    "array",
    "grid",
    "readonly",
    "readonly-array",
)
FIELD_INPUTS = (0, 11, 0xFFFFFFFF, 0x80000000, 0x89ABCDEF)


def _field_source(case):
    field = "Payload item;"
    parameters = "T delta"
    body = "item.update(uint(delta));"
    qualifiers = "thread"
    return_type = "void"
    method_name = "run"
    extra = ""
    setup = "engine.item.value = values[tid];"
    result = "engine.item.value"
    returned = "returned"
    returned_setup = "uint returned = 0u;"
    call = "engine.run(cursor++)"
    if case == "this":
        body = "this->item.update(uint(delta));"
    elif case == "dereference":
        body = "(*this).item.update(uint(delta));"
    elif case == "explicit-template":
        call = "engine.run<uint>(cursor++)"
    elif case == "nested":
        field = "Holder holder;"
        body = "holder.item.update(uint(delta));"
        setup = "engine.holder.item.value = values[tid];"
        result = "engine.holder.item.value"
    elif case == "parameter-shadow":
        parameters = "T item"
        body = "this->item.update(uint(item));"
    elif case == "local-shadow":
        body = """item.update(uint(delta));
        { Payload item; item.value = 17u; item.update(uint(delta));
          seen += item.value; }
        item.update(uint(delta) + 1u);"""
    elif case == "sibling":
        extra = """template<typename U> void step(U delta) thread {
            item.update(uint(delta));
        }"""
        body = "step<T>(delta);"
    elif case == "operator":
        method_name = "operator()"
        call = "engine(cursor++)"
    elif case == "array":
        field = "Payload items[2];"
        body = """uint index = 1u; items[index++].update(uint(delta));
                  seen += index;"""
        setup = "engine.items[0].value = 99u; engine.items[1].value = values[tid];"
        result = "engine.items[1].value"
        returned = "engine.items[0].value"
        returned_setup = ""
    elif case == "readonly":
        qualifiers = "const thread"
        return_type = "uint"
        body = "return item.read() + uint(delta);"
        call = "returned = engine.run(cursor++)"
    elif case == "grid":
        field = "Payload items[2][2];"
        body = """uint row = 1u; uint column = 0u;
                  items[row++][column++].update(uint(delta));
                  seen += row * 10u + column;"""
        setup = (
            "engine.items[0][0].value = 99u; engine.items[1][0].value = values[tid];"
        )
        result = "engine.items[1][0].value"
        returned = "engine.items[0][0].value"
        returned_setup = ""
    elif case == "readonly-array":
        field = "Payload items[2];"
        setup = "engine.items[1].value = values[tid];"
        result = "engine.items[1].value"
        qualifiers = "const thread"
        return_type = "uint"
        body = "return items[1].read() + uint(delta);"
        call = "returned = engine.run(cursor++)"
    return f"""#include <metal_stdlib>
using namespace metal;
struct Payload {{
    uint value;
    void update(uint delta) thread {{ value = value * 7u + delta; }}
    uint read() const thread {{ return value; }}
}};
struct Holder {{ Payload item; }};
struct Engine {{
    {field}
    uint seen;
    {extra}
    template<typename T> {return_type} {method_name}({parameters}) {qualifiers} {{
        {body}
    }}
}};
kernel void field_receivers(const device uint* values [[buffer(0)]],
                            device uint* results [[buffer(1)]],
                            uint tid [[thread_position_in_grid]]) {{
    Engine engine;
    {setup}
    engine.seen = 0u;
    {returned_setup}
    uint cursor = 2u;
    {call};
    {call};
    results[4 + 4 * tid] = {result};
    results[5 + 4 * tid] = engine.seen;
    results[6 + 4 * tid] = {returned};
    results[7 + 4 * tid] = cursor;
}}
"""


def _field_request(root, target, case):
    from crosstl.project import build_native_loader_dispatch_request
    from tests.test_translator.test_boolean_buffer_runtime import _bound_values
    from tests.test_translator.test_software_subgroup_product import _package

    source, descriptor, package = _package(
        root,
        target,
        "uint",
        (1, 1, 1),
        source=_field_source(case),
        software_subgroups=False,
    )
    expected = list(GUARD)
    for initial in FIELD_INPUTS:
        value = initial
        for delta in (2, 3):
            if case not in ("readonly", "readonly-array"):
                value = value * 7 + delta
                if case == "local-shadow":
                    value = value * 7 + delta + 1
        seen = {"local-shadow": 243, "array": 4, "grid": 42}.get(case, 0)
        returned = (
            initial + 3
            if case in ("readonly", "readonly-array")
            else 99 if case in ("array", "grid") else 0
        )
        expected.extend((value & 0xFFFFFFFF, seen, returned & 0xFFFFFFFF, 4))
    expected.extend(GUARD)
    inputs = _bound_values(
        descriptor,
        {
            "values": {
                "dtype": "uint32",
                "shape": [len(FIELD_INPUTS)],
                "values": list(FIELD_INPUTS),
            },
            "results": {
                "dtype": "uint32",
                "shape": [len(expected)],
                "values": GUARD + [0xDEADBEEF] * (4 * len(FIELD_INPUTS)) + GUARD,
            },
        },
    )
    outputs = _bound_values(
        descriptor,
        {
            "results": {
                "dtype": "uint32",
                "shape": [len(expected)],
                "values": expected,
            },
        },
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [len(FIELD_INPUTS), 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


@pytest.mark.parametrize("case", FIELD_CASES)
@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_template_field_receivers_project_compile(tmp_path, target, case):
    _, request, _ = _field_request(tmp_path, target, case)
    _compile(
        request.artifact_path.read_text(),
        target,
        tmp_path,
        metal_compile_flags=("-Wall", "-Wextra"),
    )


@pytest.mark.parametrize("case", FIELD_CASES)
@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_template_field_receivers_saved_intermediate(tmp_path, target, case):
    original = tmp_path / "source.metal"
    original.write_text(_field_source(case))
    intermediate = tmp_path / "source.cgl"
    intermediate.write_text(
        translate(str(original), backend="crossgl", format_output=False)
    )
    generated = translate(str(intermediate), backend=target, format_output=False)
    _compile(generated, target, tmp_path, metal_compile_flags=("-Wall", "-Wextra"))


@pytest.mark.parametrize("case", FIELD_CASES)
def test_template_field_receivers_execute(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native member lowering")
    from tests.test_translator.test_loop_updates import _execute

    target = {"darwin": "metal", "win32": "directx", "linux": "opengl"}[sys.platform]
    source, request, expected = _field_request(tmp_path, target, case)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry="field_receivers",
        metal_compile_flags=("-Wall", "-Wextra"),
    )


@pytest.mark.parametrize(
    "body",
    (
        "missing.update(uint(delta));",
        "uint item = uint(delta); item.update(1u);",
    ),
)
def test_template_field_receivers_reject_unresolved_receiver(body):
    source = _field_source("implicit").replace("item.update(uint(delta));", body)
    with pytest.raises(MetalStructMethodError) as error:
        MetalPreprocessor().preprocess(source)
    assert error.value.reason == "concrete-receiver-type-unresolved"
    assert error.value.missing_capabilities == ("metal.member-overload-resolution",)


@pytest.mark.parametrize("case", ("implicit", "nested", "array", "grid"))
def test_template_field_receivers_reject_mutation_through_const_owner(case):
    source = _field_source(case).replace(
        "void run(T delta) thread",
        "void run(T delta) const thread",
    )
    with pytest.raises(MetalStructMethodError) as error:
        MetalPreprocessor().preprocess(source)
    assert error.value.method_name == "update"
