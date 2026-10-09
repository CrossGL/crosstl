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
