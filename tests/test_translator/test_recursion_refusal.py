"""Targets that cannot execute recursion refuse a recursive call graph with a typed diagnostic (defect S30)."""

import textwrap

import pytest

import crosstl
from crosstl.project import translate_project
from crosstl.translator.codegen.call_graph import (
    RECURSION_FREE_TARGETS,
    CallGraph,
    RecursionUnsupportedError,
    validate_recursion_free,
)
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

TARGETS = sorted(RECURSION_FREE_TARGETS)

CUDA_FIB = """__device__ int fib(int n) {
    return n < 2 ? n : fib(n - 1) + fib(n - 2);
}
__global__ void kernel(const int *inp, int *outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i] = fib(inp[i]);
}
"""

GLSL_MUTUAL = """#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Outp { int o[]; };
int even(int n);
int odd(int n) { return n == 0 ? 0 : even(n - 1); }
int even(int n) { return n == 0 ? 1 : odd(n - 1); }
void main() { o[gl_GlobalInvocationID.x] = even(int(gl_GlobalInvocationID.x)); }
"""


def compute_cgl(helpers, call="helper(float(i))"):
    return textwrap.dedent("""
        shader Recursion {
            HELPERS
            compute {
                layout(std430, binding = 0) buffer float values[];
                [numthreads(64, 1, 1)]
                void main() {
                    uint i = gl_GlobalInvocationID.x;
                    values[i] = CALL;
                }
            }
        }
        """).replace("HELPERS", helpers).replace("CALL", call)


def parse_cgl(text):
    register_default_sources()
    return SOURCE_REGISTRY.get("cgl").parse(text)


def translate(tmp_path, source, backend, suffix):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("target", TARGETS)
def test_direct_recursion_from_cuda_is_refused(tmp_path, target):
    with pytest.raises(RecursionUnsupportedError) as raised:
        translate(tmp_path, CUDA_FIB, target, ".cu")
    assert raised.value.code == f"project.translate.{target}-recursion-unsupported"
    assert raised.value.project_diagnostic_code == raised.value.code
    assert raised.value.cycle == ("fib", "fib")
    assert raised.value.target == target
    assert raised.value.missing_capabilities == (f"{target}.function-recursion",)


@pytest.mark.parametrize("target", TARGETS)
def test_mutual_recursion_through_a_prototype_is_refused(tmp_path, target):
    with pytest.raises(RecursionUnsupportedError) as raised:
        translate(tmp_path, GLSL_MUTUAL, target, ".comp")
    assert raised.value.cycle in (("odd", "even", "odd"), ("even", "odd", "even"))


@pytest.mark.parametrize("target", TARGETS)
def test_mutual_recursion_in_crossgl_is_refused_with_its_cycle(target):
    ast = parse_cgl(
        compute_cgl(
            "float first(float x) { return x > 1.0 ? second(x - 1.0) : x; }\n"
            "float second(float x) { return first(x * 0.5); }\n"
            "float helper(float x) { return first(x); }",
        )
    )
    with pytest.raises(RecursionUnsupportedError) as raised:
        validate_recursion_free(ast, target)
    assert raised.value.cycle == ("first", "second", "first")
    assert "recursive call cycle 'first -> second -> first'" in str(raised.value)


@pytest.mark.parametrize("target", TARGETS)
def test_recursion_outside_the_entry_call_tree_is_refused(target):
    ast = parse_cgl(
        compute_cgl(
            "float unused(float x) { return x > 1.0 ? unused(x - 1.0) : x; }\n"
            "float helper(float x) { return x * 2.0; }",
        )
    )
    with pytest.raises(RecursionUnsupportedError):
        validate_recursion_free(ast, target)


@pytest.mark.parametrize("target", TARGETS)
def test_acyclic_call_chains_and_overloads_translate(tmp_path, target):
    source = compute_cgl(
        "float scale(float x) { return x * 2.0; }\n"
        "float scale(vec2 v) { return scale(v.x) + scale(v.y); }\n"
        "float helper(float x) { return scale(vec2(x, 1.0)) + scale(x); }",
    )
    validate_recursion_free(parse_cgl(source), target)
    output = translate(tmp_path, source, target, ".cgl")
    assert ("OpFunctionCall" if target == "vulkan" else "helper(") in output


def test_overloads_with_equal_lanes_resolve_by_argument_type():
    ast = parse_cgl(
        compute_cgl(
            "float pick(float x) { return x; }\n"
            "float pick(int x) { return pick(float(x)); }\n"
            "float helper(float x) { return pick(int(x)); }",
        )
    )
    graph = CallGraph(ast)
    names = [function.name for function in graph.functions]
    pick_int = next(
        index
        for index, function in enumerate(graph.functions)
        if function.name == "pick" and "int" in repr(function.parameters[0].param_type)
    )
    assert [names[callee] for callee in graph.edges[pick_int]] == ["pick"]
    assert graph.find_cycle() is None


AGGREGATE_OVERLOADS = {
    "cooperative-matrix": (
        "CooperativeMatrix<float, 8, 8>",
        "CooperativeMatrix<float, 8, 8> m;",
    ),
    "matrix": ("mat2", "mat2 m = mat2(v.x, v.y, v.y, v.x);"),
    "structure": ("Pair", "Pair m; m.a = v.x; m.b = v.y;"),
}


@pytest.mark.parametrize("kind", sorted(AGGREGATE_OVERLOADS))
def test_aggregate_arguments_select_the_overload_their_declared_type_names(kind):
    """A ``vec2`` overload forwarding to its aggregate overload is not a cycle.

    The aggregate local has no value type, so value-type inference alone accepts both overloads.
    """
    aggregate, local = AGGREGATE_OVERLOADS[kind]
    ast = parse_cgl(
        compute_cgl(
            "struct Pair { float a; float b; }\n"
            f"float pick({aggregate} m) {{ return 1.0; }}\n"
            f"float pick(vec2 v) {{ {local} return pick(m); }}\n"
            "float helper(float x) { return pick(vec2(x, 1.0)); }",
        )
    )
    graph = CallGraph(ast)
    by_kind = {
        (
            "vec2" if "VectorType" in repr(function.parameters[0].param_type) else kind
        ): index
        for index, function in enumerate(graph.functions)
        if function.name == "pick"
    }
    assert sorted(graph.edges[by_kind["vec2"]]) == [by_kind[kind]]
    assert graph.find_cycle() is None
    for target in TARGETS:
        validate_recursion_free(ast, target)


def test_aggregate_self_calls_remain_cycles():
    ast = parse_cgl(
        compute_cgl(
            "float pick(mat2 m) { return pick(m); }\n"
            "float pick(vec2 v) { return v.x; }\n"
            "float helper(float x) { mat2 m = mat2(x); return pick(m); }",
        )
    )
    with pytest.raises(RecursionUnsupportedError) as raised:
        validate_recursion_free(ast, "metal")
    assert raised.value.cycle == ("pick", "pick")


def test_ambiguous_overloads_keep_every_candidate():
    ast = parse_cgl(
        compute_cgl(
            "float pick(float x) { return x; }\n"
            "float pick(int x) { return pick(unknown_value); }\n"
            "float helper(float x) { return pick(x); }",
        )
    )
    with pytest.raises(RecursionUnsupportedError) as raised:
        validate_recursion_free(ast, "vulkan")
    assert raised.value.cycle == ("pick", "pick")


@pytest.mark.parametrize("target", ["cuda", "hip", "directx", "opengl"])
def test_targets_with_their_own_recursion_rules_are_not_checked_here(target):
    ast = parse_cgl(
        compute_cgl("float helper(float x) { return x > 1.0 ? helper(x - 1.0) : x; }")
    )
    validate_recursion_free(ast, target)


def test_project_report_carries_the_typed_recursion_diagnostic(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "fib.cu").write_text(CUDA_FIB, encoding="utf-8")
    payload = translate_project(
        repo, targets=["metal"], output_dir="translated", format_output=False
    ).to_json()
    artifact = payload["artifacts"][0]
    assert artifact["status"] == "failed"
    assert payload["summary"]["diagnosticsByCode"] == {
        "project.translate.metal-recursion-unsupported": 1
    }
    assert payload["diagnostics"][0]["missingCapabilities"] == [
        "metal.function-recursion"
    ]
