"""Self tail calls become loops on the targets that forbid recursion (Metal, Vulkan SPIR-V, WGSL).

A function whose every self call is a tail call is rewritten as an unconditional loop over its body, rebinding the
parameters a tail call changes; any other recursion is still refused with the typed recursion diagnostic.
"""

import os
import re
import shutil
import subprocess
import sys
import textwrap

import pytest

import crosstl
from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator import ast as A
from crosstl.translator.codegen.call_graph import (
    RECURSION_FREE_TARGETS,
    RecursionUnsupportedError,
    validate_recursion_free,
)
from crosstl.translator.semantics.rewrite import child_slots, get_child
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

TARGETS = sorted(RECURSION_FREE_TARGETS)
REQUIRE_ENV = "CROSTL_REQUIRE_SELF_TAIL_CALLS"
WORKGROUP = (64, 1, 1)
GROUPS = 2
EXTENT = 5
VALUES = [(index * 2654435761 + 7) % 1000003 for index in range(GROUPS * 64)]
GUARD = [0x6D2B79F5] * 4

# A cursor that carries into the next position by calling itself, as a row walker does, and a value-returning
# Euclid whose tail call swaps both parameters.
METAL_SOURCE = """#include <metal_stdlib>
using namespace metal;
struct Cursor {
  int index;
  int carries;
  void advance(int n, const constant int* extent) thread {
    if (n == 0) {
      return;
    }
    index += n;
    if (index >= extent[0]) {
      int extra = index - extent[0];
      index = 0;
      carries += 1;
      if (extra > 0) {
        advance(extra, extent);
      }
    }
  }
};
uint euclid(uint a, uint b) {
  if (b == 0u) {
    return a;
  }
  return euclid(b, a % b);
}
kernel void walk(const device uint* values [[buffer(0)]],
                 device uint* results [[buffer(1)]],
                 const constant int* extent [[buffer(2)]],
                 uint index [[thread_position_in_grid]]) {
  Cursor cursor{0, 0};
  cursor.advance(int(values[index] % 97u), extent);
  results[index] = uint(cursor.index) * 16777216u + uint(cursor.carries) * 65536u
      + euclid(values[index], 360u);
}
"""


def _cgl(helpers, call="helper(i, 0u)"):
    return textwrap.dedent("""
        shader TailCalls {
            HELPERS
            compute {
                layout(std430, binding = 0) buffer uint values[];
                [numthreads(64, 1, 1)]
                void main() {
                    uint i = gl_GlobalInvocationID.x;
                    values[i] = CALL;
                }
            }
        }
        """).replace("HELPERS", textwrap.dedent(helpers)).replace("CALL", call)


def _parse(text):
    register_default_sources()
    return SOURCE_REGISTRY.get("cgl").parse(text)


def _walk(node):
    yield node
    for field, key in list(child_slots(node)):
        yield from _walk(get_child(node, field, key))


def _function(program, name):
    return next(
        function
        for function in program.functions
        if function.name == name and function.body is not None
    )


def _self_calls(function):
    return [
        node
        for node in _walk(function.body)
        if isinstance(node, A.FunctionCallNode)
        and isinstance(node.function, A.IdentifierNode)
        and node.function.name == function.name
    ]


ACCUMULATE = """
    uint helper(uint n, uint acc) {
        if (n <= 1u) {
            return acc;
        }
        return helper(n - 1u, acc * n);
    }
"""

COUNT_DOWN = """
    void count(inout uint total, uint n) {
        if (n == 0u) {
            return;
        }
        total += n;
        if (n > 10u) {
            count(total, n - 2u);
            return;
        }
        count(total, n - 1u);
    }
    uint helper(uint n, uint unused) {
        uint total = 0u;
        count(total, n % 40u);
        return total;
    }
"""


@pytest.mark.parametrize("target", TARGETS)
def test_value_returning_tail_call_rebinds_both_parameters(target):
    program = _parse(_cgl(ACCUMULATE))
    validate_recursion_free(program, target)
    helper = _function(program, "helper")
    assert not _self_calls(helper)
    assert [parameter.name for parameter in helper.parameters] == [
        "n_entry",
        "acc_entry",
    ]
    *entries, result, loop, returned = helper.body.statements
    assert [(v.name, v.initial_value.name) for v in entries] == [
        ("n", "n_entry"),
        ("acc", "acc_entry"),
    ]
    assert isinstance(loop, A.LoopNode)
    assert result.name == "tail_result" and returned.value.name == "tail_result"
    # ``return acc;`` stores the result and leaves the loop.
    exits = [node for node in _walk(loop) if isinstance(node, A.BreakNode)]
    assert len(exits) == 1 and not any(
        isinstance(node, A.ReturnNode) for node in _walk(loop)
    )


@pytest.mark.parametrize("target", TARGETS)
def test_void_tail_calls_keep_unchanged_reference_parameters(target):
    program = _parse(_cgl(COUNT_DOWN))
    validate_recursion_free(program, target)
    count = _function(program, "count")
    assert not _self_calls(count)
    assert [parameter.name for parameter in count.parameters] == ["total", "n_entry"]


def test_lowered_tail_calls_translate_and_validate(tmp_path):
    for helpers in (ACCUMULATE, COUNT_DOWN):
        path = tmp_path / "tail.cgl"
        path.write_text(_cgl(helpers))
        metal = crosstl.translate(str(path), backend="metal", format_output=False)
        assert "while (true)" in metal
        if shutil.which("xcrun"):
            _compile(metal, "metal", tmp_path, metal_compile_flags=("-Wall",))
        spirv = crosstl.translate(str(path), backend="vulkan", format_output=False)
        assert "OpLoopMerge" in spirv
        wgsl = crosstl.translate(str(path), backend="wgsl", format_output=False)
        assert re.search(r"\bloop\s*\{", wgsl)
        if shutil.which("spirv-as") and shutil.which("spirv-val"):
            (tmp_path / "tail.spvasm").write_text(spirv)
            for command in (
                [
                    "spirv-as",
                    "--target-env",
                    "vulkan1.0",
                    str(tmp_path / "tail.spvasm"),
                    "-o",
                    str(tmp_path / "tail.spv"),
                ],
                ["spirv-val", "--target-env", "vulkan1.0", str(tmp_path / "tail.spv")],
            ):
                subprocess.run(command, check=True, capture_output=True)
        # WGSL has no reference parameters, so only the by-value helper is a valid module.
        if shutil.which("naga") and helpers is ACCUMULATE:
            (tmp_path / "tail.wgsl").write_text(wgsl)
            subprocess.run(
                ["naga", str(tmp_path / "tail.wgsl")], check=True, capture_output=True
            )


REFUSED = {
    "non-tail": (
        """
        void step(inout uint total, uint n) {
            if (n > 0u) {
                step(total, n - 1u);
                total += 1u;
            }
        }
    """
    ),
    "inside-loop": (
        """
        void step(inout uint total, uint n) {
            for (uint k = 0u; k < 2u; k++) {
                if (n > k) {
                    step(total, n - 1u);
                    return;
                }
            }
        }
    """
    ),
    "rebinds-reference": (
        """
        void step(inout uint total, inout uint other, uint n) {
            if (n > 0u) {
                step(other, total, n - 1u);
            }
        }
    """
    ),
    "shadowed-parameter": (
        """
        void step(inout uint total, uint n) {
            if (total > 0u) {
                uint n = total - 1u;
                total = n;
            }
            if (n > 0u) {
                step(total, n - 1u);
            }
        }
    """
    ),
    "return-inside-loop": (
        """
        uint step(uint n) {
            for (uint k = 0u; k < 2u; k++) {
                if (n == k) {
                    return k;
                }
            }
            return step(n - 1u);
        }
    """
    ),
    "call-in-argument": (
        """
        uint step(uint n) {
            if (n == 0u) {
                return 0u;
            }
            return step(step(n - 1u));
        }
    """
    ),
}


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("shape", sorted(REFUSED))
def test_recursion_that_is_not_a_lowerable_tail_call_is_refused(target, shape):
    helpers = REFUSED[shape] + """
        uint helper(uint n, uint unused) {
            uint total = n;
            uint other = 1u;
            CALL;
            return total;
        }
    """
    call = (
        "step(n)"
        if shape in ("call-in-argument", "return-inside-loop")
        else (
            "step(total, other, n)"
            if shape == "rebinds-reference"
            else "step(total, n)"
        )
    )
    program = _parse(_cgl(helpers.replace("CALL", call)))
    with pytest.raises(RecursionUnsupportedError) as raised:
        validate_recursion_free(program, target)
    assert raised.value.cycle == ("step", "step")
    assert _self_calls(_function(program, "step"))


@pytest.mark.parametrize("target", TARGETS)
def test_mutual_tail_recursion_is_still_refused(target):
    program = _parse(_cgl("""
        uint odd(uint n, uint acc);
        uint even(uint n, uint acc) {
            if (n == 0u) {
                return acc;
            }
            return odd(n - 1u, acc + 1u);
        }
        uint odd(uint n, uint acc) {
            if (n == 0u) {
                return acc;
            }
            return even(n - 1u, acc);
        }
        uint helper(uint n, uint acc) {
            return even(n, acc);
        }
    """))
    with pytest.raises(RecursionUnsupportedError):
        validate_recursion_free(program, target)


def _expected():
    words = []
    for value in VALUES:
        index, carries, n = 0, 0, value % 97
        while n:
            index += n
            n = 0
            if index >= EXTENT:
                n, index, carries = index - EXTENT, 0, carries + 1
        a, b = value, 360
        while b:
            a, b = b, a % b
        words.append(index * 16777216 + carries * 65536 + a)
    return words + GUARD


def _request(root, target):
    _, descriptor, package = _package(
        root, target, "uint", WORKGROUP, source=METAL_SOURCE, software_subgroups=False
    )
    initial = [0xDEADBEEF] * len(VALUES) + GUARD
    inputs = _bound_values(
        descriptor,
        {
            "values": {"dtype": "uint32", "shape": [len(VALUES)], "values": VALUES},
            "results": {"dtype": "uint32", "shape": [len(initial)], "values": initial},
            "extent": {"dtype": "int32", "shape": [1], "values": [EXTENT]},
        },
    )
    expected = _expected()
    outputs = _bound_values(
        descriptor,
        {"results": {"dtype": "uint32", "shape": [len(expected)], "values": expected}},
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [GROUPS, 1, 1], "workgroupSize": list(WORKGROUP)},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return request, outputs


def test_metal_row_walker_translates_to_a_loop_and_compiles(tmp_path):
    request, _ = _request(tmp_path, "metal")
    generated = request.artifact_path.read_text()
    assert "while (true)" in generated
    if shutil.which("xcrun"):
        _compile(generated, "metal", tmp_path, metal_compile_flags=("-Wall", "-Wextra"))


def test_metal_row_walker_executes_natively(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native tail-call loops")
    if sys.platform != "darwin":
        pytest.skip("the tail-call loop lowering targets Metal on macOS")
    from tests.test_translator.test_loop_updates import _execute

    request, expected = _request(tmp_path, "metal")
    _execute(request, expected, tmp_path, metal_compile_flags=("-Wall", "-Wextra"))
