"""HLSL output refuses call cycles through several functions, which DXC rejects (issue #4747)."""

import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.directx_values import DirectXRecursiveFunctionError

MUTUAL_EVEN_ODD = """
shader Rec {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* o;
        int even(int n);
        int odd(int n) {
            if (n == 0) { return 0; }
            return even(n - 1);
        }
        int even(int n) {
            if (n == 0) { return 1; }
            return odd(n - 1);
        }
        void main() {
            uint i = gl_GlobalInvocationID.x;
            o[i] = even(o[i]);
        }
    }
}
"""

VOID_PING_PONG = """
shader Rec {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* o;
        void pong(uint i, int n);
        void ping(uint i, int n) {
            if (n > 0) { pong(i, n - 1); }
        }
        void pong(uint i, int n) {
            o[i] = n;
            if (n > 0) { ping(i, n - 1); }
        }
        void main() {
            uint i = gl_GlobalInvocationID.x;
            ping(i, o[i]);
        }
    }
}
"""

# A three-function cycle at program scope, entered from a helper that also calls itself in tail position.
PROGRAM_SCOPE_TRIANGLE = """
shader Rec {
    int c(int n);
    int a(int n) { return n <= 0 ? 0 : c(n - 1) + 1; }
    int b(int n) { return a(n) * 2; }
    int c(int n) { return b(n) - 1; }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* o;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            o[i] = a(o[i]);
        }
    }
}
"""


def generate(source):
    return HLSLCodeGen().generate(crosstl.translator.parse(textwrap.dedent(source)))


@pytest.mark.parametrize(
    ("source", "cycle"),
    [
        (MUTUAL_EVEN_ODD, ("odd", "even", "odd")),
        (VOID_PING_PONG, ("ping", "pong", "ping")),
        (PROGRAM_SCOPE_TRIANGLE, ("a", "c", "b", "a")),
    ],
    ids=["non-void-mutual", "void-mutual", "three-function-cycle"],
)
def test_hlsl_refuses_call_cycles_through_several_functions(source, cycle):
    with pytest.raises(DirectXRecursiveFunctionError) as exc_info:
        generate(source)

    error = exc_info.value
    assert error.reason == "indirect-recursion-unsupported"
    assert error.cycle == cycle
    assert error.function_name == cycle[0]
    assert error.project_diagnostic_code == (
        "project.translate.directx-recursion-unsupported"
    )
    assert error.missing_capabilities == ("directx.function-recursion",)
    assert " -> ".join(cycle) in str(error)


def test_hlsl_cycle_refusal_reaches_the_public_translate_api(tmp_path):
    path = tmp_path / "mutual.cgl"
    path.write_text(textwrap.dedent(MUTUAL_EVEN_ODD))

    with pytest.raises(DirectXRecursiveFunctionError, match="odd -> even -> odd"):
        crosstl.translate(str(path), backend="directx")


def test_hlsl_keeps_lowering_direct_void_tail_recursion_next_to_acyclic_calls():
    source = """
    shader Rec {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer int* o;
            int twice(int n) { return n * 2; }
            void countdown(uint i, int n) {
                o[i] = twice(n);
                if (n > 0) { countdown(i, n - 1); }
            }
            void main() {
                uint i = gl_GlobalInvocationID.x;
                countdown(i, o[i]);
            }
        }
    }
    """

    generated = generate(source)

    assert "void countdown(" in generated
    assert "while (true)" in generated


def test_hlsl_still_refuses_direct_non_void_recursion_with_its_own_reason():
    source = """
    shader Rec {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer int* o;
            int fact(int n) {
                if (n <= 1) { return 1; }
                return n * fact(n - 1);
            }
            void main() {
                uint i = gl_GlobalInvocationID.x;
                o[i] = fact(o[i]);
            }
        }
    }
    """

    with pytest.raises(DirectXRecursiveFunctionError) as exc_info:
        generate(source)

    assert exc_info.value.reason == "non-void-recursion-unsupported"
    assert exc_info.value.cycle == ("fact", "fact")
