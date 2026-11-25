"""Slang output refuses recursive call cycles, which slangc rejects (issue #4747)."""

import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.call_graph import RecursionUnsupportedError
from crosstl.translator.codegen.slang_codegen import SlangCodeGen

DIRECT_NON_VOID_SELF_CALL = """
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

VOID_NON_TAIL_SELF_CALL = """
shader Rec {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* o;
        void fill(uint i, int n) {
            if (n > 0) { fill(i, n - 1); }
            o[i] = n;
        }
        void main() {
            uint i = gl_GlobalInvocationID.x;
            fill(i, o[i]);
        }
    }
}
"""

# A three-function cycle at program scope.
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
    return SlangCodeGen().generate(crosstl.translator.parse(textwrap.dedent(source)))


@pytest.mark.parametrize(
    ("source", "cycle"),
    [
        (DIRECT_NON_VOID_SELF_CALL, ("fact", "fact")),
        (MUTUAL_EVEN_ODD, ("odd", "even", "odd")),
        (VOID_PING_PONG, ("ping", "pong", "ping")),
        (VOID_NON_TAIL_SELF_CALL, ("fill", "fill")),
        (PROGRAM_SCOPE_TRIANGLE, ("a", "c", "b", "a")),
    ],
    ids=[
        "direct-non-void-self-call",
        "non-void-mutual",
        "void-mutual",
        "void-non-tail-self-call",
        "three-function-cycle",
    ],
)
def test_slang_refuses_recursive_call_cycles(source, cycle):
    with pytest.raises(RecursionUnsupportedError) as exc_info:
        generate(source)

    error = exc_info.value
    assert error.cycle == cycle
    assert error.target == "slang"
    assert error.code == "project.translate.slang-recursion-unsupported"
    assert error.project_diagnostic_code == error.code
    assert error.missing_capabilities == ("slang.function-recursion",)
    assert " -> ".join(cycle) in str(error)
    assert "E55201" in str(error)


def test_slang_cycle_refusal_reaches_the_public_translate_api(tmp_path):
    path = tmp_path / "mutual.cgl"
    path.write_text(textwrap.dedent(MUTUAL_EVEN_ODD))

    with pytest.raises(RecursionUnsupportedError, match="odd -> even -> odd"):
        crosstl.translate(str(path), backend="slang")


def test_slang_keeps_lowering_direct_void_tail_recursion():
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
    assert "while" in generated
