"""Module-scope private variables are per-thread kernel locals on CUDA and HIP.

CrossGL's module-scope ``static`` variable, a stage-scope private variable and a function-scope ``static`` local
each have one copy per invocation. The CUDA and HIP generators declared them as namespace-scope host variables, which
nvcc and hipcc refuse to let device code read (``identifier "G" is undefined in device code``); a ``__device__``
variable would instead be one copy shared by every thread.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics import DevicePrivateGlobalUnsupportedError
from crosstl.translator.semantics.device_private_globals import (
    lower_device_private_globals,
)

BACKENDS = ["cuda", "hip"]

PRIVATES = """
shader S {
    static int D = 4;
    compute {
        buffer float* out_values;
        int G = 7;
        int bump(int x) { G = G + x; return G + D; }
        int outer(int x) { return bump(x) * 2; }
        int pure(int x) { return x + 1; }
        [numthreads(64, 1, 1)]
        void main(uvec3 tid : SV_DispatchThreadID) {
            D = D + 1;
            out_values[tid.x] = float(outer(int(tid.x)) + pure(1));
        }
    }
}
"""

STATIC_LOCAL = """
shader S {
    compute {
        buffer int* out_values;
        int next() { static int c = 0; c = c + 1; return c; }
        [numthreads(64, 1, 1)]
        void main(uvec3 tid : SV_DispatchThreadID) {
            int a = next();
            out_values[tid.x] = a + next();
        }
    }
}
"""

SHADOWED = """
shader S {
    static int G = 1;
    compute {
        buffer int* out_values;
        int own() { int G = 2; return G; }
        int shared_value() { return G; }
        [numthreads(64, 1, 1)]
        void main(uvec3 tid : SV_DispatchThreadID) {
            int G = 5;
            out_values[tid.x] = G + own() + shared_value();
        }
    }
}
"""

AGGREGATES = """
shader S {
    struct Pair { int a; float b; }
    static float table[3] = {1.0, 2.0, 3.0};
    static Pair pair;
    static float total;
    compute {
        buffer float* out_values;
        void accumulate(int i) { total = total + table[i]; pair.a = i; }
        [numthreads(64, 1, 1)]
        void main(uvec3 tid : SV_DispatchThreadID) {
            accumulate(int(tid.x) % 3);
            out_values[tid.x] = total + float(pair.a);
        }
    }
}
"""

TWO_KERNELS = """
shader S {
    static int G = 3;
    int read_g() { return G; }
    int other() { return 1; }
    int other(int x) { return x + G; }
    @compute
    void first(int* out_values) { out_values[0] = read_g() + other(); }
    @compute
    void second(int* out_values) { out_values[0] = 2; }
}
"""

POINTER = """
shader S {
    static float* cursor;
    compute {
        buffer float* out_values;
        [numthreads(64, 1, 1)]
        void main(uvec3 tid : SV_DispatchThreadID) {
            out_values[tid.x] = cursor[0];
        }
    }
}
"""


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _body(code, signature):
    """The text of the function whose definition starts with ``signature``."""
    start = code.index(signature)
    depth = 0
    for index in range(code.index("{", start), len(code)):
        depth += {"{": 1, "}": -1}.get(code[index], 0)
        if depth == 0:
            return code[start : index + 1]
    raise AssertionError(f"unterminated function {signature}")


def _compile(tmp_path, code, backend):
    """Compile ``code`` for the device with the target's own compiler when it is installed."""
    if backend == "cuda" and shutil.which("nvcc"):
        (tmp_path / "out.cu").write_text(code)
        command = [shutil.which("nvcc"), "-std=c++17", "--ptx", "out.cu"]
    elif backend == "hip" and shutil.which("hipcc"):
        (tmp_path / "out.hip").write_text(code)
        command = [
            shutil.which("hipcc"),
            "-std=c++17",
            "--offload-device-only",
            "-c",
            "out.hip",
        ]
    else:
        return
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=tmp_path, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _module_lines(code):
    return re.findall(
        r"(?m)^(?:static )?(?:int|float|Pair) \w+(?:\[\d+\])?(?: = [^;]*)?;", code
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_private_variables_are_kernel_locals_passed_by_reference(tmp_path, backend):
    code = _translate(tmp_path, PRIVATES, backend)
    assert _module_lines(code) == []
    assert re.search(r"__device__ int bump\(int& D, int& G, int x\)", code)
    assert re.search(r"__device__ int outer\(int& D, int& G, int x\)", code)
    assert re.search(r"__device__ int pure\(int x\)", code)
    assert "return (bump(D, G, x) * 2);" in code
    kernel = _body(code, "compute_main()")
    assert kernel.index("int D = 4;") < kernel.index("int G = 7;")
    assert kernel.index("int G = 7;") < kernel.index("D = (D + 1);")
    assert "outer(D, G, " in kernel and "pure(1)" in kernel
    _compile(tmp_path, code, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_static_local_keeps_its_value_between_calls_per_thread(tmp_path, backend):
    code = _translate(tmp_path, STATIC_LOCAL, backend)
    assert re.search(r"__device__ int next\(int& next_c\)", code)
    kernel = _body(code, "compute_main()")
    assert "int next_c = 0;" in kernel
    assert kernel.count("next(next_c)") == 2
    assert _module_lines(code) == []
    _compile(tmp_path, code, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_redeclared_name_gets_a_fresh_one(tmp_path, backend):
    code = _translate(tmp_path, SHADOWED, backend)
    assert re.search(r"__device__ int own\(\)\s*\{\s*int G = 2;\s*return G;", code)
    assert re.search(
        r"__device__ int shared_value\(int& G_1\)\s*\{\s*return G_1;", code
    )
    kernel = _body(code, "compute_main()")
    assert kernel.index("int G_1 = 1;") < kernel.index("int G = 5;")
    assert "((G + own()) + shared_value(G_1))" in kernel
    _compile(tmp_path, code, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_aggregates_and_uninitialized_values(tmp_path, backend):
    code = _translate(tmp_path, AGGREGATES, backend)
    assert re.search(
        r"__device__ void accumulate\(float \(&table\)\[3\], Pair& pair, float& total, int i\)",
        code,
    )
    kernel = _body(code, "compute_main()")
    assert "float table[3] = {1.0f, 2.0f, 3.0f};" in kernel
    assert "Pair pair = {};" in kernel
    assert re.search(r"float total = 0(?:\.0)?f;", kernel)
    assert "accumulate(table, pair, total, " in kernel
    _compile(tmp_path, code, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_only_kernels_reaching_a_variable_declare_it(tmp_path, backend):
    code = _translate(tmp_path, TWO_KERNELS, backend)
    assert re.search(r"__device__ int read_g\(int& G\)", code)
    # Every overload of a name takes the same parameters, so each call passes the same arguments.
    assert re.search(r"__device__ int other\(int& G\)", code)
    assert re.search(r"__device__ int other\(int& G, int x\)", code)
    first = _body(code, "void first(")
    second = _body(code, "void second(")
    assert "int G = 3;" in first and "(read_g(G) + other(G))" in first
    assert "G" not in second
    assert _module_lines(code) == []
    _compile(tmp_path, code, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_private_variable_that_is_not_a_value_is_refused(tmp_path, backend):
    with pytest.raises(DevicePrivateGlobalUnsupportedError, match="'cursor'"):
        _translate(tmp_path, POINTER, backend)


@pytest.mark.parametrize("target", ["opengl", "directx", "metal", "vulkan", "slang"])
def test_other_targets_keep_module_scope_private_variables(target):
    program = _parse(PRIVATES)
    assert lower_device_private_globals(program, target) == 0
    assert [v.name for v in program.global_variables] == ["D"]


def test_lowering_reports_the_variables_it_moved():
    # Only the program-scope ``D`` is at module scope before the stage-scope ``G`` is hoisted.
    program = _parse(PRIVATES)
    assert lower_device_private_globals(program, "cuda") == 1
    assert program.global_variables == []
