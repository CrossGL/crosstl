"""CUDA and HIP output declare a function before a call that precedes its definition.

A CrossGL function may call a function defined after it. The CUDA importer, for one, drops a kernel's prototypes, so
``__device__ float twice(float x);`` followed by a caller and then the definition came back from the round trip with
the call ahead of every declaration of ``twice``, which nvcc and hipcc reject (``identifier "twice" is undefined``).
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.cuda_codegen import CudaCodeGen
from crosstl.translator.codegen.hip_codegen import HipCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

GENERATORS = {"cuda": CudaCodeGen, "hip": HipCodeGen}


def _program(functions, stage_functions="", body="out_[i] = quadruple(inp[i]);"):
    return f"""
shader main {{
{functions}
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* inp;
        buffer float* out_;
{stage_functions}
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            {body}
        }}
    }}
}}
"""


CALLER_FIRST = """
    float quadruple(float x) { return twice(twice(x)); }
    float twice(float x) { return x * 2.0; }
"""


def _generate(backend, source):
    return GENERATORS[backend]().generate(Parser(Lexer(source).tokens).parse())


def _declarations(code, name):
    """``(line index, is definition)`` of each declaration of function ``name``."""
    return [
        (index, not line.rstrip().endswith(";"))
        for index, line in enumerate(code.splitlines())
        if re.match(rf"^__device__ [\w ]+ {name}\(", line)
    ]


def _first_call(code, name, caller):
    lines = code.splitlines()
    for index, line in enumerate(lines):
        if re.match(rf"^__device__ [\w ]+ {caller}\(", line):
            return index
    raise AssertionError(f"{caller} is not defined:\n{code}")


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_function_called_before_its_definition_is_declared_first(backend):
    code = _generate(backend, _program(CALLER_FIRST))

    declarations = _declarations(code, "twice")
    assert [is_definition for _index, is_definition in declarations] == [False, True]
    assert declarations[0][0] < _first_call(code, "twice", "quadruple")
    assert "__device__ float twice(float x);" in code


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_functions_in_call_order_get_no_prototype(backend):
    callee_first = """
    float twice(float x) { return x * 2.0; }
    float quadruple(float x) { return twice(twice(x)); }
"""
    code = _generate(backend, _program(callee_first))

    assert [d for _i, d in _declarations(code, "twice")] == [True]
    assert [d for _i, d in _declarations(code, "quadruple")] == [True]


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_self_recursion_needs_no_prototype(backend):
    recursive = """
    float quadruple(float x) { if (x > 64.0) { return x; } return quadruple(x * 4.0); }
"""
    code = _generate(backend, _program(recursive))

    assert [d for _i, d in _declarations(code, "quadruple")] == [True]


MUTUAL = """
    uint is_odd(uint n) { if (n == 0u) { return 0u; } return is_even(n - 1u); }
    uint is_even(uint n) { if (n == 0u) { return 1u; } return is_odd(n - 1u); }
"""


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_mutual_recursion_declares_the_later_function(backend):
    code = _generate(
        backend, _program(MUTUAL, body="out_[i] = float(is_odd(uint(inp[i])));")
    )

    assert [d for _i, d in _declarations(code, "is_even")] == [False, True]
    assert [d for _i, d in _declarations(code, "is_odd")] == [True]


OVERLOADS = """
    float quadruple(float x) { return twice(twice(x)) + float(twice(int(x))); }
    float twice(float x) { return x * 2.0; }
    int twice(int x) { return x * 2; }
"""


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_every_later_overload_of_a_called_name_is_declared(backend):
    code = _generate(backend, _program(OVERLOADS))

    assert [d for _i, d in _declarations(code, "twice")] == [False, False, True, True]
    assert "__device__ int twice(int x);" in code


STAGE_FUNCTIONS = """
        float scaled(float x) { return offset(x) * 3.0; }
        float offset(float x) { return x + 1.0; }
"""


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_stage_function_called_before_its_definition_is_declared_first(backend):
    code = _generate(
        backend,
        _program("", STAGE_FUNCTIONS, body="out_[i] = scaled(inp[i]);"),
    )

    declarations = _declarations(code, "offset")
    assert [d for _i, d in declarations] == [False, True]
    assert declarations[0][0] < _first_call(code, "offset", "scaled")


CUDA_SOURCE = """__device__ float twice(float x);

__device__ float quadruple(float x) {
    return twice(twice(x));
}

__device__ float twice(float x) {
    return x * 2.0f;
}

extern "C" __global__ void probe(float* fin, float* outp) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    outp[i] = quadruple(fin[i]);
}
"""


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_cuda_kernel_with_a_prototype_keeps_the_declaration_order(tmp_path, backend):
    source = tmp_path / "probe.cu"
    source.write_text(CUDA_SOURCE)

    code = crosstl.translate(str(source), backend=backend, format_output=False)

    declarations = _declarations(code, "twice")
    assert [d for _i, d in declarations] == [False, True]
    assert declarations[0][0] < _first_call(code, "twice", "quadruple")


COMPILED = _program(
    CALLER_FIRST
    + MUTUAL
    + OVERLOADS.replace("quadruple", "octuple").replace("twice", "doubled"),
    STAGE_FUNCTIONS,
    body=(
        "out_[i] = quadruple(inp[i]) + octuple(inp[i]) + scaled(inp[i])"
        " + float(is_odd(uint(inp[i])));"
    ),
)


def _compile(command, tmp_path, name, code):
    source = tmp_path / name
    source.write_text(code)
    result = subprocess.run(
        [*command, str(source), "-o", str(tmp_path / "probe.o")],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-4000:]


def test_nvcc_accepts_the_prototypes(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not installed")
    _compile(
        [nvcc, "-arch=sm_80", "-std=c++17", "-c"],
        tmp_path,
        "probe.cu",
        _generate("cuda", COMPILED),
    )


def test_hipcc_accepts_the_prototypes(tmp_path):
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc is not installed")
    _compile(
        [hipcc, "--offload-arch=gfx1100", "-std=c++17", "-c"],
        tmp_path,
        "probe.hip",
        _generate("hip", COMPILED),
    )
