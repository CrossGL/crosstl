"""CrossGL ``out`` operands of CUDA device functions that write through a pointer: CUDA passes the address, HIP stores."""

import shutil
import subprocess

import pytest

from crosstl.translator.codegen.cuda_codegen import CudaCodeGen
from crosstl.translator.codegen.hip_codegen import HipCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.refusals import CudaRefusal, HipRefusal

STORES = ("__stcg", "__stcs", "__stwb", "__stwt")


def _program(body, extra=""):
    return f"""
shader Probe {{
    compute {{
        layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer uint* outp;
        {extra}
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            {body}
        }}
    }}
}}
"""


def _generate(generator, source):
    return generator().generate(Parser(Lexer(source).tokens).parse())


def _lines(code):
    return [line.strip() for line in code.splitlines()]


STORE_BODY = """
            int a0;
            int a1 = iin[i];
            {name}(a0, a1);
            {name}(outp[i], uint(a0));
            outp[i] = uint(a0);
"""


@pytest.mark.parametrize("name", STORES)
def test_cuda_store_with_cache_hint_takes_the_operand_address(name):
    lines = _lines(_generate(CudaCodeGen, _program(STORE_BODY.format(name=name))))
    assert f"{name}(&(a0), a1);" in lines
    assert f"{name}(&(outp[i]), uint(a0));" in lines


@pytest.mark.parametrize("name", STORES)
def test_hip_store_with_cache_hint_is_a_plain_store(name):
    code = _generate(HipCodeGen, _program(STORE_BODY.format(name=name)))
    assert name not in code
    lines = _lines(code)
    assert "(a0 = a1);" in lines
    assert any(line.startswith("(outp[i] = ") for line in lines)


MATCH_BODY = """
            uint a0 = uint(iin[i]);
            uint a1 = uint(iin[i + 1u]);
            int a2;
            uint r = __match_all_sync(a0, a1, a2);
            outp[i] = r + uint(a2);
"""


def test_cuda_match_all_takes_the_predicate_address():
    lines = _lines(_generate(CudaCodeGen, _program(MATCH_BODY)))
    assert "uint r = __match_all_sync(a0, a1, &(a2));" in lines


def test_hip_match_all_passes_the_predicate_address_to_its_helper():
    # HIP declares no __match_all_sync; HIP output computes it over a 32-lane segment of the wavefront.
    code = _generate(HipCodeGen, _program(MATCH_BODY))
    assert "unsigned int r = cgl_cuda_match_all_sync(a0, a1, &(a2));" in _lines(code)
    assert "__match_all_sync(" not in code.replace("cgl_cuda_match_all_sync(", "")


@pytest.mark.parametrize(
    "generator,refusal", [(CudaCodeGen, CudaRefusal), (HipCodeGen, HipRefusal)]
)
@pytest.mark.parametrize(
    "body,reason",
    [
        ("int a0; __stcg(a0 + 1, 2);", "neither an address"),
        ("bool b; __stcg(b, true);", "bool"),
        ("int a0; __stcg(a0);", "takes 2 operands"),
    ],
)
def test_unusable_operands_are_refused(generator, refusal, body, reason):
    with pytest.raises(refusal, match=reason):
        _generate(generator, _program(body))


# Every store once, on a local and on a buffer element, with the stored local read back.
COMPILED_STORES = (
    "int s0; int s1 = iin[i];"
    + "".join(f" {name}(s0, s1); {name}(outp[i], uint(s0));" for name in STORES)
    + " outp[i] = outp[i] + uint(s0);"
)


def _compile(command, tmp_path, name, code):
    source = tmp_path / name
    source.write_text(code)
    result = subprocess.run(
        [*command, str(source), "-o", str(tmp_path / "probe.o")],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_nvcc_accepts_the_address_operands(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is not installed")
    _compile(
        [nvcc, "-arch=sm_80", "-std=c++17", "-c"],
        tmp_path,
        "probe.cu",
        _generate(CudaCodeGen, _program(COMPILED_STORES + MATCH_BODY)),
    )


def test_hipcc_accepts_the_plain_stores(tmp_path):
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc is not installed")
    _compile(
        [hipcc, "--offload-arch=gfx1100", "-std=c++17", "-c"],
        tmp_path,
        "probe.hip",
        _generate(HipCodeGen, _program(COMPILED_STORES)),
    )
