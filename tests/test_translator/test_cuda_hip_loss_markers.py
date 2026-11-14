"""CUDA and HIP refuse a program that would carry an ``/* unsupported ... */`` loss marker.

Several CUDA and HIP emitter sites wrote such a marker in place of a construct they could not lower and still
returned the program, so ``translate`` produced source that compiles to something else or not at all. Generation
now runs through :meth:`LoweringRefusals.run`, which refuses any rendered program that still carries a marker.
"""

import pytest

import crosstl
from crosstl.translator.codegen.cuda_codegen import CudaCodeGen
from crosstl.translator.codegen.cuda_refusals import CudaLoweringRefused
from crosstl.translator.codegen.hip_codegen import HipCodeGen
from crosstl.translator.codegen.hip_refusals import HipLoweringRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from tests.lowering_refusal_helpers import (
    CUDA_REFUSAL_CODE,
    HIP_REFUSAL_CODE,
    cuda_partial_output,
    hip_partial_output,
    refused_cuda,
    refused_hip,
)

INVALID_ATOMICS = """
shader S {
    compute {
        buffer uint counters[];
        layout(local_size_x = 64) in;
        void main() {
            uint index = gl_GlobalInvocationID.x;
            uint missing = atomicAdd(counters[index]);
            uint2 badVector = atomicAdd(uint2(index, index), 1u);
        }
    }
}
"""

VALID_ATOMICS = """
shader S {
    compute {
        buffer uint counters[];
        layout(local_size_x = 64) in;
        void main() {
            uint index = gl_GlobalInvocationID.x;
            uint old = atomicAdd(counters[index], 1u);
            atomicMax(counters[0], old);
        }
    }
}
"""

BACKENDS = [
    pytest.param(
        CudaCodeGen, CudaLoweringRefused, refused_cuda, cuda_partial_output, id="cuda"
    ),
    pytest.param(
        HipCodeGen, HipLoweringRefused, refused_hip, hip_partial_output, id="hip"
    ),
]


def parse(source):
    return Parser(Lexer(source).tokens).parse()


@pytest.mark.parametrize("codegen, error_type, refuse, partial", BACKENDS)
def test_every_loss_marker_is_a_listed_refusal(codegen, error_type, refuse, partial):
    message = refuse(codegen().generate, parse(INVALID_ATOMICS))
    label = error_type.label

    assert message.startswith(f"{label} cannot lower 2 constructs: ")
    assert "atomicAdd requires 2 argument(s)" in message
    assert "atomicAdd requires assignable scalar target" in message

    # The partial program is what the emitter used to return: it carries one marker per refused site.
    output, _ = partial(codegen().generate, parse(INVALID_ATOMICS))
    assert output.count(f"/* unsupported {label} ") == 2


@pytest.mark.parametrize("codegen, error_type, refuse, partial", BACKENDS)
def test_a_program_without_markers_is_returned(codegen, error_type, refuse, partial):
    output = codegen().generate(parse(VALID_ATOMICS))

    assert "atomicAdd(&counters[index], 1u)" in output
    assert "atomicMax(&counters[0], old)" in output
    assert "unsupported" not in output


@pytest.mark.parametrize(
    "backend, error_type, code",
    [
        ("cuda", CudaLoweringRefused, CUDA_REFUSAL_CODE),
        ("hip", HipLoweringRefused, HIP_REFUSAL_CODE),
    ],
)
def test_translate_returns_no_program_with_a_loss_marker(
    tmp_path, backend, error_type, code
):
    source = tmp_path / "invalid_atomics.cgl"
    source.write_text(INVALID_ATOMICS, encoding="utf-8")

    with pytest.raises(error_type) as refusal:
        crosstl.translate(str(source), backend=backend)

    assert refusal.value.code == code
    assert len(refusal.value.reasons) == 2
    assert all("requires" in reason for reason in refusal.value.reasons)


@pytest.mark.parametrize("codegen, error_type, refuse, partial", BACKENDS)
def test_a_reused_emitter_reports_each_program_once(
    codegen, error_type, refuse, partial
):
    emitter = codegen()
    refuse(emitter.generate, parse(INVALID_ATOMICS))

    # The refusals of the first program do not leak into the next one.
    output = emitter.generate(parse(VALID_ATOMICS))
    assert "unsupported" not in output
