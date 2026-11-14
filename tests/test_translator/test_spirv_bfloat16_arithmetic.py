"""bfloat16 arithmetic in SPIR-V output rounds every result back to bfloat16.

MSL ``bfloat`` and CUDA ``__nv_bfloat16`` operators round each result to bfloat16. SPIR-V computes bfloat16
operations in binary32, so the generator must round the binary32 result back to bfloat16 before it is used;
otherwise ``float(x * y)`` keeps up to 16 extra significand bits. The runtime check is the conformance case
``x_bfloat16_arithmetic``.
"""

import re
import shutil
import subprocess

import pytest

from crosstl import translate

SOURCE = """
shader Bfloat16Product {
    compute k {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer float* inp;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            bfloat16 x = bfloat16(inp[2u * i]);
            bfloat16 y = bfloat16(inp[2u * i + 1u]);
            float product = x OPERATOR y;
            outp[i] = product;
        }
    }
}
"""

OPCODES = {"*": "OpFMul", "+": "OpFAdd", "-": "OpFSub"}


def _instructions_after(spirv, opcode):
    lines = spirv.splitlines()
    index = next(i for i, line in enumerate(lines) if opcode in line)
    result = re.match(r"\s*(%\w+) =", lines[index]).group(1)
    return result, lines[index + 1 : index + 14]


@pytest.mark.parametrize("operator", sorted(OPCODES))
def test_bfloat16_results_are_rounded_back_to_bfloat16(tmp_path, operator):
    path = tmp_path / "product.cgl"
    path.write_text(SOURCE.replace("OPERATOR", operator))
    spirv = translate(str(path), backend="vulkan", format_output=False)

    result, following = _instructions_after(spirv, OPCODES[operator])
    text = "\n".join(following)
    # The binary32 result is reinterpreted as bits, rounded to nearest even at bit 16 and narrowed to the
    # 16-bit storage of bfloat16 before it is widened again for the float assignment.
    assert re.search(rf"= OpBitcast %\w+ {result}\b", text), text
    assert "OpShiftRightLogical" in text
    assert "OpUConvert" in text
    assert (
        "OpSelect" in text
    )  # NaN payloads keep a quiet bit instead of rounding to infinity

    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        assembly = tmp_path / "product.spvasm"
        binary = tmp_path / "product.spv"
        assembly.write_text(spirv)
        subprocess.run(
            ["spirv-as", "--target-env", "vulkan1.1", str(assembly), "-o", str(binary)],
            check=True,
        )
        subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.1", str(binary)], check=True
        )


def test_bfloat16_comparisons_stay_exact_in_binary32(tmp_path):
    path = tmp_path / "compare.cgl"
    path.write_text(
        SOURCE.replace(
            "float product = x OPERATOR y;", "float product = x < y ? 1.0 : 0.0;"
        )
    )
    spirv = translate(str(path), backend="vulkan", format_output=False)
    assert "OpFOrdLessThan" in spirv
