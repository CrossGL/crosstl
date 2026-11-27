"""SPIR-V module constants fold bfloat16 constructors to scalar bit patterns."""

import re
import shutil
import subprocess

from crosstl import translate

SOURCE = """
shader Bfloat16Constants {
    compute main {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        constant bfloat16 integer_value = bfloat16(16842753u);
        const bfloat16 float_value = bfloat16(1.00390625);
        buffer float* output;
        void main() {
            output[0] = float(integer_value);
            output[1] = float(float_value);
        }
    }
}
"""


def test_bfloat16_module_initializers_are_folded_to_spirv_constants(tmp_path):
    source = tmp_path / "bfloat16_constants.cgl"
    assembly = tmp_path / "bfloat16_constants.spvasm"
    binary = tmp_path / "bfloat16_constants.spv"
    source.write_text(SOURCE)

    spirv = translate(str(source), backend="vulkan", format_output=False)

    u16_type = re.search(r"^(%\d+) = OpTypeInt 16 0$", spirv, re.MULTILINE)
    assert u16_type, spirv
    u16_constants = {
        int(value)
        for value in re.findall(
            rf"^%\d+ = OpConstant {re.escape(u16_type.group(1))} (\d+)$",
            spirv,
            re.MULTILINE,
        )
    }
    assert {0x4B81, 0x3F80} <= u16_constants

    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        assembly.write_text(spirv)
        subprocess.run(
            ["spirv-as", "--target-env", "vulkan1.1", str(assembly), "-o", str(binary)],
            check=True,
        )
        subprocess.run(
            ["spirv-val", "--target-env", "vulkan1.1", str(binary)], check=True
        )
