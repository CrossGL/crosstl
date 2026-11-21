"""SPIR-V output of fixed-array extents named by a constant of a CrossGL integer spelling.

CUDA and HIP input write ``const i32 N = 4;``, and a local ``var t: f32[N];`` needs ``N`` as a compile-time extent.
The extent resolver knew the C spellings (``int``, ``int32_t``, ``uint``) but not ``i32``, ``i64``, ``u32`` and
``u64``, so SPIR-V output refused the array as an unresolved extent.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.array_utils import collect_literal_int_constants
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

SOURCE = """\
const {type} N = 4;
@compute @workgroup_size(1, 1, 1)
fn k(@group(0) @binding(0) var<storage, read_write> o: array<f32>) {{
    var t: f32[N];
    t[0] = 1.0;
    t[3] = 2.0;
    o[0] = t[0] + t[3];
}}
"""


def validate(spv_code: str, tmp_path) -> None:
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        pytest.skip("spirv-as and spirv-val are not installed")
    asm_path = tmp_path / "shader.spvasm"
    spv_path = tmp_path / "shader.spv"
    asm_path.write_text(spv_code)
    for command in (
        [spirv_as, "--target-env", "vulkan1.0", str(asm_path), "-o", str(spv_path)],
        [spirv_val, "--target-env", "vulkan1.0", str(spv_path)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("type_name", ["i32", "i64", "u32", "u64"])
def test_constant_is_collected(type_name):
    program = Parser(Lexer(SOURCE.format(type=type_name)).tokens).parse()
    assert collect_literal_int_constants(program.constants)["N"] == 4


@pytest.mark.parametrize("type_name", ["i32", "u32", "int"])
def test_local_array_extent(type_name, tmp_path):
    path = tmp_path / "extent.cgl"
    path.write_text(SOURCE.format(type=type_name))
    spv = crosstl.translate(str(path), backend="vulkan", format_output=False)
    array = re.search(r"= OpTypeArray %\d+ (%\d+)\n", spv)
    assert array is not None
    length = re.search(rf"{array.group(1)} = OpConstant %\d+ (\d+)\n", spv)
    assert length is not None and length.group(1) == "4"
    validate(spv, tmp_path)
