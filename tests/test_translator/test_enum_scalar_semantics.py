"""CrossGL enum values are integers: they compare as scalars and initialize static locals.

The semantic passes read every entry of ``program.structs`` as a structure, enums included, so ``k == Kind::Y``
compared an "aggregate" with an operand of unknown type and Metal, HLSL and SPIR-V refused it; and an enumerator was
no constant expression, so ``static Kind fixed = Kind::Y;`` was refused on every target. Each case below fails on
that code.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.enum_values import (
    enumerator_names,
    scalar_enum_names,
)

PROGRAM = textwrap.dedent("""
    shader main {
        enum Kind {
            X,
            Y = 4,
        }
        compute {
            RWStructuredBuffer<int> data @buffer(0);
            @numthreads(1, 1, 1)
            void main(uvec3 gid @ gl_GlobalInvocationID) {
                static Kind fixed = Kind::Y;
                Kind k = data[1] > 0 ? Kind::Y : Kind::X;
                bool same = k == Kind::Y;
                bool other = k != fixed;
                data[0] = (same ? 1 : 0) + (other ? 2 : 0);
            }
        }
    }
""").strip()

TARGETS = ["opengl", "metal", "directx", "vulkan", "wgsl", "slang"]


def _validate(tmp_path, backend, output):
    if backend == "opengl":
        tool = shutil.which("glslangValidator")
        if tool is None:
            pytest.skip("glslangValidator is not installed")
        path = tmp_path / "out.comp"
        path.write_text(output)
        command = [tool, "-S", "comp", str(path)]
    elif backend == "vulkan":
        assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
        if assembler is None or validator is None:
            pytest.skip("SPIRV-Tools are not installed")
        path, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
        path.write_text(output)
        subprocess.run(
            [assembler, "--target-env", "vulkan1.1", str(path), "-o", str(binary)],
            check=True,
            capture_output=True,
        )
        command = [validator, "--target-env", "vulkan1.1", str(binary)]
    elif backend == "wgsl":
        tool = shutil.which("naga")
        if tool is None:
            pytest.skip("naga is not installed")
        path = tmp_path / "out.wgsl"
        path.write_text(output)
        command = [tool, str(path)]
    elif backend == "metal":
        xcrun = shutil.which("xcrun")
        if xcrun is None:
            pytest.skip("xcrun is not installed")
        path = tmp_path / "out.metal"
        path.write_text(output)
        command = [xcrun, "-sdk", "macosx", "metal", "-c", str(path), "-o", "/dev/null"]
    elif backend == "directx":
        tool = shutil.which("dxc")
        if tool is None:
            pytest.skip("dxc is not installed")
        path = tmp_path / "out.hlsl"
        path.write_text(output)
        command = [tool, "-HV", "2021", "-T", "cs_6_2", "-E", "CSMain", str(path)]
    else:
        tool = shutil.which("slangc")
        if tool is None:
            pytest.skip("slangc is not installed")
        path = tmp_path / "out.slang"
        path.write_text(output)
        command = [tool, str(path), "-target", "spirv", "-o", str(tmp_path / "s.spv")]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", TARGETS)
def test_enum_comparisons_and_static_enum_locals_translate(tmp_path, backend):
    path = tmp_path / "enums.cgl"
    path.write_text(PROGRAM)

    output = crosstl.translate(str(path), backend=backend)

    _validate(tmp_path, backend, output)


def test_glsl_compares_the_enum_integers(tmp_path):
    path = tmp_path / "enums.cgl"
    path.write_text(PROGRAM)

    output = crosstl.translate(str(path), backend="opengl")

    assert "int main_fixed = Kind_Y;" in output
    assert "(k == Kind_Y)" in output and "(k != main_fixed)" in output


def test_enumerators_and_scalar_enums_are_listed():
    program = Parser(Lexer(PROGRAM).get_tokens()).parse()

    assert scalar_enum_names(program) == {"Kind"}
    assert enumerator_names(program) == {"Kind::X", "Kind::Y"}
