"""SPIR-V emits the specializations a function calls before the function, wherever the generics are declared.

The SPIR-V generator lowers a call only to a function it has already emitted, so it orders functions callee first.
A specialization's body still spells its calls by the generic name (``twice(y)`` in ``quad<float>``), which matched
none of the emitted functions (``twice_float``): when ``twice`` was declared after its caller, the call was refused
with "Cannot lower call to 'twice_float'". Each case below fails on that ordering.
"""

import os
import shutil
import subprocess
import sys
import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.generic_call_dependencies import generic_callee_names
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

# Every caller is declared before its callee: generic to generic, concrete to generic, and stage-local to generic.
PROGRAM = textwrap.dedent("""
    shader main {
        generic<T> T quad(T x) {
            T y = twice(x);
            return twice<T>(y) + octo(x);
        }
        float helper(float x) {
            return quad(x) + twice(x);
        }
        generic<T> T octo(T x) {
            return twice(twice(twice(x)));
        }
        generic<T> T twice(T x) {
            return x + x;
        }
        compute {
            RWStructuredBuffer<float> data @buffer(0);
            RWStructuredBuffer<int> ints @buffer(1);
            float local_first(float x) {
                return quad(x) + octo(x);
            }
            @numthreads(64, 1, 1)
            void main(uvec3 gid @ gl_GlobalInvocationID) {
                float f = data[gid.x];
                int v = ints[gid.x];
                data[gid.x] = helper(f) + local_first(f);
                ints[gid.x] = quad(v) + octo(v);
            }
        }
    }
""").strip()


def _validate(tmp_path, output):
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
    result = subprocess.run(
        [validator, "--target-env", "vulkan1.1", str(binary)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _function_count(output):
    return sum(" OpFunction " in line for line in output.splitlines())


def test_generics_called_before_their_declaration_lower_to_spirv(tmp_path):
    path = tmp_path / "forward.cgl"
    path.write_text(PROGRAM)

    output = crosstl.translate(str(path), backend="vulkan")

    # quad, octo and twice for float and int; helper; local_first; the entry point.
    assert _function_count(output) == 9
    assert "OpFunctionCall" in output
    _validate(tmp_path, output)


def test_each_function_records_the_specializations_it_calls():
    generator = VulkanSPIRVCodeGen()
    generator.generate(Parser(Lexer(PROGRAM).get_tokens()).parse())

    clones = {
        clone.name: clone
        for clone in generator.generic_function_specializations.values()
    }
    assert generic_callee_names(generator, clones["quad_float"]) == [
        "twice_float",
        "octo_float",
    ]
    assert generic_callee_names(generator, clones["octo_int"]) == ["twice_int"]
    assert generic_callee_names(generator, clones["twice_float"]) == []


def test_specializations_are_collected_in_call_order():
    generator = VulkanSPIRVCodeGen()
    generator.generate(Parser(Lexer(PROGRAM).get_tokens()).parse())

    # Creation order follows the calls, not the hash order of the specialization keys.
    names = [
        clone.name for clone in generator.generic_function_specializations.values()
    ]
    assert names == [
        "quad_int",
        "octo_int",
        "quad_float",
        "octo_float",
        "twice_float",
        "twice_int",
    ]


# One call creates six specializations of ``outer``, and each of them one of ``inner``.
FAN_OUT = textwrap.dedent("""
    shader main {
        generic<T> T inner(T x) { return x + x; }
        generic<T> T outer(T x) { return inner(x); }
        compute {
            RWStructuredBuffer<float> data @buffer(0);
            @numthreads(1, 1, 1)
            void main(uvec3 gid @ gl_GlobalInvocationID) {
                float a = data[0];
                int b = int(a);
                uint c = uint(a);
                vec2 d = vec2(a);
                vec3 e = vec3(a);
                ivec2 f = ivec2(b);
                data[0] = outer(a) + float(outer(b)) + float(outer(c)) + outer(d).x
                    + outer(e).x + float(outer(f).x);
            }
        }
    }
""").strip()


@pytest.mark.parametrize("backend", ["opengl", "vulkan"])
def test_specialization_order_does_not_depend_on_the_hash_seed(tmp_path, backend):
    path = tmp_path / "fan_out.cgl"
    path.write_text(FAN_OUT)
    script = (
        "import sys, crosstl; "
        "sys.stdout.write(crosstl.translate(sys.argv[1], backend=sys.argv[2]))"
    )
    outputs = set()
    for seed in ("0", "1", "2", "3"):
        result = subprocess.run(
            [sys.executable, "-c", script, str(path), backend],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            check=True,
        )
        outputs.add(result.stdout)
    assert len(outputs) == 1
