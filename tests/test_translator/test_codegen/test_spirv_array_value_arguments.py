"""Array values passed to a by-value array parameter go through a Function variable in the generated SPIR-V.

A CrossGL by-value array parameter lowers to an ``OpTypePointer Function`` parameter that the callee copies on
entry. ``OpFunctionCall`` needs each argument to match its parameter type (SPIR-V 1.6, ``OpFunctionCall``), so an
array that is a value, such as a program-scope ``const`` array (an ``OpConstantComposite``), an element of a
constant array of arrays, a constant struct member, an array literal or a call result, has to be stored into a
Function variable first. The generator used to pass the value itself, which ``spirv-val`` rejects.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen

SOURCE = """
shader ArrayValueArguments {
    struct S {
        float a[4];
    };

    const float[4] W = { 1.0, 2.0, 3.0, 4.0 };
    const float[2][2] M = { { 5.0, 6.0 }, { 7.0, 8.0 } };
    const S CS = S({ 9.0, 10.0, 11.0, 12.0 });

    layout(std430, binding = 0) buffer Dst {
        float dst[8];
    } out_data;

    compute {
        layout(local_size_x = 1) in;

        float sum4(float v[4]) {
            float s = 0.0;
            for (int i = 0; i < 4; i++) {
                s += v[i] * float(i + 1);
            }
            v[0] = 100.0;
            return s + v[0];
        }

        float sum2(float v[2]) {
            v[1] = -v[1];
            return v[0] * 3.0 + v[1];
        }

        float[4] make(float k) {
            return { k, k + 1.0, k + 2.0, k + 3.0 };
        }

        void main() {
            const float L[4] = { 13.0, 14.0, 15.0, 16.0 };
            out_data.dst[0] = sum4(W);
            out_data.dst[1] = sum2(M[1]);
            out_data.dst[2] = sum4(CS.a);
            out_data.dst[3] = sum4({ 17.0, 18.0, 19.0, 20.0 });
            out_data.dst[4] = sum4(make(21.0));
            out_data.dst[5] = sum4(L);
            out_data.dst[6] = W[0] + M[1][1] + CS.a[3] + L[2];
            out_data.dst[7] = sum2(M[0]);
        }
    }
}
"""


def _generate():
    return VulkanSPIRVCodeGen().generate(crosstl.translator.parse(SOURCE))


def _function_variables(assembly):
    return set(re.findall(r"%(\w+) = OpVariable %\w+ Function\b", assembly))


def _helper_call_arguments(assembly):
    """The argument ids of every call to a function that takes a Function-pointer parameter."""
    pointer_types = set(re.findall(r"%(\w+) = OpTypePointer Function %\w+", assembly))
    pointer_functions = set()
    current = None
    for line in assembly.splitlines():
        line = line.strip()
        opened = re.match(r"%(\w+) = OpFunction\b", line)
        parameter = re.match(r"%\w+ = OpFunctionParameter %(\w+)", line)
        if opened:
            current = opened.group(1)
        elif parameter and parameter.group(1) in pointer_types:
            pointer_functions.add(current)
    calls = re.findall(r"= OpFunctionCall %\w+ %(\w+)((?: %\w+)*)", assembly)
    return [
        argument.lstrip("%")
        for callee, arguments in calls
        if callee in pointer_functions
        for argument in arguments.split()
    ]


def test_every_by_value_array_argument_is_a_function_variable():
    assembly = _generate()
    arguments = _helper_call_arguments(assembly)
    assert len(arguments) == 7, assembly
    variables = _function_variables(assembly)
    assert all(argument in variables for argument in arguments), assembly


def test_program_scope_constant_array_is_stored_before_the_call():
    assembly = _generate()
    constants = set(re.findall(r"%(\w+) = OpConstantComposite\b", assembly))
    call = re.search(r"= OpFunctionCall %\w+ %\w+ %(\w+)", assembly)
    assert call, assembly
    stores = re.findall(rf"OpStore %{call.group(1)} %(\w+)", assembly[: call.start()])
    assert stores and stores[-1] in constants, assembly


def test_array_value_argument_module_validates(tmp_path):
    assembly = _generate()
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    source = tmp_path / "shader.spvasm"
    binary = tmp_path / "shader.spv"
    source.write_text(assembly, encoding="utf-8")
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
