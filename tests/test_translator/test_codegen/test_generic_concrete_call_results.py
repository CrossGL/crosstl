"""Generic calls deduce type parameters from the results of concrete helper calls.

``generic<T> T twice(T x)`` called as ``twice(first(v))`` binds ``T`` to the declared return type of the
concrete helper ``first``. SPIR-V registers function signatures only while it emits functions, so collection
could not type the helper call and refused the generic as unspecialized, while every other backend
translated it.
"""

import os
import shutil
import subprocess
import textwrap
from types import SimpleNamespace

import pytest

import crosstl
from crosstl.translator.ast import FunctionCallNode, IdentifierNode
from crosstl.translator.codegen.generic_concrete_call_results import (
    concrete_call_result_type_name,
    record_concrete_function_overloads,
)
from tests.spirv_friendly_names import friendly_spirv

BACKENDS = ["opengl", "directx", "metal", "vulkan"]

PROGRAM = textwrap.dedent("""
    shader main {
        generic<T> T twice(T x) { return x + x; }
        struct Params { float gain; int count; };
        float first(vec4 v) { return v.x; }
        int first(ivec4 v) { return v.y; }
        Params make(float g) {
            Params p;
            p.gain = g;
            p.count = 2;
            return p;
        }
        vec2 pair(float a, float b) { return vec2(a, b); }
        compute {
            RWStructuredBuffer<float> outs @buffer(0);
            StructuredBuffer<vec4> ins @buffer(1);
            @numthreads(1, 1, 1)
            void main(uint3 gid @gl_GlobalInvocationID) {
                vec4 v = ins[gid.x];
                ivec4 iv = ivec4(v);
                outs[gid.x] = %s;
            }
        }
    }
    """)

ARGUMENTS = {
    "scalar_result": "twice(first(v))",
    "overload_by_argument": "float(twice(first(iv)))",
    "struct_member": "twice(make(v.x).gain) + float(twice(make(v.y).count))",
    "vector_result": "twice(pair(v.x, v.y)).y",
    "nested_in_operator": "twice(first(v) * 2.0)",
}

COMBINED = " + ".join(f"({expr})" for expr in ARGUMENTS.values())


def _translate(tmp_path, expression, backend, format_output=False):
    path = tmp_path / "shader.cgl"
    path.write_text(PROGRAM % expression, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=format_output)


def _run(command, cwd):
    return subprocess.run(command, capture_output=True, text=True, check=False, cwd=cwd)


def _write(workdir, name, code):
    with open(os.path.join(workdir, name), "w", encoding="utf-8") as handle:
        handle.write(code)


def _compile(tmp_path, backend, code):
    """Compile ``code`` with the target's local toolchain, when it is installed."""
    workdir = os.path.realpath(tmp_path)
    if backend == "opengl" and shutil.which("glslangValidator"):
        _write(workdir, "out.comp", code)
        result = _run(["glslangValidator", "out.comp"], workdir)
        assert result.returncode == 0, result.stdout + result.stderr
    elif backend == "directx" and shutil.which("dxc"):
        _write(workdir, "out.hlsl", code)
        result = _run(
            ["dxc", "-T", "cs_6_0", "-E", "CSMain", "out.hlsl", "-Fo", "out.dxil"],
            workdir,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    elif backend == "metal" and shutil.which("xcrun"):
        _write(workdir, "out.metal", code)
        result = _run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", "out.metal", "-o", "out.air"],
            workdir,
        )
        if "unable to find utility" not in result.stderr:
            assert result.returncode == 0, result.stderr
    elif backend == "vulkan" and shutil.which("spirv-as") and shutil.which("spirv-val"):
        _write(workdir, "out.spvasm", code)
        result = _run(
            ["spirv-as", "--target-env", "vulkan1.1", "out.spvasm", "-o", "out.spv"],
            workdir,
        )
        assert result.returncode == 0, result.stderr
        result = _run(["spirv-val", "--target-env", "vulkan1.1", "out.spv"], workdir)
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", sorted(ARGUMENTS))
def test_helper_result_binds_the_type_parameter(tmp_path, case, backend):
    code = _translate(tmp_path, ARGUMENTS[case], backend)

    assert "OpFunctionCall" in code if backend == "vulkan" else "twice" in code


def test_spirv_specializes_for_each_helper_result(tmp_path):
    # Friendly ids (``%float``) whether or not SPIRV-Tools formats the output.
    code = friendly_spirv(_translate(tmp_path, COMBINED, "vulkan"))

    definitions = [line for line in code.splitlines() if "OpFunction " in line]
    float_results = [line for line in definitions if "OpFunction %float " in line]
    int_results = [line for line in definitions if "OpFunction %int " in line]
    vec2_results = [line for line in definitions if "OpFunction %v2float " in line]
    # ``first`` and ``twice`` for float and int; ``pair`` and ``twice`` for vec2.
    assert len(float_results) == 2, definitions
    assert len(int_results) == 2, definitions
    assert len(vec2_results) == 2, definitions
    assert "OpIAdd %int" in code
    _compile(tmp_path, "vulkan", code)


@pytest.mark.parametrize("backend", ["opengl", "directx", "metal"])
def test_every_helper_result_compiles(tmp_path, backend):
    code = _translate(tmp_path, COMBINED, backend)

    assert "twice_float(first(" in code
    assert "twice_int(first(" in code
    _compile(tmp_path, backend, code)


def _function(name, return_type, *param_types, default=None):
    params = [
        SimpleNamespace(param_type=param_type, default_value=None)
        for param_type in param_types
    ]
    if params and default is not None:
        params[-1].default_value = default
    return SimpleNamespace(name=name, return_type=return_type, parameters=params)


def _generator(*functions):
    generator = SimpleNamespace(
        type_name_string=lambda value: value, generic_function_definitions={}
    )
    record_concrete_function_overloads(generator, functions, lambda func: False)
    return generator


def _result(generator, name, arg_types):
    call = FunctionCallNode(
        IdentifierNode(name), [IdentifierNode(t) for t in arg_types]
    )
    return concrete_call_result_type_name(
        generator,
        call,
        lambda gen, arg: arg.name or None,
        lambda gen, type_name: type_name,
    )


def test_overloads_select_by_arity_and_known_argument_types():
    generator = _generator(
        _function("f", "float", "vec4"),
        _function("f", "int", "ivec4"),
        _function("f", "vec2", "float", "float", default="1.0"),
    )

    assert _result(generator, "f", ["vec4"]) == "float"
    assert _result(generator, "f", ["ivec4"]) == "int"
    assert _result(generator, "f", ["float", "float"]) == "vec2"


def test_ambiguous_overloads_and_void_helpers_stay_untyped():
    generator = _generator(
        _function("f", "float", "vec4"),
        _function("f", "int", "ivec4"),
        _function("g", "void", "float"),
    )

    assert _result(generator, "f", [""]) is None
    assert _result(generator, "g", ["float"]) is None
    assert _result(generator, "missing", ["float"]) is None


def test_generic_names_are_left_to_generic_deduction():
    generator = _generator(_function("f", "float", "float"))
    generator.generic_function_definitions = {"f": [object()]}

    assert _result(generator, "f", ["float"]) is None
