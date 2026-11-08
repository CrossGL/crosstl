"""Value parameters of CrossGL generic functions are substituted into each specialization.

``generic<T, N> T scale(T x) { return x * T(N); }`` called as ``scale<float, 3>(x)`` binds ``N`` to the value ``3``.
The monomorphizing generators cloned the function and substituted only its types, so the specialization read the
undeclared ``N`` (``float(N)``, ``float values[N]``), and SPIR-V refused it. An explicit argument that reads an
enclosing value parameter (``accumulate<T, N + 1>``) kept ``N`` in the callee, and WGSL read ``true`` as a type.
Each case below fails on that code.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.enum_utils import substitute_generic_type_name
from crosstl.translator.codegen.generic_value_arguments import (
    value_argument_expression,
)
from tests.spirv_friendly_names import friendly_spirv

PROGRAM = textwrap.dedent("""
    shader main {
        const int K = 4;
        generic<T, N> T accumulate(T x) {
            T values[N];
            T total = T(0);
            for (int i = 0; i < N; i++) {
                values[i] = x * T(i + 1);
                total += values[i];
            }
            return total;
        }
        generic<T, N> T scale(T x) {
            return x * T(N);
        }
        generic<T, N> T shifted(T x) {
            return accumulate<T, N + 1>(x) + scale<T, -N>(x) + T(N * 2);
        }
        generic<T, B> T pick(T a, T b) {
            if (B) {
                return a;
            }
            return b;
        }
        compute {
            RWStructuredBuffer<float> data @buffer(0);
            @numthreads(64, 1, 1)
            void main(uvec3 gid @ gl_GlobalInvocationID) {
                float x = data[gid.x];
                float a = shifted<float, 2>(x);
                float b = accumulate<float, K>(x);
                float c = pick<float, true>(a, b) + pick<float, false>(a, b);
                data[gid.x] = a + b + c + accumulate<float, 3>(x);
            }
        }
    }
""").strip()

TARGETS = ["opengl", "metal", "directx", "vulkan", "wgsl", "slang"]


def translate(tmp_path, code, backend):
    path = tmp_path / "values.cgl"
    path.write_text(code)
    output = crosstl.translate(str(path), backend=backend, format_output=False)
    # Friendly SPIR-V ids (``%int``) whether or not SPIRV-Tools formats the output.
    return friendly_spirv(output) if backend == "vulkan" else output


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


@pytest.mark.parametrize("backend", ["opengl", "metal", "directx", "slang"])
def test_value_parameters_are_substituted_into_c_style_specializations(
    tmp_path, backend
):
    output = translate(tmp_path, PROGRAM, backend)

    assert "values[3];" in output and "values[K];" in output
    assert "values[N]" not in output and "(N)" not in output
    # ``accumulate<T, N + 1>`` in ``shifted<float, 2>`` instantiates ``accumulate`` for ``2 + 1``.
    assert "values[(2 + 1)];" in output or "values[2 + 1];" in output
    assert "i < 3" in output and "i < K" in output
    assert "(-2)" in output and "(2 * 2)" in output
    # ``if (B)`` reads ``true`` and ``false``, which some generators fold away.
    assert "(B)" not in output
    _validate(tmp_path, backend, output)


def test_value_parameters_are_substituted_into_wgsl_specializations(tmp_path):
    output = translate(tmp_path, PROGRAM, "wgsl")

    assert "array<f32, 3>" in output and "array<f32, K>" in output
    assert "array<f32, (2 + 1)>" in output
    assert "fn pick_f32_true(" in output and "if (true)" in output
    assert "true_" not in output and "N_1" not in output
    _validate(tmp_path, "wgsl", output)


def test_value_parameters_lower_to_spirv_constants(tmp_path):
    output = translate(tmp_path, PROGRAM, "vulkan")

    assert "OpConstant %int 3" in output and "OpConstant %int 4" in output
    assert "OpSNegate %int" in output
    # accumulate for 3, K and 2 + 1; scale for -2; shifted; pick for true and false; the entry point.
    assert sum(" OpFunction " in line for line in output.splitlines()) == 8
    _validate(tmp_path, "vulkan", output)


@pytest.mark.parametrize("backend", ["opengl", "metal", "directx", "slang"])
def test_distinct_values_instantiate_distinct_specializations(tmp_path, backend):
    output = translate(tmp_path, PROGRAM, backend)

    names = ["accumulate_float_3", "accumulate_float_K", "accumulate_float_2_1"]
    names += ["pick_float_true", "pick_float_false", "shifted_float_2"]
    for name in names:
        assert name in output, name


TYPE_AS_VALUE = textwrap.dedent("""
    shader main {
        generic<T, N> T scale(T x) {
            return x * T(N);
        }
        fragment {
            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                return vec4(scale<float, float>(uv.x));
            }
        }
    }
""").strip()


@pytest.mark.parametrize("backend", ["opengl", "metal", "directx", "vulkan"])
def test_a_value_parameter_bound_to_a_type_is_refused(tmp_path, backend):
    with pytest.raises(ValueError, match="'N' of generic function 'scale'"):
        translate(tmp_path, TYPE_AS_VALUE, backend)


def test_value_arguments_parse_as_expressions_and_types_do_not():
    assert value_argument_expression("3").value == 3
    assert value_argument_expression("true").value is True
    assert value_argument_expression("K").name == "K"
    assert value_argument_expression("2 + 1") is not None
    assert value_argument_expression("float") is None
    assert value_argument_expression("vec3<float>") is None


def test_explicit_arguments_substitute_enclosing_value_parameters():
    substitutions = {"T": "float", "N": "2"}

    assert substitute_generic_type_name("accumulate<T, N + 1>", substitutions) == (
        "accumulate<float, 2 + 1>"
    )
    assert substitute_generic_type_name("scale<T, -N>", substitutions) == (
        "scale<float, -2>"
    )
    assert substitute_generic_type_name("T[N]", substitutions) == "float[2]"
    # A member or a qualified name that spells a parameter is not the parameter.
    assert substitute_generic_type_name("f<s.N, ns::N>", substitutions) == (
        "f<s.N, ns::N>"
    )
    assert substitute_generic_type_name("f<N * 2>", {"N": "1 + 1"}) == (
        "f<(1 + 1) * 2>"
    )
