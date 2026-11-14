"""Generic calls deduce type parameters from elements, members, constants and other generic calls' results.

``generic<T> T twice(T x)`` binds ``T`` from the argument. Before these fixes:

* SPIR-V refused ``twice(weights[i])``, ``twice(axis)`` and ``twice(params.gain)`` on globals, because collection
  read neither the program's global types nor cbuffer members, and lowered pointers before any were declared;
* Metal typed the buffer parameter of ``uniform vec3 lights[N]`` as ``float3``, so ``lights[i]`` read as a
  component and the call needed a ``float`` specialization that was never collected;
* OpenGL refused element arguments of stage-scope uniform arrays, and neither OpenGL nor SPIR-V typed program
  constants such as ``const float LUT[3]``;
* a call on another generic call's result, as in ``twice(twice(v).y)``, read the inner definition's return type
  ``T`` and specialized ``T twice_T(T x)``, which Metal and HLSL emitted verbatim and OpenGL refused.
"""

import os
import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl._crosstl import _translate_with_codegen
from crosstl.translator.codegen.builtin_result_types import swizzle_access_type
from crosstl.translator.codegen.generic_call_results import (
    generic_call_argument_type_name,
)

BACKENDS = ["opengl", "directx", "metal", "vulkan"]

PROGRAM = textwrap.dedent("""
    shader main {
        const int N = 3;
        const float LUT[3] = {1.0, 2.0, 3.0};
        const float GAIN = 0.5;
        generic<T> T twice(T x) { return x + x; }
        generic<T> T pick(T a, T b, bool c) { return c ? a : b; }
        struct Params { float gain; int count; vec3 dir; };
        uniform float weights[4];
        uniform vec3 lights[N];
        uniform Params params;
        uniform vec3 axis;
        uniform mat3 basis;
        compute {
            RWStructuredBuffer<float> outs @buffer(0);
            uniform int counts[2];
            uniform ivec2 offs[N];
            @numthreads(1, 1, 1)
            void main(uint3 gid @gl_GlobalInvocationID) {
                int i = int(gid.x) & 1;
                float arr[4];
                arr[0] = 1.0; arr[1] = 2.0; arr[2] = 3.0; arr[3] = 4.0;
                vec3 v = vec3(1.0, 2.0, 3.0);
                mat3 m = mat3(1.0);
                Params p;
                p.gain = 2.0; p.count = 3; p.dir = v;
                outs[gid.x] = %s;
            }
        }
    }
    """)

ARGUMENTS = {
    "local_array_element": "twice(arr[i]) + twice(arr[2])",
    "local_vector_component": "twice(v.y) + twice(v[i])",
    "local_swizzle": "twice(v.xy).x",
    "local_matrix_column": "twice(m[i]).y",
    "local_struct_members": "twice(p.gain) + float(twice(p.count)) + twice(p.dir).z",
    "uniform_array_element": "twice(weights[i]) + twice(weights[1])",
    "uniform_constant_extent_element": "twice(lights[i]).x",
    "stage_uniform_array_element": "float(twice(counts[i])) + float(twice(offs[i]).y)",
    "uniform_struct_members": (
        "twice(params.gain) + float(twice(params.count))" " + twice(params.dir).x"
    ),
    "uniform_vector": "twice(axis).x + twice(axis.z) + twice(v).y",
    "uniform_matrix_column": "twice(basis[i]).x",
    "two_parameters": (
        "pick(lights[i], lights[0], i > 0).z" " + float(pick(offs[i].x, 2, true))"
    ),
    "constant_array_element": "twice(LUT[i]) + twice(LUT[1])",
    "scalar_constant": "twice(GAIN)",
    "nested_component": "twice(twice(lights[i]).y)",
    "nested_swizzle": "twice(twice(lights[i]).xy).y",
    "nested_operator": "twice(twice(lights[i]) * 2.0).z",
    "nested_ternary": "twice(i > 0 ? twice(lights[i]) : lights[0]).x",
    "nested_integer_vector": "float(twice(twice(offs[i]).x))",
    "nested_scalar_constant": "twice(twice(LUT[1]))",
    "nested_local": "twice(twice(vec3(1.0, 2.0, float(i))).y)",
}

COMBINED = " + ".join(f"({expr})" for expr in ARGUMENTS.values())

_DEFINITION = re.compile(r"\b(?:twice|pick)_(\w+)\s*\(")


def _translate(tmp_path, expression, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(PROGRAM % expression, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(command, cwd):
    return subprocess.run(command, capture_output=True, text=True, check=False, cwd=cwd)


def _compile(tmp_path, backend, code):
    """Compile ``code`` with the target's local toolchain, when it is installed."""
    workdir = os.path.realpath(tmp_path)
    if backend == "opengl" and shutil.which("glslangValidator"):
        source = os.path.join(workdir, "out.comp")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write(code)
        result = _run(["glslangValidator", source], workdir)
        assert result.returncode == 0, result.stdout + result.stderr
    elif backend == "directx" and shutil.which("dxc"):
        source = os.path.join(workdir, "out.hlsl")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write(code)
        result = _run(
            ["dxc", "-T", "cs_6_0", "-E", "CSMain", source, "-Fo", "out.dxil"],
            workdir,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    elif backend == "metal" and shutil.which("xcrun"):
        source = os.path.join(workdir, "out.metal")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write(code)
        result = _run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", source, "-o", "out.air"],
            workdir,
        )
        if "unable to find utility" not in result.stderr:
            assert result.returncode == 0, result.stderr
    elif backend == "vulkan" and shutil.which("spirv-as") and shutil.which("spirv-val"):
        assembly = os.path.join(workdir, "out.spvasm")
        with open(assembly, "w", encoding="utf-8") as handle:
            handle.write(code)
        result = _run(
            ["spirv-as", "--target-env", "vulkan1.1", assembly, "-o", "out.spv"],
            workdir,
        )
        assert result.returncode == 0, result.stderr
        result = _run(["spirv-val", "--target-env", "vulkan1.1", "out.spv"], workdir)
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", sorted(ARGUMENTS))
def test_argument_binds_the_type_parameter(tmp_path, case, backend):
    code = _translate(tmp_path, ARGUMENTS[case], backend)

    if backend == "vulkan":
        assert "OpFunctionCall" in code
        return
    suffixes = set(_DEFINITION.findall(code))
    assert suffixes, code
    assert "T" not in suffixes, code


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_argument_form_compiles(tmp_path, backend):
    code = _translate(tmp_path, COMBINED, backend)

    _compile(tmp_path, backend, code)


@pytest.mark.parametrize(
    "backend, vector, integer_vector",
    [
        ("opengl", "vec3", "ivec2"),
        ("directx", "float3", "int2"),
        ("metal", "float3", "int2"),
    ],
)
def test_nested_calls_specialize_for_the_inner_result(
    tmp_path, backend, vector, integer_vector
):
    code = _translate(
        tmp_path,
        ARGUMENTS["nested_component"] + " + " + ARGUMENTS["nested_integer_vector"],
        backend,
    )

    assert set(_DEFINITION.findall(code)) == {"float", vector, "int", integer_vector}
    assert f"twice_float(twice_{vector}(" in code
    assert f"twice_int(twice_{integer_vector}(" in code


@pytest.mark.parametrize(
    "type_name, length, expected",
    [
        ("vec3", 2, "vec2"),
        ("ivec3", 2, "ivec2"),
        ("uvec4", 3, "uvec3"),
        ("float3", 2, "float2"),
        ("int4", 1, "int"),
        ("float", 2, None),
        ("mat3", 2, None),
        ("Params", 2, None),
    ],
)
def test_swizzle_access_type_keeps_the_family_spelling(type_name, length, expected):
    assert swizzle_access_type(type_name, length) == expected


class _Generic:
    def __init__(self, *names):
        self.generic_params = [type("Param", (), {"name": name})() for name in names]


class _Generator:
    generic_function_definitions = {
        "twice": [_Generic("T")],
        "pair": [_Generic("A", "B")],
    }


@pytest.mark.parametrize(
    "fallback, actual, expected",
    [
        (None, "float", "float"),
        ("float", "T", "float"),
        ("vec3", None, "vec3"),
        ("T", "T", None),
        (None, "vector<T, 3>", None),
        ("v3float", "float", "float"),
    ],
)
def test_argument_types_spelled_with_an_open_parameter_are_discarded(
    fallback, actual, expected
):
    assert (
        generic_call_argument_type_name(_Generator(), ["twice"], fallback, actual)
        == expected
    )


def test_open_parameters_of_every_read_definition_are_discarded():
    generator = _Generator()

    assert generic_call_argument_type_name(generator, ["pair"], "B", "A") is None
    assert generic_call_argument_type_name(generator, ["pair"], None, "T") == "T"


@pytest.mark.parametrize(
    "setup, expr, expected_func",
    [
        ("float a[2]; a[0] = 1.0; a[1] = 2.0;", "pick(a[0])", "pick_float"),
        ("vec4 v = vec4(1.0, 2.0, 3.0, 4.0);", "pick(v[2])", "pick_float"),
        ("mat4 m = mat4(1.0);", "pick(m[1]).x", "pick_v4float"),
        ("float a[2][3]; a[1][2] = 3.0;", "pick(a[1][2])", "pick_float"),
        (
            "vec2 arr[2]; arr[0] = vec2(1.0); arr[1] = vec2(2.0);",
            "sel2(arr[0], arr[1]).x",
            "sel2_v2float",
        ),
        ("", "pick(outp[0])", "pick_float"),
    ],
)
def test_spirv_indexed_generic_call_arguments(tmp_path, setup, expr, expected_func):
    shader = f"""
    shader main {{
        generic<T> T pick(T x) {{ return x; }}
        generic<T> T sel2(T a, T b) {{ return a + b; }}
        compute {{
            RWStructuredBuffer<float> outp @buffer(0);
            @numthreads(1, 1, 1)
            void main(uint3 gid @gl_GlobalInvocationID) {{
                {setup}
                int i = int(gid.x);
                outp[i] = {expr};
            }}
        }}
    }}
    """
    path = tmp_path / "shader.cgl"
    path.write_text(shader, encoding="utf-8")
    code, codegen = _translate_with_codegen(str(path), backend="vulkan")

    assert "OpFunctionCall" in code
    specialized_names = set(codegen.generic_function_specialized_names.values())
    assert expected_func in specialized_names
    assert not codegen.refusals.reasons
    _compile(tmp_path, "vulkan", code)


def test_issue_3823_repro_shader(tmp_path):
    shader = """
    shader main {
        generic<T> T pick(T x) { return x; }
        generic<T> T sel2(T a, T b) { return a + b; }
        compute {
            RWStructuredBuffer<float> outp @buffer(0);
            @numthreads(1, 1, 1)
            void main(uint3 gid @gl_GlobalInvocationID) {
                float a[2];
                a[0] = 1.0;
                a[1] = 2.0;
                int i = int(gid.x);
                outp[i] = pick(a[0]) + sel2(a[0], a[1]);
            }
        }
    }
    """
    path = tmp_path / "shader.cgl"
    path.write_text(shader, encoding="utf-8")
    code, codegen = _translate_with_codegen(str(path), backend="vulkan")

    assert "OpFunctionCall" in code
    specialized_names = set(codegen.generic_function_specialized_names.values())
    assert "pick_float" in specialized_names
    assert "sel2_float" in specialized_names
    assert not codegen.refusals.reasons
    _compile(tmp_path, "vulkan", code)


def test_spirv_generic_calls_on_buffer_pointer(tmp_path):
    shader = """
    shader main {
        generic<T> T pick(T x) { return x; }
        generic<T> T sel2(T a, T b) { return a + b; }
        compute {
            buffer float* outp;
            @numthreads(1, 1, 1)
            void main(uint3 gid @gl_GlobalInvocationID) {
                float x = 1.0;
                int i = int(gid.x);
                outp[i] = pick(outp[0]) + sel2(outp[0], x);
            }
        }
    }
    """
    path = tmp_path / "shader.cgl"
    path.write_text(shader, encoding="utf-8")
    code, codegen = _translate_with_codegen(str(path), backend="vulkan")

    assert "OpFunctionCall" in code
    specialized_names = set(codegen.generic_function_specialized_names.values())
    assert "pick_float" in specialized_names
    assert "sel2_float" in specialized_names
    assert not codegen.refusals.reasons
    _compile(tmp_path, "vulkan", code)


ISSUE_3762_SHADERS = {
    "1_pick_float": (
        """
    shader P {
        T pick<T>(T x) { return x; }
        buffer float* outp;
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                float x = 1.0;
                int i = 0;
                outp[i] = pick(pick(x));
            }
        }
    }
    """
    ),
    "2_sel2_float": (
        """
    shader P {
        T sel2<T>(T a, T b) { return a; }
        buffer float* outp;
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                float x = 1.0;
                int i = 0;
                outp[i] = sel2(sel2(x, x), x);
            }
        }
    }
    """
    ),
    "3_sel2_vec2": (
        """
    shader P {
        T sel2<T>(T a, T b) { return a; }
        buffer float* outp;
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                vec2 w = vec2(1.0, 2.0);
                int i = 0;
                outp[i] = sel2(sel2(w, w), w).x;
            }
        }
    }
    """
    ),
    "4_sel_overload": (
        """
    shader P {
        T sel<T>(T a) { return a; }
        T sel<T>(T a, T b) { return b; }
        buffer float* outp;
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                vec2 w = vec2(1.0, 2.0);
                int i = 0;
                outp[i] = sel(sel(w, w), w).x;
            }
        }
    }
    """
    ),
    "5_pick_vec3": (
        """
    shader P {
        T pick<T>(T x) { return x; }
        buffer float* outp;
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                vec3 v = vec3(1.0, 2.0, 3.0);
                int i = 0;
                outp[i] = pick(pick(v)).x;
            }
        }
    }
    """
    ),
}

ISSUE_3762_EXPECTED = {
    "1_pick_float": {
        "opengl": "pick_float",
        "directx": "pick_float",
        "metal": "pick_float",
        "slang": "pick_float",
        "cuda": "pick_float",
        "hip": "pick_float",
        "wgsl": "pick_f32",
        "vulkan": "pick_float",
    },
    "2_sel2_float": {
        "opengl": "sel2_float",
        "directx": "sel2_float",
        "metal": "sel2_float",
        "slang": "sel2_float",
        "cuda": "sel2_float",
        "hip": "sel2_float",
        "wgsl": "sel2_f32",
        "vulkan": "sel2_float",
    },
    "3_sel2_vec2": {
        "opengl": "sel2_vec2",
        "directx": "sel2_float2",
        "metal": "sel2_float2",
        "slang": "sel2_vec2",
        "cuda": "sel2_vec2",
        "hip": "sel2_float2",
        "wgsl": "sel2_vec2_f32",
        "vulkan": "sel2_v2float",
    },
    "4_sel_overload": {
        "opengl": "sel_vec2_overload1",
        "directx": "sel_float2_overload1",
        "metal": "sel_float2_overload1",
        "slang": "sel_vec2_overload1",
        "cuda": "sel_vec2_overload1",
        "hip": "sel_float2_overload1",
        "wgsl": "sel_vec2_f32",
        "vulkan": "sel_v2float_overload1",
    },
    "5_pick_vec3": {
        "opengl": "pick_vec3",
        "directx": "pick_float3",
        "metal": "pick_float3",
        "slang": "pick_vec3",
        "cuda": "pick_vec3",
        "hip": "pick_float3",
        "wgsl": "pick_vec3_f32",
        "vulkan": "pick_v3float",
    },
}

ISSUE_3762_BACKENDS = [
    "opengl",
    "directx",
    "metal",
    "slang",
    "cuda",
    "hip",
    "wgsl",
    "vulkan",
]


@pytest.mark.parametrize("backend", ISSUE_3762_BACKENDS)
@pytest.mark.parametrize("case_name", sorted(ISSUE_3762_SHADERS))
def test_issue_3762_nested_generic_calls(tmp_path, case_name, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(ISSUE_3762_SHADERS[case_name], encoding="utf-8")

    expected_func = ISSUE_3762_EXPECTED[case_name][backend]

    if backend == "vulkan":
        code, codegen = _translate_with_codegen(str(path), backend="vulkan")
        assert "OpFunctionCall" in code
        specialized_names = set(codegen.generic_function_specialized_names.values())
        assert expected_func in specialized_names
        assert not any(
            name.endswith("_T") or "_T_overload" in name for name in specialized_names
        )
        assert not codegen.refusals.reasons
        _compile(tmp_path, "vulkan", code)
    else:
        code = crosstl.translate(str(path), backend=backend, format_output=False)
        assert not re.search(r"\b(pick|sel2|sel)_T(_overload\d+)?\b", code), code
        assert not re.search(r"\bT\s+(?:pick|sel2|sel)_T\b", code), code
        assert (
            expected_func in code
        ), f"Expected {expected_func} in {backend} output:\n{code}"
        _compile(tmp_path, backend, code)
