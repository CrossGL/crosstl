"""A generic overload matches a call only when the argument count lies between its parameters without defaults and all its parameters.

A call with fewer or more arguments than a generic overload takes used to be sent to that overload anyway: the
candidate match ranked by ``-arity_distance`` but never rejected a count no parameter list allows, so Metal, HLSL,
GLSL, Slang, CUDA and HIP output called a two-parameter clone such as ``inc_float(x)`` with one argument, which
their compilers reject. The call now goes to an overload that takes it, or is refused with a typed diagnostic. A
parameter with a default value still keeps working: ``scale(x)`` for ``scale<T>(T v, float k = 2.0)`` compiles.
"""

import pytest

import crosstl

C_STYLE_TARGETS = ["directx", "opengl", "metal", "cuda", "hip", "slang", "mojo"]

FEWER_ARGS = """
shader P {
    buffer float* outp;
    void inc(inout float value) { value = value + 1.0; }
    void inc<T>(inout T value, T step) { value = value + step; }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float x = outp[i];
            inc(x);
            outp[i] = x;
        }
    }
}
"""

FEWER_ARGS_NO_CONCRETE = FEWER_ARGS.replace(
    "    void inc(inout float value) { value = value + 1.0; }\n", ""
)

MORE_ARGS = """
shader P {
    buffer float* outp;
    buffer float2* outp2;
    vec2 second(float a, vec2 b, float c) { return b; }
    U second<T, U>(T a, U b) { return b; }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float x = outp[i];
            vec2 w = outp2[i];
            outp2[i] = second(x, w, x);
        }
    }
}
"""

DEFAULTED_PARAMETER = """
shader P {
    buffer float* outp;
    T scale<T>(T v, float k = 2.0) { return v * T(k); }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float x = outp[i];
            outp[i] = scale(x);
        }
    }
}
"""

TWO_GENERIC_OVERLOADS = """
shader P {
    buffer float2* outp;
    void inc<T>(inout T value) { value = value + T(1.0); }
    void inc<T>(inout T value, T step) { value = value + step; }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float2 w = outp[i];
            inc(w, w);
            outp[i] = w;
        }
    }
}
"""


def translate(tmp_path, source, backend):
    path = tmp_path / "program.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize("backend", C_STYLE_TARGETS)
def test_fewer_arguments_fall_back_to_concrete_overload(tmp_path, backend):
    code = translate(tmp_path, FEWER_ARGS, backend)
    assert "inc(x)" in code
    assert "inc_float" not in code


# CUDA and HIP rename a user function that a call uses with an argument count none of its overloads takes, so
# that a built-in of the same name stays reachable (see the clamp test below); the call then keeps the old name.
@pytest.mark.parametrize(
    "backend",
    [backend for backend in C_STYLE_TARGETS if backend not in {"cuda", "hip"}],
)
def test_fewer_arguments_without_concrete_are_refused(tmp_path, backend):
    with pytest.raises(Exception, match="inc") as error:
        translate(tmp_path, FEWER_ARGS_NO_CONCRETE, backend)
    assert "generic" in str(error.value)


BUILTIN_BESIDE_GENERIC = """
shader P {
    buffer float* outp;
    T clamp<T>(T v) { return v; }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float x = outp[i];
            outp[i] = clamp(x, 0.0, 1.0) + clamp(x);
        }
    }
}
"""


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_builtin_call_beside_a_generic_of_the_same_name_reaches_the_builtin(
    tmp_path, backend
):
    # clamp(x, 0.0, 1.0) fits no overload of the user clamp<T>(T v), so it is the built-in.
    code = translate(tmp_path, BUILTIN_BESIDE_GENERIC, backend)
    assert "fmaxf(0.0f, fminf(1.0f, x))" in code
    assert "clamp_fn_float(x)" in code


@pytest.mark.parametrize("backend", ["directx", "opengl", "metal"])
def test_more_arguments_fall_back_to_concrete_overload(tmp_path, backend):
    code = translate(tmp_path, MORE_ARGS, backend)
    assert "second(x, w, x)" in code
    assert "second_float" not in code


@pytest.mark.parametrize("backend", ["directx", "opengl", "metal"])
def test_defaulted_parameter_keeps_working(tmp_path, backend):
    code = translate(tmp_path, DEFAULTED_PARAMETER, backend)
    assert "scale_float" in code


TOO_MANY_FOR_EVERY_OVERLOAD = """
shader P {
    buffer float2* outp;
    void inc<T>(inout T value) { value = value + T(1.0); }
    void inc<T>(inout T value, T step) { value = value + step; }
    void inc<T>(inout T value, T step, T scale) { value = value + step * scale; }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float2 w = outp[i];
            inc(w, w, w, w);
            outp[i] = w;
        }
    }
}
"""


# SPIR-V output used to inline the three-parameter body and drop the fourth argument.
@pytest.mark.parametrize(
    "backend", ["directx", "opengl", "metal", "slang", "mojo", "vulkan"]
)
def test_call_no_generic_overload_takes_is_refused(tmp_path, backend):
    with pytest.raises(Exception, match="inc") as error:
        translate(tmp_path, TOO_MANY_FOR_EVERY_OVERLOAD, backend)
    assert "generic" in str(error.value)


@pytest.mark.parametrize("backend", C_STYLE_TARGETS)
def test_two_argument_call_specializes_only_the_two_parameter_overload(
    tmp_path, backend
):
    code = translate(tmp_path, TWO_GENERIC_OVERLOADS, backend)
    assert "inc_float2_overload1" in code
    assert "inc_float2_overload0" not in code
