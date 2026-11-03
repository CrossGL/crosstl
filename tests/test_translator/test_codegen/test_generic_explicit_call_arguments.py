"""CrossGL calls with explicit generic arguments (``helper<int>(x)``) bind leading parameters, deduce the rest.

The parser keeps an identifier call's explicit arguments in the callee name. Each generator used to miss the generic
definition for that name, drop the definition and emit the call verbatim, which names no function in the target.
"""

import re

import pytest

import crosstl

C_STYLE_TARGETS = ["directx", "opengl", "metal", "vulkan"]

PROGRAM = """
shader main {
    generic<T> T zero_of() {
        return T(0);
    }

    generic<T> T plus_zero(T value) {
        return zero_of<T>() + value;
    }

    generic<T, U = float> T scale_by(U factor, T value) {
        return value * T(factor);
    }

    generic<T, U = float> T widen(T value) {
        return T(U(value));
    }

    generic<T, U> T convert_to(U value) {
        return T(value);
    }

    compute {
        @ stage_entry
        void k(RWStructuredBuffer<float> values @buffer(0), RWStructuredBuffer<int> ints @buffer(1),
               uint gid @gl_GlobalInvocationID) {
            buffer_store(values, gid, zero_of<float>() + plus_zero(buffer_load(values, gid)));
            buffer_store(ints, gid, scale_by<int>(float16(2), 3) + widen<int>(3));
            buffer_store(ints, gid, convert_to<int>(buffer_load(values, gid)));
        }
    }
}
"""


def translate(tmp_path, source, backend):
    path = tmp_path / "program.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize("backend", ["directx", "opengl", "metal"])
def test_explicit_generic_arguments_select_specializations(tmp_path, backend):
    code = translate(tmp_path, PROGRAM, backend)
    for specialization in (
        "zero_of_float",
        "plus_zero_float",
        # U is deduced from the float16 argument; its float default does not apply.
        "scale_by_int_float16",
        # U is not deducible, so it takes its default.
        "widen_int_float",
        "convert_to_int_float",
    ):
        assert specialization in code
    for verbatim in ("zero_of<", "scale_by<", "widen<", "convert_to<"):
        assert verbatim not in code
    # The body of plus_zero<float> calls zero_of<float>, not an instantiation for the parameter T itself.
    assert "zero_of_T" not in code


def test_explicit_generic_arguments_lower_to_spirv_calls(tmp_path):
    module = translate(tmp_path, PROGRAM, "vulkan")
    # zero_of<float>, plus_zero<float> (which calls zero_of<float>), scale_by, widen and convert_to.
    assert module.count("OpFunctionCall") == 6


def test_deduced_parameter_keeps_the_argument_type_in_hlsl(tmp_path):
    code = translate(tmp_path, PROGRAM, "directx")
    assert "int scale_by_int_float16(float16_t factor, int value)" in code
    assert "int widen_int_float(int value)" in code


VARIABLE_ARGUMENT_OVERLOADS = """
shader main {
    buffer float* results;
    void inc<T>(inout T value) { value = value + T(1.0); }
    void inc<T>(inout T value, T step) { value = value + step; }
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        vec2 w = vec2(1.0);
        inc(w, w);
        results[0] = w.x;
    }
    }
}
"""


@pytest.mark.parametrize("backend", ["opengl", "metal", "cuda", "hip"])
def test_variable_argument_is_not_read_as_a_type_argument(tmp_path, backend):
    # inc(w, w) matches the two-parameter overload; the one-parameter overload must not be specialized for ``w``.
    code = translate(tmp_path, VARIABLE_ARGUMENT_OVERLOADS, backend)
    assert "_overload1(w, w)" in code
    assert "inc_w" not in code


EXPRESSION_ARGUMENT_OVERLOADS = """
shader main {
    buffer float* results;
    struct Q { vec2 v; };
    T sel<T>(T a) { return a; }
    T sel<T>(T a, T b) { return b; }
    compute {
    @ numthreads(1, 1, 1)
    void main() {
        Q q;
        q.v = vec2(results[0]);
        float x = results[1];
        vec2 r = sel(q.v, q.v) + sel(vec2(1.0), q.v);
        results[0] = r.x + sel(2.0, x) + sel(x + 1.0, x);
    }
    }
}
"""


@pytest.mark.parametrize(
    "backend", ["directx", "opengl", "metal", "cuda", "hip", "slang"]
)
def test_expression_argument_is_not_read_as_a_type_argument(tmp_path, backend):
    # sel(q.v, q.v), sel(vec2(1.0), q.v), sel(2.0, x) and sel(x + 1.0, x) all match sel(T a, T b); none of the
    # leading arguments is a type, so no specialization is named after one.
    code = translate(tmp_path, EXPRESSION_ARGUMENT_OVERLOADS, backend)
    specialized = set(re.findall(r"\bsel_(\w+?)_overload[01]\b", code))
    assert specialized and specialized <= {"float", "vec2", "float2"}


def test_expression_argument_overloads_lower_to_spirv(tmp_path):
    module = translate(tmp_path, EXPRESSION_ARGUMENT_OVERLOADS, "vulkan")
    assert "OpFunctionCall" in module


UNRESOLVABLE = """
shader main {
    generic<T, U> T make_value(T seed) {
        return seed + T(U(0));
    }

    compute {
        @ stage_entry
        void k(RWStructuredBuffer<int> ints @buffer(0), uint gid @gl_GlobalInvocationID) {
            buffer_store(ints, gid, CALL);
        }
    }
}
"""


@pytest.mark.parametrize(
    "call",
    [
        # U is neither supplied, deducible nor defaulted.
        "make_value<int>(1)",
        # More explicit arguments than the generic has parameters.
        "make_value<int, float, float>(1)",
    ],
)
@pytest.mark.parametrize("backend", [*C_STYLE_TARGETS, "cuda", "hip", "slang", "mojo"])
def test_unresolved_explicit_generic_call_is_refused(tmp_path, backend, call):
    with pytest.raises(Exception, match="make_value") as error:
        translate(tmp_path, UNRESOLVABLE.replace("CALL", call), backend)
    assert "generic" in str(error.value)
