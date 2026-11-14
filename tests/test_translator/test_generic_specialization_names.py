"""A generic function's concrete clone never takes a name the program already declares.

Targets without generic functions name the clone of ``pick<float>`` ``pick_float``. When the program declares its own
``pick_float``, the clone must take a free name instead; otherwise the output defines ``pick_float`` twice and both
calls name one of the two definitions.
"""

import re

import pytest

import crosstl

CLONING_BACKENDS = ["directx", "opengl", "metal", "cuda", "hip", "slang", "mojo"]

PROGRAM = """
shader main {
    generic<T> T pick(T a, T b) {
        return a > b ? a : b;
    }
    float pick_float(float a, float b) {
        return a + b;
    }
    %(extra)s
    compute {
        RWStructuredBuffer<float> o @buffer(0);
        @numthreads(1, 1, 1)
        void main(uint i @gl_GlobalInvocationID) {
            buffer_store(o, i, pick(buffer_load(o, i), 1.0f) + pick_float(buffer_load(o, i), 2.0f));
        }
    }
}
"""

# A definition header: the name, its parameter list, then the body or a Mojo return annotation.
DEFINITION = re.compile(r"\b(pick_float\w*)\s*\([^;{}()]*\)\s*(?:\{|->[^\n]*:)")


def _translate(tmp_path, backend, extra=""):
    source = tmp_path / "names.cgl"
    source.write_text(PROGRAM % {"extra": extra})
    return crosstl.translate(str(source), backend=backend, format_output=False)


def _definitions(code):
    return DEFINITION.findall(code)


def _calls_with_last_argument(code, name, literal):
    """Whether ``name`` is called with the float ``literal`` (with or without ``f``) as its last argument."""
    arguments = r"(?:[^();]|\([^()]*\))*"
    return re.search(rf"\b{name}\({arguments},\s*{re.escape(literal)}f?\)", code)


@pytest.mark.parametrize("backend", CLONING_BACKENDS)
def test_clone_takes_a_free_name_beside_a_declared_function(tmp_path, backend):
    code = _translate(tmp_path, backend)

    assert sorted(_definitions(code)) == ["pick_float", "pick_float_2"]
    # The generic call names the clone and the direct call keeps the declared function.
    assert _calls_with_last_argument(code, "pick_float_2", "1.0")
    assert _calls_with_last_argument(code, "pick_float", "2.0")


@pytest.mark.parametrize("backend", ["directx", "opengl", "metal"])
def test_clone_skips_every_taken_numbered_name(tmp_path, backend):
    extra = """float pick_float_2(float a, float b) {
        return a - b;
    }
    float useBoth(float a) {
        return pick_float_2(a, 1.0f);
    }"""
    code = _translate(tmp_path, backend, extra)

    assert sorted(_definitions(code)) == ["pick_float", "pick_float_2", "pick_float_3"]


def test_clone_keeps_its_readable_name_when_it_is_free(tmp_path):
    source = tmp_path / "free.cgl"
    source.write_text(PROGRAM.replace("pick_float", "sum_pair") % {"extra": ""})
    code = crosstl.translate(str(source), backend="directx", format_output=False)

    assert re.search(r"\bfloat pick_float\(float a, float b\) \{", code)
