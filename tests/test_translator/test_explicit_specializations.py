"""CrossGL ``@explicit_specialization``: a concrete function that replaces a generic's body for some arguments.

``float twice_f(float x) @explicit_specialization(twice, float)`` plays the part of C++'s
``template <> float twice<float>(float)`` ([temp.expl.spec]/1). Every instantiation of ``twice`` for ``float``,
including one reached only through another generic (``apply<float>``), calls ``twice_f``. Targets with native
generics that cannot select a body per argument refuse the program.
"""

import re

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

SOURCE = """
shader S {
    generic<T> T twice(T x) { return x * 2.0; }
    float twice_f(float x) @explicit_specialization(twice, float) { return x + x + 1.0; }
    generic<T> T apply(T x) { return twice(x); }
    compute {
        void main() {
            float a = 2.0;
            float b = apply(a);
            float c = twice(b);
            int n = 3;
            int m = twice(n);
        }
    }
}
"""

MONOMORPHIZING_TARGETS = [
    "directx",
    "opengl",
    "metal",
    "wgsl",
    "cuda",
    "hip",
    "slang",
    "mojo",
]


def _translate(tmp_path, backend, source=SOURCE):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_parser_keeps_the_specialization_as_an_annotation():
    ast = Parser(Lexer(SOURCE).get_tokens()).parse()
    function = next(f for f in ast.functions if f.name == "twice_f")
    assert function.annotations["explicit_specialization"] == ("twice", ("float",))
    assert not any(
        attribute.name == "explicit_specialization" for attribute in function.attributes
    )


def test_malformed_specialization_attribute_is_a_syntax_error():
    source = SOURCE.replace(
        "@explicit_specialization(twice, float)", "@explicit_specialization(twice)"
    )
    with pytest.raises(SyntaxError, match="explicit_specialization"):
        Parser(Lexer(source).get_tokens()).parse()


@pytest.mark.parametrize("backend", MONOMORPHIZING_TARGETS)
def test_instantiations_for_the_specialized_arguments_call_the_specialization(
    tmp_path, backend
):
    code = _translate(tmp_path, backend)
    assert "explicit_specialization" not in code
    # apply<float> and the direct twice(b) both reach the specialization.
    assert code.count("twice_f(") >= 3
    # twice<int> still instantiates the primary's body.
    assert "twice_int" in code or "twice_i32" in code


def test_spirv_runs_the_specialized_body(tmp_path):
    code = _translate(tmp_path, "vulkan", SOURCE.replace("int m = twice(n);", ""))
    # Only the primary's body multiplies; every float instantiation is the specialization.
    assert "OpFMul" not in code


def test_rust_generics_refuse_explicit_specializations(tmp_path):
    from crosstl.translator.codegen.rust_refusals import RustLoweringRefused

    with pytest.raises(RustLoweringRefused, match="explicit specialization twice_f"):
        _translate(tmp_path, "rust")


VECTOR_SOURCE = """
shader S {
    generic<T> T clip(T x) { return x; }
    vec4 clip_v(vec4 x) @explicit_specialization(clip, vec4) { return max(x, vec4(0.0)); }
    compute {
        void main() {
            vec4 a = vec4(1.0);
            vec4 b = clip(a);
            vec4 c = clip<vec4>(b);
        }
    }
}
"""


@pytest.mark.parametrize("backend", MONOMORPHIZING_TARGETS)
def test_vector_arguments_select_the_specialization(tmp_path, backend):
    # Deduced calls key the instantiation by the target's spelling (float4 on
    # Metal and HLSL), explicit ones by CrossGL's (vec4); both name vec4.
    code = _translate(tmp_path, backend, VECTOR_SOURCE)
    assert code.count("clip_v(") >= 3
    assert "clip_float4" not in code and "clip_vec4" not in code


def test_spirv_vector_arguments_select_the_specialization(tmp_path):
    code = _translate(tmp_path, "vulkan", VECTOR_SOURCE)
    # Only the specialization clamps; a clone of the primary would return x.
    assert code.count("OpFunctionCall") == 2
    assert code.count("= OpFunction ") == 2  # main and clip_v


DECLARED_PRIMARY_SOURCE = VECTOR_SOURCE.replace(
    "generic<T> T clip(T x) { return x; }", "generic<T> T clip(T x);"
).replace(
    "    compute {",
    "    float clip_f(float x) @explicit_specialization(clip, float) { return x; }\n"
    "    compute {",
)


@pytest.mark.parametrize("backend", ["directx", "opengl", "metal", "vulkan"])
def test_declaration_only_primary_reaches_its_specializations(tmp_path, backend):
    code = _translate(tmp_path, backend, DECLARED_PRIMARY_SOURCE)
    if backend == "vulkan":
        assert code.count("OpFunctionCall") == 2
        return
    assert code.count("clip_v(") >= 3
    assert "clip_float4" not in code and "clip_vec4" not in code


NARROW_SCALAR_SOURCE = """
shader S {
    generic<T> T top();
    uint top_u32() @explicit_specialization(top, uint) { return 4294967295u; }
    uint8 top_u8() @explicit_specialization(top, uint8) { return uint8(255); }
    compute {
        void main() {
            uint a = top<uint>();
            uint8 b = top<uint8>();
        }
    }
}
"""


@pytest.mark.parametrize("backend", MONOMORPHIZING_TARGETS)
def test_scalars_a_target_widens_keep_their_own_specializations(tmp_path, backend):
    # Metal maps uint8 to uint; comparing mapped spellings merged the two specializations.
    code = _translate(tmp_path, backend, NARROW_SCALAR_SOURCE)
    calls = [line for line in code.splitlines() if "top" in line and "=" in line]
    a_line = next(line for line in calls if re.search(r"\ba\b\s*(?::[^=]*)?=", line))
    b_line = next(line for line in calls if re.search(r"\bb\b\s*(?::[^=]*)?=", line))
    assert "top_u32()" in a_line and "top_u8" not in a_line, code
    assert "top_u8()" in b_line and "top_u32" not in b_line, code
