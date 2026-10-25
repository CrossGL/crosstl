"""The OpenGL emitter refuses string literals that GLSL cannot contain.

GLSL has no string type; a string literal may appear only in ``#include``/``#line`` directives (GLSL 4.60
section 3.1) or as the format argument of ``debugPrintfEXT`` (GL_EXT_debug_printf). A source that passes a
string to a helper must be refused, not emitted for the compiler to reject.
"""

from __future__ import annotations

import pytest

import crosstl
from crosstl.translator.codegen.glsl_refusals import OpenGLLoweringRefused
from crosstl.translator.codegen.glsl_string_literals import string_literal_reasons

STRING_ARGUMENT = """shader S {
    compute {
        layout(std430, binding = 0) buffer Out { uint values[]; } out_buf;
        void main() {
            report("file.cgl", 3);
            out_buf.values[0] = 1u;
        }
    }
}
"""

PRINTF = """shader S {
    fragment {
        vec4 main(vec3 v @ TEXCOORD0) @ gl_FragColor {
            printf("v=%v3f // not a comment", v);
            return vec4(v, 1.0);
        }
    }
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_string_argument_to_a_call_is_refused(tmp_path):
    with pytest.raises(OpenGLLoweringRefused, match=r'string literal "file\.cgl"'):
        _translate(tmp_path, STRING_ARGUMENT, "opengl")


def test_debug_printf_format_is_kept(tmp_path):
    code = _translate(tmp_path, PRINTF, "opengl")
    assert 'debugPrintfEXT("v=%v3f // not a comment", v' in code


def test_scan_skips_comments_directives_and_printf_formats():
    code = "\n".join(
        [
            '#line 4 "shader.glsl"',
            '// a "quoted" comment',
            '/* "block" */',
            "void main() {",
            "    debugPrintfEXT(",
            '        "x=%d", 1);',
            "}",
        ]
    )
    assert string_literal_reasons(code) == []


def test_scan_reports_each_literal_once_in_order():
    code = 'f("a"); g("b // c"); f("a");'
    reasons = string_literal_reasons(code)
    assert [reason.split(":")[0] for reason in reasons] == [
        'string literal "a"',
        'string literal "b // c"',
    ]
    assert "GLSL 4.60 section 3.1" in reasons[0]


def test_long_literals_are_shortened_in_the_reason():
    literal = '"' + "x" * 80 + '"'
    (reason,) = string_literal_reasons(f"f({literal});")
    assert reason.startswith('string literal "' + "x" * 35 + '..."')
