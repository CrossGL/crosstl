"""WGSL output lowers a plain enumeration to i32 constants (#3687).

WGSL has no enumerations. Every program with a shader-scope enumeration used to fail WGSL
translation with ``AttributeError: 'EnumNode' object has no attribute 'members'``.
"""

import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.wgsl_enums import WGSLEnumUnsupportedError

PROGRAM = textwrap.dedent("""
    shader Probe {
        %s
        struct E { Mode m; float w; };
        buffer float values[];
        Mode pick(Mode a) { return Mode::Third; }
        compute {
            layout(local_size_x = 1) in;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                E e;
                e.m = pick(Mode::Second);
                e.w = 1.0;
                values[i] = e.w;
            }
        }
    }
""")


def translate(tmp_path, enum):
    path = tmp_path / "probe.cgl"
    path.write_text(PROGRAM % enum)
    return crosstl.translate(str(path), backend="wgsl", format_output=False)


@pytest.mark.parametrize(
    "enum",
    [
        "enum Mode { First, Second = 5, Third }",
        "enum Mode { First, Second = 5, Third };",
    ],
)
def test_plain_enum_becomes_i32_constants(tmp_path, enum):
    output = translate(tmp_path, enum)
    assert "const Mode_First: i32 = 0;" in output
    assert "const Mode_Second: i32 = 5;" in output
    assert "const Mode_Third: i32 = 6;" in output
    assert "m: i32," in output
    assert "fn pick(a: i32) -> i32 {" in output
    assert "return Mode_Third;" in output
    assert "e.m = pick(Mode_Second);" in output
    assert "Mode::" not in output


def test_enum_with_payload_is_refused(tmp_path):
    with pytest.raises(WGSLEnumUnsupportedError, match="Mode"):
        translate(tmp_path, "enum Mode { First(float), Second, Third }")
