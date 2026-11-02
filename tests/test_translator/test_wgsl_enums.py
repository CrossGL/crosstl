"""WGSL output lowers a plain enumeration to i32 constants (#3687).

WGSL has no enumerations. Every program with a shader-scope enumeration used to fail WGSL
translation with ``AttributeError: 'EnumNode' object has no attribute 'members'``.
"""

import shutil
import subprocess
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


def wgsl(tmp_path, source):
    """WGSL output for ``source``, checked with naga when it is installed."""
    path = tmp_path / "probe.cgl"
    path.write_text(source)
    output = crosstl.translate(str(path), backend="wgsl", format_output=False)
    if shutil.which("naga"):
        (tmp_path / "out.wgsl").write_text(output)
        checked = subprocess.run(
            ["naga", "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
    return output


def translate(tmp_path, enum):
    return wgsl(tmp_path, PROGRAM % enum)


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


def test_variant_without_value_is_one_more_than_the_previous_one(tmp_path):
    output = translate(
        tmp_path,
        "const int BASE = 4;\n"
        "enum Mode { First = -5, Second, Third = 1 << 3, Fourth, "
        "Fifth = Mode::First + 20, Sixth, Seventh = BASE, Eighth }",
    )
    assert "const Mode_Second: i32 = -4;" in output
    assert "const Mode_Fourth: i32 = 9;" in output
    assert "const Mode_Sixth: i32 = 16;" in output
    assert "const Mode_Seventh: i32 = BASE;" in output
    assert "const Mode_Eighth: i32 = Mode_Seventh + 1;" in output


def test_variant_is_an_i32_value(tmp_path):
    output = wgsl(
        tmp_path,
        textwrap.dedent("""
            shader Probe {
                enum Mode { First, Second }
                buffer uint counts[];
                uint first() { return Mode::First; }
                compute {
                    layout(local_size_x = 1) in;
                    void main() {
                        uint i = gl_GlobalInvocationID.x;
                        uint u = Mode::Second;
                        float f = Mode::Second;
                        switch (int(i)) {
                            case Mode::First:
                                u = first();
                                break;
                            default:
                                break;
                        }
                        counts[i] = u + uint(f);
                    }
                }
            }
        """),
    )
    assert "return u32(Mode_First);" in output
    assert "var u: u32 = u32(Mode_Second);" in output
    assert "var f: f32 = f32(Mode_Second);" in output
    assert "case Mode_First: {" in output


def test_enum_with_payload_is_refused(tmp_path):
    with pytest.raises(WGSLEnumUnsupportedError, match="Mode"):
        translate(tmp_path, "enum Mode { First(float), Second, Third }")
