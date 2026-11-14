"""GLSL's ``int`` built-in values reach Metal's ``uint`` attributes with their signedness kept.

CrossGL spells built-in values as GLSL does, and GLSL declares ``gl_FragStencilRefARB``, ``gl_ViewIndex``,
``gl_PrimitiveID``, ``gl_Layer``, ``gl_BaseVertex`` and ``gl_BaseInstance`` ``int``. Metal's ``[[stencil]]``,
``[[amplification_id]]``, ``[[primitive_id]]``, ``[[render_target_array_index]]``, ``[[base_vertex]]`` and
``[[base_instance]]`` take ``uint`` (MSL 3.2 section 5.2.3). The generator refused the ``int`` forms with an untyped
``ValueError``, and it declared ``gl_BaseVertex`` and ``gl_BaseInstance`` members as vertex attributes.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.metal_signed_builtins import (
    metal_signed_member_assignment,
)

STAGES = """
shader Probe {
    struct VertexInput {
        vec4 position;
        int base @ gl_BaseVertex;
        int inst @ gl_BaseInstance;
    }
    struct VertexOutput {
        vec4 position @ gl_Position;
        int layer @ gl_Layer;
    }
    struct FragmentInput {
        vec4 position @ gl_FragCoord;
        int view @ gl_ViewIndex;
        int prim @ gl_PrimitiveID;
    }
    struct FragmentOutput {
        vec4 color;
        int stencil @ gl_FragStencilRefARB;
    }
    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            output.position = input.position + vec4(float(input.base - input.inst), 0.0, 0.0, 0.0);
            output.layer = input.inst - 1;
            output.layer += 2;
            return output;
        }
    }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.color = vec4(float(input.view), float(input.prim), 0.0, 1.0);
            output.stencil = input.prim - 7;
            output.stencil /= 2;
            output.stencil >>= 1;
            output.stencil++;
            int previous = output.stencil--;
            if (output.stencil - 3 < 0) {
                output.stencil = previous;
            }
            return output;
        }
    }
}
"""


def _translate(tmp_path, source):
    path = tmp_path / "probe.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _compile(tmp_path, code):
    if shutil.which("xcrun") is None:
        pytest.skip("the Metal toolchain is not installed")
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    if found.returncode != 0:
        pytest.skip("the Metal toolchain is not installed")
    artifact = tmp_path / "out.metal"
    artifact.write_text(code)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(artifact)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "warning" not in result.stderr, result.stderr


def _compact(code):
    return " ".join(code.split())


def test_int_parameter_built_ins_read_their_uint_attribute(tmp_path):
    code = _compact(_translate(tmp_path, STAGES))
    assert "uint _crossglBaseVertex [[base_vertex]]" in code
    assert "uint _crossglBaseInstance [[base_instance]]" in code
    assert "uint _crossglViewIndex [[amplification_id]]" in code
    assert "int input_base = int(_crossglBaseVertex);" in code
    assert "int input_view = int(_crossglViewIndex);" in code
    assert "[[attribute(1)]]" not in code


def test_int_member_built_ins_are_uint_storage_with_int_reads(tmp_path):
    code = _compact(_translate(tmp_path, STAGES))
    assert "uint stencil [[stencil]];" in code
    assert "uint layer [[render_target_array_index]];" in code
    assert "output.layer = uint(input.inst - 1);" in code
    assert "output.layer += uint(2);" in code
    assert "output.stencil = uint(input.prim - 7);" in code
    # Division and the right shift keep GLSL's signed meaning.
    assert "output.stencil = uint(int(output.stencil) / (2));" in code
    assert "output.stencil = uint(int(output.stencil) >> (1));" in code
    assert "(void)(int(output.stencil++));" in code
    assert "int previous = int(output.stencil--);" in code
    assert "if (int(output.stencil) - 3 < 0)" in code


def test_signed_built_in_stages_compile(tmp_path):
    _compile(tmp_path, _translate(tmp_path, STAGES))


def test_a_float_stencil_member_is_still_refused(tmp_path):
    source = STAGES.replace(
        "int stencil @ gl_FragStencilRefARB", "float stencil @ gl_FragStencilRefARB"
    ).replace("output.stencil", "output.color.x")
    with pytest.raises(ValueError, match="stencil"):
        _translate(tmp_path, source)


def test_a_signed_update_of_a_target_with_side_effects_is_refused(tmp_path):
    source = """
shader Probe {
    struct FragmentOutput {
        vec4 color;
        int stencil @ gl_FragStencilRefARB;
    }
    fragment {
        FragmentOutput main() {
            FragmentOutput outputs[2];
            int i = 0;
            outputs[i++].stencil /= 2;
            return outputs[0];
        }
    }
}
"""
    with pytest.raises(ValueError, match="side effects"):
        _translate(tmp_path, source)


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("=", "s = uint(v)"),
        ("+=", "s += uint(v)"),
        ("<<=", "s <<= uint(v)"),
        ("%=", "s = uint(int(s) % (v))"),
        (">>=", "s = uint(int(s) >> (v))"),
    ],
)
def test_member_assignment_forms(op, expected):
    assert metal_signed_member_assignment("s", op, "v", True) == expected


def test_member_assignment_refuses_a_repeated_target_with_side_effects():
    assert metal_signed_member_assignment("s", "/=", "v", False) is None


FRAGMENT_PARAMETERS = """
shader Probe {
    struct FragmentOutput {
        vec4 color;
    }
    struct FragmentInput {
        vec4 position @ gl_FragCoord;
        int sample @ gl_SampleID;
    }
    fragment {
        FragmentOutput main(FragmentInput input, int prim @ gl_PrimitiveID, int layer @ gl_Layer) {
            FragmentOutput output;
            output.color = vec4(float(prim / 2), float(layer), float(input.sample % 3), 1.0);
            return output;
        }
    }
}
"""


def test_int_fragment_parameters_keep_signed_arithmetic(tmp_path):
    code = _compact(_translate(tmp_path, FRAGMENT_PARAMETERS))
    assert "uint _crossglPrimitiveID [[primitive_id]]" in code
    assert "uint _crossglLayer [[render_target_array_index]]" in code
    assert "int prim = int(_crossglPrimitiveID);" in code
    assert "int layer = int(_crossglLayer);" in code
    assert "float(prim / 2)" in code
    assert "float(int(input.sample) % 3)" in code
    _compile(tmp_path, _translate(tmp_path, FRAGMENT_PARAMETERS))
