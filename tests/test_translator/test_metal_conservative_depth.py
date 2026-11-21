"""Metal output states CrossGL's conservative depth layouts with the depth attribute's argument.

CrossGL writes GLSL's conservative depth layout as an attribute next to the fragment depth output. MSL writes the
same promise as ``[[depth(greater)]]``, ``[[depth(less)]]`` or ``[[depth(any)]]`` on the fragment output (MSL 3.2
section 5.2.3.5). The generator gave a stage-level ``gl_FragDepth`` with a layout the vertex attribute
``[[attribute(n)]]``, which Metal rejects on a fragment output, and wrote ``depth(any)`` for an output structure
member or an output parameter, which drops the promise.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.refusals import MetalRefusal

STAGE_OUTPUT = textwrap.dedent("""
    shader S {
        struct FragmentInput { vec2 uv; }
        out vec4 color @location(0);
        out float gl_FragDepth @LAYOUT;
        fragment {
            void main(FragmentInput input) {
                color = vec4(input.uv, 0.0, 1.0);
                gl_FragDepth = input.uv.x * 0.5 + 0.25;
            }
        }
    }
    """)

STRUCT_OUTPUT = textwrap.dedent("""
    shader S {
        struct FragmentInput { vec2 uv; }
        struct FragOut {
            vec4 color @location(0);
            float depth @gl_FragDepth @LAYOUT;
        }
        fragment {
            FragOut main(FragmentInput input) {
                FragOut o;
                o.color = vec4(input.uv, 0.0, 1.0);
                o.depth = input.uv.x * 0.5;
                return o;
            }
        }
    }
    """)

PARAMETER_OUTPUT = textwrap.dedent("""
    shader S {
        struct FragmentInput { vec2 uv; }
        fragment {
            void main(FragmentInput input, out vec4 color @location(0),
                      out float d @gl_FragDepth @LAYOUT) {
                color = vec4(input.uv, 0.0, 1.0);
                d = input.uv.x * 0.5 + 0.25;
            }
        }
    }
    """)

LAYOUTS = [
    ("depth_greater", "depth(greater)"),
    ("depth_less", "depth(less)"),
    ("depth_any", "depth(any)"),
    ("depth_unchanged", "depth(any)"),
]


def _translate(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _assert_compiles(tmp_path, metal):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return
    shader = tmp_path / "out.metal"
    shader.write_text(metal, encoding="utf-8")
    result = subprocess.run(
        [xcrun, "-sdk", "macosx", "metal", "-std=metal3.0", "-c", str(shader)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
        check=False,
    )
    if "unable to find utility" not in result.stderr:
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "source, member",
    [
        (STAGE_OUTPUT, "float gl_FragDepth"),
        (STRUCT_OUTPUT, "float depth"),
        (PARAMETER_OUTPUT, "float d"),
    ],
    ids=["stage-output", "struct-member", "output-parameter"],
)
@pytest.mark.parametrize("layout, attribute", LAYOUTS, ids=[x for x, _ in LAYOUTS])
def test_depth_layout_is_the_metal_depth_argument(
    tmp_path, source, member, layout, attribute
):
    metal = _translate(tmp_path, source.replace("LAYOUT", layout))

    assert f"{member} [[{attribute}]];" in metal
    assert "[[attribute(" not in metal
    _assert_compiles(tmp_path, metal)


def test_depth_without_layout_stays_depth_any(tmp_path):
    metal = _translate(tmp_path, STAGE_OUTPUT.replace(" @LAYOUT", ""))

    assert "float gl_FragDepth [[depth(any)]];" in metal


def test_two_depth_layouts_are_refused(tmp_path):
    source = STRUCT_OUTPUT.replace("@LAYOUT", "@depth_greater @depth_less")

    with pytest.raises(MetalRefusal, match="depth_greater, depth_less"):
        _translate(tmp_path, source)


def test_depth_layout_on_a_colour_output_is_refused(tmp_path):
    source = STRUCT_OUTPUT.replace("@location(0);", "@location(0) @depth_greater;")
    source = source.replace(" @LAYOUT", "")

    with pytest.raises(MetalRefusal, match="only through the depth attribute"):
        _translate(tmp_path, source)
