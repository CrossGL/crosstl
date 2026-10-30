"""Functions, constants and structures named like ``metal`` names are renamed in Metal output."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.metal_namespace_names import METAL_NAMESPACE_NAMES


def _translate(tmp_path, source, suffix=".cgl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _compile(tmp_path, code):
    if shutil.which("xcrun") is None:
        return
    source = tmp_path / "out.metal"
    source.write_text(code)
    result = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "out.air"),
        ],
        capture_output=True,
        text=True,
    )
    if "unable to find utility" in result.stderr:
        return
    assert result.returncode == 0, result.stderr


def test_the_list_holds_the_sampler_and_sampling_argument_names():
    for name in (
        "bias",
        "level",
        "filter",
        "coord",
        "address",
        "access",
        "quad",
        "gradient2d",
        "int32_t",
    ):
        assert name in METAL_NAMESPACE_NAMES
    # Names the library leaves free, and CrossGL's own spellings of types it maps, are not listed.
    for name in ("main", "color", "ray", "intersector", "light"):
        assert name not in METAL_NAMESPACE_NAMES


@pytest.mark.parametrize("name", ["bias", "level", "filter", "quad", "access", "coord"])
def test_user_function_named_like_a_metal_type_is_renamed(tmp_path, name):
    source = f"""shader S {{
    float {name}(float x) {{ return x * 2.0; }}
    fragment {{
        vec4 main() @gl_FragColor {{
            return vec4({name}(0.25));
        }}
    }}
}}
"""
    code = _translate(tmp_path, source)
    assert not re.search(rf"\b{name}\(", code), code
    assert f"{name}_fn(0.25)" in code
    _compile(tmp_path, code)


GLSL = """#version 450
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
const float bias = 0.005;
const int level = 2;
uniform float filter_strength;
struct filter { float w; };
float shade(filter f, float bias) { return f.w + bias; }
void main() {
    filter f = filter(1.0);
    float values[level];
    values[0] = bias;
    values[1] = shade(f, 0.5);
    color = vec4(uv, values[0] + values[1], filter_strength);
}
"""


def test_glsl_constants_and_structures_named_like_metal_types_compile(tmp_path):
    code = _translate(tmp_path, GLSL, suffix=".frag")
    assert re.search(r"constant float bias_ = 0\.005;", code), code
    assert re.search(r"struct filter_ \{", code), code
    assert (
        "filter_ f = filter_{1.0}" in code or "filter_ f = filter_(1.0)" in code
    ), code
    assert re.search(r"values\[0\] = bias_;", code), code
    assert re.search(r"float values\[level_\];", code), code
    # The parameter named bias hides the library name and keeps its own.
    assert re.search(r"float shade\(filter_ f, float bias\)", code), code
    assert "f.w + bias;" in code
    _compile(tmp_path, code)


def test_metal_library_structures_keep_their_names(tmp_path):
    source = """shader S {
    struct MTLDispatchThreadgroupsIndirectArguments {
        uint threadgroupsPerGrid[3];
    };
    compute {
        buffer MTLDispatchThreadgroupsIndirectArguments args[] @buffer(0);
        void main() @workgroup_size(1, 1, 1) {
            args[0].threadgroupsPerGrid[0] = 1u;
        }
    }
}
"""
    code = _translate(tmp_path, source)
    assert "MTLDispatchThreadgroupsIndirectArguments_" not in code, code
    assert "struct MTLDispatchThreadgroupsIndirectArguments {" not in code, code
