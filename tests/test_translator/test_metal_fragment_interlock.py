"""Metal output orders a fragment interlock critical section with raster order groups.

GLSL's ``beginInvocationInterlockARB()``/``endInvocationInterlockARB()`` (and SPIR-V's interlock instructions, which
import as the same calls) were left in the Metal output, which has no such functions, so every GLSL or SPIR-V
fragment shader with a critical section was refused. Metal orders overlapping fragments' accesses per resource with
``[[raster_order_group(0)]]``; the generator now places the resources the section reaches in group 0 and removes the
calls. CrossGL's ``RasterizerOrdered*`` view types are bound as their storage views in group 0 as well, and a
subscript of a texture, which MSL cannot write, is refused instead of written as it stands.
"""

import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused

ORDERED = "[[raster_order_group(0)]]"

GLSL_BUFFER = textwrap.dedent("""\
    #version 460
    #extension GL_ARB_fragment_shader_interlock : require
    layout(pixel_interlock_ordered) in;
    layout(location = 0) in vec2 uv;
    layout(location = 0) out vec4 color;
    layout(std430, binding = 0) buffer Accum {
        vec4 accum[];
    };
    layout(std430, binding = 1) buffer Log {
        uint hits[];
    };
    void main() {
        uint k = uint(gl_FragCoord.y) * 64u + uint(gl_FragCoord.x);
        hits[k] = 1u;
        beginInvocationInterlockARB();
        accum[k] = accum[k] + vec4(uv, 0.0, 1.0);
        endInvocationInterlockARB();
        color = vec4(uv, 0.0, 1.0);
    }
    """)

GLSL_IMAGE = textwrap.dedent("""\
    #version 460
    #extension GL_ARB_fragment_shader_interlock : require
    layout(sample_interlock_unordered) in;
    layout(location = 0) in vec2 uv;
    layout(location = 0) out vec4 color;
    layout(rgba32f, binding = 0) uniform image2D accum;
    void main() {
        ivec2 p = ivec2(gl_FragCoord.xy);
        beginInvocationInterlockARB();
        imageStore(accum, p, imageLoad(accum, p) + vec4(uv, 0.0, 1.0));
        endInvocationInterlockARB();
        color = vec4(uv, 0.0, 1.0);
    }
    """)

CROSSGL_HELPER = textwrap.dedent("""\
    shader S {
        buffer vec4* accum;
        buffer vec4* other;
        struct FragmentInput {
            vec4 coord @ gl_FragCoord;
            vec2 uv;
        }
        struct FragmentOutput { vec4 color; }
        void add(uint k, vec2 uv) {
            accum[k] = accum[k] + vec4(uv, 0.0, 1.0);
        }
        fragment {
            FragmentOutput main(FragmentInput input) {
                FragmentOutput output;
                uint k = uint(input.coord.y) * 64u + uint(input.coord.x);
                beginInvocationInterlockARB();
                add(k, input.uv);
                endInvocationInterlockARB();
                other[k] = vec4(1.0);
                output.color = vec4(input.uv, 0.0, 1.0);
                return output;
            }
        }
    }
    """)

CROSSGL_ORDERED_VIEWS = textwrap.dedent("""\
    shader S {
        RasterizerOrderedTexture2D<float4> image @register(u0);
        RasterizerOrderedBuffer<uint> texels @register(u1);
        RasterizerOrderedByteAddressBuffer bytes @register(u2);
        RasterizerOrderedStructuredBuffer<float4> elements @register(u3);
        fragment {
            vec4 main(vec4 coord @ gl_FragCoord) @ gl_FragColor {
                ivec2 p = ivec2(coord.xy);
                imageStore(image, p, imageLoad(image, p) + vec4(1.0));
                imageStore(texels, p.x, imageLoad(texels, p.x) + uvec4(1u));
                bytes.Store(0u, bytes.Load(0u) + 1u);
                elements[p.x] = elements[p.x] + vec4(1.0);
                return vec4(1.0);
            }
        }
    }
    """)

CROSSGL_TEXTURE_SUBSCRIPT = textwrap.dedent("""\
    shader S {
        image2D image @format(unknown) @register(u0);
        fragment {
            vec4 main(vec4 coord @ gl_FragCoord) @ gl_FragColor {
                uvec2 p = uvec2(coord.xy);
                image[p] = image[p] + vec4(1.0);
                return vec4(1.0);
            }
        }
    }
    """)


def _translate(tmp_path, source, suffix):
    path = tmp_path / f"shader.{suffix}"
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


def _arguments(metal):
    """The fragment function's arguments, attributes included."""
    head = metal[metal.index("fragment ") :]
    head = head[head.index("(") + 1 : head.index(") {")]
    arguments, depth, current = [], 0, ""
    for character in head:
        depth += character in "<(["
        depth -= character in ">)]"
        if character == "," and depth == 0:
            arguments.append(current.strip())
            current = ""
        else:
            current += character
    return arguments + [current.strip()]


def _argument(metal, name):
    """The fragment argument whose name starts with ``name`` (GLSL block instances get a suffix)."""
    for argument in _arguments(metal):
        if re.search(rf"\b{name}\w*\s*\[\[", argument):
            return argument
    raise AssertionError(f"no argument {name!r} in {metal}")


def test_glsl_critical_section_orders_the_buffer_it_reaches(tmp_path):
    metal = _translate(tmp_path, GLSL_BUFFER, "frag")

    assert "InvocationInterlock" not in metal
    assert ORDERED in _argument(metal, "accum")
    # Only the section's accesses need ordering; the stage input is no resource.
    assert ORDERED not in _argument(metal, "log")
    assert ORDERED not in _argument(metal, "input")
    _assert_compiles(tmp_path, metal)


def test_glsl_critical_section_orders_a_storage_image(tmp_path):
    metal = _translate(tmp_path, GLSL_IMAGE, "frag")

    assert "InvocationInterlock" not in metal
    argument = _argument(metal, "accum")
    assert "texture2d<float, access::read_write>" in argument
    assert ORDERED in argument
    _assert_compiles(tmp_path, metal)


def test_critical_section_reaches_resources_through_called_functions(tmp_path):
    metal = _translate(tmp_path, CROSSGL_HELPER, "cgl")

    assert "InvocationInterlock" not in metal
    assert ORDERED in _argument(metal, "accum")
    assert ORDERED not in _argument(metal, "other")
    _assert_compiles(tmp_path, metal)


def test_rasterizer_ordered_view_types_bind_storage_views_in_group_zero(tmp_path):
    metal = _translate(tmp_path, CROSSGL_ORDERED_VIEWS, "cgl")

    image = _argument(metal, "image")
    assert "texture2d<float, access::read_write>" in image and ORDERED in image
    texels = _argument(metal, "texels")
    assert "texture_buffer<uint, access::read_write>" in texels and ORDERED in texels
    assert "device uint" in _argument(metal, "bytes")
    assert ORDERED in _argument(metal, "bytes")
    elements = _argument(metal, "elements")
    assert "device float4" in elements and ORDERED in elements
    _assert_compiles(tmp_path, metal)


def test_texture_subscript_is_refused(tmp_path):
    with pytest.raises(MetalLoweringRefused, match="subscript of texture 'image'"):
        _translate(tmp_path, CROSSGL_TEXTURE_SUBSCRIPT, "cgl")
