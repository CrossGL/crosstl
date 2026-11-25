"""The textures, images and atomic counters the OpenGL harness binds (:mod:`tests.conformance.glsl_resources`).

The parsing and unit checks run everywhere. The llvmpipe checks opt in with CROSSTL_CONFORMANCE_NATIVE=1, as
:mod:`tests.conformance.test_native_runtimes` does, and run on host Mesa or in the Mesa container: a translation that
reads the resources the native probe reads is supported even where it binds them at other units, and one that reads
another component, another texture or another counter operation is wrong.
"""

import os

import pytest

from . import glsl_resources
from .interface import InterfaceChanged, interface
from .runtimes import RUNTIMES, provision

_SHADER = """#version 450 core
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(std430, binding = 0) buffer B0 {{ {element} {input}[]; }};
layout(std430, binding = 1) buffer B1 {{ {element} outp[]; }};
{declarations}
void main() {{
    uint i = gl_GlobalInvocationID.x;
    {body}
}}
"""
_SAMPLED = "layout(binding = 8) uniform sampler2D u0;"
_SAMPLE = "outp[i] = textureLod(u0, vec2(fin[i], fin[i + 1u]), 0.0).x;"
_COUNTED = "layout(binding = 0) uniform atomic_uint u0;"
_COUNT = "outp[i] = atomicCounterIncrement(u0) + uin[i];"


def _shader(declarations=_SAMPLED, body=_SAMPLE, element="float", input="fin"):
    return _SHADER.format(
        declarations=declarations, body=body, element=element, input=input
    )


@pytest.mark.parametrize(
    "declaration, expected",
    [
        (
            "layout(binding = 8) uniform sampler2D u0;",
            {
                "kind": "sampler",
                "type": "sampler2D",
                "dimension": "2D",
                "element": "float",
                "format": "rgba32f",
                "binding": 8,
                "count": 1,
                "ordinal": 0,
            },
        ),
        (
            "layout(r32ui, binding = 4) uniform writeonly uimage2DArray img;",
            {
                "kind": "image",
                "type": "uimage2DArray",
                "dimension": "2DArray",
                "element": "uint",
                "format": "r32ui",
                "binding": 4,
                "count": 1,
                "ordinal": 0,
            },
        ),
        (
            "layout(rgba32f, CROSSGL_BINDING(0, 2, 5)) writeonly uniform image1D img;",
            {
                "kind": "image",
                "type": "image1D",
                "dimension": "1D",
                "element": "float",
                "format": "rgba32f",
                "binding": 5,
                "count": 1,
                "ordinal": 0,
            },
        ),
        (
            "layout(binding = 3) uniform isamplerCubeArray tex[2];",
            {
                "kind": "sampler",
                "type": "isamplerCubeArray",
                "dimension": "CubeArray",
                "element": "int",
                "format": "rgba32i",
                "binding": 3,
                "count": 2,
                "ordinal": 0,
            },
        ),
        (
            "layout(binding = 1, offset = 8) uniform atomic_uint c;",
            {"kind": "counter", "binding": 1, "offset": 8, "count": 1, "ordinal": 0},
        ),
    ],
)
def test_declarations_read_the_unit_type_and_format(declaration, expected):
    assert glsl_resources.declarations(declaration) == [expected]


def test_ordinals_count_array_elements_and_counters_follow_each_other():
    records = glsl_resources.declarations(
        "layout(binding = 3) uniform sampler2D tex[2];\n"
        "layout(r32f, binding = 0) uniform image2D img;\n"
        "layout(binding = 6) uniform usampler3D vol;\n"
        "layout(binding = 0) uniform atomic_uint a;\n"
        "layout(binding = 0) uniform atomic_uint b;\n"
    )
    assert [(r["kind"], r["ordinal"]) for r in records] == [
        ("sampler", 0),
        ("image", 0),
        ("sampler", 2),
        ("counter", 0),
        ("counter", 1),
    ]
    assert [r["offset"] for r in records if r["kind"] == "counter"] == [0, 4]


@pytest.mark.parametrize(
    "declaration, message",
    [
        ("layout(binding = 8) uniform sampler2DShadow u0;", "no sampler2DShadow"),
        ("layout(binding = 3) uniform texture2D t;", "no texture2D"),
        ("layout(binding = 5) uniform sampler s;", "no sampler"),
        ("uniform sampler2D u0;", "no explicit binding"),
        ("layout(binding = 3) uniform sampler2D tex[];", "without a size"),
        ("layout(binding = 4) uniform image2D img;", "no format"),
    ],
)
def test_declarations_refuse_what_the_harness_cannot_create(declaration, message):
    with pytest.raises(InterfaceChanged, match=message):
        glsl_resources.declarations(declaration)


def test_two_resources_on_one_unit_are_refused():
    records = glsl_resources.declarations(
        "layout(binding = 3) uniform sampler2D tex[2];\n"
        "layout(binding = 4) uniform sampler2D other;\n"
    )
    with pytest.raises(InterfaceChanged, match="share a unit"):
        glsl_resources.resources(records, "")
    # Samplers and images have their own units.
    records = glsl_resources.declarations(
        "layout(binding = 4) uniform sampler2D tex;\n"
        "layout(r32f, binding = 4) uniform image2D img;\n"
    )
    assert len(glsl_resources.resources(records, "")["resources"]) == 2


def test_texels_depend_on_ordinal_and_slice_and_fill_every_texel():
    [cube] = glsl_resources.declarations("layout(binding = 0) uniform samplerCube c;")
    data = glsl_resources.texels(cube, 0)
    assert len(data) == 4 * 4 * 6 * 4 * 4
    face = len(data) // 6
    assert data[:face] != data[face : 2 * face]
    assert glsl_resources.texels(cube, 1) != data
    [image] = glsl_resources.declarations(
        "layout(r32ui, binding = 0) uniform uimageBuffer b;"
    )
    assert len(glsl_resources.buffer_texels(image, 0)) == 64 * 4


def test_interface_lists_resources_and_orders_atomic_shaders(tmp_path):
    path = tmp_path / "probe.comp"
    path.write_text(_shader())
    found = interface(path, "glsl", 1)
    assert found["bindings"] == (0, 1)
    assert [record["kind"] for record in found["resources"]] == ["sampler"]
    assert "ordered" not in found
    path.write_text(_shader(_COUNTED, _COUNT, "uint", "uin"))
    found = interface(path, "glsl", 1)
    assert found["resources"][0]["kind"] == "counter" and found["ordered"]
    # A default-block uniform the host does not set is still refused.
    path.write_text(_shader(_SAMPLED + "\nuniform float scale;"))
    with pytest.raises(InterfaceChanged, match="uniforms the host does not provide"):
        interface(path, "glsl", 1)
    path.write_text(_shader(body="outp[i] = fin[i];", declarations=""))
    assert "resources" not in interface(path, "glsl", 1)


_NATIVE = os.environ.get("CROSSTL_CONFORMANCE_NATIVE") == "1"
_TWO = (
    "layout(binding = 3) uniform sampler2D a;\nlayout(binding = 4) uniform sampler2D b;"
)
_TWO_SAMPLE = (
    "outp[i] = textureLod(a, vec2(fin[i], 0.5), 0.0).x "
    "- textureLod(b, vec2(fin[i], 0.5), 0.0).y;"
)


def _run(native, translated, name):
    from tools.coverage import execution
    from tools.coverage.model import Feature, Result

    runtime = RUNTIMES["opengl-llvmpipe"]
    provided = provision(runtime)
    if not provided.available:
        pytest.skip(provided.reason)
    feature = Feature("glsl", "construct", name, None)
    [result], used = execution.run(
        [Result(feature, "output", "supported", sources=(native, translated))],
        runtimes=runtime.id,
    )
    assert used == [runtime.id]
    return result


@pytest.mark.skipif(
    not _NATIVE,
    reason="set CROSSTL_CONFORMANCE_NATIVE=1 to execute on native runtimes",
)
@pytest.mark.parametrize(
    "native, translated, expected",
    [
        (_shader(), _shader(), "supported"),
        (
            _shader(),
            _shader("layout(binding = 0) uniform sampler2D u0;"),
            "supported",
        ),
        (_shader(), _shader(body=_SAMPLE.replace(").x;", ").y;")), "wrong"),
        (
            _shader(_TWO, _TWO_SAMPLE),
            _shader(
                "layout(binding = 0) uniform sampler2D a;\n"
                "layout(binding = 1) uniform sampler2D b;",
                _TWO_SAMPLE,
            ),
            "supported",
        ),
        (
            _shader(_TWO, _TWO_SAMPLE),
            _shader(
                "layout(binding = 4) uniform sampler2D b;\n"
                "layout(binding = 3) uniform sampler2D a;",
                _TWO_SAMPLE,
            ),
            "wrong",
        ),
        (
            _shader(_TWO, _TWO_SAMPLE),
            _shader(_TWO, _TWO_SAMPLE.replace("textureLod(b,", "textureLod(a,")),
            "wrong",
        ),
        (
            _shader(_COUNTED, _COUNT, "uint", "uin"),
            _shader(_COUNTED, _COUNT, "uint", "uin"),
            "supported",
        ),
        (
            _shader(_COUNTED, _COUNT, "uint", "uin"),
            _shader(
                _COUNTED,
                _COUNT.replace("Increment(u0)", "Decrement(u0)"),
                "uint",
                "uin",
            ),
            "wrong",
        ),
        (
            _shader(
                "layout(r32ui, binding = 8) uniform uimage2D u0;",
                "outp[i] = imageAtomicAdd(u0, ivec2(uin[i] % 4u, 1), uin[i + 1u]);",
                "uint",
                "uin",
            ),
            _shader(
                "layout(r32ui, binding = 0) uniform uimage2D u0;",
                "outp[i] = imageAtomicAdd(u0, ivec2(uin[i] % 4u, 1), uin[i + 1u]);",
                "uint",
                "uin",
            ),
            "supported",
        ),
    ],
    ids=[
        "same",
        "other-unit",
        "component",
        "two-textures",
        "swapped-textures",
        "one-texture-twice",
        "counter",
        "other-counter-operation",
        "image-atomic",
    ],
)
def test_coverage_compares_resource_reads_on_llvmpipe(native, translated, expected):
    result = _run(native, translated, "sampled-texture")
    assert result.outcome == expected, result
