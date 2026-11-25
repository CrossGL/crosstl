"""Metal output declares a run-time indexed CrossGL texture array of unknown length as an argument buffer.

``uniform sampler2D tex[];`` indexed at run time is a runtime descriptor array whose length the host chooses. A Metal
entry point binds a fixed number of textures, so the generator sized the array from the values its index could take
(``int(x) & 1`` gave ``array<texture2d<float>, 2>``) and refused an index with no bound. An argument buffer, a
buffer of structures that each hold a texture, is Metal's array of unknown length.
"""

import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.resource_array_bounds import (
    ResourceArrayIndexUnboundedError,
)

COMPUTE = textwrap.dedent("""
    shader S {
        DECLARATIONS
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer float* fin;
            buffer float* outp;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                int k = int(fin[0]);
                BODY
            }
        }
    }
    """)

# The AIR of an argument buffer: a structure type that holds a texture (tools/coverage/decorations.py).
ARGUMENT_BUFFER = re.compile(
    r'^%struct\.\w+ = type \{[^\n]*%"struct\.metal::texture\w*"', re.MULTILINE
)


def _translate(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _compute(declarations, body):
    return COMPUTE.replace("DECLARATIONS", declarations).replace("BODY", body)


def _flat(text):
    """``text`` on one line, with one space between tokens and the pointer star written ``T *name``."""
    return re.sub(r"\s*\*\s*(?=\w)", " *", " ".join(text.split()))


def _air(tmp_path, metal):
    """The AIR text xcrun compiles ``metal`` to, or None without the Metal compiler."""
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return None
    shader = tmp_path / "out.metal"
    shader.write_text(metal, encoding="utf-8")
    air = tmp_path / "out.ll"
    result = subprocess.run(
        [xcrun, "-sdk", "macosx", "metal", "-std=metal3.0", "-S", "-emit-llvm"]
        + ["-c", str(shader), "-o", str(air)],
        capture_output=True,
        text=True,
        check=False,
    )
    if "unable to find utility" in result.stderr:
        return None
    assert result.returncode == 0, result.stderr
    return air.read_text(encoding="utf-8")


def _assert_argument_buffer(tmp_path, metal):
    air = _air(tmp_path, metal)
    if air is not None:
        assert ARGUMENT_BUFFER.search(air), "no argument buffer structure in the AIR"


def test_combined_sampler_array_indexed_at_run_time_is_an_argument_buffer(tmp_path):
    metal = _translate(
        tmp_path,
        _compute(
            "uniform sampler2D tex[];",
            "outp[i] = textureLod(tex[k & 1], vec2(fin[i], 0.5), 0.0).x;",
        ),
    )
    flat = _flat(metal)
    assert (
        "struct tex_slot { texture2d<float> texture; sampler sampler_state; };" in flat
    )
    assert re.search(r"const device tex_slot \*tex \[\[buffer\(\d+\)\]\]", flat)
    assert "tex[k & 1].texture.sample(tex[k & 1].sampler_state," in flat
    assert "array<texture2d<float>" not in flat
    _assert_argument_buffer(tmp_path, metal)


def test_unbounded_index_translates_instead_of_refusing(tmp_path):
    metal = _translate(
        tmp_path,
        _compute(
            "uniform sampler2D tex[];",
            "outp[i] = textureLod(tex[k], vec2(fin[i], 0.5), 0.0).x;",
        ),
    )
    assert "tex[k].texture.sample(tex[k].sampler_state," in _flat(metal)
    _assert_argument_buffer(tmp_path, metal)


def test_program_scope_array_with_nonuniform_index_and_glsl_binding(tmp_path):
    source = textwrap.dedent("""
        shader S {
            layout(binding = 0) uniform sampler2D tex[];
            compute {
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                layout(std430, binding = 0) buffer Data { float v[]; } data;
                void main() {
                    int k = int(data.v[0]);
                    data.v[1] = textureLod(tex[nonuniformEXT(k)], vec2(0.5), 0.0).x;
                }
            }
        }
        """)
    metal = _translate(tmp_path, source)
    flat = _flat(metal)
    # The GLSL binding names a texture unit, so the argument buffer takes a buffer slot the block does not hold.
    data_slot = re.search(r"\*data \[\[buffer\((\d+)\)\]\]", flat).group(1)
    tex_slot = re.search(r"\*tex \[\[buffer\((\d+)\)\]\]", flat).group(1)
    assert data_slot != tex_slot
    assert "nonuniformEXT" not in flat
    _assert_argument_buffer(tmp_path, metal)


def test_separate_texture_array_samples_through_its_sampler(tmp_path):
    metal = _translate(
        tmp_path,
        _compute(
            "texture2D tex[];\nsampler smp;",
            "outp[i] = textureLod(tex[k], smp, vec2(0.5), 0.0).x;",
        ),
    )
    flat = _flat(metal)
    assert "struct tex_slot { texture2d<float> texture; };" in flat
    assert "tex[k].texture.sample(smp," in flat
    _assert_argument_buffer(tmp_path, metal)


def test_hlsl_unbounded_texture_array_keeps_its_register(tmp_path):
    source = textwrap.dedent("""
        StructuredBuffer<float> fin : register(t0);
        RWStructuredBuffer<float> outp : register(u0);
        Texture2D tex[] : register(t1);
        SamplerState smp : register(s0);
        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) {
            uint k = uint(fin[0]);
            outp[id.x] = tex[k].SampleLevel(smp, float2(0.5, 0.5), 0.0).x;
        }
        """)
    path = tmp_path / "shader.hlsl"
    path.write_text(source, encoding="utf-8")
    metal = crosstl.translate(str(path), backend="metal", format_output=False)
    flat = _flat(metal)
    assert "const device tex_slot *tex [[buffer(1)]]" in flat
    assert "tex[k].texture.sample(smp," in flat
    _assert_argument_buffer(tmp_path, metal)


def test_texel_calls_read_the_element_texture(tmp_path):
    metal = _translate(
        tmp_path,
        _compute(
            "uniform isampler2D tex[];",
            "ivec2 size = textureSize(tex[k], 0);\n"
            "outp[i] = float(texelFetch(tex[k], ivec2(0, 0), 0).x + size.x);",
        ),
    )
    flat = _flat(metal)
    assert "struct tex_slot { texture2d<int> texture; sampler sampler_state; };" in flat
    assert "tex[k].texture.read(" in flat
    assert "tex[k].texture.get_width(" in flat
    _assert_argument_buffer(tmp_path, metal)


def test_helper_function_takes_the_argument_buffer(tmp_path):
    source = textwrap.dedent("""
        shader S {
            uniform sampler2D tex[];
            float lookup(int k) { return textureLod(tex[k], vec2(0.5), 0.0).x; }
            compute {
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                buffer float* fin;
                buffer float* outp;
                void main() { outp[0] = lookup(int(fin[0])); }
            }
        }
        """)
    metal = _translate(tmp_path, source)
    flat = _flat(metal)
    assert "float lookup(int k, const device tex_slot *tex)" in flat
    _assert_argument_buffer(tmp_path, metal)


def test_fragment_stage_array(tmp_path):
    source = textwrap.dedent("""
        shader S {
            uniform sampler2D tex[];
            uniform int layer;
            fragment {
                vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor { return texture(tex[layer], uv); }
            }
        }
        """)
    metal = _translate(tmp_path, source)
    assert "tex[layer].texture.sample(tex[layer].sampler_state, uv)" in _flat(metal)
    _assert_argument_buffer(tmp_path, metal)


@pytest.mark.parametrize(
    ("declaration", "texture"),
    [
        ("uniform samplerCubeArray tex[];", "texturecube_array<float>"),
        ("uniform usampler3D tex[];", "texture3d<uint>"),
        ("uniform sampler2DArray tex[];", "texture2d_array<float>"),
    ],
)
def test_texture_dimensions(tmp_path, declaration, texture):
    metal = _translate(
        tmp_path,
        _compute(declaration, "outp[i] = float(textureSize(tex[k], 0).x);"),
    )
    assert f"struct tex_slot {{ {texture} texture;" in _flat(metal)
    _assert_argument_buffer(tmp_path, metal)


def test_constant_indices_keep_the_implicitly_sized_array(tmp_path):
    metal = _translate(
        tmp_path,
        _compute(
            "uniform sampler2D tex[];",
            "outp[i] = textureLod(tex[1], vec2(0.5), 0.0).x;",
        ),
    )
    flat = _flat(metal)
    assert "array<texture2d<float>, 2> tex [[texture(0)]]" in flat
    assert "tex_slot" not in flat


def test_combined_sampler_index_with_side_effects_is_not_duplicated(tmp_path):
    source = _compute(
        "uniform sampler2D tex[];",
        "int j = 0;\noutp[i] = textureLod(tex[j++], vec2(0.5), 0.0).x;",
    )
    # Sampling the element twice would increment j twice, so the array keeps the sized path, which refuses it.
    with pytest.raises(ResourceArrayIndexUnboundedError):
        _translate(tmp_path, source)
