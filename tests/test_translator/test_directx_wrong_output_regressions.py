"""HLSL output for CrossGL constructs that DXC rejected or that lost their meaning.

Each case names the coverage feature it came from. Where DXC is on ``PATH`` the generated HLSL is compiled too.
"""

import re
import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.refusals import DirectXRefusal


def _hlsl(tmp_path, source, name="probe.cgl"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(source).strip() + "\n", encoding="utf-8")
    return crosstl.translate(str(path), backend="directx")


def _compile(tmp_path, generated, profile, entry=None):
    compiler = shutil.which("dxc")
    if compiler is None:
        return
    source = tmp_path / "generated.hlsl"
    source.write_text(generated, encoding="utf-8")
    command = [compiler, "-T", profile, "-HV", "2021", "-enable-16bit-types"]
    command += ["-E", entry] if entry else []
    command += [str(source), "-Fo", str(tmp_path / "generated.bin")]
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_stage_helper_calls_take_their_default_arguments(tmp_path):
    # default-argument: a helper declared inside the stage block kept calls that omit defaulted parameters.
    generated = _hlsl(
        tmp_path,
        """
        shader Probe {
            compute {
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                buffer float* outp;
                float blend(float a, float b = 2.0, int k = 3) {
                    return a * b + float(k);
                }
                void main() {
                    uint i = gl_GlobalInvocationID.x;
                    outp[i] = blend(1.0) + blend(1.0, 4.0) + blend(1.0, 0.5, 1);
                }
            }
        }
        """,
    )
    assert "blend(1.0, 2.0, 3)" in generated
    assert "blend(1.0, 4.0, 3)" in generated
    assert "blend(1.0, 0.5, 1)" in generated
    _compile(tmp_path, generated, "cs_6_2", "CSMain")


@pytest.mark.parametrize(
    ("layout", "semantic"),
    [
        ("depth_greater", "SV_DepthGreaterEqual"),
        ("depth_less", "SV_DepthLessEqual"),
        ("depth_any", "SV_Depth"),
        ("depth_unchanged", "SV_Depth"),
    ],
)
def test_conservative_depth_layout_selects_the_depth_semantic(
    tmp_path, layout, semantic
):
    # conservative-depth: '@depth_greater' was written as the user semantic 'depth_greater'.
    generated = _hlsl(
        tmp_path,
        f"""
        shader Probe {{
            struct FragmentInput {{
                vec2 uv;
            }}
            out vec4 color @location(0);
            out float gl_FragDepth @{layout};
            fragment {{
                void main(FragmentInput input) {{
                    color = vec4(input.uv, 0.0, 1.0);
                    gl_FragDepth = input.uv.x * 0.5 + 0.25;
                }}
            }}
        }}
        """,
    )
    assert re.search(rf"float gl_FragDepth\s*:\s*{semantic}\s*;", generated)
    assert layout not in generated
    _compile(tmp_path, generated, "ps_6_2", "PSMain")


def test_conservative_depth_on_an_output_structure_member(tmp_path):
    generated = _hlsl(
        tmp_path,
        """
        shader Probe {
            struct FragmentOutput {
                vec4 color @ gl_FragColor;
                float depth @ gl_FragDepth @ depth_less;
            }
            fragment {
                FragmentOutput main(vec2 uv @ TexCoord0) {
                    FragmentOutput output;
                    output.color = vec4(uv, 0.0, 1.0);
                    output.depth = uv.x * 0.5;
                    return output;
                }
            }
        }
        """,
    )
    assert re.search(r"float depth\s*:\s*SV_DepthLessEqual\s*;", generated)
    _compile(tmp_path, generated, "ps_6_2", "PSMain")


def test_conservative_depth_layout_on_a_colour_output_is_refused(tmp_path):
    with pytest.raises(DirectXRefusal, match="conservative depth layout"):
        _hlsl(
            tmp_path,
            """
            shader Probe {
                struct FragmentOutput {
                    vec4 color @ gl_FragColor @ depth_greater;
                }
                fragment {
                    FragmentOutput main(vec2 uv @ TexCoord0) {
                        FragmentOutput output;
                        output.color = vec4(uv, 0.0, 1.0);
                        return output;
                    }
                }
            }
            """,
        )


@pytest.mark.parametrize(
    "call",
    [
        "obj.Invoke(obj, payload);",
        "dx::HitObject r = obj.TraceRay(scene, 0u, 255u, 0u, 1u, 0u, ray, payload);",
    ],
)
def test_hit_object_member_calls_mark_their_payload(tmp_path, call):
    # DxHitObject.Invoke and DxHitObject.TraceRay: DXC requires the payload type to be a [raypayload] structure.
    generated = _hlsl(
        tmp_path,
        f"""
        shader Probe {{
            struct Payload {{
                vec4 color;
            }}
            ray_generation {{
                buffer uint* outp;
                uniform accelerationStructureEXT scene;
                void main() {{
                    Payload payload;
                    dx::HitObject obj;
                    RayDesc ray;
                    ray.Origin = vec3(0.0, 0.0, 0.0);
                    ray.TMin = 0.0;
                    ray.Direction = vec3(0.0, 0.0, 1.0);
                    ray.TMax = 1.0;
                    {call}
                    outp[0] = 0u;
                }}
            }}
        }}
        """,
    )
    assert re.search(r"struct\s*\[raypayload\]\s*Payload", generated)
    assert "read(caller, closesthit, miss, anyhit)" in generated
    _compile(tmp_path, generated, "lib_6_9")


def test_typedef_array_return_uses_the_lowered_return_type(tmp_path):
    # array-return (input): 'typedef float Triple[3]' as a return type was written as 'float[3] spread(...)'.
    source = tmp_path / "probe.hlsl"
    source.write_text(
        textwrap.dedent("""
            RWStructuredBuffer<float> outp : register(u0);
            typedef float Triple[3];
            Triple spread(float x) {
                Triple r = {x, x * 2.0, x + 1.0};
                return r;
            }
            [numthreads(1, 1, 1)]
            void main(uint3 id : SV_DispatchThreadID) {
                Triple a = spread(outp[id.x]);
                outp[id.x] = a[0] + a[2] + spread(2.0)[1];
            }
            """).strip() + "\n",
        encoding="utf-8",
    )
    generated = crosstl.translate(str(source), backend="directx")
    assert "float3 spread(float x)" in generated
    assert "float[3]" not in generated
    _compile(tmp_path, generated, "cs_6_2", "CSMain")


def test_long_array_return_builds_its_structure_with_a_constructor(tmp_path):
    # A five-element return lowers to a structure, which HLSL cannot construct with a cast-style call.
    generated = _hlsl(
        tmp_path,
        """
        shader Probe {
            compute {
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                buffer float* outp;
                float[5] spread(float x) {
                    float r[5] = float[5](x, x * 2.0, x + 1.0, x - 1.0, 7.0);
                    return r;
                }
                void main() {
                    uint i = gl_GlobalInvocationID.x;
                    float a[5] = spread(outp[i]);
                    outp[i] = a[0] + a[4];
                }
            }
        }
        """,
    )
    assert "return CrossGLFixedArray_float_5(" not in generated
    assert "return __crossgl_construct_CrossGLFixedArray_float_5(r[0]" in generated
    _compile(tmp_path, generated, "cs_6_2", "CSMain")


def test_row_major_buffer_member_keeps_its_row_layout(tmp_path):
    # buffer-row-major-non-square: the member was split into column-major words at another stride.
    generated = _hlsl(
        tmp_path,
        """
        shader Probe {
            struct Rows {
                @ row_major mat2x3 m;
                vec2 w;
            };
            compute {
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                buffer float* fin;
                buffer Rows* outp;
                void main() {
                    uint i = gl_GlobalInvocationID.x;
                    outp[i].m = mat2x3(fin[i], 1.0, 2.0, 3.0, fin[i] * 0.5, 4.0);
                    outp[i].w = vec2(outp[i].m[0][2], 1.0);
                }
            }
        }
        """,
    )
    assert "RWStructuredBuffer<Rows> outp" in generated
    assert re.search(r"row_major float3x2 m;", generated)
    assert "outp[i].m[2][0]" in generated
    _compile(tmp_path, generated, "cs_6_2", "CSMain")
