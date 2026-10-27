"""CrossGL ``%`` on floating-point operands is the truncated remainder in every generator (issue #3293).

``x % y`` is ``x - y * trunc(x / y)``, whose sign is the dividend's, as HLSL ``%`` and ``fmod``, MSL ``fmod``,
C ``fmod``, Slang ``%``, WGSL ``%`` and SPIR-V ``OpFRem`` compute it. Slang, CUDA and HIP output floored it instead,
as GLSL ``mod`` does; Metal kept a floating ``%=`` that the Metal compiler rejects, and OpenGL refused it. GLSL
``mod`` keeps its floored meaning everywhere.
"""

import shutil
import subprocess

import pytest

import crosstl

REMAINDER = """shader P {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer float* outp;

        void main() {
            uint i = gl_GlobalInvocationID.x;
            float a = float(iin[i]) - 7.5;
            float d = a * 3.0;
            d %= 4.0;
            vec2 v = vec2(a, -a) % vec2(4.0, -2.0);
            outp[i] = d + v.x * 8.0 + v.y * 64.0 + mod(a, 2.0);
        }
    }
}
"""

SIDE_EFFECT_TARGET = """shader P {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;

        void main() {
            float t[2];
            int k = 0;
            t[0] = outp[0];
            t[1] = outp[1];
            t[k++] %= 2.0;
            outp[0] = t[0] + float(k);
        }
    }
}
"""

EXPECTED = {
    "metal": ["d = fmod(d, 4.0);", "fmod(float2(a, -a), float2(4.0, -2.0))"],
    "opengl": [
        "d = crossgl_remainder(d, 4.0);",
        "crossgl_remainder(vec2(a, (-a)), vec2(4.0, (-2.0)))",
        "return x - y * trunc(x / y);",
    ],
    "slang": ["d %= 4.0;", "float2(a, -a) % float2(4.0, -2.0)"],
    "cuda": [
        "d = fmodf(d, 4.0);",
        "make_float2(fmodf(lhs.x, rhs.x), fmodf(lhs.y, rhs.y))",
    ],
    "hip": [
        "d = fmodf(d, 4.0);",
        "make_float2(fmodf(lhs.x, rhs.x), fmodf(lhs.y, rhs.y))",
    ],
    "directx": ["d %= 4.0;"],
}

# GLSL mod() keeps floor semantics: the only floor in each output belongs to it.
FLOORED_MOD = {
    "metal": "((a) - ((2.0) * floor((a) / (2.0))))",
    "opengl": "mod(a, 2.0)",
    "slang": "((a) - (floor((a) / (2.0)) * (2.0)))",
    "cuda": "cgl_mod_float(a, 2.0)",
    "hip": "cgl_mod_float(a, 2.0)",
}


def _translate(tmp_path, source, backend):
    path = tmp_path / "remainder.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize("backend", sorted(EXPECTED))
def test_float_remainder_is_truncated(tmp_path, backend):
    code = _translate(tmp_path, REMAINDER, backend)
    for fragment in EXPECTED[backend]:
        assert fragment in code


@pytest.mark.parametrize("backend", sorted(FLOORED_MOD))
def test_mod_builtin_keeps_floor_semantics(tmp_path, backend):
    code = _translate(tmp_path, REMAINDER, backend)
    assert FLOORED_MOD[backend] in code
    floor_name = "floorf" if backend in {"cuda", "hip"} else "floor"
    # The only floor is mod(a, 2.0)'s (CUDA and HIP: inside the cgl_mod_float
    # helper); GLSL output calls its own mod.
    assert code.count(f"{floor_name}(") == (0 if backend == "opengl" else 1)


def test_spirv_uses_frem_for_the_operator_and_fmod_for_mod(tmp_path):
    code = _translate(tmp_path, REMAINDER, "vulkan")
    assert code.count("OpFRem") == 2
    assert code.count("OpFMod") == 1


@pytest.mark.parametrize("backend", ["metal", "opengl", "cuda", "hip"])
def test_side_effecting_remainder_target_is_refused(tmp_path, backend):
    with pytest.raises(Exception, match="float-remainder-assignment-unsupported"):
        _translate(tmp_path, SIDE_EFFECT_TARGET, backend)


@pytest.mark.parametrize("backend", ["slang", "directx"])
def test_native_remainder_assignment_evaluates_the_target_once(tmp_path, backend):
    code = _translate(tmp_path, SIDE_EFFECT_TARGET, backend)
    assert code.count("k++") == 1
    assert "%= 2.0;" in code


def test_webgl_fragment_remainder_assignment(tmp_path):
    source = """shader P {
        fragment {
            uniform float scale;
            vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
                float x = uv.x * scale;
                x %= 0.25;
                return vec4(x, uv.y % 0.5, 0.0, 1.0);
            }
        }
    }"""
    code = _translate(tmp_path, source, "webgl")
    assert "x = crossgl_remainder(x, 0.25);" in code
    assert "crossgl_remainder(" in code and "%" not in code.replace("%=", "")


@pytest.mark.skipif(shutil.which("glslangValidator") is None, reason="glslang")
def test_glslang_accepts_the_remainders(tmp_path):
    shader = tmp_path / "remainder.comp"
    shader.write_text(_translate(tmp_path, REMAINDER, "opengl"))
    result = subprocess.run(
        ["glslangValidator", str(shader)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout


@pytest.mark.skipif(shutil.which("xcrun") is None, reason="Metal toolchain")
def test_metal_compiles_the_remainders(tmp_path):
    source = tmp_path / "remainder.metal"
    source.write_text(_translate(tmp_path, REMAINDER, "metal"))
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(source)]
        + ["-o", str(tmp_path / "remainder.air")],
        capture_output=True,
        text=True,
    )
    if "unable to find utility" in result.stderr:
        pytest.skip("the Metal compiler is not installed")
    assert result.returncode == 0, result.stderr
