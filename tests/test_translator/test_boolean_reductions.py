"""``any`` and ``all`` of numeric vectors and scalars on targets that reduce only booleans.

CrossGL takes HLSL's forms: ``any(x)``/``all(x)`` test the components of a numeric or boolean scalar or vector
against zero. GLSL and GLSL ES accept only ``bvecN`` operands, WGSL ``bool``/``vecN<bool>`` and MSL ``bool``/``boolN``,
so the generators emitted ``any(a)`` on an ``ivec3`` (glslang: "no matching overloaded function found"), ``all(u)``
on a ``uint4`` (metal: "no known conversion from 'uint4' to 'bool4'") and ``any(a)`` on ``vec3<i32>`` (naga:
"Relational argument is not a boolean vector").
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.boolean_reductions import lower_boolean_reductions

COMPUTE = """
RWStructuredBuffer<float> outp : register(u0);
[numthreads(1, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    int3 a = int3(id);
    float3 b = float3(id) - 1.0;
    bool2 c = bool2(true, false);
    uint4 u = uint4(1, 2, 3, id.x);
    float s = 0.0;
    if (any(a)) s += 1.0;
    if (all(b)) s += 2.0;
    if (any(c)) s += 4.0;
    if (all(s)) s += 8.0;
    if (all(u)) s += 16.0;
    if (!any(b.xy)) s += 32.0;
    if (all(any(a))) s += 64.0;
    outp[0] = s;
}
"""

FRAGMENT = """
shader S {
    fragment {
        uniform ivec2 mask;
        uniform vec4 tint;
        vec4 main(vec2 uv @ TEXCOORD0) @ gl_FragColor {
            float k = 0.0;
            if (any(mask)) { k += 0.25; }
            if (all(tint.xyz)) { k += 0.5; }
            if (!all(uv)) { k += 0.125; }
            if (any(k)) { k += 0.0625; }
            return vec4(k);
        }
    }
}
"""


def _translate(tmp_path, source, backend, suffix="hlsl"):
    path = tmp_path / f"shader.{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _flat(code):
    return " ".join(code.split())


def _run(command, cwd):
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=cwd, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _validate(tmp_path, backend, code, stage="comp"):
    if backend == "directx":
        if shutil.which("dxc") is None:
            pytest.skip("dxc is not installed")
        (tmp_path / "out.hlsl").write_text(code)
        _run(["dxc", "-T", "cs_6_0", "-E", "CSMain", "out.hlsl"], tmp_path)
    elif backend in ("opengl", "webgl"):
        if shutil.which("glslangValidator") is None:
            pytest.skip("glslangValidator is not installed")
        (tmp_path / f"out.{stage}").write_text(code)
        _run(["glslangValidator", f"out.{stage}"], tmp_path)
    elif backend == "wgsl":
        if shutil.which("naga") is None:
            pytest.skip("naga is not installed")
        (tmp_path / "out.wgsl").write_text(code)
        _run(["naga", "out.wgsl"], tmp_path)
    elif backend == "metal":
        if shutil.which("xcrun") is None:
            pytest.skip("the Metal compiler is not installed")
        (tmp_path / "out.metal").write_text(code)
        _run(["xcrun", "metal", "-c", "out.metal", "-o", "out.air"], tmp_path)
    elif backend == "vulkan":
        assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
        if assembler is None or validator is None:
            pytest.skip("SPIRV-Tools is not installed")
        (tmp_path / "out.spvasm").write_text(code)
        _run(
            [assembler, "--target-env", "vulkan1.1", "out.spvasm", "-o", "out.spv"],
            tmp_path,
        )
        _run([validator, "--target-env", "vulkan1.1", "out.spv"], tmp_path)


def test_glsl_converts_numeric_operands_to_boolean_vectors(tmp_path):
    code = _flat(_translate(tmp_path, COMPUTE, "opengl"))
    assert "any(bvec3(a))" in code
    assert "all(bvec3(b))" in code
    assert "all(bvec4(u))" in code
    assert "any(bvec2(b.xy))" in code
    assert "any(c)" in code and "bvec2(c)" not in code
    assert "any(a)" not in code and "all(u)" not in code


def test_glsl_reduces_a_scalar_to_its_boolean_value(tmp_path):
    code = _flat(_translate(tmp_path, COMPUTE, "opengl"))
    assert "if (bool(s))" in code
    assert "all(s)" not in code
    # The reduction of a reduction is that reduction.
    assert "all(any(" not in code


def test_metal_converts_numeric_operands_to_boolean_vectors(tmp_path):
    code = _flat(_translate(tmp_path, COMPUTE, "metal"))
    assert "any(bool3(a))" in code
    assert "all(bool4(u))" in code
    assert "any(c)" in code


def test_wgsl_converts_numeric_operands_to_boolean_vectors(tmp_path):
    code = _flat(_translate(tmp_path, COMPUTE, "wgsl"))
    assert "any(vec3<bool>(a))" in code
    assert "all(vec4<bool>(u))" in code
    assert "any(c)" in code


@pytest.mark.parametrize("backend", ["directx", "slang"])
def test_hlsl_family_keeps_numeric_reductions(tmp_path, backend):
    code = _flat(_translate(tmp_path, COMPUTE, backend))
    assert "any(a)" in code
    assert "all(u)" in code
    assert "all(s)" in code


@pytest.mark.parametrize("backend", ["directx", "opengl", "wgsl", "metal", "vulkan"])
def test_numeric_reductions_validate(tmp_path, backend):
    _validate(tmp_path, backend, _translate(tmp_path, COMPUTE, backend))


@pytest.mark.parametrize("backend", ["opengl", "webgl", "wgsl", "metal"])
def test_fragment_numeric_reductions_validate(tmp_path, backend):
    code = _translate(tmp_path, FRAGMENT, backend, suffix="cgl")
    if backend in ("opengl", "webgl"):
        assert "any(bvec2(mask))" in code
        assert "all(bvec3(tint.xyz))" in code
    _validate(tmp_path, backend, code, stage="frag")


def test_pass_runs_only_for_boolean_only_targets():
    source = """
    shader S {
        compute {
            void main() {
                ivec3 a = ivec3(1, 0, 2);
                bvec3 m = bvec3(true, false, true);
                float s = 1.0;
                bool r = any(a) || all(m) || any(s);
            }
        }
    }
    """
    program = Parser(Lexer(source).get_tokens()).parse()
    assert lower_boolean_reductions(program, "directx") == 0
    assert lower_boolean_reductions(program, "vulkan") == 0
    assert lower_boolean_reductions(program, "cgl") == 0
    # ``any(a)`` is converted and ``any(s)`` reduced; ``all(m)`` already reduces a boolean vector.
    assert lower_boolean_reductions(program, "opengl") == 2
