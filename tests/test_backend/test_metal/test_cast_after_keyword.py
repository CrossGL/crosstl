"""A cast after ``return`` is not a temporary function object.

The Metal preprocessor scans for ``T(...)(...)``, a temporary of type ``T``
whose ``operator()`` is called. It also matched the keyword ``return``, so
``return (float)(1.0 / x);`` was refused as a temporary
``return(float)`` called with ``(1.0 / x)``. Keywords that can precede a
parenthesized expression are now excluded, and the cast is imported.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl

KERNEL = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    static float smith(float ndotv, float a)
    {
        return (float)(1.0 / (ndotv + sqrt(a + a - a * a)));
    }

    static half3 unpremultiply(float4 color)
    {
        return (half3)(color.rgb * 1.0f / color.a);
    }

    static float clamped(float x)
    {
        if ((bool)(x > 1.0)) { return 1.0; }
        return x;
    }

    kernel void k(device float4 *colors [[buffer(0)]],
                  device float *out [[buffer(1)]],
                  uint i [[thread_position_in_grid]])
    {
        out[i] = smith(out[i], 0.5) + float(unpremultiply(colors[i]).x)
            + clamped(out[i]);
    }
    """)


def _translate(tmp_path, backend):
    source = tmp_path / "casts.metal"
    source.write_text(KERNEL, encoding="utf-8")
    return crosstl.translate(str(source), backend=backend)


def test_casts_after_return_are_imported(tmp_path):
    code = _translate(tmp_path, "cgl")
    assert "return float(1.0 / (ndotv + sqrt(a + a - a * a)));" in code
    assert "return f16vec3(color.rgb * 1.0f / color.a);" in code


_VALIDATORS = {
    "opengl": (
        "casts.comp",
        [["glslangValidator", "-S", "comp", "casts.comp"]],
    ),
    "vulkan": (
        "casts.spvasm",
        [
            ["spirv-as", "--target-env", "vulkan1.1", "casts.spvasm", "-o", "c.spv"],
            ["spirv-val", "--target-env", "vulkan1.1", "c.spv"],
        ],
    ),
    "directx": (
        "casts.hlsl",
        [["dxc", "-T", "cs_6_2", "-enable-16bit-types", "-E", "CSMain", "casts.hlsl"]],
    ),
}


@pytest.mark.parametrize("backend", sorted(_VALIDATORS))
def test_casts_after_return_reach_every_target(tmp_path, backend):
    file_name, commands = _VALIDATORS[backend]
    code = _translate(tmp_path, backend)
    missing = [command[0] for command in commands if shutil.which(command[0]) is None]
    if missing:
        pytest.skip(f"{', '.join(missing)} not installed")
    (tmp_path / file_name).write_text(code, encoding="utf-8")
    for command in commands:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=300, cwd=tmp_path
        )
        assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
