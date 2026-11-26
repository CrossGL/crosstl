"""GLSL output writes the HLSL and Slang interpolation spellings CrossGL accepts as GLSL qualifiers.

CrossGL accepts ``nointerpolation`` and the ``linear_*`` spellings beside GLSL's ``flat``, ``noperspective``,
``centroid`` and ``sample`` (``validation.INTERPOLATION_MODE_METADATA_NAMES``), and the Slang importer keeps them on
entry parameters. The GLSL generator dropped them, so ``nointerpolation float2 uv`` became a smoothly interpolated
``in vec2 uv;``. They are now ``flat``, ``centroid``, ``sample`` and ``noperspective``; plain ``linear`` is GLSL's
default interpolation.
"""

import shutil
import subprocess

import pytest

import crosstl

GLSLANG = shutil.which("glslangValidator")


def fragment(qualifier):
    return f"""
shader main {{
    fragment {{
        vec4 main({qualifier} vec2 uv @ TexCoord0) @ gl_FragColor {{
            return vec4(uv, 0.0, 1.0);
        }}
    }}
}}
"""


def translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def input_declaration(code):
    return next(line for line in code.splitlines() if line.endswith(" vec2 uv;"))


@pytest.mark.parametrize(
    ("qualifier", "expected"),
    [
        ("nointerpolation", "flat in"),
        ("linear_centroid", "centroid in"),
        ("linear_sample", "sample in"),
        ("linear_noperspective", "noperspective in"),
        ("linear_noperspective_centroid", "noperspective centroid in"),
    ],
)
@pytest.mark.parametrize("backend", ["opengl", "gles"])
def test_interpolation_spelling_becomes_the_glsl_qualifier(
    tmp_path, backend, qualifier, expected
):
    if backend == "gles" and expected not in ("flat in", "centroid in"):
        # GLSL ES has sample only from 3.20 and noperspective only under an NV extension; the ES version and
        # extension a qualifier needs are the version requirements' concern, not the spelling's.
        pytest.skip("qualifier outside GLSL ES 3.10")
    code = translate(tmp_path, fragment(qualifier), backend)

    assert f" {expected} " in input_declaration(code)
    if GLSLANG is not None:
        result = subprocess.run(
            [GLSLANG, "--stdin", "-S", "frag"],
            input=code,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_linear_keeps_the_default_interpolation(tmp_path):
    declaration = input_declaration(translate(tmp_path, fragment("linear"), "opengl"))

    assert declaration.endswith(" in vec2 uv;")
    assert not {"flat", "smooth", "noperspective", "centroid", "sample"} & set(
        declaration.split()
    )


def test_slang_nointerpolation_parameter_is_flat(tmp_path):
    path = tmp_path / "shader.slang"
    path.write_text(
        '[shader("fragment")]\n'
        "float4 main(nointerpolation float2 uv : TEXCOORD0) : SV_Target {\n"
        "    return float4(uv, 0.0, 1.0);\n"
        "}\n"
    )

    assert " flat in " in input_declaration(
        crosstl.translate(str(path), backend="opengl")
    )
