"""Untyped globals must retain their initializer through the public translator."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.hlsl_refusals import DirectXLoweringRefused


def translate(tmp_path, declarations, use="scale"):
    source = tmp_path / "shader.cgl"
    source.write_text(
        f"shader S {{ {declarations} compute {{ buffer float data[]; "
        f"void main() {{ data[0] = float({use}); }} }} }}",
        encoding="utf-8",
    )
    return crosstl.translate(str(source), backend="directx")


@pytest.mark.parametrize("binding", ["let", "var"])
@pytest.mark.parametrize(
    "value, typename, use",
    [
        ("2.0", "float", "scale"),
        ("2", "int", "scale"),
        ("true", "bool", "scale"),
        ("vec3(1.0, 2.0, 3.0)", "float3", "scale.x"),
    ],
)
def test_global_type_and_initializer_survive_translation(
    tmp_path, binding, value, typename, use
):
    output = translate(tmp_path, f"{binding} scale = {value};", use)
    assert re.search(rf"\bstatic {typename} scale\s*=", output)
    assert "None scale" not in output
    assert re.search(r"scale\s*=.*(?:2|true|float3)", output)


def test_later_global_uses_the_inferred_global_type(tmp_path):
    output = translate(tmp_path, "let base = 2.0; let scale = base * 3.0;")
    assert "float base = 2.0;" in output
    assert re.search(r"float scale\s*=.*base.*3", output)


def test_uninferable_global_is_refused(tmp_path):
    with pytest.raises(
        DirectXLoweringRefused, match="global.*scale.*cannot be inferred"
    ):
        translate(tmp_path, "let scale = missing_symbol;")


def test_explicit_constant_still_uses_its_existing_lowering(tmp_path):
    output = translate(tmp_path, "const float scale = 2.0;")
    assert "static const float scale = 2.0;" in output


@pytest.mark.skipif(shutil.which("dxc") is None, reason="DXC is not installed")
@pytest.mark.parametrize("binding", ["let", "var"])
@pytest.mark.parametrize(
    "value, use",
    [
        ("2.0", "scale"),
        ("2", "scale"),
        ("true", "scale"),
        ("vec3(1.0, 2.0, 3.0)", "scale.x"),
    ],
)
def test_inferred_global_compiles_with_dxc(tmp_path, binding, value, use):
    output = translate(tmp_path, f"{binding} scale = {value};", use)
    shader = tmp_path / "shader.hlsl"
    shader.write_text(output, encoding="utf-8")
    binary = tmp_path / "shader.dxil"
    result = subprocess.run(
        [
            shutil.which("dxc"),
            "-T",
            "cs_6_0",
            "-E",
            "CSMain",
            "-Fo",
            str(binary),
            str(shader),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert binary.stat().st_size > 0
