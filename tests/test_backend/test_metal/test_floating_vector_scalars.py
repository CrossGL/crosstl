"""MSL converts a scalar operand to a floating vector's element type (MSL section 2.13).

``half3 * float`` is a ``half3`` in MSL, whereas a ``float16`` and a ``float`` make a ``float`` operation in CrossGL, so
the importer spells the scalar conversion. A scalar of the element's rank or below needs no conversion.
"""

import shutil
import subprocess

import pytest

import crosstl

HEADER = "#include <metal_stdlib>\nusing namespace metal;\n"


def _translate(tmp_path, source, backend="cgl"):
    path = tmp_path / "vectors.metal"
    path.write_text(HEADER + source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize(
    "operator", ["+", "-", "*", "/", "<", "<=", ">", ">=", "==", "!="]
)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("vector", ["half2", "half3", "half4"])
def test_wider_floating_scalar_converts_to_the_vector_element(
    tmp_path, operator, reverse, vector
):
    left, right = ("scalar", "value") if reverse else ("value", "scalar")
    source = f"""{vector} combine({vector} value, float scalar) {{
    auto result = {left} {operator} {right};
    return {vector}(result);
}}
"""
    generated = _translate(tmp_path, source)
    expected = (
        f"float16(scalar) {operator} value"
        if reverse
        else f"value {operator} float16(scalar)"
    )
    assert expected in generated


def test_floating_vector_result_keeps_the_element_type(tmp_path):
    source = """half3 scale(half3 value, float scalar) {
    auto product = value * scalar;
    return product;
}
"""
    generated = _translate(tmp_path, source)
    assert "f16vec3 product = value * float16(scalar);" in generated


@pytest.mark.parametrize("operator", ["+=", "-=", "*=", "/="])
def test_floating_vector_compound_assignment_converts_the_scalar(tmp_path, operator):
    source = f"""half3 scale(half3 value, float scalar) {{
    value {operator} scalar;
    return value;
}}
"""
    generated = _translate(tmp_path, source)
    assert f"value {operator} float16(scalar);" in generated


def test_narrower_or_integer_scalars_need_no_conversion(tmp_path):
    source = """float3 widen(float3 value, half scalar, int count) {
    return value * scalar + value * count;
}
half3 count(half3 value, int count, short small) {
    return value * count + value * small;
}
"""
    generated = _translate(tmp_path, source)
    assert "value * scalar + value * count" in generated
    assert "value * count + value * small" in generated
    assert "float(scalar)" not in generated


def test_floating_vector_scalar_arithmetic_compiles_as_half(tmp_path):
    source = """fragment half4 shade(float4 position [[position]]) {
    half3 color = half3(position.xyz);
    constexpr float bands = 2.0;
    color = saturate(fmod(color * bands, 1.0));
    half3 scaled = color * position.w;
    scaled *= position.w;
    return half4(color + scaled, 1.0h);
}
"""
    generated = _translate(tmp_path, source, backend="metal")
    assert "color * half(bands)" in generated
    assert "scaled *= half(position.w);" in generated
    assert "float3(color)" not in generated
    if shutil.which("xcrun") is None:
        return
    artifact = tmp_path / "translated.metal"
    artifact.write_text(generated)
    compiled = subprocess.run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-c",
            str(artifact),
            "-o",
            str(tmp_path / "translated.air"),
        ],
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
