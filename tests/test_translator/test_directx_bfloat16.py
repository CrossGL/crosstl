import json
import shutil
import struct
import subprocess
import textwrap

import pytest

import crosstl.translator
from crosstl.project import (
    build_runtime_artifact_manifest,
    build_runtime_loader_manifest,
    build_runtime_package,
    load_project_config,
    translate_project,
)
from crosstl.translator.codegen.directx_codegen import (
    DirectXBFloat16UnsupportedError,
    HLSLCodeGen,
)

BFLOAT_RESOURCE_SHADER = """
shader ExactBFloatStorage {
    StructuredBuffer<bfloat16_t> input;
    RWStructuredBuffer<bfloat16_t> output;

    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            bfloat16_t value = input[0];
            output[0] = value;
        }
    }
}
"""


def _float32_from_bits(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


@pytest.mark.parametrize(
    ("float_bits", "bfloat_bits"),
    [
        (0x00000000, 0x0000),
        (0x80000000, 0x8000),
        (0x3F800000, 0x3F80),
        (0x3F807FFF, 0x3F80),
        (0x3F808000, 0x3F80),
        (0x3F808001, 0x3F81),
        (0x3F818000, 0x3F82),
        (0x7F800000, 0x7F80),
        (0xFF800000, 0xFF80),
        (0x7F800001, 0x7FC0),
    ],
)
def test_directx_bfloat16_constant_rounding_is_binary32_rne(float_bits, bfloat_bits):
    assert (
        HLSLCodeGen.bfloat16_bits_for_float(_float32_from_bits(float_bits))
        == bfloat_bits
    )


def test_directx_bfloat16_numeric_and_bitcast_helpers_are_exact():
    shader = """
    shader ExactBFloatConversions {
        bfloat16_t from_float(float value) {
            return bfloat16_t(value);
        }

        float to_float(bfloat16_t value) {
            return float(value);
        }

        bfloat16_t from_bits(uint16_t bits) {
            return as_type<bfloat16_t>(bits);
        }

        uint16_t to_bits(bfloat16_t value) {
            return as_type<uint16_t>(value);
        }
    }
    """

    generated = HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert "uint from_float(float value)" in generated
    assert "float to_float(uint value)" in generated
    assert "uint from_bits(min16uint bits)" in generated
    assert "min16uint to_bits(uint value)" in generated
    assert "uint roundingBias = 0x7fffu + (upperBits & 1u);" in generated
    assert "absoluteBits > 0x7f800000u" in generated
    assert "(upperBits | 0x40u) & 0xffffu" in generated
    assert "asfloat((value & 0xffffu) << 16u)" in generated
    assert "__crossgl_bfloat16_from_uint16(min16uint(bits))" in generated
    assert "__crossgl_bfloat16_to_uint16(uint(value))" in generated
    assert "half" not in generated


def test_directx_bfloat16_storage_uses_native_uint16_and_rejects_dx11():
    ast = crosstl.translator.parse(BFLOAT_RESOURCE_SHADER)

    generated = HLSLCodeGen(target_profile="dx12").generate(ast)

    assert "StructuredBuffer<uint16_t> input" in generated
    assert "RWStructuredBuffer<uint16_t> output" in generated
    assert "StructuredBuffer<min16uint>" not in generated
    assert "Shader Model 6.2" in generated
    assert "dxc -enable-16bit-types" in generated

    with pytest.raises(DirectXBFloat16UnsupportedError) as exc_info:
        HLSLCodeGen(target_profile="dx11").generate(ast)

    diagnostic = exc_info.value
    assert diagnostic.project_diagnostic_code == (
        "project.translate.directx-bfloat16-unsupported"
    )
    assert diagnostic.missing_capabilities == ("directx.exact-bfloat16-lowering",)
    assert diagnostic.target_profile == "dx11"
    assert diagnostic.reason == "target-profile-lacks-native-16bit-storage"


def test_directx_bfloat16_unproven_builtin_fails_closed():
    shader = """
    shader UnsupportedBFloatBuiltin {
        bfloat16_t apply_sine(bfloat16_t value) {
            return sin(value);
        }
    }
    """

    with pytest.raises(DirectXBFloat16UnsupportedError) as exc_info:
        HLSLCodeGen().generate(crosstl.translator.parse(shader))

    assert exc_info.value.operation == "sin"
    assert exc_info.value.reason == "unsupported-bfloat16-builtin"


def test_mlx_style_metal_bfloat16_project_lowers_exactly_to_directx(tmp_path):
    repo = tmp_path / "mlx-bfloat16"
    kernels = repo / "kernels"
    kernels.mkdir(parents=True)
    (kernels / "roundtrip.metal").write_text(
        textwrap.dedent("""
            #include <metal_stdlib>
            using namespace metal;
            typedef bfloat bfloat16_t;

            kernel void roundtrip(
                const device bfloat16_t* input [[buffer(0)]],
                device bfloat16_t* output [[buffer(1)]],
                uint index [[thread_position_in_grid]]) {
                bfloat16_t value = input[index];
                uint16_t bits = as_type<uint16_t>(value);
                bfloat16_t restored = as_type<bfloat16_t>(bits);
                float expanded = float(restored);
                output[index] = bfloat16_t(expanded + 1.0f);
            }
            """).strip(),
        encoding="utf-8",
    )
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["kernels"]
            include = ["kernels/*.metal"]
            targets = ["directx"]
            output_dir = "translated"
            """).strip(),
        encoding="utf-8",
    )

    report = translate_project(load_project_config(repo), format_output=False)
    payload = report.to_json()

    assert payload["diagnostics"] == []
    assert payload["summary"]["translatedCount"] == 1
    assert payload["summary"]["failedCount"] == 0
    artifact = payload["artifacts"][0]
    assert artifact["requiredCapabilities"] == ["directx.native-16bit-types"]
    generated = (repo / artifact["path"]).read_text(encoding="utf-8")
    assert "StructuredBuffer<uint16_t> input" in generated
    assert "RWStructuredBuffer<uint16_t> output" in generated
    assert "__crossgl_bfloat16_from_float" in generated
    assert "__crossgl_bfloat16_to_float" in generated
    assert "__crossgl_bfloat16_from_uint16" in generated
    assert "__crossgl_bfloat16_to_uint16" in generated
    assert "StructuredBuffer<half>" not in generated

    report_path = repo / "portability-report.json"
    report.write_json(report_path)
    manifest = build_runtime_artifact_manifest(report_path)
    manifest_path = repo / "runtime-artifact-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    package_dir = repo / "runtime-package"
    build_runtime_package(manifest_path, package_dir)
    loader = build_runtime_loader_manifest(package_dir / "runtime-package.json")
    load_unit = loader["loadUnits"][0]
    load_metadata = load_unit["loadSteps"][0]["metadata"]
    command_input = load_unit["loadSteps"][-1]["metadata"]["commandInput"]

    assert load_metadata["targetProfiles"] == ["directx-12"]
    assert load_metadata["minimumShaderModel"] == "6.2"
    assert load_metadata["compilerArguments"] == ["-enable-16bit-types"]
    assert load_metadata["entryProfiles"] == [{"entry": "CSMain", "profile": "cs_6_2"}]
    assert command_input["shaderModel"] == "cs_6_2"
    assert command_input["commands"][0][1:5] == [
        "-T",
        "cs_6_2",
        "-enable-16bit-types",
        "-E",
    ]


def test_project_bfloat16_failure_reports_actionable_lowering_contract(tmp_path):
    repo = tmp_path / "unsupported-bfloat16"
    kernels = repo / "kernels"
    kernels.mkdir(parents=True)
    (kernels / "unsupported.cgl").write_text(
        textwrap.dedent("""
            shader UnsupportedBFloatBuiltin {
                bfloat16_t apply_sine(bfloat16_t value) {
                    return sin(value);
                }
            }
            """).strip(),
        encoding="utf-8",
    )
    (repo / "crosstl.toml").write_text(
        textwrap.dedent("""
            [project]
            source_roots = ["kernels"]
            include = ["kernels/*.cgl"]
            targets = ["directx"]
            output_dir = "translated"
            """).strip(),
        encoding="utf-8",
    )

    payload = translate_project(
        load_project_config(repo), format_output=False
    ).to_json()

    assert payload["summary"]["translatedCount"] == 0
    assert payload["summary"]["failedCount"] == 1
    diagnostic = next(
        item
        for item in payload["diagnostics"]
        if item["code"] == "project.translate.directx-bfloat16-unsupported"
    )
    assert diagnostic["details"] == {
        "bfloat16Lowering": {
            "operation": "sin",
            "reason": "unsupported-bfloat16-builtin",
            "sourceType": "bfloat16_t",
        },
        "sourcePath": "kernels/unsupported.cgl",
        "targetArtifact": "translated/directx/kernels/unsupported.hlsl",
    }


def test_directx_bfloat16_storage_validates_with_dxc_when_available(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")

    generated = HLSLCodeGen(target_profile="dx12").generate(
        crosstl.translator.parse(BFLOAT_RESOURCE_SHADER)
    )
    shader_path = tmp_path / "bfloat16.hlsl"
    output_path = tmp_path / "bfloat16.dxil"
    shader_path.write_text(generated, encoding="utf-8")
    validation = subprocess.run(
        [
            dxc,
            "-T",
            "cs_6_2",
            "-E",
            "CSMain",
            "-enable-16bit-types",
            str(shader_path),
            "-Fo",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert validation.returncode == 0, validation.stdout + validation.stderr
    assert output_path.exists()
