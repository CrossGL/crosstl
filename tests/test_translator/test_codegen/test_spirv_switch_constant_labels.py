"""SPIR-V OpSwitch labels written as named integral constants (GLSL 4.60 section 6.2; HLSL and MSL follow C++)."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def generate(source_code):
    return VulkanSPIRVCodeGen().generate(Parser(Lexer(source_code).tokens).parse())


def switch_literals(spv_code):
    match = re.search(r"OpSwitch %\d+ %\d+((?: -?\d+ %\d+)*)", spv_code)
    assert match, spv_code
    return [int(value) for value in re.findall(r" (-?\d+) %\d+", match.group(1))]


def assert_validates(spv_code, tmp_path):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not spirv_as or not spirv_val:
        pytest.skip("spirv-as and spirv-val are not installed")
    asm_path = tmp_path / "shader.spvasm"
    spv_path = tmp_path / "shader.spv"
    asm_path.write_text(spv_code)
    assemble = subprocess.run(
        [spirv_as, "--target-env", "vulkan1.1", str(asm_path), "-o", str(spv_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert assemble.returncode == 0, assemble.stderr
    validate = subprocess.run(
        [spirv_val, "--target-env", "vulkan1.1", str(spv_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr


def test_local_constant_case_labels_fold_to_literals(tmp_path):
    spv_code = generate("""
    shader Switches {
        fragment {
            vec4 main(uint rate @ location(0) @ flat) @ gl_FragColor {
                const uint PER_PIXEL = 0u;
                const uint PER_2X2 = 5u;
                const uint PER_4X4 = 10u;
                switch (rate) {
                    case PER_PIXEL:
                        return vec4(1.0, 0.0, 0.0, 1.0);
                    case PER_2X2:
                        return vec4(0.0, 1.0, 0.0, 1.0);
                    case PER_4X4:
                        return vec4(0.0, 0.0, 1.0, 1.0);
                    default:
                        return vec4(0.0);
                }
            }
        }
    }
    """)

    assert switch_literals(spv_code) == [0, 5, 10]
    assert_validates(spv_code, tmp_path)


def test_module_constant_case_labels_fold_to_literals(tmp_path):
    spv_code = generate("""
    shader Switches {
        const int FIRST = -1;
        const int SECOND = FIRST + 3;

        fragment {
            vec4 main(int mode @ location(0) @ flat) @ gl_FragColor {
                vec4 color = vec4(0.0);
                switch (mode) {
                    case FIRST:
                        color = vec4(1.0);
                        break;
                    case SECOND:
                        color = vec4(0.5);
                        break;
                    default:
                        break;
                }
                return color;
            }
        }
    }
    """)

    assert switch_literals(spv_code) == [-1, 2]
    assert_validates(spv_code, tmp_path)


def test_non_constant_case_label_is_still_refused():
    with pytest.raises(ValueError, match="scalar integer literals"):
        generate("""
        shader Switches {
            fragment {
                vec4 main(int mode @ location(0) @ flat, int other @ location(1) @ flat) @ gl_FragColor {
                    switch (mode) {
                        case other:
                            return vec4(1.0);
                        default:
                            return vec4(0.0);
                    }
                }
            }
        }
        """)


def test_hlsl_local_const_case_labels_translate_to_spirv(tmp_path):
    source = tmp_path / "rates.hlsl"
    source.write_text("""
    float4 main(nointerpolation uint rate : TEXCOORD0) : SV_TARGET
    {
        const uint PER_PIXEL = 0;
        const uint PER_2X1 = 4;
        switch (rate) {
            case PER_PIXEL:
                return float4(1.0, 0.0, 0.0, 1.0);
            case PER_2X1:
                return float4(0.0, 1.0, 0.0, 1.0);
            default:
                return float4(0.0, 0.0, 0.0, 1.0);
        }
    }
    """)

    spv_code = crosstl.translate(str(source), backend="vulkan")

    assert switch_literals(spv_code) == [0, 4]
    assert_validates(spv_code, tmp_path)
