"""A WaveSize contract on a compute entry whose GLSL subgroups are emulated in software."""

import shutil
import subprocess

import pytest

import crosstl.translator
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.opengl_values import OpenGLSoftwareSubgroupError


def _reduction(attribute):
    return f"""
    shader SoftwareWaveSize {{
        compute {{
            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
            buffer float* values;
            void main(){attribute} {{
                uint lane = gl_LocalInvocationID.x;
                values[lane] = WaveActiveSum(values[lane]);
            }}
        }}
    }}
    """


def _generate(attribute, **options):
    return GLSLCodeGen(**options).generate(
        crosstl.translator.parse(_reduction(attribute))
    )


def test_equal_wave_size_adds_no_hardware_subgroup_guard():
    with_contract = _generate(" @ WaveSize(32)", software_subgroup_width=32)
    without_contract = _generate("", software_subgroup_width=32)

    # Emulated subgroups have exactly 32 lanes, so the contract changes nothing.
    assert with_contract == without_contract
    assert "#define CROSSTL_SOFTWARE_SUBGROUP_WIDTH 32u" in with_contract
    assert "CROSSTL_REQUIRED_SUBGROUP_WIDTH" not in with_contract
    assert "gl_SubgroupSize" not in with_contract
    assert "GL_KHR_shader_subgroup" not in with_contract


@pytest.mark.parametrize("contract", [16, 64])
def test_unequal_wave_size_is_refused(contract):
    with pytest.raises(OpenGLSoftwareSubgroupError) as raised:
        _generate(f" @ WaveSize({contract})", software_subgroup_width=32)

    assert raised.value.reason == "hardware-contract-conflict"
    assert raised.value.software_subgroup_width == 32


def test_hardware_subgroups_keep_the_width_guard():
    generated = _generate(" @ WaveSize(32)")

    assert "#define CROSSTL_REQUIRED_SUBGROUP_WIDTH 32u" in generated
    assert "if (gl_SubgroupSize != CROSSTL_REQUIRED_SUBGROUP_WIDTH)" in generated
    assert "CROSSTL_SOFTWARE_SUBGROUP_WIDTH" not in generated


@pytest.mark.skipif(
    shutil.which("glslangValidator") is None, reason="glslangValidator is not on PATH"
)
def test_equal_wave_size_output_compiles(tmp_path):
    shader = tmp_path / "software_wave_size.comp"
    shader.write_text(_generate(" @ WaveSize(32)", software_subgroup_width=32))

    result = subprocess.run(
        ["glslangValidator", str(shader)], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stdout + result.stderr
