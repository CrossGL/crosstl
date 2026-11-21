"""Metal output never names an explicit ``@stage_entry`` function ``main``.

The Metal compiler rejects kernel, vertex and fragment functions called ``main``,
so an explicit entry with that name gets the same stage prefix as an entry
without the attribute. Explicit entries with any other name keep it.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

import crosstl
from crosstl.translator.codegen.metal_codegen import MetalCodeGen

COMPUTE = """
shader Test {
    compute {
        @stage_entry
        void main(RWStructuredBuffer<float> out_ @buffer(1), uint id @gl_GlobalInvocationID) {
            out_[id] = 1.0;
        }
    }
}
"""

FRAGMENT = """
shader Test {
    fragment {
        @stage_entry
        vec4 main(vec4 pos @gl_FragCoord) @gl_FragColor {
            return pos;
        }
    }
}
"""

VERTEX = """
shader Test {
    vertex {
        @stage_entry
        vec4 main(uint vid @gl_VertexID) @gl_Position {
            return vec4(0.0, 0.0, 0.0, 1.0);
        }
    }
}
"""


def _generate(shader, stage):
    return MetalCodeGen().generate_stage(crosstl.translator.parse(shader), stage)


def _compile_metal(generated):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not available")
    lookup = subprocess.run(
        [xcrun, "-f", "metal"], capture_output=True, check=False, text=True
    )
    if lookup.returncode != 0:
        pytest.skip("Metal compiler is not available")
    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = Path(temp_dir) / "shader.metal"
        output_path = Path(temp_dir) / "shader.air"
        source_path.write_text(generated, encoding="utf-8")
        result = subprocess.run(
            [
                xcrun,
                "-sdk",
                "macosx",
                "metal",
                "-c",
                str(source_path),
                "-o",
                str(output_path),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode != 0 and "metal toolchain" in result.stderr.lower():
            pytest.skip("Metal toolchain is not installed")
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "shader, stage, signature",
    [
        (COMPUTE, "compute", "kernel void kernel_main("),
        (FRAGMENT, "fragment", "fragment float4 fragment_main("),
        (VERTEX, "vertex", "vertex float4 vertex_main("),
    ],
)
def test_metal_explicit_stage_entry_named_main_gets_stage_name(
    shader, stage, signature
):
    generated = _generate(shader, stage)

    assert signature in generated
    assert " main(" not in generated
    _compile_metal(generated)


def test_metal_explicit_stage_entry_named_main_avoids_helper_collision():
    shader = """
    shader Test {
        float kernel_main(float x) {
            return x * 2.0;
        }

        compute {
            @stage_entry
            void main(RWStructuredBuffer<float> out_ @buffer(1), uint id @gl_GlobalInvocationID) {
                out_[id] = kernel_main(1.0);
            }
        }
    }
    """

    generated = _generate(shader, "compute")

    assert "float kernel_main(float x)" in generated
    assert "kernel void kernel_main_2(" in generated
    assert " main(" not in generated
    _compile_metal(generated)


def test_metal_explicit_stage_entry_keeps_other_names():
    shader = """
    shader Test {
        compute {
            @stage_entry
            void fill(RWStructuredBuffer<float> out_ @buffer(1), uint id @gl_GlobalInvocationID) {
                out_[id] = 1.0;
            }
        }
    }
    """

    generated = _generate(shader, "compute")

    assert "kernel void fill(" in generated
