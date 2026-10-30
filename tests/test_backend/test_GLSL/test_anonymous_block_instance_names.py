"""GLSL import names an anonymous buffer block's instance so that no variable of the program shadows it.

The importer derived the instance name from the block name (``Data`` became ``data``) and checked it against global
declarations only, so a local or parameter called ``data`` captured every member access in its function and the
translation read ``data.values`` from a ``float``.
"""

import shutil
import subprocess

import pytest

import crosstl

COMPUTE = """#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Data { float values[]; };
float twice(float data) { return data * 2.0 + values[1]; }
void main() {
    float data = values[0];
    values[0] = twice(data) + 1.0;
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.comp"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return " ".join(code.split())


def test_instance_name_avoids_locals_and_parameters(tmp_path):
    code = _flat(_translate(tmp_path, COMPUTE, "cgl"))
    assert "} data_2;" in code
    assert "float data = data_2.values[0];" in code
    assert "data_2.values[0] = (twice(data) + 1.0);" in code
    assert "return ((data * 2.0) + data_2.values[1]);" in code
    assert "data.values" not in code


def test_instance_name_skips_every_taken_suffix(tmp_path):
    source = COMPUTE.replace(
        "    float data = values[0];\n",
        "    float data_2 = 0.0;\n    float data = values[0] + data_2;\n",
    )
    code = _flat(_translate(tmp_path, source, "cgl"))
    assert "} data_3;" in code
    assert "float data = (data_3.values[0] + data_2);" in code


def test_instance_name_without_collision_is_unchanged(tmp_path):
    source = COMPUTE.replace("data", "x")
    code = _flat(_translate(tmp_path, source, "cgl"))
    assert "} data;" in code
    assert "float x = data.values[0];" in code


@pytest.mark.skipif(
    shutil.which("glslangValidator") is None, reason="glslangValidator unavailable"
)
def test_glsl_output_compiles(tmp_path):
    output = tmp_path / "out.comp"
    output.write_text(_translate(tmp_path, COMPUTE, "opengl"))
    result = subprocess.run(
        ["glslangValidator", "-S", "comp", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
