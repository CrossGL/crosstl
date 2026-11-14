"""An unqualified CrossGL array parameter is copied in, in GLSL output as in every other target (issue #3201).

GLSL 4.60 section 6.1.1 copies an ``in`` (unqualified) parameter into the callee, so a callee's writes to its array
parameter stay local, which is also what the HLSL, MSL, SPIR-V and Slang outputs do. The GLSL generator gave every
written fixed-array parameter ``inout``, because asking whether a helper writes its array parameter also marked the
parameter for writeback. Only a caller that passes caller-visible storage, such as ``&local``, keeps the writeback.
"""

import shutil
import subprocess

import pytest

import crosstl

COPY_IN = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer float* outp;
        float total(float v[4]) {
            v[0] = 0.0;
            return v[0] + v[1] + v[2] + v[3];
        }
        float relay(float v[4]) {
            return total(v) + v[0];
        }
        void main() {
            uint i = gl_GlobalInvocationID.x;
            float values[4];
            for (int k = 0; k < 4; k++) {
                values[k] = float(iin[i] + k);
            }
            outp[i] = total(values) * 10.0 + relay(values) + values[0];
        }
    }
}
"""

COPY_IN_FRAGMENT = """
shader Probe {
    fragment {
        float total(float v[2]) {
            v[0] = 0.0;
            return v[0] + v[1];
        }
        vec4 main() @ gl_FragColor {
            float values[2];
            values[0] = 0.25;
            values[1] = 0.5;
            return vec4(total(values), values[0], 0.0, 1.0);
        }
    }
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "probe.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _glslang(tmp_path, code, stage):
    if shutil.which("glslangValidator") is None:
        return
    shader = tmp_path / f"probe.{stage}"
    shader.write_text(code)
    result = subprocess.run(
        ["glslangValidator", str(shader)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_opengl_copies_written_array_parameters_in(tmp_path):
    code = _translate(tmp_path, COPY_IN, "opengl")
    assert "inout" not in code
    compact = " ".join(code.split())
    assert "float total(float v[4])" in compact
    assert "float relay(float v[4])" in compact
    _glslang(tmp_path, code, "comp")


def test_webgl_copies_written_array_parameters_in(tmp_path):
    code = _translate(tmp_path, COPY_IN_FRAGMENT, "webgl")
    assert "inout" not in code
    assert "float total(float v[2])" in " ".join(code.split())
    _glslang(tmp_path, code, "frag")


@pytest.mark.parametrize("backend", ["opengl", "directx", "slang"])
def test_targets_agree_that_array_parameters_are_values(tmp_path, backend):
    code = _translate(tmp_path, COPY_IN, backend)
    assert "inout float v[4]" not in " ".join(code.split())


def test_caller_visible_storage_keeps_its_writeback(tmp_path):
    code = _translate(
        tmp_path,
        """
        shader View {
            void update(float value[1]) { value[0] += 1.0; }
            void relay(float value[1]) { update(value); }
            compute {
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                void main(RWStructuredBuffer<float> result @buffer(0)) {
                    float total = 2.0;
                    relay(&total);
                    result[0] = total;
                }
            }
        }
        """,
        "opengl",
    )
    compact = " ".join(code.split())
    # &total is the caller's own storage, so relay, and update, which relay forwards it to, write it back.
    assert "void update(inout float value[1])" in compact
    assert "void relay(inout float value[1])" in compact
    _glslang(tmp_path, code, "comp")
