"""WGSL output writes barriers in helper functions that only compute entry points reach.

WGSL output used to refuse every barrier outside an entry point ("keep barriers directly in compute entry points").
A helper that only compute entries reach, directly or through other helpers, is now written in the compute stage's
context, so its barriers become ``workgroupBarrier()`` and its implicit-LOD lookups use level 0 as the entry point's
own do. A barrier in a helper a vertex or fragment entry also reaches is still refused.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.refusals import WGSLRefusal

EXCHANGE = """
shader Exchange {
    compute {
        layout(local_size_x = 2, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer float* outp;
        shared float tile[2];

        void stash(float x) {
            tile[0] = x;
            tile[1] = x * 0.5;
        }

        float exchange(float x) {
            stash(x);
            barrier();
            return tile[0] * 2.0 + tile[1];
        }
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = exchange(float(iin[i])) + 1.0;
        }
    }
}
"""


def _translate(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend="wgsl")


def _naga(code, tmp_path):
    naga = shutil.which("naga")
    if naga is None:
        return
    (tmp_path / "out.wgsl").write_text(code)
    result = subprocess.run(
        [naga, "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr + code


def test_a_stage_local_helper_keeps_its_barrier(tmp_path):
    code = _translate(tmp_path, EXCHANGE)
    helper = code.split("fn exchange(", 1)[1].split("\n}\n", 1)[0]
    assert "workgroupBarrier();" in helper
    _naga(code, tmp_path)


def test_a_program_helper_reached_through_another_helper_keeps_its_barrier(tmp_path):
    source = """
    shader Chain {
        shared float tile[4];
        void sync() {
            memoryBarrierShared();
            barrier();
        }
        float load(uint i) {
            sync();
            return tile[i];
        }
        compute {
            layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;
            buffer float* outp;
            void main() {
                uint i = gl_LocalInvocationID.x;
                tile[i] = float(i);
                outp[i] = load(3u - i);
            }
        }
    }
    """
    code = _translate(tmp_path, source)
    helper = code.split("fn sync(", 1)[1].split("\n}\n", 1)[0]
    assert "workgroupBarrier();" in helper
    _naga(code, tmp_path)


def test_a_compute_helper_samples_at_level_zero(tmp_path):
    source = """
    shader HelperSample {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer vec4* outp;
            uniform sampler2D colorMap;
            vec4 fetch(vec2 uv) {
                return texture(colorMap, uv);
            }
            void main() {
                outp[gl_GlobalInvocationID.x] = fetch(vec2(0.5, 0.5));
            }
        }
    }
    """
    code = _translate(tmp_path, source)
    assert "textureSampleLevel(colorMap, colorMap_sampler, uv, 0.0)" in code
    _naga(code, tmp_path)


@pytest.mark.parametrize(
    "stages",
    [
        """
        fragment {
            vec4 main() @ gl_FragColor {
                sync();
                return vec4(1.0);
            }
        }
        """,
        """
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                sync();
            }
        }
        fragment {
            vec4 main() @ gl_FragColor {
                sync();
                return vec4(1.0);
            }
        }
        """,
    ],
    ids=["fragment-only", "compute-and-fragment"],
)
def test_a_barrier_a_graphics_entry_reaches_is_refused(tmp_path, stages):
    source = f"""
    shader GraphicsBarrier {{
        void sync() {{
            barrier();
        }}
        {stages}
    }}
    """
    with pytest.raises(WGSLRefusal, match="not reached only from compute entry points"):
        _translate(tmp_path, source)
