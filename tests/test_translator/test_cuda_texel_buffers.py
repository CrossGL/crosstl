"""CrossGL sampled texel buffers in CUDA and HIP output.

A CrossGL ``samplerBuffer``, ``isamplerBuffer`` or ``usamplerBuffer`` is a texture object bound to linear memory, and
``texelFetch(buffer, index)`` reads one texel of it with ``tex1Dfetch<T4>`` (CUDA C++ Programming Guide, "Texture
Object API"; HIP mirrors it). ``textureSize(buffer)`` reads the width the host records in the resource's query
metadata. GLSL defines no other operation on a texel buffer (GLSL 4.60, 8.9.1 and 8.9.2), so the generators refuse
the rest. CUDA and HIP output refused every texel buffer before.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.lowering_refusals import LoweringRefused

BACKENDS = ["cuda", "hip"]
TEXTURE_OBJECT = {"cuda": "cudaTextureObject_t", "hip": "hipTextureObject_t"}

TEXEL_BUFFERS = """
shader Probe {
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        buffer int* outi;
        buffer uint* outu;
        uniform samplerBuffer f;
        uniform isamplerBuffer s;
        uniform usamplerBuffer u;
        vec4 first(samplerBuffer b, int k) { return texelFetch(b, k); }
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int n = textureSize(f);
            outp[i] = texelFetch(f, int(i) % n).x + first(f, 0).y;
            outi[i] = texelFetch(s, int(i)).w + textureSize(s);
            outu[i] = texelFetch(u, int(i)).z + uint(textureSize(u));
        }
    }
}
"""

INTEGER_TEXTURE_SIZE = """
shader Probe {
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer int* outi;
        uniform isampler2D s;
        uniform usampler3D u;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outi[i] = textureSize(s, 0).x + textureSize(u, 0).z;
        }
    }
}
"""

REFUSED_CALL = """
shader Probe {
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        uniform samplerBuffer f;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            outp[i] = %s;
        }
    }
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "texel_buffers.cgl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _compile(tmp_path, code, backend):
    """Compile ``code`` with the target's own compiler when it is installed."""
    if backend == "cuda" and shutil.which("nvcc"):
        (tmp_path / "out.cu").write_text(code)
        command = [shutil.which("nvcc"), "-std=c++17", "--ptx", "out.cu"]
    elif backend == "hip" and shutil.which("hipcc"):
        (tmp_path / "out.hip").write_text(code)
        command = [
            shutil.which("hipcc"),
            "-std=c++17",
            "--offload-device-only",
            "-c",
            "out.hip",
        ]
    else:
        return
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=tmp_path, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", BACKENDS)
def test_texel_buffers_are_texture_objects(tmp_path, backend):
    code = _translate(tmp_path, TEXEL_BUFFERS, backend)
    texture_object = TEXTURE_OBJECT[backend]
    for name in ("f", "s", "u"):
        assert re.search(rf"\b{texture_object} {name};", code), name
    assert re.search(rf"float4 first\({texture_object} b, int k\)", code)


@pytest.mark.parametrize("backend", BACKENDS)
def test_texel_fetch_reads_with_tex1dfetch(tmp_path, backend):
    code = _translate(tmp_path, TEXEL_BUFFERS, backend)
    assert "return tex1Dfetch<float4>(b, k);" in code
    assert "tex1Dfetch<float4>(f, (int(i) % n)).x" in code
    assert "tex1Dfetch<int4>(s, int(i)).w" in code
    assert "tex1Dfetch<uint4>(u, int(i)).z" in code
    assert "texelFetch" not in code
    _compile(tmp_path, code, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_texture_size_reads_the_buffer_width(tmp_path, backend):
    code = _translate(tmp_path, TEXEL_BUFFERS, backend)
    for sampler in ("samplerBuffer", "isamplerBuffer", "usamplerBuffer"):
        helper = re.search(
            rf"int cgl_textureSize_{sampler}\(\s*CglResourceQueryInfo info\)"
            rf"\s*\{{\s*return info\.width;\s*\}}",
            code,
        )
        assert helper, sampler
    assert "int n = cgl_textureSize_samplerBuffer(f_metadata);" in code
    assert "cgl_textureSize_isamplerBuffer(s_metadata)" in code
    assert "cgl_textureSize_usamplerBuffer(u_metadata)" in code


@pytest.mark.parametrize("backend", BACKENDS)
def test_integer_textures_have_size_queries(tmp_path, backend):
    code = _translate(tmp_path, INTEGER_TEXTURE_SIZE, backend)
    texture_object = TEXTURE_OBJECT[backend]
    assert re.search(rf"\b{texture_object} s;", code)
    assert re.search(rf"\b{texture_object} u;", code)
    assert re.search(r"cgl_textureSize_isampler2D\(s_metadata, 0\)\.x", code)
    assert re.search(r"cgl_textureSize_usampler3D\(u_metadata, 0\)\.z", code)
    _compile(tmp_path, code, backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "call, message",
    [
        ("texture(f, 0.5).x", "'texture' is not an operation on a samplerBuffer"),
        (
            "float(textureSize(f, 0))",
            "textureSize on samplerBuffer takes no level-of-detail argument",
        ),
        (
            "texelFetch(f, 0, 0).x",
            "texelFetch on samplerBuffer takes the buffer and one integer index",
        ),
        ("textureGather(f, 0.5).x", "'textureGather' is not an operation on a"),
    ],
)
def test_other_texel_buffer_operations_are_refused(tmp_path, backend, call, message):
    with pytest.raises(LoweringRefused, match=re.escape(message)):
        _translate(tmp_path, REFUSED_CALL % call, backend)
