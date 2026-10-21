"""HLSL operand typing: mixed-signedness atomics, double vectors and matrices, double fma and typed refusals."""

import shutil
import subprocess

import pytest

from crosstl.translator import parse
from crosstl.translator.codegen.directx_codegen import (
    DirectXContextualConversionError,
    HLSLCodeGen,
)

MIXED_ATOMICS = """shader Atomics {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer uint* uin;
        buffer int* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            uint a1 = uin[i + 0u];
            atomicAdd(outp[i], a1);
            int old = atomicXor(outp[i], a1);
            atomicMax(outp[i], a1);
        }
    }
}"""

DOUBLES = """shader Doubles {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer vec2* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            dvec2 a0 = dvec2(double(fin[i + 0u]), double(fin[i + 1u]));
            dvec2 a1 = dvec2(double(fin[i + 2u]), double(fin[i + 3u]));
            dvec2 r = fma(a0, a1, a0);
            dmat2x3 m = dmat2x3(1.0);
            dvec3 c = m * r;
            vec2 f = fma(vec2(r), vec2(1.0), vec2(0.0));
            outp[i] = f + vec2(c.xy);
        }
    }
}"""

NON_INTEGER_INDEX = """shader Index {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer vec2* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            vec2 a0 = vec2(fin[i + 0u], fin[i + 1u]);
            outp[i] = NonUniformResourceIndex(a0);
        }
    }
}"""


def _dxc(hlsl, tmp_path):
    dxc = shutil.which("dxc")
    if not dxc:
        return
    (tmp_path / "out.hlsl").write_text(hlsl, encoding="utf-8")
    result = subprocess.run(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", "out.hlsl", "-Fo", "out.o"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_atomic_operands_take_the_memory_signedness(tmp_path):
    # GLSL, SPIR-V and Metal convert the operand to the memory type; so does HLSL.
    hlsl = HLSLCodeGen().generate(parse(MIXED_ATOMICS))

    assert "InterlockedAdd(outp[i], int(a1));" in hlsl
    assert "InterlockedXor(outp[i], int(a1), old);" in hlsl
    assert "InterlockedMax(outp[i], int(a1));" in hlsl
    _dxc(hlsl, tmp_path)


def test_double_vectors_and_matrices_map_to_hlsl_doubles(tmp_path):
    hlsl = HLSLCodeGen().generate(parse(DOUBLES))

    assert "dvec" not in hlsl and "dmat" not in hlsl
    assert "double2 a0 = double2(" in hlsl
    assert "double3x2 m = double3x2(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);" in hlsl
    # Double fma stays fused; float fma keeps HLSL mad.
    assert "double2 r = fma(a0, a1, a0);" in hlsl
    assert "float2 f = mad(" in hlsl
    _dxc(hlsl, tmp_path)


def test_non_integer_resource_index_is_a_typed_refusal():
    with pytest.raises(DirectXContextualConversionError) as refused:
        HLSLCodeGen().generate(parse(NON_INTEGER_INDEX))

    assert refused.value.reason == "non-uniform-resource-index-not-integer"
    assert "got float2" in str(refused.value)
