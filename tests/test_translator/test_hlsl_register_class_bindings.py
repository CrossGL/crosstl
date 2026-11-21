"""HLSL register classes do not collide in CUDA, HIP, Rust or Mojo bindings.

HLSL keeps separate register ranges for ``b`` (constant buffers), ``t`` (SRVs),
``u`` (UAVs) and ``s`` (samplers). The CUDA, HIP, Rust and Mojo generators all
map buffers into one namespace, so they used to reject ``register(b0)`` and
``register(u0)`` as a binding conflict. They should only conflict when the class
letter is the same.
"""

import pytest

import crosstl
from crosstl.translator.codegen.cuda_codegen import CudaCodeGen
from crosstl.translator.codegen.hip_codegen import HipCodeGen
from crosstl.translator.codegen.mojo_codegen import MojoCodeGen
from crosstl.translator.codegen.rust_codegen import RustCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.refusals import (
    CudaRefusal,
    HipRefusal,
    MojoRefusal,
    RustRefusal,
)

BACKENDS = ["cuda", "hip", "rust", "mojo"]

BACKEND_REFUSALS = [
    ("cuda", CudaRefusal),
    ("hip", HipRefusal),
    ("rust", RustRefusal),
    ("mojo", MojoRefusal),
]

CROSSGL_BACKENDS = [
    ("cuda", CudaCodeGen),
    ("hip", HipCodeGen),
    ("rust", RustCodeGen),
    ("mojo", MojoCodeGen),
]


def _translate_crossgl(source, codegen_cls):
    return codegen_cls().generate(Parser(Lexer(source).tokens).parse())


def _write_and_translate(tmp_path, hlsl, backend):
    path = tmp_path / "test.hlsl"
    path.write_text(hlsl)
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_hlsl_b0_and_u0_translate(backend_name, tmp_path):
    """register(b0) cbuffer and register(u0) RWStructuredBuffer coexist."""
    hlsl = """
    cbuffer Params : register(b0) {
        float scale;
    };
    RWStructuredBuffer<float> outp : register(u0);
    [numthreads(64, 1, 1)]
    void main(uint3 tid : SV_DispatchThreadID) {
        outp[tid.x] = outp[tid.x] * scale;
    }
    """
    code = _write_and_translate(tmp_path, hlsl, backend_name)
    assert "Params" in code
    assert "outp" in code


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_hlsl_t0_and_u0_translate(backend_name, tmp_path):
    """register(t0) StructuredBuffer and register(u0) RWStructuredBuffer coexist."""
    hlsl = """
    StructuredBuffer<float> inp : register(t0);
    RWStructuredBuffer<float> outp : register(u0);
    [numthreads(64, 1, 1)]
    void main(uint3 tid : SV_DispatchThreadID) {
        outp[tid.x] = inp[tid.x];
    }
    """
    code = _write_and_translate(tmp_path, hlsl, backend_name)
    assert "inp" in code
    assert "outp" in code


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_hlsl_b0_t0_and_u1_translate(backend_name, tmp_path):
    """A cbuffer, SRV and UAV with distinct class letters translate together."""
    hlsl = """
    cbuffer Params : register(b0) {
        float scale;
    };
    StructuredBuffer<float> inp : register(t0);
    RWStructuredBuffer<float> outp : register(u1);
    [numthreads(64, 1, 1)]
    void main(uint3 tid : SV_DispatchThreadID) {
        outp[tid.x] = inp[tid.x] * scale;
    }
    """
    code = _write_and_translate(tmp_path, hlsl, backend_name)
    assert "Params" in code
    assert "inp" in code
    assert "outp" in code


@pytest.mark.parametrize("backend_name, refusal_cls", BACKEND_REFUSALS)
def test_hlsl_same_register_class_same_number_still_refused(
    backend_name, refusal_cls, tmp_path
):
    """Two resources in the same HLSL register class and number still conflict."""
    hlsl = """
    StructuredBuffer<float> inp : register(t0);
    StructuredBuffer<float> inp2 : register(t0);
    [numthreads(64, 1, 1)]
    void main(uint3 tid : SV_DispatchThreadID) {
        inp2[tid.x] = inp[tid.x];
    }
    """
    with pytest.raises(refusal_cls, match="Conflicting .* resource binding"):
        _write_and_translate(tmp_path, hlsl, backend_name)


@pytest.mark.parametrize("backend_name, codegen_cls", CROSSGL_BACKENDS)
def test_crossgl_register_class_is_classified(backend_name, codegen_cls):
    """CrossGL ``@register(b0)`` and ``@register(u0)`` are not treated as overlapping."""
    source = """
    shader Test {
        @register(b0) cbuffer Params { float scale; };
        RWStructuredBuffer<float> outp @register(u0);
        compute {
            [numthreads(64, 1, 1)]
            void main(uint3 tid : SV_DispatchThreadID) {
                outp[tid.x] = outp[tid.x] * scale;
            }
        }
    }
    """
    code = _translate_crossgl(source, codegen_cls)
    assert "Params" in code
    assert "outp" in code


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_hlsl_auto_resource_skips_register_t0(backend_name, tmp_path):
    """An auto-assigned buffer skips the number occupied by ``register(t0)``."""
    hlsl = """
    StructuredBuffer<float> a : register(t0);
    StructuredBuffer<float> b;
    [numthreads(64, 1, 1)]
    void main(uint3 tid : SV_DispatchThreadID) {
        b[tid.x] = a[tid.x];
    }
    """
    code = _write_and_translate(tmp_path, hlsl, backend_name)
    assert "name=a kind=buffer set=0 binding=0" in code
    assert "name=b kind=buffer set=0 binding=1" in code


@pytest.mark.parametrize("backend_name, codegen_cls", CROSSGL_BACKENDS)
def test_crossgl_binding_zero_conflicts_with_register_t0(backend_name, codegen_cls):
    """A classless ``@binding(0)`` still conflicts with ``@register(t0)``."""
    source = """
    shader Test {
        @register(t0) StructuredBuffer<float> a;
        StructuredBuffer<float> b @binding(0);
        compute {
            [numthreads(64, 1, 1)]
            void main(uint3 tid : SV_DispatchThreadID) {
                b[tid.x] = a[tid.x];
            }
        }
    }
    """
    with pytest.raises(Exception, match="Conflicting .* resource binding"):
        _translate_crossgl(source, codegen_cls)
