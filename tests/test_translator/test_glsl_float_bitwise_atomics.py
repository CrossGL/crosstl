"""HLSL ``InterlockedCompareExchangeFloatBitwise`` and ``InterlockedCompareStoreFloatBitwise`` in GLSL.

Shader model 6.6 compares and swaps a ``float`` by its bits. The HLSL importer passes both calls through with HLSL's
argument lists, and every generator but HLSL's refused them as undeclared functions.
:func:`crosstl.translator.semantics.atomic_forms.canonicalize_atomic_calls` now rewrites them to CrossGL's
``atomicCompSwap``, which on a ``float`` compares bits: GLSL output keeps the destination buffer in ``uint`` words
and compares ``floatBitsToUint`` of the operands.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.glsl_float_atomic_storage import (
    OpenGLFloatAtomicStorageError,
)
from crosstl.translator.diagnostics import TranslationDiagnostic

GLSLANG = shutil.which("glslangValidator")

COMPARE_EXCHANGE = """RWStructuredBuffer<float> fin : register(u0);
RWStructuredBuffer<float> outp : register(u1);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    float a1 = fin[i + 0u];
    float a2 = fin[i + 1u];
    float a3;
    InterlockedCompareExchangeFloatBitwise(outp[i], a1, a2, a3);
    fin[i] = a3;
}
"""

COMPARE_STORE_MEMBER = """struct Pair { float a; float b; };
RWStructuredBuffer<Pair> pairs : register(u0);
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    InterlockedCompareStoreFloatBitwise(pairs[id.x].b, -0.0, 2.0);
    pairs[id.x].a = 1.0;
}
"""

GROUP_SHARED = """RWStructuredBuffer<float> outp : register(u0);
groupshared float shared_value;
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    if (gi == 0) shared_value = 1.0;
    GroupMemoryBarrierWithGroupSync();
    float seen;
    InterlockedCompareExchangeFloatBitwise(shared_value, 1.0, float(gi), seen);
    outp[id.x] = seen;
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "probe.hlsl"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _glslang(tmp_path, glsl):
    if GLSLANG is None:
        pytest.skip("glslangValidator is not on PATH")
    path = tmp_path / "translated.comp"
    path.write_text(glsl)
    result = subprocess.run(
        [GLSLANG, str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr + glsl


def _collapsed(code):
    return " ".join(code.split())


def test_compare_exchange_returns_the_previous_value_through_its_out_argument(
    tmp_path,
):
    glsl = _translate(tmp_path, COMPARE_EXCHANGE, "opengl")
    flat = _collapsed(glsl)
    # The destination buffer holds words, so the comparison is of the operands' bits.
    assert re.search(r"buffer \w+ \{ uint outp\[\]; \};", flat)
    assert re.search(
        r"uint \w+ = atomicCompSwap\( outp\[\w+\], floatBitsToUint\(\w+\), floatBitsToUint\(\w+\)\); "
        r"return uintBitsToFloat\(\w+\);",
        flat,
    )
    assert re.search(r"a3 = \w+\(uint\(i\), a1, a2\);", flat)
    assert "InterlockedCompareExchangeFloatBitwise" not in glsl
    _glslang(tmp_path, glsl)


def test_compare_store_of_a_structure_member_discards_the_previous_value(tmp_path):
    glsl = _translate(tmp_path, COMPARE_STORE_MEMBER, "opengl")
    flat = _collapsed(glsl)
    assert re.search(r"atomicCompSwap\( pairs\[\w+\]\.b,", flat)
    # A statement of its own: the previous value is discarded. -0.0 compares unequal to a stored +0.0.
    assert re.search(
        r"[;{] \w+\(uint\(gl_GlobalInvocationID\.x\), \(-0\.0\), 2\.0\);", flat
    )
    # The rest of the structure is stored as words too.
    assert "pairs[gl_GlobalInvocationID.x].a = floatBitsToUint(1.0);" in flat
    assert "InterlockedCompareStoreFloatBitwise" not in glsl
    _glslang(tmp_path, glsl)


@pytest.mark.parametrize("backend", ["vulkan", "metal", "cuda", "hip", "rust"])
def test_targets_without_float_compare_exchange_refuse_it(tmp_path, backend):
    with pytest.raises(TranslationDiagnostic, match="atomicCompSwap"):
        _translate(tmp_path, COMPARE_EXCHANGE, backend)


def test_a_group_shared_destination_is_refused(tmp_path):
    with pytest.raises(OpenGLFloatAtomicStorageError):
        _translate(tmp_path, GROUP_SHARED, "opengl")


def test_hlsl_output_keeps_the_bitwise_call(tmp_path):
    hlsl = _translate(tmp_path, COMPARE_EXCHANGE, "directx")
    assert "InterlockedCompareExchangeFloatBitwise(outp[i], a1, a2, a3);" in hlsl
