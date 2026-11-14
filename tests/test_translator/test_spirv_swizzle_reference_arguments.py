"""A swizzle of several vector components is an ``out`` or ``inout`` argument on SPIR-V.

GLSL 4.60 section 6.1.1 and HLSL pass ``out`` and ``inout`` arguments by copy: the argument's l-value is fixed when
the arguments are evaluated, left to right, an ``inout`` value is copied in, and the parameter's final value is copied
back on return. ``v.xz`` is such an l-value when no component repeats (GLSL 4.60 section 5.8). ``OpAccessChain``
selects one vector component per index (SPIR-V 1.6, section 3.56.7), so the SPIR-V emitter, which inlines callees
with reference parameters and binds each to its argument's pointer, refused these calls as "not addressable". It now
gives such an argument its own ``Function`` variable, copies the selected components in for ``inout`` and stores the
variable back into them after the inlined body, through the base pointer taken at the call.

Each argument is evaluated once. Overload matching asked for a storage argument's type by resolving its pointer,
which lowered the expression, so ``inc(buf.v[j++].z)`` ran ``j++`` once per query as well as for the call and wrote
an element several places further on. The query now resolves a copy whose subscripts with side effects read ``0``.

``EXPECTED`` and ``ONCE_EXPECTED`` follow from section 6.1.1 by hand; the runtime tests compare the emitted SPIR-V,
and the GLSL emitter's output compiled by glslang, against them on Metal through SPIRV-Cross.
"""

import re
import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.SPIRV_codegen import UnsupportedSPIRVFeatureError
from tests.spirv_friendly_names import translate_spirv

FORMS = """
shader Forms {
    void bump(inout ivec2 p, int k) { p.x = p.x + k; p.y = p.y * 2; }
    void setPair(out ivec2 p) { p = ivec2(40, 50); }
    void step2(inout ivec2 p, inout int k) { p = p + ivec2(k, 100); k = k + 1; }
    void clampTo(inout ivec2 p, int lim) {
        if (p.x > lim) { p.x = lim; return; }
        p.y = lim;
    }
    int sumBump(inout ivec2 p) { p = p + ivec2(1, 1); return p.x + p.y; }
    void scale3(inout vec3 c, float f) { c = c * f; }
    compute {
        layout(std430, binding = 0) buffer Out { ivec4 vecs[2]; int values[]; } outBuf;
        shared ivec4 sv;
        layout(local_size_x = 1) in;
        void main() {
            ivec4 v = ivec4(1, 2, 3, 4);
            setPair(v.wy);
            ivec2 arr[2];
            arr[0] = ivec2(7, 8);
            arr[1] = ivec2(9, 10);
            int i = 0;
            step2(arr[i].yx, i);
            outBuf.vecs[0] = ivec4(0, 0, 0, 0);
            outBuf.vecs[1] = ivec4(1, 2, 3, 4);
            bump(outBuf.vecs[1].wy, 3);
            ivec3 q = ivec3(20, 1, 30);
            clampTo(q.zx, 25);
            ivec3 r = ivec3(1, 50, 2);
            clampTo(r.xz, 25);
            int s = sumBump(v.zw) + 1000;
            sv = ivec4(5, 6, 7, 8);
            bump(sv.xw, 5);
            mat3 m = mat3(1.0);
            scale3(m[1].zyx, 3.0);
            outBuf.values[0] = v.x; outBuf.values[1] = v.y;
            outBuf.values[2] = v.z; outBuf.values[3] = v.w;
            outBuf.values[4] = arr[0].x; outBuf.values[5] = arr[0].y;
            outBuf.values[6] = arr[1].x; outBuf.values[7] = i;
            outBuf.values[8] = q.x; outBuf.values[9] = q.y; outBuf.values[10] = q.z;
            outBuf.values[11] = r.x; outBuf.values[12] = r.y; outBuf.values[13] = r.z;
            outBuf.values[14] = s;
            outBuf.values[15] = sv.x; outBuf.values[16] = sv.y;
            outBuf.values[17] = sv.z; outBuf.values[18] = sv.w;
            outBuf.values[19] = int(m[1].x); outBuf.values[20] = int(m[1].y);
            outBuf.values[21] = int(m[1].z);
        }
    }
}
"""

# vecs[0], vecs[1] (bump of .wy), then values: v after setPair(.wy) and sumBump(.zw); arr[0] after
# step2(arr[0].yx) with i fixed at 0, arr[1].x, i; q after the early return; r; s; sv after bump(.xw);
# the column m[1] after scale3(.zyx).
EXPECTED = (
    (0, 0, 0, 0, 1, 4, 3, 7)
    + (1, 50, 4, 41, 107, 8, 9, 1)
    + (20, 1, 25, 1, 50, 25, 1045)
    + (10, 6, 7, 16, 0, 3, 0)
)

HLSL_FORMS = """
RWStructuredBuffer<int> Out : register(u0);
void bump(inout int2 p, int k) { p.x += k; p.y *= 2; }
void put(out int2 p, int k) { p = int2(k, k + 1); }
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    int4 v = int4(1, 2, 3, 4);
    bump(v.zx, 10);
    put(v.wy, 7);
    Out[0] = v.x; Out[1] = v.y; Out[2] = v.z; Out[3] = v.w;
}
"""

HLSL_EXPECTED = (2, 8, 13, 7)


ONCE = """
shader Once {
    void bump(inout ivec2 p, int k) { p.x = p.x + k; p.y = p.y * 2; }
    void inc(inout int p) { p = p + 5; }
    compute {
        layout(std430, binding = 0) buffer Out { ivec4 vecs[3]; int values[]; } outBuf;
        layout(local_size_x = 1) in;
        void main() {
            outBuf.vecs[0] = ivec4(1, 2, 3, 4);
            outBuf.vecs[1] = ivec4(1, 2, 3, 4);
            outBuf.vecs[2] = ivec4(1, 2, 3, 4);
            int i = 0;
            bump(outBuf.vecs[i++].wy, 3);
            int j = 1;
            inc(outBuf.vecs[j++].z);
            outBuf.values[0] = i;
            outBuf.values[1] = j;
        }
    }
}
"""

HLSL_ONCE = """
RWStructuredBuffer<int4> Vecs : register(u0);
void inc(inout int p) { p += 5; }
void bump(inout int2 p, int k) { p.x += k; p.y *= 2; }
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    Vecs[0] = int4(1, 2, 3, 4);
    Vecs[1] = int4(1, 2, 3, 4);
    Vecs[2] = int4(1, 2, 3, 4);
    uint j = 1;
    inc(Vecs[j++].z);
    uint i = 0;
    bump(Vecs[i++].wy, 3);
    Vecs[3] = int4(int(i), int(j), 0, 0);
}
"""

# vecs[0] after bump(.wy) with i at 0, vecs[1] after inc(.z) with j at 1, vecs[2] untouched, then i and j.
ONCE_EXPECTED = (1, 4, 3, 7, 1, 2, 8, 4, 1, 2, 3, 4, 1, 2)


def _shader(call, declaration="ivec4 v = ivec4(1, 2, 3, 4);"):
    return f"""
shader Probe {{
    void bump(inout ivec2 p, int k) {{ p.x = p.x + k; p.y = p.y * 2; }}
    void setPair(out ivec2 p) {{ p = ivec2(40, 50); }}
    compute {{
        layout(std430, binding = 0) buffer Out {{ ivec4 vecs[2]; int values[]; }} outBuf;
        layout(local_size_x = 1) in;
        void main() {{
            {declaration}
            {call}
            outBuf.values[0] = outBuf.vecs[0].x;
        }}
    }}
}}
"""


def _write(tmp_path, source, name):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def _translate(tmp_path, source, name, backend="vulkan"):
    path = _write(tmp_path, source, name)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _tool(name):
    path = shutil.which(name)
    if path is None:
        pytest.skip(f"{name} is not installed")
    return path


def _run(command):
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


def _assemble(tmp_path, code):
    asm = tmp_path / "out.spvasm"
    asm.write_text(code, encoding="utf-8")
    binary = tmp_path / "out.spv"
    _run([_tool("spirv-as"), "--target-env", "vulkan1.1", str(asm), "-o", str(binary)])
    _run([_tool("spirv-val"), "--target-env", "vulkan1.1", str(binary)])
    return binary


def _run_spirv_on_metal(spirv, result_count):
    msl = _run(
        [_tool("spirv-cross"), "--msl", "--msl-version", "20100", str(spirv)]
    ).stdout
    entry = re.search(r"kernel\s+void\s+(\w+)", msl).group(1)
    Metal = pytest.importorskip("Metal")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(
        msl, Metal.MTLCompileOptions.new(), None
    )
    assert library is not None, error
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_(entry), None
    )
    assert pipeline is not None, error
    size = 4 * result_count
    results = device.newBufferWithBytes_length_options_(
        struct.pack(f"<{result_count}i", *([-99] * result_count)), size, 0
    )
    command = device.newCommandQueue().commandBuffer()
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(results, 0, 0)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1), Metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    assert command.error() is None, command.error()
    data = bytes(results.contents().as_buffer(size))
    return struct.unpack(f"<{result_count}i", data)


def test_swizzle_arguments_translate_to_valid_spirv(tmp_path):
    code = _translate(tmp_path, FORMS, "forms.cgl")
    assert "OpVectorShuffle" in code
    _assemble(tmp_path, code)


def test_swizzle_arguments_follow_copy_in_copy_out_on_metal(tmp_path):
    binary = _assemble(tmp_path, _translate(tmp_path, FORMS, "forms.cgl"))
    assert _run_spirv_on_metal(binary, len(EXPECTED)) == EXPECTED


def test_glsl_emitter_output_agrees_through_glslang(tmp_path):
    glslang = _tool("glslangValidator")
    shader = tmp_path / "forms.comp"
    shader.write_text(_translate(tmp_path, FORMS, "forms.cgl", "opengl"))
    binary = tmp_path / "glslang.spv"
    _run([glslang, "-V", "-S", "comp", str(shader), "-o", str(binary)])
    assert _run_spirv_on_metal(binary, len(EXPECTED)) == EXPECTED


def test_hlsl_swizzle_arguments_follow_copy_in_copy_out_on_metal(tmp_path):
    binary = _assemble(tmp_path, _translate(tmp_path, HLSL_FORMS, "forms.hlsl"))
    assert _run_spirv_on_metal(binary, len(HLSL_EXPECTED)) == HLSL_EXPECTED


def test_out_swizzle_argument_is_not_copied_in(tmp_path):
    code = _translate(tmp_path, _shader("setPair(v.wy);"), "out.cgl")
    assert "OpVectorShuffle" not in code, code
    _assemble(tmp_path, code)


def test_buffer_swizzle_argument_touches_only_its_components(tmp_path):
    code = translate_spirv(
        _write(tmp_path, _shader("bump(outBuf.vecs[1].wy, 3);", ""), "buf.cgl")
    )
    elements = set(
        re.findall(r"%(\w+) = OpAccessChain %_ptr_\w+_v4int %\w+ %int_1\n", code)
    )
    assert elements, code
    for element in elements:
        assert not re.search(rf"OpStore %{element}\b", code), code
        assert not re.search(rf"OpLoad %\w+ %{element}\b", code), code
    components = [
        index
        for base, index in re.findall(
            r"OpAccessChain %_ptr_\w+_int %(\w+) %(int_\d)\n", code
        )
        if base in elements
    ]
    # Copy-in reads .w and .y; copy-out stores them, each through its own pointer.
    assert components == ["int_3", "int_1", "int_3", "int_1"], code
    _assemble(tmp_path, code)


@pytest.mark.parametrize(
    ("call", "feature"),
    [
        ("bump(v.xx, 1);", "duplicate-swizzle-reference-argument"),
        ("bump(v.xyz, 1);", "incompatible-reference-argument"),
    ],
)
def test_swizzle_arguments_that_are_not_lvalues_are_refused(tmp_path, call, feature):
    with pytest.raises(UnsupportedSPIRVFeatureError) as raised:
        _translate(tmp_path, _shader(call), "refused.cgl")
    assert raised.value.feature == feature


def test_reference_argument_subscripts_are_evaluated_once(tmp_path):
    code = translate_spirv(_write(tmp_path, ONCE, "once.cgl"))
    assert len(re.findall(r"OpStore %j\b", code)) == 2, code
    assert len(re.findall(r"OpStore %i\b", code)) == 2, code
    binary = _assemble(tmp_path, code)
    assert _run_spirv_on_metal(binary, len(ONCE_EXPECTED)) == ONCE_EXPECTED


def test_hlsl_reference_argument_subscripts_are_evaluated_once(tmp_path):
    binary = _assemble(tmp_path, _translate(tmp_path, HLSL_ONCE, "once.hlsl"))
    expected = ONCE_EXPECTED + (0, 0)
    assert _run_spirv_on_metal(binary, len(expected)) == expected
