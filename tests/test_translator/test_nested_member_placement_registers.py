"""Member placements inside the structs a constant buffer holds, on GLSL, GLSL ES and HLSL output.

``@ offset(N)`` and ``@ align(N)`` on a member of a struct place it within that struct wherever a constant buffer
holds it (DXC writes the same for ``[[vk::offset(N)]]`` on a struct member). GLSL ``layout(offset)`` and HLSL
``packoffset`` apply to the members of a block only (GLSL 4.60 section 4.4.5), and both languages declare a struct
once, laid out by the block's rules alone, so GLSL, GLSL ES and HLSL output refused such a buffer (#4294). They now
declare the buffer as 16-byte registers and copy every member out of the bytes its placement names, as they already
did for a buffer whose own members their rules cannot place.

Every probe fills the bytes the placements do not name with a decoy value, so a read from the byte the target's own
rules would pick returns the decoy. The GLSL output runs on Vulkan through glslangValidator and on OpenGL through
moderngl; the HLSL output is validated by DXC as DXIL and runs on Vulkan through DXC's SPIR-V output.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import sys

import pytest

import crosstl
from crosstl.translator.semantics.member_placements import MemberPlacementError

REQUIRE_ENV = "CROSSTL_REQUIRE_NESTED_MEMBER_PLACEMENT_EXECUTION"
REQUIRE_OPENGL_ENV = "CROSSTL_REQUIRE_NESTED_MEMBER_PLACEMENT_OPENGL_EXECUTION"
DECOY = 99.0
HOST_WORDS = 128

BODY = """
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            %s
        }
    }
}
"""

# WebGL 2 has no compute stage, so GLSL ES output is checked with the same reads in a fragment shader.
FRAGMENT = """
    fragment {
        vec4 main() @ gl_FragColor {
            %s
        }
    }
}
"""

INNER = "struct Inner { float s; vec4 v @offset(32); };"

# (id, declarations, body, {byte offset: value}, expected result). Each buffer that declares no layout is placed
# alike by std140 and HLSL packing apart from the struct placements, so one set of host bytes serves every target.
CASES = [
    (
        "nested-struct",
        f"{INNER}\n cbuffer C {{ Inner n; }};",
        "outp[i] = n.s + n.v.y;",
        {0: 1.5, 36: 20.0},
        21.5,
    ),
    (
        "array-of-nested-structs",
        f"{INNER}\n cbuffer C {{ Inner n[2]; float z; }};",
        "outp[i] = n[1].s + n[1].v.y + z;",
        {48: 1.0, 84: 2.0, 96: 4.0},
        7.0,
    ),
    (
        "two-levels-copied",
        "struct Deep { float d; float e @offset(12); };\n"
        " struct Inner { float s; Deep q @offset(16); };\n cbuffer C { Inner n; };",
        "Inner c = n; outp[i] = c.q.e + c.s;",
        {0: 1.0, 28: 5.0},
        6.0,
    ),
    (
        "aligned-member",
        "struct Inner { float s; float t @align(16); };\n cbuffer C { float a; Inner n; };",
        "outp[i] = a + n.s + n.t;",
        {0: 1.0, 16: 2.0, 32: 4.0},
        7.0,
    ),
    (
        "hlsl-cbuffer-member-after-an-offset",
        # t is placed at byte 8 of Inner; u follows it by HLSL packing, at byte 12.
        "struct Inner { float s; float t @offset(8); float u; };\n"
        " cbuffer C @buffer_layout(hlsl_cbuffer) { float a; Inner n; };",
        "outp[i] = a + n.s + n.t + n.u;",
        {0: 1.0, 16: 2.0, 24: 4.0, 28: 8.0},
        15.0,
    ),
    (
        "declared-std430-layout",
        # std430 aligns Inner to 4 bytes, so n starts at byte 4 and t at byte 12.
        "struct Inner { float s; float t @offset(8); };\n"
        " cbuffer C @buffer_layout(std430) { float a; Inner n; };",
        "outp[i] = a + n.s + n.t;",
        {0: 1.0, 4: 2.0, 12: 4.0},
        7.0,
    ),
]
CASE_IDS = [case[0] for case in CASES]

# HLSL as the Vulkan samples write it: the struct's member placement is all the buffer has.
HLSL_SOURCE = """
struct UBO
{
    float scale;
    [[vk::offset(272)]] float alpha;
};
cbuffer ubo : register(b0) { UBO ubo; }
RWStructuredBuffer<float> outp : register(u1);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    outp[id.x] = ubo.scale + ubo.alpha;
}
"""
HLSL_FRAGMENT_SOURCE = HLSL_SOURCE.split("RWStructuredBuffer")[0] + """
float4 main() : SV_Target
{
    return float4(ubo.scale + ubo.alpha, 0.0, 0.0, 1.0);
}
"""
HLSL_VALUES = {0: 1.0, 272: 2.0}


def _source(declarations, body, backend="opengl"):
    if backend == "webgl":
        body = re.sub(r"outp\[i\] = (.*);$", r"return vec4(\1, 0.0, 0.0, 1.0);", body)
        return "shader Probe {\n " + declarations + "\n" + FRAGMENT % body
    return "shader Probe {\n " + declarations + "\n" + BODY % body


def _translate(tmp_path, source, backend, suffix="cgl"):
    path = tmp_path / f"probe-{backend}.{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _host_bytes(values):
    words = [DECOY] * HOST_WORDS
    for offset, value in values.items():
        words[offset // 4] = value
    return struct.pack(f"<{HOST_WORDS}f", *words)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, (result.stdout + result.stderr)[-4000:]
    return result


def _vulkan_runtime():
    """A host Vulkan runtime and ``vkrun``, or ``None``: lavapipe, else MoltenVK or KosmicKrisp."""
    try:
        from tests.conformance.runtimes import provision
    except ImportError:
        return None
    for runtime in ("vulkan-lavapipe", "vulkan-moltenvk", "vulkan-kosmickrisp"):
        try:
            provisioned = provision(runtime)
        except Exception:
            continue
        if provisioned.available and provisioned.mode == "host":
            env = dict(provisioned.env)
            if env.get("CROSSTL_CONFORMANCE_VKRUN"):
                return env
    return None


def _require_vulkan():
    env = _vulkan_runtime()
    if env is None:
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(
                f"{REQUIRE_ENV}=1 but no host Vulkan runtime with vkrun is available"
            )
        pytest.skip(f"no host Vulkan runtime; set {REQUIRE_ENV}=1 to require one")
    return env


def _dispatch(tmp_path, env, binary, entry, uniform_binding, output_binding, values):
    constants = tmp_path / "constants.bin"
    constants.write_bytes(_host_bytes(values))
    result_path = tmp_path / "result.bin"
    command = [
        env["CROSSTL_CONFORMANCE_VKRUN"],
        str(binary),
        entry,
        "1",
        "1",
        "1",
        f"{uniform_binding}:uniform:{constants}",
        f"{output_binding}:out:{result_path}:16",
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=120, env={**os.environ, **env}
    )
    assert result.returncode == 0 and result.stdout.startswith("ok"), (
        result.stdout + result.stderr
    )
    (value,) = struct.unpack("<f", result_path.read_bytes()[:4])
    return value


def _glsl_bindings(glsl):
    uniform = re.search(r"layout\(std140, binding = (\d+)\) uniform", glsl).group(1)
    output = re.search(r"layout\(std430, binding = (\d+)\) buffer", glsl).group(1)
    return int(uniform), int(output)


def _run_glsl_on_vulkan(tmp_path, glsl, values):
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not on PATH")
    env = _require_vulkan()
    source = tmp_path / "probe.comp"
    binary = tmp_path / "probe-glsl.spv"
    source.write_text(glsl)
    # OpenGL numbers uniform and storage blocks apart; Vulkan gives them one namespace, so storage moves to 16 + N.
    _run(
        [
            "glslangValidator",
            "-V",
            "--target-env",
            "vulkan1.1",
            "--shift-ssbo-binding",
            "16",
            str(source),
            "-o",
            str(binary),
        ]
    )
    uniform, output = _glsl_bindings(glsl)
    return _dispatch(tmp_path, env, binary, "main", uniform, 16 + output, values)


def _hlsl_entry(hlsl):
    return re.search(r"\[numthreads\([^)]*\)\]\s*void\s+(\w+)\s*\(", hlsl).group(1)


def _dxc(tmp_path, hlsl, *arguments, output):
    dxc = shutil.which("dxc")
    if not dxc:
        pytest.skip("dxc is not on PATH")
    source = tmp_path / "probe.hlsl"
    source.write_text(hlsl)
    _run(
        [
            dxc,
            "-T",
            "cs_6_0",
            "-E",
            _hlsl_entry(hlsl),
            "-WX",
            *arguments,
            str(source),
            "-Fo",
            str(output),
        ]
    )
    return output


def _run_hlsl_on_vulkan(tmp_path, hlsl, values):
    # DXC gives register bN binding N and uN binding 16 + N, so the two never share a binding.
    binary = _dxc(
        tmp_path,
        hlsl,
        "-spirv",
        "-fvk-u-shift",
        "16",
        "0",
        output=tmp_path / "probe-hlsl.spv",
    )
    env = _require_vulkan()
    uniform = int(re.search(r"cbuffer\s+\w+\s*:\s*register\(b(\d+)\)", hlsl).group(1))
    output = 16 + int(
        re.search(
            r"RWStructuredBuffer<float>\s+\w+\s*:\s*register\(u(\d+)\)", hlsl
        ).group(1)
    )
    return _dispatch(tmp_path, env, binary, _hlsl_entry(hlsl), uniform, output, values)


def _run_glsl_on_opengl(glsl, values):
    if os.environ.get(REQUIRE_OPENGL_ENV) != "1" or sys.platform == "darwin":
        pytest.skip(
            f"set {REQUIRE_OPENGL_ENV}=1 on Linux to require the OpenGL dispatch"
        )
    import moderngl

    uniform, output = _glsl_bindings(glsl)
    context = moderngl.create_standalone_context(require=430, backend="egl")
    buffers = []
    try:
        shader = context.compute_shader(glsl)
        constants = context.buffer(_host_bytes(values))
        constants.bind_to_uniform_block(uniform)
        result = context.buffer(bytes(16))
        result.bind_to_storage_buffer(output)
        buffers += [constants, result]
        shader.run(group_x=1)
        context.finish()
        (value,) = struct.unpack("<f", result.read()[:4])
        return value
    finally:
        for buffer in buffers:
            buffer.release()
        context.release()


# Translation ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["opengl", "webgl", "directx"])
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_placed_struct_members_are_read_from_registers(tmp_path, case, backend):
    _name, declarations, body, _values, _expected = case
    code = _translate(tmp_path, _source(declarations, body, backend), backend)

    if backend == "directx":
        assert re.search(r"uint4 __crossgl_C_registers\[\d+\];", code)
        assert "__crossgl_load_C();" in code
    else:
        assert re.search(r"uvec4 crossgl_cbuffer_registers\w*\[\d+\];", code)
        assert re.search(r"crossgl_load_C\w*\(\);", code)


def test_glsl_reads_a_placed_member_at_its_byte(tmp_path):
    # Before: refused; std140 alone would put v at byte 16, register 1.
    code = _flat(_translate(tmp_path, _source(CASES[0][1], CASES[0][2]), "opengl"))

    assert re.search(
        r"n\.v = uintBitsToFloat\(crossgl_cbuffer_registers\w*\[2\]\.xyzw\);", code
    )


def test_hlsl_reads_a_member_after_a_placed_member_by_packing(tmp_path):
    code = _flat(_translate(tmp_path, _source(CASES[4][1], CASES[4][2]), "directx"))

    assert "n.t = asfloat(__crossgl_C_registers[1].z);" in code
    assert "n.u = asfloat(__crossgl_C_registers[1].w);" in code


@pytest.mark.parametrize("backend", ["opengl", "webgl", "directx"])
def test_hlsl_vk_offset_on_a_struct_member_is_read_from_its_register(tmp_path, backend):
    source = HLSL_FRAGMENT_SOURCE if backend == "webgl" else HLSL_SOURCE
    code = _flat(_translate(tmp_path, source, backend, suffix="hlsl"))

    # Byte 272 is the first word of register 17.
    reader = "asfloat" if backend == "directx" else "uintBitsToFloat"
    assert re.search(rf"ubo\.alpha = {reader}\(\w+\[17\]\.x\);", code)


@pytest.mark.parametrize("backend", ["opengl", "directx"])
def test_a_nested_placement_that_matches_the_layout_keeps_the_plain_declaration(
    tmp_path, backend
):
    source = _source(
        "struct Inner { float s @offset(0); vec4 v @offset(16); };\n cbuffer C { Inner n; };",
        "outp[i] = n.s + n.v.y;",
    )

    code = _translate(tmp_path, source, backend)

    assert "registers" not in code


@pytest.mark.parametrize("backend", ["opengl", "directx"])
def test_a_push_constant_block_holding_a_placed_struct_is_still_refused(
    tmp_path, backend
):
    # GLSL's Vulkan push_constant branch and HLSL's [[vk::push_constant]] block declare the struct unplaced.
    source = _source(
        f"{INNER}\n cbuffer P @push_constant {{ float a; Inner n; }};",
        "outp[i] = a + n.v.y;",
    )

    with pytest.raises(MemberPlacementError, match="constant buffer 'P'"):
        _translate(tmp_path, source, backend)


def test_an_offset_before_the_member_above_it_is_refused(tmp_path):
    source = _source(
        "struct Inner { float s; float t @offset(8); float u @offset(4); };\n"
        " cbuffer C @buffer_layout(hlsl_cbuffer) { Inner n; };",
        "outp[i] = n.u;",
    )

    with pytest.raises(Exception, match="lies within or before the member above it"):
        _translate(tmp_path, source, "directx")


# Validation and execution --------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_glsl_and_glsl_es_output_compile(tmp_path, case):
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not on PATH")
    _name, declarations, body, _values, _expected = case
    for backend, stage in (("opengl", "probe.comp"), ("webgl", "probe-es.frag")):
        path = tmp_path / stage
        path.write_text(
            _translate(tmp_path, _source(declarations, body, backend), backend)
        )
        _run(["glslangValidator", str(path)])


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_hlsl_output_compiles_to_dxil(tmp_path, case):
    _name, declarations, body, _values, _expected = case
    hlsl = _translate(tmp_path, _source(declarations, body), "directx")

    assert _dxc(tmp_path, hlsl, output=tmp_path / "probe.dxil").stat().st_size > 0


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_glsl_output_reads_the_host_bytes_on_vulkan(tmp_path, case):
    _name, declarations, body, values, expected = case
    glsl = _translate(tmp_path, _source(declarations, body), "opengl")

    assert _run_glsl_on_vulkan(tmp_path, glsl, values) == expected


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_hlsl_output_reads_the_host_bytes_on_vulkan(tmp_path, case):
    _name, declarations, body, values, expected = case
    hlsl = _translate(tmp_path, _source(declarations, body), "directx")

    assert _run_hlsl_on_vulkan(tmp_path, hlsl, values) == expected


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_glsl_output_reads_the_host_bytes_on_opengl(tmp_path, case):
    _name, declarations, body, values, expected = case
    glsl = _translate(tmp_path, _source(declarations, body), "opengl")

    assert _run_glsl_on_opengl(glsl, values) == expected


@pytest.mark.parametrize("backend", ["opengl", "directx"])
def test_hlsl_vk_offset_struct_member_executes(tmp_path, backend):
    code = _translate(tmp_path, HLSL_SOURCE, backend, suffix="hlsl")
    run = _run_glsl_on_vulkan if backend == "opengl" else _run_hlsl_on_vulkan

    assert run(tmp_path, code, HLSL_VALUES) == 3.0
