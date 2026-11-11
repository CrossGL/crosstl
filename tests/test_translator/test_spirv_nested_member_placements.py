"""Member placements inside the structs a constant buffer holds, and in arrays of ``ConstantBuffer<T>``.

``@ offset(N)`` and ``@ align(N)`` on a member of a struct place it within that struct, wherever the struct is held.
SPIR-V decorates every struct type's members with ``Offset`` (SPIR-V 1.6 section 2.16.2), so SPIR-V output places
them: the layout clone of a nested struct carries its declared offsets, and the element block of an array of
``ConstantBuffer<T>`` is planned like one ``ConstantBuffer<T>``. SPIR-V output used to decorate ``struct Inner { float
s; vec4 v @offset(32); }`` with ``Offset 16`` for ``v`` (std140) and was refused; an array of ``ConstantBuffer<T>``
lost the placements of ``T`` on SPIR-V, HLSL, GLSL and Slang output with no diagnostic.

HLSL and GLSL output declare the ``T`` of such an array once, by their own constant-buffer rules, so they refuse an
array whose declared layout or placements move a member under those rules. Every probe below fills the bytes the
placements do not name with a decoy value, so a read from the target's default offset returns the decoy.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.SPIRV_codegen import UnsupportedSPIRVFeatureError
from crosstl.translator.semantics.member_placements import (
    ConstantBufferArrayLayoutError,
    MemberPlacementError,
)
from tests.spirv_friendly_names import friendly_spirv

REQUIRE_ENV = "CROSSTL_REQUIRE_NESTED_MEMBER_PLACEMENT_EXECUTION"
DECOY = 99.0

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

INNER = "struct Inner { float s; vec4 v @offset(32); };"

# (id, declarations, body, {byte offset: value}, expected result, SPIR-V decorations the placements need)
CASES = [
    (
        "nested-struct",
        f"{INNER}\n cbuffer C {{ Inner n; }};",
        "outp[i] = n.s + n.v.y;",
        {0: 1.5, 36: 20.0},
        21.5,
        [r"%Inner_placed_std140_\d+ 1 Offset 32"],
    ),
    (
        "array-of-nested-structs",
        f"{INNER}\n cbuffer C {{ Inner n[2]; float z; }};",
        "outp[i] = n[1].s + n[1].v.y + z;",
        {48: 1.0, 84: 2.0, 96: 4.0},
        7.0,
        [r"%_arr_Inner_placed_std140_\d+_int_2 ArrayStride 48", r"%C 1 Offset 96"],
    ),
    (
        "two-levels-copied",
        "struct Deep { float d; float e @offset(12); };\n"
        " struct Inner { float s; Deep q @offset(16); };\n cbuffer C { Inner n; };",
        "Inner c = n; outp[i] = c.q.e + c.s;",
        {0: 1.0, 28: 5.0},
        6.0,
        [r"%Deep_placed_std140_\d+ 1 Offset 12"],
    ),
    (
        "struct-in-constant-buffer-of-t",
        f"{INNER}\n struct T {{ float a; Inner n; }};\n ConstantBuffer<T> cb @binding(0);",
        "outp[i] = cb.a + cb.n.v.y;",
        {0: 1.0, 52: 2.0},
        3.0,
        [r"%Inner_placed_std140_\d+ 1 Offset 32"],
    ),
    (
        "declared-scalar-layout",
        "struct Inner { float s; vec3 v @offset(8); };\n struct T { float a; Inner n; };\n"
        " ConstantBuffer<T> cb @binding(0) @buffer_layout(scalar);",
        "outp[i] = cb.a + cb.n.v.y;",
        {0: 1.0, 16: 2.0},
        3.0,
        [r"%Inner_placed_scalar_\d+ 1 Offset 8"],
    ),
    (
        "declared-std140-layout",
        "struct T { float s; vec3 v @offset(32); };\n"
        " ConstantBuffer<T> cb @binding(0) @buffer_layout(std140);",
        "outp[i] = cb.s + cb.v.y;",
        {0: 1.0, 36: 2.0},
        3.0,
        [r"%T_placed_std140_block_\d+ 1 Offset 32"],
    ),
    (
        "hlsl-cbuffer-nested",
        "struct Inner { float s @offset(0); float t @offset(8); };\n"
        " cbuffer C @buffer_layout(hlsl_cbuffer) { float a; Inner n; };",
        "outp[i] = a + n.s + n.t;",
        {0: 1.0, 16: 2.0, 24: 4.0},
        7.0,
        [r"%Inner_placed_hlsl_cbuffer_\d+ 1 Offset 8"],
    ),
    (
        "push-constant-nested",
        f"{INNER}\n cbuffer P @push_constant {{ float a; Inner n; }};",
        "outp[i] = a + n.v.y;",
        {0: 1.0, 52: 2.0},
        3.0,
        [r"%Inner_placed_std430_\d+ 1 Offset 32"],
    ),
    (
        "array-of-constant-buffers",
        "struct T { float s; vec4 v @offset(32); };\n ConstantBuffer<T> cbs[2] @binding(0);",
        "T t = cbs[0]; outp[i] = t.s + t.v.y;",
        {0: 1.0, 36: 2.0},
        3.0,
        [r"%T_placed_std140_block_\d+ 1 Offset 32"],
    ),
    (
        "array-of-hlsl-cbuffers",
        "struct T { float s; vec2 v; };\n"
        " ConstantBuffer<T> cbs[2] @binding(0) @buffer_layout(hlsl_cbuffer);",
        "outp[i] = cbs[0].s + cbs[0].v.y;",
        {0: 1.0, 8: 2.0},
        3.0,
        [r"%T_placed_hlsl_cbuffer_block_\d+ 1 Offset 4"],
    ),
]

CASE_IDS = [case[0] for case in CASES]


def _source(declarations, body):
    return "shader Probe {\n " + declarations + "\n" + BODY % body


def _translate(tmp_path, source, backend):
    path = tmp_path / f"{backend}.cgl"
    path.write_text(source)
    if backend != "vulkan":
        return crosstl.translate(str(path), backend=backend)
    # The same friendly ids with or without SPIRV-Tools on PATH.
    return friendly_spirv(
        crosstl.translate(str(path), backend=backend, format_output=False)
    )


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _assemble(tmp_path, spirv):
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if not (assembler and validator):
        pytest.skip("SPIRV-Tools is not installed")
    source = tmp_path / "probe.spvasm"
    binary = tmp_path / "probe.spv"
    source.write_text(spirv)
    for command in (
        [assembler, "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        [validator, "--target-env", "vulkan1.1", "--scalar-block-layout", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr or result.stdout
    return binary


def _host_bytes(values):
    words = [DECOY] * 32
    for offset, value in values.items():
        words[offset // 4] = value
    return struct.pack(f"<{len(words)}f", *words)


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


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_spirv_decorates_the_placements_of_nested_members(tmp_path, case):
    _name, declarations, body, _values, _expected, decorations = case
    spirv = _translate(tmp_path, _source(declarations, body), "vulkan")

    for decoration in decorations:
        assert re.search(
            r"Op(?:Member)?Decorate " + decoration + r"\b", spirv
        ), decoration
    _assemble(tmp_path, spirv)


def test_spirv_no_longer_writes_the_layout_offset_of_a_placed_nested_member(tmp_path):
    # Before: 'OpMemberDecorate %Inner_std140_9 1 Offset 16' under a refusal; the planned clone holds byte 32.
    spirv = _translate(tmp_path, _source(CASES[0][1], CASES[0][2]), "vulkan")

    assert not re.search(r"%Inner\w* 1 Offset 16\b", spirv)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_spirv_reads_nested_placements_from_the_host_bytes(tmp_path, case):
    _name, declarations, body, values, expected, _decorations = case
    env = _vulkan_runtime()
    if env is None:
        if os.environ.get(REQUIRE_ENV) == "1":
            pytest.fail(
                f"{REQUIRE_ENV}=1 but no host Vulkan runtime with vkrun is available"
            )
        pytest.skip(f"no host Vulkan runtime; set {REQUIRE_ENV}=1 to require one")
    spirv = _translate(tmp_path, _source(declarations, body), "vulkan")
    binary = _assemble(tmp_path, spirv)
    entry = re.search(r'OpEntryPoint GLCompute %\w+ "(\w+)"', spirv).group(1)
    bindings = dict(re.findall(r"OpDecorate %(\w+) Binding (\d+)", spirv))
    output = next(
        binding for name, binding in bindings.items() if name.startswith("outp")
    )
    constants = tmp_path / "constants.bin"
    constants.write_bytes(_host_bytes(values))
    result_path = tmp_path / "result.bin"
    command = [env["CROSSTL_CONFORMANCE_VKRUN"], str(binary), entry, "1", "1", "1"]
    command += [
        f"{binding}:uniform:{constants}"
        for name, binding in sorted(bindings.items())
        if not name.startswith("outp")
    ]
    command.append(f"{output}:out:{result_path}:16")
    if "@push_constant" in declarations:
        command.append(f"push:{constants}")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **env},
    )

    assert result.returncode == 0 and result.stdout.startswith("ok"), (
        result.stdout + result.stderr
    )
    (value,) = struct.unpack("<f", result_path.read_bytes()[:4])
    assert value == expected


def test_spirv_refuses_an_array_of_std430_constant_buffers_without_placements(tmp_path):
    source = _source(
        "struct T { float s; vec2 v; };\n"
        " ConstantBuffer<T> cbs[2] @binding(0) @buffer_layout(std430);",
        "outp[i] = cbs[0].s + cbs[0].v.y;",
    )

    with pytest.raises(UnsupportedSPIRVFeatureError, match="std430 layout"):
        _translate(tmp_path, source, "vulkan")


ARRAY_PLACED = (
    "struct T { float s; vec4 v @offset(32); };\n ConstantBuffer<T> cbs[2] @binding(0);"
)
ARRAY_STD140 = "struct T { float s; vec3 v; };\n ConstantBuffer<T> cbs[2] @binding(0) @buffer_layout(std140);"
ARRAY_HLSL = (
    "struct T { float s; vec2 v; };\n"
    " ConstantBuffer<T> cbs[2] @binding(0) @buffer_layout(hlsl_cbuffer);"
)
ARRAY_NESTED = f"{INNER}\n struct T {{ float a; Inner n; }};\n ConstantBuffer<T> cbs[2] @binding(0);"


@pytest.mark.parametrize(
    "declarations, backend",
    [
        # HLSL packing puts v at byte 16 and std140 at byte 16 too, but the placement asks for byte 32.
        (ARRAY_PLACED, "directx"),
        (ARRAY_PLACED, "opengl"),
        # std140 puts the vec3 at byte 16, HLSL packing at byte 4.
        (ARRAY_STD140, "directx"),
        # HLSL packing puts the vec2 at byte 4, std140 at byte 8.
        (ARRAY_HLSL, "opengl"),
        (ARRAY_NESTED, "directx"),
        (ARRAY_NESTED, "opengl"),
    ],
)
def test_hlsl_and_glsl_refuse_an_array_of_constant_buffers_they_would_lay_out_elsewhere(
    tmp_path, declarations, backend
):
    read = "cbs[1].n.v.y" if "Inner n" in declarations else "cbs[1].s"
    source = _source(declarations, f"outp[i] = {read};")

    with pytest.raises(
        ConstantBufferArrayLayoutError, match="array of constant buffers 'cbs'"
    ):
        _translate(tmp_path, source, backend)


@pytest.mark.parametrize(
    "declarations, backend",
    [(ARRAY_STD140, "opengl"), (ARRAY_HLSL, "directx")],
)
def test_an_array_of_constant_buffers_in_the_targets_own_layout_is_accepted(
    tmp_path, declarations, backend
):
    source = _source(declarations, "outp[i] = cbs[1].s + cbs[1].v.y;")

    assert _translate(tmp_path, source, backend)


def test_slang_refuses_the_placements_of_an_array_of_constant_buffers(tmp_path):
    source = _source(ARRAY_PLACED, "outp[i] = cbs[1].s + cbs[1].v.y;")

    with pytest.raises(MemberPlacementError, match="'cbs'"):
        _translate(tmp_path, source, "slang")
