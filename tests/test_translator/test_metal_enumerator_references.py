"""References to Metal enumerators reach every target as CrossGL's scoped ``E::A``.

MSL follows C++: an unscoped enum's enumerators are names of the enclosing scope (C++17 section 10.2), so shaders
write ``AAPLNumMeshTextures`` rather than ``AAPLTextureIndices::AAPLNumMeshTextures``. CrossGL enumerators are
scoped, and the generators declare ``E::A`` as their own constants, so a bare ``A`` named nothing in HLSL or GLSL
and SPIR-V refused it. Scoped (``enum class``) references were mangled to ``Scoped_u3a_u3aTwo``.
"""

import shutil
import subprocess

import pytest

import crosstl

SOURCE = """
#include <metal_stdlib>
using namespace metal;

typedef enum AAPLTextureIndices {
    AAPLTextureIndexBaseColor = 0,
    AAPLTextureIndexNormal = 1,
    AAPLNumMeshTextures = AAPLTextureIndexNormal + 1
} AAPLTextureIndices;

enum class Channel : int { Red = 1, Green = 2 };

kernel void counts(device int *out [[buffer(0)]],
                   uint tid [[thread_position_in_grid]]) {
    int base = AAPLNumMeshTextures * AAPLTextureIndexNormal;
    out[tid] = base + int(Channel::Green) + (tid == 0 ? AAPLTextureIndexBaseColor : 1);
}
"""


def _translate(tmp_path, backend):
    path = tmp_path / "counts.metal"
    path.write_text(SOURCE, encoding="utf-8")
    return crosstl.translate(
        str(path), backend=backend, source_options={"threadgroup_size": (64, 1, 1)}
    )


def _run(command, **kwargs):
    completed = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-2000:]


def test_enumerator_references_import_scoped(tmp_path):
    crossgl = _translate(tmp_path, "cgl")
    assert (
        "AAPLNumMeshTextures = AAPLTextureIndices::AAPLTextureIndexNormal + 1"
        in crossgl
    )
    assert (
        "int base = AAPLTextureIndices::AAPLNumMeshTextures * "
        "AAPLTextureIndices::AAPLTextureIndexNormal;"
    ) in crossgl
    assert "int(Channel::Green)" in crossgl
    assert "_u3a" not in crossgl


@pytest.mark.parametrize("backend", ["directx", "opengl", "vulkan", "metal"])
def test_enumerator_references_compile_on_every_target(tmp_path, backend):
    code = _translate(tmp_path, backend)
    if backend == "directx" and shutil.which("dxc"):
        path = tmp_path / "counts.hlsl"
        path.write_text(code, encoding="utf-8")
        _run(
            [shutil.which("dxc"), "-T", "cs_6_0", "-E", "CSMain", str(path)],
            timeout=180,
        )
    elif backend == "opengl" and shutil.which("glslangValidator"):
        path = tmp_path / "counts.comp"
        path.write_text(code, encoding="utf-8")
        _run([shutil.which("glslangValidator"), str(path)])
    elif backend == "vulkan" and shutil.which("spirv-as") and shutil.which("spirv-val"):
        assembly, module = tmp_path / "counts.spvasm", tmp_path / "counts.spv"
        assembly.write_text(code, encoding="utf-8")
        _run(
            ["spirv-as", "--target-env", "vulkan1.1", str(assembly), "-o", str(module)]
        )
        _run(["spirv-val", "--target-env", "vulkan1.1", str(module)])
    elif backend == "metal" and shutil.which("xcrun"):
        if subprocess.run(
            ["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True
        ).returncode:
            return
        path = tmp_path / "counts.metal.out.metal"
        path.write_text(code, encoding="utf-8")
        _run(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-c",
                str(path),
                "-o",
                str(tmp_path / "counts.air"),
            ],
            timeout=180,
        )


def test_a_variable_named_like_an_enumerator_hides_it(tmp_path):
    source = SOURCE.replace(
        "int base = AAPLNumMeshTextures * AAPLTextureIndexNormal;",
        "int AAPLNumMeshTextures = 5;\n    int base = AAPLNumMeshTextures * AAPLTextureIndexNormal;",
    )
    path = tmp_path / "shadow.metal"
    path.write_text(source, encoding="utf-8")
    crossgl = crosstl.translate(str(path), backend="cgl")
    assert (
        "int base = AAPLNumMeshTextures * AAPLTextureIndices::AAPLTextureIndexNormal;"
        in crossgl
    )


MODES = """
#include <metal_stdlib>
using namespace metal;

typedef enum { ModeAdd = 0, ModeMul = 1, ModeCount } Mode;

kernel void apply(device float *out [[buffer(0)]],
                  constant int &mode [[buffer(1)]],
                  uint tid [[thread_position_in_grid]]) {
    float weights[ModeCount];
    weights[ModeAdd] = 1.0f;
    weights[ModeMul] = 2.0f;
    Mode selected = Mode(mode);
    switch (selected) {
        case ModeAdd: out[tid] += weights[0]; break;
        case ModeMul: out[tid] *= weights[1]; break;
        default: break;
    }
}
"""


def test_enum_values_follow_c_rules():
    from crosstl.translator.semantics.enum_values import enum_values

    program = crosstl.translator.parse("""
        shader main {
            enum Flags { A = 1, B, C = Flags::B << 2, D = C + 1, }
            compute { @ stage_entry @ workgroup_size(1, 1, 1) void k() {} }
        }
        """)
    values = enum_values(program)
    assert [values[f"Flags::{name}"] for name in "ABCD"] == [1, 2, 8, 9]


@pytest.mark.parametrize("backend", ["directx", "opengl", "vulkan", "metal"])
def test_enum_extents_and_conversions_compile(tmp_path, backend):
    path = tmp_path / "modes.metal"
    path.write_text(MODES, encoding="utf-8")
    code = crosstl.translate(
        str(path), backend=backend, source_options={"threadgroup_size": (64, 1, 1)}
    )
    assert "ModeCount]" not in code
    assert "Mode(" not in code.replace("Mode_", "")
    if backend == "directx" and shutil.which("dxc"):
        hlsl = tmp_path / "modes.hlsl"
        hlsl.write_text(code, encoding="utf-8")
        _run(
            [shutil.which("dxc"), "-T", "cs_6_0", "-E", "CSMain", str(hlsl)],
            timeout=180,
        )
    elif backend == "opengl" and shutil.which("glslangValidator"):
        glsl = tmp_path / "modes.comp"
        glsl.write_text(code, encoding="utf-8")
        _run([shutil.which("glslangValidator"), str(glsl)])
    elif backend == "vulkan" and shutil.which("spirv-as") and shutil.which("spirv-val"):
        assembly, module = tmp_path / "modes.spvasm", tmp_path / "modes.spv"
        assembly.write_text(code, encoding="utf-8")
        _run(
            ["spirv-as", "--target-env", "vulkan1.1", str(assembly), "-o", str(module)]
        )
        _run(["spirv-val", "--target-env", "vulkan1.1", str(module)])
    elif backend == "metal" and shutil.which("xcrun"):
        if subprocess.run(
            ["xcrun", "-sdk", "macosx", "-f", "metal"], capture_output=True
        ).returncode:
            return
        out = tmp_path / "modes.out.metal"
        out.write_text(code, encoding="utf-8")
        _run(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-c",
                str(out),
                "-o",
                str(tmp_path / "modes.air"),
            ],
            timeout=180,
        )
