"""Coherence of storage textures in Metal output.

Metal output used to drop ``coherent`` and ``globallycoherent`` on storage images. MSL 3.2 states texture
coherence as a template argument, ``texture2d<T, access::A, memory_coherence::memory_coherence_device>``; the
default is threadgroup coherence and there is no system scope. A texture type has to match the helper parameter
it is passed to, so every texture and parameter that calls connect to a coherent one becomes device coherent.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused
from crosstl.translator.codegen.metal_texture_coherence import (
    with_device_texture_coherence,
)

DEVICE = "memory_coherence::memory_coherence_device"


def translate(tmp_path, source, backend, suffix=".cgl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def flat(code):
    """Join the declarations that clang-format wraps, for substring checks."""
    return re.sub(r"\s+", " ", code)


def metal_ir(tmp_path, code):
    """Compile with ``xcrun metal -std=metal3.2`` and return the textual AIR."""
    if shutil.which("xcrun") is None:
        pytest.skip("xcrun is not installed")
    (tmp_path / "out.metal").write_text(code)
    command = ["xcrun", "-sdk", "macosx", "metal", "-std=metal3.2", "-S"]
    command += ["-emit-llvm", "-c", "out.metal", "-o", "-"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    if result.returncode != 0 and "missing Metal Toolchain" in result.stderr:
        pytest.skip("the Metal toolchain is not installed")
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_coherent_storage_image_is_device_coherent(tmp_path):
    code = translate(
        tmp_path,
        """shader C {
    uimage2D img @binding(0) @r32ui @coherent;
    compute {
        void main() {
            imageStore(img, ivec2(0), imageLoad(img, ivec2(1)));
        }
    }
}""",
        "metal",
    )

    assert f"texture2d<uint, access::read_write, {DEVICE}> img [[texture(0)]]" in flat(
        code
    )
    ir = metal_ir(tmp_path, code)
    assert "air.read_texture_2d.device_coherent" in ir
    assert "air.write_texture_2d.device_coherent" in ir


def test_hlsl_globallycoherent_texture_reaches_helper_parameter(tmp_path):
    crossgl = translate(
        tmp_path,
        """globallycoherent RWTexture2D<uint> img : register(u0);
void put(RWTexture2D<uint> t) { t[uint2(0, 0)] = 1u; }
[numthreads(1, 1, 1)] void main() { put(img); }
""",
        "cgl",
        suffix=".hlsl",
    )
    code = translate(tmp_path, crossgl, "metal")

    assert f"void put(texture2d<uint, access::read_write, {DEVICE}> t)" in flat(code)
    assert f"texture2d<uint, access::read_write, {DEVICE}> img [[texture(0)]]" in flat(
        code
    )
    ir = metal_ir(tmp_path, code)
    assert "air.write_texture_2d.device_coherent" in ir
    assert not re.search(r"air\.write_texture_2d\.(?!device_coherent)", ir)


def test_glsl_coherent_image_and_parameter(tmp_path):
    crossgl = translate(
        tmp_path,
        """#version 450
layout(local_size_x = 1) in;
layout(r32ui, binding = 0) coherent uniform uimage2D img;
void put(coherent uimage2D t) { imageStore(t, ivec2(0), uvec4(1u)); }
void main() { put(img); }
""",
        "cgl",
        suffix=".comp",
    )
    code = translate(tmp_path, crossgl, "metal")

    assert f"void put(texture2d<uint, access::read_write, {DEVICE}> t)" in flat(code)
    assert f"texture2d<uint, access::read_write, {DEVICE}> img [[texture(0)]]" in flat(
        code
    )
    metal_ir(tmp_path, code)


def test_coherent_parameter_makes_every_connected_texture_coherent(tmp_path):
    code = translate(
        tmp_path,
        """shader N {
    uimage2D plain @binding(0) @r32ui;
    uimage2D other @binding(1) @r32ui;
    void inner(uimage2D t @r32ui @coherent) { imageStore(t, ivec2(0), uvec4(1u)); }
    void outer(uimage2D t @r32ui) { inner(t); }
    compute {
        void main() {
            outer(plain);
            imageStore(other, ivec2(0), uvec4(2u));
        }
    }
}""",
        "metal",
    )

    assert f"void inner(texture2d<uint, access::read_write, {DEVICE}> t)" in flat(code)
    assert f"void outer(texture2d<uint, access::read_write, {DEVICE}> t)" in flat(code)
    assert f"{DEVICE}> plain [[texture(0)]]" in flat(code)
    assert "texture2d<uint, access::read_write> other [[texture(1)]]" in flat(code)
    metal_ir(tmp_path, code)


def test_coherent_global_used_by_helper_keeps_coherence(tmp_path):
    code = translate(
        tmp_path,
        """shader C {
    uimage2D img @binding(0) @r32ui @coherent;
    void put() { imageStore(img, ivec2(0), uvec4(1u)); }
    compute { void main() { put(); } }
}""",
        "metal",
    )

    assert f"void put(texture2d<uint, access::read_write, {DEVICE}> img)" in flat(code)
    metal_ir(tmp_path, code)


def test_texture_arrays_texel_buffers_and_readonly_images(tmp_path):
    code = translate(
        tmp_path,
        """shader C {
    image2D imgs[2] @binding(0) @rgba8 @coherent;
    uimageBuffer tb @binding(2) @r32ui @coherent;
    uimage2D src @binding(3) @r32ui @coherent @readonly;
    uimage2D dst @binding(4) @r32ui;
    compute {
        void main() {
            imageStore(imgs[1], ivec2(0), vec4(1.0));
            imageStore(tb, 0, uvec4(1u));
            imageStore(dst, ivec2(0), imageLoad(src, ivec2(0)));
        }
    }
}""",
        "metal",
    )

    assert f"array<texture2d<float, access::read_write, {DEVICE}>, 2> imgs" in flat(
        code
    )
    assert f"texture_buffer<uint, access::read_write, {DEVICE}> tb" in flat(code)
    assert f"texture2d<uint, access::read, {DEVICE}> src" in flat(code)
    assert "texture2d<uint, access::read_write> dst" in flat(code)
    metal_ir(tmp_path, code)


def test_threadgroup_coherence_is_the_texture_default(tmp_path):
    code = translate(
        tmp_path,
        """shader C {
    uimage2D img @binding(0) @r32ui @coherent(threadgroup);
    compute { void main() { imageStore(img, ivec2(0), uvec4(1u)); } }
}""",
        "metal",
    )

    assert "memory_coherence" not in code
    assert "texture2d<uint, access::read_write> img" in flat(code)


@pytest.mark.parametrize(
    ("attribute", "match"),
    [
        ("@volatile", "volatile storage texture 'img'"),
        ("@coherent(system)", "storage texture 'img' is coherent at system scope"),
    ],
)
def test_texture_coherence_msl_cannot_state_is_refused(tmp_path, attribute, match):
    source = f"""shader C {{
    uimage2D img @binding(0) @r32ui {attribute};
    compute {{ void main() {{ imageStore(img, ivec2(0), uvec4(1u)); }} }}
}}"""

    with pytest.raises(MetalLoweringRefused, match=re.escape(match)):
        translate(tmp_path, source, "metal")


@pytest.mark.parametrize(
    ("type_text", "expected"),
    [
        (
            "texture2d<uint, access::read_write>",
            f"texture2d<uint, access::read_write, {DEVICE}>",
        ),
        ("texture2d<float>", f"texture2d<float, access::sample, {DEVICE}>"),
        ("texture_buffer<uint>", f"texture_buffer<uint, access::read, {DEVICE}>"),
        (
            f"texture3d<half, access::write, {DEVICE}>",
            f"texture3d<half, access::write, {DEVICE}>",
        ),
        ("device uint*", "device uint*"),
    ],
)
def test_with_device_texture_coherence(type_text, expected):
    assert with_device_texture_coherence(type_text) == expected
