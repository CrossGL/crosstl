"""CrossGL atomics on workgroup-shared values and on texels reach Slang and Metal as valid code.

Slang has no ``atomicAdd``: an atomic on a ``groupshared`` value is an
``Interlocked*`` call that returns the original value through an output
argument. Metal's ``imageAtomicCompSwap`` helper takes signed coordinates, so
unsigned ones are converted, and a discarded image atomic does not leave an
unused ``.x`` selection.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.slang_refusals import SlangLoweringRefused

WORKGROUP_ATOMICS = """shader WorkgroupAtomics {
    layout(std430, binding = 0) buffer Results {
        int data[];
    } results;
    shared uint count;
    shared int bins[64];
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        void main() {
            atomicAdd(count, 1u);
            int prior = atomicMin(bins[gl_LocalInvocationIndex], 3) + 1;
            int swapped = atomicCompSwap(bins[0], 1, 2);
            barrier();
            results.data[gl_LocalInvocationIndex] = prior + swapped + int(count);
        }
    }
}
"""

TEXEL_ATOMICS = """shader TexelAtomics {
    @ format(r32ui) uimage2D counters;
    compute {
        layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
        void main() {
            uvec2 pixel = gl_GlobalInvocationID.xy;
            imageAtomicAdd(counters, ivec2(pixel), 1u);
            uint prior = imageAtomicCompSwap(counters, pixel, 0u, 5u);
            imageAtomicMax(counters, ivec2(pixel), prior);
        }
    }
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _run(tmp_path, command):
    result = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, check=False, timeout=180
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]


def test_slang_writes_interlocked_calls_for_groupshared_atomics(tmp_path):
    slang = _translate(tmp_path, WORKGROUP_ATOMICS, "slang")
    statements = [line.strip() for line in slang.splitlines()]

    for statement in (
        "InterlockedAdd(count, 1u);",
        "int cgl_atomicMin_original;",
        "InterlockedMin(bins[gl_LocalInvocationIndex], 3, cgl_atomicMin_original);",
        "int prior = cgl_atomicMin_original + 1;",
        "InterlockedCompareExchange(bins[0], 1, 2, cgl_atomicCompSwap_original);",
        "int swapped = cgl_atomicCompSwap_original;",
    ):
        assert statement in statements, slang
    assert "atomicAdd(" not in slang and "atomicMin(" not in slang

    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    (tmp_path / "out.slang").write_text(slang, encoding="utf-8")
    _run(
        tmp_path,
        [slangc, "out.slang", "-target", "spirv", "-entry", "main", "-stage", "compute"]
        + ["-o", "out.spv"],
    )


@pytest.mark.parametrize(
    ("declaration", "call", "reason"),
    [
        (
            "shared float total;",
            "atomicAdd(total, 1.0);",
            "needs an int or uint target",
        ),
        ("shared uint count;", "atomicAdd(count);", "takes a target and 1 value"),
        (
            "shared uint count;",
            "atomicCompSwap(count, 1u);",
            "takes a target and 2 value",
        ),
    ],
)
def test_slang_refuses_groupshared_atomics_it_cannot_write(
    tmp_path, declaration, call, reason
):
    source = WORKGROUP_ATOMICS.replace(
        "    shared uint count;\n", f"    {declaration}\n    shared uint count_;\n"
    ).replace("atomicAdd(count, 1u);", call)
    source = source.replace("int(count)", "0")
    with pytest.raises(SlangLoweringRefused, match=reason):
        _translate(tmp_path, source, "slang")


def test_metal_converts_coordinates_and_discards_image_atomic_results(tmp_path):
    metal = _translate(tmp_path, TEXEL_ATOMICS, "metal")
    statements = [line.strip() for line in metal.splitlines()]

    assert (
        "uint prior = imageAtomicCompSwap_uimage2D(counters, int2(pixel), 0u, 5u);" in (
            statements
        )
    ), (metal)
    assert (
        "(void)(counters.atomic_fetch_add(uint2(int2(pixel)), 1u).x);" in statements
    ), metal

    if shutil.which("xcrun") is None:
        pytest.skip("the Metal toolchain is not installed")
    (tmp_path / "out.metal").write_text(metal, encoding="utf-8")
    _run(
        tmp_path,
        [
            "xcrun",
            "-sdk",
            "macosx",
            "metal",
            "-Werror",
            "-c",
            "out.metal",
            "-o",
            "out.air",
        ],
    )
