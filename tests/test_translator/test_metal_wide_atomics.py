"""Metal output writes CrossGL's 64-bit unsigned atomicMax and atomicMin with Metal's 64-bit atomics.

Metal's only 64-bit atomics are ``atomic_max_explicit`` and ``atomic_min_explicit`` on device ``atomic_ulong`` (MSL
4.1, section 6.15.4.7), and they return nothing. The generator used to refuse every 64-bit atomicMax and atomicMin.
A call whose result is discarded now becomes the Metal operation; one whose result is used, or whose target is
threadgroup memory, is still refused, since Metal cannot express it.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused

CROSSGL = textwrap.dedent("""
    shader S {
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            buffer uint* uin;
            buffer uint64_t* outp;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                uint64_t a1 = uint64_t(uin[i]);
                atomicMax(outp[i], a1);
                atomicMin(outp[i + 1u], a1 + uint64_t(1));
            }
        }
    }
    """)


def _translate(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal", format_output=False)


def test_discarded_wide_atomic_min_and_max_use_metal_64_bit_atomics(tmp_path):
    metal = _translate(tmp_path, CROSSGL)

    assert (
        "atomic_max_explicit(reinterpret_cast<device atomic_ulong*>(&outp[i]), a1, "
        "memory_order_relaxed);"
    ) in metal
    assert (
        "atomic_min_explicit(reinterpret_cast<device atomic_ulong*>(&outp[i + 1u]), "
    ) in metal

    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return
    shader = tmp_path / "out.metal"
    shader.write_text(metal, encoding="utf-8")
    result = subprocess.run(
        [xcrun, "-sdk", "macosx", "metal", "-std=metal3.1", "-c", str(shader)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
        check=False,
    )
    if "unable to find utility" not in result.stderr:
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("replacement", "reason"),
    [
        ("uint64_t old = atomicMax(outp[i], a1);", "return no value"),
        ("outp[i] = atomicMin(outp[i], a1);", "return no value"),
    ],
)
def test_used_wide_atomic_result_is_refused(tmp_path, replacement, reason):
    source = CROSSGL.replace("atomicMax(outp[i], a1);", replacement)
    with pytest.raises(MetalLoweringRefused, match=reason):
        _translate(tmp_path, source)


def test_threadgroup_wide_atomic_is_refused(tmp_path):
    source = CROSSGL.replace(
        "        void main() {",
        "        shared uint64_t tile[4];\n        void main() {",
    ).replace("atomicMax(outp[i], a1);", "atomicMax(tile[i], a1);")
    assert "shared uint64_t tile[4];" in source
    with pytest.raises(MetalLoweringRefused, match="requires device memory"):
        _translate(tmp_path, source)
