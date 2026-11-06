"""Read-only locals end with their block in Metal code generation.

The Metal generator refuses a store to a ``const`` local. It recorded read-only locals by name for
the whole function, so a ``const int v`` declared in a loop body or a nested block made a later,
writable ``int v`` of the outer scope look read-only, and every store to it was refused as
"const-qualified local". The read-only set now follows block scope, and a writable declaration
hides an earlier read-only one of the same name.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused

METAL = textwrap.dedent("""\
    #include <metal_stdlib>
    using namespace metal;

    kernel void k(device int * dst [[buffer(0)]],
                  device const int * src [[buffer(1)]],
                  uint tid [[thread_position_in_grid]])
    {
        int low = 0;
        while (low < int(tid)) {
            const int idx0 = src[low];
            low += idx0 > 0 ? 1 : 2;
        }
        int idx0 = 0;
        if (low < 4) {
            idx0 = src[low];
        }
        int v = int(tid);
        {
            const int v = 3;
            dst[0] = v;
        }
        v = v + idx0;
        dst[tid] = v;
    }
    """)

GLSL = textwrap.dedent("""\
    #version 450
    layout(local_size_x = 64) in;
    layout(std430, binding = 0) buffer Out { int dst[]; };
    layout(std430, binding = 1) readonly buffer In { int src[]; };
    void main() {
        uint tid = gl_GlobalInvocationID.x;
        int low = 0;
        while (low < int(tid)) {
            const int idx0 = src[low];
            low += idx0 > 0 ? 1 : 2;
        }
        int idx0 = 0;
        if (low < 4) {
            idx0 = src[low];
        }
        dst[tid] = idx0;
    }
    """)

STILL_CONST = textwrap.dedent("""\
    #include <metal_stdlib>
    using namespace metal;

    kernel void k(device int * dst [[buffer(0)]],
                  uint tid [[thread_position_in_grid]])
    {
        int v = int(tid);
        {
            const int v = 3;
            v = 4;
            dst[0] = v;
        }
        dst[tid] = v;
    }
    """)


def _translate(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal", format_output=False)


def test_outer_local_is_writable_after_a_const_local_of_a_loop_body(tmp_path):
    code = _translate(tmp_path, "scopes.metal", METAL)
    assert "const int idx0 = src[low];" in code
    assert "int idx0 = 0;" in code
    assert "idx0 = src[low];" in code
    assert "v = v + idx0;" in code


def test_glsl_outer_local_is_writable_after_a_const_local(tmp_path):
    code = _translate(tmp_path, "scopes.comp", GLSL)
    assert "int idx0 = 0;" in code
    assert "idx0 = (*reinterpret_cast<const device int*>(in_ + (low * 4)));" in code
    assert "(*reinterpret_cast<device int*>(out + (tid * 4))) = idx0;" in code


def test_store_to_const_local_inside_its_block_is_still_refused(tmp_path):
    with pytest.raises(MetalLoweringRefused, match="const-qualified local"):
        _translate(tmp_path, "const.metal", STILL_CONST)


def _metal_compiler_available():
    if shutil.which("xcrun") is None:
        return False
    probe = subprocess.run(
        ["xcrun", "-f", "metal"], capture_output=True, text=True, timeout=60
    )
    return probe.returncode == 0


@pytest.mark.parametrize(
    "name,source",
    [("scopes.metal", METAL), ("scopes.comp", GLSL)],
    ids=["metal", "glsl"],
)
def test_outputs_compile_with_the_metal_compiler(tmp_path, name, source):
    if not _metal_compiler_available():
        pytest.skip("xcrun metal not installed")
    code = _translate(tmp_path, name, source)
    (tmp_path / "out.metal").write_text(code, encoding="utf-8")
    result = subprocess.run(
        ["xcrun", "metal", "-c", "out.metal", "-o", "out.air"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
