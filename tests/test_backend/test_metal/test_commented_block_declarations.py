"""A comment before a nested block does not hide the block's declarations.

Template helper calls are materialized by deducing their arguments from the declarations in scope. The
statement scanner treated any text before a ``{`` as a statement prefix, so a comment line before a bare block
merged the brace into the first declaration inside it (``{ float value;``). That declaration was lost, the
helper's template argument could not be deduced, and template-hostile targets stopped with "missing out_t".
ggml's flash-attention kernels comment their nested blocks this way.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl

KERNEL = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    template <typename out_t>
    void widen(device const half *src, thread out_t &out) { out = out_t(*src); }

    kernel void k(device const half *src [[buffer(0)]],
                  device float *dst [[buffer(1)]],
                  uint gid [[thread_position_in_grid]])
    {
        if (gid < 64) {
            // The comment before this block is not part of a statement.
            {
                float value;
                widen(src + gid, value);
                dst[gid] = value;
            }
            /* Nor is a block comment. */ {
                int count;
                widen(src + gid + 1, count);
                dst[gid] += float(count);
            }
        }
    }
    """)


def _translate(tmp_path, backend):
    source = tmp_path / "blocks.metal"
    source.write_text(KERNEL, encoding="utf-8")
    return crosstl.translate(str(source), backend=backend)


def test_helper_arguments_are_deduced_inside_commented_blocks(tmp_path):
    code = _translate(tmp_path, "opengl")
    assert "void widen_float_" in code
    assert "void widen_int_" in code


_VALIDATORS = {
    "opengl": (
        "blocks.comp",
        [["glslangValidator", "-S", "comp", "blocks.comp"]],
    ),
    "directx": (
        "blocks.hlsl",
        [["dxc", "-T", "cs_6_2", "-enable-16bit-types", "-E", "CSMain", "blocks.hlsl"]],
    ),
    "metal": (
        "blocks.metal",
        [["xcrun", "-sdk", "macosx", "metal", "-c", "blocks.metal", "-o", "b.air"]],
    ),
}


@pytest.mark.parametrize("backend", sorted(_VALIDATORS))
def test_commented_blocks_reach_every_target(tmp_path, backend):
    file_name, commands = _VALIDATORS[backend]
    code = _translate(tmp_path, backend)
    missing = [command[0] for command in commands if shutil.which(command[0]) is None]
    if missing:
        pytest.skip(f"{', '.join(missing)} not installed")
    (tmp_path / file_name).write_text(code, encoding="utf-8")
    for command in commands:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=300, cwd=tmp_path
        )
        assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
