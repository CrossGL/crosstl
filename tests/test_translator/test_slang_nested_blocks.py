"""Slang output writes a nested ``{ ... }`` block as a block (issue #3270).

The Slang emitter had no case for a ``BlockNode`` statement and wrote its repr, ``BlockNode(statements=2);``, which
slangc rejects. A block now keeps its braces, and a declaration in it ends with the block, so a later use of a
shadowed name reads the outer variable's type again.
"""

import shutil
import subprocess

import crosstl

BLOCKS = """
shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer int* iin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int x = iin[i];
            float r = 0.0;
            {
                float x = 2.0;
                r += x;
            }
            switch (iin[i] & 1) {
                case 0: {
                    float t = fin[i];
                    r += t;
                    break;
                }
                default: {
                    int t = iin[i] * 4;
                    r += float(t);
                    break;
                }
            }
            outp[i] = r * 100.0 + float(x / 2);
        }
    }
}
"""


def _slang(tmp_path):
    path = tmp_path / "blocks.cgl"
    path.write_text(BLOCKS)
    return crosstl.translate(str(path), backend="slang", format_output=False)


def test_nested_blocks_keep_their_braces(tmp_path):
    code = _slang(tmp_path)
    assert "BlockNode" not in code
    compact = " ".join(code.split())
    assert "{ float x = 2.0; r += x; }" in compact
    assert "int t = iin[i] * 4;" in compact
    # The outer int x is in scope again after the block: x / 2 stays an integer division.
    assert "float(x / 2)" in compact


def test_slangc_accepts_nested_blocks(tmp_path):
    slangc = shutil.which("slangc")
    if slangc is None:
        return
    source = tmp_path / "blocks.slang"
    source.write_text(_slang(tmp_path))
    result = subprocess.run(
        [slangc, str(source), "-target", "spirv", "-o", str(tmp_path / "blocks.spv")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
