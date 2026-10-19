"""Integer locals of precise functions written from block-scoped locals."""

import shutil
import subprocess
import textwrap

import crosstl

PRECISE_LOOP_LOCALS = """
shader PreciseLoopLocals {
    @ precise
    float Root(uint bits) {
        uint root = 0u;
        uint remainder = 0u;
        int total = 0;
        for (int digit = 0; digit < 4; digit += 1) {
            remainder = (remainder << 2u) | (bits >> 30u);
            bits <<= 2u;
            root <<= 1u;
            uint trial = (root << 1u) | 1u;
            float weight = float(digit) * 0.5;
            if (remainder >= trial) {
                remainder -= trial;
                root |= 1u;
            }
            total += weight;
        }
        return float(root) + float(total);
    }

    compute {
        buffer float results[];

        @ NumThreads(1, 1, 1)
        void main() {
            results[0] = Root(7u);
        }
    }
}
"""


def translate_opengl(tmp_path):
    path = tmp_path / "precise_loop_locals.cgl"
    path.write_text(textwrap.dedent(PRECISE_LOOP_LOCALS), encoding="utf-8")
    return crosstl.translate(str(path), backend="opengl", format_output=False)


def test_integer_writes_from_integer_block_locals_stay_unqualified(tmp_path):
    code = translate_opengl(tmp_path)

    # remainder -= trial subtracts a uint declared later, inside the loop.
    assert "\n    uint remainder = 0u;\n" in code
    assert "\n    uint root = 0u;\n" in code
    assert "remainder -= trial;" in code


def test_integer_writes_from_float_block_locals_stay_precise(tmp_path):
    code = translate_opengl(tmp_path)

    # total += weight adds a float declared later, inside the loop.
    assert "\n    precise int total = 0;\n" in code
    assert "precise float weight = (float(digit) * 0.5);" in code
    assert "precise float crossglPreciseReturn" in code

    validator = shutil.which("glslangValidator")
    if validator is not None:
        shader = tmp_path / "shader.comp"
        shader.write_text(code, encoding="utf-8")
        completed = subprocess.run(
            [validator, "-S", "comp", str(shader)], capture_output=True, text=True
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
