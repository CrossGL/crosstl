"""Reference arguments to Metal helper parameters that carry an address space.

An ``inout`` parameter qualified ``device`` or ``threadgroup`` lowers to a Metal reference in that address space
(``device int&``). Such a reference binds an lvalue of the same address space directly; a ``thread`` temporary
cannot bind it (Metal Shading Language 4.0, section 4: "Address Spaces"), so storage of another address space is
refused rather than copied.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

MATCHING_SPACES = """shader S {
    void increment(inout device int value) { value += 3; }
    void accumulate(inout device int value, inout threadgroup int total) { value *= 2; total += value; }
    compute {
        @ stage_entry
        void mutate(device int& value @buffer(0), RWStructuredBuffer<int> result @buffer(1)) {
            threadgroup int[2] tile;
            tile[0] = 0;
            value = value * 2;
            increment(value);
            increment(result[1]);
            accumulate(result[0], tile[0]);
            buffer_store(result, 2, tile[0]);
        }
    }
}
"""

THREAD_ARGUMENT = """shader S {
    void increment(inout device int value) { value += 3; }
    compute {
        @ stage_entry
        void mutate(RWStructuredBuffer<int> result @buffer(0)) {
            int local = 1;
            increment(local);
            buffer_store(result, 0, local);
        }
    }
}
"""


def _translate(tmp_path, source):
    path = tmp_path / "shader.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal")


def test_matching_address_space_binds_without_temporaries(tmp_path):
    code = _translate(tmp_path, MATCHING_SPACES)
    assert "__crossgl_ref" not in code
    # Output is clang-formatted only where clang-format is installed.
    assert re.search(r"void increment\(device int\s*&\s*value\)", code)
    assert "increment(value);" in code
    assert "increment(result[1]);" in code
    assert "accumulate(result[0], tile[0]);" in code
    if shutil.which("xcrun") is None:
        return
    source = tmp_path / "shader.metal"
    source.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [
            "xcrun",
            "--sdk",
            "macosx",
            "metal",
            "-Werror",
            "-c",
            str(source),
            "-o",
            str(tmp_path / "shader.air"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_other_address_space_argument_is_refused(tmp_path):
    with pytest.raises(Exception, match="device reference parameter of 'increment'"):
        _translate(tmp_path, THREAD_ARGUMENT)
