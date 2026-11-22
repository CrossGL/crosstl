"""Slang output writes the CrossGL shader clock built-ins, and Slang input imports Slang's device-scope clock reads.

Slang v2026.20's core module declares ``clockARB()``, ``clock2x32ARB()`` (subgroup scope) and ``getRealtimeClock()``
(device scope, two 32-bit words, low word first) with ``getRealtimeClockLow()``. The Slang generator refused every
CrossGL clock built-in as one Slang has no built-in for. It now writes the Slang reads, joining the two words of one
``getRealtimeClock()`` read for the 64-bit ``clockRealtimeEXT()``.
"""

import os
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.slang_codegen import SlangCodeGen
from crosstl.translator.codegen.slang_refusals import SlangLoweringRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

CLOCKS = """
shader Clocks {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer uint64_t* wide;
        buffer uvec2* pairs;

        void main() {
            wide[0] = clockARB();
            wide[1] = clockRealtimeEXT();
            pairs[0] = clock2x32ARB();
            pairs[1] = clockRealtime2x32EXT();
        }
    }
}
"""

SLANG_CLOCKS = """RWStructuredBuffer<uint> result;

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint2 words = getRealtimeClock();
    uint low = getRealtimeClockLow();
    uint64_t ticks = clockARB();
    uint2 pair = clock2x32ARB();
    result[0] = words.x + low + uint(ticks) + pair.y;
}
"""


def _slang(source):
    return SlangCodeGen().generate(Parser(Lexer(source).get_tokens()).parse())


def _slangc_spirv(tmp_path, source):
    slangc = shutil.which("slangc")
    if slangc is None:
        pytest.skip("slangc is not installed")
    path = tmp_path / "clocks.slang"
    path.write_text(source)
    compiled = subprocess.run(
        [
            slangc,
            str(path),
            "-entry",
            "main",
            "-stage",
            "compute",
            "-target",
            "spirv",
            "-o",
            str(tmp_path / "clocks.spv"),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "SLANG_RUN_SPIRV_VALIDATION": "1"},
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr


def test_slang_output_writes_each_clock_read():
    slang = _slang(CLOCKS)

    assert "wide[0] = clockARB();" in slang
    assert "pairs[0] = clock2x32ARB();" in slang
    assert "pairs[1] = getRealtimeClock();" in slang
    assert "wide[1] = crosstl_clock_realtime();" in slang
    # One read of the device clock gives both words; the first is the low word.
    assert "uint2 words = getRealtimeClock();" in slang
    assert "return (uint64_t(words.y) << 32) | uint64_t(words.x);" in slang
    assert slang.count("uint64_t crosstl_clock_realtime()") == 1


def test_slang_output_of_the_clock_reads_compiles_to_spirv(tmp_path):
    _slangc_spirv(tmp_path, _slang(CLOCKS))


def test_the_realtime_helper_avoids_a_user_function_of_its_name():
    slang = _slang("""
    shader Clocks {
        uint crosstl_clock_realtime() { return 7u; }
        compute {
            layout(local_size_x = 1) in;
            buffer uint64_t* wide;
            void main() {
                wide[0] = clockRealtimeEXT() + uint64_t(crosstl_clock_realtime());
            }
        }
    }
    """)

    assert "uint64_t crosstl_clock_realtime_1()" in slang
    assert "crosstl_clock_realtime_1() + uint64_t(crosstl_clock_realtime())" in slang


def test_a_clock_read_with_arguments_is_refused():
    with pytest.raises(SlangLoweringRefused, match="'clockARB' takes no arguments"):
        _slang("""
        shader Clocks {
            compute {
                layout(local_size_x = 1) in;
                buffer uint64_t* wide;
                void main() {
                    wide[0] = clockARB(1u);
                }
            }
        }
        """)


def test_slang_input_imports_the_device_clock_reads(tmp_path):
    path = tmp_path / "clocks.slang"
    path.write_text(SLANG_CLOCKS)

    crossgl = crosstl.translate(str(path), backend="cgl")

    assert "uvec2 words = clockRealtime2x32EXT();" in crossgl
    assert "uint low = clockRealtime2x32EXT().x;" in crossgl
    assert "uint64_t ticks = clockARB();" in crossgl
    assert "uvec2 pair = clock2x32ARB();" in crossgl
    assert "getRealtimeClock" not in crossgl

    slang = crosstl.translate(str(path), backend="slang")

    assert "uint2 words = getRealtimeClock();" in slang
    assert "uint low = getRealtimeClock().x;" in slang
    _slangc_spirv(tmp_path, slang)


def test_slang_input_keeps_a_user_function_named_like_a_clock_read(tmp_path):
    path = tmp_path / "clocks.slang"
    path.write_text("""RWStructuredBuffer<uint> result;

uint getRealtimeClockLow() { return 7u; }

[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    result[0] = getRealtimeClockLow();
}
""")

    crossgl = crosstl.translate(str(path), backend="cgl")

    assert "clockRealtime2x32EXT" not in crossgl
    assert "getRealtimeClockLow()" in crossgl
