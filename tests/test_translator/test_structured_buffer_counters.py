"""Hidden ``RWStructuredBuffer`` counters reach Metal and GLSL as explicit counter buffers.

HLSL ``IncrementCounter()`` returns the counter before the increment and ``DecrementCounter()`` the counter after
the decrement. Neither target has a hidden counter, so each counted buffer gets a ``<name>Counter`` buffer that the
host binds, like the counter of an ``AppendStructuredBuffer``, and helper functions receive it beside the buffer.
"""

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.structured_buffer_counters import (
    materialize_structured_buffer_counters,
)

ROOT = Path(__file__).resolve().parents[2]
REQUIRE_ENV = "CROSSTL_REQUIRE_BUFFER_COUNTER_EXECUTION"

HLSL_COUNTERS = """
struct Item { uint value; };
RWStructuredBuffer<Item> items : register(u0);
AppendStructuredBuffer<uint> appended : register(u1);

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint slot = items.IncrementCounter();
    items[slot].value = tid.x;
    uint back = items.DecrementCounter();
    appended.Append(back);
}
"""

HLSL_HELPERS = """
RWStructuredBuffer<uint> ups : register(u0);
RWStructuredBuffer<uint> downs : register(u1);

uint push(RWStructuredBuffer<uint> q, uint v) { uint i = q.IncrementCounter(); q[i] = v; return i; }
uint pop(RWStructuredBuffer<uint> q) { return q.DecrementCounter(); }
uint forward(RWStructuredBuffer<uint> q, uint v) { return push(q, v); }

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    forward(ups, tid.x + 1u);
    uint slot = pop(downs);
    downs[slot] = tid.x + 1u;
}
"""


def translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"shader.{suffix}"
    path.write_text(textwrap.dedent(source).strip() + "\n", encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


def run(command):
    completed = subprocess.run(
        [str(item) for item in command], capture_output=True, text=True, timeout=180
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def metal_compiler():
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        return None
    probe = subprocess.run(
        [xcrun, "-sdk", "macosx", "-f", "metal"], capture_output=True, text=True
    )
    return xcrun if probe.returncode == 0 else None


def compile_metal(source, tmp_path):
    """Compile ``source``; returns the library, or ``None`` without a Metal toolchain."""
    xcrun = metal_compiler()
    if xcrun is None:
        return None
    shader = tmp_path / "counters.metal"
    shader.write_text(source, encoding="utf-8")
    air, library = tmp_path / "counters.air", tmp_path / "counters.metallib"
    run([xcrun, "-sdk", "macosx", "metal", "-Werror", "-c", shader, "-o", air])
    run([xcrun, "-sdk", "macosx", "metallib", air, "-o", library])
    return library


def validate_glsl(source, tmp_path):
    validator = shutil.which("glslangValidator")
    if validator is None:
        return
    shader = tmp_path / "counters.comp"
    shader.write_text(source, encoding="utf-8")
    run([validator, "-S", "comp", shader])


def kernel_slots(metal):
    kernel = re.search(r"kernel void (\w+)\((.*?)\) \{", metal, re.S)
    assert kernel is not None, metal
    slots = {
        name: int(index)
        for name, index in re.findall(r"(\w+) \[\[buffer\((\d+)\)\]\]", kernel.group(2))
    }
    return kernel.group(1), slots


def flat(code):
    """``code`` on one line, with no space after an opening or before a closing parenthesis."""
    return re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", re.sub(r"\s+", " ", code)))


def parse_crossgl(source):
    return Parser(Lexer(textwrap.dedent(source)).get_tokens()).parse()


def test_metal_counter_methods_use_a_counter_buffer(tmp_path):
    metal = translate(tmp_path, HLSL_COUNTERS, "hlsl", "metal")
    assert "buffer_increment_counter" not in metal
    assert "buffer_decrement_counter" not in metal
    _, slots = kernel_slots(metal)
    # Explicit registers keep their slots; the counters take the next free ones.
    assert slots == {"items": 0, "itemsCounter": 2, "appended": 1, "appendedCounter": 3}
    compact = flat(metal)
    # IncrementCounter returns the old value, DecrementCounter the new one.
    assert (
        "uint slot = atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>"
        "(&itemsCounter[0u]), 1u, memory_order_relaxed);"
    ) in compact, metal
    assert (
        "uint back = atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>"
        "(&itemsCounter[0u]), 4294967295u, memory_order_relaxed) - 1u;"
    ) in compact, metal
    compile_metal(metal, tmp_path)


def test_glsl_counter_methods_use_a_counter_buffer(tmp_path):
    glsl = translate(tmp_path, HLSL_COUNTERS, "hlsl", "opengl")
    assert "buffer_increment_counter" not in glsl
    assert (
        "layout(std430, binding = 2) buffer itemsCounterBuffer { uint itemsCounter[]; };"
        in glsl
    )
    assert "uint slot = atomicAdd(itemsCounter[0u], 1u);" in glsl
    assert "uint back = (atomicAdd(itemsCounter[0u], 4294967295u) - 1u);" in glsl
    validate_glsl(glsl, tmp_path)


def test_helpers_receive_the_counter_beside_the_buffer(tmp_path):
    metal = translate(tmp_path, HLSL_HELPERS, "hlsl", "metal")
    compact = flat(metal)
    assert "uint push(device uint* q, device uint* qCounter, uint v)" in compact
    assert "uint forward(device uint* q, device uint* qCounter, uint v)" in compact
    assert "push(q, qCounter, v)" in compact
    assert "forward(ups, upsCounter, tid.x + 1u)" in compact
    assert "pop(downs, downsCounter)" in compact
    _, slots = kernel_slots(metal)
    assert slots == {"ups": 0, "upsCounter": 2, "downs": 1, "downsCounter": 3}
    compile_metal(metal, tmp_path)

    glsl = translate(tmp_path, HLSL_HELPERS, "hlsl", "opengl")
    assert "buffer_increment_counter" not in glsl
    assert re.search(r"atomicAdd\(upsCounter\[\w+\], 1u\)", glsl), glsl
    assert re.search(r"atomicAdd\(downsCounter\[\w+\], 4294967295u\) - 1u", glsl), glsl
    validate_glsl(glsl, tmp_path)


def test_glsl_arrays_of_counted_buffers_get_arrays_of_counters(tmp_path):
    glsl = translate(
        tmp_path,
        """
        RWStructuredBuffer<uint> queues[2] : register(u0);
        [numthreads(64, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID)
        {
            uint k = tid.x & 1u;
            uint slot = queues[k].IncrementCounter();
            queues[k][slot] = tid.x;
        }
        """,
        "hlsl",
        "opengl",
    )
    compact = flat(glsl)
    assert "buffer_increment_counter" not in glsl
    assert (
        "layout(std430, binding = 2) buffer queuesCounterBuffer { uint data[]; } queuesCounter[2];"
        in compact
    ), glsl
    # One buffer of a block array keeps its elements in the block member.
    assert "uint slot = atomicAdd(queuesCounter[k].data[0u], 1u);" in compact
    assert "queues[k].data[slot] = gl_GlobalInvocationID.x;" in compact
    validate_glsl(glsl, tmp_path)


def test_counter_names_avoid_program_identifiers(tmp_path):
    metal = translate(
        tmp_path,
        """
        RWStructuredBuffer<uint> items : register(u0);
        RWStructuredBuffer<uint> itemsCounter : register(u1);
        [numthreads(1, 1, 1)]
        void main(uint3 tid : SV_DispatchThreadID)
        {
            itemsCounter[0] = items.IncrementCounter();
        }
        """,
        "hlsl",
        "metal",
    )
    _, slots = kernel_slots(metal)
    assert slots == {"items": 0, "itemsCounter": 1, "itemsCounter_1": 2}
    assert "&itemsCounter_1[0u]" in metal
    compile_metal(metal, tmp_path)


@pytest.mark.parametrize("backend", ["metal", "opengl"])
@pytest.mark.parametrize(
    "declaration, call, reason",
    [
        (
            "StructuredBuffer<uint> data;",
            "buffer_increment_counter(data)",
            "'data' is not an RWStructuredBuffer",
        ),
        (
            "AppendStructuredBuffer<uint> data;",
            "buffer_decrement_counter(data)",
            "'data' is not an RWStructuredBuffer",
        ),
        (
            "RWStructuredBuffer<uint> data;",
            "buffer_increment_counter(data, 1u)",
            "takes exactly one buffer operand",
        ),
    ],
)
def test_counters_without_a_counted_buffer_are_refused(
    tmp_path, backend, declaration, call, reason
):
    source = f"""
    shader main {{
        {declaration}
        RWStructuredBuffer<uint> results;
        compute {{
            @ numthreads(1, 1, 1)
            void main() {{
                results[0] = {call};
            }}
        }}
    }}
    """
    with pytest.raises(Exception, match=re.escape(reason)):
        translate(tmp_path, source, "cgl", backend)


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_cuda_and_hip_counters_use_a_counter_buffer(tmp_path, backend):
    code = translate(tmp_path, HLSL_HELPERS, "hlsl", backend)
    assert "buffer_increment_counter" not in code
    assert "buffer_decrement_counter" not in code
    assert "name=upsCounter kind=buffer set=0 binding=2" in code
    assert "name=downsCounter kind=buffer set=0 binding=3" in code
    assert "atomicAdd(&qCounter[0u], 1u)" in code
    assert "(atomicAdd(&qCounter[0u], 4294967295u) - 1u)" in code
    with pytest.raises(ValueError, match="'data' is not an RWStructuredBuffer"):
        translate(
            tmp_path,
            """
            shader main {
                StructuredBuffer<uint> data;
                RWStructuredBuffer<uint> results;
                compute {
                    @ numthreads(1, 1, 1)
                    void main() { results[0] = buffer_increment_counter(data); }
                }
            }
            """,
            "cgl",
            backend,
        )


def test_slang_keeps_the_counter_methods(tmp_path):
    slang = translate(tmp_path, HLSL_HELPERS, "hlsl", "slang")
    assert "uint i = q.IncrementCounter();" in slang
    assert "return q.DecrementCounter();" in slang
    with pytest.raises(Exception, match="requires a read-write RWStructuredBuffer"):
        translate(
            tmp_path,
            """
            shader main {
                StructuredBuffer<uint> data;
                RWStructuredBuffer<uint> results;
                compute {
                    @ numthreads(1, 1, 1)
                    void main() { results[0] = buffer_increment_counter(data); }
                }
            }
            """,
            "cgl",
            "slang",
        )


def test_mojo_refuses_counter_calls(tmp_path):
    with pytest.raises(
        ValueError, match="structured-buffer counters are not available"
    ):
        translate(tmp_path, HLSL_COUNTERS, "hlsl", "mojo")


def test_counter_through_ambiguous_overloads_is_refused(tmp_path):
    source = """
    shader main {
        RWStructuredBuffer<uint> data;
        RWStructuredBuffer<float> other;
        uint bump(RWStructuredBuffer<uint> q) { return buffer_increment_counter(q); }
        uint bump(RWStructuredBuffer<float> q) { return 0u; }
        compute {
            @ numthreads(1, 1, 1)
            void main() {
                data[0] = bump(data);
            }
        }
    }
    """
    for backend in ("metal", "opengl"):
        with pytest.raises(Exception, match="overloads of 'bump' with 1 parameters"):
            translate(tmp_path, source, "cgl", backend)


def test_counter_index_with_side_effects_is_refused(tmp_path):
    source = """
    shader main {
        RWStructuredBuffer<uint> queues[2];
        uint bump(RWStructuredBuffer<uint> q) { return buffer_increment_counter(q); }
        compute {
            @ numthreads(1, 1, 1)
            void main() {
                int k = 0;
                queues[0][0] = bump(queues[k++]);
            }
        }
    }
    """
    with pytest.raises(Exception, match="the buffer index has side effects"):
        translate(tmp_path, source, "cgl", "opengl")


def test_materialization_leaves_its_input_unchanged():
    ast = parse_crossgl("""
        shader main {
            RWStructuredBuffer<uint> data;
            compute {
                @ numthreads(1, 1, 1)
                void main() { data[0] = buffer_increment_counter(data); }
            }
        }
        """)
    lowered = materialize_structured_buffer_counters(ast)
    assert lowered is not ast
    assert [v.name for v in ast.global_variables] == ["data"]
    assert [v.name for v in lowered.global_variables] == ["data", "dataCounter"]
    names = {getattr(getattr(n, "function", None), "name", None) for n in ast.walk()}
    assert "buffer_increment_counter" in names
    lowered_names = {
        getattr(getattr(n, "function", None), "name", None) for n in lowered.walk()
    }
    assert "buffer_increment_counter" not in lowered_names
    assert "atomicAdd" in lowered_names
    assert materialize_structured_buffer_counters(lowered) is lowered


def test_counters_execute_on_metal(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1" or sys.platform != "darwin":
        pytest.skip(f"set {REQUIRE_ENV}=1 on macOS to require the Metal dispatch")
    metal = translate(tmp_path, HLSL_HELPERS, "hlsl", "metal")
    library = compile_metal(metal, tmp_path)
    assert library is not None, "the Metal toolchain is required"
    entry, slots = kernel_slots(metal)
    threads = 64
    payloads = {
        "ups": struct.pack(f"<{threads}I", *([0] * threads)),
        "upsCounter": struct.pack("<I", 0),
        "downs": struct.pack(f"<{threads}I", *([0] * threads)),
        "downsCounter": struct.pack("<I", threads),
    }
    assert sorted(slots) == sorted(payloads), slots
    files = []
    for name, index in sorted(slots.items(), key=lambda item: item[1]):
        path = tmp_path / f"buffer-{index}.in"
        path.write_bytes(payloads[name])
        files.append(str(path))
    runner = tmp_path / "runner"
    run(
        [
            "swiftc",
            ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
            "-o",
            runner,
        ]
    )
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": files,
                "workgroupCount": [1, 1, 1],
                "workgroupSize": [threads, 1, 1],
                "simdWidth": 32,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    run([runner, library, entry, request, output])

    def words(name, count):
        data = (output / f"buffer-{slots[name]}.bin").read_bytes()
        return list(struct.unpack(f"<{count}I", data[: 4 * count]))

    # Every increment hands out a distinct old value 0..63, and every decrement a distinct new value 63..0.
    assert words("upsCounter", 1) == [threads]
    assert words("downsCounter", 1) == [0]
    assert sorted(words("ups", threads)) == list(range(1, threads + 1))
    assert sorted(words("downs", threads)) == list(range(1, threads + 1))
