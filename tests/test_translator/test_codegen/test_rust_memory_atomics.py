"""CrossGL atomics in the rust-gpu output.

The Rust generator used to write ``atomicAdd(counter.total, 1u)`` as a call into a host ``gpu`` runtime that no
rust-gpu crate has. The rust-gpu output does not lower CrossGL atomics yet (``spirv_std::arch::atomic_*`` takes the
scope and memory semantics as const generic arguments), so each atomic call is refused by name. A program function
that shares an atomic's name is still called as declared.
"""

from __future__ import annotations

import pytest

import crosstl
from crosstl.translator.codegen.rust_refusals import RustLoweringRefused

from .test_rust_codegen import (
    assert_generated_rust_smoke_compiles,
    generate_code,
    parse_code,
    tokenize_code,
)

BUFFER_ATOMICS = """
shader AtomicUse {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer uint* slots;
        buffer int* signedSlots;
        void main() {
            uint previous = atomicAdd(slots[0], 1u);
            int low = atomicMin(signedSlots[0], -3);
            int high = atomicMax(signedSlots[1], 9);
            uint masked = atomicAnd(slots[1], 3u);
            uint joined = atomicOr(slots[2], 4u);
            uint flipped = atomicXor(slots[3], 8u);
            uint swapped = atomicCompSwap(slots[4], previous, 7u);
            uint replaced = atomicExchange(slots[5], 5u);
            uint tally = 2u;
            uint before = atomicAdd(tally, 3u);
            slots[6] = previous + masked + joined + flipped + swapped + replaced + before + uint(low + high);
        }
    }
}
"""


def _rust(source):
    return generate_code(parse_code(tokenize_code(source)))


def _translate(tmp_path, source, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="rust", format_output=False)


def test_buffer_and_local_atomics_are_refused_by_name():
    with pytest.raises(RustLoweringRefused) as refused:
        _rust(BUFFER_ATOMICS)
    assert refused.value.code == "project.translate.rust-lowering-unsupported"
    assert refused.value.reasons == tuple(
        f"function: '{name}' has no rust-gpu lowering for these operands"
        for name in (
            "atomicAdd",
            "atomicMin",
            "atomicMax",
            "atomicAnd",
            "atomicOr",
            "atomicXor",
            "atomicCompSwap",
            "atomicExchange",
        )
    )


def test_hlsl_style_atomics_are_canonicalized_before_rust_generation(tmp_path):
    # HLSL's out-argument forms reach the Rust generator as the CrossGL atomics, which it refuses by name.
    with pytest.raises(RustLoweringRefused) as refused:
        _translate(
            tmp_path,
            """
            shader OutArgument {
                compute {
                    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                    buffer uint* slots;
                    void main() {
                        uint original;
                        atomicAdd(slots[0], 2u, original);
                        uint exchanged;
                        atomicCompSwap(slots[1], 1u, original, exchanged);
                        slots[2] = original + exchanged;
                    }
                }
            }
            """,
        )
    assert refused.value.reasons == (
        "function: 'atomicAdd' has no rust-gpu lowering for these operands",
        "function: 'atomicCompSwap' has no rust-gpu lowering for these operands",
    )


def test_program_functions_with_atomic_names_are_called_as_declared(tmp_path):
    code = _rust("""
        shader UserAtomic {
            uint atomicAdd(uint a, uint b) { return a - b; }
            uint atomic_add(uint a, uint b) { return a * b; }
            compute {
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                buffer uint* slots;
                void main() {
                    uint i = gl_GlobalInvocationID.x;
                    slots[i] = atomicAdd(slots[i], 1u) + atomic_add(slots[i], 3u);
                }
            }
        }
        """)
    assert "pub fn atomicAdd(a: u32, b: u32) -> u32 {" in code, code
    assert (
        "slots[i as usize] = (atomicAdd(slots[i as usize], 1) + atomic_add(slots[i as usize], 3));"
        in code
    ), code
    assert_generated_rust_smoke_compiles(code, tmp_path)
