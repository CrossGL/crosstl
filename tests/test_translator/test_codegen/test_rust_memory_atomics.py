"""CrossGL atomics on buffer and workgroup memory, lowered to the Rust runtime's ``&mut`` place helpers.

The Rust generator wrote ``atomicAdd(counter.total, 1u)`` as it stood, which calls a function no Rust module
defines. Each CrossGL atomic now lowers to the ``gpu`` runtime function that takes the location as ``&mut`` and
returns its previous value; forms with no such spelling are refused.
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
    struct Counters {
        uint total;
        int signedTotal;
        uint slots[4];
    };
    buffer Counters counters;
    compute {
        void main() {
            uint previous = atomicAdd(counters.total, 1u);
            int low = atomicMin(counters.signedTotal, -3);
            int high = atomicMax(counters.signedTotal, 9);
            uint masked = atomicAnd(counters.slots[0], 3u);
            uint joined = atomicOr(counters.slots[1], 4u);
            uint flipped = atomicXor(counters.slots[2], 8u);
            uint swapped = atomicCompSwap(counters.slots[3], previous, 7u);
            uint replaced = atomicExchange(counters.total, 5u);
            counters.slots[0] = previous + masked + joined + flipped + swapped + replaced + uint(low + high);
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


def test_buffer_atomics_borrow_the_location_and_return_its_previous_value():
    code = _rust(BUFFER_ATOMICS)
    for line in (
        "let previous: u32 = atomic_add(&mut (*COUNTERS).total, 1);",
        "let low: i32 = atomic_min(&mut (*COUNTERS).signedTotal, (-3));",
        "let high: i32 = atomic_max(&mut (*COUNTERS).signedTotal, 9);",
        "let masked: u32 = atomic_and(&mut (*COUNTERS).slots[0], 3);",
        "let joined: u32 = atomic_or(&mut (*COUNTERS).slots[1], 4);",
        "let flipped: u32 = atomic_xor(&mut (*COUNTERS).slots[2], 8);",
        "let swapped: u32 = atomic_comp_swap(&mut (*COUNTERS).slots[3], previous, 7);",
        "let replaced: u32 = atomic_exchange(&mut (*COUNTERS).total, 5);",
    ):
        assert line in code, code
    assert "atomicAdd" not in code
    assert "atomicCompSwap" not in code


def test_hlsl_style_atomics_are_canonicalized_before_rust_generation(tmp_path):
    code = _translate(
        tmp_path,
        """
        shader OutArgument {
            struct Counters { uint total; uint slot; };
            buffer Counters counters;
            compute {
                void main() {
                    uint original;
                    atomicAdd(counters.total, 2u, original);
                    uint exchanged;
                    atomicCompSwap(counters.slot, 1u, original, exchanged);
                    counters.total = original + exchanged;
                }
            }
        }
        """,
    )
    assert "original = atomic_add(&mut (*COUNTERS).total, 2);" in code, code
    assert (
        "exchanged = atomic_comp_swap(&mut (*COUNTERS).slot, 1, original);" in code
    ), code


def test_atomic_on_a_local_makes_its_binding_mutable_and_compiles(tmp_path):
    code = _rust("""
        shader LocalAtomics {
            compute {
                uint main(uint seed) {
                    uint tally = seed;
                    uint slots[2];
                    slots[0] = 0u;
                    slots[1] = 0u;
                    uint before = atomicAdd(tally, 3u);
                    atomicMax(slots[1], before);
                    uint swapped = atomicCompSwap(tally, before + 3u, 11u);
                    return swapped + atomicExchange(slots[0], tally) + slots[1];
                }
            }
        }
        """)
    assert "let mut tally: u32 = seed;" in code, code
    assert "atomic_add(&mut tally, 3)" in code, code
    assert "atomic_max(&mut slots[1 as usize], before)" in code or (
        "atomic_max(&mut slots[1], before)" in code
    ), code
    assert_generated_rust_smoke_compiles(code, tmp_path)


def test_a_program_function_of_the_runtime_name_gets_the_module_path():
    code = _rust("""
        shader Shadowed {
            uint atomic_add(uint a, uint b) { return a * b; }
            compute {
                uint main(uint seed) {
                    uint tally = seed;
                    return atomicAdd(tally, 2u) + atomic_add(seed, 3u);
                }
            }
        }
        """)
    assert "gpu::atomic_add(&mut tally, 2)" in code, code
    assert "atomic_add(seed, 3)" in code, code


@pytest.mark.parametrize(
    "call, reason",
    [
        ("atomicAdd(seed + 1u, 2u)", "first argument is not a memory location"),
        ("atomicAdd(counters.total)", "with 1 arguments"),
        ("atomicCompSwap(counters.total, 1u)", "with 2 arguments"),
    ],
)
def test_atomic_forms_with_no_rust_spelling_are_refused(call, reason):
    with pytest.raises(RustLoweringRefused, match=reason):
        _rust(f"""
            shader Refused {{
                struct Counters {{ uint total; }};
                buffer Counters counters;
                compute {{
                    uint main(uint seed) {{
                        return {call};
                    }}
                }}
            }}
            """)


def test_a_program_atomic_function_is_called_as_declared():
    code = _rust("""
        shader UserAtomic {
            uint atomicAdd(uint a, uint b) { return a - b; }
            compute {
                uint main(uint seed) {
                    return atomicAdd(seed, 1u);
                }
            }
        }
        """)
    assert "atomic_add(&mut" not in code, code
