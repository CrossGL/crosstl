"""CrossGL atomics in the rust-gpu output.

The Rust generator writes each CrossGL atomic on a storage buffer or workgroup memory as the spirv_std::arch function
of the same operation, with the Device or Workgroup scope and relaxed semantics as constant generic arguments. An
atomic on memory no other invocation shares, a write to a read-only buffer and a type SPIR-V has no atomic for are
refused by name. A program function that shares an atomic's name is still called as declared.
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
        buffer float* sums;
        void main() {
            uint previous = atomicAdd(slots[0], 1u);
            int low = atomicMin(signedSlots[0], -3);
            int high = atomicMax(signedSlots[1], 9);
            uint masked = atomicAnd(slots[1], 3u);
            uint joined = atomicOr(slots[2], 4u);
            uint flipped = atomicXor(slots[3], 8u);
            uint swapped = atomicCompSwap(slots[4], previous, 7u);
            uint replaced = atomicExchange(slots[5], 5u);
            float total = atomicAdd(sums[0], 1.5);
            float smallest = atomicMin(sums[1], total);
            uint seen = atomicLoad(slots[7]);
            atomicStore(slots[8], seen);
            uint loaded = atomic_load(slots[9]);
            atomic_store(slots[10], loaded);
            slots[6] = previous + masked + joined + flipped + swapped + replaced + uint(low + high) + uint(smallest);
        }
    }
}
"""

_DEVICE = "{ spirv_std::memory::Scope::Device as u32 }, { spirv_std::memory::Semantics::NONE.bits() }"


def _rust(source):
    return generate_code(parse_code(tokenize_code(source)))


def _translate(tmp_path, source, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="rust", format_output=False)


def test_buffer_atomics_are_spirv_std_atomics(tmp_path):
    code = _rust(BUFFER_ATOMICS)
    for call in (
        f"let previous: u32 = unsafe {{ spirv_std::arch::atomic_i_add::<_, {_DEVICE}>(&mut slots[0], 1) }};",
        f"spirv_std::arch::atomic_s_min::<_, {_DEVICE}>(&mut signedSlots[0], (-3))",
        f"spirv_std::arch::atomic_s_max::<_, {_DEVICE}>(&mut signedSlots[1], 9)",
        f"spirv_std::arch::atomic_and::<_, {_DEVICE}>(&mut slots[1], 3)",
        f"spirv_std::arch::atomic_or::<_, {_DEVICE}>(&mut slots[2], 4)",
        f"spirv_std::arch::atomic_xor::<_, {_DEVICE}>(&mut slots[3], 8)",
        f"spirv_std::arch::atomic_exchange::<_, {_DEVICE}>(&mut slots[5], 5)",
        f"spirv_std::arch::atomic_f_add::<_, {_DEVICE}>(&mut sums[0], 1.5)",
        f"spirv_std::arch::atomic_f_min::<_, {_DEVICE}>(&mut sums[1], total)",
        f"spirv_std::arch::atomic_load::<_, {_DEVICE}>(&slots[7])",
        f"spirv_std::arch::atomic_store::<_, {_DEVICE}>(&mut slots[8], seen)",
        f"spirv_std::arch::atomic_load::<_, {_DEVICE}>(&slots[9])",
        f"spirv_std::arch::atomic_store::<_, {_DEVICE}>(&mut slots[10], loaded)",
    ):
        assert call in code, (call, code)
    # spirv_std takes the value to store before the comparator, CrossGL's atomicCompSwap after it.
    assert (
        "spirv_std::arch::atomic_compare_exchange::<_, { spirv_std::memory::Scope::Device as u32 }, "
        "{ spirv_std::memory::Semantics::NONE.bits() }, { spirv_std::memory::Semantics::NONE.bits() }>"
        "(&mut slots[4], 7, previous)" in code
    ), code
    assert_generated_rust_smoke_compiles(code, tmp_path)


def test_workgroup_atomics_use_the_workgroup_scope(tmp_path):
    code = _rust("""
        shader SharedCounter {
            compute {
                layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
                buffer uint* outp;
                shared uint counts[4];
                void main() {
                    uint i = gl_LocalInvocationID.x;
                    uint before = atomicAdd(counts[i % 4u], 1u);
                    barrier();
                    outp[gl_GlobalInvocationID.x] = before + counts[i % 4u];
                }
            }
        }
        """)
    assert (
        "spirv_std::arch::atomic_i_add::<_, { spirv_std::memory::Scope::Workgroup as u32 }, "
        "{ spirv_std::memory::Semantics::NONE.bits() }>(&mut counts[(i % 4) as usize], 1)"
        in code
    ), code
    assert_generated_rust_smoke_compiles(code, tmp_path)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (
            "uint tally = 2u; slots[0] = atomicAdd(tally, 3u);",
            "atomic: atomicAdd's location is not in a storage buffer or workgroup memory, so no other invocation "
            "shares it",
        ),
        (
            "atomicAdd(frozen[0], 1u);",
            "atomic: atomicAdd writes to the read-only storage buffer 'frozen'",
        ),
        (
            "atomicAnd(sums[0], 1.0);",
            "atomic: atomicAnd on a f32 location has no spirv_std atomic",
        ),
    ],
)
def test_atomics_without_shared_memory_or_a_spirv_form_are_refused(body, reason):
    source = f"""
        shader Refused {{
            compute {{
                layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
                buffer uint* slots;
                buffer float* sums;
                readonly buffer uint* frozen;
                void main() {{
                    {body}
                }}
            }}
        }}
        """
    with pytest.raises(RustLoweringRefused) as refused:
        _rust(source)
    assert refused.value.code == "project.translate.rust-lowering-unsupported"
    assert reason in refused.value.reasons, refused.value.reasons


def test_hlsl_style_atomics_are_canonicalized_before_rust_generation(tmp_path):
    # HLSL's out-argument forms reach the Rust generator as the CrossGL atomics that return the previous value.
    code = _translate(
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
    assert (
        f"spirv_std::arch::atomic_i_add::<_, {_DEVICE}>(&mut slots[0], 2)" in code
    ), code
    assert "atomic_compare_exchange::" in code, code
    assert "(&mut slots[1], original, 1)" in code, code
    assert_generated_rust_smoke_compiles(code, tmp_path)


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
