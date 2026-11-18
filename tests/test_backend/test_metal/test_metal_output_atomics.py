"""Metal output lowering for the five atomics of CrossGL/crosstl #2337.

CrossGL has no subtraction operation, and the coverage probes call the five
Metal atomics (``atomic_fetch_sub``, ``atomic_load``, ``atomic_store`` and the
64-bit ``atomic_max``/``atomic_min``) under their Metal names. The Metal
generator used to emit those calls verbatim onto plain ``device`` pointers,
which the Metal compiler rejects with ``no matching function for call``.

The 32-bit spellings lower to the ``atomic_*_explicit`` fetch forms on a
``reinterpret_cast<device atomic_int*>`` target, as ``atomicAdd`` already did.
The 64-bit min/max are different: MSL 3.0 section 6.15.2.6 provides only the
void ``atomic_max_explicit``/``atomic_min_explicit`` on ``atomic_ulong``, so a
discarded call lowers to those and a call whose value is used is refused.
Metal has no ``atomic_long`` and no 64-bit load, store, fetch or
compare-exchange, so those stay refused too.
"""

import pytest

import crosstl
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.refusals import MetalRefusal

PROBE = """shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer int* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int a1 = iin[i + 0u];
            atomic_fetch_sub(outp[i], a1);
        }
    }
}
"""

BLOCK_PROBE = """shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        layout(std430, binding = 0) buffer OutBlock { uint64_t values[]; } outp;
        layout(std430, binding = 1) buffer InBlock { uint values[]; } inp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            uint64_t a1 = uint64_t(inp.values[i]);
            atomicMax(outp.values[i], a1);
        }
    }
}
"""


def _translate(tmp_path, source):
    path = tmp_path / "probe.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _with_buffers(body):
    return f"""shader Probe {{
    compute {{
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer uint* uin;
        buffer uint64_t* outp;
        buffer int64_t* outi;
        buffer int* outi32;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            {body}
        }}
    }}
}}
"""


def test_atomic_fetch_sub_lowers_to_fetch_sub_explicit(tmp_path):
    metal = _translate(tmp_path, PROBE)

    assert (
        "atomic_fetch_sub_explicit(reinterpret_cast<device atomic_int*"
        ">(&outp[i]), a1, memory_order_relaxed);"
    ) in metal
    assert "atomic_fetch_sub(outp[i], a1);" not in metal


def test_atomic_fetch_sub_result_is_the_previous_value(tmp_path):
    metal = _translate(
        tmp_path,
        _with_buffers(
            "int a1 = int(uin[i]); "
            "int old = atomic_fetch_sub(outi32[i], a1); "
            "outi32[i] = old;"
        ),
    )

    assert (
        "int old = atomic_fetch_sub_explicit("
        "reinterpret_cast<device atomic_int*>(&outi32[i]), a1, "
        "memory_order_relaxed);"
    ) in metal


def test_metal_atomic_load_and_store_spellings_lower(tmp_path):
    metal = _translate(
        tmp_path,
        _with_buffers(
            "int a1 = int(uin[i]); "
            "int loaded = atomic_load(outi32[i]); "
            "atomic_store(outi32[i], a1 + loaded);"
        ),
    )

    assert (
        "int loaded = atomic_load_explicit("
        "reinterpret_cast<device atomic_int*>(&outi32[i]), "
        "memory_order_relaxed);"
    ) in metal
    assert (
        "atomic_store_explicit(reinterpret_cast<device atomic_int*"
        ">(&outi32[i]), a1 + loaded, memory_order_relaxed);"
    ) in metal


def test_atomic_max_and_min_on_uint64_lower_to_void_modify_forms(tmp_path):
    metal = _translate(
        tmp_path,
        _with_buffers(
            "uint64_t a1 = uint64_t(uin[i]); "
            "atomicMax(outp[i], a1); "
            "atomicMin(outp[i], a1);"
        ),
    )

    assert (
        "atomic_max_explicit(reinterpret_cast<device atomic_ulong*"
        ">(&outp[i]), a1, memory_order_relaxed);"
    ) in metal
    assert (
        "atomic_min_explicit(reinterpret_cast<device atomic_ulong*"
        ">(&outp[i]), a1, memory_order_relaxed);"
    ) in metal
    assert "atomic_fetch_max_explicit" not in metal
    assert "atomic_fetch_min_explicit" not in metal
    assert "atomic_uint64_t" not in metal


def test_metal_atomic_fetch_min_and_max_spellings_on_uint64(tmp_path):
    metal = _translate(
        tmp_path,
        _with_buffers(
            "uint64_t a1 = uint64_t(uin[i]); "
            "atomic_fetch_max(outp[i], a1); "
            "atomic_fetch_min(outp[i], a1);"
        ),
    )

    assert (
        "atomic_max_explicit(reinterpret_cast<device atomic_ulong*"
        ">(&outp[i]), a1, memory_order_relaxed);"
    ) in metal
    assert (
        "atomic_min_explicit(reinterpret_cast<device atomic_ulong*"
        ">(&outp[i]), a1, memory_order_relaxed);"
    ) in metal


def test_atomic_max_on_uint64_buffer_block_member(tmp_path):
    metal = _translate(tmp_path, BLOCK_PROBE)

    assert (
        "atomic_max_explicit(reinterpret_cast<device atomic_ulong*"
        ">(outp + (i * 8)), a1, memory_order_relaxed);"
    ) in metal


def test_value_use_of_64_bit_min_max_is_refused(tmp_path):
    with pytest.raises(Exception, match="64-bit atomic min/max return no value"):
        _translate(
            tmp_path,
            _with_buffers(
                "uint64_t a1 = uint64_t(uin[i]); "
                "uint64_t old = atomicMax(outp[i], a1); "
                "outp[i] = old;"
            ),
        )


def test_signed_64_bit_atomic_max_is_refused(tmp_path):
    with pytest.raises(Exception, match="got int64_t"):
        _translate(
            tmp_path,
            _with_buffers("int64_t a1 = int64_t(uin[i]); atomicMax(outi[i], a1);"),
        )


def test_64_bit_atomic_add_and_load_stay_refused(tmp_path):
    with pytest.raises(Exception, match="got uint64_t"):
        _translate(
            tmp_path,
            _with_buffers("uint64_t a1 = uint64_t(uin[i]); atomicAdd(outp[i], a1);"),
        )
    with pytest.raises(Exception, match="got uint64_t"):
        _translate(
            tmp_path,
            _with_buffers("uint64_t v = atomicLoad(outp[i]); outp[i] = v;"),
        )


def test_unmapped_atomic_scalar_type_is_refused_not_passed_through():
    # Regression test for the fail-closed gate: the scalar mapping used to
    # fall back to the input name verbatim (``.get(name, name)``), which
    # would have emitted e.g. ``atomic_long`` — a type Metal does not have.
    # Metal has no atomic_long, so a signed 64-bit scalar (and anything else
    # unmapped) must be refused.
    for scalar in ("long", "int64_t", "double"):
        with pytest.raises(MetalRefusal, match="unsupported Metal atomic scalar type"):
            MetalCodeGen.metal_atomic_scalar_type(scalar)
    assert MetalCodeGen.metal_atomic_scalar_type("uint64_t") == "ulong"
    assert MetalCodeGen.metal_atomic_scalar_type("float", bitwise_float=True) == "uint"
