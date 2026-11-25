"""Metal output lowering for 64-bit atomic min/max on GLSL buffer blocks.

CrossGL/crosstl #2337: Metal's only 64-bit atomics are the void
``atomic_max_explicit``/``atomic_min_explicit`` on ``atomic_ulong`` (MSL 4.1,
section 6.15.4.7), so a discarded ``atomicMax``/``atomicMin`` on a
``uint64_t`` target lowers to those. The buffer-resource path landed in
#4812; this file covers the GLSL buffer-block member, which lowers by byte
offset from the block's device pointer instead of by reinterpret cast, plus
the refusals: a used result has no lowering, and threadgroup memory has no
64-bit atomic form at all.
"""

import pytest

import crosstl
from crosstl.translator.codegen.metal_refusals import MetalLoweringRefused

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

THREADGROUP_PROBE = """shader Probe {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer uint* uin;
        buffer uint64_t* outp;
        shared uint64_t data[1];
        void main() {
            uint i = gl_LocalInvocationID.x;
            if (i == 0u) { data[0] = uint64_t(0); }
            barrier();
            atomicMax(data[0], uint64_t(uin[i]));
            barrier();
            if (i == 0u) { outp[0] = data[0]; }
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


def test_atomic_max_on_uint64_buffer_block_member(tmp_path):
    metal = _translate(tmp_path, BLOCK_PROBE)

    assert (
        "atomic_max_explicit(reinterpret_cast<device atomic_ulong*"
        ">(outp + (i * 8)), a1, memory_order_relaxed);"
    ) in metal


def test_value_use_of_64_bit_min_max_is_refused(tmp_path):
    with pytest.raises(
        MetalLoweringRefused, match="64-bit atomic min/max return no value"
    ):
        _translate(
            tmp_path,
            _with_buffers(
                "uint64_t a1 = uint64_t(uin[i]); "
                "uint64_t old = atomicMax(outp[i], a1); "
                "outp[i] = old;"
            ),
        )


def test_threadgroup_64_bit_min_max_is_refused(tmp_path):
    # MSL only defines atomic_max_explicit/atomic_min_explicit on
    # atomic_ulong for device memory; a threadgroup target has no form.
    with pytest.raises(MetalLoweringRefused, match="requires device memory"):
        _translate(tmp_path, THREADGROUP_PROBE)


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
