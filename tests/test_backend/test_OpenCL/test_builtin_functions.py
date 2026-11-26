"""OpenCL C built-in functions imported as the CrossGL built-ins that compute the same values.

crosstl/backend/OpenCL/builtin_functions.py: integer and reduced-accuracy math functions, ``as_<type>``
reinterpretations, scalar relational functions, ``bitselect``, the ``cl_khr_extended_bit_ops`` bitfield functions and
the ``cl_khr_subgroup*`` functions, and the ``atom_`` spellings of the OpenCL C 1.0 atomics extensions.
"""

import pytest

import crosstl

from .test_work_item_queries import to_crossgl

KERNEL = """
__kernel __attribute__((reqd_work_group_size(1, 1, 1)))
void k(__global int* iin, __global float* fin, __global int* o, __global float* fo, __global uint* uo) {{
    uint i = get_global_id(0);
    {body}
}}
"""


def stored(body):
    """The CrossGL statement the kernel writes for its last OpenCL C statement ``body``."""
    crossgl = to_crossgl(KERNEL.format(body=body))
    lines = [line.strip() for line in crossgl.splitlines() if line.strip()]
    return lines[lines.index("var i: u32 = gl_GlobalInvocationID.x;") + 1]


@pytest.mark.parametrize(
    "body, expected",
    [
        pytest.param(
            "o[i] = mul_hi(iin[i], iin[i + 1u]);",
            "o[i] = mulhi(iin[i], iin[(i + 1u)]);",
            id="mul_hi",
        ),
        pytest.param(
            "uo[i] = abs_diff(iin[i], iin[i + 1u]);",
            "uo[i] = absdiff(iin[i], iin[(i + 1u)]);",
            id="abs_diff",
        ),
        pytest.param(
            # 6.15.2: the native_ and half_ functions have implementation-defined accuracy; the full-accuracy
            # function is one such implementation.
            "fo[i] = native_sin(fin[i]) + half_rsqrt(fin[i]) + native_recip(fin[i]);",
            "fo[i] = ((sin(fin[i]) + inversesqrt(fin[i])) + rcp(fin[i]));",
            id="native-and-half",
        ),
        pytest.param(
            "fo[i] = native_divide(fin[i], fin[i + 1u]);",
            "fo[i] = ((fin[i]) / (fin[(i + 1u)]));",
            id="native_divide",
        ),
        pytest.param(
            "fo[i] = fast_length(fin[i]);", "fo[i] = length(fin[i]);", id="fast_length"
        ),
        pytest.param(
            "fo[i] = as_float(iin[i]);",
            "fo[i] = intBitsToFloat(iin[i]);",
            id="as_float",
        ),
        pytest.param(
            "o[i] = as_int(fin[i]);", "o[i] = floatBitsToInt(fin[i]);", id="as_int"
        ),
        pytest.param(
            # Integers of one width keep their bits through a conversion.
            "uo[i] = as_uint(iin[i]);",
            "uo[i] = u32(iin[i]);",
            id="as_uint-of-int",
        ),
        pytest.param(
            "o[i] = isequal(fin[i], fin[i + 1u]);",
            "o[i] = i32(fin[i] == fin[(i + 1u)]);",
            id="isequal",
        ),
        pytest.param(
            # isnotequal is true for unordered operands, as != is.
            "o[i] = isnotequal(fin[i], fin[i + 1u]);",
            "o[i] = i32(fin[i] != fin[(i + 1u)]);",
            id="isnotequal",
        ),
        pytest.param(
            "o[i] = islessgreater(fin[i], fin[i + 1u]);",
            "o[i] = i32((fin[i] < fin[(i + 1u)]) || (fin[i] > fin[(i + 1u)]));",
            id="islessgreater",
        ),
        pytest.param(
            "o[i] = bitselect(iin[i], iin[i + 1u], iin[i + 2u]);",
            "o[i] = ((iin[i] & ~iin[(i + 2u)]) | (iin[(i + 1u)] & iin[(i + 2u)]));",
            id="bitselect-int",
        ),
        pytest.param(
            "fo[i] = bitselect(fin[i], fin[i + 1u], fin[i + 2u]);",
            "fo[i] = uintBitsToFloat((floatBitsToUint(fin[i]) & ~floatBitsToUint(fin[(i + 2u)]))"
            " | (floatBitsToUint(fin[(i + 1u)]) & floatBitsToUint(fin[(i + 2u)])));",
            id="bitselect-float",
        ),
        pytest.param(
            "uo[i] = bitfield_extract_unsigned(iin[i], 2u, 3u);",
            "uo[i] = bitfieldExtract(u32(iin[i]), i32(2u), i32(3u));",
            id="bitfield_extract_unsigned",
        ),
        pytest.param(
            "o[i] = sub_group_reduce_add(iin[i]) + sub_group_all(iin[i]);",
            "o[i] = (subgroupAdd(iin[i]) + i32(subgroupAll((iin[i]) != 0)));",
            id="sub-group-reduce-and-predicate",
        ),
        pytest.param(
            "uo[i] = get_sub_group_local_id();",
            "uo[i] = gl_SubgroupInvocationID;",
            id="get_sub_group_local_id",
        ),
        pytest.param(
            # 6.15.17: the last sub-group of a work-group holds fewer work-items when the size is not a multiple.
            "uo[i] = get_sub_group_size();",
            "uo[i] = min(gl_SubgroupSize, gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z"
            " - gl_SubgroupID * gl_SubgroupSize);",
            id="get_sub_group_size",
        ),
        pytest.param(
            # OpenCL C 1.1 made the cl_khr_*_atomics atom_ functions core as atomic_.
            "atom_add(&o[0], 1);",
            "atomicAdd(o[0], 1);",
            id="atom_add",
        ),
        pytest.param(
            "fo[i] = work_group_reduce_max(fin[i]) + work_group_scan_exclusive_add(fin[i]);",
            "fo[i] = (workgroupMax(fin[i]) + workgroupExclusiveAdd(fin[i]));",
            id="work-group-reduce-and-scan",
        ),
        pytest.param(
            # cl_khr_work_group_uniform_arithmetic: the bitwise operators combine the integers themselves.
            "o[i] = work_group_scan_inclusive_xor(iin[i]);",
            "o[i] = workgroupInclusiveXor(iin[i]);",
            id="work-group-bitwise-scan",
        ),
        pytest.param(
            "o[i] = work_group_all(iin[i]) + work_group_reduce_logical_or(iin[i]);",
            "o[i] = (i32(workgroupAll((iin[i]) != 0)) + i32(workgroupOr((iin[i]) != 0)));",
            id="work-group-predicates",
        ),
        pytest.param(
            "fo[i] = work_group_broadcast(fin[i], 3);",
            "fo[i] = workgroupBroadcast(fin[i], u32(3));",
            id="work-group-broadcast-1d",
        ),
        pytest.param(
            # 6.15.1: the local linear id is (z * size_y + y) * size_x + x, CrossGL's local invocation index.
            "fo[i] = work_group_broadcast(fin[i], 1u, 2u, 0u);",
            "fo[i] = workgroupBroadcast(fin[i], u32(1u) + u32(2u) * gl_WorkGroupSize.x"
            " + u32(0u) * gl_WorkGroupSize.x * gl_WorkGroupSize.y);",
            id="work-group-broadcast-3d",
        ),
    ],
)
def test_builtin_calls_import_as_crossgl_builtins(body, expected):
    assert stored(body) == expected


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            # A second read of an operand with a side effect would repeat it.
            "o[i] = islessgreater(fin[i++], fin[i]);",
            id="islessgreater-side-effect",
        ),
        pytest.param(
            # DXC declares WaveActiveBitAnd for unsigned integers only, and the HLSL output does not convert.
            "o[i] = sub_group_non_uniform_reduce_and(iin[i]);",
            id="signed-bitwise-reduction",
        ),
    ],
)
def test_calls_without_a_faithful_form_stay_refused(body, tmp_path):
    path = tmp_path / "kernel.cl"
    path.write_text(KERNEL.format(body=body))
    with pytest.raises(Exception, match="is not declared"):
        crosstl.translate(str(path), backend="directx")


@pytest.mark.parametrize("backend", ["directx", "opengl", "metal"])
def test_imported_builtins_translate_to_each_target(backend, tmp_path):
    path = tmp_path / "kernel.cl"
    path.write_text(KERNEL.format(body="""
    fo[i] = as_float(iin[i]) + native_cos(fin[i]) + half_exp2(fin[i]);
    o[i] = isgreater(fin[i], fin[i + 1u]) + bitselect(iin[i], iin[i + 1u], iin[i + 2u]);
    """))
    code = crosstl.translate(str(path), backend=backend)
    assert "native_cos" not in code
    assert "isgreater" not in code
    assert "bitselect" not in code


@pytest.mark.parametrize("backend", ["directx", "opengl", "metal"])
def test_work_group_collectives_lower_for_each_target(backend, tmp_path):
    # Targets without work-group collectives combine the values through workgroup memory
    # (crosstl/translator/semantics/workgroup_collectives.py), which the fixed work-group size allows.
    path = tmp_path / "kernel.cl"
    path.write_text(KERNEL.format(body="""
    fo[i] = work_group_reduce_add(fin[i]) + work_group_broadcast(fin[i], 0);
    o[i] = work_group_any(iin[i]) + work_group_scan_inclusive_min(iin[i]);
    """))
    code = crosstl.translate(str(path), backend=backend)
    assert "work_group_" not in code
    assert "workgroupAdd(" not in code
