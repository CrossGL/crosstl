"""WGSL and Metal probes."""

from dataclasses import replace

import pytest

from tools.coverage import lowering, probes, semantic, validators
from tools.coverage.features import metal
from tools.coverage.model import (
    STRING,
    Feature,
    Param,
    Signature,
    Type,
)

from .glsl_features import VEC3, feature

WGSL_SIGNATURES = [
    feature("mix", VEC3, VEC3, VEC3, VEC3),
    feature("select", VEC3, VEC3, VEC3, Type("bool")),
    feature("determinant", Type("float"), Type("float", (3, 3))),
    feature("atomicAdd", Type("int"), Param(Type("int"), "inout"), Type("int")),
    feature("workgroupBarrier", Type.other("void")),
    feature(
        "insertBits",
        Type("int", (3,)),
        Type("int", (3,)),
        Type("int", (3,)),
        Type("uint"),
        Type("uint"),
    ),
    feature("atomicLoad", Type("int"), Param(Type("int"), "inout")),
    feature(
        "atomicStore", Type.other("void"), Param(Type("int"), "inout"), Type("int")
    ),
]


@pytest.mark.skipif(not validators.available("wgsl"), reason="naga is not installed")
@pytest.mark.parametrize("probe_feature", WGSL_SIGNATURES, ids=lambda f: f.name)
@pytest.mark.parametrize("baseline", [False, True], ids=["probe", "baseline"])
def test_wgsl_probes_are_valid_wgsl(probe_feature, baseline):
    probe = probes.build(probe_feature, "wgsl", baseline=baseline)
    verdict = validators.validate("wgsl", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log


def test_wgsl_probe_binds_atomics_through_a_pointer():
    source = probes.build(WGSL_SIGNATURES[3], "wgsl").source
    assert "var<storage, read_write> outp: array<atomic<i32>>;" in source
    assert "atomicAdd(&outp[i], a1)" in source
    # The baseline cannot read an atomic element directly, so it leaves the buffer alone.
    assert (
        "outp[i]" not in probes.build(WGSL_SIGNATURES[3], "wgsl", baseline=True).source
    )


def test_wgsl_atomic_probes_drop_a_returned_value_through_a_phony_assignment():
    add, load, store = (
        next(f for f in WGSL_SIGNATURES if f.name == name)
        for name in ("atomicAdd", "atomicLoad", "atomicStore")
    )
    assert "_ = atomicAdd(&outp[i], a1);" in probes.build(add, "wgsl").source
    assert "_ = atomicLoad(&outp[i]);" in probes.build(load, "wgsl").source
    # atomicStore returns nothing, and CrossGL drops a value with a call statement.
    assert "    atomicStore(&outp[i], a1);" in probes.build(store, "wgsl").source
    assert "_ =" not in probes.build(load, "crossgl").source


@pytest.mark.skipif(
    not lowering.can_lower("wgsl"), reason="naga or spirv-dis is missing"
)
def test_wgsl_atomic_load_lowers_to_its_instruction():
    (load,) = (f for f in WGSL_SIGNATURES if f.name == "atomicLoad")
    assert semantic.lower(replace(load, language="wgsl")) == ("OpAtomicLoad", "")


def test_wgsl_has_no_output_parameters():
    frexp = feature("frexp", VEC3, VEC3, Param(Type("int", (3,)), "out"))
    with pytest.raises(probes.ProbeError, match="output parameters"):
        probes.build(frexp, "wgsl")


WORKGROUP_LOAD = Feature(
    "wgsl",
    "function",
    "workgroupUniformLoad",
    Signature(Type("float"), (Param(Type("float"), "workgroup"),)),
    crossgl="workgroupUniformLoad",
)


def test_workgroup_parameter_takes_a_shared_variable_the_probe_fills():
    source = probes.build(WORKGROUP_LOAD, "wgsl").source
    assert "var<workgroup> a0: f32;" in source
    assert "    a0 = fin[i + 0u];" in source
    assert "    var r: f32 = workgroupUniformLoad(&a0);" in source
    crossgl = probes.build(WORKGROUP_LOAD, "crossgl").source
    assert "        shared float a0;" in crossgl
    assert "            float r = workgroupUniformLoad(a0);" in crossgl


@pytest.mark.skipif(not validators.available("wgsl"), reason="naga is not installed")
def test_workgroup_parameter_probe_is_valid_wgsl():
    probe = probes.build(WORKGROUP_LOAD, "wgsl")
    verdict = validators.validate("wgsl", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log


METAL_SIGNATURES = [
    feature("fma", VEC3, VEC3, VEC3, VEC3),
    feature("select", VEC3, VEC3, VEC3, Type("bool", (3,))),
    feature("frexp", VEC3, VEC3, Param(Type("int", (3,)), "out")),
    feature("simd_sum", Type("float"), Type("float")),
    feature(
        "atomic_fetch_add_explicit",
        Type("int"),
        Param(Type("int"), "inout"),
        Type("int"),
        Param(Type.other("memory_order"), "const"),
    ),
    feature(
        "atomic_max", Type.other("void"), Param(Type("uint64"), "inout"), Type("uint64")
    ),
    feature(
        "threadgroup_barrier",
        Type.other("void"),
        Param(Type.other("mem_flags"), "const"),
    ),
    feature(
        "atomic_compare_exchange_weak",
        Type("bool"),
        Param(Type("int"), "inout"),
        Param(Type("int"), "address"),
        Type("int"),
    ),
    feature("simd_ballot", Type.other("simd_vote"), Type("bool")),
    feature("quad_active_threads_mask", Type.other("quad_vote")),
    feature("pack_half_to_unorm4x8", Type("uint"), Type("half", (4,))),
    feature("unpack_unorm565_to_half", Type("half", (3,)), Type("uint")),
]


@pytest.mark.skipif(
    not validators.available("metal"), reason="the Metal toolchain is not installed"
)
@pytest.mark.parametrize("probe_feature", METAL_SIGNATURES, ids=lambda f: f.name)
def test_metal_probes_are_valid_metal(probe_feature):
    probe = probes.build(probe_feature, "metal")
    verdict = validators.validate("metal", probe.source, entry=probe.entry)
    assert verdict.accepted, verdict.log


def test_cuda_hip_and_crossgl_pass_the_address_of_a_local_for_a_pointer_to_values():
    norm = Feature(
        "cuda",
        "function",
        "normf",
        Signature(Type("float"), (Param(Type("int")), Param(Type("float"), "address"))),
        crossgl="norm",
    )
    for language in ("cuda", "hip"):
        source = probes.build(replace(norm, language=language), language).source
        assert "    float a1 = fin[i + 0u];" in source
        assert "    float r = normf(a0, &a1);" in source
    source = probes.build(norm, "crossgl").source
    assert "            float a1 = fin[i + 0u];" in source
    assert "            float r = norm(a0, &a1);" in source
    # A language without pointers to locals says so rather than spelling one.
    with pytest.raises(probes.ProbeError, match="no probe passes glsl a pointer"):
        probes.build(replace(norm, language="glsl"), "glsl")


def test_hip_and_crossgl_pass_the_empty_string_for_a_c_string():
    nan = Feature(
        "hip",
        "function",
        "nanf",
        Signature(Type("float"), (Param(STRING, "const"),)),
        crossgl="nan",
    )
    assert '    float r = nanf("");' in probes.build(nan, "hip").source
    assert '            float r = nan("");' in probes.build(nan, "crossgl").source
    # Languages without strings say so rather than spelling one.
    with pytest.raises(probes.ProbeError, match="parameters of type string"):
        probes.build(replace(nan, language="glsl"), "glsl")


def test_metal_compare_exchange_takes_the_address_of_the_expected_value():
    exchange = next(
        f for f in METAL_SIGNATURES if f.name == "atomic_compare_exchange_weak"
    )
    source = probes.build(exchange, "metal").source
    assert "    int a1 = iin[i + 0u];" in source
    assert "atomic_compare_exchange_weak(&outp[i], &a1, a2);" in source
    with pytest.raises(probes.ProbeError, match="no probe passes hlsl a pointer"):
        probes.build(exchange, "hlsl")


def test_metal_ballots_are_read_as_their_vote_integer():
    ballot = feature("simd_ballot", Type.other("simd_vote"), Type("bool"))
    source = probes.build(ballot, "metal").source
    # The 64-bit mask is stored as a 32-bit buffer element, as every 64-bit result is.
    assert "device uint* outp" in source
    assert "ulong r = static_cast<simd_vote::vote_t>(simd_ballot(a0));" in source
    assert "outp[i] = uint(r);" in source
    mask = feature("quad_active_threads_mask", Type.other("quad_vote"))
    source = probes.build(mask, "metal").source
    assert (
        "uint r = static_cast<quad_vote::vote_t>(quad_active_threads_mask());" in source
    )
    # CrossGL has no mask type to read, so the operation is an IR gap, not an untested probe.
    with pytest.raises(probes.NoSpelling, match="no type for Metal's quad_vote"):
        probes.build(mask, "crossgl")


def test_metal_half_arguments_are_read_from_float_buffers():
    pack, unpack = METAL_SIGNATURES[-2:]
    source = probes.build(pack, "metal").source
    assert "half4 a0 = half4(half(fin[i + 0u]), half(fin[i + 1u])" in source
    assert "uint r = pack_half_to_unorm4x8(a0);" in source
    source = probes.build(unpack, "metal").source
    assert "device float3* outp [[buffer(1)]]" in source
    assert "outp[i] = float3(r);" in source


def test_metal_atomic_probe_passes_the_output_element_and_a_relaxed_order():
    add = feature(
        "atomic_fetch_add_explicit",
        Type("int"),
        Param(Type("int"), "inout"),
        Type("int"),
        Param(Type.other("memory_order"), "const"),
    )
    source = probes.build(add, "metal").source
    assert "device atomic_int* outp [[buffer(1)]]" in source
    assert "atomic_fetch_add_explicit(&outp[i], a1, memory_order_relaxed);" in source


def test_metal_atomic_load_and_store_probes_call_crossgl_load_and_store():
    load, store = metal.parse(
        "metal_atomic atomic_load_explicit int(inout int, memory_order)\n"
        "metal_atomic atomic_store void(inout int, int)\n"
    )
    assert "atomic_load_explicit(&outp[i], memory_order_relaxed);" in (
        probes.build(load, "metal").source
    )
    crossgl = probes.build(load, "crossgl").source
    assert "buffer int* outp;" in crossgl
    assert "atomicLoad(outp[i])" in crossgl
    assert "atomic_store(&outp[i], a1);" in probes.build(store, "metal").source
    assert "atomicStore(outp[i], a1);" in probes.build(store, "crossgl").source


def test_metal_wide_atomic_max_probe_calls_crossgl_atomic_max():
    (wide,) = metal.parse("metal_atomic atomic_min void(inout ulong, ulong)\n")
    assert "atomic_min(&outp[i], a1);" in probes.build(wide, "metal").source
    crossgl = probes.build(wide, "crossgl").source
    assert "buffer uint64_t* outp;" in crossgl
    assert "atomicMin(outp[i], a1);" in crossgl


def test_metal_memory_flag_probes_call_the_crossgl_operations_the_importer_writes():
    sub, fence, barrier = metal.parse(
        "metal_atomic atomic_fetch_sub_explicit int(inout int, int, memory_order)\n"
        "metal_atomic atomic_thread_fence void(mem_flags, memory_order)\n"
        "metal_compute simdgroup_barrier void(mem_flags)\n"
    )
    assert "atomic_fetch_sub_explicit(&outp[i], a1, memory_order_relaxed);" in (
        probes.build(sub, "metal").source
    )
    crossgl = probes.build(sub, "crossgl").source
    assert "buffer int* outp;" in crossgl
    assert "atomicAdd(outp[i], -(a1));" in crossgl
    assert (
        "atomic_thread_fence(mem_flags::mem_none, memory_order_relaxed);"
        in probes.build(fence, "metal").source
    )
    assert (
        "atomicThreadFence(mem_none, memory_order_relaxed, thread_scope_device);"
        in probes.build(fence, "crossgl").source
    )
    assert "simdgroup_barrier(mem_flags::mem_none);" in (
        probes.build(barrier, "metal").source
    )
    assert "subgroupExecutionBarrier();" in probes.build(barrier, "crossgl").source
