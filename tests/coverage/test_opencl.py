import pytest

from tools.coverage import issues, probes, report, runner, validators
from tools.coverage.features import opencl
from tools.coverage.model import VOID, Feature, Param, Result, Signature, Type

HEADER = """
#define __ovld __attribute__((overloadable))
// OpenCL v1.1 s6.11.1, v1.2 s6.12.1, v2.0 s6.13.1 - Work-item Functions
size_t __ovld __cnfn get_global_id(uint dimindx);

// OpenCL v1.1 s6.11.2, v1.2 s6.12.2, v2.0 s6.13.2 - Math functions
char __ovld __cnfn clz(char);
int __ovld __cnfn clz(int);
float2 __ovld __cnfn sin(float2);
float __ovld __cnfn sin(float);
#if defined(__opencl_c_generic_address_space)
float __ovld sincos(float x, float *cosval);
#endif
float __ovld sincos(float x, __global float *cosval);
/**
 * A comment that mentions float __ovld fake(float); is not a declaration.
 */

// OpenCL v1.1/1.2/2.0 s6.2.3 - Explicit conversions
int __ovld __cnfn convert_int_rte(float);
char __ovld __cnfn convert_char(float);
uchar4 __ovld __cnfn convert_uchar4_sat(int4);
int4 __ovld __cnfn convert_int4_rte(float4);

// OpenCL v1.1 s6.11.7, v1.2 s6.12.7, v2.0 s6.13.7 - Vector Data Load and Store Functions
float2 __ovld __purefn vload2(size_t, const float *);
float4 __ovld __purefn vload4(size_t, const float *);
void __ovld vstore4(float4 data, size_t offset, float *p);

// OpenCL v1.1 s6.11.8, v1.2 s6.12.8, v2.0 s6.13.8 - Synchronization Functions
void __ovld __conv barrier(cl_mem_fence_flags);

// OpenCL v1.1 s6.11.1, v1.2 s6.12.11 - Atomic Functions
int __ovld atomic_add(volatile __local int *, int);
int __ovld atomic_add(volatile __global int *, int);

// OpenCL v2.0 s6.13.11 - Atomics Functions
#if defined(__opencl_c_generic_address_space)
float __ovld atomic_fetch_add(volatile atomic_float *, float);
int __ovld atomic_fetch_add(volatile atomic_int *, int);
#endif
int __ovld atomic_fetch_add(volatile __global atomic_int *, int);
bool __ovld atomic_flag_test_and_set(volatile __global atomic_flag *);

// OpenCL v1.1 s6.11.3, v1.2 s6.12.14, v2.0 s6.13.14 - Image Read and Write Functions
float4 __ovld __purefn read_imagef(read_only image1d_t, sampler_t, float);
float4 __ovld __purefn read_imagef(read_only image2d_t, sampler_t, int2);
float4 __ovld __purefn read_imagef(read_only image2d_t, sampler_t, float2);
void __ovld write_imagef(write_only image2d_t, int2, float4);
int __ovld __cnfn get_image_depth(read_only image3d_t);

// OpenCL v1.1 s6.11.10, v1.2 s6.12.10, v2.0 s6.13.10 - Async Copies from Global to Local Memory, Local to Global Memory, and Prefetch
event_t __ovld async_work_group_copy(__local char *, const __global char *, size_t, event_t);
event_t __ovld async_work_group_copy(__local float *, const __global float *, size_t, event_t);
event_t __ovld async_work_group_copy(__local float2 *, const __global float2 *, size_t, event_t);
void __ovld wait_group_events(int, event_t *);

// OpenCL v2.0 s6.13.9 - Address Space Qualifier Functions
cl_mem_fence_flags __ovld get_fence(const void *ptr);

// OpenCL v2.0 s6.13.16 - Pipe Functions
bool __ovld is_valid_reserve_id(reserve_id_t reserve_id);

// OpenCL v2.0 s6.13.17 - Enqueue Kernels
ndrange_t __ovld ndrange_2D(const size_t[2]);
int __ovld enqueue_marker(queue_t, uint, const clk_event_t*, clk_event_t*);
void __ovld capture_event_profiling_info(clk_event_t, clk_profiling_info, __global void*);

#ifdef __OPENCL_CPP_VERSION__
float __ovld cpp_only(float);
#else
float __ovld c_only(float);
#endif

// OpenCL Extension v2.0 s9.17 - Sub-groups
#if defined(cl_khr_subgroup_ballot)
uint4 __ovld sub_group_ballot(int predicate);
#endif
#if defined(cl_intel_subgroups)
float __ovld __conv intel_sub_group_shuffle(float, uint);
#endif
"""


def by_name(features):
    return {feature.name: feature for feature in features}


def test_opencl_lists_each_name_once_in_its_section_or_extension():
    features = by_name(opencl.parse(HEADER))
    assert features["sin"].group == "math"
    assert features["get_global_id"].group == "work-item"
    assert features["convert_int_rte"].group == "explicit-conversions"
    assert features["sub_group_ballot"].group == "cl_khr_subgroup_ballot"
    assert "fake" not in features
    assert all(
        f.language == "opencl" and f.category == "function" for f in features.values()
    )


def test_opencl_leaves_out_vendor_extensions_and_cpp_only_declarations():
    features = by_name(opencl.parse(HEADER))
    assert "intel_sub_group_shuffle" not in features
    assert "cpp_only" not in features
    assert "c_only" in features


def test_opencl_folds_vector_widths_the_specification_writes_as_n():
    features = by_name(opencl.parse(HEADER))
    # A conversion's widths join the scalar conversion.
    assert "convert_int4_rte" not in features
    assert "convert_intn_rte" not in features
    # A load or store with no scalar form is one feature, probed at width 4.
    assert "vload2" not in features and "vload4" not in features
    assert features["vloadn"].spelling == "vload4"
    assert features["vloadn"].signature.result == Type("float", (4,))
    assert features["vstoren"].spelling == "vstore4"


def test_opencl_probes_the_plainest_overload():
    features = by_name(opencl.parse(HEADER))
    assert features["clz"].signature == Signature(Type("int"), (Param(Type("int")),))
    assert features["sin"].signature == Signature(
        Type("float"), (Param(Type("float")),)
    )
    # An out pointer takes a private local's address, so the generic overload is chosen; an atomic takes a global
    # buffer element.
    assert features["sincos"].signature.params[1] == Param(Type("float"), "out")
    assert features["vloadn"].signature.params == (
        Param(Type("uint64")),
        Param(Type("float"), "address"),
    )
    assert features["atomic_add"].signature.params[0] == Param(Type("int"), "inout")


def test_opencl_passes_dimensions_and_fence_flags_as_constants():
    features = by_name(opencl.parse(HEADER))
    assert features["get_global_id"].signature.params == (Param(Type("uint"), "const"),)
    flags = features["barrier"].signature.params[0]
    assert flags == Param(Type("uint", opaque="cl_mem_fence_flags"), "const")
    assert features["barrier"].signature.result == VOID


def test_opencl_parameter_names_are_not_types():
    assert opencl.parameter("float x", False, False)[0] == Param(Type("float"))
    assert opencl.parameter("unsigned int", False, False)[0] == Param(Type("uint"))
    assert opencl.parameter("float *cosval", False, False) == (
        Param(Type("float"), "out"),
        "",
    )
    # A global or local pointer outside an atomic stays opaque, for the dialect to pass (OpenCL.handle).
    assert opencl.parameter("__global float *cosval", False, False) == (
        Param(Type.other("__global float *")),
        "__global",
    )
    assert opencl.parameter("int8", False, False)[0] == Param(Type.other("int8"))


def _probe(name):
    return probes.build(by_name(opencl.parse(HEADER))[name], "opencl").source


def test_opencl_probe_is_a_kernel_over_global_buffers():
    source = _probe("sincos")
    assert "__kernel void probe(__global float* fin, __global float* outp) {" in source
    assert "    uint i = get_global_id(0);" in source
    assert "float r = sincos(a0, &a1);" in source


def test_opencl_probe_spells_casts_vector_literals_and_constants():
    source = _probe("vloadn")
    assert "ulong a0 = (ulong)uin[i + 0u];" in source
    assert "float4 r = vload4(a0, &a1);" in source
    assert "barrier(CLK_LOCAL_MEM_FENCE);" in _probe("barrier")
    assert "get_global_id(1u)" in _probe("get_global_id")
    assert "atomic_add(&outp[i], a1);" in _probe("atomic_add")
    vector = Feature(
        "opencl",
        "function",
        "f",
        Signature(Type("float", (3,)), (Param(Type("int", (3,))),)),
    )
    built = probes.build(vector, "opencl").source
    assert "int3 a0 = (int3)(iin[i + 0u], iin[i + 1u], iin[i + 2u]);" in built
    half = Feature(
        "opencl", "function", "h", Signature(Type("half"), (Param(Type("half")),))
    )
    assert probes.build(half, "opencl").source.startswith(
        "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
    )


def test_opencl_probes_8_bit_scalars_through_32_bit_buffers():
    features = by_name(opencl.parse(HEADER))
    assert features["convert_char"].signature.result == Type("int8")
    assert opencl.opencl_type("uchar4") == Type("uint8", (4,))
    source = _probe("convert_char")
    assert "char r = convert_char(a0);" in source
    assert "outp[i] = (int)r;" in source


def test_opencl_probes_c11_atomics_on_an_atomic_int_buffer():
    features = by_name(opencl.parse(HEADER))
    # The core atomic_int overload, not the cl_ext_float_atomics one, and its global form.
    assert features["atomic_fetch_add"].signature.params == (
        Param(Type("int", opaque="atomic_int"), "inout"),
        Param(Type("int")),
    )
    source = _probe("atomic_fetch_add")
    assert "__global atomic_int* outp" in source
    assert "atomic_fetch_add(&outp[i], a1);" in source
    assert "__global atomic_flag* outp" in _probe("atomic_flag_test_and_set")


def test_opencl_passes_2d_images_as_kernel_arguments():
    features = by_name(opencl.parse(HEADER))
    params = features["read_imagef"].signature.params
    assert [p.type for p in params] == [
        Type.other("read_only image2d_t"),
        Type.other("sampler_t"),
        Type("float", (2,)),
    ]
    source = _probe("read_imagef")
    kernel = source.index("__kernel void probe(")
    assert source.index("__constant sampler_t u1 = ") < kernel
    assert ", read_only image2d_t u0) {" in source
    assert "float4 r = read_imagef(u0, u1, a2);" in source
    assert ", write_only image2d_t u0) {" in _probe("write_imagef")
    # Images no other function takes are kernel arguments of the probe's own.
    assert ", read_only image3d_t a0) {" in _probe("get_image_depth")


def test_opencl_makes_events_and_queues_with_the_built_ins_that_return_them():
    source = _probe("enqueue_marker")
    assert "queue_t a0 = get_default_queue();" in source
    assert "clk_event_t a2 = create_user_event();" in source
    assert "int r = enqueue_marker(a0, a1, &a2, &a3);" in source
    ranged = _probe("ndrange_2D")
    assert "size_t a0[2] = {1, 1};" in ranged
    assert "ndrange_t r = ndrange_2D(a0);" in ranged
    assert "event_t a1 = 0;" in _probe("wait_group_events")
    profiled = _probe("capture_event_profiling_info")
    assert (
        "capture_event_profiling_info(a0, CLK_PROFILING_COMMAND_EXEC_TIME, (__global void*)outp);"
        in profiled
    )


def test_opencl_passes_local_and_global_pointers_by_their_address_space():
    feature = by_name(opencl.parse(HEADER))["async_work_group_copy"]
    # The pointers are opaque, so the type they point at ranks the overload.
    assert feature.signature.params[0] == Param(Type.other("__local float *"))
    source = _probe("async_work_group_copy")
    kernel = source.index("__kernel void probe(")
    assert source.index("__local float a0[1];") > kernel
    assert (
        "event_t r = async_work_group_copy(a0, (const __global float *)outp, a2, a3);"
        in source
    )
    assert "uint r = (uint)get_fence(outp);" in _probe("get_fence")
    constant = Feature(
        "opencl",
        "function",
        "f",
        Signature(VOID, (Param(Type.other("__constant float *")),)),
    )
    with pytest.raises(probes.ProbeError, match="__constant float"):
        probes.build(constant, "opencl")


def test_opencl_reserves_pipe_packets_from_a_pipe_kernel_argument():
    source = _probe("is_valid_reserve_id")
    assert ", read_only pipe int a0_pipe) {" in source
    assert "reserve_id_t a0 = reserve_read_pipe(a0_pipe, 1u);" in source


def test_opencl_declares_workgroup_variables_inside_the_kernel():
    shared = Feature(
        "opencl",
        "function",
        "f",
        Signature(Type("float"), (Param(Type("float"), "workgroup"),)),
    )
    source = probes.build(shared, "opencl").source
    kernel = source.index("__kernel void probe(")
    assert source.index("__local float a0;") > kernel


@pytest.mark.skipif(
    not validators.available("opencl"), reason="clang or DXC is missing"
)
def test_opencl_probes_are_valid_opencl_c():
    for name in (
        "sincos",
        "vloadn",
        "barrier",
        "get_global_id",
        "atomic_add",
        "clz",
        "convert_char",
        "atomic_fetch_add",
        "atomic_flag_test_and_set",
        "read_imagef",
        "write_imagef",
        "get_image_depth",
        "async_work_group_copy",
        "wait_group_events",
        "get_fence",
        "is_valid_reserve_id",
        "ndrange_2D",
        "enqueue_marker",
        "capture_event_profiling_info",
    ):
        verdict = validators.validate("opencl", _probe(name))
        assert verdict.accepted, (name, verdict.log)


def test_opencl_input_is_emitted_as_hlsl_and_checked_by_dxc(monkeypatch):
    feature = by_name(opencl.parse(HEADER))["sin"]
    calls = []

    def translate(source, suffix, backend, source_backend):
        calls.append((suffix, backend, source_backend))
        return "hlsl text"

    def validated(language, source, stage, entry="", target=""):
        calls.append((language, source))
        return validators.Verdict(True)

    monkeypatch.setattr(validators, "available", lambda language, target="": True)
    monkeypatch.setattr(runner, "translate", translate)
    monkeypatch.setattr(runner, "_validated", validated)
    result = runner.run_input(feature, "opencl")
    assert result.outcome == "supported"
    # The native probe, then one translation from OpenCL C to HLSL, then DXC.
    assert calls[1] == (".cl", "directx", "opencl")
    assert calls[2] == ("hlsl", "hlsl text")


def test_opencl_has_no_output_row_and_its_issues_name_the_hlsl_check():
    feature = by_name(opencl.parse(HEADER))["sin"]
    data = report.build(
        [Result(feature, "input", "wrong", "error: x", "clang")],
        ["opencl"],
        ["output", "input"],
    )
    rows = {row["direction"]: row for row in data["backends"]}
    assert rows["output"]["offered"] is False
    assert "opencl" not in runner.OUTPUT_BACKENDS
    body = issues.tracking_body(rows["input"], "CrossGL/crosstl", {})
    assert "translates it to HLSL that DXC accepts" in body
