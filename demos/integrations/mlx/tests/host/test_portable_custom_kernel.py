"""Custom kernels of the portable host: read mx.fast.metal_kernel's signature
back from its source, check every slot buffer and the exact grid, translate one
package per source and launch (dispatch regions on Direct3D and OpenGL),
dispatch through the reflected bindings and audit the guards and packages."""

import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from crosstl.translator.dispatch_regions import plan_dispatch_regions
from demos.integrations.mlx.portable_host import (
    custom_kernel_dispatch,
    custom_kernel_evidence,
    custom_kernel_layout,
    custom_kernel_packages,
    element_storage,
    prepare,
    runtime,
    upstream_module_groups,
    verify_upstream_ops,
)

layout = custom_kernel_layout
HOST = Path(layout.__file__).parent
GUARDS = element_storage.GUARDS
GUARD_COUNT = element_storage.GUARD_COUNT
# dispatch.h's CROSTL_MLX_BUFFER_INOUT.
INOUT = 2

BASIC = "custom_kernel_basic_floatc_float"
BASIC_SOURCE = """[[kernel]] void custom_kernel_basic_floatc_float(
  const constant float* a [[buffer(0)]],
  device float* out1 [[buffer(1)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]]) {

                uint elem = thread_position_in_grid.x;
                out1[elem] = a[elem];

}
"""
STRIDED = "custom_kernel_myexpFalse__float_float_float"
STRIDED_SOURCE = """template <typename T>
[[kernel]] void custom_kernel_myexpFalse__float_float_float(
  const device float* inp [[buffer(0)]],
  const constant int* inp_shape [[buffer(1)]],
  const constant int64_t* inp_strides [[buffer(2)]],
  const constant int& inp_ndim [[buffer(3)]],
  device float* out [[buffer(4)]],
  uint3 thread_position_in_grid [[thread_position_in_grid]],
  uint threads_per_simdgroup [[threads_per_simdgroup]]) {

                uint elem = thread_position_in_grid.x;
                uint loc = elem_to_loc(elem, inp_shape, inp_strides, inp_ndim);
                T tmp = inp[loc];
                out[elem] = metal::precise::exp(tmp) * threads_per_simdgroup;

}

template [[host_name("custom_kernel_myexpFalse__float_float_float")]] [[kernel]] \
decltype(custom_kernel_myexpFalse__float_float_float<float>) \
custom_kernel_myexpFalse__float_float_float<float>;
"""
HELPER = "custom_kernel_helper_floatc_float"
# write_signature appends the declaration to the kernel's header as is.
HELPER_SOURCE = """
            template <typename T>
            T do_exp(T x) {
                return metal::precise::exp(x);
            }
            [[kernel]] void custom_kernel_helper_floatc_float(
  const constant float* a [[buffer(0)]],
  device atomic<float>* out1 [[buffer(1)]]) {
out1[0] = do_exp(a[0]);
}
"""


def test_signatures_follow_write_signature():
    basic = layout.parse(BASIC, BASIC_SOURCE)
    assert [item.to_json() for item in basic.parameters] == [
        ["a", "float32", "input", 0],
        ["out1", "float32", "output", 1],
    ]
    assert basic.attributes == ("thread_position_in_grid",) and not basic.template
    strided = layout.parse(STRIDED, STRIDED_SOURCE)
    assert [item.to_json() for item in strided.parameters] == [
        ["inp", "float32", "input", 0],
        ["inp_shape", "int32", "shape", 1],
        ["inp_strides", "int64", "strides", 2],
        ["inp_ndim", "int32", "ndim", 3],
        ["out", "float32", "output", 4],
    ]
    assert {item.owner for item in strided.parameters[1:4]} == {"inp"}
    assert strided.template
    assert strided.attributes == ("thread_position_in_grid", "threads_per_simdgroup")
    helper = layout.parse(HELPER, HELPER_SOURCE)
    assert helper.outputs[0].atomic and helper.attributes == ()
    assert [item.buffer for item in helper.parameters] == ["buffer0", "buffer1"]


@pytest.mark.parametrize(
    ("entry", "source", "message"),
    (
        ("compiled_x", BASIC_SOURCE, "entry or source"),
        (BASIC, BASIC_SOURCE.replace("buffer(1)", "buffer(2)"), "consecutive"),
        (BASIC, BASIC_SOURCE + BASIC_SOURCE, "exactly its entry"),
        (
            BASIC,
            BASIC_SOURCE.replace("  device float* out1", "  float* out1"),
            "no output",
        ),
        (
            BASIC,
            BASIC_SOURCE.replace("[[thread_position_in_grid]]", "[[thread_index]]"),
            "write_signature",
        ),
        (
            BASIC,
            BASIC_SOURCE.replace(
                "uint3 thread_position_in_grid", "uint thread_position_in_grid"
            ),
            "write_signature",
        ),
        (STRIDED, STRIDED_SOURCE.split("\ntemplate [[host_name")[0], "instantiated"),
        (BASIC, "template <typename T>\n" + BASIC_SOURCE, "instantiated"),
        (
            BASIC,
            BASIC_SOURCE.replace("device float* out1", "device float* a"),
            "unique",
        ),
    ),
)
def test_signatures_refuse_other_sources(entry, source, message):
    with pytest.raises(ValueError, match=message):
        layout.parse(entry, source)


def buffers(signature, values, directions=None, metadata=None):
    """Native slot buffers for ``signature``: ``values`` by parameter name."""
    memory, entries = {}, []
    for item in signature.parameters:
        words = values[item.name]
        ctype = {
            "int32": ctypes.c_int32,
            "int64": ctypes.c_int64,
        }.get(item.dtype, element_storage.CTYPES[item.dtype])
        memory[item.name] = (ctype * len(words))(*words)
        direction = (directions or {}).get(item.name, int(item.kind == "output"))
        entries.append(
            runtime.Buffer(
                item.buffer.encode(),
                item.dtype.encode(),
                ctypes.addressof(memory[item.name]),
                len(words),
                direction,
            )
        )
    return entries, memory


def supplied(entries, *, source, mode=0):
    """The callback's buffer array with the host's source and options."""
    text = source.encode()
    option = (ctypes.c_int32 * 1)(mode)
    keep = [text, option]
    every = [
        *entries,
        runtime.Buffer(
            b"kernel_source", b"utf8", ctypes.cast(text, ctypes.c_void_p), len(text), 0
        ),
        runtime.Buffer(b"compile_options", b"int32", ctypes.addressof(option), 1, 0),
    ]
    return (runtime.Buffer * len(every))(*every), keep


STRIDED_VALUES = {
    "inp": [0x3F800000] * 28,
    "inp_shape": [12, 6],
    "inp_strides": [2, 1],
    "inp_ndim": [2],
    "out": [0] * 72,
}


def strided_launch(grid=72, group=32):
    return runtime.Launch(
        (ctypes.c_uint32 * 3)(-(-grid // group), 1, 1),
        (ctypes.c_uint32 * 3)(group, 1, 1),
        (ctypes.c_uint32 * 3)(grid, 1, 1),
    )


def test_validation_checks_slots_metadata_and_the_exact_grid():
    signature = layout.parse(STRIDED, STRIDED_SOURCE)
    entries, _ = buffers(signature, STRIDED_VALUES)
    named = {entry.name.decode(): entry for entry in entries}
    execution = strided_launch().execution()
    metadata = layout.validate(STRIDED, signature, named, 72, execution, 2)
    assert metadata == {
        "mathMode": "fast",
        "threadGridSize": [72, 1, 1],
        "workgroupSize": [32, 1, 1],
        "template": True,
        "attributes": ["thread_position_in_grid", "threads_per_simdgroup"],
        "initialized": [],
    }
    with pytest.raises(ValueError, match="exact grid"):
        layout.validate(STRIDED, signature, named, 64, execution, 0)
    with pytest.raises(ValueError, match="exact grid"):
        layout.validate(
            STRIDED, signature, named, 72, {**execution, "threadGridSize": []}, 0
        )
    with pytest.raises(ValueError, match="slots"):
        layout.validate(
            STRIDED, signature, {**named, "extra": named["buffer0"]}, 72, execution, 0
        )
    for name, values, message in (
        ("inp", [0x3F800000] * 27, "extent differs"),
        ("inp_strides", [-2, 1], "non-negative"),
        ("inp_shape", [12, 6, 1], "rank"),
        ("inp_ndim", [3], "rank"),
        ("inp_shape", [0, 6], "positive"),
    ):
        changed, _ = buffers(signature, {**STRIDED_VALUES, name: values})
        changed = {entry.name.decode(): entry for entry in changed}
        with pytest.raises(ValueError, match=message):
            layout.validate(STRIDED, signature, changed, 72, execution, 0)
    for directions, message in (({"out": 0}, "buffer out"), ({"inp": 1}, "buffer inp")):
        changed, _ = buffers(signature, STRIDED_VALUES, directions)
        changed = {entry.name.decode(): entry for entry in changed}
        with pytest.raises(ValueError, match=message):
            layout.validate(STRIDED, signature, changed, 72, execution, 0)


def test_launch_geometry_carries_partial_edge_groups():
    assert strided_launch(72, 32).execution() == {
        "workgroupCount": [3, 1, 1],
        "workgroupSize": [32, 1, 1],
        "threadGridSize": [72, 1, 1],
    }
    regions = plan_dispatch_regions((72, 1, 1), (32, 1, 1))
    assert [region.workgroup_size for region in regions] == [(32, 1, 1), (8, 1, 1)]


def test_packages_translate_each_region_with_its_options(tmp_path):
    region = plan_dispatch_regions((6, 1, 1), (4, 1, 1))[1]
    identity = {
        "target": "directx",
        "entry": BASIC,
        "workgroupSize": [2, 1, 1],
        "mathMode": "fast",
    }
    config = custom_kernel_packages.project_config(
        tmp_path, "work", identity, region, True
    )
    assert config.defines == {"__FAST_MATH__": "1"}
    options = config.source_options["metal"]["target_options"]["directx"]
    assert options == {
        "dispatch_region": region.to_json(),
        "software_subgroup_width": 32,
        "relative_wave_shuffle_out_of_range": "self",
    }
    assert config.source_options["metal"]["preserve_resource_origins"]
    safe = custom_kernel_packages.project_config(
        tmp_path,
        "work",
        {**identity, "target": "opengl", "mathMode": "safe"},
        region,
        False,
    )
    assert safe.defines == {}
    assert safe.source_options["metal"]["target_options"] == {
        "opengl": {"dispatch_region": region.to_json()}
    }
    metal = custom_kernel_packages.target_options("metal", None, True)
    assert metal == {}
    absent = {
        "diagnostics": [
            {
                "severity": "error",
                "code": "project.translate.opengl-software-subgroup-invalid",
            }
        ]
    }
    assert custom_kernel_packages._subgroups_absent(absent, "opengl")
    assert not custom_kernel_packages._subgroups_absent(absent, "directx")
    assert not custom_kernel_packages._subgroups_absent(
        {
            "diagnostics": (
                absent["diagnostics"] * 2 + [{"severity": "error", "code": "x"}]
            )
        },
        "opengl",
    )


def test_cached_sources_keep_their_hash_and_mlx_headers(tmp_path):
    source = BASIC_SOURCE
    digest = custom_kernel_packages._sha256(source.encode())
    (tmp_path / custom_kernel_packages.SOURCE_FILE).write_text(layout.PREAMBLE + source)
    assert (
        custom_kernel_packages.CustomKernelPackageCache.source(tmp_path, digest)
        == source
    )
    (tmp_path / custom_kernel_packages.SOURCE_FILE).write_text(source)
    with pytest.raises(ValueError, match="changed after translation"):
        custom_kernel_packages.CustomKernelPackageCache.source(tmp_path, digest)


class FakeCache:
    """Packages that bind every slot with its origin, by target."""

    def __init__(self, target, *, drop=(), read_only=()):
        self.target, self.drop, self.read_only = target, set(drop), set(read_only)
        self.requests = []

    def get(self, entry, source, *, math_mode, thread_grid_size, workgroup_size):
        self.requests.append((entry, math_mode, tuple(thread_grid_size)))
        signature = layout.parse(entry, source)
        bindings = []
        for item in signature.parameters:
            if item.name in self.drop:
                continue
            value = element_storage.packet(item.dtype, self.target, [])
            bindings.append(
                {
                    "name": item.name + "Binding",
                    "access": (
                        "read_write"
                        if item.kind == "output" and item.name not in self.read_only
                        else "read"
                    ),
                    "provenance": {"sourceResource": {"parameter": item.name}},
                    "scalarLayout": {
                        "elementType": value["dtype"],
                        "elementStrideBytes": element_storage.PHYSICAL_SIZES[
                            value["dtype"]
                        ],
                    },
                }
            )
        regions = (
            [None]
            if self.target == "metal"
            else plan_dispatch_regions(thread_grid_size, workgroup_size)
        )
        return tuple(
            (
                {
                    "bindings": bindings,
                    "artifact": {"id": entry, "region": index},
                    "provenance": (
                        {} if region is None else {"dispatchRegion": region.to_json()}
                    ),
                    "entryPoint": {"name": "main"},
                },
                Path(f"/packages/{index}"),
            )
            for index, region in enumerate(regions)
        )


def run(monkeypatch, tmp_path, target, *, cache=None, fault=None, directions=None):
    signature = layout.parse(STRIDED, STRIDED_SOURCE)
    values = dict(STRIDED_VALUES)
    if directions:
        values["out"] = [0x40000000] * 72
    entries, memory = buffers(signature, values, directions)
    array, keep = supplied(entries, source=STRIDED_SOURCE)
    host = SimpleNamespace(
        custom_kernels=cache or FakeCache(target),
        target=target,
        trace=tmp_path / "dispatch.jsonl",
        trace_values=False,
        dispatch_count=0,
    )
    seen = {}

    def result(inputs, expected):
        seen.update(inputs)
        outputs = {}
        for binding, shape in expected.items():
            words = element_storage.unpack("float32", target, inputs[binding]["values"])
            words[:72] = [0x42000000 + index for index in range(72)]
            if fault == "guard":
                words[-1] ^= 1
            outputs[binding] = {
                **shape,
                "values": element_storage.packet("float32", target, words)["values"],
            }
        details = {"request": {}}
        if target != "metal":
            details["regions"] = []
        return SimpleNamespace(status="ok", outputs=outputs, details=details)

    monkeypatch.setattr(
        custom_kernel_dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **_: (
            inputs,
            outputs,
        ),
    )
    monkeypatch.setattr(
        custom_kernel_dispatch, "execute", lambda _, request: result(*request)
    )
    regions = []
    monkeypatch.setattr(
        custom_kernel_dispatch.region_dispatch,
        "run_regions",
        lambda host, packages, inputs, outputs, execution: (
            regions.append(len(packages)) or result(inputs, outputs)
        ),
    )
    custom_kernel_dispatch.dispatch(
        host, STRIDED, array, len(array), 72, strided_launch()
    )
    del keep
    event = json.loads(host.trace.read_text().splitlines()[-1])
    return memory, seen, event, host, regions


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_dispatch_binds_guards_and_writes_back_each_output(
    monkeypatch, tmp_path, target
):
    memory, seen, event, host, regions = run(monkeypatch, tmp_path, target)
    assert host.dispatch_count == 1
    assert host.custom_kernels.requests == [(STRIDED, 0, (72, 1, 1))]
    assert regions == ([] if target == "metal" else [2])
    # A fresh output starts as guards, and its guard tail is checked.
    out = element_storage.unpack("float32", target, seen["outBinding"]["values"])
    assert out == [GUARDS["float32"]] * (72 + GUARD_COUNT)
    assert list(memory["out"]) == [0x42000000 + index for index in range(72)]
    assert list(memory["inp_strides"]) == [2, 1]
    assert event["customKernel"]["threadGridSize"] == [72, 1, 1]
    assert event["boundParameters"] == sorted(STRIDED_VALUES)
    assert set(event["outputHashes"]) == {"out"}


def test_dispatch_uploads_an_initialized_output(monkeypatch, tmp_path):
    # The host fills an output with the kernel's initial value and binds it
    # as an initialized input/output, which uploads its values.
    _, seen, event, _, _ = run(
        monkeypatch, tmp_path, "metal", directions={"out": INOUT}
    )
    out = element_storage.unpack("float32", "metal", seen["outBinding"]["values"])
    assert out[:72] == [0x40000000] * 72
    assert out[72:] == [GUARDS["float32"]] * GUARD_COUNT
    assert event["customKernel"]["initialized"] == ["out"]


@pytest.mark.parametrize(
    ("cache", "fault", "error", "message"),
    (
        (FakeCache("metal", drop=("out",)), None, ValueError, "every output"),
        (FakeCache("metal", read_only=("out",)), None, ValueError, "not writable"),
        (None, "guard", RuntimeError, "guard"),
    ),
)
def test_dispatch_refuses_unbound_outputs_and_changed_guards(
    monkeypatch, tmp_path, cache, fault, error, message
):
    with pytest.raises(error, match=message):
        run(monkeypatch, tmp_path, "metal", cache=cache, fault=fault)
    assert not (tmp_path / "dispatch.jsonl").exists()


def test_dispatch_accepts_a_package_that_drops_an_unread_input(monkeypatch, tmp_path):
    _, seen, event, _, _ = run(
        monkeypatch, tmp_path, "directx", cache=FakeCache("directx", drop=("inp_ndim",))
    )
    assert "inp_ndimBinding" not in seen and "inp_ndim" not in event["boundParameters"]


def test_audit_checks_every_region_and_its_module(monkeypatch, tmp_path):
    modules = tmp_path / "modules"
    modules.mkdir()
    module = modules / "region.dxil"
    module.write_bytes(b"dxil")
    digest = custom_kernel_evidence.hashlib.sha256(b"dxil").hexdigest()
    cache = FakeCache("directx")
    packages = cache.get(
        STRIDED,
        STRIDED_SOURCE,
        math_mode=0,
        thread_grid_size=(72, 1, 1),
        workgroup_size=(32, 1, 1),
    )
    regions = [
        {
            "artifact": descriptor["artifact"],
            "provenance": descriptor["provenance"],
            "packageRoot": str(directory),
            "workgroupCount": descriptor["provenance"]["dispatchRegion"][
                "workgroupCount"
            ],
            "workgroupSize": descriptor["provenance"]["dispatchRegion"][
                "workgroupSize"
            ],
            "moduleFile": str(module),
            "moduleHash": digest,
        }
        for descriptor, directory in packages
    ]
    event = {"details": {"regions": regions}}
    custom_kernel_evidence.audit_regions(event, packages, modules)
    with pytest.raises(ValueError, match="coverage"):
        custom_kernel_evidence.audit_regions(event, packages[:1], modules)
    module.write_bytes(b"other")
    with pytest.raises(ValueError, match="identity"):
        custom_kernel_evidence.audit_regions(event, packages, modules)


def test_math_modes_come_from_the_compile_options_buffer():
    for value, name in layout.MATH_MODES.items():
        option = (ctypes.c_int32 * 1)(value)
        buffer = runtime.Buffer(
            b"compile_options", b"int32", ctypes.addressof(option), 1, 0
        )
        assert layout.math_mode({"compile_options": buffer}) == value
        assert custom_kernel_evidence.MODES[name] == value
    option = (ctypes.c_int32 * 1)(3)
    buffer = runtime.Buffer(
        b"compile_options", b"int32", ctypes.addressof(option), 1, 0
    )
    with pytest.raises(ValueError, match="unknown"):
        layout.math_mode({"compile_options": buffer})
    with pytest.raises(ValueError, match="compile options"):
        layout.math_mode({})


def test_host_port_follows_custom_kernel_cpp():
    port = (HOST / "custom_kernel.inc").read_text(encoding="utf-8")
    for text in (
        "void CustomKernel::eval_gpu(",
        "ensure_row_contiguous_ && !in.flags().row_contiguous",
        "contiguous_copy(in)",
        "std::min(tx, gx)",
        "launch.thread_grid_size[axis] = extent[axis];",
        "init_value_ ? CROSTL_MLX_BUFFER_INOUT : 1",
        '"kernel_source", "utf8"',
        '"compile_options", "int32"',
        "is_precompiled_ || shared_memory_ != 0 || !scalar_arguments_.empty()",
    ):
        assert text in port, text
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    assert backend.rstrip().endswith(
        '#include "mlx/backend/no_gpu/crosstl_custom_kernel.inc"'
    )
    assert "CustomKernel" in prepare.FAST_PRIMITIVES


def test_preparation_replaces_metal_kernel_and_the_stub(monkeypatch):
    replaced = []

    def record(text, before, after):
        replaced.append(before)
        return text

    monkeypatch.setattr(prepare, "replace_once", record)
    sources = prepare.adapted_sources("", "", "", "")
    assert "NO_GPU_MULTI(CustomKernel)" in replaced
    assert prepare.METAL_KERNEL_STREAM in replaced
    assert (
        sources["crosstl_custom_kernel.inc"]
        == (HOST / "custom_kernel.inc").read_bytes()
    )
    assert "crosstl_metal_kernel.cpp" in sources
    assert "crosstl_metal_kernel.cpp" in sources["CMakeLists.txt"].decode()


def test_runtime_routes_custom_kernels_and_groups_list_them():
    source = (HOST / "runtime.py").read_text(encoding="utf-8")
    assert "custom_kernel_layout.is_entry(entry)" in source
    tests = upstream_module_groups.CUSTOM_KERNEL_TESTS
    assert len(tests) == len(set(tests)) == 13
    assert verify_upstream_ops.GROUPS["custom-kernels"] == tests
    assert set(tests) == (
        set(upstream_module_groups.METAL_SOURCE_TESTS)
        | (set(tests) & verify_upstream_ops.METAL_PATH_TESTS)
    )
    assert len(set(tests) & verify_upstream_ops.METAL_PATH_TESTS) == 5
    cpu = set(upstream_module_groups.METAL_SOURCE_CPU_TESTS)
    assert len(cpu) == 2 and cpu <= set(upstream_module_groups.METAL_SOURCE_TESTS)
    # The CPU worker skips every test but the two that check MLX's refusal.
    assert set(tests) - cpu <= verify_upstream_ops.GPU_ONLY_TESTS
    assert not cpu & verify_upstream_ops.GPU_ONLY_TESTS


def test_metal_source_reports_metal_only_for_its_native_tests():
    class Metal:
        @staticmethod
        def is_available():
            return False

    mx = SimpleNamespace(metal=Metal(), gpu="gpu", is_available=lambda device: True)
    name = upstream_module_groups.METAL_SOURCE_TESTS[0]
    with verify_upstream_ops.metal_source(mx, name, native=True):
        assert mx.metal.is_available()
    assert not mx.metal.is_available()
    with verify_upstream_ops.metal_source(mx, name, native=False):
        assert not mx.metal.is_available()
    other = upstream_module_groups.CUSTOM_KERNEL_TESTS[3]
    assert other not in upstream_module_groups.METAL_SOURCE_TESTS
    with verify_upstream_ops.metal_source(mx, other, native=True):
        assert not mx.metal.is_available()
    mx.is_available = lambda device: False
    with pytest.raises(RuntimeError, match="native host"):
        with verify_upstream_ops.metal_source(mx, name, native=True):
            pass
