"""Compiled-kernel parsing, launch contracts, guarded commits and audits."""

import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import compiled_dispatch as dispatch
from demos.integrations.mlx.portable_host import compiled_evidence as evidence
from demos.integrations.mlx.portable_host import compiled_layout as layout
from demos.integrations.mlx.portable_host import compiled_packages as packages
from demos.integrations.mlx.portable_host import element_storage, runtime

HOST = Path(layout.__file__).parent
CONTIGUOUS = "compiled_0123456789abcdef_contiguous"
STRIDED = "compiled_0123456789abcdef_strided_2"
# What compiled.inc's build_kernel port emits for (x + y) * 2 over float32
# vectors, and for where(m, exp(a), a * 0.5) over a transposed float16 matrix.
CONTIGUOUS_SOURCE = layout.PREAMBLE + f"""[[host_name("{CONTIGUOUS}")]]
[[kernel]] void {CONTIGUOUS}(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant const uint& size [[buffer(3)]],
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {{
  constexpr int N_ = 1;
  uint index = N_ * pos.x;
  float tmp_A = A[index];
  float tmp_B = B[index];
  auto tmp_D = static_cast<float>(2);
  float tmp_E = Add()(tmp_A, tmp_B);
  float tmp_C = Multiply()(tmp_E, tmp_D);
  C[index] = tmp_C;
}}
"""
STRIDED_SOURCE = layout.PREAMBLE + f"""[[host_name("{STRIDED}")]]
[[kernel]] void {STRIDED}(
    device const float16_t* A [[buffer(0)]],
    device const bool* B [[buffer(1)]],
    constant const int64_t* in_strides [[buffer(2)]],
    device float16_t* C [[buffer(3)]],
    constant const int* output_shape [[buffer(4)]],
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {{
  constexpr int N_ = 1;
  uint index = pos.x + grid.x * (pos.y + uint(grid.y) * pos.z);
  auto tmp_D = static_cast<float16_t>(0.5);
  uint index_A = elem_to_loc_2<uint>({{pos.x, pos.y}}, in_strides + 0);
  uint index_B = elem_to_loc_2<uint>({{pos.x, pos.y}}, in_strides + 2);
  float16_t tmp_A = A[index_A];
  bool tmp_B = B[index_B];
  float16_t tmp_E = Exp()(tmp_A);
  float16_t tmp_F = Multiply()(tmp_A, tmp_D);
  float16_t tmp_C = Select()(tmp_B, tmp_E, tmp_F);
  C[index] = tmp_C;
}}
"""
CTYPES = element_storage.CTYPES


def test_parse_reads_exactly_build_kernel_parameters():
    assert layout.parse(CONTIGUOUS, CONTIGUOUS_SOURCE) == (
        ("A", "float32", 0),
        ("B", "float32", 0),
        ("C", "float32", 1),
        ("size", "uint32", 0),
    )
    assert layout.parse(STRIDED, STRIDED_SOURCE) == (
        ("A", "float16", 0),
        ("B", "bool_", 0),
        ("in_strides", "int64", 0),
        ("C", "float16", 1),
        ("output_shape", "int32", 0),
    )
    assert layout.strided_inputs(STRIDED_SOURCE) == ("A", "B")
    assert layout.work_per_thread(STRIDED_SOURCE) == 1
    assert [
        layout.variant(f"compiled_{'0' * 16}_{kind}")
        for kind in (
            "contiguous",
            "contiguous_n",
            "strided_1",
            "strided_7",
            "strided_dynamic",
        )
    ] == ["contiguous", "contiguous_n", "strided_1", "strided_7", "strided_dynamic"]


def test_parse_binds_input_strides_only_for_non_scalar_inputs():
    # mx.compile of broadcast_to(a, (1,)) + itself: one scalar input, read as
    # A[0], so build_kernel emits no in_strides parameter.
    entry = "compiled_0123456789abcdef_strided_1"
    source = layout.PREAMBLE + f"""[[host_name("{entry}")]]
[[kernel]] void {entry}(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant const int* output_shape [[buffer(2)]],
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {{
  constexpr int N_ = 1;
  uint index = pos.x + grid.x * (pos.y + uint(grid.y) * pos.z);
  float tmp_A = A[0];
  float tmp_B = Add()(tmp_A, tmp_A);
  B[index] = tmp_B;
}}
"""
    assert layout.parse(entry, source) == (
        ("A", "float32", 0),
        ("B", "float32", 1),
        ("output_shape", "int32", 0),
    )
    assert layout.strided_inputs(source) == ()
    # A strided read still needs its strides.
    unbound = STRIDED_SOURCE.replace(
        "    constant const int64_t* in_strides [[buffer(2)]],\n", ""
    )
    with pytest.raises(ValueError, match="input strides"):
        layout.parse(STRIDED, unbound)


@pytest.mark.parametrize(
    "entry",
    (
        "compiled_0123456789abcdef_contiguous_large",
        "compiled_0123456789abcdef_strided_8",
        "compiled_0123456789ABCDEF_contiguous",
        "compiled_contiguous",
        "v_Absfloat32float32",
        None,
    ),
)
def test_variant_refuses_other_names(entry):
    with pytest.raises(ValueError, match="compiled entry"):
        layout.variant(entry)


@pytest.mark.parametrize(
    "edit",
    (
        lambda s: s.replace('#include "mlx/backend/metal/kernels/utils.h"\n', ""),
        lambda s: s.replace(f'host_name("{CONTIGUOUS}")', 'host_name("other")'),
        lambda s: s.replace("[[buffer(1)]]", "[[buffer(4)]]"),
        lambda s: s.replace("    constant const uint& size [[buffer(3)]],\n", ""),
        lambda s: s.replace("device float* C", "device const float* C"),
        lambda s: s.replace("uint3 grid [[threads_per_grid]]", "uint3 grid"),
        lambda s: s + "[[kernel]] void again() {}\n",
    ),
)
def test_parse_refuses_sources_build_kernel_does_not_emit(edit):
    with pytest.raises(ValueError, match="Compiled source"):
        layout.parse(CONTIGUOUS, edit(CONTIGUOUS_SOURCE))


def test_operations_take_the_vector_profiles_of_each_step():
    assert layout.operations(STRIDED_SOURCE) == (
        ("v_", "Exp", "float16"),
        ("vv_", "Multiply", "float16"),
        ("ternary_v", "Select", "float16"),
    )
    selected = dict(packages.profiles(CONTIGUOUS_SOURCE))
    assert selected["binary32_additive_profile"] == "rne-flush"
    assert selected["binary32_multiplication_profile"] == "rne-flush"
    assert selected["binary32_fma_profile"] == "rne-flush"
    assert "binary32_additive_profile" not in dict(packages.profiles(STRIDED_SOURCE))
    cast = CONTIGUOUS_SOURCE.replace(
        "float tmp_C = Multiply()(tmp_E, tmp_D);",
        "float tmp_C = cast_to<float>(tmp_E);",
    )
    assert layout.operations(cast)[-1] == ("cast", "float32", "float32")


@pytest.mark.parametrize(
    ("operation", "dtype", "refused"),
    (
        ("Divide", "int32", True),
        ("Remainder", "uint8", True),
        ("LeftShift", "int64", True),
        ("RightShift", "bool_", True),
        ("Divide", "float32", False),
        ("Remainder", "bfloat16", False),
        ("Add", "int32", False),
    ),
)
def test_integer_divisions_and_shifts_are_refused(operation, dtype, refused):
    steps = (("vv_", operation, dtype),)
    if refused:
        with pytest.raises(ValueError, match="integer divisions"):
            layout.require_defined(steps)
    else:
        assert layout.require_defined(steps) == steps


def native_buffers(parameters, values):
    memory, buffers = {}, []
    for name, dtype, direction in parameters:
        words = values[name]
        ctype = CTYPES.get(dtype, ctypes.c_uint8)
        memory[name] = (ctype * len(words))(*words)
        buffers.append(
            runtime.Buffer(
                name.encode(),
                dtype.encode(),
                ctypes.addressof(memory[name]),
                len(words),
                direction,
            )
        )
    return memory, buffers


def contiguous_buffers(count=5):
    values = {
        "A": [0x3F800000] * count,
        "B": [0x40000000] * count,
        "C": [0] * count,
        "size": [count],
    }
    return native_buffers(layout.parse(CONTIGUOUS, CONTIGUOUS_SOURCE), values)


def strided_buffers():
    # A is a transposed 3 x 4 float16 matrix viewed as 4 x 3; B is a
    # broadcast row of three Booleans.
    values = {
        "A": [0x3C00] * 12,
        "B": [1, 0, 1],
        "in_strides": [1, 4, 0, 1],
        "C": [0] * 12,
        "output_shape": [4, 3],
    }
    parameters = layout.parse(STRIDED, STRIDED_SOURCE)
    memory, buffers = native_buffers(
        [(n, "int64" if n == "in_strides" else d, o) for n, d, o in parameters], values
    )
    return memory, buffers


def execution(groups):
    return {"workgroupCount": list(groups), "workgroupSize": [1, 1, 1]}


def by_name(buffers):
    return {buffer.name.decode(): buffer for buffer in buffers}


def test_validate_accepts_the_launches_eval_gpu_makes():
    _, buffers = contiguous_buffers()
    parameters = layout.parse(CONTIGUOUS, CONTIGUOUS_SOURCE)
    assert layout.validate(
        CONTIGUOUS,
        CONTIGUOUS_SOURCE,
        parameters,
        by_name(buffers),
        5,
        execution((5, 1, 1)),
    ) == {"variant": "contiguous", "workPerThread": 1, "groups": [5, 1, 1], "size": 5}
    _, buffers = strided_buffers()
    parameters = layout.parse(STRIDED, STRIDED_SOURCE)
    assert layout.validate(
        STRIDED, STRIDED_SOURCE, parameters, by_name(buffers), 12, execution((3, 4, 1))
    ) == {
        "variant": "strided_2",
        "workPerThread": 1,
        "groups": [3, 4, 1],
        "shape": [4, 3],
        "strides": {"A": [1, 4], "B": [0, 1]},
    }


def test_work_per_thread_follows_each_variant():
    # Compiled::eval_gpu covers two elements per thread in the strided kernels
    # of more than three dimensions and one in those of at most three.
    _, buffers = strided_buffers()
    parameters = layout.parse(STRIDED, STRIDED_SOURCE)
    arguments = (parameters, by_name(buffers), 12, execution((3, 4, 1)))
    two = STRIDED_SOURCE.replace("constexpr int N_ = 1;", "constexpr int N_ = 2;")
    with pytest.raises(ValueError, match="work per thread"):
        layout.validate(STRIDED, two, *arguments)
    wide = STRIDED.replace("strided_2", "strided_4")
    four = two.replace(STRIDED, wide)
    # The work check passes, and the two-dimensional buffers fail the rank check.
    with pytest.raises(ValueError, match="rank differs"):
        layout.validate(wide, four, *arguments)
    with pytest.raises(ValueError, match="work per thread"):
        layout.validate(wide, four.replace("N_ = 2;", "N_ = 1;"), *arguments)


@pytest.mark.parametrize(
    "fault",
    ("groups", "size", "extent", "stride", "span", "dtype", "direction", "missing"),
)
def test_validate_refuses_launches_that_differ_from_mlx(fault):
    memory, buffers = strided_buffers()
    supplied = by_name(buffers)
    groups = (3, 4, 1)
    if fault == "groups":
        groups = (4, 3, 1)
    elif fault == "size":
        supplied["A"].output = 0
        execution_size = {"workgroupCount": [3, 4, 1], "workgroupSize": [2, 1, 1]}
    elif fault == "extent":
        supplied["C"].count = 11
    elif fault == "stride":
        memory["in_strides"][0] = -1
    elif fault == "span":
        supplied["A"].count = 11
    elif fault == "dtype":
        supplied["B"].dtype = b"uint8"
    elif fault == "direction":
        supplied["C"].output = 0
    elif fault == "missing":
        del supplied["output_shape"]
    launch = execution_size if fault == "size" else execution(groups)
    with pytest.raises(ValueError):
        layout.validate(
            STRIDED,
            STRIDED_SOURCE,
            layout.parse(STRIDED, STRIDED_SOURCE),
            supplied,
            12,
            launch,
        )


def physical_layout(target, dtype):
    value = element_storage.packet(dtype, target, [0])
    return {
        "elementType": value["dtype"],
        "elementStrideBytes": element_storage.PHYSICAL_SIZES[value["dtype"]],
    }


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    "fault",
    (None, "guard", "missing", "shape", "trace", "unbound", "readonly", "source"),
)
def test_dispatch_commits_outputs_only_after_validation(
    tmp_path, monkeypatch, target, fault
):
    memory, buffers = contiguous_buffers()
    source = CONTIGUOUS_SOURCE if fault != "source" else CONTIGUOUS_SOURCE[1:]
    encoded = ctypes.create_string_buffer(source.encode())
    buffers.append(
        runtime.Buffer(
            b"kernel_source", b"utf8", ctypes.addressof(encoded), len(source), 0
        )
    )
    descriptor = {"artifact": {}, "bindings": []}
    for name, dtype, direction in layout.parse(CONTIGUOUS, CONTIGUOUS_SOURCE):
        if fault == "unbound" and name == "B":
            continue
        descriptor["bindings"].append(
            {
                "name": f"binding_{name}",
                "kind": "buffer",
                "access": "read_write" if direction and fault != "readonly" else "read",
                "provenance": {"sourceResource": {"parameter": name}},
                "scalarLayout": physical_layout(target, dtype),
            }
        )
    host = SimpleNamespace(
        target=target,
        compiled=SimpleNamespace(get=lambda entry, text: (descriptor, tmp_path)),
        trace=tmp_path / "trace.jsonl",
        dispatch_count=0,
    )
    requests = []

    def request(descriptor, directory, inputs, outputs, execution, **kwargs):
        requests.append((inputs, kwargs))
        return {name: dict(inputs[name]) for name in outputs}

    monkeypatch.setattr(dispatch, "build_native_loader_dispatch_request", request)

    def execute(host, outputs):
        outputs = json.loads(json.dumps(outputs))
        value = outputs["binding_C"]
        value["values"][:5] = element_storage.packet(
            "float32", target, [0x40C00000] * 5
        )["values"]
        if fault == "guard":
            value["values"][-1] = value["values"][0]
        elif fault == "missing":
            outputs.clear()
        elif fault == "shape":
            value["shape"] = [1]
        return SimpleNamespace(status="ok", outputs=outputs, details={})

    monkeypatch.setattr(dispatch, "execute", execute)
    if fault == "trace":
        host.trace.mkdir()
    supplied = (runtime.Buffer * len(buffers))(*buffers)
    launch = runtime.Launch((5, 1, 1), (1, 1, 1))
    call = lambda: dispatch.dispatch(
        host, CONTIGUOUS, supplied, len(buffers), 5, launch
    )
    if fault:
        with pytest.raises((ValueError, RuntimeError, OSError)):
            call()
        assert host.dispatch_count == 0
        assert not any(memory["C"])
        return
    call()
    assert list(memory["C"]) == [0x40C00000] * 5
    assert host.dispatch_count == 1
    ((inputs, kwargs),) = requests
    assert kwargs["expected_target"] == target
    assert set(kwargs["observe_inputs"]) == {"binding_A", "binding_B", "binding_size"}
    assert len(inputs["binding_C"]["values"]) == 5 + element_storage.GUARD_COUNT
    (event,) = (json.loads(line) for line in host.trace.read_text().splitlines())
    assert event["entry"] == CONTIGUOUS and event["target"] == target
    assert event["compiledMetadata"]["size"] == 5
    assert event["parameters"][-1] == ["size", "uint32", 0]
    assert event["outputValues"] == {"C": [0x40C00000] * 5}


def test_runtime_routes_compiled_entries(monkeypatch):
    host = object.__new__(runtime.HostRuntime)
    host.descriptors, host.scans, host.gathers = {}, None, None
    host.quantization = host.quantized_matmuls = None
    host.fp_quantized_matmuls = host.quantized_qmms = host.fp_quantized_qmms = None
    host.gemvs = host.dots = host.gemms = host.splitk_gemms = None
    host.compiled = None
    for name in (
        "copies",
        "elementwise",
        "aranges",
        "arg_reductions",
        "softmaxes",
        "sorts",
        "searchsorted",
        "norms",
        "wide_reductions",
        "divmods",
    ):
        setattr(host, name, None)
    assert host._entry_available(CONTIGUOUS.encode()) == 0
    host.compiled = object()
    assert host._entry_available(CONTIGUOUS.encode()) == 1
    assert host._entry_available(b"compiled_0123456789abcdef_contiguous_large") == 0
    calls = []
    monkeypatch.setattr(
        runtime.compiled_dispatch, "dispatch", lambda *args: calls.append(args[1])
    )
    host.dispatch(STRIDED, [], 6, 12, launch=None)
    assert calls == [STRIDED]


def test_package_config_scopes_the_generated_source(tmp_path):
    options = packages.profiles(CONTIGUOUS_SOURCE)
    config = packages.project_config(tmp_path, "work", "opengl", CONTIGUOUS, options)
    assert tuple(config.include_patterns) == ("work/kernel.metal",)
    assert config.entry_points == {"work/kernel.metal": (CONTIGUOUS,)}
    assert tuple(config.workgroup_size) == (1, 1, 1)
    metal = config.source_options["metal"]
    assert metal["preserve_resource_origins"] is True
    assert metal["binary32_additive_profile"] == "rne-flush"
    with pytest.raises(ValueError, match="compiled-kernel target"):
        packages.CompiledPackageCache(tmp_path, tmp_path, "vulkan")


def test_audit_finds_the_package_of_the_traced_source(tmp_path, monkeypatch):
    descriptor = {"artifact": {"packagePath": "a"}, "entryPoint": {"name": "CSMain"}}
    found = []

    class Cache:
        target = "directx"

        def find(self, entry, source_hash):
            found.append((entry, source_hash))
            return (descriptor, tmp_path), CONTIGUOUS_SOURCE

    monkeypatch.setattr(evidence, "require_unchanged", lambda *args: None)
    audited = []
    monkeypatch.setattr(evidence, "audit_native_execution", audited.append)
    event = {
        "entry": CONTIGUOUS,
        "target": "directx",
        "sourceHash": "f" * 64,
        "artifact": descriptor["artifact"],
        "dispatchVersion": 3,
        "packageRoot": str(tmp_path),
        "parameters": [
            list(item) for item in layout.parse(CONTIGUOUS, CONTIGUOUS_SOURCE)
        ],
        "compiledMetadata": {"variant": "contiguous", "workPerThread": 1},
        "workgroupCount": [5, 1, 1],
        "workgroupSize": [1, 1, 1],
        "details": {
            "request": {
                "target": "directx",
                "entryPoint": "CSMain",
                "dispatch": {"workgroupCount": [5, 1, 1], "workgroupSize": [1, 1, 1]},
            }
        },
    }
    assert evidence.is_compiled(event) and not evidence.is_compiled({"entry": "v_Abs"})
    evidence.audit_event(event, Cache())
    assert found == [(CONTIGUOUS, "f" * 64)] and audited == [event]
    for key, value in (
        ("packageRoot", str(tmp_path / "other")),
        ("parameters", []),
        ("target", "opengl"),
    ):
        with pytest.raises(ValueError):
            evidence.audit_event({**event, key: value}, Cache())


def test_backend_ports_build_kernel_and_the_variant_selection():
    port = (HOST / "compiled.inc").read_text(encoding="utf-8")
    for text in (
        '"[[host_name(\\"{0}\\")]]\\n[[kernel]] void {0}(\\n", kernel_name);',
        '"    constant const int64_t* in_strides [[buffer({0})]],\\n", cnt++);',
        '"  {0} index = pos.x + grid.x * (pos.y + {0}(grid.y) * pos.z);\\n",',
        '"cast_to<{0}>(tmp_{1});\\n",',
        "compiled_collapse_contiguous_dims(inputs, outputs[0], is_constant_);",
        "compiled_allocate_outputs(inputs, outputs, is_constant_, contiguous);",
        '{"kernel_source", "utf8", const_cast<char*>(source.data()), source.size(), 0}',
        "work_per_thread = dynamic ? 2 : ndim > 3 ? 2 : 1;",
    ):
        assert text in port, text
    assert port.count("namer.get_name(inputs_[i]);") == 1
    for header in packages.HEADERS:
        assert f'"#include \\"{header}\\"\\n"' in port, header
    backend = (HOST / "backend.cpp").read_text(encoding="utf-8")
    tail = backend[backend.rindex("} // namespace mlx::core::fast") :]
    assert '#include "mlx/backend/no_gpu/crosstl_compiled.inc"\n' in tail
    adapter = (HOST / "prepare.py").read_text(encoding="utf-8")
    assert '"crosstl_compiled.inc": (HERE / "compiled.inc").read_bytes(),' in adapter
    assert '"Compiled", "DivMod", "Quantize"' in adapter
