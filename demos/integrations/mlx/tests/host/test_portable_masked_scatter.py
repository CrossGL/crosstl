"""Masked scatters format MLX's generated masked_assign kernel exactly, check
their launches and operands, and recompute every output from the trace."""

import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    element_storage,
    masked_scatter_dispatch,
    masked_scatter_evidence,
    masked_scatter_packages,
    prepare,
    runtime,
)

ROOT = Path(__file__).resolve().parents[5]
HOST = ROOT / "demos/integrations/mlx/portable_host"
TARGETS = ("metal", "directx", "opengl")
# The masked_assign template of the pinned mlx/backend/metal/jit/indexing.h.
INDEXING = """
constexpr std::string_view gather_kernels = R"(
)";

constexpr std::string_view masked_assign_kernel = R"(
template [[host_name("{0}")]] [[kernel]] decltype(masked_assign_impl<{1}, {2}>) masked_assign_impl<{1}, {2}>;
)";

constexpr std::string_view slice_update_op_kernel = R"(
)";
"""


def test_render_formats_the_pinned_template_as_fmt_does():
    source = masked_scatter_packages.render(INDEXING, "masked_assign_complex64_false")
    assert source == (
        '#include "mlx/backend/metal/kernels/utils.h"\n'
        '#include "mlx/backend/metal/kernels/indexing/masked_scatter.h"\n'
        '\ntemplate [[host_name("masked_assign_complex64_false")]] [[kernel]] '
        "decltype(masked_assign_impl<complex64_t, false>) "
        "masked_assign_impl<complex64_t, false>;\n"
    )
    assert len(masked_scatter_packages.ENTRIES) == 26
    assert masked_scatter_packages.signature("masked_assign_bool__true") == (
        "bool_",
        True,
    )


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "single masked assign"),
        (INDEXING + INDEXING, "single masked assign"),
        (
            INDEXING.replace('masked_assign_kernel = R"(', "masked_assign_kernel = R("),
            "no source template",
        ),
        (INDEXING.replace("{2}>)", "{3}>)"), "Unsupported field"),
    ],
)
def test_render_refuses_a_changed_template(text, message):
    with pytest.raises(ValueError, match=message):
        masked_scatter_packages.render(text, "masked_assign_float32_true")


def test_unknown_entries_are_refused():
    for entry in ("masked_assign_float64_true", "masked_assign_float32", None):
        with pytest.raises(ValueError, match="Unknown masked assign entry"):
            masked_scatter_packages.signature(entry)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("contiguous", (True, False))
def test_only_contiguous_opengl_reads_assert_their_signed_index(target, contiguous):
    entry = f"masked_assign_int32_{str(contiguous).lower()}"
    config = masked_scatter_packages.project_config(ROOT, "work", target, entry)
    expected = (
        [("work/kernel.metal", "batch_idx * src_batch_size + src_index", 0, 2**31 - 1)]
        if target == "opengl" and contiguous
        else []
    )
    assert [
        (
            assertion.source,
            assertion.expression,
            assertion.value_range.minimum,
            assertion.value_range.maximum,
        )
        for assertion in config.index_range_assertions
    ] == expected
    assert config.workgroup_size == (1, 1, 1)


def test_offsets_are_exclusive_counts_within_each_batch():
    mask = [1, 0, 1, 1, 0, 0, 1, 1]
    assert masked_scatter_evidence.expected_offsets(mask, 4) == [0, 1, 1, 2, 0, 0, 0, 1]
    assert masked_scatter_evidence.expected_offsets(mask, 8) == [0, 1, 1, 2, 3, 3, 3, 4]


def test_strided_sources_read_through_elem_to_loc():
    # A (1, 2, 3) source transposed from (1, 3, 2): element 4 is (0, 1, 1).
    assert masked_scatter_dispatch.source_location(4, [1, 2, 3], [6, 1, 2], False) == 3
    assert masked_scatter_dispatch.source_location(4, [1, 2, 3], [6, 1, 2], True) == 4
    assert masked_scatter_dispatch.row_contiguous([1, 2, 3], [7, 3, 1])
    assert not masked_scatter_dispatch.row_contiguous([1, 2, 3], [6, 1, 2])


def native_buffers(dtype, mask, offsets, source, shape, strides, initial, **faults):
    ctype = element_storage.CTYPES[dtype]
    batches = shape[0]
    memory = {
        "mask": (ctypes.c_uint8 * len(mask))(*mask),
        "scatter_offsets": (ctypes.c_uint32 * len(offsets))(*offsets),
        "src": (ctype * len(source))(*source),
        "out": (ctype * len(initial))(*initial),
        "src_shapes": (ctypes.c_int32 * len(shape))(*shape),
        "src_strides": (ctypes.c_int64 * len(strides))(*strides),
        "src_ndim": (ctypes.c_int32 * 1)(len(shape)),
        "src_batch_size": (ctypes.c_int64 * 1)(
            faults.get("src_batch", len(source) // batches if strides else 0)
        ),
        "mask_batch_size": (ctypes.c_int64 * 1)(len(mask) // batches),
    }
    entry = "masked_assign_{}_{}".format(dtype, faults.get("contiguous", "true"))
    types = masked_scatter_dispatch.operand_types(entry)
    array = (runtime.Buffer * 9)(
        *(
            runtime.Buffer(
                name.encode(),
                types[name].encode(),
                ctypes.addressof(memory[name]),
                len(memory[name]),
                int(name == "out"),
            )
            for name in memory
        )
    )
    return entry, array, memory


class FakeCache:
    def __init__(self, target):
        self.target = target

    def get(self, entry):
        types = masked_scatter_dispatch.operand_types(entry)
        bindings = []
        for name, dtype in types.items():
            value = element_storage.packet(dtype, self.target, [])
            physical = value["dtype"]
            layout = {
                "elementType": physical,
                "elementStrideBytes": element_storage.PHYSICAL_SIZES[physical],
                # Direct3D prefixes its constant buffer members with the entry.
                "memberName": entry + "_" + name if name in ("src_ndim",) else name,
            }
            if dtype == element_storage.COMPLEX:
                layout = dict(element_storage.COMPLEX_LAYOUT, memberName=name)
            bindings.append({"name": name + "Binding", "scalarLayout": layout})
        return {"bindings": bindings, "artifact": {"id": entry}}, Path("/packages")


def run(monkeypatch, tmp_path, target, dtype, case, **options):
    fault = options.pop("fault", None)
    threads = len(case["mask"])
    count = options.pop("count", (threads, 1, 1))
    entry, array, memory = native_buffers(dtype, **case, **options)
    host = SimpleNamespace(
        masked_scatters=FakeCache(target),
        target=target,
        trace=tmp_path / "dispatch.jsonl",
        trace_values=True,
        dispatch_count=0,
    )

    def execute(_, request):
        inputs = request["inputs"]
        words = {
            name: element_storage.unpack(
                kind, target, inputs[name + "Binding"]["values"]
            )
            for name, kind in masked_scatter_dispatch.operand_types(entry).items()
        }
        metadata = {
            "mask": words["mask"],
            "scatterOffsets": words["scatter_offsets"],
            "sourceShape": words["src_shapes"],
            "sourceStrides": words["src_strides"],
            "sourceBatchSize": words["src_batch_size"][0],
            "maskBatchSize": words["mask_batch_size"][0],
            "sourceContiguous": entry.endswith("_true"),
        }
        result = (
            masked_scatter_evidence.expected_output(
                metadata, words["src"], words["out"][:threads]
            )
            + words["out"][threads:]
        )
        if fault == "result":
            result[0] ^= 1
        if fault == "guard":
            result[-1] ^= 1
        return SimpleNamespace(
            status="ok",
            outputs={"outBinding": element_storage.packet(dtype, target, result)},
            details={"request": {}},
        )

    monkeypatch.setattr(
        masked_scatter_dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **_: {
            "inputs": inputs,
            "outputs": outputs,
            "execution": execution,
        },
    )
    monkeypatch.setattr(masked_scatter_dispatch, "execute", execute)
    launch = runtime.Launch(
        (ctypes.c_uint32 * 3)(*count),
        (ctypes.c_uint32 * 3)(1, 1, 1),
    )
    masked_scatter_dispatch.dispatch(host, entry, array, 9, threads, launch)
    event = json.loads(host.trace.read_text().splitlines()[-1])
    return list(memory["out"]), event, host


# Two batches of four mask elements over a (2, 3) contiguous source; the
# second batch sets a fourth element, past its source row, which keeps the
# destination's value.
CONTIGUOUS = {
    "mask": [1, 0, 1, 0, 1, 1, 1, 1],
    "offsets": [0, 1, 1, 2, 0, 1, 2, 3],
    "source": [10, 11, 12, 20, 21, 22],
    "shape": [2, 3],
    "strides": [3, 1],
    "initial": [0, 1, 2, 3, 4, 5, 6, 7],
}
# One batch over a transposed (1, 2, 3) source.
STRIDED = {
    "mask": [0, 1, 1, 0, 1, 1],
    "offsets": [0, 0, 1, 2, 2, 3],
    "source": [30, 31, 32, 33, 34, 35],
    "shape": [1, 2, 3],
    "strides": [6, 1, 2],
    "initial": [9, 9, 9, 9, 9, 9],
}


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "dtype, case, contiguous, expected",
    [
        ("int32", CONTIGUOUS, "true", [10, 1, 11, 3, 20, 21, 22, 7]),
        ("uint64", CONTIGUOUS, "true", [10, 1, 11, 3, 20, 21, 22, 7]),
        ("int16", STRIDED, "false", [9, 30, 32, 9, 34, 31]),
    ],
)
def test_outputs_are_recomputed_from_the_trace(
    monkeypatch, tmp_path, target, dtype, case, contiguous, expected
):
    values, event, host = run(
        monkeypatch, tmp_path, target, dtype, case, contiguous=contiguous
    )
    assert values == expected and host.dispatch_count == 1
    assert event["outputValues"] == expected
    assert event["initialValues"] == case["initial"]
    assert masked_scatter_evidence.audit_event(event, native=False) == 1
    changed = dict(event, outputValues=[expected[0] + 1, *expected[1:]])
    with pytest.raises(ValueError, match="differs from its operands"):
        masked_scatter_evidence.audit_event(changed, native=False)
    metadata = dict(event["maskedScatterMetadata"])
    metadata["scatterOffsets"] = [0] * len(metadata["mask"])
    with pytest.raises(ValueError, match="exclusive counts"):
        masked_scatter_evidence.audit_event(
            dict(event, maskedScatterMetadata=metadata), native=False
        )


def test_float32_and_boolean_outputs_keep_their_storage(monkeypatch, tmp_path):
    bits = [0x3F800000, 0x40000000, 0x40400000, 0x40800000]
    case = dict(STRIDED, source=bits + bits[:2], initial=[0] * 6)
    values, event, _ = run(
        monkeypatch, tmp_path, "opengl", "float32", case, contiguous="false"
    )
    assert values == [0, bits[0], bits[2], 0, bits[0], bits[1]]
    assert masked_scatter_evidence.audit_event(event, native=False) == 1
    case = dict(CONTIGUOUS, source=[1, 0, 1, 0, 1, 1], initial=[0] * 8)
    values, event, _ = run(monkeypatch, tmp_path, "directx", "bool_", case)
    assert values == [1, 0, 0, 0, 0, 1, 1, 0]
    assert masked_scatter_evidence.audit_event(event, native=False) == 1


@pytest.mark.parametrize(
    "options, message",
    [
        ({"fault": "guard"}, "output guard"),
        ({"fault": "result"}, None),
        ({"count": (4, 1, 1)}, "one thread per element"),
        ({"contiguous": "false"}, "source layout"),
        ({"src_batch": 2}, "source layout"),
    ],
)
def test_dispatch_refuses_changed_guards_launches_and_layouts(
    monkeypatch, tmp_path, options, message
):
    if message is None:
        _, event, _ = run(
            monkeypatch, tmp_path, "metal", "int32", CONTIGUOUS, **options
        )
        with pytest.raises(ValueError, match="differs from its operands"):
            masked_scatter_evidence.audit_event(event, native=False)
        return
    with pytest.raises((ValueError, RuntimeError), match=message):
        run(monkeypatch, tmp_path, "metal", "int32", CONTIGUOUS, **options)


def test_runtime_routes_masked_scatters_through_their_cache():
    source = (HOST / "runtime.py").read_text()
    assert "masked_scatter_packages.MaskedScatterPackageCache(" in source
    assert "if entry in masked_scatter_packages.ENTRIES:" in source
    # The streaming audit checks owned families against their package caches.
    audit = (HOST / "verify_upstream_ops.py").read_text()
    assert (
        "            masked_scatter_evidence.audit_event,\n"
        "            masked_scatter_packages.MaskedScatterPackageCache(\n"
    ) in audit
    assert "lambda event: check(event, cache)" in audit


def test_backend_ports_masked_scatter_through_a_scan_and_masked_assign():
    assert '        "MaskedScatter",\n' in (HOST / "prepare.py").read_text()
    assert prepare.HERE == HOST
    source = (HOST / "scatter.inc").read_text()
    assert "void MaskedScatter::eval_gpu(" in source
    assert "dispatch_scan(mask_flat, scatter_offsets, Scan::Sum, 1, false, false);" in (
        source
    )
    assert 'std::string("masked_assign_") + value_type' in source
    assert "{{static_cast<uint32_t>(total), 1, 1}, {1, 1, 1}}" in source
    backend = (HOST / "backend.cpp").read_text()
    assert (
        "dispatch_scan(inputs[0], out, reduce_type_, axis_, reverse_, inclusive_);" in (
            backend
        )
    )
    assert "(out.dtype() == int32 || out.dtype() == uint32)" in backend


def test_the_runtime_imports_the_dispatch_without_the_audit():
    # The evidence modules import the runtime, so the dispatch must not.
    lines = (HOST / "masked_scatter_dispatch.py").read_text().splitlines()
    imports = [line for line in lines if line.startswith(("from ", "import "))]
    assert imports and not any("_evidence" in line for line in imports)
