"""Dynamic slice offsets format MLX's generated kernel exactly, check their
launches and operands, and recompute every offset from the retained trace."""

import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import (
    dynamic_offset_dispatch,
    dynamic_offset_evidence,
    dynamic_offset_packages,
    element_storage,
    prepare,
    runtime,
)

ROOT = Path(__file__).resolve().parents[5]
HOST = ROOT / "demos/integrations/mlx/portable_host"
TARGETS = ("metal", "directx", "opengl")
# The formatting function of the pinned mlx/backend/metal/slicing.cpp.
SLICING = """
array compute_dynamic_offset(
    const array& indices,
    const Strides& strides,
    const std::vector<int>& axes,
    const Stream& s) {
  auto lib = d.get_library(lib_name, [dtype]() {
    return fmt::format(
        R"(
        [[kernel]] void compute_dynamic_offset_{0}(
            constant const {1}* indices [[buffer(0)]],
            device int64_t& offset [[buffer(1)]],
            constant const int64_t* strides [[buffer(2)]],
            constant const int* axes [[buffer(3)]],
            constant const int& n_axes [[buffer(4)]],
            uint index [[thread_position_in_grid]]) {{
          int64_t acc = 0;
          for (int i = 0; i < n_axes; ++i) {{
            acc += indices[i] * strides[axes[i]];
          }}
          offset = acc;
        }})",
        type_to_name(dtype),
        get_type_string(dtype));
  });
}
"""


def test_render_formats_the_pinned_template_as_fmt_does():
    source = dynamic_offset_packages.render(SLICING, "compute_dynamic_offset_uint32")
    assert "void compute_dynamic_offset_uint32(" in source
    assert "constant const uint32_t* indices [[buffer(0)]]" in source
    assert "{{" not in source and "}}" not in source
    assert source.count("{") == source.count("}") == 2
    assert set(dynamic_offset_packages.ENTRIES.values()) == {
        "int32",
        "uint32",
        "int64",
        "uint64",
    }


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "single offset kernel"),
        (SLICING + SLICING, "single offset kernel"),
        (SLICING.replace('R"(', "R("), "no source template"),
        (SLICING.replace("{0}(", "{2}("), "Unsupported field"),
        (SLICING.replace("{0}(", "{name}("), "Unsupported field"),
        (SLICING.replace("}})", "})"), "Unmatched closing brace"),
    ],
)
def test_render_refuses_a_changed_template(text, message):
    with pytest.raises(ValueError, match=message):
        dynamic_offset_packages.render(text, "compute_dynamic_offset_int32")


def test_unknown_entries_are_refused():
    with pytest.raises(ValueError, match="Unknown dynamic offset entry"):
        dynamic_offset_packages.signature("compute_dynamic_offset_int16")


def native_buffers(dtype, indices, strides, axes, *, n_axes=None):
    ctype = element_storage.CTYPES[dtype]
    memory = {
        "indices": (ctype * len(indices))(*indices),
        "offset": (ctypes.c_int64 * 1)(),
        "strides": (ctypes.c_int64 * len(strides))(*strides),
        "axes": (ctypes.c_int32 * len(axes))(*axes),
        "n_axes": (ctypes.c_int32 * 1)(len(axes) if n_axes is None else n_axes),
    }
    types = dynamic_offset_dispatch.operand_types("compute_dynamic_offset_" + dtype)
    array = (runtime.Buffer * 5)(
        *(
            runtime.Buffer(
                name.encode(),
                types[name].encode(),
                ctypes.addressof(memory[name]),
                len(memory[name]),
                int(name == "offset"),
            )
            for name in memory
        )
    )
    return array, memory


class FakeCache:
    def __init__(self, target):
        self.target = target

    def get(self, entry):
        types = dynamic_offset_dispatch.operand_types(entry)
        # Direct3D prefixes its constant buffer member with the entry.
        bindings = []
        for name, dtype in types.items():
            physical = element_storage.packet(dtype, self.target, [])["dtype"]
            member = entry + "_" + name if name == "n_axes" else name
            bindings.append(
                {
                    "name": name + "Binding",
                    "scalarLayout": {
                        "elementType": physical,
                        "elementStrideBytes": element_storage.PHYSICAL_SIZES[physical],
                        "memberName": member,
                    },
                }
            )
        return {"bindings": bindings, "artifact": {"id": entry}}, Path("/packages")


def run(monkeypatch, tmp_path, target, dtype, indices, strides, axes, **options):
    fault = options.pop("fault", None)
    count = options.pop("count", (1, 1, 1))
    entry = "compute_dynamic_offset_" + dtype
    array, memory = native_buffers(dtype, indices, strides, axes, **options)
    host = SimpleNamespace(
        dynamic_offsets=FakeCache(target),
        target=target,
        trace=tmp_path / "dispatch.jsonl",
        trace_values=True,
        dispatch_count=0,
    )

    def execute(_, request):
        inputs = request["inputs"]
        words = {
            name: element_storage.unpack(
                types, target, inputs[name + "Binding"]["values"]
            )
            for name, types in (
                ("indices", dtype),
                ("strides", "int64"),
                ("axes", "int32"),
            )
        }
        initial = element_storage.unpack(
            "int64", target, inputs["offsetBinding"]["values"]
        )
        offset = dynamic_offset_evidence.expected_offset(
            words["indices"], words["strides"], words["axes"]
        )
        result = [offset ^ (fault == "result")] + initial[1:]
        if fault == "guard":
            result[-1] ^= 1
        return SimpleNamespace(
            status="ok",
            outputs={"offsetBinding": element_storage.packet("int64", target, result)},
            details={"request": {}},
        )

    monkeypatch.setattr(
        dynamic_offset_dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **_: {
            "inputs": inputs,
            "outputs": outputs,
            "execution": execution,
        },
    )
    monkeypatch.setattr(dynamic_offset_dispatch, "execute", execute)
    launch = runtime.Launch(
        (ctypes.c_uint32 * 3)(*count),
        (ctypes.c_uint32 * 3)(1, 1, 1),
    )
    dynamic_offset_dispatch.dispatch(host, entry, array, 5, 1, launch)
    event = json.loads(host.trace.read_text().splitlines()[-1])
    return memory["offset"][0], event, host


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize(
    "dtype, indices, strides, axes, offset",
    [
        ("int32", [1], [1], [0], 1),
        ("int32", [1, 2, 3], [42, 7, 1], [0, 1, 2], 59),
        ("uint32", [2, 3], [20, 5, 1], [2, 0], 62),
        ("int64", [-1, 4], [6, 2, 1], [0, 1], 2),
        # A uint64 product wraps modulo 2**64 before the int64 store.
        ("uint64", [2**64 - 1], [3], [0], -3),
    ],
)
def test_offsets_are_recomputed_from_the_trace(
    monkeypatch, tmp_path, target, dtype, indices, strides, axes, offset
):
    value, event, host = run(
        monkeypatch, tmp_path, target, dtype, indices, strides, axes
    )
    assert value == offset and host.dispatch_count == 1
    assert event["dynamicOffsetMetadata"] == {
        "dtype": dtype,
        "indices": indices,
        "strides": strides,
        "axes": axes,
    }
    assert dynamic_offset_evidence.audit_event(event, native=False) == 1
    changed = dict(event, outputValues=[offset + 1])
    with pytest.raises(ValueError, match="differs from its operands"):
        dynamic_offset_evidence.audit_event(changed, native=False)
    metadata = dict(event["dynamicOffsetMetadata"], axes=[len(strides)] * len(axes))
    with pytest.raises(ValueError, match="metadata"):
        dynamic_offset_evidence.audit_event(
            dict(event, dynamicOffsetMetadata=metadata), native=False
        )


@pytest.mark.parametrize(
    "options, message",
    [
        ({"fault": "guard"}, "output guard"),
        ({"count": (2, 1, 1)}, "one thread"),
        ({"n_axes": 2}, "axes exceed"),
    ],
)
def test_dispatch_refuses_changed_guards_launches_and_operands(
    monkeypatch, tmp_path, options, message
):
    with pytest.raises((ValueError, RuntimeError), match=message):
        run(monkeypatch, tmp_path, "metal", "int32", [1], [4, 1], [0], **options)


@pytest.mark.parametrize("axes", [[2], [0, 0]])
def test_dispatch_refuses_axes_outside_the_strides(monkeypatch, tmp_path, axes):
    with pytest.raises(ValueError, match="axes exceed"):
        run(monkeypatch, tmp_path, "opengl", "int32", [1] * len(axes), [4, 1], axes)


def test_runtime_routes_dynamic_offsets_through_their_cache():
    source = (HOST / "runtime.py").read_text()
    assert "dynamic_offset_packages.DynamicOffsetPackageCache(" in source
    assert "if entry in dynamic_offset_packages.ENTRIES:" in source


def test_backend_ports_both_dynamic_slice_primitives():
    assert prepare.DYNAMIC_SLICE_PRIMITIVES == ("DynamicSlice", "DynamicSliceUpdate")
    source = (HOST / "backend.cpp").read_text()
    for name in prepare.DYNAMIC_SLICE_PRIMITIVES:
        assert f"void {name}::eval_gpu(" in source
    # The slice reads from the dynamic source offset, the update writes at the
    # dynamic destination offset of its row-contiguous copy.
    assert "dynamic_offset(start, in.strides(), axes_)" in source
    assert "dynamic_offset(start_indices, out.strides(), axes_)" in source
    assert 'std::string("compute_dynamic_offset_") + type' in source
    assert "origin + src_offset + low < 0" in source


def test_the_runtime_imports_the_dispatch_without_the_audit():
    # The evidence modules import the runtime, so the dispatch must not.
    lines = (HOST / "dynamic_offset_dispatch.py").read_text().splitlines()
    imports = [line for line in lines if line.startswith(("from ", "import "))]
    assert imports and not any("_evidence" in line for line in imports)
