"""DivMod callbacks check binary_op_gpu layouts and audit exact results."""

import ctypes
import hashlib
import json
from pathlib import Path

import pytest

from demos.integrations.mlx.portable_host import (
    divmod_dispatch,
    divmod_evidence,
    divmod_packages,
    runtime,
)

ROOT = Path(__file__).resolve().parents[5]
CONTRACT = (
    ROOT / "demos/integrations/mlx/contracts/binary_two.current-tree.translation.json"
)


def buffers(kernel, dtype, shape, a_strides, b_strides):
    """Allocate a callback's buffers for ``shape`` with the given operand strides."""
    count = 1
    for size in shape:
        count *= size
    spans = [
        1 + sum((size - 1) * stride for size, stride in zip(shape, strides))
        for strides in (a_strides, b_strides)
    ]
    data = {
        "a": (dtype, spans[0]),
        "b": (dtype, spans[1]),
        "c": (dtype, count),
        "d": (dtype, count),
    }
    values = {}
    if kernel in {"sv", "vs", "vv"}:
        data["size"], values["size"] = ("uint32", 1), [count]
    elif kernel == "g1":
        data["a_stride"], data["b_stride"] = ("int64", 1), ("int64", 1)
        values["a_stride"], values["b_stride"] = a_strides, b_strides
    elif kernel in {"g2", "g3"}:
        data["a_strides"], data["b_strides"] = ("int64", len(shape)), (
            "int64",
            len(shape),
        )
        values["a_strides"], values["b_strides"] = a_strides, b_strides
    elif kernel == "gn2":
        data.update(
            shape=("int32", len(shape)),
            a_strides=("int64", len(shape)),
            b_strides=("int64", len(shape)),
            ndim=("int32", 1),
        )
        values.update(
            shape=shape, a_strides=a_strides, b_strides=b_strides, ndim=[len(shape)]
        )
    memory = {
        name: (
            (divmod_dispatch.CTYPES.get(kind) or divmod_dispatch.METADATA_CTYPES[kind])
            * size
        )(*values.get(name, ()))
        for name, (kind, size) in data.items()
    }
    supplied = {
        name: runtime.Buffer(
            name.encode(),
            kind.encode(),
            ctypes.addressof(memory[name]),
            size,
            int(name in "cd"),
        )
        for name, (kind, size) in data.items()
    }
    return supplied, memory, count


def launch(kernel, shape, count):
    grid = (
        [count, 1, 1]
        if kernel in {"ss", "sv", "vs", "vv", "g1"}
        else divmod_dispatch.general_launch(kernel, shape)
    )
    return {"workgroupCount": grid, "workgroupSize": [1, 1, 1]}


def test_entries_are_pinned_binary_two_instantiations():
    pinned = json.loads(CONTRACT.read_text(encoding="utf-8"))["artifacts"]
    for target in ("metal", "directx", "opengl"):
        assert set(divmod_packages.ENTRIES) <= set(pinned[target])
    assert len(divmod_packages.ENTRIES) == 88
    assert divmod_packages.signature("gn2_DivModuint16") == ("gn2", "uint16")
    for entry in ("vv_DivModbool_", "vv_DivModcomplex64", "vvn_DivModfloat32", None):
        with pytest.raises(ValueError, match="Unknown DivMod entry"):
            divmod_packages.signature(entry)
    assert all(
        entry.startswith(divmod_dispatch.ENTRY_PREFIXES)
        for entry in divmod_packages.ENTRIES
    )


@pytest.mark.parametrize(
    ("kernel", "shape", "a_strides", "b_strides"),
    [
        ("ss", [5], [0], [0]),
        ("sv", [7], [0], [1]),
        ("vs", [7], [1], [0]),
        ("vv", [9], [1], [1]),
        ("g1", [6], [2], [0]),
        ("g2", [2, 3], [0, 1], [3, 0]),
        ("g3", [2, 2, 2], [4, 2, 1], [0, 1, 0]),
        ("gn2", [2, 1, 2, 3], [6, 0, 3, 1], [0, 0, 1, 0]),
    ],
)
def test_validate_derives_each_kernel_layout(kernel, shape, a_strides, b_strides):
    supplied, _, count = buffers(kernel, "int16", shape, a_strides, b_strides)
    execution = launch(kernel, shape, count)
    metadata = divmod_dispatch.validate(
        f"{kernel}_DivModint16", supplied, count, execution
    )
    expected = shape if kernel not in {"ss", "sv", "vs", "vv"} else [count]
    assert metadata["shape"] == expected
    assert metadata["kernel"] == kernel and metadata["dtype"] == "int16"


def test_validate_rejects_mismatched_layouts():
    shape, a_strides, b_strides = [2, 2, 2], [4, 2, 1], [0, 1, 0]
    supplied, memory, count = buffers("g3", "uint16", shape, a_strides, b_strides)
    entry, execution = "g3_DivModuint16", launch("g3", shape, count)
    with pytest.raises(ValueError, match="launch does not match"):
        divmod_dispatch.validate(
            entry, supplied, count, {**execution, "workgroupCount": [2, 2, 1]}
        )
    with pytest.raises(ValueError, match="buffers do not match"):
        divmod_dispatch.validate("vv_DivModuint16", supplied, count, execution)
    with pytest.raises(ValueError, match="type, extent"):
        divmod_dispatch.validate("g2_DivModuint16", supplied, count, execution)
    with pytest.raises(ValueError, match="type, extent"):
        divmod_dispatch.validate("g3_DivModint16", supplied, count, execution)
    memory["b_strides"][1] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        divmod_dispatch.validate(entry, supplied, count, execution)
    memory["b_strides"][1] = 2
    with pytest.raises(ValueError, match="span does not match"):
        divmod_dispatch.validate(entry, supplied, count, execution)
    memory["b_strides"][1] = 1
    overlapping = dict(supplied)
    overlapping["d"] = runtime.Buffer(b"d", b"uint16", supplied["c"].data, count, 1)
    with pytest.raises(ValueError, match="overlap"):
        divmod_dispatch.validate(entry, overlapping, count, execution)
    reversed_direction = dict(supplied)
    reversed_direction["c"] = runtime.Buffer(
        b"c", b"uint16", supplied["c"].data, count, 0
    )
    with pytest.raises(ValueError, match="direction"):
        divmod_dispatch.validate(entry, reversed_direction, count, execution)
    assert divmod_dispatch.validate(entry, supplied, count, execution)["shape"] == shape


def test_logical_operands_follow_broadcast_strides():
    metadata = {"shape": [2, 3], "aStrides": [0, 1], "bStrides": [1, 0]}
    pairs = divmod_dispatch.logical_operands(metadata, [10, 20, 30], [1, 2])
    assert pairs == [(10, 1), (20, 1), (30, 1), (10, 2), (20, 2), (30, 2)]


@pytest.mark.parametrize(
    ("dtype", "pair"),
    [
        ("int32", (5, 0)),
        ("uint16", (5, 0)),
        ("int8", (-128, -1)),
        ("int64", (-(1 << 63), -1)),
    ],
)
def test_integer_operands_refuse_undefined_division(dtype, pair):
    with pytest.raises(ValueError, match="nonzero divisors"):
        divmod_dispatch.check_integer_operands(dtype, [pair])


def test_float_operands_are_not_integer_checked():
    divmod_dispatch.check_integer_operands("float32", [(0x3F800000, 0)])


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
@pytest.mark.parametrize(
    ("dtype", "words"),
    [
        ("float16", [0x3C00, 0x8001, 0x7BFF]),
        ("bfloat16", [0x3F80, 0xC120]),
        ("uint16", [0, 65535]),
        ("int8", [-128, 127]),
        ("uint64", [1 << 63]),
    ],
)
def test_storage_packets_round_trip_exactly(target, dtype, words):
    value = divmod_dispatch.packet(dtype, target, words)
    assert value["shape"] == [len(words)]
    assert divmod_dispatch.unpack(dtype, target, value["values"]) == words
    if dtype in {"uint16", "int8"} and target == "opengl":
        assert value["dtype"] in {"uint32", "int32"}
    if dtype == "float16" and target == "directx":
        assert divmod_dispatch.reflected_layout_matches(
            {
                "elementType": "uint16",
                "elementStrideBytes": 2,
                "storageEncoding": {
                    "encoding": "ieee754-binary16",
                    "logicalElementType": "float16",
                },
            },
            value,
        )


def test_unpack_rejects_out_of_range_narrow_results():
    with pytest.raises(ValueError, match="invalid scalar storage"):
        divmod_dispatch.unpack("uint16", "opengl", [65536])
    with pytest.raises(ValueError, match="invalid scalar storage"):
        divmod_dispatch.unpack("int8", "directx", [128])


def test_integer_reference_floors_like_python():
    # The unchanged upstream test_divmod mixed-sign operands.
    av = [-7, 7, -7, 7, -1, 1, -5, 5, 6, -6]
    bv = [2, 2, -2, -2, 3, -3, 3, -3, 3, 3]
    for x, y in zip(av, bv):
        assert divmod_evidence.integer_divmod(x, y, "int32") == (x // y, x % y)
    assert divmod_evidence.integer_divmod(255, 7, "uint8") == (36, 3)
    with pytest.raises(ValueError, match="overflows"):
        divmod_evidence.integer_divmod(-128, -1, "int8")


def words(values, dtype):
    return [divmod_evidence.float_word(value, dtype) for value in values]


@pytest.mark.parametrize("dtype", ("float32", "float16", "bfloat16"))
def test_float_reference_floors_and_moves_remainders(dtype):
    a, b = words([-7.0, 7.0, -7.5, 7.5], dtype), words([2.0, -2.0, 2.0, -2.0], dtype)
    results = [divmod_evidence.float_divmod(x, y, dtype) for x, y in zip(a, b)]
    assert [q for q, _ in results] == words([-4.0] * 4, dtype)
    assert [r for _, r in results] == words([1.0, -1.0, 0.5, -0.5], dtype)
    with pytest.raises(ValueError, match="normal binary operands"):
        divmod_evidence.float_divmod(1, a[0], dtype)


def event(target="opengl"):
    metadata = {
        "kernel": "g2",
        "dtype": "uint16",
        "shape": [2, 3],
        "aStrides": [0, 1],
        "bStrides": [1, 0],
    }
    operands = {"a": [10, 20, 30], "b": [3, 7]}
    pairs = divmod_dispatch.logical_operands(metadata, operands["a"], operands["b"])
    guard = [divmod_dispatch.GUARDS["uint16"]] * divmod_dispatch.GUARD_COUNT
    outputs = {"c": [x // y for x, y in pairs], "d": [x % y for x, y in pairs]}
    payload = b"".join(bytes((ctypes.c_uint16 * 6)(*outputs[name])) for name in "cd")
    return {
        "entry": "g2_DivModuint16",
        "target": target,
        "threads": 6,
        "divmodMetadata": metadata,
        "operandValues": operands,
        "inputs": {
            name: divmod_dispatch.packet("uint16", target, operands[name] + guard)
            for name in "ab"
        },
        "outputValues": outputs,
        "outputGuards": {"c": guard, "d": guard},
        "outputHash": hashlib.sha256(payload).hexdigest(),
    }


def test_audit_recomputes_every_output_from_retained_operands():
    assert divmod_evidence.audit_event(event(), native=False) == 6
    changed = event()
    changed["outputValues"]["d"][4] += 1
    with pytest.raises(ValueError, match="result 4"):
        divmod_evidence.audit_event(changed, native=False)
    uploaded = event()
    uploaded["inputs"]["a"]["values"][0] = 11
    with pytest.raises(ValueError, match="upload of a"):
        divmod_evidence.audit_event(uploaded, native=False)
    guarded = event()
    guarded["outputGuards"]["c"] = guarded["outputGuards"]["c"][1:] + [0]
    with pytest.raises(ValueError, match="guard"):
        divmod_evidence.audit_event(guarded, native=False)
    hashed = event()
    hashed["outputHash"] = "0" * 64
    with pytest.raises(ValueError, match="output hash"):
        divmod_evidence.audit_event(hashed, native=False)


def test_audit_checks_the_native_request_and_modules(monkeypatch):
    seen = []
    monkeypatch.setattr(divmod_evidence, "audit_native_execution", seen.append)
    record = event()
    request = {
        "target": "opengl",
        "entryPoint": "main",
        "dispatch": {"workgroupCount": [3, 2, 1], "workgroupSize": [1, 1, 1]},
    }
    record.update(
        workgroupCount=[3, 2, 1],
        workgroupSize=[1, 1, 1],
        dispatchVersion=3,
        details={"request": request},
    )
    assert divmod_evidence.audit_event(record) == 6 and seen == [record]
    request["dispatch"]["workgroupCount"] = [6, 1, 1]
    with pytest.raises(ValueError, match="native request changed"):
        divmod_evidence.audit_event(record)


def test_package_config_pins_profiles_and_opengl_index_bounds():
    config = divmod_packages._config("out", "opengl", "g3_DivModfloat32")
    assert 'binary32_division_profile = "rne-flush"' in config
    assert config.count("[[project.index_range_assertions]]") == len(
        divmod_packages.OPENGL_INDEX_EXPRESSIONS
    )
    half = divmod_packages._config("out", "directx", "vv_DivModfloat16")
    assert 'binary16_remainder_profile = "binary32-quotient"' in half
    assert "index_range_assertions" not in half
    integer = divmod_packages._config("out", "metal", "ss_DivModint8")
    assert "profile" not in integer
