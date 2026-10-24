"""Lazy unary, binary and ternary vectors mirror the pinned instantiations, check
their launches and operands, and recompute integer and Boolean results exactly."""

import ctypes
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from demos.integrations.mlx.portable_host import (
    element_storage,
    elementwise_dispatch,
    elementwise_evidence,
    elementwise_packages,
    elementwise_warm,
    packages,
    runtime,
)

ROOT = Path(__file__).resolve().parents[5]
CONTRACTS = ROOT / "demos/integrations/mlx/contracts"
TARGETS = ("metal", "directx", "opengl")
INTEGERS = ("uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64")


def pinned_vectors():
    names = set()
    for source in ("unary", "binary"):
        path = CONTRACTS / f"{source}.current-tree.translation.json"
        entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
        names |= {entry for entry in entries if re.match(r"vv?_[A-Z]", entry)}
    return names


def pinned_selections():
    path = CONTRACTS / "ternary.current-tree.translation.json"
    entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
    return {entry for entry in entries if entry.startswith("v_Select")}


def test_entries_are_every_pinned_vector_instantiation():
    pinned = pinned_vectors()
    portable = {name for name in pinned if "FP8" not in name}
    selections = pinned_selections()
    assert len(selections) == 13
    lazy_selections = selections - set(packages.SELECTION_ENTRIES)
    assert set(elementwise_packages.ENTRIES) == pinned | lazy_selections
    assert len(portable) == 415 and len(pinned) - len(portable) == 6
    assert len(lazy_selections) == 9 and len(elementwise_packages.ENTRIES) == 430
    # fast::ConvertFP8 converts each real float to and from E4M3 bytes.
    assert {
        name: elementwise_packages.signature(name) for name in pinned - portable
    } == {
        f"v_{operation}{source}{destination}": ("v_", operation, source, destination)
        for operation, source, destination in elementwise_packages.FP8_CONVERSIONS
    }
    assert sum("complex64" in name for name in portable) == 40
    assert elementwise_packages.signature("v_Selectfloat16") == (
        "ternary_v",
        "Select",
        "float16",
        "float16",
    )
    assert elementwise_packages.operand_types("v_Selectcomplex64") == {
        "a": "bool_",
        "b": "complex64",
        "c": "complex64",
        "d": "complex64",
    }
    signature = elementwise_packages.signature
    assert signature("vv_Lessbfloat16") == ("vv_", "Less", "bfloat16", "bool_")
    assert signature("v_Absint8int8") == ("v_", "Abs", "int8", "int8")
    assert signature("vv_Addbool_") == ("vv_", "Add", "bool_", "bool_")
    assert signature("v_Realcomplex64float32") == ("v_", "Real", "complex64", "float32")
    assert signature("vv_NaNEqualcomplex64") == (
        "vv_",
        "NaNEqual",
        "complex64",
        "bool_",
    )
    for entry in (
        "v_Sinint32int32",
        "vv_LeftShiftbool_",
        "v_copyint8int8",
        "v_ArcTanhcomplex64complex64",
        "vv_ArcTan2complex64",
        "v_Selectfloat32",
        "v_Selectbool_",
        None,
    ):
        with pytest.raises(ValueError, match="Unknown elementwise entry"):
            signature(entry)
    with pytest.raises(ValueError, match="Unknown elementwise entry"):
        elementwise_packages.UNARY_RECIPE.signature("vv_Addint8")
    with pytest.raises(ValueError, match="Unknown elementwise entry"):
        elementwise_packages.BINARY_RECIPE.signature("v_Absint8int8")
    with pytest.raises(ValueError, match="Unknown elementwise entry"):
        elementwise_packages.UNARY_RECIPE.signature("v_Selectint8")
    assert elementwise_packages.TERNARY_RECIPE.source == packages.SELECTION_SOURCE


@pytest.mark.parametrize(
    ("entry", "profiles"),
    [
        ("vv_Addint8", set()),
        ("vv_Addbfloat16", {"binary32_additive_profile"}),
        ("vv_Multiplyfloat16", set()),
        ("vv_Multiplybfloat16", {"binary32_multiplication_profile"}),
        ("vv_Dividebfloat16", {"binary32_division_profile"}),
        (
            "vv_Remainderbfloat16",
            {
                "binary32_comparison_profile",
                "binary32_remainder_profile",
                "binary32_additive_profile",
            },
        ),
        ("vv_Remainderfloat16", {"binary16_remainder_profile"}),
        ("vv_Maximumbfloat16", {"binary32_comparison_profile"}),
        (
            "vv_Powerfloat32",
            {"binary32_power_operand_profile", "binary32_power_accuracy_profile"},
        ),
        ("vv_Powerbfloat16", {"binary32_power_operand_profile"}),
        ("vv_ArcTan2float16", {"binary32_atan2_profile"}),
        ("v_Logfloat16float16", {"binary32_log_profile"}),
        ("v_Log2float16float16", set()),
        ("v_Logcomplex64complex64", {"binary32_log_profile", "binary32_atan2_profile"}),
        ("v_ArcCoscomplex64complex64", {"binary32_atan2_profile"}),
        ("v_ArcTancomplex64complex64", set()),
        ("vv_Multiplycomplex64", set()),
        ("vv_Powercomplex64", set()),
        ("v_Selectbfloat16", set()),
        ("v_Selectcomplex64", set()),
    ],
)
def test_profiles_follow_the_kernel_proofs(entry, profiles):
    for target in TARGETS:
        text = elementwise_packages.options(target, entry)
        options = dict(re.findall(r'^(\w+) = "([^"]+)"$', text, re.M))
        base = {name for name, _ in elementwise_packages.PROFILES}
        assert set(options) == base | profiles
        assert "index_range" not in text


def native_buffers(entry, operands, *, donate=False):
    """A callback's buffers for one vector: inputs, output and size."""
    kernel, _, source, destination = elementwise_packages.signature(entry)
    names, output = elementwise_dispatch.operands(kernel)
    types = {**elementwise_packages.operand_types(entry), "size": "uint32"}
    threads = len(operands[names[0]])
    memory = {
        name: (element_storage.CTYPES[types[name]] * threads)(*operands[name])
        for name in names
    }
    memory[output] = (
        memory[names[0]]
        if donate
        else (element_storage.CTYPES[destination] * threads)()
    )
    memory["size"] = (ctypes.c_uint32 * 1)(threads)
    array = (runtime.Buffer * len(memory))(
        *(
            runtime.Buffer(
                name.encode(),
                types[name].encode(),
                ctypes.addressof(memory[name]),
                len(memory[name]),
                int(name == output),
            )
            for name in memory
        )
    )
    return array, memory, threads


class FakeCache:
    def __init__(self, target):
        self.target = target

    def get(self, entry):
        kernel, _, _, _ = elementwise_packages.signature(entry)
        names, output = elementwise_dispatch.operands(kernel)
        types = {**elementwise_packages.operand_types(entry), "size": "uint32"}
        prefix = "" if self.target == "metal" else entry.rstrip("_") + "_"
        bindings = []
        for name in (*names, output, "size"):
            dtype = types[name]
            physical = (
                element_storage.packet(dtype, self.target, [])["dtype"]
                if name != "size"
                else "uint32"
            )
            member = {"in": "in_", "out": "out_"}.get(name, name)
            layout = (
                dict(element_storage.COMPLEX_LAYOUT)
                if dtype == "complex64"
                else {
                    "elementType": physical,
                    "elementStrideBytes": element_storage.PHYSICAL_SIZES[physical],
                }
            )
            bindings.append(
                {
                    "name": name + "Binding",
                    "scalarLayout": {**layout, "memberName": prefix + member},
                }
            )
        return {"bindings": bindings, "artifact": {"id": entry}}, Path("/packages")


def run(
    monkeypatch,
    tmp_path,
    target,
    entry,
    operands,
    *,
    fault=None,
    donate=False,
    compute=None,
):
    kernel, operation, source, destination = elementwise_packages.signature(entry)
    names, output = elementwise_dispatch.operands(kernel)
    array, memory, threads = native_buffers(entry, operands, donate=donate)
    host = SimpleNamespace(
        elementwise=FakeCache(target),
        target=target,
        trace=tmp_path / "dispatch.jsonl",
        trace_values=True,
        dispatch_count=0,
    )

    types = elementwise_packages.operand_types(entry)

    def execute(_, request):
        values = {
            name: element_storage.unpack(
                types[name], target, request["inputs"][name + "Binding"]["values"]
            )
            for name in names
        }
        initial = element_storage.unpack(
            destination, target, request["inputs"][output + "Binding"]["values"]
        )
        result = (compute or elementwise_evidence.expected_words)(
            kernel, operation, source, destination, values
        )
        result += initial[threads:]
        if fault == "guard":
            result[-1] ^= 1
        if fault == "result":
            result[0] ^= 1
        return SimpleNamespace(
            status="ok",
            outputs={
                output + "Binding": element_storage.packet(destination, target, result)
            },
            details={"request": {}},
        )

    monkeypatch.setattr(
        elementwise_dispatch,
        "build_native_loader_dispatch_request",
        lambda descriptor, directory, inputs, outputs, execution, **_: {
            "inputs": inputs,
            "outputs": outputs,
            "execution": execution,
        },
    )
    monkeypatch.setattr(elementwise_dispatch, "execute", execute)
    launch = runtime.Launch(
        (ctypes.c_uint32 * 3)(threads, 1, 1), (ctypes.c_uint32 * 3)(1, 1, 1)
    )
    elementwise_dispatch.dispatch(host, entry, array, len(memory), threads, launch)
    event = json.loads(host.trace.read_text().splitlines()[-1])
    return list(memory[output]), event, host


@pytest.mark.parametrize("target", TARGETS)
def test_narrow_vectors_dispatch_through_reflected_carriers(
    monkeypatch, tmp_path, target
):
    a, b = [-128, -1, 0, 1, 127], [-1, 127, 5, 1, 1]
    words, event, host = run(
        monkeypatch, tmp_path, target, "vv_Addint8", {"a": a, "b": b}
    )
    assert host.dispatch_count == 1
    assert words == np.add(np.int8(a), np.int8(b)).tolist()
    assert elementwise_evidence.audit_event(event, native=False) == 5
    words, event, _ = run(
        monkeypatch, tmp_path, target, "vv_Lessuint16", {"a": [1, 9], "b": [2, 3]}
    )
    assert words == [1, 0]
    words, event, _ = run(
        monkeypatch, tmp_path, target, "v_Negativeuint8uint8", {"in": [0, 1, 255]}
    )
    assert words == [0, 255, 1]
    assert elementwise_evidence.audit_event(event, native=False) == 3
    with pytest.raises(RuntimeError, match="guard"):
        run(
            monkeypatch,
            tmp_path,
            target,
            "vv_Multiplyint16",
            {"a": [1], "b": [2]},
            fault="guard",
        )


def complex_words(values, dtype=np.complex64):
    """MLX's complex64 storage words: real bits low, imaginary bits high."""
    return np.asarray(values, dtype=dtype).view(np.uint64).tolist()


def numpy_complex(kernel, operation, source, destination, values):
    """The float32 pair arithmetic of a complex fake; the audit never recomputes it."""
    arrays = {
        name: np.asarray(words, dtype=np.uint64).view(np.complex64)
        for name, words in values.items()
    }
    if operation == "Multiply":
        return complex_words(arrays["a"] * arrays["b"])
    if operation == "Equal":
        return (arrays["a"] == arrays["b"]).astype(int).tolist()
    if operation == "Real":
        return arrays["in"].real.view(np.uint32).tolist()
    assert operation == "Conjugate"
    return complex_words(np.conj(arrays["in"]))


@pytest.mark.parametrize("target", TARGETS)
def test_complex_vectors_dispatch_through_packed_float_pairs(
    monkeypatch, tmp_path, target
):
    a, b = [1 + 2j, -0.5 + 0j, 3 - 4j], [2 - 1j, 4 + 4j, 3 - 4j]
    words, event, host = run(
        monkeypatch,
        tmp_path,
        target,
        "vv_Multiplycomplex64",
        {"a": complex_words(a), "b": complex_words(b)},
        compute=numpy_complex,
    )
    assert host.dispatch_count == 1
    assert words == complex_words(np.complex64(a) * np.complex64(b))
    upload = event["inputs"]["aBinding"]
    assert upload["dtype"] == "float32" and upload["shape"] == [6]
    assert upload["values"][:2] == np.float32([1, 2]).view(np.uint32).tolist()
    # Floating and complex results are checked by package and native identity.
    assert elementwise_evidence.audit_event(event, native=False) == 3
    words, _, _ = run(
        monkeypatch,
        tmp_path,
        target,
        "vv_Equalcomplex64",
        {"a": complex_words(a), "b": complex_words(b)},
        compute=numpy_complex,
    )
    assert words == [0, 0, 1]
    words, _, _ = run(
        monkeypatch,
        tmp_path,
        target,
        "v_Realcomplex64float32",
        {"in": complex_words(a)},
        compute=numpy_complex,
    )
    assert words == np.float32([1, -0.5, 3]).view(np.uint32).tolist()
    words, _, _ = run(
        monkeypatch,
        tmp_path,
        target,
        "v_Conjugatecomplex64complex64",
        {"in": complex_words(a)},
        compute=numpy_complex,
        donate=True,
    )
    assert words == complex_words(np.conj(np.complex64(a)))
    with pytest.raises(RuntimeError, match="guard"):
        run(
            monkeypatch,
            tmp_path,
            target,
            "vv_Multiplycomplex64",
            {"a": complex_words(a), "b": complex_words(b)},
            compute=numpy_complex,
            fault="guard",
        )


def numpy_select(kernel, operation, source, destination, values):
    """A floating or complex selection fake; the audit never recomputes it."""
    assert (kernel, operation) == ("ternary_v", "Select")
    return [x if c else y for c, x, y in zip(values["a"], values["b"], values["c"])]


@pytest.mark.parametrize("target", TARGETS)
def test_selections_dispatch_with_a_boolean_condition(monkeypatch, tmp_path, target):
    condition = [1, 0, 1, 0]
    words, event, host = run(
        monkeypatch,
        tmp_path,
        target,
        "v_Selectint8",
        {"a": condition, "b": [-128, 5, 127, -1], "c": [1, 2, 3, 4]},
    )
    assert host.dispatch_count == 1 and words == [-128, 2, 127, 4]
    assert event["inputs"]["aBinding"] == element_storage.packet(
        "bool_", target, condition
    )
    assert elementwise_evidence.audit_event(event, native=False) == 4
    words, event, _ = run(
        monkeypatch,
        tmp_path,
        target,
        "v_Selectuint64",
        {"a": [0, 1], "b": [2**64 - 1, 7], "c": [0, 2**63]},
    )
    assert words == [0, 7]
    assert elementwise_evidence.audit_event(event, native=False) == 2
    _, event, _ = run(
        monkeypatch,
        tmp_path,
        target,
        "v_Selectint16",
        {"a": [1, 0], "b": [1, 2], "c": [3, 4]},
        fault="result",
    )
    with pytest.raises(ValueError, match="result 0 of v_Selectint16 differs"):
        elementwise_evidence.audit_event(event, native=False)
    a, b = [1 + 2j, -0.0 - 1j], [3 + 0j, 4 - 4j]
    words, event, _ = run(
        monkeypatch,
        tmp_path,
        target,
        "v_Selectcomplex64",
        {"a": [0, 1], "b": complex_words(a), "c": complex_words(b)},
        compute=numpy_select,
    )
    assert words == complex_words([3 + 0j, -0.0 - 1j])
    assert not elementwise_evidence.recomputable("v_Selectcomplex64")
    assert not elementwise_evidence.recomputable("v_Selectbfloat16")
    assert elementwise_evidence.recomputable("v_Selectint64")


def test_selection_recompute_requires_boolean_conditions():
    with pytest.raises(ValueError, match="outside its storage range"):
        elementwise_evidence.expected_words(
            "ternary_v", "Select", "int32", "int32", {"a": [2], "b": [1], "c": [0]}
        )
    assert elementwise_evidence.expected_words(
        "ternary_v", "Select", "int32", "int32", {"a": [1], "b": [-5], "c": [0]}
    ) == [-5]


def test_complex_reflection_must_be_the_packed_float_pair():
    value = element_storage.packet("complex64", "opengl", [0x3F800000])
    layout = dict(element_storage.COMPLEX_LAYOUT)
    matches = element_storage.reflected_layout_matches
    assert matches(layout, value, "complex64")
    # A struct binding never carries a real packet, nor a complex one as scalars.
    assert not matches(layout, element_storage.packet("float32", "opengl", [0]))
    assert not matches(layout, value)
    assert not matches(
        {"elementType": "float32", "elementStrideBytes": 4}, value, "complex64"
    )
    for key, changed in (
        ("componentCount", 4),
        ("elementStrideBytes", 16),
        ("physicalType", "float2"),
        ("structMembers", [*layout["structMembers"]][::-1]),
    ):
        assert not matches({**layout, key: changed}, value, "complex64")
    assert not matches(layout, {**value, "shape": [3]}, "complex64")


def test_donated_outputs_and_audit_rejections(monkeypatch, tmp_path):
    words, event, _ = run(
        monkeypatch,
        tmp_path,
        "opengl",
        "v_BitwiseInvertint8int8",
        {"in": [0, -1, 5]},
        donate=True,
    )
    assert words == [-1, 0, -6]
    _, event, _ = run(
        monkeypatch,
        tmp_path,
        "directx",
        "vv_Subtractbool_",
        {"a": [1, 1, 0], "b": [0, 1, 1]},
        fault="result",
    )
    with pytest.raises(ValueError, match="result 0 of vv_Subtractbool_"):
        elementwise_evidence.audit_event(event, native=False)
    _, event, _ = run(
        monkeypatch, tmp_path, "metal", "vv_Maximumint64", {"a": [-5], "b": [3]}
    )
    uploaded = json.loads(json.dumps(event))
    uploaded["operandValues"]["a"][0] = 4
    with pytest.raises(ValueError, match="upload differs"):
        elementwise_evidence.audit_event(uploaded, native=False)
    with pytest.raises(ValueError, match="output hash"):
        elementwise_evidence.audit_event(dict(event, outputHash="0" * 64), native=False)
    with pytest.raises(ValueError, match="metadata"):
        elementwise_evidence.audit_event(dict(event, entry="vv_Minimumint64"))


def test_vector_layout_rejections():
    array, memory, threads = native_buffers("vv_Addint8", {"a": [1, 2], "b": [3, 4]})
    supplied = {name: array[i] for i, name in enumerate(memory)}
    execution = {"workgroupCount": [2, 1, 1], "workgroupSize": [1, 1, 1]}
    validate = elementwise_dispatch.validate
    assert validate("vv_Addint8", supplied, 2, execution)["operation"] == "Add"
    with pytest.raises(ValueError, match="launch does not match"):
        validate("vv_Addint8", supplied, 2, {**execution, "workgroupCount": [3, 1, 1]})
    with pytest.raises(ValueError, match="type, extent"):
        validate("vv_Adduint8", supplied, 2, execution)
    with pytest.raises(ValueError, match="host contract"):
        validate("vv_Addint8", supplied, 65536, execution)
    with pytest.raises(ValueError, match="buffers do not match"):
        validate("v_Absint8int8", supplied, 2, execution)
    memory["size"][0] = 1
    with pytest.raises(ValueError, match="size does not match"):
        validate("vv_Addint8", supplied, 2, execution)
    memory["size"][0] = 2
    # An output that partly overlaps an input is never a donated buffer.
    shifted = runtime.Buffer(b"c", b"int8", supplied["a"].data + 1, 2, 1)
    with pytest.raises(ValueError, match="allocations overlap"):
        validate("vv_Addint8", {**supplied, "c": shifted}, 2, execution)
    donated = runtime.Buffer(b"c", b"int8", supplied["b"].data, 2, 1)
    assert validate("vv_Addint8", {**supplied, "c": donated}, 2, execution)


@pytest.mark.parametrize(
    ("operation", "dtype", "values", "message"),
    [
        ("LeftShift", "int8", {"a": [1], "b": [32]}, "counts in \\[0, 31\\]"),
        ("RightShift", "uint16", {"a": [1], "b": [-1]}, "counts in \\[0, 31\\]"),
        ("LeftShift", "int64", {"a": [1], "b": [64]}, "\\[0, 63\\]"),
        ("Divide", "uint8", {"a": [1], "b": [0]}, "nonzero divisors"),
        ("Remainder", "int16", {"a": [-32768], "b": [-1]}, "signed overflow"),
        ("Divide", "bool_", {"a": [1], "b": [0]}, "nonzero divisors"),
    ],
)
def test_undefined_operands_are_refused(operation, dtype, values, message):
    with pytest.raises(ValueError, match=message):
        elementwise_dispatch.require_defined(operation, dtype, values)


def test_defined_operands_pass():
    elementwise_dispatch.require_defined("LeftShift", "int8", {"a": [1], "b": [31]})
    elementwise_dispatch.require_defined("LeftShift", "uint64", {"a": [1], "b": [63]})
    elementwise_dispatch.require_defined("Divide", "int8", {"a": [-128], "b": [1]})
    elementwise_dispatch.require_defined("Divide", "float16", {"a": [1], "b": [0]})


def wrapped(dtype, values):
    modulus = 1 << np.iinfo(dtype).bits
    residues = np.array([int(value) % modulus for value in values], dtype=np.uint64)
    return residues.astype(dtype).tolist()


@pytest.mark.parametrize("dtype", INTEGERS)
def test_integer_results_wrap_like_numpy(dtype):
    rng = np.random.default_rng(len(dtype))
    info = np.iinfo(dtype)
    a = rng.integers(info.min, info.max, 512, dtype=dtype, endpoint=True)
    b = rng.integers(info.min, info.max, 512, dtype=dtype, endpoint=True)
    b[b == 0] = 1
    if info.min:
        b[(a == info.min) & (b == -1)] = 1
    values = {"a": a.tolist(), "b": b.tolist()}
    expected_words = elementwise_evidence.expected_words
    with np.errstate(over="ignore"):
        checks = {
            "Add": a + b,
            "Subtract": a - b,
            "Multiply": a * b,
            "Maximum": np.maximum(a, b),
            "Minimum": np.minimum(a, b),
            "Remainder": np.remainder(a, b),
            "BitwiseAnd": a & b,
            "BitwiseOr": a | b,
            "BitwiseXor": a ^ b,
            "Divide": (
                (np.abs(a.astype(object)) // np.abs(b.astype(object)))
                * np.sign(a.astype(object))
                * np.sign(b.astype(object))
            ),
        }
    for operation, expected in checks.items():
        actual = expected_words("vv_", operation, dtype, dtype, values)
        assert actual == wrapped(dtype, expected.tolist()), operation
    for operation, function in (("Less", np.less), ("Equal", np.equal)):
        actual = expected_words("vv_", operation, dtype, "bool_", values)
        assert actual == function(a, b).astype(int).tolist()
    bits = np.iinfo(dtype).bits
    counts = rng.integers(0, bits, 512).astype(dtype)
    shifts = {"a": values["a"], "b": counts.tolist()}
    left = expected_words("vv_", "LeftShift", dtype, dtype, shifts)
    right = expected_words("vv_", "RightShift", dtype, dtype, shifts)
    assert left == np.left_shift(a, counts).tolist()
    assert right == np.right_shift(a, counts).tolist()
    for operation, expected in (
        ("Negative", -a),
        ("Square", a * a),
        ("BitwiseInvert", ~a),
        ("Abs", np.abs(a) if info.min else a),
        ("Sign", np.sign(a)),
        ("Ceil", a),
    ):
        actual = expected_words("v_", operation, dtype, dtype, {"in": a.tolist()})
        assert actual == wrapped(dtype, expected.tolist()), operation


def test_promoted_shifts_powers_and_booleans():
    expected_words = elementwise_evidence.expected_words
    shifts = {"a": [1, -128, 255 - 256, 3], "b": [9, 9, 31, 7]}
    assert expected_words("vv_", "LeftShift", "int8", "int8", shifts) == [0, 0, 0, -128]
    assert expected_words("vv_", "RightShift", "int8", "int8", shifts) == [0, -1, -1, 0]
    assert expected_words(
        "vv_", "RightShift", "uint8", "uint8", {"a": [255], "b": [9]}
    ) == [0]
    powers = {"a": [3, -2, 2, 7], "b": [5, 3, 9, -1]}
    assert expected_words("vv_", "Power", "int8", "int8", powers) == [-13, -8, 0, 0]
    booleans = {"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]}
    results = {
        "Add": [0, 1, 1, 1],
        "Subtract": [0, 1, 1, 0],
        "Multiply": [0, 0, 0, 1],
        "Power": [1, 0, 1, 1],
        "Maximum": [0, 1, 1, 1],
        "LogicalAnd": [0, 0, 0, 1],
        "BitwiseXor": [0, 1, 1, 0],
        "GreaterEqual": [1, 0, 1, 1],
    }
    for operation, expected in results.items():
        assert expected_words("vv_", operation, "bool_", "bool_", booleans) == expected
    for operation, expected in (
        ("Negative", [0, 1]),
        ("Sign", [0, 1]),
        ("LogicalNot", [1, 0]),
        ("Square", [0, 1]),
    ):
        assert (
            expected_words("v_", operation, "bool_", "bool_", {"in": [0, 1]})
            == expected
        )
    with pytest.raises(ValueError, match="integer operands"):
        expected_words("vv_", "Add", "float16", "float16", {"a": [0], "b": [0]})
    with pytest.raises(ValueError, match="storage range"):
        expected_words("v_", "Abs", "int8", "int8", {"in": [128]})


def test_float_events_check_package_and_native_identity(monkeypatch, tmp_path):
    def native(event):
        seen.append(event["entry"])

    seen = []
    monkeypatch.setattr(elementwise_evidence, "audit_native_execution", native)
    package = tmp_path / "package"
    package.mkdir()
    (package / "artifact.glsl").write_bytes(b"kernel")
    artifact = {
        "packagePath": "artifact.glsl",
        "sizeBytes": 6,
        "hash": {"algorithm": "sha256", "value": hashlib.sha256(b"kernel").hexdigest()},
    }
    cache = SimpleNamespace(get=lambda entry: ({"artifact": artifact}, package))
    event = {
        "entry": "vv_Equalfloat16",
        "target": "opengl",
        "threads": 2,
        "workgroupCount": [2, 1, 1],
        "workgroupSize": [1, 1, 1],
        "dispatchVersion": 3,
        "artifact": artifact,
        "packageRoot": str(package),
        "elementwiseMetadata": {
            "kernel": "vv_",
            "operation": "Equal",
            "source": "float16",
            "destination": "bool_",
        },
        "details": {
            "request": {
                "target": "opengl",
                "entryPoint": "main",
                "dispatch": {"workgroupCount": [2, 1, 1], "workgroupSize": [1, 1, 1]},
            }
        },
    }
    assert not elementwise_evidence.recomputable("vv_Equalfloat16")
    assert not elementwise_evidence.recomputable("vv_Addcomplex64")
    assert not elementwise_evidence.recomputable("v_Realcomplex64float32")
    # FP8 conversions produce or read floats; the upstream tests compare them.
    assert not elementwise_evidence.recomputable("v_ToFP8float32uint8")
    assert not elementwise_evidence.recomputable("v_FromFP8uint8bfloat16")
    assert elementwise_evidence.audit_event(event, cache) == 2
    assert seen == ["vv_Equalfloat16"]
    (package / "artifact.glsl").write_bytes(b"change")
    with pytest.raises(ValueError, match="changed after translation"):
        elementwise_evidence.audit_event(event, cache)


def test_runtime_prefers_installed_families_over_lazy_vectors():
    host = SimpleNamespace(descriptors={"vv_Addint32": {}}, elementwise=object())
    lazy = runtime.HostRuntime.lazy_elementwise
    assert lazy(host, "vv_Addint8") and lazy(host, "v_Sinfloat16float16")
    assert not lazy(host, "vv_Addint32")
    assert not lazy(host, "v_copyint8float32")
    # The selection family installs float32, 32-bit integer and Boolean selections.
    assert lazy(host, "v_Selectint64") and lazy(host, "v_Selectcomplex64")
    assert not lazy(host, "v_Selectfloat32") and not lazy(host, "v_Selectbool_")
    host.elementwise = None
    assert not lazy(host, "vv_Addint8")


class InlinePool:
    """A process pool stand-in that runs each batch in this process."""

    def __init__(self, jobs):
        self.jobs = jobs

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def map(self, function, batches):
        return [function(batch) for batch in batches]


@pytest.mark.parametrize("fault", [None, "missing"])
def test_warm_translates_every_entry_in_parallel_batches(monkeypatch, tmp_path, fault):
    requested = []

    class Cache:
        def __init__(self, root, directory, target):
            assert directory == (tmp_path / "packages" / "elementwise").resolve()
            assert target == "opengl"

        def get(self, entry):
            requested.append(entry)
            return {"artifact": {"hash": {"value": entry[::-1]}}}, None

    monkeypatch.setattr(elementwise_warm, "ElementwisePackageCache", Cache)
    monkeypatch.setattr(elementwise_warm, "ProcessPoolExecutor", InlinePool)
    if fault == "missing":
        monkeypatch.setattr(
            elementwise_warm,
            "_translate",
            lambda job: [(entry, "0") for entry in job[3][1:]],
        )
        with pytest.raises(ValueError, match="every entry"):
            elementwise_warm.warm(tmp_path, tmp_path / "packages", "opengl", 3)
        return
    summary = tmp_path / "summary.json"
    elementwise_warm.main(
        [
            "--mlx-root",
            str(tmp_path),
            "--target",
            "opengl",
            "--packages",
            str(tmp_path / "packages"),
            "--jobs",
            "3",
            "--summary",
            str(summary),
        ]
    )
    assert sorted(requested) == sorted(elementwise_packages.ENTRIES)
    record = json.loads(summary.read_text())
    assert record["entries"] == 430 and record["target"] == "opengl"
    assert record["artifacts"]["vv_Addint8"] == "vv_Addint8"[::-1]


def test_warm_refuses_unknown_targets_and_job_counts(tmp_path):
    with pytest.raises(ValueError, match="target"):
        elementwise_warm.warm(tmp_path, tmp_path, "vulkan", 2)
    for jobs in (0, 17, 2.0):
        with pytest.raises(ValueError, match="1 to 16 jobs"):
            elementwise_warm.warm(tmp_path, tmp_path, "metal", jobs)
