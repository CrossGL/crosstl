"""Bulk runtime value helpers agree with the element-wise paths they skip."""

import copy
import math
import random
import re
import struct
from collections.abc import Sequence

import pytest

from crosstl.project import native_loader_dispatch, native_runtime_drivers
from crosstl.project.runtime_value_encoding import (
    BFLOAT16_BITS,
    FLOAT16_BITS,
    FLOAT32_BITS,
    encoded_storage,
    storage_bytes,
    storage_words,
    validate_value_encoding,
)
from crosstl.project.runtime_values import (
    copy_values,
    exact_types_within,
    flatten_scalars,
    flatten_values,
    integer_storage,
    pack_scalars,
    scalar_types,
    scalars_fit,
)


class Words(list):
    """A list subclass, which the bulk paths leave to the general code."""


class Number(float):
    pass


def element_flatten(value, keep_none=False):
    if value is None:
        return [None] if keep_none else []
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return [value]
    return [item for part in value for item in element_flatten(part, keep_none)]


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        [1, 2.5, True, "nan", b"x", bytearray(b"y")],
        [[1, [2, None]], (3, 4.0), None, "ab"],
        (Words([1, 2]), [Number(1.5)]),
        "text",
        7,
    ],
)
@pytest.mark.parametrize("keep_none", [False, True])
def test_flatten_values_matches_the_recursive_definition(value, keep_none):
    assert flatten_values(value, keep_none=keep_none) == element_flatten(
        value, keep_none
    )


@pytest.mark.parametrize(
    "values",
    [[1, 2.0, "nan", None, True], [[1], 2], (1, 2), [bytearray(b"a")], Words([1])],
)
def test_copy_values_matches_deepcopy_and_never_shares_mutable_parts(values):
    copied = copy_values(values)
    assert copied == copy.deepcopy(values) and type(copied) is type(values)
    if isinstance(values, list):
        assert copied is not values
    nested = [item for item in values if isinstance(item, (list, bytearray))]
    for original, duplicate in zip(nested, [i for i in copied if i in nested]):
        assert duplicate is not original


def element_fits(values, dtype):
    for value in values:
        if dtype == "bool":
            ok = type(value) is bool
        elif dtype.startswith(("int", "uint")):
            width = int(dtype.removeprefix("u").removeprefix("int"))
            low, high = (
                (0, (1 << width) - 1)
                if dtype.startswith("u")
                else (-(1 << (width - 1)), (1 << (width - 1)) - 1)
            )
            ok = isinstance(value, int) and not isinstance(value, bool)
            ok = ok and low <= value <= high
        else:
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
            try:
                ok = ok and math.isfinite(float(value))
                if ok:
                    struct.pack("<e" if dtype == "float16" else "<f", float(value))
            except (OverflowError, struct.error, ValueError):
                ok = False
        if not ok:
            return False
    return True


@pytest.mark.parametrize(
    "dtype, values",
    [
        ("bool", [True, False]),
        ("bool", [True, 1]),
        ("int8", [-128, 127]),
        ("int8", [128]),
        ("uint32", [0, 2**32 - 1]),
        ("uint32", [-1]),
        ("int64", [2**63]),
        ("int32", [True]),
        ("float32", [0, 1.5, -2.0e38]),
        ("float32", [1e39]),
        ("float32", [math.inf]),
        ("float32", ["nan"]),
        ("float32", [10**400]),
        ("float32", [False]),
        ("float16", [65504.0, -0.5]),
        ("float16", [70000.0]),
        ("float16", [math.nan]),
        ("bfloat16", [1.0]),
        ("float32", []),
    ],
)
def test_scalars_fit_only_accepts_what_the_element_checks_accept(dtype, values):
    fits = scalars_fit(values, dtype)
    if fits:
        assert element_fits(values, dtype)
    if dtype != "bfloat16" and all(type(v) in (int, float, bool) for v in values):
        # For exact built-in scalars the bulk check decides alone.
        assert fits == element_fits(values, dtype)


def test_subclassed_scalars_fall_back_to_the_element_checks():
    assert not scalars_fit([Number(1.0)], "float32")
    native_loader_dispatch._validate_buffer_values(
        [Number(1.0)], dtype="float32", path="$", name="x", role="input"
    )


@pytest.mark.parametrize(
    "dtype, values",
    [("float32", [1.0, 1e39]), ("int32", [2**31]), ("bool", [1]), ("float16", [7e4])],
)
def test_rejected_values_report_the_first_offending_index(dtype, values):
    with pytest.raises(native_loader_dispatch.NativeLoaderDispatchError) as error:
        native_loader_dispatch._validate_buffer_values(
            values, dtype=dtype, path="$.v", name="x", role="input"
        )
    assert error.value.path == f"$.v[{len(values) - 1}]"


@pytest.mark.parametrize(
    "dtype, values",
    [
        ("float32", [0.1, -3, 2.5e-41, 1e38]),
        ("float32", ["nan", 1.0, "+infinity"]),
        ("float32", [True, 2.0]),
        ("float16", [0.1, -3, 6e-8]),
        ("float16", ["-infinity", 0.5]),
    ],
)
def test_bulk_float_packing_matches_element_packing(dtype, values):
    special = {
        "float32": {
            "nan": 0x7FC00000,
            "+infinity": 0x7F800000,
            "-infinity": 0xFF800000,
        },
        "float16": {"nan": 0x7E00, "+infinity": 0x7C00, "-infinity": 0xFC00},
    }[dtype]
    expected = b"".join(
        (
            struct.pack("<I" if dtype == "float32" else "<H", special[item])
            if isinstance(item, str)
            else struct.pack("<f" if dtype == "float32" else "<e", item)
        )
        for item in values
    )
    packed = native_runtime_drivers._pack_values(
        values, dtype, expected_count=len(values)
    )
    assert packed == expected


@pytest.mark.parametrize("width", [16, 32])
def test_storage_words_bulk_path_keeps_range_checks(width):
    assert storage_words([0, (1 << width) - 1], width=width) == [0, (1 << width) - 1]
    assert storage_words([[1], (2, 3)], width=width) == [1, 2, 3]
    for invalid in ([1 << width], [-1], [True], [1.0]):
        with pytest.raises(ValueError, match="storage values"):
            storage_words(invalid, width=width)


def test_exact_types_within_matches_exact_types_only():
    ints = frozenset({int})
    assert exact_types_within([], ints)
    assert exact_types_within([0, -1, 2**70], ints)
    assert not exact_types_within([1, True], ints)
    assert not exact_types_within([1, 1.0], ints)
    assert not exact_types_within([Number(1.0)], frozenset({float}))
    assert exact_types_within((value for value in (1.0, 2.5)), frozenset({float}))


@pytest.mark.parametrize(
    "dtype, values, accepted",
    [
        ("bool", [True, False], True),
        ("bool", [True, 1], False),
        ("int8", [-128, 0, 127], True),
        ("int8", [128], False),
        ("uint8", [-1], False),
        ("uint16", [65535, True], False),
        ("int16", [1.0], False),
    ],
)
def test_bulk_packing_keeps_bool_and_narrow_integer_checks(dtype, values, accepted):
    if not accepted:
        with pytest.raises(native_runtime_drivers.RuntimeExecutorUnavailable):
            native_runtime_drivers._pack_values(
                values, dtype, expected_count=len(values)
            )
        return
    packed = native_runtime_drivers._pack_values(
        values, dtype, expected_count=len(values)
    )
    code = native_runtime_drivers._dtype_format(dtype)
    assert packed == struct.pack("<" + code * len(values), *values)


def test_unpacking_rejects_noncanonical_bool_bytes_and_keeps_order():
    assert native_runtime_drivers._unpack_values(b"\x01\x00\x01", "bool") == [
        True,
        False,
        True,
    ]
    with pytest.raises(native_runtime_drivers.RuntimeAdapterDispatchError):
        native_runtime_drivers._unpack_values(b"\x01\x02", "bool")
    words = [0, 1, 0xFFFFFFFF]
    payload = struct.pack("<3I", *words)
    assert native_runtime_drivers._unpack_values(payload, "uint32") == words
    assert native_runtime_drivers._unpack_values(b"", "float32") == []


FIT_CASES = [
    ("bool", [True, False]),
    ("bool", [True, 1]),
    ("int8", [-128, 127]),
    ("int8", [128]),
    ("uint8", [0, 255]),
    ("int16", [-32768, 32767]),
    ("uint16", [65536]),
    ("int32", [-(2**31), 2**31 - 1]),
    ("uint32", [0, 2**32 - 1]),
    ("int64", [-(2**63), 2**63 - 1]),
    ("uint64", [2**64 - 1]),
    ("uint64", [2**64]),
    ("int32", [True]),
    ("float32", [0, 1.5, -2.0e38]),
    ("float32", [1e39]),
    ("float32", ["nan"]),
    ("float16", [65504.0, -0.5]),
    ("float16", [70000.0]),
    ("bfloat16", [1.0]),
    ("float32", []),
    ("int32", [Words([1])]),
    ("float32", [1e308, 1e308]),
    ("float32", [3.0e38, 3.0e38, -math.inf]),
    ("float16", [math.inf, -math.inf]),
    ("float32", [2**63, 0.5]),
    ("float32", [10**400, 0.5]),
]


def test_float_fits_decide_as_the_element_checks_on_mixed_lists():
    rng = random.Random(5)
    pool = [0, 1, -1, 0.5, 1e38, -3.4e38, 3.5e38, 65504.0, 70000.0, math.inf]
    pool += [-math.inf, math.nan, 10**308, 10**400, 2**63, 1e-45, 6e-8, -0.0, 1e308]
    for _ in range(4000):
        values = [rng.choice(pool) for _ in range(rng.randint(0, 6))]
        for dtype in ("float16", "float32"):
            assert scalars_fit(values, dtype) == element_fits(values, dtype), values


@pytest.mark.parametrize("dtype, values", FIT_CASES)
def test_pack_scalars_packs_exactly_what_scalars_fit_accepts(dtype, values):
    packed = pack_scalars(values, dtype)
    assert (packed is not None) == scalars_fit(values, dtype)
    if packed is not None:
        code = native_runtime_drivers._dtype_format(dtype)
        assert packed == struct.pack("<" + code * len(values), *values)


@pytest.mark.parametrize("dtype, values", FIT_CASES)
def test_precomputed_types_decide_as_a_fresh_scan(dtype, values):
    types = scalar_types(values)
    assert scalars_fit(values, dtype, types=types) == scalars_fit(values, dtype)
    assert pack_scalars(values, dtype, types=types) == pack_scalars(values, dtype)


def test_scalar_types_are_exact_and_drive_flattening_and_copies():
    assert scalar_types([1, True, 1.0, Number(2.0)]) == {int, bool, float, Number}
    assert scalar_types([]) == frozenset()
    nested = [[1, 2], [3.0]]
    types = scalar_types(nested)
    assert types == {list}
    copied = copy_values(nested, types=types)
    assert copied == nested and copied[0] is not nested[0]
    assert flatten_values(nested, types=types) == [1, 2, 3.0]
    flat, flat_types = flatten_scalars(nested, types=types)
    assert flat == [1, 2, 3.0] and flat_types == {int, float}
    values = [1, 2.5, "nan"]
    flat, flat_types = flatten_scalars(values)
    assert flat == values and flat is not values and flat_types == {int, float, str}
    flat, flat_types = flatten_scalars((1, (2,)))
    assert flat == [1, 2] and flat_types == {int}
    assert flatten_scalars(None) == ([], frozenset())


def test_typed_runtime_values_check_their_values_through_one_scan():
    value = native_loader_dispatch._typed_value(
        {
            "name": "x",
            "kind": "buffer",
            "dtype": "float32",
            "shape": [3],
            "values": [1, 2.5, -3.0],
        },
        key="x",
        role="input",
        path="$.inputValues.x",
    )
    assert value.values == [1, 2.5, -3.0]
    for values, field in (([1.0, 1e39, 2.0], "[1]"), ([1.0, 2.0], "")):
        with pytest.raises(native_loader_dispatch.NativeLoaderDispatchError) as error:
            native_loader_dispatch._typed_value(
                {
                    "name": "x",
                    "kind": "buffer",
                    "dtype": "float32",
                    "shape": [3],
                    "values": values,
                },
                key="x",
                role="input",
                path="$.inputValues.x",
            )
        assert error.value.path == "$.inputValues.x.values" + field


INTEGER_DTYPES = tuple(
    f"{sign}int{width}" for sign in ("", "u") for width in (8, 16, 32, 64)
)


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_integer_storage_packs_in_range_ints_and_refuses_the_rest(dtype):
    code = native_runtime_drivers._dtype_format(dtype)
    width = struct.calcsize(code) * 8
    if dtype.startswith("int"):
        low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
    else:
        low, high = 0, (1 << width) - 1
    rng = random.Random(dtype)
    accepted = [[], [low, high], [high, 0, low], [0] * 3]
    accepted.append([rng.randint(low, high) for _ in range(500)])
    for values in accepted:
        expected = struct.pack(f"<{len(values)}{code}", *values)
        assert integer_storage(values, dtype) == expected
        assert integer_storage(tuple(values), dtype) == expected
    # A bytes-like sequence is read item by item, as ``struct`` reads it.
    assert integer_storage(b"\x01\x02", dtype) == struct.pack(f"<2{code}", 1, 2)
    for values in ([low - 1], [high + 1], [0, high + 1, 0], [low, low - 1]):
        assert integer_storage(values, dtype) is None
        assert pack_scalars(values, dtype) is None
        assert not scalars_fit(values, dtype)


@pytest.mark.parametrize("width", [16, 32])
def test_storage_bytes_packs_exactly_what_storage_words_accepts(width):
    code = "H" if width == 16 else "I"
    for values in ([], [0, (1 << width) - 1], [[1], (2, 3)], [5, 0] * 300):
        words = storage_words(values, width=width)
        expected = struct.pack(f"<{len(words)}{code}", *words)
        assert storage_bytes(values, width=width) == expected
        types = scalar_types(values)
        assert storage_bytes(values, width=width, types=types) == expected
        assert storage_words(values, width=width, types=types) == words
    for invalid in ([1 << width], [-1], [True], [1.0], [[1 << width]], 7.0, [0, -1]):
        with pytest.raises(ValueError) as expected:
            storage_words(invalid, width=width)
        message = re.escape(str(expected.value))
        with pytest.raises(ValueError, match=message):
            storage_bytes(invalid, width=width)
        if type(invalid) is list:
            with pytest.raises(ValueError, match=message):
                storage_bytes(invalid, width=width, types=scalar_types(invalid))
    with pytest.raises(ValueError, match="width"):
        storage_bytes([1], width=8)


@pytest.mark.parametrize(
    "encoding, dtype, values",
    [
        (FLOAT32_BITS, "float32", [0, 0x7FC00000, 2**32 - 1]),
        (FLOAT16_BITS, "float16", [0x7E00, 1]),
        (BFLOAT16_BITS, "bfloat16", [[0x3F80], [0]]),
        (FLOAT32_BITS, "float16", [0]),
        ("other", "float32", [0]),
        (["unhashable"], "float32", [0]),
        (FLOAT16_BITS, "float16", [1 << 16]),
        (BFLOAT16_BITS, "bfloat16", [-1]),
        (FLOAT32_BITS, "float32", [True]),
        (FLOAT32_BITS, "float32", [0.0]),
    ],
)
def test_encoded_storage_checks_and_packs_as_validation_and_packing_do(
    encoding, dtype, values
):
    types = scalar_types(values)
    try:
        validate_value_encoding(encoding, dtype, values)
    except ValueError as error:
        message = re.escape(str(error))
        for known in (None, types):
            with pytest.raises(ValueError, match=message):
                validate_value_encoding(encoding, dtype, values, types=known)
            with pytest.raises(ValueError, match=message):
                encoded_storage(encoding, dtype, values, types=known)
        with pytest.raises(
            native_runtime_drivers.RuntimeExecutorUnavailable, match=message
        ):
            native_runtime_drivers._pack_values(
                values, dtype, expected_count=len(values), encoding=encoding
            )
        return
    validate_value_encoding(encoding, dtype, values, types=types)
    words = storage_words(values, width=32 if dtype == "float32" else 16)
    code = "I" if dtype == "float32" else "H"
    expected = struct.pack(f"<{len(words)}{code}", *words)
    assert encoded_storage(encoding, dtype, values) == expected
    assert encoded_storage(encoding, dtype, values, types=types) == expected
    packed = native_runtime_drivers._pack_values(
        values, dtype, expected_count=len(words), encoding=encoding
    )
    assert packed == expected
