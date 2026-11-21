"""Prove scaled_dot_product_attention_vjp.metal's 33 entries at the current pin.

``ScaledDotProductAttentionVJP::eval_gpu`` runs ``sdpa_vjp_odo`` (each query
row's float32 dot product of the forward output and its cotangent, summed
with ``simd_sum``) and, per key and query block, ``sdpa_vjp_ds`` (the block's
probabilities and score gradients, with causal and fully masked rows set to
zero) and ``sdpa_vjp_reduce_add`` / ``sdpa_vjp_reduce_set`` (grouped-head
gradient sums into float32 accumulators and the gradient outputs). The
file instantiates the row product at eight head dimensions and every kernel
for float, float16_t and bfloat16_t: 24 + 3 + 6 entries.

Each entry is translated with the host's recipe
(``sdpa_vjp_packages.project_config``), pinned per target and run natively
on launches ``scaled_dot_product_attention.cpp`` makes, checked by
``sdpa_vjp_layout.plan_launch``: partial threadgroups on both axes, a causal
block across the diagonal with a nonzero ``diag_off``, a fully masked row and
reductions into one window of a larger accumulator. Every output is checked
against a double-precision reference within bounds derived from each
kernel's float32 arithmetic, every guard and every accumulator row outside
the launch's window must come back unchanged and no input may change. On
Metal every buffer must also equal the unchanged upstream kernel's, word for
word.
"""

import hashlib
import json
import math
import re
import shutil
import struct
import tempfile
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from crosstl.project import (
    build_native_loader_dispatch_request,
    translate_project,
    validate_project_report,
)
from demos.integrations.mlx.portable_host import sdpa_vjp_dispatch as dispatch
from demos.integrations.mlx.portable_host import sdpa_vjp_layout as layout
from demos.integrations.mlx.portable_host import sdpa_vjp_packages as packages
from demos.integrations.mlx.portable_host.element_storage import (
    GUARD_COUNT,
    GUARDS,
    unpack,
)
from demos.integrations.mlx.portable_host.sdpa_dispatch import POISON
from demos.integrations.mlx.tests.kernels.current_gate import (
    assert_identical,
    compact_contract_text,
    parse_pin,
    require_pinned_checkout,
)
from demos.integrations.mlx.tests.kernels.test_current_arg_reduce import (
    _metal_library,
    _run,
)
from demos.integrations.mlx.tests.kernels.test_current_complex_power import (
    MLX_COMMIT,
    ROOT,
)
from demos.integrations.mlx.tests.pins import check_generated_artifact
from tests.runtime_helpers import _prepare_native_package, _validate_generated_artifact
from tests.test_translator.test_native_loader_dispatch_integration import _executor

REQUIRE_ENV = "CROSTL_REQUIRE_MLX_CURRENT_SDPA_VJP"
CONTRACT_PATH = (
    ROOT / "demos/integrations/mlx/contracts/sdpa_vjp.current-tree.translation.json"
)
TARGETS = ("metal", "opengl", "directx")
SOURCE = layout.SOURCE
SOURCE_SHA256 = "e4939faab6c21b0e733d60bb828338696ecafe737db4bbe63e819bd69f186661"
HOST_SOURCE = "mlx/backend/metal/scaled_dot_product_attention.cpp"
HOST_SHA256 = "c8a29b3ee6e5e0f29f247f36c57bcc54c0f99e0f6358de6195889c258a05676f"
DTYPES = tuple(layout.TYPE_NAMES)
ENTRIES = (
    *(
        layout.odo_entry(dtype, dv)
        for dtype in DTYPES
        for dv in sorted(layout.ODO_DIMS)
    ),
    *(layout.ds_entry(dtype) for dtype in DTYPES),
    *(
        layout.reduce_entry(dtype, accumulate)
        for dtype in DTYPES
        for accumulate in (True, False)
    ),
)
SIMD_WIDTH = layout.SIMD_SIZE
# Unit roundoff of the kernels' float32 arithmetic.
UNIT = 2.0**-24
# (significand bits, minimum normal exponent) of each storage type.
FORMATS = {"float32": (24, -126), "float16": (11, -14), "bfloat16": (8, -126)}
# exp's relative error: GLSL bounds it by 3 + 2|x| ULP and Direct3D's exp2 by
# 2^-21, after scaling the argument by log2(e).
EXP_ULPS = (3, 2)
EXP2_RELATIVE = 2.0**-21
NAN_WORDS = {"float32": 0x7FC00000, "float16": 0x7E00, "bfloat16": 0x7FC0}

ODO_CASES = (
    {"label": "rows-6", "heads": 2, "qL": 3},
    {"label": "rows-15", "heads": 3, "qL": 5},
)
# Blocks as sdpa_vjp_blocked launches them: the first leaves partial
# threadgroups on both axes and kills one row (lse = -inf); the second is a
# causal block across the diagonal of a query suffix (diag_off = kL - qL).
DS_CASES = (
    {
        "label": "block",
        "heads": 2,
        "qL": 20,
        "bq": 12,
        "bk": 40,
        "i0": 8,
        "j0": 24,
        "scale": 0.125,
        "diag_off": 0,
        "causal": 0,
        "dead": (1, 3),
    },
    {
        "label": "causal",
        "heads": 1,
        "qL": 9,
        "bq": 9,
        "bk": 33,
        "i0": 0,
        "j0": 4,
        "scale": 0.5,
        "diag_off": 5,
        "causal": 1,
        "dead": None,
    },
)
# A grouped-head reduction into rows 4..13 of 16, and a whole one.
REDUCE_CASES = (
    {
        "label": "window",
        "heads": 2,
        "group": 3,
        "rows": 10,
        "dim": 40,
        "acc_rows": 16,
        "row_off": 4,
    },
    {
        "label": "whole",
        "heads": 1,
        "group": 1,
        "rows": 8,
        "dim": 64,
        "acc_rows": 8,
        "row_off": 0,
    },
)
CASES = {"odo": ODO_CASES, "ds": DS_CASES, "reduce": REDUCE_CASES}


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Storage words and references
# ---------------------------------------------------------------------------


def to_words(dtype, values):
    """Round float values to ``dtype``'s storage words, to nearest even."""
    single = np.ascontiguousarray(values, dtype=np.float64).astype(np.float32)
    return round_words(dtype, single.view(np.uint32).ravel().tolist())


def round_words(dtype, words):
    """Round float32 words to ``dtype``'s storage words, to nearest even."""
    bits = np.array(words, dtype=np.uint64)
    if dtype == "float32":
        return [int(word) for word in bits]
    if dtype == "float16":
        single = bits.astype(np.uint32).view(np.float32)
        return single.astype(np.float16).view(np.uint16).astype(int).tolist()
    assert not np.any((bits & 0x7F800000) == 0x7F800000), "bfloat16 of non-finite"
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(int).tolist()


def from_words(dtype, words):
    """The float64 values of storage words."""
    bits = np.array(words, dtype=np.uint64)
    if dtype == "float32":
        return bits.astype(np.uint32).view(np.float32).astype(np.float64)
    if dtype == "float16":
        return bits.astype(np.uint16).view(np.float16).astype(np.float64)
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def ulp(dtype, magnitude):
    """The spacing of ``dtype`` at each magnitude, subnormals included."""
    bits, emin = FORMATS[dtype]
    magnitude = np.abs(np.asarray(magnitude, dtype=np.float64))
    with np.errstate(divide="ignore"):
        exponent = np.floor(np.log2(np.where(magnitude > 0, magnitude, 1.0)))
    exponent = np.where(magnitude > 0, np.maximum(exponent, emin), emin)
    return np.exp2(exponent - (bits - 1))


def gamma(count):
    """Higham's bound on ``count`` float32 roundings, in any order."""
    return count * UNIT / (1 - count * UNIT)


def _rng(entry, label):
    return np.random.default_rng(zlib.crc32(f"{entry}:{label}".encode()))


def _sample(dtype, rng, count, low, high):
    return to_words(dtype, rng.uniform(low, high, count))


def _counts(grid, workgroup):
    return [-(-extent // width) for extent, width in zip(grid, workgroup)]


def _tile_words(case):
    scale = struct.unpack("<I", struct.pack("<f", case["scale"]))[0]
    return [scale if name == "scale" else case[name] for name in layout.TILE_FIELDS]


def _arrays(entry):
    return [
        name
        for name in layout.parameters(entry)
        if name not in layout.scalars(entry) + layout.records(entry)
    ]


def _odo_case(entry, info, case):
    rng = _rng(entry, case["label"])
    rows, dv = case["heads"] * case["qL"], info["Dv"]
    data = {
        name: _sample(info["dtype"], rng, rows * dv, -2.0, 2.0)
        for name in ("o", "cot_o")
    }
    terms = (
        from_words(info["dtype"], data["o"]) * from_words(info["dtype"], data["cot_o"])
    ).reshape(rows, dv)
    grid = [SIMD_WIDTH, case["qL"], case["heads"]]
    reference = {
        "odo": {
            "values": terms.sum(axis=1),
            # Dv products and Dv - 1 sums, in whatever order the target adds.
            "bound": gamma(dv + 1) * np.abs(terms).sum(axis=1),
        }
    }
    spans = {"o": rows * dv, "cot_o": rows * dv, "odo": rows}
    return data, {"qL": case["qL"]}, {}, grid, spans, reference


def _ds_reference(dtype, case, S, dP, lse, odo):
    """Each element's probability and score gradient, the elements the
    kernel must zero, and their float32 error bounds."""
    heads, bq, bk = case["heads"], case["bq"], case["bk"]
    scale = case["scale"]
    rows = np.arange(bq).reshape(1, bq, 1)
    cols = np.arange(bk).reshape(1, 1, bk)
    qi = (np.arange(heads).reshape(heads, 1, 1) * case["qL"]) + case["i0"] + rows
    level, delta = lse[qi], odo[qi]
    dead = np.broadcast_to(np.isneginf(level), (heads, bq, bk))
    masked = dead | (
        (case["causal"] != 0)
        & (case["j0"] + cols > case["i0"] + rows + case["diag_off"])
    )
    product = S * scale
    with np.errstate(invalid="ignore"):
        x = np.where(masked, 0.0, product - level)
    probability = np.where(masked, 0.0, np.exp(x))
    gradient = probability * (dP - delta) * scale
    # The argument's two roundings move exp by their size; exp itself errs
    # by GLSL's ULP bound or Direct3D's exp2 bound plus its scaled argument.
    relative = (
        2 * UNIT * (np.abs(product) + np.abs(x))
        + EXP2_RELATIVE
        + (EXP_ULPS[0] + EXP_ULPS[1] * np.abs(x)) * 2.0**-23
    )
    return masked, {
        "P": {"values": probability, "relative": relative, "dtype": dtype},
        # One subtraction and two products more.
        "dS": {"values": gradient, "relative": relative + 3 * UNIT, "dtype": dtype},
    }


def _ds_case(entry, info, case):
    rng = _rng(entry, case["label"])
    dtype = info["dtype"]
    heads, bq, bk = case["heads"], case["bq"], case["bk"]
    block, total = heads * bq * bk, heads * case["qL"]
    data = {
        "S": _sample(dtype, rng, block, -8.0, 8.0),
        "dP": _sample(dtype, rng, block, -3.0, 3.0),
        "odo": to_words("float32", rng.uniform(-2.0, 2.0, total)),
    }
    S = from_words(dtype, data["S"]).reshape(heads, bq, bk)
    product = from_words("float32", to_words("float32", S * case["scale"]))
    product = product.reshape(heads, bq, bk)
    lse = rng.uniform(-1.0, 1.0, total)
    for head in range(heads):
        for row in range(bq):
            # A logsumexp at or above the row's largest scaled score.
            lse[head * case["qL"] + case["i0"] + row] = product[head, row].max() + (
                rng.uniform(0.0, 2.0)
            )
    if case["dead"] is not None:
        head, row = case["dead"]
        lse[head * case["qL"] + case["i0"] + row] = -np.inf
    data["lse"] = to_words("float32", lse)
    masked, reference = _ds_reference(
        dtype,
        case,
        S,
        from_words(dtype, data["dP"]).reshape(heads, bq, bk),
        from_words("float32", data["lse"]),
        from_words("float32", data["odo"]),
    )
    reference["masked"] = masked
    spans = {**dict.fromkeys(("S", "dP", "dS", "P"), block), "lse": total, "odo": total}
    return data, {}, {"p": _tile_words(case)}, [bk, bq, heads], spans, reference


def _reduce_case(entry, info, case):
    rng = _rng(entry, case["label"])
    dtype = info["dtype"]
    heads, group, rows, dim = (case[key] for key in ("heads", "group", "rows", "dim"))
    acc_rows, row_off = case["acc_rows"], case["row_off"]
    total = heads * acc_rows * dim
    data = {
        "src": _sample(dtype, rng, heads * group * rows * dim, -4.0, 4.0),
        "acc": to_words("float32", rng.uniform(-8.0, 8.0, total)),
        "out": _sample(dtype, rng, total, -1.0, 1.0),
    }
    src = from_words(dtype, data["src"]).reshape(heads, group, rows, dim)
    start = from_words("float32", data["acc"]).reshape(heads, acc_rows, dim)
    start = start[:, row_off : row_off + rows]
    accumulate = float(info["accumulate"])
    reference = {
        "acc": {
            "values": accumulate * start + src.sum(axis=1),
            # group additions in the kernel's order, one more to accumulate.
            "bound": (
                gamma(group + 1)
                * (accumulate * np.abs(start) + np.abs(src).sum(axis=1))
            ),
        }
    }
    scalars = {name: case[name] for name in layout.REDUCE_SCALARS}
    spans = {"src": len(data["src"]), "acc": total, "out": total}
    return data, scalars, {}, [dim, rows, heads], spans, reference


def build_case(entry, case):
    """One launch: storage words, scalars, record words, the checked plan
    and the reference."""
    info = layout.signature(entry)
    builder = {"odo": _odo_case, "ds": _ds_case, "reduce": _reduce_case}[info["kind"]]
    data, values, words, grid, spans, reference = builder(entry, info, case)
    workgroup = list(info["workgroupSize"])
    execution = {"workgroupCount": _counts(grid, workgroup), "workgroupSize": workgroup}
    metadata = layout.plan_launch(
        entry, spans, values, words, spans[layout.outputs(entry)[-1]], execution
    )
    directions = layout.directions(entry)
    storage = {}
    for name in _arrays(entry):
        dtype = layout.parameter_type(entry, name)
        if directions[name] == 1:
            storage[name] = [GUARDS[dtype]] * (spans[name] + GUARD_COUNT)
        elif directions[name] == layout.INOUT:
            storage[name] = data[name] + [GUARDS[dtype]] * GUARD_COUNT
        else:
            storage[name] = data[name] + [POISON[dtype]] * GUARD_COUNT
    return {
        "label": case["label"],
        "storage": storage,
        "spans": spans,
        "values": values,
        "words": words,
        "execution": execution,
        "metadata": metadata,
        "reference": reference,
    }


def _window(entry, built):
    """The accumulator elements one reduction writes, in its storage order."""
    metadata = built["metadata"]
    dim, acc_rows = metadata["dim"], metadata["accRows"]
    start, rows = metadata["rowOffset"], metadata["rows"]
    return np.array(
        [
            (head * acc_rows + start + row) * dim + column
            for head in range(metadata["heads"])
            for row in range(rows)
            for column in range(dim)
        ]
    )


def check_outputs(entry, built, outputs):
    """Check each output's logical words against the reference; return the
    worst error as a fraction of its bound."""
    info = layout.signature(entry)
    reference = built["reference"]
    worst = 0.0

    def within(name, actual, expected, bound):
        nonlocal worst
        error = np.abs(actual - expected)
        assert np.all(np.isfinite(actual)), (entry, name, "non-finite output")
        assert np.all(error <= bound), (entry, name, float(np.max(error - bound)))
        worst = max(worst, float(np.max(error / np.where(bound > 0, bound, 1.0))))

    if info["kind"] == "odo":
        item = reference["odo"]
        within(
            "odo", from_words("float32", outputs["odo"]), item["values"], item["bound"]
        )
    elif info["kind"] == "ds":
        masked = reference["masked"].ravel()
        for name in ("P", "dS"):
            item = reference[name]
            actual = from_words(info["dtype"], outputs[name])
            expected = item["values"].ravel()
            relative = item["relative"].ravel()
            # The float32 value's error, then one rounding to the storage type.
            bound = relative * np.abs(expected) + ulp(
                info["dtype"], np.abs(expected) * (1 + relative)
            )
            assert np.all(actual[masked] == 0.0), (entry, name, "masked element")
            within(name, actual[~masked], expected[~masked], bound[~masked])
    else:
        window = _window(entry, built)
        item = reference["acc"]
        acc = np.array(outputs["acc"])[window]
        within(
            "acc",
            from_words("float32", acc),
            item["values"].ravel(),
            item["bound"].ravel(),
        )
        # The gradient output is the accumulator rounded once, exactly.
        assert np.array(outputs["out"])[window].tolist() == round_words(
            info["dtype"], acc.tolist()
        ), (entry, "out")
    return worst


def ideal_outputs(entry, built):
    """The reference rounded to each output's storage, in place in the
    launch's initial storage."""
    info = layout.signature(entry)
    reference = built["reference"]
    outputs = {
        name: list(built["storage"][name][: built["spans"][name]])
        for name in layout.outputs(entry)
    }
    if info["kind"] == "odo":
        outputs["odo"] = to_words("float32", reference["odo"]["values"])
    elif info["kind"] == "ds":
        for name in ("P", "dS"):
            outputs[name] = to_words(info["dtype"], reference[name]["values"].ravel())
    else:
        window = _window(entry, built)
        acc = to_words("float32", reference["acc"]["values"].ravel())
        for index, word in zip(window, acc):
            outputs["acc"][index] = word
        for index, word in zip(window, round_words(info["dtype"], acc)):
            outputs["out"][index] = word
    return outputs


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def _expected_validation_contract():
    return {
        "workgroups": {kind: list(size) for kind, size in layout.WORKGROUPS.items()},
        # As JSON stores them: the dead row's (head, row) becomes a list.
        "cases": json.loads(json.dumps(CASES)),
        "reference": "float64 from the stored inputs",
        "bounds": {
            "odo": "gamma(Dv + 1) * sum |o * cot_o|",
            "ds": (
                "2u(|S scale| + |x|) + 2^-21 + (3 + 2|x|) 2^-23 relative for P, "
                "3u more for dS, then one ULP of the storage type; masked "
                "elements exactly zero"
            ),
            "reduce": (
                "gamma(group + 1) * (|acc| + sum |src|) for acc; out is acc "
                "rounded to nearest even, exactly"
            ),
        },
        "guards": (
            f"{GUARD_COUNT} guards after every output and every accumulator row "
            "outside the window stay untouched; inputs unchanged"
        ),
        "metalUpstreamComparison": (
            "the unchanged upstream kernel, word for word over every buffer"
        ),
    }


def expected_contract(artifacts=None):
    if artifacts is None:
        artifacts = _contract()["artifacts"]
    dispatches = sum(len(CASES[layout.signature(entry)["kind"]]) for entry in ENTRIES)
    return {
        "schemaVersion": 1,
        "kind": "crosstl-mlx-current-tree-sdpa-vjp-contract",
        "provenance": {
            "repository": "https://github.com/ml-explore/mlx.git",
            "commit": MLX_COMMIT,
            "source": SOURCE,
            "sourceSha256": SOURCE_SHA256,
            "hostSource": {"path": HOST_SOURCE, "sha256": HOST_SHA256},
            "recipe": "demos/integrations/mlx/portable_host/sdpa_vjp_packages.py",
        },
        "corpusCensus": {
            "metalUnitCount": 49,
            "entryCount": 17832,
            "discoveryDiagnosticCount": 0,
        },
        "entries": list(ENTRIES),
        "artifacts": artifacts,
        "validationContract": _expected_validation_contract(),
        "scope": {
            "instantiatedEntryCount": len(ENTRIES),
            "coveredEntryCount": len(ENTRIES),
            "artifactCountByTarget": {target: len(ENTRIES) for target in TARGETS},
            "dispatchCountByTarget": {target: dispatches for target in TARGETS},
            "completeCorpusEntryCount": 17832,
            "upstreamMlxTestSuiteExecuted": False,
            "mlxHostRuntimeRedirectionImplemented": False,
        },
    }


def test_current_mlx_sdpa_vjp_contract_is_exact():
    contract = _contract()
    assert contract == expected_contract()
    assert CONTRACT_PATH.read_text(encoding="utf-8") == compact_contract_text(contract)
    assert list(contract["artifacts"]) == list(ENTRIES)
    for pins in contract["artifacts"].values():
        assert list(pins) == list(TARGETS)
        for pin in pins.values():
            record = parse_pin(pin)
            assert re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
            assert record["sizeBytes"] > 0
    readme = (ROOT / "demos/integrations/mlx/README.md").read_text(encoding="utf-8")
    assert "sdpa_vjp.current-tree.translation.json" in readme


def test_sdpa_vjp_entries_follow_the_instantiations():
    assert len(ENTRIES) == len(set(ENTRIES)) == 33
    kinds = [layout.signature(entry)["kind"] for entry in ENTRIES]
    assert kinds.count("odo") == 24 and kinds.count("ds") == 3
    assert kinds.count("reduce") == 6
    assert set(packages.verification_entries()) <= set(ENTRIES)


@pytest.mark.parametrize("entry", ENTRIES)
def test_sdpa_vjp_cases_follow_the_host_plans(entry):
    info = layout.signature(entry)
    for case in CASES[info["kind"]]:
        built = build_case(entry, case)
        execution = built["execution"]
        assert built["metadata"]["elements"] == built["spans"]
        assert built["metadata"]["grid"][2] == case["heads"]
        for name, words in built["storage"].items():
            assert len(words) == built["spans"][name] + GUARD_COUNT
            if layout.directions(entry)[name] == 0:
                dtype = layout.parameter_type(entry, name)
                assert words[-GUARD_COUNT:] == [POISON[dtype]] * GUARD_COUNT
        if info["kind"] == "ds":
            # Partial threadgroups on both axes or a causal diagonal.
            grid = built["metadata"]["grid"]
            assert execution["workgroupCount"][0] * 32 > grid[0]
            masked = built["reference"]["masked"]
            assert masked.any() and not masked.all()
        if info["kind"] == "reduce" and case["label"] == "window":
            assert len(_window(entry, built)) < built["spans"]["acc"]
        assert check_outputs(entry, built, ideal_outputs(entry, built)) <= 1.0


def _damage(entry, built, kind):
    outputs = ideal_outputs(entry, built)
    name = layout.outputs(entry)[0]
    if layout.signature(entry)["kind"] == "ds":
        masked = built["reference"]["masked"].ravel()
        index = int(np.flatnonzero(masked if kind == "masked" else ~masked)[0])
        name = "P"
    elif layout.signature(entry)["kind"] == "reduce":
        index = int(_window(entry, built)[0])
        name = "out" if kind == "rounding" else "acc"
    else:
        index = 0
    dtype = layout.parameter_type(entry, name)
    value = from_words(dtype, [outputs[name][index]])[0]
    if kind == "masked":
        outputs[name][index] = to_words(dtype, [0.5])[0]
    elif kind == "rounding":
        outputs[name][index] ^= 1
    elif kind == "nan":
        outputs[name][index] = NAN_WORDS[dtype]
    else:
        outputs[name][index] = to_words(dtype, [value * 1.01 + 1e-3])[0]
    return outputs


@pytest.mark.parametrize(
    "entry,kind",
    [
        ("sdpa_vjp_odo_float_64", "value"),
        ("sdpa_vjp_odo_bfloat16_t_512", "nan"),
        ("sdpa_vjp_ds_float16_t", "value"),
        ("sdpa_vjp_ds_float", "masked"),
        ("sdpa_vjp_reduce_add_bfloat16_t", "value"),
        ("sdpa_vjp_reduce_set_float16_t", "rounding"),
    ],
)
def test_sdpa_vjp_checks_reject_defects(entry, kind):
    built = build_case(entry, CASES[layout.signature(entry)["kind"]][0])
    with pytest.raises(AssertionError):
        check_outputs(entry, built, _damage(entry, built, kind))


def test_sdpa_vjp_bounds_stay_a_few_roundings():
    """Each bound counts the kernel's float32 roundings, far inside
    test_fast_sdpa's 1e-2 gradient tolerance."""
    for case in DS_CASES:
        dS = build_case("sdpa_vjp_ds_float", case)["reference"]["dS"]
        assert float(dS["relative"].max()) < 1e-5
    # Even the longest row product is held to about 3e-5 of its terms.
    assert gamma(max(layout.ODO_DIMS) + 1) < 3.1e-5


@pytest.mark.parametrize("target", TARGETS)
def test_sdpa_vjp_configs_follow_the_host_recipe(tmp_path, target):
    for entry in ENTRIES:
        info = layout.signature(entry)
        config = _config(tmp_path / "work", tmp_path, target, entry)
        assert config.entry_points == {SOURCE: (entry,)}
        assert list(config.targets) == [target]
        options = config.source_options["metal"]["target_options"]
        # Only the row product sums across a simdgroup.
        if info["kind"] == "odo":
            assert options == packages.TARGET_OPTIONS
        else:
            assert options == {}
        waves = config.subgroup_width_rules
        assert waves == ({SOURCE: str(SIMD_WIDTH)} if target == "directx" else {})
        assert bool(config.index_range_assertions) == (target == "opengl")


def _config(work, root, target, entry):
    return packages.project_config(
        root, str(Path(work).relative_to(root) / "out"), target, entry
    )


# ---------------------------------------------------------------------------
# Native parity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def current_mlx():
    root, target = require_pinned_checkout(REQUIRE_ENV, "SDPA VJP", TARGETS, MLX_COMMIT)
    source = (root / SOURCE).read_text(encoding="utf-8")
    assert hashlib.sha256(source.encode("utf-8")).hexdigest() == SOURCE_SHA256
    host_bytes = (root / HOST_SOURCE).read_bytes()
    assert hashlib.sha256(host_bytes).hexdigest() == HOST_SHA256
    host = host_bytes.decode("utf-8")
    dims = {
        int(dim) for dim in re.findall(r"instantiate_odo\(in_type, (\d+)\)", source)
    }
    assert dims == set(layout.ODO_DIMS)
    for macro in ("odo_shapes", "sdpa_vjp_ds", "sdpa_vjp_reduce"):
        names = re.findall(rf"^instantiate_{macro}\((\w+)", source, re.MULTILINE)
        assert sorted(names) == sorted(layout.TYPES), macro
    # The launches the cases follow.
    for launch in (
        "MTL::Size(std::min(dim, 32), 8, 1)",
        "compute_encoder.set_bytes(params, 6);",
        "compute_encoder.set_bytes(qL, 3);",
    ):
        assert launch in host, launch
    return root, target


@pytest.fixture(scope="module")
def metal_reference(current_mlx, tmp_path_factory):
    root, target = current_mlx
    if target != "metal":
        return None
    work = tmp_path_factory.mktemp("sdpa-vjp-metal-reference")
    runner = work / "readback"
    _run(
        [
            "xcrun",
            "swiftc",
            ROOT / "tests/fixtures/runtime_verification/metal_raw_buffers.swift",
            "-o",
            runner,
        ],
        work,
        "build-runner",
    )
    return runner, _metal_library(
        root / SOURCE, work / "upstream.metallib", root, upstream=True
    )


def _translate(root, work, target, entry, record):
    """Translate one entry, check its pin and prepare its native package."""
    info = layout.signature(entry)
    translation = translate_project(
        _config(work, root, target, entry),
        format_output=False,
        validate=True,
        run_toolchains=False,
    )
    report = translation.to_json()
    translation.write_json(work / "report.json")
    assert validate_project_report(work / "report.json")["success"] is True
    assert report["diagnostics"] == []
    assert report["summary"]["translatedCount"] == 1
    (artifact,) = report["artifacts"]
    assert artifact["sourceHash"] == {"algorithm": "sha256", "value": SOURCE_SHA256}
    check_generated_artifact(
        artifact, sha256=record["sha256"], size_bytes=record["sizeBytes"]
    )
    (point,) = artifact["execution"]["entryPoints"]
    assert point["workgroupSize"] == list(info["workgroupSize"])
    generated = root / artifact["path"]
    text = generated.read_text(encoding="utf-8")
    assert "unsupported" not in text
    if target == "metal":
        # The kernels' own exp stays the fast one upstream calls.
        if info["kind"] == "ds":
            assert "metal::fast::exp(" in text
        _metal_library(generated, work / "translated.metallib", root)
    else:
        _validate_generated_artifact(generated, work, target)
    return _prepare_native_package(translation, work)


def _raw(entry, name, built):
    """One ``[[buffer(i)]]``'s bytes as the upstream kernel reads them."""
    if name in layout.records(entry):
        fields = built["metadata"]
        return struct.pack("<5if2i", *(fields[field] for field in layout.TILE_FIELDS))
    if name in layout.scalars(entry):
        return struct.pack("<i", built["values"][name])
    words = built["storage"][name]
    fmt = "I" if layout.parameter_type(entry, name) == "float32" else "H"
    return struct.pack(f"<{len(words)}{fmt}", *words)


def _upstream(work, reference, entry, built, translated):
    """Run the unchanged upstream kernel on the same storage; compare every
    output word for word and check every input is unchanged."""
    runner, library = reference
    label = built["label"]
    files, payloads = [], []
    for index, name in enumerate(layout.parameters(entry)):
        payloads.append(_raw(entry, name, built))
        path = work / f"{label}-upstream-input-{index}.bin"
        path.write_bytes(payloads[-1])
        files.append(str(path))
    request = work / f"{label}-upstream-request.json"
    request.write_text(
        json.dumps(
            {
                "buffers": files,
                "workgroupCount": built["execution"]["workgroupCount"],
                "workgroupSize": built["execution"]["workgroupSize"],
                "simdWidth": SIMD_WIDTH,
            }
        ),
        encoding="utf-8",
    )
    readback = work / f"{label}-upstream-readback"
    device = json.loads(
        _run([runner, library, entry, request, readback], work, f"{label}-upstream")
    )
    for index, name in enumerate(layout.parameters(entry)):
        raw = (readback / f"buffer-{index}.bin").read_bytes()
        if name not in layout.outputs(entry):
            assert raw == payloads[index], name
            continue
        size = built["spans"][name]
        expected = translated[name] + built["storage"][name][size:]
        fmt = "I" if layout.parameter_type(entry, name) == "float32" else "H"
        upstream = list(struct.unpack(f"<{len(raw) // struct.calcsize(fmt)}{fmt}", raw))
        assert_identical(f"{label}-{name}", expected, upstream)
    return device


def _run_case(work, target, entry, built, executor, translated, reference):
    """Dispatch one launch on the translation and, on Metal, upstream."""
    descriptor, package = translated
    host = SimpleNamespace(target=target)
    storage = built["storage"]
    inputs, observed, outputs, _ = dispatch._inputs(
        host, entry, descriptor, storage, built["values"], built["metadata"]
    )
    requested = {
        binding: {key: item for key, item in inputs[binding].items() if key != "values"}
        for binding in outputs.values()
    }
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        requested,
        built["execution"],
        expected_target=target,
        observe_inputs=tuple(observed),
    )
    availability = executor.is_available(request)
    assert availability.available, availability.reason
    result = executor.run(request)
    assert result.status == "ok", result.details
    observations = result.details["inputObservations"]
    assert len(observations) == len(observed)
    assert all(item["unchanged"] for item in observations.values())
    # Guards, and accumulator rows outside the window, come back unchanged.
    payloads = dispatch._readback(
        entry, result, requested, outputs, storage, built["metadata"], target
    )
    words = {name: items for name, (items, _) in payloads.items()}
    for name, binding in outputs.items():
        full = unpack(
            layout.parameter_type(entry, name),
            target,
            result.outputs[binding]["values"],
        )
        assert full[: built["spans"][name]] == words[name]
    worst = check_outputs(entry, built, words)
    device = None
    if target == "metal":
        device = _upstream(work, reference, entry, built, words)
    return {
        "case": built["label"],
        "execution": built["execution"],
        "worstErrorToBound": worst,
        "outputHashes": {
            name: hashlib.sha256(json.dumps(items).encode("utf-8")).hexdigest()
            for name, items in words.items()
        },
        "upstream": None if device is None else {"device": device, "identical": True},
    }


@pytest.mark.parametrize("entry", ENTRIES)
def test_current_mlx_sdpa_vjp_native_parity(
    current_mlx, metal_reference, tmp_path, entry
):
    root, target = current_mlx
    pin = _contract()["artifacts"][entry][target]
    with tempfile.TemporaryDirectory(prefix=".current-sdpa-vjp-", dir=root) as path:
        work = Path(path)
        try:
            translated = _translate(root, work, target, entry, parse_pin(pin))
            executor = _executor(target)
            try:
                evidence = [
                    _run_case(
                        work,
                        target,
                        entry,
                        build_case(entry, case),
                        executor,
                        translated,
                        metal_reference,
                    )
                    for case in CASES[layout.signature(entry)["kind"]]
                ]
            finally:
                close = getattr(executor.runtime_adapter.runtime, "close", None)
                if callable(close):
                    close()
            assert all(item["worstErrorToBound"] <= 1.0 for item in evidence)
            assert math.isfinite(max(item["worstErrorToBound"] for item in evidence))
            (work / "parity.json").write_text(
                json.dumps(
                    {
                        "commit": MLX_COMMIT,
                        "entry": entry,
                        "target": target,
                        "artifact": pin,
                        "cases": evidence,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            shutil.copytree(
                work,
                tmp_path / "evidence",
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    "*.air", "*.metallib", "out", "*.bin", "*-readback", "package"
                ),
            )


def test_ci_requires_native_sdpa_vjp_parity():
    from tests.ci_helpers import assert_workflow_triggers, current_parity_step

    workflow = (ROOT / ".github/workflows/demo-project-testing.yml").read_text()
    step = current_parity_step(
        workflow, "Prove current MLX SDPA VJP native parity", group="backward"
    )
    for required in (
        f'{REQUIRE_ENV}: "1"',
        "CROSTL_MLX_CURRENT_ROOT: ${{ github.workspace }}/mlx-current-tree-upstream",
        "pytest -q -n auto",
        "--basetemp=mlx-current-results/sdpa-vjp",
        "--junitxml=mlx-current-results/sdpa-vjp.xml",
        "demos/integrations/mlx/tests/kernels/test_current_sdpa_vjp.py",
    ):
        assert required in step, required
    assert "continue-on-error" not in step
    assert_workflow_triggers(
        workflow, "demos/integrations/mlx/tests/kernels/test_current_sdpa_vjp.py"
    )
