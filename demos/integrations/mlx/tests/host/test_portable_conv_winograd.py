"""conv.cpp's Winograd route on the host: batch step, launches and reference."""

import copy
import math
import tomllib
from pathlib import Path

import pytest

from demos.integrations.mlx.portable_host import conv_layout as layout
from demos.integrations.mlx.portable_host import conv_packages
from demos.integrations.mlx.portable_host import conv_winograd_layout as winograd
from demos.integrations.mlx.portable_host import conv_winograd_reference as reference
from demos.integrations.mlx.portable_host import verify_conv as verify
from demos.integrations.mlx.tests.kernels import test_current_winograd as gate

HOST = Path(layout.__file__).parent
RECORDS = [verify.case_record(case) for case in verify.CASES]
WINOGRAD = {
    record["name"]: record for record in RECORDS if record["route"] == winograd.ROUTE
}
WS, TILE = winograd.WORKING_SET_ENV, winograd.TILE_BATCH_ENV


def _params(n, h, w, c, o, pad=1):
    out = [h + 2 * pad - 2, w + 2 * pad - 2]
    return {
        "N": n,
        "C": c,
        "O": o,
        "iS": [h, w],
        "wS": [3, 3],
        "oS": out,
        "str": [1, 1],
        "pad": [pad, pad],
        "kdil": [1, 1],
        "idil": [1, 1],
        "in_strides": layout.contiguous([n, h, w, c]),
        "wt_strides": layout.contiguous([o, 3, 3, c]),
        "out_strides": layout.contiguous([n, *out, o]),
        "groups": 1,
        "flip": False,
    }


def winograd_launches(record):
    """``(entry, counts, parameters, threads, execution)`` per conv.inc transform."""
    p, dtype = record["params"], record["dtype"]
    weight, inputs, outputs = winograd.entries(dtype)
    tiles_h, tiles_w = winograd.tile_grid(p)
    padded = winograd.padded_size(p)
    result = []
    for rows in record["batchRows"]:
        filters = 64 * p["C"] * p["O"]
        result.append(
            (
                weight,
                {"wt_in": math.prod(record["wtShape"]), "wt_out": filters},
                {"C": p["C"], "O": p["O"]},
                filters,
                {"workgroupCount": [p["O"] // 4, 1, 1], "workgroupSize": [32, 4, 1]},
            )
        )
        step = rows[0]
        for tile_n in rows:
            transformed = {**winograd.transform_params(p), "N": tile_n}
            execution = {
                "workgroupCount": [tiles_w, tiles_h, tile_n],
                "workgroupSize": [32, 2, 2],
            }
            tiles = tile_n * tiles_h * tiles_w
            domain_in = 64 * tiles * p["C"]
            result.append(
                (
                    inputs,
                    {
                        "inp_in": step * padded[0] * padded[1] * p["C"],
                        "inp_out": domain_in,
                    },
                    {"params": copy.deepcopy(transformed)},
                    domain_in,
                    copy.deepcopy(execution),
                )
            )
            written = tile_n * math.prod(p["oS"]) * p["O"]
            result.append(
                (
                    outputs,
                    {"out_in": 64 * tiles * p["O"], "out_out": written},
                    {"params": copy.deepcopy(transformed)},
                    written,
                    execution,
                )
            )
    return result


def test_entries_name_conv_cpp_pipelines():
    for dtype in layout.DTYPES:
        names = winograd.entries(dtype)
        assert names == [
            f"winograd_conv_2d_weight_transform_{dtype}_bc32",
            f"winograd_conv_2d_input_transform_{dtype}_bc32",
            f"winograd_conv_2d_output_transform_{dtype}_bo32",
        ]
        for stage, name in zip(winograd.STAGES, names):
            info = layout.signature(name)
            assert (info["kind"], info["stage"], info["dtype"]) == (
                f"winograd_{stage}",
                stage,
                dtype,
            )
            assert layout.source(name) == layout.SOURCE
            assert layout.source_entry(name) == name
            assert layout.workgroup_size(name) == winograd.WORKGROUPS[stage]
            assert layout.host_parameters(name) == winograd.PARAMETERS[stage]
    assert set(gate.ENTRIES) == {
        name for dtype in layout.DTYPES for name in winograd.entries(dtype)
    }
    for name in (
        "winograd_conv_2d_output_transform_float32_bc32",
        "winograd_conv_2d_weight_transform_float32_bo32",
        "winograd_conv_2d_input_transform_float64_bc32",
    ):
        with pytest.raises(ValueError, match="Unsupported"):
            layout.signature(name)


def test_layout_follows_the_winograd_gate():
    assert winograd.WORKGROUPS == gate.WORKGROUPS
    assert winograd.OPENGL_INDEX_RANGE_ASSERTIONS == gate.OPENGL_INDEX_RANGE_ASSERTIONS
    assert winograd.METADATA_PRECONDITIONS == gate.METADATA_PRECONDITIONS
    assert winograd.PARAMETERS == gate.BUFFER_ORDER
    assert winograd.MMA_PROFILE == gate.MMA_PROFILE
    assert winograd.FRAGMENT_MAPPING == gate.FRAGMENT_MAPPING
    assert winograd.FRAGMENT_MAPPING_PROVENANCE == gate.FRAGMENT_MAPPING_PROVENANCE
    for name, rows, _ in gate.TABLES.values():
        table = {
            "wt_transform": reference.WT_TRANSFORM,
            "in_transform": reference.IN_TRANSFORM,
            "out_transform": reference.OUT_TRANSFORM,
        }[name]
        from fractions import Fraction

        widened = tuple(
            tuple(Fraction(value) for value in row) + (Fraction(0),) * (8 - len(row))
            for row in table
        )
        assert widened == rows, name


@pytest.mark.parametrize("target", ("metal", "opengl", "directx"))
def test_packages_follow_the_winograd_gate(tmp_path, target):
    for entry in gate.ENTRIES:
        built = conv_packages.ConvPackageCache(Path("/pinned"), tmp_path, target)
        config = built.project_config("out", entry)
        expected = tomllib.loads(gate._config("out", target, entry))["project"]
        assert config.entry_points == {layout.SOURCE: (entry,)}
        (rules,) = config.entry_workgroup_size_rules.values()
        assert {name: list(map(int, size)) for name, size in rules.items()} == {
            entry: expected["entry_workgroup_size_rules"][layout.SOURCE][entry]
        }
        assert config.source_options == expected["source_options"]
        gated = type(config)(
            root=Path("/pinned"),
            index_range_assertions=expected.get("index_range_assertions", ()),
            metadata_preconditions=expected.get("metadata_preconditions", ()),
        )
        for name in ("index_range_assertions", "metadata_preconditions"):
            assert [item.to_json() for item in getattr(config, name)] == [
                item.to_json() for item in getattr(gated, name)
            ], name


def test_batch_step_follows_winograd_batch_step():
    # test_conv2d_winograd_batch_tiling's shapes and budget arithmetic.
    for in_shape, wt_shape in (
        ((8, 48, 48, 64), (192, 3, 3, 64)),
        ((5, 52, 44, 128), (128, 3, 3, 128)),
        ((4, 48, 48, 192), (96, 3, 3, 192)),
    ):
        n, h, w, c = in_shape
        o = wt_shape[0]
        p = _params(n, h, w, c, o)
        assert layout.route(p, "float32", {}) == winograd.ROUTE
        assert winograd.batch_step(p, "float32", {}) == n
        assert winograd.batch_step(p, "float32", {TILE: "3"}) == 3
        padded_h = 6 * ((h + 5) // 6) + 2
        padded_w = 6 * ((w + 5) // 6) + 2
        per_n = (
            padded_h * padded_w * c * 4
            + 64 * ((h + 5) // 6) * ((w + 5) // 6) * (c + o) * 4
        )
        used = (n * h * w * (c + o) + 64 * c * o) * 4
        budget = str(int((used + 5 * per_n // 2) / 0.75))
        assert winograd.batch_step(p, "float32", {WS: budget}) == 2
        assert winograd.batch_step(p, "float32", {WS: "1"}) == 0
        assert winograd.batch_step(p, "float32", {WS: "1", TILE: "1"}) == 0
        # A forced tile is still capped by the budget.
        assert winograd.batch_step(p, "float32", {WS: budget, TILE: "3"}) == 2
        # Half elements need half the budget per row.
        assert winograd.batch_step(p, "float16", {WS: budget}) > 2
        assert layout.route(p, "float32", {WS: "1"}) != winograd.ROUTE
    # env::get_var's atoi takes a leading integer and ignores the rest.
    p = _params(4, 48, 48, 192, 96)
    assert winograd.batch_step(p, "float32", {TILE: " 3x"}) == 3
    assert winograd.batch_step(p, "float32", {TILE: "x3"}) == 4
    assert winograd.batch_step(p, "float32", {TILE: "-2"}) == 4
    with pytest.raises(ValueError, match="element type"):
        winograd.batch_step(p, None, {WS: "100"})


def test_batch_step_takes_the_host_bound_for_the_working_set():
    # 64 * tiles * O words per batch row: the domain output caps the tile.
    p = _params(64, 192, 192, 32, 512)
    tiles = 32 * 32
    cap = winograd.MAX_ELEMENTS // (64 * tiles * 512)
    assert cap < 64
    assert winograd.batch_step(p, "float32", {}) == cap
    rows = winograd.scratch_elements(p, cap)
    assert max(rows.values()) <= winograd.MAX_ELEMENTS
    assert max(winograd.scratch_elements(p, cap + 1).values()) > winograd.MAX_ELEMENTS
    # Below 32 GEMM rows per tile the route falls back unless forced.
    small = _params(70000, 8, 8, 32, 32, pad=0)
    assert winograd.tile_grid(small) == (1, 1)
    # 65535 batch rows, the Z workgroups of one launch.
    assert winograd.batch_step(small, "float32", {}) == winograd.MAX_GROUPS
    narrow = _params(2, 64, 64, 4096, 4096)
    assert winograd.batch_step(narrow, "float32", {}) == 0


def test_routing_follows_the_working_set():
    base = _params(1, 64, 64, 32, 224, pad=0)
    assert layout.route(base, "float32", {}) == winograd.ROUTE
    assert layout.route(base, "float32", {WS: "1"}) == "implicit_gemm_conv_2D_gpu"
    assert layout.route({**base, "flip": True}, "float32", {}) != winograd.ROUTE
    assert layout.route({**base, "C": 16}, "float32", {}) != winograd.ROUTE
    assert layout.route({**base, "iS": [63, 64]}, "float32", {}) != winograd.ROUTE
    assert layout.route({**base, "O": 192}, "float32", {}) != winograd.ROUTE
    # The implicit GEMM plan accepts the fallback launch only when the
    # working set refuses Winograd.
    record = {record["name"]: record for record in RECORDS}["winograd-2d-fallback"]
    assert record["route"] == "implicit_gemm_conv_2D_gpu"


def test_cases_cover_the_winograd_route():
    assert set(WINOGRAD) == {
        "winograd-2d",
        "winograd-2d-tiled",
        "winograd-2d-moved",
        "winograd-2d-working-set",
    }
    assert WINOGRAD["winograd-2d"]["batchRows"] == [[1]]
    assert WINOGRAD["winograd-2d-tiled"]["batchRows"] == [[2, 1], [3]]
    assert WINOGRAD["winograd-2d-working-set"]["batchRows"] == [[2, 2, 1]]
    # The channels-first copy, the zero fill and one tile's input copy.
    assert WINOGRAD["winograd-2d-moved"]["copies"] == 1 + 1 + 1
    tiled = WINOGRAD["winograd-2d-tiled"]
    assert tiled["copies"] == (1 + 2) + (1 + 1)
    weight, inputs, outputs = winograd.entries("float16")
    first, second = tiled["launches"][:7], tiled["launches"][7:]
    assert first[0] == second[0] == weight
    assert first[1::3] == [inputs, inputs] and second[1::3] == [inputs]
    assert first[3::3] == [outputs, outputs] and second[3::3] == [outputs]
    # 2 * 42 and 1 * 42 GEMM rows, and 3 * 42 untiled: none fills whole
    # 64-row tiles, so one pipeline serves them all.
    assert tiled["productEntries"] == [gemm_entry("float16", 84, 192, 64)]
    assert {gemm_entry("float16", rows, 192, 64) for rows in (84, 42, 126)} == set(
        tiled["productEntries"]
    )


def gemm_entry(dtype, m, n, k):
    from demos.integrations.mlx.portable_host import gemm_layout

    return gemm_layout.select(dtype, False, False, m, n, k, 1)


@pytest.mark.parametrize("name", sorted(WINOGRAD))
def test_plan_accepts_each_transform(name):
    record = WINOGRAD[name]
    planned = winograd_launches(record)
    convs = [entry for entry in record["launches"] if entry in record["convEntries"]]
    assert [entry for entry, *_ in planned] == convs
    for entry, counts, parameters, threads, execution in planned:
        metadata = layout.plan(entry, counts, parameters, threads, execution)
        assert metadata["route"] == winograd.ROUTE
        assert metadata["stage"] == layout.signature(entry)["stage"]


def _launch(stage):
    record = WINOGRAD["winograd-2d-tiled"]
    planned = winograd_launches(record)
    index = {"weight": 0, "input": 1, "output": 2}[stage]
    return copy.deepcopy(planned[index])


@pytest.mark.parametrize(
    "stage, fault, message",
    (
        ("weight", lambda c, p, e: p.update(C=48), "channels"),
        ("weight", lambda c, p, e: p.update(O=16), "channels"),
        ("weight", lambda c, p, e: e.update(workgroupSize=[32, 2, 2]), "threadgroups"),
        (
            "weight",
            lambda c, p, e: e["workgroupCount"].__setitem__(0, 1),
            "threadgroups",
        ),
        ("weight", lambda c, p, e: c.update(wt_in=c["wt_in"] - 1), "past its weights"),
        ("weight", lambda c, p, e: c.update(wt_out=c["wt_out"] - 1), "output differs"),
        ("input", lambda c, p, e: p["params"].update(pad=[1, 1]), "invalid"),
        ("input", lambda c, p, e: p["params"].update(flip=True), "invalid"),
        ("input", lambda c, p, e: p["params"].update(C=48), "invalid"),
        ("input", lambda c, p, e: p["params"]["iS"].__setitem__(0, 40), "strides"),
        (
            "input",
            lambda c, p, e: p["params"]["in_strides"].__setitem__(1, 1),
            "strides",
        ),
        (
            "input",
            lambda c, p, e: p["params"]["out_strides"].__setitem__(0, 1),
            "strides",
        ),
        (
            "input",
            lambda c, p, e: e["workgroupCount"].__setitem__(2, 1),
            "threadgroups",
        ),
        (
            "input",
            lambda c, p, e: e["workgroupCount"].__setitem__(0, 7),
            "threadgroups",
        ),
        ("input", lambda c, p, e: c.update(inp_in=c["inp_in"] - 64), "past its rows"),
        ("input", lambda c, p, e: c.update(inp_out=c["inp_out"] + 1), "output differs"),
        ("output", lambda c, p, e: c.update(out_in=c["out_in"] + 1), "reads differ"),
        (
            "output",
            lambda c, p, e: c.update(out_out=c["out_out"] - 1),
            "output differs",
        ),
        ("output", lambda c, p, e: p["params"].update(O=224), "strides"),
    ),
)
def test_plan_rejects_transforms_that_differ_from_conv_cpp(stage, fault, message):
    entry, counts, parameters, threads, execution = _launch(stage)
    layout.plan(entry, counts, parameters, threads, execution)
    fault(counts, parameters, execution)
    with pytest.raises(ValueError, match=message):
        layout.plan(entry, counts, parameters, threads, execution)


def _emulate(np, x, w, pad, dtype):
    """The kernels' stages in float32, products in conv.metal's order."""
    f = np.float32
    if dtype == "bfloat16":

        def store(values):
            bits = values.astype(f).view(np.uint32).astype(np.uint64)
            bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
            return bits.astype(np.uint32).view(f)

    else:
        kind = {"float32": np.float32, "float16": np.float16}[dtype]

        def store(values):
            return values.astype(kind).astype(f)

    def table(rows):
        return reference._table(np, rows).astype(f)

    g, b, a = (
        table(reference.WT_TRANSFORM),
        table(reference.IN_TRANSFORM),
        table(reference.OUT_TRANSFORM),
    )
    n, h, wd, c = x.shape
    out_h, out_w = h + 2 * pad[0] - 2, wd + 2 * pad[1] - 2
    th, tw = -(-out_h // 6), -(-out_w // 6)
    filters = store(np.einsum("ai,oijc,bj->abco", g, w.astype(f), g, optimize=False))
    padded = np.zeros((n, 6 * th + 2, 6 * tw + 2, c), dtype=f)
    padded[:, pad[0] : pad[0] + h, pad[1] : pad[1] + wd] = x
    tiles = np.stack(
        [
            np.stack(
                [padded[:, 6 * i : 6 * i + 8, 6 * j : 6 * j + 8] for j in range(tw)], 1
            )
            for i in range(th)
        ],
        1,
    )
    first = np.einsum("ra,ntuabc->nturbc", b.T, tiles)
    inputs = store(np.einsum("nturbc,bs->ntursc", first, b))
    products = store(np.einsum("ntuabc,abco->ntuabo", inputs, filters))
    first = np.einsum("ntuabo,bs->ntuaso", products, a)
    outputs = store(np.einsum("ra,ntuaso->nturso", a.T, first))
    out = outputs.transpose(0, 1, 3, 2, 4, 5).reshape(n, th * 6, tw * 6, -1)
    return out[:, :out_h, :out_w]


@pytest.mark.parametrize("dtype", layout.DTYPES)
def test_reference_certifies_the_kernels_rounding(dtype):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(7)
    shape, filters = (2, 20, 26, 32), 224
    x = verify._nonzero(rng, shape, 0.25)
    w = verify._nonzero(rng, (filters, 3, 3, 32), 0.5)
    value, uncertainty = reference.stages(np, x, w, (1, 1), dtype)
    assert value.shape == uncertainty.shape == (2, 20, 26, filters)
    exact = verify.reference(
        np,
        x,
        w,
        {
            "options": {
                "groups": 1,
                "idil": [1, 1],
                "padLo": [1, 1],
                "padHi": [1, 1],
                "flip": False,
                "kdil": [1, 1],
                "stride": [1, 1],
            },
            "params": {"C": 32, "O": filters},
            "outShape": [2, 20, 26, filters],
        },
    )
    emulated = _emulate(np, x, w, (1, 1), dtype)
    assert reference.contains(np, emulated, value, uncertainty)
    # Half the outputs are certain to within 2^-7 of the convolution's
    # magnitude: far narrower than any misplaced value.
    scale = np.maximum(np.abs(exact), 1.0)
    assert np.median(uncertainty / scale) <= 2.0**-7
    # A tile from the wrong batch row, a channel block shifted by one and
    # swapped output rows each leave the interval.
    moved = emulated.copy()
    moved[0, :6, :6] = emulated[1, :6, :6]
    assert not reference.contains(np, moved, value, uncertainty)
    shifted = emulated.copy()
    shifted[..., :32] = emulated[..., 1:33]
    assert not reference.contains(np, shifted, value, uncertainty)
    swapped = emulated.copy()
    swapped[:, [5, 6]] = emulated[:, [6, 5]]
    assert not reference.contains(np, swapped, value, uncertainty)


def test_reference_rejects_stage_overflow():
    np = pytest.importorskip("numpy")
    x = np.full((1, 64, 64, 32), 60000.0)
    w = np.ones((224, 3, 3, 32))
    with pytest.raises(ValueError, match="storage range"):
        reference.stages(np, x, w, (0, 0), "float16")


def test_backend_ports_winograd_conv_2d_gpu():
    port = (HOST / "conv.inc").read_text(encoding="utf-8")
    assert "conv_unported(" not in port
    start = port.index("void conv_dispatch_2d(")
    body = port[start : port.index("\n}\n", start)]
    assert (
        "if (int n_step = conv_winograd_batch_step(in, conv_params); n_step > 0) {\n"
        "      return conv_winograd(in, wt, out, conv_params, n_step);"
    ) in body
    step = port[port.index("int conv_winograd_batch_step(") :]
    step = step[: step.index("\n}\n")]
    for text in (
        'env::get_var("MLX_CONV_WINOGRAD_WORKING_SET", 0)',
        'env::get_var("MLX_CONV_WINOGRAD_TILE_BATCH", 0)',
        "constexpr int min_rows_per_tile = 32;",
        "size_t(env_ws) / 4 * 3",
        "65535",
    ):
        assert text in step, text
    route = port[port.index("void conv_winograd(") :]
    route = route[: route.index("\n}\n")]
    order = [
        '"winograd_conv_2d_weight_transform_"',
        "dispatch_copy(fill, in_padded);",
        "for (int n_offset = 0; n_offset < conv_params.N; n_offset += n_step) {",
        "dispatch_strided_copy(",
        '"winograd_conv_2d_input_transform_"',
        "collapse_batches(inp_wg_tile, filt_wg);",
        "dispatch_steel_gemm(",
        '"winograd_conv_2d_output_transform_"',
    ]
    assert [route.index(text) for text in order] == sorted(
        route.index(text) for text in order
    )
    for stage in winograd.STAGES:
        for name in winograd.PARAMETERS[stage]:
            assert f'{{"{name}", ' in route, name
