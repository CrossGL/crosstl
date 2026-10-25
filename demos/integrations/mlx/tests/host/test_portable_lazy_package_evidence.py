import hashlib
import json
from types import SimpleNamespace

import pytest

from demos.integrations.mlx.portable_host import lazy_package_evidence as lazy
from demos.integrations.mlx.portable_host import trace_records
from demos.integrations.mlx.portable_host.runtime import HostRuntime


def package(tmp_path, name, data=b"kernel source"):
    root = tmp_path / name
    (root / "artifacts").mkdir(parents=True)
    (root / "artifacts/kernel.metal").write_bytes(data)
    artifact = {
        "packagePath": "artifacts/kernel.metal",
        "hash": {"algorithm": "sha256", "value": hashlib.sha256(data).hexdigest()},
        "sizeBytes": len(data),
    }
    return {"artifact": artifact, "provenance": {}}, root


class Packages:
    def __init__(self, packages, target="metal"):
        self.target = target
        self.items = packages
        self.requests = []

    def packages(self, event):
        self.requests.append(event["entry"])
        return self.items


def event(entry, descriptor, **extra):
    return {
        "entry": entry,
        "target": "metal",
        "dispatchVersion": 3,
        "artifact": descriptor["artifact"],
        "workgroupCount": [1, 1, 1],
        "workgroupSize": [32, 1, 1],
        "details": {},
        **extra,
    }


@pytest.mark.parametrize(
    ("entry", "family"),
    [
        ("contig_scan_inclusive_sum_int32_int32", "scan"),
        ("strided_scan_reverse_exclusive_max_float32_float32", "scan"),
        ("row_reduce_small_1_reduce_sumint32", "small-row"),
        ("col_reduce_small_1_reduce_orbool_", "small-column"),
        ("gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0", "gemv"),
        (
            "steel_gemm_fused_nn_float32_float32_bm64_bn64_bk16_wm2_wn2"
            "_has_batch_n_use_out_source_n_do_axpby_n_align_M_n_align_N_n_align_K_n",
            "gemm",
        ),
        ("steel_gemm_splitk_accum_float32_float32", "gemm-splitk"),
        # The gathered gemv prefixes would also match GEMV's.
        ("gemv_gather_float32_bm4_bn1_sm1_sn32_tm4_tn4", "gather-mv"),
        ("gemv_t_gather_float32_bm1_bn8_sm1_sn32_tm4_tn4", "gather-mv"),
        (
            "steel_gather_mm_nn_float32_float32_bm64_bn64_bk16_wm2_wn2_has_batch_n"
            "_align_M_n_align_N_n_align_K_t",
            "gather-mm",
        ),
        ("gather_mm_offsets", "gather-mm-offsets"),
        ("gather_mm_offsets_extra", None),
        (
            "steel_segmented_mm_tn_float32_float32_bm64_bn64_bk16_wm2_wn2"
            "_segments_contiguous_n_align_M_n_align_N_n",
            "segmented-mm",
        ),
        ("row_reduce_looped_1_32_32_reduce_sumint32", None),
        ("all_reduce_sumint32", None),
        ("vv_Addint64", None),
        (None, None),
    ],
)
def test_lazy_families_cover_only_cached_translations(entry, family):
    assert lazy.family(entry) == family


def test_small_reductions_keep_the_callers_native_check(tmp_path):
    descriptor, root = package(tmp_path, "row")
    packages = Packages([(descriptor, root)])
    record = event("row_reduce_small_1_reduce_sumint32", descriptor)
    assert lazy.audit_event(record, packages, tmp_path) is False
    assert packages.requests == [record["entry"]]


@pytest.mark.parametrize("fault", ["artifact", "source", "target", "root", "version"])
def test_lazy_dispatch_must_name_its_unchanged_cached_package(tmp_path, fault):
    descriptor, root = package(tmp_path, "column")
    record = event("col_reduce_small_1_reduce_sumint32", descriptor)
    if fault == "artifact":
        record["artifact"] = {**descriptor["artifact"], "sizeBytes": 1}
    elif fault == "source":
        (root / "artifacts/kernel.metal").write_bytes(b"changed source")
    elif fault == "target":
        record["target"] = "opengl"
    elif fault == "root":
        record["packageRoot"] = str(tmp_path)
    else:
        record["dispatchVersion"] = 2
    with pytest.raises(ValueError):
        lazy.audit_event(record, Packages([(descriptor, root)]), tmp_path)


def test_scans_require_their_package_root_and_native_execution(tmp_path, monkeypatch):
    descriptor, root = package(tmp_path, "scan")
    entry = "contig_scan_inclusive_sum_int32_int32"
    request = {
        "target": "metal",
        "entryPoint": entry,
        "dispatch": {"workgroupCount": [1, 1, 1], "workgroupSize": [32, 1, 1]},
    }
    record = event(entry, descriptor, packageRoot=str(root))
    record["details"] = {"request": request}
    audited = []
    monkeypatch.setattr(lazy, "audit_native_execution", audited.append)
    assert lazy.audit_event(record, Packages([(descriptor, root)]), tmp_path) is True
    assert audited == [record]
    for fault in ("root", "launch"):
        broken = json.loads(json.dumps(record))
        if fault == "root":
            del broken["packageRoot"]
        else:
            broken["details"]["request"]["dispatch"]["workgroupSize"] = [64, 1, 1]
        with pytest.raises(ValueError):
            lazy.audit_event(broken, Packages([(descriptor, root)]), tmp_path)


@pytest.mark.parametrize("fault", ["none", "root", "entry", "launch", "target"])
def test_matrix_kernels_require_their_package_entry_and_native_execution(
    tmp_path, monkeypatch, fault
):
    descriptor, root = package(tmp_path, "gemm")
    name = "steel_gemm_fused_nn_float32_float32_bm64_bn64"
    descriptor["entryPoint"] = {"name": name, "stage": "compute"}
    entry = name + "_has_batch_n_use_out_source_n"
    request = {
        "target": "metal",
        "entryPoint": name,
        "dispatch": {"workgroupCount": [1, 1, 1], "workgroupSize": [32, 1, 1]},
    }
    record = event(entry, descriptor, packageRoot=str(root))
    record["details"] = {"request": request}
    if fault == "root":
        del record["packageRoot"]
    elif fault == "entry":
        request["entryPoint"] = entry
    elif fault == "launch":
        request["dispatch"]["workgroupCount"] = [2, 1, 1]
    elif fault == "target":
        request["target"] = "opengl"
    audited = []
    monkeypatch.setattr(lazy, "audit_native_execution", audited.append)
    packages = Packages([(descriptor, root)])
    if fault != "none":
        with pytest.raises(ValueError):
            lazy.audit_event(record, packages, tmp_path)
        assert audited == []
        return
    assert lazy.audit_event(record, packages, tmp_path) is True
    assert audited == [record] and packages.requests == [entry]


def test_matrix_caches_are_derived_per_entry(tmp_path):
    lazy_packages = lazy.LazyPackages(tmp_path / "mlx", tmp_path / "packages", "metal")
    requested = []
    for name in lazy.MATRIX_FAMILIES:
        lazy_packages.caches[name] = SimpleNamespace(
            get=lambda entry, name=name: requested.append((name, entry)) or ("d", "p")
        )
    entries = [
        "gemv_t_gather_float32_bm1_bn8_sm1_sn32_tm4_tn4",
        "steel_gather_mm_rhs_nt_float32_float32_bm16_bn64_bk16_wm1_wn2",
        "gather_mm_offsets",
        "steel_segmented_mm_nn_float16_float16_bm64_bn64_bk16_wm2_wn2",
        "gemv_t_float32_bm1_bn8_sm1_sn32_tm4_tn4_nc0_axpby0",
        "dot_product_bfloat16_it32_tg512_sg16",
        "steel_gemm_fused_tn_float16_float16_bm64_bn64_bk16_wm2_wn2",
        "steel_gemm_splitk_nn_float32_float32_bm16_bn32_bk16_wm2_wn2",
    ]
    for entry in entries:
        assert lazy_packages.packages({"entry": entry}) == [("d", "p")]
    assert requested == list(zip(lazy.MATRIX_FAMILIES, entries))


@pytest.mark.parametrize("fault", ["none", "count", "module", "outside", "root"])
def test_small_row_regions_name_each_package_and_module(tmp_path, fault):
    modules = tmp_path / "native-modules"
    modules.mkdir()
    items, regions = [], []
    for index in range(2):
        descriptor, root = package(tmp_path, f"region{index}", f"r{index}".encode())
        geometry = {"workgroupCount": [index + 1, 1, 1], "workgroupSize": [32, 1, 1]}
        descriptor["provenance"] = {"dispatchRegion": geometry}
        module = modules / f"{index}.glsl"
        module.write_bytes(b"module %d" % index)
        items.append((descriptor, root))
        regions.append(
            {
                "artifact": descriptor["artifact"],
                "provenance": descriptor["provenance"],
                "packageRoot": str(root),
                "moduleFile": str(module),
                "moduleHash": hashlib.sha256(module.read_bytes()).hexdigest(),
                **geometry,
            }
        )
    record = event("row_reduce_small_1_reduce_sumint32", items[0][0])
    record["details"] = {"regions": regions}
    if fault == "count":
        regions.pop()
    elif fault == "module":
        regions[1]["moduleHash"] = "0" * 64
    elif fault == "outside":
        outside = tmp_path / "outside.glsl"
        outside.write_bytes(b"module 1")
        regions[1]["moduleFile"] = str(outside)
    elif fault == "root":
        regions[0]["packageRoot"] = str(tmp_path)
    if fault == "none":
        assert lazy.audit_event(record, Packages(items), modules) is True
    else:
        with pytest.raises(ValueError):
            lazy.audit_event(record, Packages(items), modules)


def test_prefix_retention_compacts_only_the_other_entries(tmp_path):
    host = SimpleNamespace(trace=tmp_path / "trace.jsonl", trace_values=("gather",))
    values = list(range(trace_records.COMPACT_TRACE_LIMIT + 1))
    trace_records.write_trace(host, {"entry": "gatherint32int32_2_1_int", "v": values})
    trace_records.write_trace(host, {"entry": "vv_Addint64", "v": values})
    kept, compacted = (json.loads(line) for line in host.trace.read_text().splitlines())
    assert kept["v"] == values
    assert compacted["v"]["compactedValues"]["length"] == len(values)
    assert trace_records.retains_values(SimpleNamespace(trace_values=True), None)
    assert not trace_records.retains_values(SimpleNamespace(trace_values=False), "x")
    assert not trace_records.retains_values(host, None)


@pytest.mark.parametrize("value", [(), ("",), ["gather"], ("gather", 1), "gather"])
def test_runtime_refuses_malformed_trace_retention(tmp_path, value):
    with pytest.raises(ValueError, match="Trace value retention"):
        HostRuntime(tmp_path, tmp_path / "trace.jsonl", trace_values=value)
