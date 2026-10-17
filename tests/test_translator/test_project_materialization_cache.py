import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

from crosstl.backend.DirectX.preprocessor import Macro
from crosstl.backend.Metal.preprocessor import MetalTemplateSpecializationError
from crosstl.project import ProjectConfig, pipeline, translate_project
from crosstl.translator.dispatch_regions import plan_dispatch_regions


@pytest.fixture(autouse=True)
def clear_materialization_cache():
    with pipeline._metal_materialization_cache_lock:
        pipeline._metal_materialization_cache.clear()
    yield
    with pipeline._metal_materialization_cache_lock:
        pipeline._metal_materialization_cache.clear()


@pytest.fixture
def materialization_request(tmp_path):
    (tmp_path / "bias.h").write_text("#define BIAS 3\n", encoding="utf-8")
    source = """#include "bias.h"
template <typename T> T adjust(T value) { return value + T(1); }
template <typename T, int Offset>
[[kernel]] void launch(device T* out [[buffer(0)]],
                      uint gid [[thread_position_in_grid]]) {
    out[gid] = adjust<T>(T(gid + Offset + BIAS));
}
instantiate_kernel("first", launch, float, 1)
instantiate_kernel("second", launch, float, 2)
"""
    path = tmp_path / "kernel.metal"
    path.write_text(source, encoding="utf-8")
    return dict(
        unit=pipeline.ProjectTranslationUnit(
            path=path,
            relative_path=path.name,
            source_backend="metal",
            extension=".metal",
            source_hash={
                "algorithm": "sha256",
                "value": hashlib.sha256(source.encode()).hexdigest(),
            },
            source_size_bytes=len(source.encode()),
        ),
        target="metal",
        variant=None,
        defines={},
        include_paths=[str(tmp_path)],
        source_options={"max_template_materialization_work": 32},
        entry_point="first",
        fail_closed=True,
    )


def materialize(request):
    result = pipeline._project_template_materialization_for_artifact(**request)
    assert result is not None
    assert not result.blocked
    assert result.diagnostics == ()
    return result


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_materialization_reuse_returns_independent_records(
    materialization_request, monkeypatch, target
):
    request = {**materialization_request, "target": target}
    original = materialize(request)
    expected = materialize(request)
    assert len(pipeline._metal_materialization_cache) == 1

    def unexpected(*args, **kwargs):
        pytest.fail("A successful materialization should have been reused")

    monkeypatch.setattr(pipeline, "_retain_metal_materialization", unexpected)
    original.metadata["specializations"][0]["name"] = "changed"
    original.source_options["max_template_materialization_work"] = 1
    assert materialize(request) == expected
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(lambda _: materialize(request), range(8)))
    assert all(result == expected for result in results)
    results[0].metadata["specializations"].clear()
    assert all(result == expected for result in results[1:])
    assert materialize(request) == expected


@pytest.mark.parametrize("change", ["include", "source", "macro-state"])
def test_materialization_reuse_reads_current_content(materialization_request, change):
    request = materialization_request
    original = materialize(request)
    path = request["unit"].path
    if change == "source":
        old, new = "gid + Offset", "gid - Offset"
    else:
        path = path.parent / "bias.h"
        if change == "macro-state":
            path.write_text("#define BIAS 3\n#define UNUSED 1\n", encoding="utf-8")
            original = materialize(request)
            old, new = "UNUSED 1", "UNUSED 2"
        else:
            old, new = "BIAS 3", "BIAS 7"
    stat = path.stat()
    path.write_text(path.read_text().replace(old, new), encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    count = len(pipeline._metal_materialization_cache)
    updated = materialize(request)
    assert len(pipeline._metal_materialization_cache) == count + 1
    if change == "macro-state":
        assert updated == original
    else:
        assert updated.text != original.text


def test_materialization_reuse_does_not_hide_preprocessing_errors(
    materialization_request,
):
    materialize(materialization_request)
    header = materialization_request["unit"].path.parent / "bias.h"
    header.write_text("#error changed header\n", encoding="utf-8")
    with pytest.raises(SyntaxError, match="changed header"):
        pipeline._project_template_materialization_for_artifact(
            **materialization_request
        )
    assert len(pipeline._metal_materialization_cache) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"entry_point": "second"},
        {"target": "opengl"},
        {"variant": "another"},
        {"defines": {"UNUSED": "2"}},
        {"define_sources": {"UNUSED": "project-variant"}},
        {"source_options": {"max_template_materialization_work": 64}},
        {"source_options": {"max_template_specializations": 128}},
        {"source_options": {"preserve_resource_origins": True}},
        {"source_options": {"promote_derived_pointer_members": True}},
    ],
)
def test_materialization_reuse_keys_frontend_inputs(materialization_request, change):
    materialize(materialization_request)
    materialize({**materialization_request, **change})
    assert len(pipeline._metal_materialization_cache) == 2


def test_materialization_reuse_preserves_source_provenance(materialization_request):
    original = materialize(materialization_request)
    unit = materialization_request["unit"]
    moved = materialize(
        {
            **materialization_request,
            "unit": replace(unit, relative_path="nested/kernel.metal"),
        }
    )
    assert len(pipeline._metal_materialization_cache) == 2
    assert moved.text == original.text


@pytest.mark.parametrize(
    "options",
    [
        # Project translation flattens target_options before materialization.
        {
            "dispatch_region": {"workgroupCount": [2, 1, 1]},
            "software_subgroup_width": 32,
            "software_subgroup_applicability": "required",
            "relative_wave_shuffle_out_of_range": "self",
            "private_pointer_out_of_bounds_read": "zero",
            "cooperative_matrix_software_lowering": True,
            "cooperative_matrix_mma_profile": "profile",
        },
        {"target_options": {"metal": {"dispatch_region": {"workgroupCount": [2]}}}},
    ],
)
def test_materialization_reuse_ignores_withheld_target_options(
    materialization_request, monkeypatch, options
):
    original = materialize(materialization_request)
    monkeypatch.setattr(
        pipeline,
        "_retain_metal_materialization",
        lambda *args: pytest.fail("cache miss"),
    )
    changed = {
        **materialization_request,
        "target_artifact": "new-output/kernel.metal",
        "source_options": {**materialization_request["source_options"], **options},
    }
    assert materialize(changed) == original


@pytest.mark.parametrize(
    "options",
    [
        {"template_variants": {"*": {"*.metal": {"*": {"Offset": "4"}}}}},
        {"template_specialization_limit_source": "project.limit"},
        {"template_materialization_work_limit_source": "project.work"},
    ],
)
def test_materialization_reuse_keys_options_materialization_reads(
    materialization_request, options
):
    materialize(materialization_request)
    materialize(
        {
            **materialization_request,
            "source_options": {**materialization_request["source_options"], **options},
        }
    )
    assert len(pipeline._metal_materialization_cache) == 2


def test_withheld_options_are_absent_from_frontend_options():
    assert pipeline.METAL_MATERIALIZATION_WITHHELD_SOURCE_OPTIONS < (
        pipeline.FRONTEND_EXCLUDED_SOURCE_OPTIONS
    )
    assert not pipeline.METAL_MATERIALIZATION_WITHHELD_SOURCE_OPTIONS & {
        pipeline.TEMPLATE_VARIANTS_SOURCE_OPTION,
        pipeline.METAL_TEMPLATE_SPECIALIZATION_LIMIT_SOURCE_OPTION,
        pipeline.METAL_TEMPLATE_MATERIALIZATION_WORK_LIMIT_SOURCE_OPTION,
    }


def _key(**changes):
    unit = pipeline.ProjectTranslationUnit(
        path=pipeline.Path("/project/kernel.metal"),
        relative_path="kernel.metal",
        source_backend="metal",
        extension=".metal",
        source_hash={"algorithm": "sha256", "value": "0" * 64},
        source_size_bytes=1,
    )
    preprocessor = SimpleNamespace(macros={"A": Macro(name="A", replacement="1")})
    arguments = dict(
        unit=unit,
        target="opengl",
        variant=None,
        entry_point="first",
        source="source",
        expanded_discovery_source="discovery",
        discovery_preprocessor=preprocessor,
        expanded_source="expanded",
        preprocessor=preprocessor,
        defines={"A": "1"},
        define_sources=None,
        include_paths=["include"],
        source_options={"first": 1, "second": [1, 2]},
    )
    arguments.update(changes)
    return pipeline._metal_materialization_cache_key(**arguments)


@pytest.mark.parametrize(
    "change",
    [
        {"source_options": {"second": [1, 2], "first": 1}},
        {"source_options": {"first": 1, "second": (1, 2)}},
        {"source_options": {"first": True, "second": [1, 2]}},
        {"source_options": {"first": 1.0, "second": [1, 2]}},
        {"source_options": {"first": "1", "second": [1, 2]}},
        {"source_options": {1: 1, "second": [1, 2]}},
        {"defines": {"A": "2"}},
        {"define_sources": {}},
        {"include_paths": ["include", "other"]},
        {"expanded_source": "expanded "},
        {"expanded_discovery_source": "discovery "},
        {"source": "source "},
        {
            "preprocessor": SimpleNamespace(
                macros={"A": Macro(name="A", params=[], replacement="1")}
            )
        },
        {
            "discovery_preprocessor": SimpleNamespace(
                macros={"A": Macro(name="A", replacement="2")}
            )
        },
        {"entry_point": None},
        {"variant": "variant"},
        {"target": "directx"},
    ],
)
def test_materialization_key_distinguishes_inputs_exactly(change):
    assert _key() == _key()
    assert _key(**change) not in {None, _key()}


@pytest.mark.parametrize(
    "change",
    [
        {"source_options": {"first": object()}},
        {"source_options": {"first": {1, 2}}},
        {"preprocessor": SimpleNamespace(macros={"A": "not a macro"})},
    ],
)
def test_materialization_key_bypasses_unsupported_inputs(change):
    assert _key(**change) is None


def test_materialization_reuse_is_bounded_and_tracks_recent_hits(
    materialization_request,
):
    original = materialize(materialization_request)
    first_key = next(iter(pipeline._metal_materialization_cache))
    for index in range(pipeline.METAL_MATERIALIZATION_CACHE_LIMIT - 1):
        materialize({**materialization_request, "variant": str(index)})
    oldest_key = list(pipeline._metal_materialization_cache)[1]
    assert materialize(materialization_request) == original
    materialize({**materialization_request, "variant": "new"})
    assert (
        len(pipeline._metal_materialization_cache)
        == pipeline.METAL_MATERIALIZATION_CACHE_LIMIT
    )
    assert first_key in pipeline._metal_materialization_cache
    assert oldest_key not in pipeline._metal_materialization_cache


def test_materialization_failures_are_not_retained(materialization_request):
    request = {
        **materialization_request,
        "source": "template <typename T> kernel void unresolved(device T* out) {}",
    }
    for _ in range(2):
        result = pipeline._project_template_materialization_for_artifact(**request)
        assert result is not None and result.blocked
        assert not pipeline._metal_materialization_cache


def test_materialization_reuse_cannot_bypass_lower_work_limit(materialization_request):
    materialize(materialization_request)
    with pytest.raises(MetalTemplateSpecializationError, match="work budget exceeded"):
        pipeline._project_template_materialization_for_artifact(
            **{
                **materialization_request,
                "source_options": {"max_template_materialization_work": 1},
            }
        )
    assert len(pipeline._metal_materialization_cache) == 1


def test_materialization_reuse_bypasses_unserializable_options(materialization_request):
    request = {
        **materialization_request,
        "source_options": {"extra": object()},
    }
    for _ in range(2):
        materialize(request)
        assert not pipeline._metal_materialization_cache


SUBGROUP_SOURCE = """#include <metal_stdlib>
using namespace metal;
template <typename T>
[[kernel]] void subgroup_copy(device T* output [[buffer(0)]],
                              uint tid [[thread_position_in_grid]]) {
    output[tid] = T(simd_sum(tid));
}
instantiate_kernel("subgroup_uint", subgroup_copy, uint)
"""


def _region_project(root, target, region, output):
    root.mkdir(parents=True, exist_ok=True)
    (root / "kernel.metal").write_text(SUBGROUP_SOURCE, encoding="utf-8")
    options = {"dispatch_region": region.to_json(), "software_subgroup_width": 32}
    if target == "directx":
        options["relative_wave_shuffle_out_of_range"] = "self"
    report = translate_project(
        ProjectConfig(
            root=root,
            include_patterns=("kernel.metal",),
            targets=(target,),
            output_dir=output,
            workgroup_size=region.workgroup_size,
            entry_points={"kernel.metal": ("subgroup_uint",)},
            source_options={"metal": {"target_options": {target: options}}},
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["failedCount"] == 0, report["diagnostics"]
    (artifact,) = report["artifacts"]
    assert artifact["provenance"]["dispatchRegion"] == region.to_json()
    assert artifact["templateMaterialization"]["status"] == "materialized"
    return (root / artifact["path"]).read_text(encoding="utf-8")


@pytest.mark.parametrize("target", ["opengl", "directx"])
def test_one_materialization_serves_each_dispatch_region(tmp_path, monkeypatch, target):
    # Checked host packages translate each dispatch from one source root.
    regions = plan_dispatch_regions([37], [32])
    assert len(regions) == 2
    cached = [
        _region_project(tmp_path, target, region, f"cached-{index}")
        for index, region in enumerate(regions)
    ]
    assert len(pipeline._metal_materialization_cache) == 1
    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "_cached_metal_materialization", lambda key: None)
        patch.setattr(pipeline, "_retain_metal_materialization", lambda *args: None)
        uncached = [
            _region_project(tmp_path, target, region, f"uncached-{index}")
            for index, region in enumerate(regions)
        ]
    assert cached == uncached
    assert cached[0] != cached[1]
