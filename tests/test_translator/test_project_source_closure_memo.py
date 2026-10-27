from dataclasses import replace
from types import SimpleNamespace

import pytest

from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.project import ProjectConfig, pipeline, scan_project
from crosstl.project import source_closure_memo as memo_module
from crosstl.project.source_closure_memo import (
    SourceClosureMemo,
    UncacheableSource,
    include_closure_fingerprint,
    memoized_preprocessor_run,
)


@pytest.fixture(autouse=True)
def clear_memos():
    for memo in (
        memo_module.ENTRY_DISCOVERY_MEMO,
        memo_module.SOURCE_HOST_INTERFACE_MEMO,
        memo_module.PREPROCESSOR_RUN_MEMO,
    ):
        memo.clear()
    yield
    for memo in (
        memo_module.ENTRY_DISCOVERY_MEMO,
        memo_module.SOURCE_HOST_INTERFACE_MEMO,
        memo_module.PREPROCESSOR_RUN_MEMO,
    ):
        memo.clear()


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_fingerprint_tracks_included_bytes(tmp_path):
    source = '#include "common.h"\nkernel void k() {}\n'
    main = _write(tmp_path / "src" / "main.metal", source)
    header = _write(tmp_path / "src" / "common.h", '#include "nested.h"\n')
    nested = _write(tmp_path / "src" / "nested.h", "#define WIDTH 4\n")
    first = include_closure_fingerprint(source, str(main), [])
    assert include_closure_fingerprint(source, str(main), []) == first
    nested.write_text("#define WIDTH 8\n", encoding="utf-8")
    second = include_closure_fingerprint(source, str(main), [])
    assert second != first
    header.write_text("#define WIDTH 8\n", encoding="utf-8")
    assert include_closure_fingerprint(source, str(main), []) not in {first, second}


def test_fingerprint_tracks_absent_candidates_in_search_order(tmp_path):
    source = '#include "shared.h"\n#if __has_include(<optional.h>)\n#endif\n'
    main = _write(tmp_path / "src" / "main.metal", source)
    include_dir = tmp_path / "include"
    _write(include_dir / "shared.h", "#define VALUE 1\n")
    before = include_closure_fingerprint(source, str(main), [str(include_dir)])
    # A header next to the source now shadows the include-path copy.
    _write(tmp_path / "src" / "shared.h", "#define VALUE 2\n")
    shadowed = include_closure_fingerprint(source, str(main), [str(include_dir)])
    assert shadowed != before
    # __has_include operands are probed even without a matching #include.
    _write(include_dir / "optional.h", "\n")
    assert (
        include_closure_fingerprint(source, str(main), [str(include_dir)]) != shadowed
    )


def test_fingerprint_terminates_on_include_cycles(tmp_path):
    main = _write(tmp_path / "main.metal", '#include "a.h"\n')
    _write(tmp_path / "a.h", '#pragma once\n#include "b.h"\n')
    _write(tmp_path / "b.h", '#pragma once\n#include "a.h"\n')
    assert include_closure_fingerprint(main.read_text(), str(main), [])


@pytest.mark.parametrize(
    "source",
    [
        "#define HEADER <x.h>\n#include HEADER\n",
        '#include_next "x.h"\n',
        "#if __has_include(HEADER)\n#endif\n",
        "#if __has_include_next(<x.h>)\n#endif\n",
    ],
)
def test_operands_that_are_not_modelled_are_uncacheable(tmp_path, source):
    main = _write(tmp_path / "main.metal", source)
    with pytest.raises(UncacheableSource):
        include_closure_fingerprint(source, str(main), [str(tmp_path)])


def test_memo_reuses_only_identical_inputs_and_guards(tmp_path):
    source = '#include "h.h"\n'
    main = _write(tmp_path / "main.metal", source)
    header = _write(tmp_path / "h.h", "1\n")
    memo = SourceClosureMemo(limit=4)
    calls = []

    def produce():
        calls.append(1)
        return {"values": [len(calls)]}

    def compute(inputs=("a",), guards=(produce,)):
        return memo.compute(
            source=source,
            file_path=str(main),
            include_paths=[],
            inputs=list(inputs),
            guards=guards,
            produce=produce,
        )

    first = compute()
    first["values"].append("mutated")
    assert compute() == {"values": [1]}
    assert (memo.hits, memo.misses) == (1, 1)
    assert compute(inputs=("b",)) == {"values": [2]}
    assert compute(guards=(object(),)) == {"values": [3]}
    header.write_text("2\n", encoding="utf-8")
    assert compute() == {"values": [4]}
    assert len(calls) == 4


def test_memo_never_stores_uncacheable_sources(tmp_path):
    source = "#include MACRO_HEADER\n"
    main = _write(tmp_path / "main.metal", source)
    memo = SourceClosureMemo()
    calls = []
    for _ in range(2):
        memo.compute(
            source=source,
            file_path=str(main),
            include_paths=[],
            inputs=[],
            guards=(),
            produce=lambda: calls.append(1),
        )
    assert len(calls) == 2 and len(memo) == 0


def _expand(preprocessor, code, *, file_path):
    return preprocessor.preprocess(code, file_path=file_path)


def test_preprocessor_run_restores_the_state_a_fresh_run_leaves(tmp_path):
    _write(tmp_path / "defs.h", "#define SCALE 3\n#define TWICE(x) ((x) * 2)\n")
    source = '#include "defs.h"\nint value = TWICE(SCALE);\n'
    main = _write(tmp_path / "main.metal", source)

    def run():
        preprocessor = MetalPreprocessor(include_paths=[str(tmp_path)], defines={})
        text = memoized_preprocessor_run(
            preprocessor,
            source,
            file_path=str(main),
            source_backend="metal",
            run=_expand,
        )
        return text, preprocessor

    fresh_text, fresh = run()
    cached_text, cached = run()
    assert memo_module.PREPROCESSOR_RUN_MEMO.hits == 1
    assert cached_text == fresh_text and "((3) * 2)" in cached_text
    assert sorted(cached.macros) == sorted(fresh.macros)
    assert cached.macros["SCALE"] is not fresh.macros["SCALE"]
    assert cached._entered_source_files == fresh._entered_source_files

    # Different initial state (a configured define) is a different key.
    preprocessor = MetalPreprocessor(
        include_paths=[str(tmp_path)], defines={"EXTRA": "1"}
    )
    memoized_preprocessor_run(
        preprocessor, source, file_path=str(main), source_backend="metal", run=_expand
    )
    assert memo_module.PREPROCESSOR_RUN_MEMO.misses == 2
    assert "EXTRA" in preprocessor.macros


METAL_KERNEL = """#include <metal_stdlib>
#include "names.h"
using namespace metal;
kernel void KERNEL_NAME(device float* out [[buffer(0)]],
                        uint gid [[thread_position_in_grid]]) {
    out[gid] = float(gid);
}
"""


def _metal_project(tmp_path):
    _write(tmp_path / "kernels" / "names.h", "#define KERNEL_NAME first_kernel\n")
    _write(tmp_path / "kernels" / "kernel.metal", METAL_KERNEL)
    return ProjectConfig(root=tmp_path, include_dirs=("kernels",))


def _entry_names(report):
    (unit,) = report.units
    return [entry.name for entry in unit.entry_discovery.entries]


def test_project_rescans_reuse_discovery_until_a_header_changes(tmp_path):
    config = _metal_project(tmp_path)
    first = scan_project(config)
    assert _entry_names(first) == ["first_kernel"]
    second = scan_project(config)
    assert memo_module.ENTRY_DISCOVERY_MEMO.hits == 1
    assert [unit.entry_discovery for unit in second.units] == [
        unit.entry_discovery for unit in first.units
    ]

    _write(tmp_path / "kernels" / "names.h", "#define KERNEL_NAME second_kernel\n")
    third = scan_project(config)
    assert _entry_names(third) == ["second_kernel"]
    assert memo_module.ENTRY_DISCOVERY_MEMO.misses == 2


def test_discovery_is_not_reused_across_replaced_discoverers(tmp_path):
    _metal_project(tmp_path)
    path = tmp_path / "kernels" / "kernel.metal"
    spec = pipeline.SOURCE_REGISTRY.get("metal")
    calls = []

    def discoverer(code, **kwargs):
        calls.append(kwargs["file_path"])
        return spec.entry_point_discoverer(code, **kwargs)

    replaced = replace(spec, entry_point_discoverer=discoverer)
    results = [
        memo_module.memoized_entry_discovery(
            candidate,
            path.read_text(encoding="utf-8"),
            file_path=str(path),
            include_paths=[str(tmp_path / "kernels")],
            defines={},
            source_options={},
        )
        for candidate in (spec, replaced, replaced)
    ]
    assert len(calls) == 1
    assert results[0] == results[1] == results[2]


HLSL_KERNEL = """#include "common.hlsli"
RWStructuredBuffer<float> output : register(u0);
[numthreads(GROUP, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    output[id.x] = (float)id.x;
}
"""


def test_source_host_interface_reuse_is_scoped_to_identical_sources(tmp_path):
    root = tmp_path.resolve()
    _write(root / "kernels" / "common.hlsli", "#define GROUP 4\n")
    _write(root / "kernels" / "kernel.hlsl", HLSL_KERNEL)
    artifact = {
        "source": "kernels/kernel.hlsl",
        "sourceBackend": "directx",
        "target": "metal",
    }
    first = pipeline._runtime_manifest_source_host_interface(root, artifact)
    second = pipeline._runtime_manifest_source_host_interface(root, artifact)
    assert first == second and first["entryPointCount"] == 1
    assert memo_module.SOURCE_HOST_INTERFACE_MEMO.hits == 1
    first["entryPoints"].clear()
    assert pipeline._runtime_manifest_source_host_interface(root, artifact) == second
    _write(root / "kernels" / "common.hlsli", "#define GROUP 8\n")
    pipeline._runtime_manifest_source_host_interface(root, artifact)
    assert memo_module.SOURCE_HOST_INTERFACE_MEMO.misses == 2


def test_unmodelled_backends_bypass_the_memo(tmp_path):
    calls = []
    spec = SimpleNamespace(
        name="glsl",
        discover_entry_points=lambda *args, **kwargs: calls.append(1) or "result",
    )
    for _ in range(2):
        assert (
            memo_module.memoized_entry_discovery(
                spec,
                "void main() {}\n",
                file_path=str(tmp_path / "main.comp"),
                include_paths=[],
                defines={},
                source_options={},
            )
            == "result"
        )
    assert len(calls) == 2 and len(memo_module.ENTRY_DISCOVERY_MEMO) == 0
