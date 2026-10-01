"""Exact-grid regions preserve source coordinates and shared native allocations."""

import hashlib
import itertools
import json
import math
import os
import sys
from pathlib import Path

import pytest

from crosstl.project.native_runtime_drivers import (
    DirectXComputeRuntime,
    OpenGLComputeRuntime,
)
from crosstl.project.runtime_verification import (
    NativeRuntimeBufferBinding,
    NativeRuntimeDispatchRequest,
    RuntimeAllocationView,
    RuntimeDispatchGeometry,
    RuntimeResourceBinding,
    RuntimeValue,
)
from crosstl.translator import parse
from crosstl.translator.dispatch_region_lowering import specialize_dispatch_region
from crosstl.translator.dispatch_regions import DispatchRegion, plan_dispatch_regions
from tests.test_backend.test_metal.test_codegen import convert, parse_crossgl
from tests.test_translator.test_exact_thread_grid_runtime import (
    GRIDS,
    SOURCE,
    _coordinates,
)
from tests.test_translator.test_exact_thread_grid_runtime import (
    test_metal_exact_thread_grid_native as _metal_control,
)
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_votes import _codegen

REQUIRE_ENV = "CROSTL_REQUIRE_DISPATCH_REGIONS"


def test_region_execution_gate_is_required_on_every_native_target():
    from tools import ci_coverage

    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/mlx-portable-host.yml").read_text()
    step = ci_coverage.workflow_step_section(
        workflow, "Validate exact dispatch regions"
    )
    assert "test_dispatch_regions.py" in step
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "-n auto" in step and "continue-on-error" not in step and "if:" not in step
    assert (
        "--timeout-seconds 300" in step
        and "--junitxml=" in step
        and "--basetemp=" in step
    )
    for event in ("pull_request", "push"):
        for name in ("test_dispatch_regions.py", "test_exact_thread_grid_runtime.py"):
            assert (
                f"tests/test_translator/{name}"
                in ci_coverage.workflow_event_path_filters(workflow, event)
            )


def test_region_plan_exhaustively_covers_small_3d_grids():
    for grid in itertools.product(range(1, 5), repeat=3):
        for size in itertools.product(range(1, 4), repeat=3):
            regions = plan_dispatch_regions(grid, size)
            assert 1 <= len(regions) <= 8
            visited = set()
            for region in regions:
                for group in _coordinates(region.workgroup_count):
                    source_group = tuple(
                        g + o for g, o in zip(group, region.workgroup_offset)
                    )
                    for local in _coordinates(region.workgroup_size):
                        physical = tuple(
                            g * w + t
                            for g, w, t in zip(group, region.workgroup_size, local)
                        )
                        point = tuple(
                            t + o for t, o in zip(physical, region.thread_offset)
                        )
                        assert (
                            tuple(t // w for t, w in zip(point, size)) == source_group
                        )
                        assert all(t < n for t, n in zip(point, grid))
                        assert point not in visited
                        visited.add(point)
            assert visited == set(_coordinates(grid))


def test_region_plan_normalizes_dimensions_and_keeps_uint32_boundaries():
    grid = [0xFFFFFFFF, 9]
    size = [1024, 4]
    regions = plan_dispatch_regions(grid, size)
    assert len(regions) == 4
    grid[0] = size[0] = 1
    assert regions[-1].thread_offset == (0xFFFFFC00, 8, 0)
    assert regions[-1].workgroup_size == (1023, 1, 1)
    assert regions[-1].source_workgroup_count == (4194304, 3, 1)
    assert plan_dispatch_regions((5,), (32,))[0].to_json() == {
        "threadGridSize": [5, 1, 1],
        "sourceWorkgroupSize": [32, 1, 1],
        "workgroupOffset": [0, 0, 0],
        "workgroupCount": [1, 1, 1],
        "workgroupSize": [5, 1, 1],
    }


@pytest.mark.parametrize(
    "bad", [None, [], [0], [-1], [True], [1.5], [1 << 32], [1, 1, 1, 1], "32"]
)
@pytest.mark.parametrize("field", ["grid", "size"])
def test_region_plan_rejects_invalid_geometry(bad, field):
    with pytest.raises(ValueError):
        plan_dispatch_regions(
            bad if field == "grid" else [37], bad if field == "size" else [32]
        )


@pytest.mark.parametrize(
    "offset,count,size",
    [
        ([0], [2], [32]),
        ([1], [1], [32]),
        ([2], [1], [5]),
        ([-1], [1], [32]),
        ([0], [0], [32]),
    ],
)
def test_region_rejects_mixed_or_out_of_range_groups(offset, count, size):
    with pytest.raises(ValueError):
        DispatchRegion((37,), (32,), offset, count, size)


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("grid,size", GRIDS)
def test_region_specialization_compiles_without_changing_source_ast(
    tmp_path, target, grid, size
):
    ast = parse_crossgl(convert(SOURCE))
    before = _codegen(target).generate(ast)
    for index, region in enumerate(plan_dispatch_regions(grid, size)):
        lowered = specialize_dispatch_region(ast, region)
        generated = _codegen(target).generate(lowered)
        directory = tmp_path / str(index)
        directory.mkdir()
        _compile(generated, target, directory)
        assert lowered.annotations["dispatch_region"] == region.to_json()
        assert "crossglNumWorkGroups" not in generated
        with pytest.raises(ValueError, match="already specialized"):
            specialize_dispatch_region(lowered, region)
    assert not ast.annotations
    assert _codegen(target).generate(ast) == before


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("kind", ["uint", "int", "uvec2", "ivec2", "uvec3", "ivec3"])
def test_region_entry_coordinates_preserve_parameter_widths(tmp_path, target, kind):
    component = "" if kind in {"uint", "int"} else ".x"
    ast = parse(f"""shader Region {{
        RWStructuredBuffer<uint> result @register(u0);
        compute {{ void main({kind} tid @gl_GlobalInvocationID,
                            {kind} gid @gl_WorkGroupID,
                            {kind} grid @threads_per_grid,
                            {kind} count @gl_NumWorkGroups) {{
            uint crossglPhysical_tid = 7u;
            result[uint(tid{component})] = uint(gid{component} + grid{component} + count{component}) + crossglPhysical_tid;
        }} }}
    }}""")
    region = plan_dispatch_regions((37,), (32,))[-1]
    generated = _codegen(target, software=False).generate(
        specialize_dispatch_region(ast, region)
    )
    _compile(generated, target, tmp_path)
    assert "37" in generated and "32" in generated


@pytest.mark.parametrize("target", ["directx", "opengl"])
def test_region_rebases_builtin_reads_in_helpers(tmp_path, target):
    ast = parse("""shader Region {
        RWStructuredBuffer<uint> result @register(u0);
        uint helper() { return gl_GlobalInvocationID.x + gl_WorkGroupID.x + gl_NumWorkGroups.x; }
        compute { void main() {
            uint index = gl_GlobalInvocationID.x;
            uint total = helper() + threads_per_grid.x;
            result[index] = total + threads_per_grid.x;
        } }
    }""")
    lowered = specialize_dispatch_region(ast, plan_dispatch_regions((37,), (32,))[-1])
    generated = _codegen(target, software=False).generate(lowered)
    _compile(generated, target, tmp_path)
    assert "crossglNumWorkGroups" not in generated


def test_region_ast_preserves_source_owned_builtin_names():
    ast = parse("""shader Region {
        RWStructuredBuffer<uint> result @register(u0);
        uint owned(uint threads_per_grid) { return threads_per_grid; }
        compute { void main() {
            uint total = threads_per_grid.x;
            { uint threads_per_grid = 7u; total += owned(threads_per_grid); }
            for (uint threads_per_grid = 0u; threads_per_grid < 2u; ++threads_per_grid) { total += threads_per_grid; }
            result[gl_GlobalInvocationID.x] = total + threads_per_grid.x;
        } }
    }""")
    lowered = specialize_dispatch_region(ast, plan_dispatch_regions((37,), (32,))[-1])
    from crosstl.translator.ast import IdentifierNode, VariableNode
    from crosstl.translator.dispatch_region_lowering import _walk

    references = [
        n
        for n in _walk(lowered)
        if isinstance(n, IdentifierNode) and n.name == "threads_per_grid"
    ]
    declarations = [
        n
        for n in _walk(lowered)
        if isinstance(n, VariableNode) and n.name == "threads_per_grid"
    ]
    assert len(references) == 5
    assert len(declarations) == 2


def test_region_specialization_requires_one_compute_entry_and_valid_coordinates():
    region = plan_dispatch_regions([37], [32])[-1]
    for source in (
        "shader Empty {}",
        "shader Graphics { vertex { void main() {} } }",
        "shader Multiple { compute { void first() {} } compute { void second() {} } }",
        "shader Bad { compute { void main(float tid @gl_GlobalInvocationID) {} } }",
    ):
        with pytest.raises(ValueError):
            specialize_dispatch_region(parse(source), region)


def test_region_rejects_builtin_global_initializer_before_emission():
    ast = parse(
        "shader Init { uint x = gl_GlobalInvocationID.x; compute { void main() {} } }"
    )
    with pytest.raises(ValueError, match="global initializers"):
        specialize_dispatch_region(ast, plan_dispatch_regions([37], [32])[-1])


def test_region_requires_one_entry_even_with_qualified_graphics_functions():
    from crosstl.translator.ast import BlockNode, FunctionNode, PrimitiveType

    ast = parse("shader Mixed { compute { void main() {} } }")
    ast.functions.append(
        FunctionNode(
            "geometry_entry",
            PrimitiveType("void"),
            [],
            BlockNode([]),
            qualifiers=["geometry"],
        )
    )
    with pytest.raises(ValueError, match="exactly one compute entry"):
        specialize_dispatch_region(ast, plan_dispatch_regions([37], [32])[-1])


def test_region_global_initializer_can_call_source_owned_function():
    ast = parse(
        "shader Owned { uint threads_per_grid() { return 7u; } uint x = threads_per_grid(); compute { void main() {} } }"
    )
    lowered = specialize_dispatch_region(ast, plan_dispatch_regions([37], [32])[-1])
    assert lowered.global_variables[0].initial_value.function.name == "threads_per_grid"


@pytest.mark.parametrize(
    "semantic",
    ["gl_GlobalInvocationID", "gl_WorkGroupID", "threads_per_grid", "gl_NumWorkGroups"],
)
def test_region_rejects_signed_coordinate_overflow(semantic):
    ast = parse(
        f"shader Signed {{ compute {{ void main(int tid @{semantic}) {{}} }} }}"
    )
    with pytest.raises(ValueError, match="signed parameter range"):
        specialize_dispatch_region(ast, plan_dispatch_regions([0xFFFFFFFF], [1])[0])


def test_region_preserves_source_owned_global_and_function_identifiers():
    from crosstl.translator.ast import IdentifierNode
    from crosstl.translator.dispatch_region_lowering import _walk

    for declaration, expression in (
        ("uint threads_per_grid = 7u;", "threads_per_grid"),
        ("uint threads_per_grid() { return 7u; }", "threads_per_grid()"),
    ):
        ast = parse(
            f"shader Owned {{ {declaration} compute {{ void main() {{ uint value = {expression}; }} }} }}"
        )
        lowered = specialize_dispatch_region(ast, plan_dispatch_regions([37], [32])[-1])
        assert any(
            isinstance(node, IdentifierNode) and node.name == "threads_per_grid"
            for node in _walk(lowered)
        )


def _case(grid, size):
    counts = tuple((n + w - 1) // w for n, w in zip(grid, size))
    padded = math.prod(n * w for n, w in zip(counts, size))
    inputs = [(i * 17 + 3) % 251 for i in range(padded + 17)]
    initial = [0xFFFFFFFF] * (34 + padded * 22)
    expected = initial.copy()
    for tid in _coordinates(grid):
        gid = tuple(t // w for t, w in zip(tid, size))
        lid = tuple(t % w for t, w in zip(tid, size))
        active = tuple(min(w, n - g * w) for n, w, g in zip(grid, size, gid))
        local_index = lid[0] + active[0] * (lid[1] + active[1] * lid[2])
        sid, lane = divmod(local_index, 32)
        total = 0
        for linear in range(sid * 32, min((sid + 1) * 32, math.prod(active))):
            local = (
                linear % active[0],
                (linear // active[0]) % active[1],
                linear // (active[0] * active[1]),
            )
            point = tuple(g * w + l for g, w, l in zip(gid, size, local))
            index = point[0] + grid[0] * (point[1] + grid[1] * point[2])
            total += inputs[index]
        index = tid[0] + grid[0] * (tid[1] + grid[1] * tid[2])
        expected[17 + index * 22 : 17 + (index + 1) * 22] = [
            *tid,
            *gid,
            *grid,
            *active,
            *lid,
            *counts,
            lane,
            sid,
            (math.prod(active) + 31) // 32,
            total,
        ]
    return inputs, initial, expected


@pytest.mark.parametrize("grid,size", GRIDS)
def test_exact_grid_regions_execute_natively(tmp_path, monkeypatch, grid, size):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 to require native dispatch region checks")
    if sys.platform == "darwin":
        monkeypatch.setenv("CROSTL_REQUIRE_METAL_PACKAGE_RUNTIME", "1")
        _metal_control(tmp_path, grid, size)
        return
    target = {"win32": "directx", "linux": "opengl"}[sys.platform]
    inputs, initial, expected = _case(grid, size)
    ast = parse_crossgl(convert(SOURCE))
    requests = []
    evidence = []
    for index, region in enumerate(plan_dispatch_regions(grid, size)):
        directory = tmp_path / str(index)
        directory.mkdir()
        generated = _codegen(target).generate(specialize_dispatch_region(ast, region))
        artifact, module = _compile(generated, target, directory)
        assert module.is_file(), "Required native compiler did not produce a module"
        bindings = {}
        for binding, (name, values) in enumerate(
            (("inputWords", inputs), ("outputWords", initial))
        ):
            bindings[name] = NativeRuntimeBufferBinding(
                name=name,
                binding=RuntimeResourceBinding(
                    name=name,
                    kind="storage-buffer",
                    type_name="RWStructuredBuffer<uint>",
                    set=0,
                    binding=binding,
                    access="read_write",
                ),
                value=values if index == 0 else None,
                source="input" if index == 0 else None,
                dtype="uint32",
                shape=(len(values),),
                allocation=RuntimeAllocationView(
                    allocation_id=name,
                    byte_length=4 * len(values),
                    allocation_byte_length=4 * len(values),
                ),
                expected_output=RuntimeValue(
                    name=name,
                    kind="buffer",
                    dtype="uint32",
                    shape=(len(values),),
                    values=inputs if name == "inputWords" else expected,
                ),
            )
        entry = "CSMain" if target == "directx" else "main"
        requests.append(
            NativeRuntimeDispatchRequest(
                target=target,
                artifact={"target": target, "id": str(index)},
                artifact_path=artifact,
                module_path=module,
                loaded_artifact=(
                    module.read_bytes() if target == "directx" else generated
                ),
                buffers=bindings,
                constants={},
                entry_point=entry,
                dispatch=RuntimeDispatchGeometry(
                    entry_point=entry,
                    workgroup_size=region.workgroup_size,
                    workgroup_count=region.workgroup_count,
                ),
            )
        )
        evidence.append(
            {
                "region": region.to_json(),
                "artifactSha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "moduleSha256": hashlib.sha256(module.read_bytes()).hexdigest(),
            }
        )
    runtime = (
        DirectXComputeRuntime()
        if target == "directx"
        else OpenGLComputeRuntime(context_backends=("egl",))
    )
    result = runtime.dispatch_sequence(None, None, requests)
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            {
                "grid": grid,
                "size": size,
                "sourceSha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
                "regions": evidence,
                "inputs": inputs,
                "initial": initial,
                "expected": expected,
                "result": result,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert result["inputWords"]["values"] == inputs
    assert result["outputWords"]["values"] == expected
