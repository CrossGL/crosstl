"""HLSL output for CrossGL work-graph nodes: ``[Shader("node")]`` entries of a shader model 6.8 library."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.hlsl_work_graph_nodes import HlslWorkGraphNodeError
from crosstl.translator.semantics import work_graph_nodes
from tests.test_translator.test_work_graph_nodes import FULL, PROBES


def _hlsl(tmp_path, source, name="probe.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="directx")


def _dxc(tmp_path, code, name="node.hlsl"):
    """Compile ``code`` as a shader model 6.8 library with warnings as errors, when DXC is installed."""
    if shutil.which("dxc") is None:
        return
    path = tmp_path / name
    path.write_text(code)
    result = subprocess.run(
        ["dxc", "-T", "lib_6_8", "-WX", "-Fo", str(tmp_path / "node.dxil"), str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _compact(code):
    return " ".join(code.split())


def _node(parameters, body, attributes, members="uint value;", extra=""):
    lines = "\n".join(f"            {line}" for line in body)
    attributes = "\n".join(f"            {line}" for line in attributes)
    return f"""
shader Probe {{
    struct Record {{
        {members}
    }}
    {extra}
    node {{
        buffer uint* out_buf;
        groupshared uint total;
        void main({parameters})
{attributes} {{
{lines}
        }}
    }}
}}
"""


BROADCASTING = (
    "@node_launch(broadcasting)",
    "@node_dispatch_grid(2, 1, 1)",
    "@numthreads(4, 1, 1)",
)


@pytest.mark.parametrize("target", ["directx", "hlsl", "dx12", "d3d12"])
def test_hlsl_target_names_write_node_shaders(target):
    assert target in work_graph_nodes.work_graph_targets()


@pytest.mark.parametrize("name", sorted(PROBES))
def test_probes_are_node_shaders_dxc_accepts(name, tmp_path):
    code = _hlsl(tmp_path, PROBES[name])
    assert '[shader("node")]' in code
    _dxc(tmp_path, code)


def test_every_attribute_is_written_in_its_hlsl_spelling(tmp_path):
    code = _compact(_hlsl(tmp_path, FULL))
    for spelling in (
        '[NodeLaunch("broadcasting")]',
        "[NodeIsProgramEntry]",
        '[NodeID("first", 2)]',
        "[NodeLocalRootArgumentsTableIndex(0)]",
        '[NodeShareInputOf("other")]',
        "[NodeMaxDispatchGrid(4, 1, 1)]",
        "[NodeMaxRecursionDepth(3)]",
        "[NumThreads(4, 1, 1)]",
        "globallycoherent RWDispatchNodeInputRecord<Record> obj",
        '[MaxRecords(4)][NodeID("next")][NodeArraySize(3)] NodeOutputArray<Record> outs',
        "[MaxRecordsSharedWith(outs)][AllowSparseNodes] EmptyNodeOutput done",
        "uint3 grid : SV_DispatchGrid;",
        "Barrier(NODE_OUTPUT_MEMORY, DEVICE_SCOPE);",
        "obj.FinishedCrossGroupSharing()",
    ):
        assert spelling in code, spelling
    # The attribute sits between ``struct`` and the name (D3D12 Work Graphs, "NodeTrackRWInputSharing").
    assert re.search(r"struct\s*\[NodeTrackRWInputSharing\]\s*Record\b", code)
    # Entry attributes come before [shader("node")] in the order the D3D12 work-graph specification lists them.
    assert (
        code.index("[NodeLaunch(")
        < code.index("[NodeMaxDispatchGrid(")
        < code.index("[NumThreads(")
    )
    assert code.index("[NumThreads(") < code.index('[shader("node")]')


def test_full_node_compiles(tmp_path):
    # FULL reads ``r`` in a barrier after ``r.OutputComplete()``, which DXC's validator rejects as a read of an
    # uninitialized handle; completing the records after the barrier gives the same node a valid body.
    complete, barrier = "r.OutputComplete();", "Barrier(r, GROUP_SCOPE | GROUP_SYNC);"
    assert FULL.index(complete) < FULL.index(barrier)
    valid = FULL.replace(complete, "", 1).replace(barrier, f"{barrier} {complete}", 1)
    _dxc(tmp_path, _hlsl(tmp_path, valid))


def test_system_values_become_entry_parameters_of_a_broadcasting_node(tmp_path):
    source = _node(
        "DispatchNodeInputRecord<Record> input",
        [
            "out_buf[gl_GlobalInvocationID.x] = input.Get().value + gl_WorkGroupID.x"
            " + gl_LocalInvocationID.x + gl_LocalInvocationIndex;"
        ],
        BROADCASTING,
    )
    code = _compact(_hlsl(tmp_path, source))
    for parameter in (
        "uint3 dispatchThreadID : SV_DispatchThreadID",
        "uint3 groupThreadID : SV_GroupThreadID",
        "uint3 groupID : SV_GroupID",
        "uint groupIndex : SV_GroupIndex",
    ):
        assert parameter in code, parameter
    _dxc(tmp_path, code)


def test_coalescing_node_reads_its_place_in_the_group(tmp_path):
    source = _node(
        "@max_records(8) GroupNodeInputRecords<Record> inputs, uint tid @ gl_LocalInvocationIndex",
        [
            "if (tid < inputs.Count()) {",
            "    InterlockedAdd(total, inputs[tid].value);",
            "}",
            "Barrier(GROUP_SHARED_MEMORY, GROUP_SCOPE | GROUP_SYNC);",
            "if (gl_LocalInvocationID.x == 0u) {",
            "    out_buf[0] = total;",
            "}",
        ],
        ("@node_launch(coalescing)", "@numthreads(8, 1, 1)"),
    )
    code = _compact(_hlsl(tmp_path, source))
    assert '[NodeLaunch("coalescing")]' in code
    assert "[MaxRecords(8)] GroupNodeInputRecords<Record> inputs" in code
    assert "SV_GroupIndex" in code and "SV_GroupThreadID" in code
    _dxc(tmp_path, code)


@pytest.mark.parametrize(
    "launch, builtin, read",
    [
        ("coalescing", "gl_GlobalInvocationID", "gl_GlobalInvocationID.x"),
        ("coalescing", "gl_WorkGroupID", "gl_WorkGroupID.x"),
        ("thread", "gl_LocalInvocationIndex", "gl_LocalInvocationIndex"),
        ("thread", "gl_LocalInvocationID", "gl_LocalInvocationID.x"),
    ],
)
def test_system_values_the_launch_lacks_are_refused(launch, builtin, read, tmp_path):
    record = (
        "@max_records(4) GroupNodeInputRecords<Record> inputs"
        if launch == "coalescing"
        else "ThreadNodeInputRecord<Record> inputs"
    )
    attributes = (f"@node_launch({launch})",) + (
        ("@numthreads(4, 1, 1)",) if launch == "coalescing" else ()
    )
    source = _node(record, [f"out_buf[0] = {read};"], attributes)
    with pytest.raises(HlslWorkGraphNodeError) as raised:
        _hlsl(tmp_path, source)
    assert raised.value.code.endswith("directx-work-graph-system-value")
    assert builtin in str(raised.value)


def test_thread_launch_recursion_writes_the_node_id_it_calls(tmp_path):
    source = _node(
        'RWThreadNodeInputRecord<Record> input, @max_records(1) @node_id("Self") NodeOutput<Record> self_out',
        [
            "input.Get().value += 1u;",
            "if (GetRemainingRecursionLevels() > 0u) {",
            "    ThreadNodeOutputRecords<Record> r = self_out.GetThreadNodeOutputRecords(1u);",
            "    r.Get().value = input.Get().value;",
            "    r.OutputComplete();",
            "}",
            "out_buf[0] = input.Get().value;",
        ],
        ("@node_launch(thread)", "@node_max_recursion_depth(4)", '@node_id("Self")'),
    )
    code = _compact(_hlsl(tmp_path, source))
    assert '[NodeLaunch("thread")]' in code
    assert '[NodeID("Self")]' in code and "[NodeMaxRecursionDepth(4)]" in code
    _dxc(tmp_path, code)


def test_renamed_outputs_keep_their_crossgl_node_ids(tmp_path):
    """HLSL reserves ``line`` and ``point``; the outputs are renamed and [NodeID] keeps the node names."""
    source = _node(
        "@max_records(2) NodeOutput<Record> line, @max_records_shared_with(line) EmptyNodeOutput point",
        [
            "ThreadNodeOutputRecords<Record> r = line.GetThreadNodeOutputRecords(1u);",
            "r.Get().value = 3u;",
            "r.OutputComplete();",
            "point.ThreadIncrementOutputCount(1u);",
        ],
        BROADCASTING,
    )
    code = _compact(_hlsl(tmp_path, source))
    assert '[NodeID("line")][MaxRecords(2)] NodeOutput<Record> line_' in code
    assert (
        '[NodeID("point")][MaxRecordsSharedWith(line_)] EmptyNodeOutput point_' in code
    )
    _dxc(tmp_path, code)


@pytest.mark.parametrize(
    "parameters, message",
    [
        (
            "@max_records(2) @node_id(3) NodeOutput<Record> outs",
            "'@node_id' names a node with 3, not a string",
        ),
        (
            "@max_records(2) NodeOutput<Record> outs, @max_records_shared_with(nothing) EmptyNodeOutput done",
            "'@max_records_shared_with(nothing)' names no parameter of the node",
        ),
    ],
)
def test_attributes_hlsl_cannot_write_are_refused(parameters, message, tmp_path):
    source = _node(parameters, ["out_buf[0] = 1u;"], BROADCASTING)
    with pytest.raises(HlslWorkGraphNodeError) as raised:
        _hlsl(tmp_path, source)
    assert raised.value.code.endswith("directx-work-graph-attribute")
    assert message in str(raised.value)
