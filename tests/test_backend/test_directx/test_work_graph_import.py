"""HLSL input for work-graph nodes: a ``[Shader("node")]`` entry imports as a CrossGL ``node`` stage."""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.stage_refusals import StageUnsupportedError

RECORDS = """
struct [NodeTrackRWInputSharing] Record
{
    uint3 grid : SV_DispatchGrid;
    uint value;
};

struct Child
{
    uint id;
};

RWStructuredBuffer<uint> out_buf : register(u0);
groupshared uint total;
"""

BROADCASTING = RECORDS + """
[SHADER("node")]
[NodeLaunch("broadcasting")]
[NodeMaxDispatchGrid(4, 1, 1)]
[NodeMaxRecursionDepth(3)]
[NodeID("first", 2)]
[NodeShareInputOf("other")]
[NodeIsProgramEntry]
[NodeLocalRootArgumentsTableIndex(0)]
[NumThreads(4, 1, 1)]
void main(globallycoherent RWDispatchNodeInputRecord<Record> obj,
          uint3 group : SV_GroupID,
          uint index : SV_GroupIndex,
          [MaxRecords(4)] [NodeID("next")] [NodeArraySize(3)] NodeOutputArray<Child> outs,
          [MaxRecordsSharedWith(outs)] [AllowSparseNodes] EmptyNodeOutput done)
{
    GroupNodeOutputRecords<Child> r = outs[1].GetGroupNodeOutputRecords(1);
    r[0].id = obj.Get().value + group.x + index;
    Barrier(r, GROUP_SCOPE | GROUP_SYNC);
    r.OutputComplete();
    done.ThreadIncrementOutputCount(1);
    Barrier(NODE_OUTPUT_MEMORY, DEVICE_SCOPE);
    out_buf[index] = obj.FinishedCrossGroupSharing() ? GetRemainingRecursionLevels() : 0u;
}
"""

THREAD = RECORDS + """
[Shader("node")]
[NodeLaunch("thread")]
[NodeID("Self")]
[NodeMaxRecursionDepth(4)]
void main(RWThreadNodeInputRecord<Child> input, [MaxRecords(1)] [NodeID("Self")] NodeOutput<Child> self_out)
{
    input.Get().id += 1;
    if (GetRemainingRecursionLevels() > 0)
    {
        ThreadNodeOutputRecords<Child> r = self_out.GetThreadNodeOutputRecords(1);
        r.Get().id = input.Get().id;
        r.OutputComplete();
    }
    out_buf[0] = input.Get().id;
}
"""

COALESCING = RECORDS + """
[Shader("node")]
[NodeLaunch("coalescing")]
[NumThreads(8, 1, 1)]
void main([MaxRecords(8)] GroupNodeInputRecords<Child> inputs, uint tid : SV_GroupIndex,
          [MaxRecords(1)] EmptyNodeOutput done)
{
    if (tid < inputs.Count())
    {
        InterlockedAdd(total, inputs[tid].id);
    }
    Barrier(GROUP_SHARED_MEMORY, GROUP_SCOPE | GROUP_SYNC);
    if (tid == 0)
    {
        out_buf[0] = total;
        done.GroupIncrementOutputCount(1);
    }
}
"""


def _write(tmp_path, source, name="node.hlsl"):
    path = tmp_path / name
    path.write_text(source)
    return path


def _crossgl(tmp_path, source):
    return crosstl.translate(str(_write(tmp_path, source)), backend="cgl")


def _compact(code):
    return " ".join(code.split())


def _dxc(tmp_path, code, name):
    """Compile ``code`` as a shader model 6.8 library with warnings as errors, when DXC is installed."""
    if shutil.which("dxc") is None:
        return
    path = _write(tmp_path, code, name)
    result = subprocess.run(
        [
            "dxc",
            "-T",
            "lib_6_8",
            "-WX",
            "-Fo",
            str(path.with_suffix(".dxil")),
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _round_trip(tmp_path, source):
    """The HLSL CrossTL writes back from the CrossGL it imports ``source`` to; DXC must accept both."""
    _dxc(tmp_path, source, "source.hlsl")
    crossgl = _write(tmp_path, _crossgl(tmp_path, source), "node.cgl")
    hlsl = crosstl.translate(str(crossgl), backend="directx")
    _dxc(tmp_path, hlsl, "round_trip.hlsl")
    return _compact(hlsl)


@pytest.mark.parametrize("shader", ["Shader", "shader"])
def test_broadcasting_node_imports_every_attribute(shader, tmp_path):
    code = _compact(_crossgl(tmp_path, BROADCASTING.replace("SHADER", shader)))
    assert "node {" in code
    for spelling in (
        "@node_track_rw_input_sharing struct Record",
        "uvec3 grid @ SV_DispatchGrid;",
        '@ node_launch("broadcasting")',
        "@ node_max_dispatch_grid(4, 1, 1)",
        "@ node_max_recursion_depth(3)",
        '@ node_id("first", 2)',
        '@ node_share_input_of("other")',
        "@ node_is_program_entry",
        "@ node_local_root_arguments_table_index(0)",
        "@ numthreads(4, 1, 1)",
        "@globallycoherent RWDispatchNodeInputRecord<Record> obj",
        "uvec3 group @ gl_WorkGroupID",
        "uint index @ gl_LocalInvocationIndex",
        '@max_records(4) @node_id("next") @node_array_size(3) NodeOutputArray<Child> outs',
        "@max_records_shared_with(outs) @allow_sparse_nodes EmptyNodeOutput done",
        "Barrier(r, GROUP_SCOPE | GROUP_SYNC);",
        "obj.FinishedCrossGroupSharing() ? GetRemainingRecursionLevels() : 0u",
    ):
        assert spelling in code, spelling
    assert "compute {" not in code


@pytest.mark.parametrize(
    "source, expected",
    [
        (
            BROADCASTING.replace("SHADER", "Shader"),
            (
                '[NodeLaunch("broadcasting")]',
                '[NodeID("first", 2)]',
                '[NodeShareInputOf("other")]',
                "globallycoherent RWDispatchNodeInputRecord<Record> obj",
                '[MaxRecords(4)][NodeID("next")][NodeArraySize(3)] NodeOutputArray<Child> outs',
                "[MaxRecordsSharedWith(outs)][AllowSparseNodes] EmptyNodeOutput done",
            ),
        ),
        (
            THREAD,
            (
                '[NodeLaunch("thread")]',
                '[NodeID("Self")]',
                "[NodeMaxRecursionDepth(4)]",
                '[MaxRecords(1)][NodeID("Self")] NodeOutput<Child> self_out',
            ),
        ),
        (
            COALESCING,
            (
                '[NodeLaunch("coalescing")]',
                "[NumThreads(8, 1, 1)]",
                "[MaxRecords(8)] GroupNodeInputRecords<Child> inputs",
                ": SV_GroupIndex",
            ),
        ),
    ],
    ids=["broadcasting", "thread", "coalescing"],
)
def test_nodes_round_trip_to_hlsl_dxc_accepts(source, expected, tmp_path):
    code = _round_trip(tmp_path, source)
    assert '[shader("node")]' in code
    for spelling in expected:
        assert spelling in code, spelling


@pytest.mark.parametrize("target", ["metal", "opengl", "vulkan", "cuda"])
def test_imported_nodes_are_refused_by_targets_without_node_shaders(target, tmp_path):
    crossgl = _write(tmp_path, _crossgl(tmp_path, THREAD), "node.cgl")
    with pytest.raises(StageUnsupportedError) as raised:
        crosstl.translate(str(crossgl), backend=target)
    assert raised.value.stages == ("node",)
