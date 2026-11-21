"""CrossGL work-graph nodes: the node stage, the rules DXC enforces for it, and the refusal for other targets."""

import pytest

import crosstl
from crosstl.translator import parse
from crosstl.translator.ast import ShaderStage
from crosstl.translator.codegen.stage_refusals import StageUnsupportedError
from crosstl.translator.semantics import (
    WorkGraphNodeError,
    WorkGraphUnsupportedError,
    apply_rule_carrying,
    work_graph_nodes,
)
from crosstl.translator.semantics.work_graph_nodes import (
    check_work_graph_nodes,
    refuse_work_graph_nodes,
    register_work_graph_target,
)


def _node(parameters, body, attributes, record_attribute="", members="uint value;"):
    """A program whose node stage takes ``parameters`` and runs ``body`` under the entry ``attributes``."""
    lines = "\n".join(f"            {line}" for line in body)
    attributes = "\n".join(f"            {line}" for line in attributes)
    return f"""
shader Probe {{
    {record_attribute}
    struct Record {{
        {members}
    }}
    node {{
        buffer uint* out_buf;
        void main({parameters})
{attributes} {{
            uint i = 0u;
{lines}
        }}
    }}
}}
"""


BROADCASTING = (
    "@node_launch(broadcasting)",
    "@node_dispatch_grid(1, 1, 1)",
    "@numthreads(1, 1, 1)",
)
COALESCING = ("@node_launch(coalescing)", "@numthreads(1, 1, 1)")

# The CrossGL forms of the eleven HLSL work-graph probes (tools/coverage/hlsl.py, ``HLSL.node``).
PROBES = {
    "GetRemainingRecursionLevels": _node(
        "", ["out_buf[i] = GetRemainingRecursionLevels();"], BROADCASTING
    ),
    "EmptyNodeInput.Count": _node(
        "@max_records(1) EmptyNodeInput obj", ["out_buf[i] = obj.Count();"], COALESCING
    ),
    "EmptyNodeOutput.GroupIncrementOutputCount": _node(
        "@max_records(1) EmptyNodeOutput obj",
        ["obj.GroupIncrementOutputCount(1u);"],
        BROADCASTING,
    ),
    "EmptyNodeOutput.IsValid": _node(
        "@max_records(1) EmptyNodeOutput obj",
        ["out_buf[i] = obj.IsValid() ? 1u : 0u;"],
        BROADCASTING,
    ),
    "EmptyNodeOutput.ThreadIncrementOutputCount": _node(
        "@max_records(1) EmptyNodeOutput obj",
        ["obj.ThreadIncrementOutputCount(1u);"],
        BROADCASTING,
    ),
    "GroupNodeInputRecords.Count": _node(
        "@max_records(1) GroupNodeInputRecords<Record> obj",
        ["out_buf[i] = obj.Count();"],
        COALESCING,
    ),
    "GroupOrThreadNodeOutputRecords.OutputComplete": _node(
        "@max_records(1) NodeOutput<Record> output",
        [
            "ThreadNodeOutputRecords<Record> obj = output.GetThreadNodeOutputRecords(1u);",
            "obj.Get().value = 7u;",
            "obj.OutputComplete();",
        ],
        BROADCASTING,
    ),
    "NodeOutput.GetGroupNodeOutputRecords": _node(
        "@max_records(1) NodeOutput<Record> obj",
        [
            "GroupNodeOutputRecords<Record> r = obj.GetGroupNodeOutputRecords(1u);",
            "r[0].value = 7u;",
            "r.OutputComplete();",
        ],
        BROADCASTING,
    ),
    "NodeOutput.GetThreadNodeOutputRecords": _node(
        "@max_records(1) NodeOutput<Record> obj",
        [
            "ThreadNodeOutputRecords<Record> r = obj.GetThreadNodeOutputRecords(1u);",
            "r.Get().value = 7u;",
            "r.OutputComplete();",
        ],
        BROADCASTING,
    ),
    "NodeOutput.IsValid": _node(
        "@max_records(1) NodeOutput<Record> obj",
        ["out_buf[i] = obj.IsValid() ? 1u : 0u;"],
        BROADCASTING,
    ),
    "RWDispatchNodeInputRecord.FinishedCrossGroupSharing": _node(
        "@globallycoherent RWDispatchNodeInputRecord<Record> obj",
        ["out_buf[i] = obj.FinishedCrossGroupSharing() ? 1u : 0u;"],
        BROADCASTING,
        record_attribute="@node_track_rw_input_sharing",
    ),
}

# Every attribute, input form, system value and barrier the IR defines, in one broadcasting node.
FULL = _node(
    "@globallycoherent RWDispatchNodeInputRecord<Record> obj, uvec3 group @ SV_GroupID, "
    "uvec3 thread @ gl_LocalInvocationID, uvec3 dispatch @ gl_GlobalInvocationID, uint index @ SV_GroupIndex, "
    '@max_records(4) @node_id("next") @node_array_size(3) NodeOutputArray<Record> outs, '
    "@max_records_shared_with(outs) @allow_sparse_nodes EmptyNodeOutput done",
    [
        "GroupNodeOutputRecords<Record> r = outs[1].GetGroupNodeOutputRecords(1u);",
        "r[0].value = obj.Get().value + group.x + thread.x + dispatch.x + index;",
        "r.OutputComplete();",
        "done.ThreadIncrementOutputCount(1u);",
        "Barrier(NODE_OUTPUT_MEMORY, DEVICE_SCOPE);",
        "Barrier(r, GROUP_SCOPE | GROUP_SYNC);",
        "out_buf[i] = obj.FinishedCrossGroupSharing() ? GetRemainingRecursionLevels() : 0u;",
    ],
    (
        '@node_launch("broadcasting")',
        "@node_max_dispatch_grid(4, 1, 1)",
        "@node_max_recursion_depth(3)",
        '@node_id("first", 2)',
        '@node_share_input_of("other")',
        "@node_is_program_entry",
        "@node_local_root_arguments_table_index(0)",
        "@numthreads(4, 1, 1)",
    ),
    record_attribute="@node_track_rw_input_sharing",
    members="uvec3 grid @ SV_DispatchGrid;\n        uint value;",
)

ALL_TARGETS = [
    "directx",
    "metal",
    "opengl",
    "vulkan",
    "slang",
    "wgsl",
    "webgl",
    "cuda",
    "hip",
    "rust",
    "mojo",
]


@pytest.fixture
def node_target(monkeypatch):
    """Register a stand-in target that writes node shaders, for the duration of one test."""
    monkeypatch.setattr(work_graph_nodes, "_TARGETS", set(work_graph_nodes._TARGETS))
    register_work_graph_target("node-test")
    return "node-test"


def test_node_block_is_a_node_stage():
    program = parse(FULL)
    (stage,) = program.stages.values()
    assert stage.stage is ShaderStage.NODE
    names = [attribute.name for attribute in stage.entry_point.attributes]
    assert names[:2] == ["node_launch", "node_max_dispatch_grid"]


@pytest.mark.parametrize("name", sorted(PROBES))
def test_crossgl_output_keeps_the_node(name, tmp_path):
    path = tmp_path / "probe.cgl"
    path.write_text(PROBES[name], newline="")
    first = crosstl.translate(str(path), backend="cgl")
    assert "node {" in first
    path.write_text(first, newline="")
    assert crosstl.translate(str(path), backend="cgl") == first


@pytest.mark.parametrize("name", sorted(PROBES))
def test_probes_pass_the_shared_passes_for_a_node_target(name, node_target):
    apply_rule_carrying(parse(PROBES[name]), "cgl", node_target)


def test_every_construct_passes_the_shared_passes_for_a_node_target(node_target):
    apply_rule_carrying(parse(FULL), "cgl", node_target)


@pytest.mark.parametrize("target", ALL_TARGETS)
def test_targets_without_node_shaders_refuse_the_stage(target, tmp_path):
    path = tmp_path / "probe.cgl"
    path.write_text(FULL)
    with pytest.raises(StageUnsupportedError) as raised:
        crosstl.translate(str(path), backend=target)
    assert raised.value.stages == ("node",)


COMPUTE_WITH = """
shader Probe {{
    struct Record {{
        {member}
    }}
    compute {{
        buffer uint* out_buf;
        void main() @numthreads(1, 1, 1) {{
            {statement}
        }}
    }}
}}
"""


@pytest.mark.parametrize(
    ("member", "statement", "construct"),
    [
        ("uint value;", "out_buf[0] = GetRemainingRecursionLevels();", "intrinsic"),
        ("uint value;", "Barrier(NODE_INPUT_MEMORY, GROUP_SCOPE);", "barrier flag"),
        ("uvec3 grid @ SV_DispatchGrid;", "out_buf[0] = 1u;", "SV_DispatchGrid"),
    ],
)
def test_work_graph_constructs_outside_a_node_stage_are_refused(
    member, statement, construct, node_target
):
    program = parse(COMPUTE_WITH.format(member=member, statement=statement))
    for target in ("opengl", node_target):
        with pytest.raises(WorkGraphUnsupportedError) as raised:
            refuse_work_graph_nodes(program, target)
        assert construct in str(raised.value)


def test_work_graph_types_outside_a_node_stage_are_refused(node_target):
    source = """
shader Probe {
    struct Record { uint value; }
    node {
        void main() @node_dispatch_grid(1, 1, 1) @numthreads(1, 1, 1) {}
    }
    compute {
        void main(@max_records(1) NodeOutput<Record> obj) @numthreads(1, 1, 1) {}
    }
}
"""
    with pytest.raises(WorkGraphUnsupportedError, match="work-graph type NodeOutput"):
        refuse_work_graph_nodes(parse(source), node_target)


def _rule(source):
    with pytest.raises(WorkGraphNodeError) as raised:
        check_work_graph_nodes(parse(source), "node-test")
    return raised.value.rule


# Each program breaks one rule that DXC 1.9.2602.24 enforces under lib_6_8; the comment gives DXC's error.
VIOLATIONS = {
    "unknown launch mode": (
        _node("", [], ("@node_launch(sometimes)", "@numthreads(1, 1, 1)")),
        "launch-mode",
    ),
    # "Broadcasting node shader 'b1' must have either the NodeDispatchGrid or NodeMaxDispatchGrid attribute"
    "broadcasting without a grid": (
        _node("", [], ("@node_launch(broadcasting)", "@numthreads(1, 1, 1)")),
        "dispatch-grid",
    ),
    # A missing launch mode is broadcasting.
    "default launch without a grid": (
        _node("", [], ("@numthreads(1, 1, 1)",)),
        "dispatch-grid",
    ),
    # "Node shader 'b2' may not use both 'nodemaxdispatchgrid' and 'nodedispatchgrid'"
    "both grids": (
        _node(
            "DispatchNodeInputRecord<Record> obj",
            [],
            (
                "@node_dispatch_grid(1, 1, 1)",
                "@node_max_dispatch_grid(4, 1, 1)",
                "@numthreads(1, 1, 1)",
            ),
            members="uvec3 grid @ SV_DispatchGrid;",
        ),
        "dispatch-grid",
    ),
    # "... with NodeMaxDispatchGrid attribute must declare an input record containing a field with SV_DispatchGrid"
    "maximum grid without SV_DispatchGrid": (
        _node(
            "DispatchNodeInputRecord<Record> obj",
            [],
            ("@node_max_dispatch_grid(4, 1, 1)", "@numthreads(1, 1, 1)"),
        ),
        "dispatch-grid",
    ),
    "maximum grid without an input": (
        _node("", [], ("@node_max_dispatch_grid(4, 1, 1)", "@numthreads(1, 1, 1)")),
        "dispatch-grid",
    ),
    "SV_DispatchGrid of four components": (
        _node(
            "DispatchNodeInputRecord<Record> obj",
            [],
            ("@node_max_dispatch_grid(4, 1, 1)", "@numthreads(1, 1, 1)"),
            members="uvec4 grid @ SV_DispatchGrid;",
        ),
        "dispatch-grid",
    ),
    # "'nodedispatchgrid' may only be used with broadcasting nodes"
    "grid on a coalescing node": (
        _node(
            "",
            [],
            (
                "@node_launch(coalescing)",
                "@node_dispatch_grid(1, 1, 1)",
                "@numthreads(1, 1, 1)",
            ),
        ),
        "dispatch-grid",
    ),
    # "Thread launch nodes must have a thread group size of (1,1,1)"
    "thread launch of two threads": (
        _node(
            "ThreadNodeInputRecord<Record> obj",
            [],
            ("@node_launch(thread)", "@numthreads(2, 1, 1)"),
        ),
        "thread-group",
    ),
    # "Node shader 'c2' may not have more than one input record"
    "two inputs": (
        _node(
            "@max_records(2) GroupNodeInputRecords<Record> records, EmptyNodeInput empty",
            [],
            COALESCING,
        ),
        "input-count",
    ),
    # "'DispatchNodeInputRecord' may not be used with coalescing nodes (only GroupNodeInputRecords, ...)"
    "broadcasting input on a coalescing node": (
        _node("DispatchNodeInputRecord<Record> obj", [], COALESCING),
        "input-kind",
    ),
    "group input on a thread node": (
        _node("GroupNodeInputRecords<Record> obj", [], ("@node_launch(thread)",)),
        "input-kind",
    ),
    # "Invalid system value semantic 'SV_GroupID' for launchtype 'Coalescing'"
    "group id on a coalescing node": (
        _node("uvec3 group @ SV_GroupID", [], COALESCING),
        "system-value",
    ),
    "dispatch id on a coalescing node": (
        _node("uvec3 id @ gl_GlobalInvocationID", [], COALESCING),
        "system-value",
    ),
    # "Invalid system value semantic 'SV_GroupIndex' for launchtype 'Thread'"
    "group index on a thread node": (
        _node("uint index @ SV_GroupIndex", [], ("@node_launch(thread)",)),
        "system-value",
    ),
    # "'globallycoherent' is not a valid modifier for a declaration of type 'DispatchNodeInputRecord<Record>'"
    "globallycoherent read-only record": (
        _node(
            "@globallycoherent DispatchNodeInputRecord<Record> obj", [], BROADCASTING
        ),
        "globallycoherent",
    ),
    # "Use of FinishedCrossGroupSharing() requires NodeTrackRWInputSharing attribute to be specified on the record"
    "cross-group sharing without tracking": (
        _node(
            "@globallycoherent RWDispatchNodeInputRecord<Record> obj",
            ["out_buf[i] = obj.FinishedCrossGroupSharing() ? 1u : 0u;"],
            BROADCASTING,
        ),
        "cross-group-sharing",
    ),
    # "attribute 'NodeArraySize' may only be used with node output arrays (NodeOutputArray or EmptyNodeOutputArray)"
    "array size on a single output": (
        _node(
            "@max_records(1) @node_array_size(2) NodeOutput<Record> obj",
            [],
            BROADCASTING,
        ),
        "output-array-attribute",
    ),
}


@pytest.mark.parametrize("case", sorted(VIOLATIONS))
def test_rule_violations_are_refused(case):
    source, rule = VIOLATIONS[case]
    assert _rule(source) == rule


@pytest.mark.parametrize(
    "case",
    [
        # DXC accepts these.
        _node(
            "ThreadNodeInputRecord<Record> obj",
            ["out_buf[i] = obj.Get().value;"],
            ("@node_launch(thread)",),
        ),
        _node(
            "RWThreadNodeInputRecord<Record> obj",
            ["out_buf[i] = obj.Get().value;"],
            ("@node_launch(thread)", "@numthreads(1, 1, 1)"),
        ),
        _node(
            "GroupNodeInputRecords<Record> obj, uvec3 thread @ SV_GroupThreadID, uint index @ gl_LocalInvocationIndex",
            ["out_buf[index] = obj.Count() + thread.x;"],
            COALESCING,
        ),
        _node(
            "",
            ["out_buf[i] = GetRemainingRecursionLevels();"],
            ("@node_dispatch_grid(1, 1, 1)", "@numthreads(1, 1, 1)"),
        ),
        _node(
            "DispatchNodeInputRecord<Record> obj",
            ["out_buf[i] = obj.Get().value;"],
            BROADCASTING,
            members="uvec3 grid @ SV_DispatchGrid;\n        uint value;",
        ),
    ],
)
def test_nodes_dxc_accepts_pass(case):
    check_work_graph_nodes(parse(case), "node-test")


def test_registration_is_by_target_name(node_target):
    assert node_target in work_graph_nodes.work_graph_targets()
    assert "opengl" not in work_graph_nodes.work_graph_targets()
