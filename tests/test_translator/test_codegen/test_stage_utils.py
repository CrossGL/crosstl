from enum import Enum

from crosstl.translator import stage_utils as shared_stage_utils
from crosstl.translator.ast import StageMap
from crosstl.translator.codegen.stage_utils import (
    assign_stage_entry_names,
    collect_stage_entry_records,
    collect_stage_entry_reserved_function_names,
    collect_stage_local_cbuffers,
    collect_stage_local_structs,
    collect_stage_local_variables,
    compute_local_size,
    deduplicate_named_declarations,
    normalize_stage_name,
    order_functions_by_dependencies,
    should_emit_qualified_function,
    stage_layout_entries,
    stage_layout_entry,
    stage_layout_entry_arguments,
    stage_layout_entry_value,
    stage_layout_qualifiers,
    stage_matches,
)


class DummyStage(Enum):
    VERTEX = "vertex"
    FRAGMENT = "fragment"


class DummyFunction:
    def __init__(self, name, body=None, qualifiers=None):
        self.name = name
        self.body = body or []
        self.parameters = []
        self.qualifiers = qualifiers or []


class DummyCall:
    def __init__(self, name):
        self.name = name


class DummyVariable:
    def __init__(self, name, var_type, attributes=None):
        self.name = name
        self.var_type = var_type
        self.attributes = attributes or []


class DummyLayout:
    def __init__(self, direction=None, entries=None):
        self.direction = direction
        self.entries = entries or []


class DummyLayoutEntry:
    def __init__(self, name, arguments=None):
        self.name = name
        self.arguments = arguments or []


class DummyLayoutArgument:
    def __init__(self, value=None, name=None):
        self.value = value
        self.name = name


class DummyBinaryLayoutArgument:
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right


class DummyUnaryLayoutArgument:
    def __init__(self, op, operand, is_postfix=False):
        self.op = op
        self.operand = operand
        self.is_postfix = is_postfix


class DummyStageNode:
    def __init__(
        self,
        entry_point=None,
        local_functions=None,
        local_variables=None,
        local_structs=None,
        local_cbuffers=None,
        layout_qualifiers=None,
    ):
        self.entry_point = entry_point
        self.local_functions = local_functions or []
        self.local_variables = local_variables or []
        self.local_structs = local_structs or []
        self.local_cbuffers = local_cbuffers or []
        self.layout_qualifiers = layout_qualifiers or []


class DummyAst:
    def __init__(self, functions=None, stages=None):
        self.functions = functions or []
        self.stages = stages or {}


def walk_dummy_nodes(root):
    if isinstance(root, list):
        yield from root
    elif root is not None:
        yield root


def dummy_call_name(call):
    return call.name


def test_normalize_stage_name_accepts_strings_and_enums():
    assert normalize_stage_name(None) is None
    assert normalize_stage_name("fragment") == "fragment"
    assert normalize_stage_name("ShaderStage.VERTEX") == "vertex"
    assert normalize_stage_name(DummyStage.FRAGMENT) == "fragment"


def test_codegen_stage_utils_uses_shared_stage_normalizer():
    assert normalize_stage_name is shared_stage_utils.normalize_stage_name
    assert shared_stage_utils.shader_stage_from_name("pixel-shader").value == "fragment"


def test_normalize_stage_name_canonicalizes_backend_aliases():
    assert normalize_stage_name("vs") == "vertex"
    assert normalize_stage_name("vertex-shader") == "vertex"
    assert normalize_stage_name("pixel") == "fragment"
    assert normalize_stage_name("ps") == "fragment"
    assert normalize_stage_name("frag") == "fragment"
    assert normalize_stage_name("hull") == "tessellation_control"
    assert normalize_stage_name("hs") == "tessellation_control"
    assert normalize_stage_name("domain") == "tessellation_evaluation"
    assert normalize_stage_name("ds") == "tessellation_evaluation"
    assert normalize_stage_name("cs") == "compute"
    assert normalize_stage_name("kernel") == "compute"
    assert normalize_stage_name("gs") == "geometry"
    assert normalize_stage_name("as") == "amplification"
    assert normalize_stage_name("ms") == "mesh"
    assert normalize_stage_name("raygeneration") == "ray_generation"
    assert normalize_stage_name("raygen") == "ray_generation"
    assert normalize_stage_name("rgen") == "ray_generation"
    assert normalize_stage_name("closesthit") == "ray_closest_hit"
    assert normalize_stage_name("closest-hit") == "ray_closest_hit"
    assert normalize_stage_name("rchit") == "ray_closest_hit"
    assert normalize_stage_name("anyhit") == "ray_any_hit"
    assert normalize_stage_name("any-hit") == "ray_any_hit"
    assert normalize_stage_name("rahit") == "ray_any_hit"
    assert normalize_stage_name("rint") == "ray_intersection"
    assert normalize_stage_name("rmiss") == "ray_miss"
    assert normalize_stage_name("rcall") == "ray_callable"
    assert normalize_stage_name("callable") == "ray_callable"
    assert normalize_stage_name("pixel-shader") == "fragment"


def test_stage_matches_normalizes_target_and_stage():
    assert stage_matches(None, "vertex")
    assert stage_matches("ShaderStage.VERTEX", DummyStage.VERTEX)
    assert stage_matches("hull", "ShaderStage.TESSELLATION_CONTROL")
    assert stage_matches("domain", "tessellation_evaluation")
    assert stage_matches("raygeneration", "ray_generation")
    assert stage_matches("closesthit", "ray_closest_hit")
    assert not stage_matches("fragment", DummyStage.VERTEX)
    assert not stage_matches("domain", "tessellation_control")


def test_should_emit_qualified_function_keeps_helpers_and_filters_stages():
    assert should_emit_qualified_function("vertex", None)
    assert should_emit_qualified_function("vertex", "inline")
    assert should_emit_qualified_function("vertex", "ShaderStage.VERTEX")
    assert should_emit_qualified_function("tessellation_control", "hull")
    assert should_emit_qualified_function("ray_closest_hit", "closesthit")
    assert not should_emit_qualified_function("vertex", "fragment")
    assert not should_emit_qualified_function("domain", "hull")
    assert not should_emit_qualified_function("vertex", "ray_miss")


def test_compute_local_size_uses_common_config_keys():
    assert compute_local_size() == ("1", "1", "1")
    assert compute_local_size({"local_size": (8, 4, 2)}) == ("8", "4", "2")
    assert compute_local_size({"workgroup_size": [16, 2, 1]}) == ("16", "2", "1")
    assert compute_local_size({"numthreads": (3, 5, 7)}) == ("3", "5", "7")
    assert compute_local_size({"local_size_x": 9, "local_size_y": 8}) == (
        "9",
        "8",
        "1",
    )


def test_stage_entry_name_helpers_reserve_global_and_local_helpers():
    helper = DummyFunction("PSMain")
    local_helper = DummyFunction("PSMain_2")
    entry = DummyFunction("main")
    ast = DummyAst(
        functions=[helper],
        stages={"fragment": DummyStageNode(entry, [local_helper])},
    )
    stage_entry_types = {"fragment"}

    entries = collect_stage_entry_records(ast, "fragment", stage_entry_types)
    reserved = collect_stage_entry_reserved_function_names(
        ast, "fragment", stage_entry_types
    )
    names = assign_stage_entry_names(
        entries,
        reserved,
        lambda _stage_name, _func: "PSMain",
    )

    assert entries == [(id(entry), "fragment", entry)]
    assert reserved == {"PSMain", "PSMain_2"}
    assert names[id(entry)] == "PSMain_3"


def test_stage_entry_name_helpers_canonicalize_alias_qualifiers():
    entry = DummyFunction("main", qualifiers=["hull"])
    ast = DummyAst(functions=[entry])

    entries = collect_stage_entry_records(
        ast, "tessellation_control", {"tessellation_control"}
    )

    assert entries == [(id(entry), "tessellation_control", entry)]


def test_stage_entry_name_helpers_can_keep_single_default_entry():
    entry = DummyFunction("main")
    ast = DummyAst(stages={"vertex": DummyStageNode(entry)})
    entries = collect_stage_entry_records(ast, None, {"vertex"})
    reserved = collect_stage_entry_reserved_function_names(ast, None, {"vertex"})

    names = assign_stage_entry_names(
        entries,
        reserved,
        lambda stage_name, _func: f"{stage_name}_main",
        single_entry_default="main",
    )

    assert names == {}


def test_stage_entry_name_helpers_rename_single_default_when_reserved():
    helper = DummyFunction("main")
    entry = DummyFunction("main")
    ast = DummyAst(functions=[helper], stages={"vertex": DummyStageNode(entry)})
    entries = collect_stage_entry_records(ast, None, {"vertex"})
    reserved = collect_stage_entry_reserved_function_names(ast, None, {"vertex"})

    names = assign_stage_entry_names(
        entries,
        reserved,
        lambda stage_name, _func: f"{stage_name}_main",
        single_entry_default="main",
    )

    assert names[id(entry)] == "vertex_main"


def test_stage_helpers_preserve_multiple_entries_for_same_stage():
    main_entry = DummyFunction("main")
    spawn_entry = DummyFunction("spawn")
    stages = StageMap()
    stages.append("compute", DummyStageNode(main_entry))
    stages.append("compute", DummyStageNode(spawn_entry))
    ast = DummyAst(stages=stages)

    entries = collect_stage_entry_records(ast, "compute", {"compute"})

    assert entries == [
        (id(main_entry), "compute", main_entry),
        (id(spawn_entry), "compute", spawn_entry),
    ]


def test_collect_stage_local_variables_filters_by_stage_and_predicate():
    vertex_tex = DummyVariable("vertexTex", "sampler2D")
    fragment_tex = DummyVariable("fragmentTex", "sampler2D")
    fragment_scalar = DummyVariable("exposure", "float")
    ast = DummyAst(
        stages={
            "vertex": DummyStageNode(local_variables=[vertex_tex]),
            "fragment": DummyStageNode(local_variables=[fragment_tex, fragment_scalar]),
        }
    )

    variables = collect_stage_local_variables(
        ast, "fragment", lambda node: node.var_type == "sampler2D"
    )

    assert variables == [fragment_tex]


def test_collect_stage_local_structs_filters_by_canonical_stage_aliases():
    hull_payload = DummyVariable("HullPatch", "struct")
    domain_payload = DummyVariable("DomainPatch", "struct")
    fragment_payload = DummyVariable("FragmentPayload", "struct")
    ast = DummyAst(
        stages={
            "hull": DummyStageNode(local_structs=[hull_payload]),
            "ShaderStage.TESSELLATION_EVALUATION": DummyStageNode(
                local_structs=[domain_payload]
            ),
            "fragment": DummyStageNode(local_structs=[fragment_payload]),
        }
    )

    assert collect_stage_local_structs(ast, "tessellation_control") == [hull_payload]
    assert collect_stage_local_structs(ast, "domain") == [domain_payload]
    assert collect_stage_local_structs(ast, None) == [
        hull_payload,
        domain_payload,
        fragment_payload,
    ]


def test_collect_stage_local_cbuffers_filters_by_stage():
    vertex_camera = DummyVariable("VertexCamera", "cbuffer")
    fragment_camera = DummyVariable("FragmentCamera", "cbuffer")
    ast = DummyAst(
        stages={
            "vertex": DummyStageNode(local_cbuffers=[vertex_camera]),
            "fragment": DummyStageNode(local_cbuffers=[fragment_camera]),
        }
    )

    assert collect_stage_local_cbuffers(ast, "fragment") == [fragment_camera]
    assert collect_stage_local_cbuffers(ast, None) == [vertex_camera, fragment_camera]


def test_stage_layout_helpers_filter_entries_and_values():
    geometry_stage = DummyStageNode(
        layout_qualifiers=[
            DummyLayout("in", [DummyLayoutEntry("triangles")]),
            DummyLayout(
                "out",
                [
                    DummyLayoutEntry("triangle_strip"),
                    DummyLayoutEntry("max_vertices", [DummyLayoutArgument(value=3)]),
                ],
            ),
        ]
    )

    assert stage_layout_qualifiers(geometry_stage, "in") == [
        geometry_stage.layout_qualifiers[0]
    ]
    assert stage_layout_entries(geometry_stage, "out") == [
        geometry_stage.layout_qualifiers[1].entries[0],
        geometry_stage.layout_qualifiers[1].entries[1],
    ]
    assert (
        stage_layout_entry(geometry_stage, "TRIANGLE_STRIP", "out")
        is geometry_stage.layout_qualifiers[1].entries[0]
    )
    assert stage_layout_entry_arguments(geometry_stage, "max_vertices", "out") == [
        geometry_stage.layout_qualifiers[1].entries[1].arguments[0]
    ]
    assert stage_layout_entry_value(geometry_stage, "max_vertices", "out") == "3"
    assert stage_layout_entry_value(geometry_stage, "invocations", "out", "1") == "1"


def test_stage_layout_entry_value_handles_identifier_arguments():
    stage = DummyStageNode(
        layout_qualifiers=[
            DummyLayout(
                "in",
                [
                    DummyLayoutEntry(
                        "local_size_x", [DummyLayoutArgument(name="GROUP_SIZE")]
                    )
                ],
            )
        ]
    )

    assert stage_layout_entry_value(stage, "local_size_x", "in") == "GROUP_SIZE"


def test_stage_layout_entry_value_formats_expression_arguments():
    stage = DummyStageNode(
        layout_qualifiers=[
            DummyLayout(
                "in",
                [
                    DummyLayoutEntry(
                        "local_size_x",
                        [
                            DummyBinaryLayoutArgument(
                                DummyLayoutArgument(name="GROUP_SIZE"),
                                "*",
                                DummyUnaryLayoutArgument(
                                    "-",
                                    DummyLayoutArgument(value=2),
                                ),
                            )
                        ],
                    )
                ],
            ),
            DummyLayout(
                "out",
                [
                    DummyLayoutEntry(
                        "stream",
                        [
                            DummyUnaryLayoutArgument(
                                "++",
                                DummyLayoutArgument(name="streamIndex"),
                                is_postfix=True,
                            )
                        ],
                    )
                ],
            ),
        ]
    )

    assert stage_layout_entry_value(stage, "local_size_x", "in") == "GROUP_SIZE * -2"
    assert stage_layout_entry_value(stage, "stream", "out") == "streamIndex++"


def test_deduplicate_named_declarations_reuses_matching_stage_resources():
    first = DummyVariable("sharedTex", "sampler2D")
    second = DummyVariable("sharedTex", "sampler2D")

    assert deduplicate_named_declarations([first, second], "resource") == [first]


def test_deduplicate_named_declarations_rejects_conflicting_stage_resources():
    first = DummyVariable("sharedTex", "sampler2D")
    second = DummyVariable("sharedTex", "image2D")

    try:
        deduplicate_named_declarations([first, second], "resource")
    except ValueError as exc:
        assert str(exc) == "Conflicting resource declaration for 'sharedTex'"
    else:
        raise AssertionError("expected conflicting resource declaration")


def test_order_functions_by_dependencies_places_callees_first():
    first = DummyFunction("first", [DummyCall("second")])
    second = DummyFunction("second")

    ordered = order_functions_by_dependencies(
        [first, second],
        walk_dummy_nodes,
        dummy_call_name,
        DummyCall,
    )

    assert ordered == [second, first]
