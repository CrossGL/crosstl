from enum import Enum

from crosstl.translator.codegen.stage_utils import (
    assign_stage_entry_names,
    collect_stage_entry_records,
    collect_stage_entry_reserved_function_names,
    collect_stage_local_variables,
    compute_local_size,
    deduplicate_named_declarations,
    normalize_stage_name,
    order_functions_by_dependencies,
    should_emit_qualified_function,
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


class DummyStageNode:
    def __init__(self, entry_point=None, local_functions=None, local_variables=None):
        self.entry_point = entry_point
        self.local_functions = local_functions or []
        self.local_variables = local_variables or []


class DummyAst:
    def __init__(self, functions=None, stages=None):
        self.functions = functions or []
        self.stages = stages or {}


def walk_dummy_nodes(root):
    if isinstance(root, list):
        for item in root:
            yield item
    elif root is not None:
        yield root


def dummy_call_name(call):
    return call.name


def test_normalize_stage_name_accepts_strings_and_enums():
    assert normalize_stage_name(None) is None
    assert normalize_stage_name("fragment") == "fragment"
    assert normalize_stage_name("ShaderStage.VERTEX") == "vertex"
    assert normalize_stage_name(DummyStage.FRAGMENT) == "fragment"


def test_stage_matches_normalizes_target_and_stage():
    assert stage_matches(None, "vertex")
    assert stage_matches("ShaderStage.VERTEX", DummyStage.VERTEX)
    assert not stage_matches("fragment", DummyStage.VERTEX)


def test_should_emit_qualified_function_keeps_helpers_and_filters_stages():
    assert should_emit_qualified_function("vertex", None)
    assert should_emit_qualified_function("vertex", "inline")
    assert should_emit_qualified_function("vertex", "ShaderStage.VERTEX")
    assert not should_emit_qualified_function("vertex", "fragment")
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


def test_collect_stage_local_variables_filters_by_stage_and_predicate():
    vertex_tex = DummyVariable("vertexTex", "sampler2D")
    fragment_tex = DummyVariable("fragmentTex", "sampler2D")
    fragment_scalar = DummyVariable("exposure", "float")
    ast = DummyAst(
        stages={
            "vertex": DummyStageNode(local_variables=[vertex_tex]),
            "fragment": DummyStageNode(
                local_variables=[fragment_tex, fragment_scalar]
            ),
        }
    )

    variables = collect_stage_local_variables(
        ast, "fragment", lambda node: node.var_type == "sampler2D"
    )

    assert variables == [fragment_tex]


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
