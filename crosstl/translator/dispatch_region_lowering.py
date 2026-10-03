"""Specialize a compute AST for one exact-grid dispatch region."""

from __future__ import annotations

import copy

from .ast import (
    AssignmentNode,
    ASTNode,
    AttributeNode,
    BinaryOpNode,
    BlockNode,
    BuiltinVariableNode,
    ConstantNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MemberAccessNode,
    ParameterNode,
    PrimitiveType,
    TypeNode,
    VariableNode,
    VectorType,
)
from .codegen.stage_utils import collect_stage_entry_records
from .dispatch_regions import DispatchRegion
from .stage_utils import STAGE_QUALIFIER_NAMES, normalize_stage_name

_SEMANTICS = {
    "gl_GlobalInvocationID": "thread",
    "SV_DispatchThreadID": "thread",
    "SV_DispatchThreadId": "thread",
    "sv_dispatch_thread_id": "thread",
    "sv_dispatchthreadid": "thread",
    "gl_WorkGroupID": "group",
    "SV_GroupID": "group",
    "SV_GroupId": "group",
    "sv_group_id": "group",
    "sv_groupid": "group",
    "gl_NumWorkGroups": "count",
    "threads_per_grid": "grid",
}
_SKIP_FIELDS = {"parent", "annotations", "source_location"}


def _walk(value):
    seen = set()
    pending = [value]
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, TypeNode):
            continue
        if isinstance(node, ASTNode):
            yield node
            pending.extend(v for k, v in vars(node).items() if k not in _SKIP_FIELDS)
        elif isinstance(node, dict):
            pending.extend(node.values())
        elif isinstance(node, (list, tuple)):
            pending.extend(node)


def _coordinate_type(value):
    scalar = value.element_type if isinstance(value, VectorType) else value
    width = value.size if isinstance(value, VectorType) else 1
    if (
        not isinstance(scalar, PrimitiveType)
        or scalar.name not in {"uint", "int"}
        or width not in (1, 2, 3)
    ):
        raise ValueError(
            "Dispatch coordinates require int/uint scalars or vectors of width 1..3"
        )
    return scalar.name, width


def _constant(values, value_type):
    scalar, width = _coordinate_type(value_type)
    if scalar == "int" and any(n > 0x7FFFFFFF for n in values[:width]):
        raise ValueError("Dispatch coordinate exceeds the signed parameter range")
    args = [LiteralNode(n, PrimitiveType(scalar)) for n in values[:width]]
    if width == 1:
        return args[0]
    name = ("uvec" if scalar == "uint" else "ivec") + str(width)
    return FunctionCallNode(IdentifierNode(name), args)


def _expression(kind, physical, value_type, region):
    if kind == "grid":
        return _constant(region.thread_grid_size, value_type)
    if kind == "count":
        return _constant(region.source_workgroup_count, value_type)
    offset = region.thread_offset if kind == "thread" else region.workgroup_offset
    return BinaryOpNode(physical, "+", _constant(offset, value_type))


def _rewrite_builtins(value, bound, region, memo, capture):
    """Retain lexical declarations and shared legacy AST aliases during rewriting."""
    if not isinstance(value, (ASTNode, list, tuple, dict)) or isinstance(
        value, TypeNode
    ):
        return value
    key = (id(value), frozenset(bound))
    if key in memo:
        return memo[key]
    if isinstance(value, IdentifierNode) and value.name not in bound:
        kind = _SEMANTICS.get(value.name)
        if kind:
            physical = capture(kind) if kind in {"thread", "group"} else None
            result = _expression(
                kind, physical, VectorType(PrimitiveType("uint"), 3), region
            )
            memo[key] = result
            return result
    if isinstance(value, BuiltinVariableNode):
        kind = _SEMANTICS.get(value.builtin_name)
        if kind:
            physical = capture(kind) if kind in {"thread", "group"} else None
            result = _expression(
                kind, physical, VectorType(PrimitiveType("uint"), 3), region
            )
            if value.component:
                result = MemberAccessNode(result, value.component)
            memo[key] = result
            return result
    if isinstance(value, (list, tuple)):
        items = [_rewrite_builtins(v, bound, region, memo, capture) for v in value]
        result = tuple(items) if isinstance(value, tuple) else items
    elif isinstance(value, dict):
        result = {
            k: _rewrite_builtins(v, bound, region, memo, capture)
            for k, v in value.items()
        }
    else:
        result = copy.copy(value)
        memo[key] = result
        if isinstance(value, BlockNode):
            scope = set(bound)
            result.statements = []
            for statement in value.statements:
                if isinstance(statement, (VariableNode, ConstantNode)):
                    scope.add(statement.name)
                result.statements.append(
                    _rewrite_builtins(statement, scope, region, memo, capture)
                )
        else:
            scope = set(bound)
            if isinstance(value, ForNode) and isinstance(
                value.init, (VariableNode, ConstantNode)
            ):
                scope.add(value.init.name)
            for name, child in vars(value).items():
                if name in _SKIP_FIELDS:
                    continue
                child_scope = scope
                if isinstance(value, ForInNode) and name == "body":
                    child_scope = scope | {value.pattern}
                setattr(
                    result,
                    name,
                    _rewrite_builtins(child, child_scope, region, memo, capture),
                )
    memo[key] = result
    return result


def specialize_dispatch_region(ast, region: DispatchRegion):
    """Return an independent AST for a single region; never mutate the source AST.

    The caller must select one compute entry, dispatch exactly the region's
    physical counts/size, and share resources across every region in the plan.
    This pass does not submit work or silently round a native dispatch.
    """
    if not isinstance(region, DispatchRegion):
        raise TypeError("region must be a DispatchRegion")
    if ast.annotations.get("dispatch_region") is not None:
        raise ValueError("Compute AST is already specialized for a dispatch region")
    entries = collect_stage_entry_records(ast, None, STAGE_QUALIFIER_NAMES)
    if len(entries) != 1 or entries[0][1] != "compute":
        raise ValueError(
            "Dispatch region specialization requires exactly one compute entry"
        )
    result = copy.deepcopy(ast)
    entries = collect_stage_entry_records(result, None, STAGE_QUALIFIER_NAMES)
    entry = entries[0][2]
    if not isinstance(entry.body, BlockNode):
        raise ValueError("Dispatch region specialization requires a compute entry body")
    names = {getattr(node, "name", None) for node in _walk(result)}
    names.discard(None)
    captures = {}

    def unique(name):
        while name in names:
            name += "_"
        names.add(name)
        return name

    def capture(kind):
        if kind not in captures:
            captures[kind] = (
                unique(f"crossglPhysical_{kind}"),
                unique(f"crossglDispatch_{kind}"),
            )
        return IdentifierNode(captures[kind][0])

    globals_ = {node.name for node in result.global_variables + result.constants}
    for block in result.cbuffers:
        globals_.update(member.name for member in block.members)
    functions = list(result.functions)
    declarations = list(result.global_variables + result.constants)
    for stage in result.stages.values():
        globals_.update(variable.name for variable in stage.local_variables)
        declarations.extend(stage.local_variables)
        for block in stage.local_cbuffers:
            globals_.update(member.name for member in block.members)
        functions.extend(stage.local_functions)
        if stage.entry_point is not None:
            functions.append(stage.entry_point)
    globals_.update(function.name for function in functions)
    for declaration in declarations:
        initializer = getattr(
            declaration, "initial_value", getattr(declaration, "value", None)
        )
        for node in _walk(initializer):
            if (
                isinstance(node, IdentifierNode)
                and node.name in _SEMANTICS
                and node.name not in globals_
            ) or (
                isinstance(node, BuiltinVariableNode)
                and node.builtin_name in _SEMANTICS
            ):
                raise ValueError(
                    "Dispatch coordinates in global initializers require entry-local initialization"
                )
    visited = set()
    for function in functions:
        if id(function) in visited:
            continue
        visited.add(id(function))
        bound = globals_ | {p.name for p in function.parameters}
        function.body = _rewrite_builtins(function.body, bound, region, {}, capture)
    prologue = []
    parameters = []
    for parameter in entry.parameters:
        kinds = {
            _SEMANTICS[a.name] for a in parameter.attributes if a.name in _SEMANTICS
        }
        if not kinds:
            parameters.append(parameter)
            continue
        if len(kinds) != 1:
            raise ValueError("Entry parameter has conflicting dispatch semantics")
        kind = kinds.pop()
        scalar, width = _coordinate_type(parameter.param_type)
        physical = None
        if kind in {"thread", "group"}:
            extent = (
                region.thread_grid_size
                if kind == "thread"
                else region.source_workgroup_count
            )
            if scalar == "int" and any(n - 1 > 0x7FFFFFFF for n in extent[:width]):
                raise ValueError(
                    "Dispatch coordinate exceeds the signed parameter range"
                )
            physical = capture(kind)
            if width < 3:
                physical = MemberAccessNode(physical, "xyz"[:width])
            if scalar == "int":
                constructor = "int" if width == 1 else f"ivec{width}"
                physical = FunctionCallNode(IdentifierNode(constructor), [physical])
        prologue.append(
            VariableNode(
                parameter.name,
                copy.deepcopy(parameter.param_type),
                _expression(kind, physical, parameter.param_type, region),
            )
        )
    capture_prologue = []
    for kind, (global_name, parameter_name) in captures.items():
        value_type = VectorType(PrimitiveType("uint"), 3)
        semantic = "gl_GlobalInvocationID" if kind == "thread" else "gl_WorkGroupID"
        result.global_variables.append(
            VariableNode(global_name, value_type, qualifiers=["thread"])
        )
        parameters.append(
            ParameterNode(
                parameter_name, value_type, attributes=[AttributeNode(semantic)]
            )
        )
        capture_prologue.append(
            AssignmentNode(IdentifierNode(global_name), IdentifierNode(parameter_name))
        )
    entry.parameters = parameters
    entry.body.statements = capture_prologue + prologue + entry.body.statements
    entry.attributes = [
        a for a in entry.attributes if a.name not in {"numthreads", "local_size"}
    ]
    entry.attributes.append(
        AttributeNode(
            "numthreads",
            [LiteralNode(n, PrimitiveType("int")) for n in region.workgroup_size],
        )
    )
    for kind, stage in result.stages.items():
        if normalize_stage_name(kind) == "compute":
            x, y, z = map(str, region.workgroup_size)
            stage.execution_config.update(
                {
                    "numthreads": (x, y, z),
                    "local_size": (x, y, z),
                    "local_size_x": x,
                    "local_size_y": y,
                    "local_size_z": z,
                }
            )
    result.annotations["dispatch_region"] = region.to_json()
    return result
