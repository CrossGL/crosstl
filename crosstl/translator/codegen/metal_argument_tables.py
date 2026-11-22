"""Per-entry argument tables for the resources that belong to one stage entry, for Metal.

A Metal entry function takes its resources as parameters, and ``[[buffer(n)]]``, ``[[texture(n)]]`` and
``[[sampler(n)]]`` index that function's own argument table (Metal Shading Language Specification, section 5.2.1),
so two kernels of one file may bind different buffers at the same index. CrossGL has two kinds of declarations that
belong to one entry:

- a resource declared in a stage block, which only that stage's functions name;
- a resource or ``<kernel>_Args`` constant block that OpenCL C and CUDA input write for a kernel parameter. The
  importer hoists them to program scope and marks each with the kernel it came from (the ``entry_argument``
  annotation of ``crosstl.backend.OpenCL.target_lowering``).

:func:`assign_metal_argument_tables` gives every stage entry of a program with several entries an argument table
number and marks each such declaration with the tables of the entries it belongs to. The generator then passes the
declaration to those entries only, checks its binding against the program-scope resources and the declarations of
the same tables only, and reads a constant-block member name in an entry through the blocks that entry takes, so
``fill_Args`` and ``scale_Args`` may both declare ``n``. Declarations of several entries that agree in name, type
and binding are one declaration that all of them take. One that has the name of another entry's declaration but
differs from it is renamed, together with the references in its own entry and stage functions; a program function
that names it could mean either and is refused.
"""

from __future__ import annotations

from copy import deepcopy

from .. import ast as A
from ..refusals import MetalRefusal
from ..semantics.rewrite import child_slots, get_child
from ..semantics.scoped_walk import program_stages

__all__ = [
    "ENTRY_ARGUMENT_ANNOTATION",
    "assign_metal_argument_tables",
    "metal_argument_tables",
    "metal_entry_argument_table",
    "metal_argument_tables_meet",
]

# Written by the OpenCL C and CUDA bridge (crosstl/backend/OpenCL/target_lowering.py): the kernel whose parameter a
# program-scope declaration was.
ENTRY_ARGUMENT_ANNOTATION = "entry_argument"
_ENTRY_TABLE = "metal.argument_table"
_DECLARATION_TABLES = "metal.argument_tables"

# Statements that open a scope; a local declared in one hides a program-scope name to the end of that scope.
_SCOPE_NODES = (
    A.BlockNode,
    A.ForNode,
    A.IfNode,
    A.WhileNode,
    A.DoWhileNode,
    A.SwitchNode,
)
_SHAPE_SKIPPED = frozenset({"parent", "annotations", "source_location"})


def metal_argument_tables(node):
    """The argument tables a declaration belongs to, or None for a declaration every entry takes."""
    get = getattr(node, "get_annotation", None)
    return get(_DECLARATION_TABLES) if callable(get) else None


def metal_entry_argument_table(function):
    """The argument table number of a stage entry, or None for a helper or a program with one entry."""
    get = getattr(function, "get_annotation", None)
    return get(_ENTRY_TABLE) if callable(get) else None


def metal_argument_tables_meet(first, second):
    """Whether one entry can take a declaration of each of two table sets (None: every entry)."""
    return first is None or second is None or bool(first & second)


def _entries(ast):
    return [
        (index, stage, stage.entry_point)
        for index, stage in enumerate(program_stages(ast))
        if isinstance(getattr(stage, "entry_point", None), A.FunctionNode)
    ]


def _owned_declarations(ast, entries, is_stage_resource):
    """``(node, container, tables, stage)`` for each declaration that belongs to some entries, in program order."""
    entries_by_name = {}
    for index, _stage, entry in entries:
        entries_by_name.setdefault(entry.name, []).append(index)
    owned = []
    for field in ("global_variables", "cbuffers"):
        container = getattr(ast, field, None) or []
        for node in container:
            get = getattr(node, "get_annotation", None)
            kernel = get(ENTRY_ARGUMENT_ANNOTATION) if callable(get) else None
            indices = entries_by_name.get(kernel, [])
            # A kernel name two entries share names neither of them; the declaration stays program-scope.
            if len(indices) == 1:
                owned.append((node, container, frozenset(indices), None))
    for index, stage, _entry in entries:
        container = getattr(stage, "local_variables", None) or []
        for node in container:
            if isinstance(getattr(node, "name", None), str) and is_stage_resource(node):
                owned.append((node, container, frozenset({index}), stage))
    return owned


def _shape(node, path=frozenset(), skipped=_SHAPE_SKIPPED | {"name"}):
    """A structural key of a declaration, apart from its name: kind, type, qualifiers, attributes and arguments."""
    if isinstance(node, A.ASTNode):
        if id(node) in path:
            return ("cycle", type(node).__name__)
        path = path | {id(node)}
        return (
            type(node).__name__,
            tuple(
                (field, _shape(value, path, _SHAPE_SKIPPED))
                for field, value in sorted(vars(node).items())
                if field not in skipped
            ),
        )
    if isinstance(node, (list, tuple)):
        return tuple(_shape(item, path, _SHAPE_SKIPPED) for item in node)
    if isinstance(node, dict):
        return tuple(
            sorted(
                (str(key), _shape(value, path, _SHAPE_SKIPPED))
                for key, value in node.items()
            )
        )
    if node is None or isinstance(node, (str, int, float, bool)):
        return node
    return (type(node).__name__, str(getattr(node, "value", node)))


class _ReferenceRenamer:
    """Renames the references to a program-scope name that no local declaration hides, in one function."""

    def __init__(self, old, new):
        self.old = old
        self.new = new
        self.count = 0

    def run(self, function):
        parameters = getattr(function, "parameters", None) or []
        if any(
            getattr(parameter, "name", None) == self.old for parameter in parameters
        ):
            return
        scopes = [set()]
        body = getattr(function, "body", None)
        for node in body if isinstance(body, list) else [body]:
            self.walk(node, scopes)

    def walk(self, node, scopes):
        if node is None or isinstance(node, (A.FunctionNode, A.StructNode)):
            return
        scoped = isinstance(node, _SCOPE_NODES)
        if scoped:
            scopes.append(set())
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            if isinstance(child, A.IdentifierNode):
                # A callee names a function or type, never a variable.
                if not (isinstance(node, A.FunctionCallNode) and field == "function"):
                    self.reference(child, scopes)
                continue
            self.walk(child, scopes)
        # A local's own initializer still reads the outer name.
        if isinstance(node, A.VariableNode) and node.name == self.old:
            scopes[-1].add(self.old)
        if scoped:
            scopes.pop()

    def reference(self, node, scopes):
        if node.name != self.old or any(self.old in scope for scope in scopes):
            return
        node.name = self.new
        node.identifier = self.new
        self.count += 1


def _program_function_names_reading(ast, name):
    """Program functions (not stage entries or stage functions) that reference ``name`` with no local declaring it."""
    readers = []
    for function in getattr(ast, "functions", None) or []:
        probe = _ReferenceRenamer(name, name)
        probe.run(function)
        if probe.count:
            readers.append(function.name)
    return readers


def _fresh_name(name, used):
    index = 1
    candidate = f"{name}_{index}"
    while candidate in used:
        index += 1
        candidate = f"{name}_{index}"
    used.add(candidate)
    return candidate


def _rename_references(stage, old, new):
    renamer = _ReferenceRenamer(old, new)
    renamer.run(stage.entry_point)
    for function in getattr(stage, "local_functions", None) or []:
        renamer.run(function)


def _callee_name(call):
    function = getattr(call, "function", None)
    return function if isinstance(function, str) else getattr(function, "name", None)


def _rename_calls(stage, old, new):
    functions = [stage.entry_point, *(getattr(stage, "local_functions", None) or [])]
    for function in functions:
        for node in function.walk() if hasattr(function, "walk") else []:
            if not isinstance(node, A.FunctionCallNode) or _callee_name(node) != old:
                continue
            if isinstance(node.function, str):
                node.function = new
            else:
                node.function.name = new
                if hasattr(node.function, "identifier"):
                    node.function.identifier = new


def _colliding_stage_function_names(ast, entries):
    """Names that the stage functions of two entries both declare and no program function does."""
    program_names = {
        getattr(function, "name", None)
        for function in getattr(ast, "functions", None) or []
    }
    seen, colliding = set(), set()
    for _index, stage, _entry in entries:
        names = {
            function.name
            for function in getattr(stage, "local_functions", None) or []
            if isinstance(getattr(function, "name", None), str)
        }
        colliding |= (names & seen) - program_names
        seen |= names
    return colliding


def _rename_colliding_stage_functions(entries, colliding, used_names):
    """Give each later stage's functions of a name another stage's functions have a name of their own.

    Metal declares every stage's functions at program scope, so two stages' ``helper`` were two definitions of one
    function, which the Metal compiler rejects.
    """
    claimed = set()
    for _index, stage, _entry in entries:
        functions = getattr(stage, "local_functions", None) or []
        names = {function.name for function in functions} & colliding
        for name in sorted(names):
            if name not in claimed:
                claimed.add(name)
                continue
            renamed = _fresh_name(name, used_names)
            _rename_calls(stage, name, renamed)
            for function in functions:
                if function.name == name:
                    function.name = renamed


def assign_metal_argument_tables(ast, codegen):
    """Return ``ast``, or a copy whose entries and entry-owned declarations carry their argument tables."""
    is_stage_resource = codegen.is_stage_local_resource_variable
    entries = _entries(ast)
    if len(entries) < 2:
        return ast
    owned = _owned_declarations(ast, entries, is_stage_resource)
    colliding = _colliding_stage_function_names(ast, entries)
    if not owned and not colliding:
        return ast
    ast = deepcopy(ast)
    entries = _entries(ast)
    used_names = codegen.collect_metal_program_identifier_names(ast)
    _rename_colliding_stage_functions(entries, colliding, used_names)
    if owned:
        _assign_tables(ast, entries, is_stage_resource, used_names)
    return ast


def _assign_tables(ast, entries, is_stage_resource, used_names):
    stages = {index: stage for index, stage, _entry in entries}
    for index, _stage, entry in entries:
        entry.add_annotation(_ENTRY_TABLE, index)
    kept_by_name = {}
    for node, container, tables, _stage in _owned_declarations(
        ast, entries, is_stage_resource
    ):
        source_name = node.name
        kept = kept_by_name.setdefault(source_name, [])
        shape = _shape(node)
        same = next(
            (declaration for declaration in kept if declaration[1] == shape), None
        )
        if same is not None:
            # The entries agree on this declaration: it is one declaration, which each of them takes.
            declaration = same[0]
            if declaration.name != source_name and not isinstance(node, A.StructNode):
                for index in tables:
                    _rename_references(stages[index], source_name, declaration.name)
            declaration.add_annotation(
                _DECLARATION_TABLES, metal_argument_tables(declaration) | tables
            )
            container[:] = [item for item in container if item is not node]
            continue
        if kept:
            if isinstance(node, A.StructNode):
                raise MetalRefusal(
                    f"Metal output cannot declare two different constant blocks named '{source_name}' for "
                    "two entries of one program"
                )
            readers = _program_function_names_reading(ast, source_name)
            if readers:
                raise MetalRefusal(
                    f"Metal output cannot tell which entry's '{source_name}' function '{readers[0]}' reads: "
                    "several entries declare a different resource of that name"
                )
            renamed = _fresh_name(source_name, used_names)
            for index in tables:
                _rename_references(stages[index], source_name, renamed)
            node.name = renamed
        node.add_annotation(_DECLARATION_TABLES, tables)
        kept.append((node, shape))
