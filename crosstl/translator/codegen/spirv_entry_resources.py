"""Resources that belong to one compute kernel of a SPIR-V module with several entry points.

CUDA and OpenCL C input hoist each kernel parameter to a program-scope declaration and mark it with the kernel it
came from (the ``entry_argument`` annotation of ``crosstl.backend.OpenCL.target_lowering``): a buffer or texture
keeps the binding the kernel gave it, and the plain values become a ``<kernel>_Args`` constant block bound after
them. Every kernel numbers its own parameters from binding 0, so the kernels of one file bind different buffers at
the same descriptor slot and may give their parameters the same names.

A SPIR-V module can hold all of them: the host builds one compute pipeline per entry point, and a pipeline's layout
only describes the resources its entry point statically uses (the Vulkan specification, "Pipeline Layouts" and
"Shader Resource Interface"). :func:`scope_spirv_entry_resources` therefore gives every declaration that belongs to
one entry of a program with several entries the ``spirv.entry_scope`` annotation, the set of entry names that take
it. The generator then

- checks a declaration's descriptor slot only against the declarations its entries can also see
  (:func:`spirv_entry_scopes_meet`), so ``a``'s ``o`` and ``b``'s ``p`` may both be set 0, binding 0;
- resolves the member names of an entry's constant blocks only inside that entry, so ``a_Args`` and ``b_Args``
  may both declare ``n``.

Declarations of several entries that agree in name, type and binding become one declaration that all of them take.
One that has the name of another entry's declaration but differs from it is renamed, together with the references
in its entry and that stage's functions; a program function that names it could mean either and is refused, as
:mod:`.metal_argument_tables` refuses it for Metal.
"""

from __future__ import annotations

from copy import deepcopy

from .. import ast as A
from ..refusals import VulkanRefusal
from ..semantics.scoped_walk import program_stages
from .metal_argument_tables import (
    ENTRY_ARGUMENT_ANNOTATION,
    _fresh_name,
    _program_function_names_reading,
    _rename_references,
    _shape,
)

__all__ = [
    "SPIRV_ENTRY_SCOPE_ANNOTATION",
    "scope_spirv_entry_resources",
    "spirv_entry_scope",
    "spirv_entry_scopes_meet",
]

SPIRV_ENTRY_SCOPE_ANNOTATION = "spirv.entry_scope"


def spirv_entry_scope(node):
    """The names of the entries that take ``node``, or None for a declaration every entry can take."""
    get = getattr(node, "get_annotation", None)
    return get(SPIRV_ENTRY_SCOPE_ANNOTATION) if callable(get) else None


def spirv_entry_scopes_meet(first, second):
    """Whether one entry can see a declaration of each of two scopes (None: every entry)."""
    return first is None or second is None or bool(first & second)


def _entries(ast):
    return [
        (stage, stage.entry_point)
        for stage in program_stages(ast)
        if isinstance(getattr(stage, "entry_point", None), A.FunctionNode)
        and isinstance(getattr(stage.entry_point, "name", None), str)
    ]


def _owned_declarations(ast, entries):
    """``(node, container, entry name)`` for each declaration that belongs to exactly one entry, in program order."""
    count = {}
    for _stage, entry in entries:
        count[entry.name] = count.get(entry.name, 0) + 1
    owned = []
    for field in ("global_variables", "cbuffers"):
        container = getattr(ast, field, None) or []
        for node in container:
            get = getattr(node, "get_annotation", None)
            kernel = get(ENTRY_ARGUMENT_ANNOTATION) if callable(get) else None
            # A kernel name two entries share names neither of them; the declaration stays program-scope.
            if isinstance(kernel, str) and count.get(kernel) == 1:
                owned.append((node, container, kernel))
    return owned


def _program_names(ast):
    names = set()
    for field in ("global_variables", "cbuffers", "functions", "structs", "constants"):
        for node in getattr(ast, field, None) or []:
            name = getattr(node, "name", None)
            if isinstance(name, str):
                names.add(name)
    for stage, entry in _entries(ast):
        names.add(entry.name)
        for node in getattr(stage, "local_functions", None) or []:
            name = getattr(node, "name", None)
            if isinstance(name, str):
                names.add(name)
    return names


def scope_spirv_entry_resources(ast):
    """Return ``ast``, or a copy whose entry-owned declarations carry the entries that take them."""
    entries = _entries(ast)
    if len(entries) < 2 or not _owned_declarations(ast, entries):
        return ast
    ast = deepcopy(ast)
    entries = _entries(ast)
    stages = {entry.name: stage for stage, entry in entries}
    used_names = _program_names(ast)
    kept_by_name = {}
    for node, container, kernel in _owned_declarations(ast, entries):
        source_name = node.name
        kept = kept_by_name.setdefault(source_name, [])
        shape = _shape(node)
        same = next((item for item in kept if item[1] == shape), None)
        if same is not None:
            # The entries agree on this declaration: it is one declaration, which each of them takes.
            declaration = same[0]
            if declaration.name != source_name:
                _rename_references(stages[kernel], source_name, declaration.name)
            declaration.add_annotation(
                SPIRV_ENTRY_SCOPE_ANNOTATION,
                spirv_entry_scope(declaration) | {kernel},
            )
            container[:] = [item for item in container if item is not node]
            continue
        if kept:
            if isinstance(node, A.StructNode):
                raise VulkanRefusal(
                    f"SPIR-V output cannot declare two different constant blocks named '{source_name}' for two "
                    "entries of one module"
                )
            readers = _program_function_names_reading(ast, source_name)
            if readers:
                raise VulkanRefusal(
                    f"SPIR-V output cannot tell which entry's '{source_name}' function '{readers[0]}' reads: "
                    "several entries declare a different resource of that name"
                )
            renamed = _fresh_name(source_name, used_names)
            _rename_references(stages[kernel], source_name, renamed)
            node.name = renamed
        node.add_annotation(SPIRV_ENTRY_SCOPE_ANNOTATION, frozenset({kernel}))
        kept.append((node, shape))
    return ast
