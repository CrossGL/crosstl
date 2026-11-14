"""Vertex inputs bound to a CrossGL built-in value that Metal has no vertex-function attribute for.

Metal's vertex functions read the draw's system values through ``[[vertex_id]]``, ``[[instance_id]]``,
``[[base_vertex]]``, ``[[base_instance]]``, ``[[amplification_id]]`` and ``[[amplification_count]]`` (Metal Shading
Language Specification, section 5.2.3.1, "Vertex Function Input Attributes"); every other ``[[stage_in]]`` member is a
vertex-buffer attribute. The emitter gave a vertex input whose ``gl_`` built-in has no such attribute the next free
``[[attribute(n)]]``, so ``uint draw @ gl_DrawID`` (or an HLSL or Slang ``SV_DrawIndex`` parameter) compiled and read
vertex attribute ``n`` instead of the draw index. Metal has no draw-index attribute, and the fragment-input path
already refuses a built-in it cannot express (:mod:`.metal_varyings`).

:func:`refuse_unbound_metal_vertex_inputs` refuses each such input of a vertex entry point: a parameter, or a member
of the structure a parameter takes by value. It records the refusal through the generator's ``refuse_lowering`` and
does not modify the AST.
"""

from __future__ import annotations

from ..ast import NamedType
from ..semantics.builtin_names import is_crossgl_gl_builtin
from .stage_utils import collect_stage_entry_records

__all__ = ["refuse_unbound_metal_vertex_inputs"]

_OUTPUT_QUALIFIERS = frozenset({"out", "inout"})


def _structs(ast):
    """Every structure declared at program or stage scope, by name; the first declaration of a name wins."""
    structs = {}
    containers = [getattr(ast, "structs", None)]
    for stage in (getattr(ast, "stages", None) or {}).values():
        containers.append(getattr(stage, "local_structs", None))
    for container in containers:
        for struct in container or []:
            name = getattr(struct, "name", None)
            if isinstance(name, str) and name:
                structs.setdefault(name, struct)
    return structs


def _is_input(parameter):
    qualifiers = {str(q).lower() for q in getattr(parameter, "qualifiers", None) or []}
    return not qualifiers & _OUTPUT_QUALIFIERS


def _input_nodes(parameter, structs):
    """``(name, node)`` for the parameter, or for each member of the structure it takes by value."""
    param_type = getattr(parameter, "param_type", None)
    if isinstance(param_type, NamedType) and not param_type.generic_args:
        struct = structs.get(param_type.name)
        if struct is not None:
            for member in getattr(struct, "members", None) or []:
                yield f"{struct.name}.{getattr(member, 'name', '?')}", member
            return
    yield getattr(parameter, "name", "?"), parameter


def refuse_unbound_metal_vertex_inputs(ast, codegen):
    """Refuse every vertex-entry input whose ``gl_`` built-in Metal would bind as a vertex-buffer attribute."""
    structs = _structs(ast)
    refused = set()
    for _id, _stage, function in collect_stage_entry_records(ast, None, {"vertex"}):
        for parameter in getattr(function, "parameters", None) or []:
            if not _is_input(parameter):
                continue
            for name, node in _input_nodes(parameter, structs):
                semantic = codegen.semantic_from_node(node)
                if semantic is None or not is_crossgl_gl_builtin(semantic):
                    continue
                # A built-in Metal spells with one of its own attributes keeps it; one whose spelling is still
                # the ``gl_`` name would take the next free ``[[attribute(n)]]``.
                canonical = str(codegen.canonical_metal_semantic(semantic))
                if not canonical.lower().startswith("gl_"):
                    continue
                if not codegen.is_metal_user_stage_io_semantic(semantic):
                    continue
                key = (name, str(semantic))
                if key in refused:
                    continue
                refused.add(key)
                codegen.refuse_lowering(
                    f"vertex input '{name}': Metal has no vertex-function attribute for the built-in "
                    f"@{semantic}; it would read a vertex-buffer attribute instead"
                )
