"""Declare the Metal standard library structures that a unit uses without declaring them.

``<metal_stdlib>`` declares plain-data structures such as ``MTLQuadTessellationFactorsHalf`` that shaders name
without a declaration (:mod:`crosstl.translator.codegen.metal_library_structs`). CrossGL has no such library,
so the importer declares each one the unit names as an ordinary structure with the library's members.
"""

import re

from ...translator.codegen.metal_library_structs import METAL_LIBRARY_STRUCTS

_NAME_PATTERN = re.compile(
    r"\b(?:metal::)?(" + "|".join(sorted(METAL_LIBRARY_STRUCTS)) + r")\b"
)


def metal_library_struct_source(name):
    """Return the Metal declaration of library structure ``name``."""
    members = "".join(
        f" {element_type} {member_name}{'' if length is None else f'[{length}]'};"
        for element_type, member_name, length in METAL_LIBRARY_STRUCTS[name]
    )
    return f"struct {name} {{{members} }};"


def _strings(node, seen):
    """Yield every string held anywhere in AST ``node``."""
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, str):
            yield current
        elif isinstance(current, (list, tuple, set, frozenset)):
            stack.extend(current)
        elif isinstance(current, dict):
            stack.extend(current.keys())
            stack.extend(current.values())
        else:
            attributes = getattr(current, "__dict__", None)
            if attributes is not None and id(current) not in seen:
                seen.add(id(current))
                stack.extend(attributes.values())


def referenced_library_struct_names(ast):
    """Return the library structures ``ast`` names but does not declare, in table order."""
    declared = {
        getattr(node, "name", None)
        for attribute in ("structs", "typedefs")
        for node in getattr(ast, attribute, None) or []
    }
    referenced = set()
    for text in _strings(ast, set()):
        if "MTL" in text:
            referenced.update(_NAME_PATTERN.findall(text))
    return [
        name
        for name in METAL_LIBRARY_STRUCTS
        if name in referenced and name not in declared
    ]


def declare_referenced_library_structs(ast, parse_structs):
    """Prepend to ``ast.structs`` the library structures it uses without declaring.

    ``parse_structs`` parses Metal source text and returns its structure declarations.
    """
    names = referenced_library_struct_names(ast)
    if names:
        source = "\n".join(metal_library_struct_source(name) for name in names)
        ast.structs = list(parse_structs(source)) + list(
            getattr(ast, "structs", None) or []
        )
    return ast
