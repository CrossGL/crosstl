"""Fragment outputs with a ``COLORn`` semantic write colour attachment ``n``.

Direct3D 9 pixel shaders name their render-target outputs ``COLOR0`` to ``COLOR3``, and Direct3D 10 and later read a
``COLORn`` pixel-shader output as ``SV_Targetn`` (HLSL "Semantics", Direct3D 9 VS/PS semantics and the
"Semantics" mapping table for Direct3D 10). A ``COLOR`` semantic without an index has index 0. The HLSL importer
keeps such an output as the CrossGL user semantic ``@ Color0``, which every generator then read as a vertex colour
varying: GLSL declared it at varying location 13 instead of attachment 0, Metal gave the fragment output a vertex
``[[attribute(13)]]`` (which does not compile), and a sibling ``SV_Target2`` member was declared at location 0 by GLSL.

:func:`name_fragment_colour_outputs` rewrites ``@ COLORn`` (any case, ``n`` defaulting to 0) on a fragment entry
point's return value, on its ``out`` parameters and on the members of the structures they carry to
``@ gl_FragData[n]``, which every generator emits as colour attachment ``n``. A structure that a fragment entry point
writes this way and that another entry point also uses as an interface would need both meanings at once; it is
refused with :class:`FragmentColourOutputUnsupportedError`.
"""

from __future__ import annotations

import re

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import iter_structs

__all__ = ["FragmentColourOutputUnsupportedError", "name_fragment_colour_outputs"]

_COLOUR = re.compile(r"color(\d*)", re.IGNORECASE)


class FragmentColourOutputUnsupportedError(TranslationDiagnostic):
    """A structure with ``COLORn`` members written by a fragment entry point and used by another interface."""

    def __init__(self, message, target):
        super().__init__(
            message,
            target=target,
            code=f"{CODE_PREFIX}fragment-colour-output-unsupported",
        )


def name_fragment_colour_outputs(program, target=None):
    """Rewrite the ``COLORn`` fragment outputs of ``program`` to ``gl_FragData[n]``, in place."""
    structs = {getattr(struct, "name", None): struct for struct in iter_structs(program)}
    output_structs = {}
    other_uses = {}
    for stage_key, stage in _stages(program):
        function = getattr(stage, "entry_point", None)
        if function is None:
            continue
        fragment = _stage_name(stage_key, stage) == "fragment"
        name = getattr(function, "name", "?")
        values = [(function, getattr(function, "return_type", None), True)]
        values.extend(
            (parameter, getattr(parameter, "param_type", None), _is_output(parameter))
            for parameter in getattr(function, "parameters", None) or []
        )
        for node, type_node, output in values:
            struct_name = _type_name(type_node)
            if fragment and output:
                if node is not function or not _is_void(type_node):
                    _rename_colour(getattr(node, "attributes", None))
                if struct_name in structs:
                    output_structs.setdefault(struct_name, name)
            elif struct_name in structs:
                other_uses.setdefault(struct_name, name)

    for struct_name in sorted(output_structs):
        members = getattr(structs[struct_name], "members", None) or []
        if not any(_colour_index(member) is not None for member in members):
            continue
        if struct_name in other_uses:
            raise FragmentColourOutputUnsupportedError(
                f"Structure '{struct_name}' carries COLOR fragment outputs of "
                f"'{output_structs[struct_name]}' and is also an interface of "
                f"'{other_uses[struct_name]}'; give the fragment outputs their own structure",
                target,
            )
        for member in members:
            _rename_colour(getattr(member, "attributes", None))
    return program


def _rename_colour(attributes):
    for index, attribute in enumerate(attributes or []):
        match = _COLOUR.fullmatch(str(getattr(attribute, "name", "") or ""))
        if match is None or getattr(attribute, "arguments", None):
            continue
        attachment = int(match.group(1) or 0)
        attributes[index] = A.AttributeNode(
            "gl_FragData",
            [A.LiteralNode(attachment, A.PrimitiveType("int"))],
            source_location=getattr(attribute, "source_location", None),
        )


def _colour_index(member):
    for attribute in getattr(member, "attributes", None) or []:
        match = _COLOUR.fullmatch(str(getattr(attribute, "name", "") or ""))
        if match is not None and not getattr(attribute, "arguments", None):
            return int(match.group(1) or 0)
    return None


def _is_output(parameter):
    qualifiers = {
        str(qualifier).lower() for qualifier in getattr(parameter, "qualifiers", None) or []
    }
    return bool(qualifiers & {"out", "inout"})


def _is_void(type_node):
    return _type_name(type_node) == "void"


def _type_name(type_node):
    while isinstance(type_node, A.ArrayType):
        type_node = type_node.element_type
    name = getattr(type_node, "name", None)
    return name if isinstance(name, str) else str(type_node or "")


def _stage_name(key, stage):
    value = getattr(key, "value", key)
    return str(value or getattr(stage, "stage", "") or "").lower()


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    if isinstance(stages, dict):
        return list(stages.items())
    return [(getattr(stage, "stage", None), stage) for stage in stages]
