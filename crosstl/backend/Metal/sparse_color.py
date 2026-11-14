"""Import MSL sparse texture reads, which return ``sparse_color<T>``, as CrossGL texture residency queries.

The ``sparse_sample``, ``sparse_sample_compare``, ``sparse_read``, ``sparse_gather`` and ``sparse_gather_compare``
texture members return ``sparse_color<T>``: ``value()`` is the texel and ``resident()`` whether every texel the read
touched was resident (Metal Shading Language specification, section 6.12). CrossGL has no such aggregate; it wraps
the ordinary read instead (:mod:`crosstl.translator.codegen.texture_residency`). A local initialised from a sparse
read becomes the texel and its residency code::

    sparse_color<float4> c = t.sparse_sample(s, uv);   // MSL
    vec4 crosstl_sparse_c_value;                        // CrossGL
    int crosstl_sparse_c_code = textureResidency(texture(t, s, uv), crosstl_sparse_c_value);

and ``c.value()`` reads ``crosstl_sparse_c_value`` while ``c.resident()`` is ``textureResident(crosstl_sparse_c_code)``.
The read itself goes through the importer's ordinary texture path, as ``t.sample(s, uv)`` would. ``value()`` of a
sparse read used directly in an expression is that ordinary read. Every other use of a ``sparse_color`` value (a
parameter, struct member, copy or return, or ``resident()`` of a read with no local to hold its texel) has no CrossGL
form and is refused.
"""

from __future__ import annotations

import copy
import re

from ...translator.codegen.texture_residency import (
    TEXTURE_RESIDENCY_OPERATION,
    TEXTURE_RESIDENCY_READS,
    TEXTURE_RESIDENT_OPERATION,
)
from .MetalAst import AssignmentNode, MethodCallNode, TextureSampleNode, VariableNode

__all__ = [
    "METAL_SPARSE_TEXTURE_METHODS",
    "import_sparse_color_assignment",
    "import_sparse_color_member",
    "is_sparse_color_type",
    "refuse_sparse_color_reference",
    "refuse_sparse_color_type",
]

# MSL texture member functions that return sparse_color<T>: the texel and whether its tile is resident.
METAL_SPARSE_TEXTURE_METHODS = frozenset(
    {
        "sparse_sample",
        "sparse_sample_compare",
        "sparse_read",
        "sparse_gather",
        "sparse_gather_compare",
    }
)
_SPARSE_PREFIX = "sparse_"
_SPARSE_COLOR = re.compile(
    r"^(?:const\s+)?(?:metal\s*::\s*)?sparse_color\s*<\s*(?P<element>.+?)\s*>\s*(?:const)?$"
)
_SPARSE_COLOR_ANYWHERE = re.compile(r"(?<![A-Za-z0-9_])sparse_color\s*<")
_TEXTURE_TYPE = re.compile(
    r"^(?:const\s+)?(?:metal\s*::\s*)?(?P<kind>texture|depth)[A-Za-z0-9_]*\s*<\s*(?P<scalar>[A-Za-z_][A-Za-z0-9_]*)"
)
_AUTO = frozenset({"auto", "const auto"})
_MEMBERS = frozenset({"value", "resident"})


class _SparseColorLocal(str):
    """The type recorded for a local imported as a texel and residency code, as opposed to a ``sparse_color``
    parameter, which has neither."""


def sparse_color_element(metal_type):
    """``T`` of ``sparse_color<T>``, or None for any other type."""
    match = _SPARSE_COLOR.match(str(metal_type or "").strip())
    return match.group("element") if match else None


def is_sparse_color_type(metal_type) -> bool:
    """Whether ``metal_type`` mentions ``sparse_color<...>`` anywhere (an array or template of it included)."""
    return bool(_SPARSE_COLOR_ANYWHERE.search(str(metal_type or "")))


def refuse_sparse_color_type(codegen, metal_type):
    """Refuse a ``sparse_color`` type that reached type mapping; None for any other type."""
    if not is_sparse_color_type(metal_type):
        return None
    return codegen.refuse_import(
        f"type '{str(metal_type).strip()}': CrossGL has no sparse_color aggregate; only a local initialised from a "
        "sparse texture read is imported (as the texel and a textureResidency code)",
        placeholder="int",
    )


def _local_names(name):
    return f"crosstl_sparse_{name}_value", f"crosstl_sparse_{name}_code"


def _sparse_local(codegen, expr):
    """The source name of the imported ``sparse_color`` local ``expr`` names, or None."""
    if not isinstance(expr, VariableNode) or getattr(expr, "vtype", None):
        return None
    recorded = getattr(codegen, "current_variable_types", {}).get(expr.name)
    return expr.name if isinstance(recorded, _SparseColorLocal) else None


def _plain_read(expr):
    """The ordinary texture read of a ``t.sparse_*(...)`` call, or None for any other expression."""
    if (
        not isinstance(expr, MethodCallNode)
        or expr.method not in METAL_SPARSE_TEXTURE_METHODS
    ):
        return None
    method = expr.method[len(_SPARSE_PREFIX) :]
    if method == "sample":
        # The parser builds ``t.sample(...)`` as a TextureSampleNode; build the same node for its sparse form.
        args = list(expr.args or [])
        options = args[2:]
        read = TextureSampleNode(
            expr.object,
            args[0] if args else None,
            args[1] if len(args) > 1 else None,
            options[0] if options else None,
        )
        if options:
            read.options = options
    else:
        read = copy.copy(expr)
        read.method = method
    location = getattr(expr, "source_location", None)
    if location is not None:
        read.source_location = location
    return read


def _render_read(codegen, sparse_call, read, is_main):
    """CrossGL for the ordinary read, refused when it is not a read CrossGL can report residency for."""
    pending = len(codegen.refusals.reasons)
    text = codegen.generate_expression(read, is_main)
    if len(codegen.refusals.reasons) > pending:
        return text
    name = text.split("(", 1)[0].strip() if text.endswith(")") else ""
    if name not in TEXTURE_RESIDENCY_READS:
        return codegen.refuse_import(
            f"sparse texture {sparse_call.method}: the read imports as '{text}', which has no residency-reporting "
            "CrossGL form",
            placeholder=text,
        )
    return text


def _element_type(codegen, declaration, sparse_call, read):
    """``T`` of the declared ``sparse_color<T>``, or for ``auto`` the texel type of the read (MSL section 6.12)."""
    element = sparse_color_element(getattr(declaration, "vtype", None))
    if element is not None:
        return element
    inferred = codegen.expression_metal_type(read)
    if inferred:
        return str(inferred).strip()
    texture = _TEXTURE_TYPE.match(
        str(codegen.expression_metal_type(sparse_call.object) or "").strip()
    )
    if texture is None:
        return None
    # A depth texture's sample, compare and read return one component; every gather returns four.
    if texture.group("kind") == "depth" and "gather" not in sparse_call.method:
        return texture.group("scalar")
    return f"{texture.group('scalar')}4"


def import_sparse_color_assignment(codegen, stmt, is_main, indent):
    """CrossGL statements for a ``sparse_color`` local's declaration or assignment, or None for any other."""
    if not isinstance(stmt, AssignmentNode):
        return None
    left = stmt.left
    if not isinstance(left, VariableNode):
        return None
    vtype = str(getattr(left, "vtype", None) or "").strip()
    read = _plain_read(stmt.right)
    declares = bool(vtype)
    if declares:
        if sparse_color_element(vtype) is None and not (
            read is not None and vtype in _AUTO
        ):
            return None
    elif _sparse_local(codegen, left) is None:
        return None
    if read is None or stmt.operator != "=" or getattr(left, "array_sizes", None):
        return codegen.refuse_import(
            f"sparse_color local '{left.name}': only a single local assigned directly from a sparse texture read "
            "is imported (as the texel and a textureResidency code)",
            placeholder=f"int {left.name}",
        )
    value_name, code_name = _local_names(left.name)
    read_text = _render_read(codegen, stmt.right, read, is_main)
    query = f"{TEXTURE_RESIDENCY_OPERATION}({read_text}, {value_name})"
    if not declares:
        return f"{code_name} = {query};\n"
    element = _element_type(codegen, left, stmt.right, read)
    if not element:
        return codegen.refuse_import(
            f"sparse texture {stmt.right.method}: the texel type of '{left.name}' could not be determined",
            placeholder="\n",
        )
    codegen.current_variable_types[left.name] = _SparseColorLocal(
        f"sparse_color<{element}>"
    )
    getattr(codegen, "current_variable_type_qualifiers", {}).pop(left.name, None)
    pad = "    " * indent
    return (
        f"{codegen.map_type(element)} {value_name};\n"
        f"{pad}int {code_name} = {query};\n"
    )


def import_sparse_color_member(codegen, expr, is_main):
    """CrossGL for ``value()`` or ``resident()`` of a ``sparse_color``, or None when ``expr`` is not one."""
    if not isinstance(expr, MethodCallNode):
        return None
    local = _sparse_local(codegen, expr.object)
    read = None if local is not None else _plain_read(expr.object)
    if local is None and read is None:
        if isinstance(expr.object, VariableNode) and is_sparse_color_type(
            getattr(codegen, "current_variable_types", {}).get(expr.object.name)
        ):
            return refuse_sparse_color_reference(codegen, expr.object)
        return None
    if expr.method not in _MEMBERS or expr.args:
        return codegen.refuse_import(
            f"sparse_color member '{expr.method}': only value() and resident() of a sparse texture read are imported"
        )
    if local is not None:
        value_name, code_name = _local_names(local)
        if expr.method == "value":
            return value_name
        return f"{TEXTURE_RESIDENT_OPERATION}({code_name})"
    if expr.method == "value":
        return _render_read(codegen, expr.object, read, is_main)
    return codegen.refuse_import(
        f"sparse texture {expr.object.method}(...).resident(): CrossGL reports residency through "
        "textureResidency, which needs a local to receive the texel; declare the sparse_color as a local"
    )


def refuse_sparse_color_reference(codegen, expr):
    """Refuse a ``sparse_color`` value used other than through ``value()`` or ``resident()``; None otherwise."""
    if not isinstance(expr, VariableNode) or getattr(expr, "vtype", None):
        return None
    if not is_sparse_color_type(
        getattr(codegen, "current_variable_types", {}).get(expr.name)
    ):
        return None
    return codegen.refuse_import(
        f"sparse_color value '{expr.name}': CrossGL has no sparse_color aggregate, so only its value() and "
        "resident() are imported",
        placeholder=expr.name,
    )
