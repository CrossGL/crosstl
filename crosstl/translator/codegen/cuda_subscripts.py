"""Subscripts of CrossGL vectors and matrices in CUDA and HIP output.

CrossGL, like GLSL, reads column ``i`` of a matrix with ``m[i]`` and element ``j`` of that column with ``m[i][j]``
(GLSL 4.60 section 5.6), and component ``i`` of a vector with ``v[i]``; ``i`` may be a runtime value. CUDA's and
HIP's vector types (``float3``, ``int4`` ...) have no subscript operator, and the matrix structures the generators
define store their elements column by column in one flat array, so neither subscript can be copied unchanged.

* A matrix's ``operator[]`` returns a column view: a small structure of references to the column's elements, named
  like the vector's components, with a subscript, assignment from a vector and conversion to one. ``m[i] = v``,
  ``m[i][j] += x`` and ``m[i].y = x`` write through it.
* A column read as a value is spelled ``m.column(i)``, which returns the column vector, so overloaded and template
  functions receive a real vector. A column passed to an ``out`` or ``inout`` parameter is refused: a C++ reference
  to the vector cannot bind to it.
* A vector subscript with a constant index names the component (``v.y``). A runtime index calls
  ``cgl_vector_componentN(v, i)``, which returns a reference to the component, so the subscript can be read,
  assigned or passed to an ``out`` parameter, and the index expression is evaluated once.

GLSL leaves an index past the last column or component undefined (section 5.11), as HLSL and Metal do. The helpers
return the last column or component for such an index, so the access stays inside the value. A constant index past
the end does not compile in the source languages and is refused.
"""

from __future__ import annotations

import re

from ..matrix_shape import crossgl_matrix_shape
from .c_reference_parameters import is_output_parameter

_COMPONENTS = ("x", "y", "z", "w")
_REFERENCE_NODES = "_cgl_subscript_reference_nodes"
# The generators' own matrix structures: ``float3x2`` has 3 columns of ``float2``, like CrossGL's ``mat3x2``.
_GENERATED_MATRIX = re.compile(r"(float|double)([234])x([234])")


def matrix_shape(type_name):
    """``(scalar, columns, rows)`` of a CrossGL or generated CUDA/HIP matrix type name, or ``None``."""
    if not isinstance(type_name, str):
        return None
    text = type_name.strip()
    generated = _GENERATED_MATRIX.fullmatch(text)
    if generated is not None:
        return generated.group(1), int(generated.group(2)), int(generated.group(3))
    return crossgl_matrix_shape(text)


def matrix_column_type(type_name):
    """The CrossGL column vector type of the matrix type ``type_name``, or ``None`` when it is not a matrix."""
    shape = matrix_shape(type_name)
    if shape is None:
        return None
    scalar, _columns, rows = shape
    return f"{'d' if scalar == 'double' else ''}vec{rows}"


def _select_component(names, index="index"):
    """A conditional expression choosing one of ``names`` by ``index``; the last one for any other index."""
    expression = names[-1]
    for position in range(len(names) - 2, -1, -1):
        expression = f"{index} == {position} ? {names[position]} : ({expression})"
    return expression


def _column_view_lines(scalar, rows):
    vector = f"{scalar}{rows}"
    names = _COMPONENTS[:rows]
    load = f"make_{vector}({', '.join(names)})"
    lines = []
    for const in (False, True):
        view = f"cgl_{vector}_{'const_' if const else ''}column"
        element = f"{'const ' if const else ''}{scalar}"
        bindings = ", ".join(f"{name}(p[{row}])" for row, name in enumerate(names))
        lines.append(f"struct {view} {{")
        lines.extend(f"    {element}& {name};" for name in names)
        lines.extend(
            [
                f"    __host__ __device__ explicit {view}({element}* p) : {bindings} {{ }}",
                f"    __host__ __device__ operator {vector}() const {{ return {load}; }}",
                (
                    f"    __host__ __device__ {element}& operator[](int index) const "
                    f"{{ return {_select_component(names)}; }}"
                ),
            ]
        )
        if not const:
            stores = " ".join(f"{name} = value.{name};" for name in names)
            lines.extend(
                [
                    f"    __host__ __device__ {view}& operator=(const {vector}& value) {{ {stores} return *this; }}",
                    (
                        f"    __host__ __device__ {view}& operator=(const {view}& value) "
                        f"{{ return *this = static_cast<{vector}>(value); }}"
                    ),
                ]
            )
        lines.extend(["};", ""])
    return lines


def matrix_column_view_helpers():
    """C++ column views for every matrix column type, emitted before the matrix structures that return them."""
    lines = [
        "// CrossGL matrix columns: m[i] is column i of a column-major matrix and m[i][j] its element j.",
    ]
    for scalar in ("float", "double"):
        for rows in range(2, 5):
            lines.extend(_column_view_lines(scalar, rows))
    return lines


def matrix_column_members(scalar, columns, rows):
    """The members of the ``{scalar}{columns}x{rows}`` structure that subscript it by column."""
    vector = f"{scalar}{rows}"
    view = f"cgl_{vector}_column"
    const_view = f"cgl_{vector}_const_column"
    start = f"(static_cast<unsigned int>(index) < {columns}u ? index : {columns - 1}) * {rows}"
    load = ", ".join(f"m[start + {row}]" for row in range(rows))
    return [
        f"    __host__ __device__ {view} operator[](int index) {{ return {view}(m + {start}); }}",
        f"    __host__ __device__ {const_view} operator[](int index) const {{ return {const_view}(m + {start}); }}",
        (
            f"    __host__ __device__ {vector} column(int index) const "
            f"{{ const int start = {start}; return make_{vector}({load}); }}"
        ),
    ]


def _vector_component_helper(count):
    name = f"cgl_vector_component{count}"
    names = [f"v.{component}" for component in _COMPONENTS[:count]]
    text = (
        "template <typename V>\n"
        f"__host__ __device__ inline auto {name}(V&& v, int index) -> decltype((v.x))\n"
        "{\n"
        f"    return {_select_component(names)};\n"
        "}"
    )
    return name, text


def _reference_nodes(generator):
    nodes = getattr(generator, _REFERENCE_NODES, None)
    if nodes is None:
        nodes = set()
        setattr(generator, _REFERENCE_NODES, nodes)
    return nodes


def visit_as_reference(generator, node):
    """Visit ``node`` so that a matrix column it names stays a writable column view."""
    nodes = _reference_nodes(generator)
    key = id(node)
    added = key not in nodes
    nodes.add(key)
    try:
        return generator.visit(node)
    finally:
        if added:
            nodes.discard(key)


def _type_name(generator, type_name):
    if type_name is None or isinstance(type_name, str):
        return type_name
    converter = getattr(generator, "convert_type_node_to_string", None)
    return converter(type_name) if converter is not None else None


def _vector_components(generator, type_name):
    info = generator.vector_type_info(type_name)
    if info is not None:
        return tuple(info["components"])
    component_type = generator.native_vector_component_type(type_name)
    if component_type is not None and isinstance(type_name, str):
        width = type_name[-1]
        if width in "234":
            return _COMPONENTS[: int(width)]
    return None


def _visit_index(generator, index_node):
    generator.subscript_index_depth += 1
    try:
        return generator.visit(index_node)
    finally:
        generator.subscript_index_depth -= 1


def _needs_reference(generator, node):
    if id(node) in _reference_nodes(generator):
        return True
    return generator.assignment_lhs_depth > 0 and generator.subscript_index_depth == 0


def vector_matrix_subscript(generator, node, refused, target):
    """The CUDA or HIP text of the subscript ``node`` when it indexes a vector or a matrix, else ``None``.

    ``refused`` is the generator's typed lowering diagnostic and ``target`` its language name.
    """
    array_node = getattr(node, "array_expr", getattr(node, "array", None))
    index_node = getattr(node, "index_expr", getattr(node, "index", None))
    if array_node is None or index_node is None:
        return None
    base_type = _type_name(generator, generator.expression_result_type(array_node))
    shape = matrix_shape(base_type)
    if shape is not None:
        _scalar, columns, _rows = shape
        index = generator.literal_int_value(index_node)
        if index is not None and not 0 <= index < columns:
            raise refused(
                [
                    f"cannot subscript the {base_type} matrix with index {index} in {target}: "
                    f"its columns are 0 to {columns - 1}"
                ]
            )
        array = generator.visit(array_node)
        index_text = _visit_index(generator, index_node)
        if _needs_reference(generator, node):
            return f"{array}[{index_text}]"
        return f"{array}.column({index_text})"

    components = _vector_components(generator, base_type)
    if components is None:
        return None
    index = generator.literal_int_value(index_node)
    if index is not None:
        if not 0 <= index < len(components):
            raise refused(
                [
                    f"cannot subscript the {base_type} vector with index {index} in {target}: "
                    f"its components are 0 to {len(components) - 1}"
                ]
            )
        return f"{visit_as_reference(generator, array_node)}.{components[index]}"
    array = visit_as_reference(generator, array_node)
    index_text = _visit_index(generator, index_node)
    name, text = _vector_component_helper(len(components))
    generator.helper_functions.setdefault(name, text)
    return f"{name}({array}, {index_text})"


def matrix_column_output_arguments(generator, params, raw_args):
    """Positions (1-based) of matrix columns ``m[i]`` passed to ``out`` or ``inout`` parameters."""
    positions = []
    for position, (param, arg) in enumerate(zip(params, raw_args), start=1):
        if not is_output_parameter(param):
            continue
        if type(arg).__name__ != "ArrayAccessNode":
            continue
        array_node = getattr(arg, "array_expr", getattr(arg, "array", None))
        base_type = _type_name(generator, generator.expression_result_type(array_node))
        if matrix_column_type(base_type) is not None:
            positions.append(position)
    return positions
