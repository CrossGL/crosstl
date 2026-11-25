"""Geometry entry points for rust-gpu, written from CrossGL's GLSL form of a geometry stage.

CrossGL writes a geometry stage the way GLSL 4.60 does (sections 4.4.1.2, 4.4.2.2 and 8.16): stage layouts name the
input primitive, the output primitive, the maximum vertex count and the invocation count; each per-vertex input is a
module-scope ``in`` array with one element per vertex of the input primitive; the stage writes its module-scope
``out`` values and the built-in outputs, then calls ``EmitVertex()``::

    in vec4 inPosition[2] @location(0);
    geometry {
        layout(lines, invocations = 2) in;
        layout(points, max_vertices = 1) out;
        void main() { gl_Position = inPosition[0] + inPosition[1]; EmitVertex(); EndPrimitive(); }
    }

rust-gpu (0.11) declares the same shader as one entry point, ``#[spirv(geometry(input_lines, output_points,
output_vertices = 1, invocations = 2))]``, whose parameters are the interface (rustc_codegen_spirv/src/symbols.rs,
``EXECUTION_MODES`` and ``BUILTINS``): an input array is a parameter taken by value, ``[Vec4; 2]``, and an output a
``&mut`` reference into the Output storage class that the stage writes before each ``spirv_std::arch::emit_vertex()``.
:func:`plan_rust_geometry_interface` gives each module-scope input and output, and each geometry built-in the program
names (``gl_InvocationID``, ``gl_PrimitiveIDIn``, ``gl_Position``, ...), an entry-parameter record that
:func:`.rust_resource_bindings.resource_parameters` threads to the helpers that use it. A user value keeps its
explicit location, and the others take the first free ones in declaration order
(:func:`.stage_io_semantic_locations.allocate_locations`). Outputs are written through their reference
(:mod:`.rust_workgroup_references`). ``EmitVertex``, ``EndPrimitive``, ``EmitStreamVertex(k)`` and
``EndStreamPrimitive(k)`` call the ``spirv_std::arch`` functions of the same name; a stream number must be an
integer literal, since rust-gpu takes it as a const generic.

Refused, with the value named: an entry point that takes parameters or returns a value (HLSL's stream-object form),
module-scope inputs or outputs in a program with another graphics stage (they would belong to no single entry point),
missing stage layouts, an input that is not an array of the input primitive's vertex count, interpolation modes
rust-gpu has no attribute for (``noperspective``, ``centroid``, ``sample``) and outputs in a vertex stream other than
0, which rust-gpu cannot decorate.
"""

from __future__ import annotations

from ..ast import (
    ArrayType,
    IdentifierNode,
    LiteralNode,
    PrimitiveType,
    VariableNode,
    VectorType,
)
from ..semantics.stage_varyings import explicit_location, location_span
from .ast_walk import walk_ast_values
from .rust_gpu_interface import workgroup_type_name
from .stage_io_semantic_locations import allocate_locations

__all__ = [
    "GEOMETRY_STAGE",
    "STAGE_INPUT_KIND",
    "STAGE_OUTPUT_KIND",
    "generate_rust_geometry_call",
    "plan_rust_geometry_interface",
    "refuse_rust_geometry_entry",
    "rust_geometry_stage_attribute",
]

GEOMETRY_STAGE = "geometry"
STAGE_INPUT_KIND = "stage_input"
STAGE_OUTPUT_KIND = "stage_output"

# GLSL input primitive -> (rust-gpu execution mode, vertices per primitive).
_INPUT_PRIMITIVES = {
    "points": ("input_points", 1),
    "lines": ("input_lines", 2),
    "lines_adjacency": ("input_lines_adjacency", 4),
    "triangles": ("triangles", 3),
    "triangles_adjacency": ("input_triangles_adjacency", 6),
}
_OUTPUT_PRIMITIVES = {
    "points": "output_points",
    "line_strip": "output_line_strip",
    "triangle_strip": "output_triangle_strip",
}
# The geometry built-ins (GLSL 4.60 section 7.1.3): CrossGL name -> (rust-gpu built-in, CrossGL type).
_INPUT_BUILTINS = {
    "gl_InvocationID": ("invocation_id", "int"),
    "gl_PrimitiveIDIn": ("primitive_id", "int"),
}
_OUTPUT_BUILTINS = {
    "gl_Position": ("position", "vec4"),
    "gl_PointSize": ("point_size", "float"),
    "gl_Layer": ("layer", "int"),
    "gl_ViewportIndex": ("viewport_index", "int"),
    "gl_PrimitiveID": ("primitive_id", "int"),
}
_CALLS = {"EmitVertex": "emit_vertex", "EndPrimitive": "end_primitive"}
_STREAM_CALLS = {
    "EmitStreamVertex": "emit_stream_vertex",
    "EndStreamPrimitive": "end_stream_primitive",
}
_OTHER_GRAPHICS_STAGES = frozenset(
    {
        "vertex",
        "fragment",
        "tessellation_control",
        "tessellation_evaluation",
        "mesh",
        "task",
        "amplification",
        "object",
    }
)
_UNSUPPORTED_INTERPOLATION = ("noperspective", "centroid", "sample")
_DIRECTIONS = {"in": "in", "input": "in", "out": "out", "output": "out"}


def _names(items):
    return [str(getattr(item, "name", item)).lower() for item in items or []]


def _direction(node):
    """``"in"`` or ``"out"`` for a module-scope stage input or output declaration, else ``None``."""
    if not isinstance(node, VariableNode):
        return None
    for name in _names(getattr(node, "qualifiers", None)) + _names(
        getattr(node, "attributes", None)
    ):
        if name in _DIRECTIONS:
            return _DIRECTIONS[name]
    return None


def _literal(node):
    """The value of an integer literal (``3``, ``3u``), else ``None``."""
    value = node.value if isinstance(node, LiteralNode) else node
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip().rstrip("uU") if isinstance(value, str) else ""
    return int(text) if text.isdigit() else None


def _geometry_stage(ast):
    for stage_type, stage in (getattr(ast, "stages", None) or {}).items():
        if str(getattr(stage_type, "value", stage_type)).lower() == GEOMETRY_STAGE:
            return stage
    return None


def _layout_modes(generator, stage):
    """The rust-gpu execution modes of ``stage``'s layouts and its input vertex count, refusing what is missing."""
    entries = {"in": {}, "out": {}}
    for layout in getattr(stage, "layout_qualifiers", None) or []:
        direction = _DIRECTIONS.get(str(getattr(layout, "direction", "")).lower())
        if direction is None:
            continue
        for entry in getattr(layout, "entries", None) or []:
            arguments = getattr(entry, "arguments", None) or []
            value = _literal(arguments[0]) if arguments else None
            entries[direction][str(getattr(entry, "name", entry)).lower()] = value
    inputs = [name for name in entries["in"] if name in _INPUT_PRIMITIVES]
    outputs = [name for name in entries["out"] if name in _OUTPUT_PRIMITIVES]
    max_vertices = entries["out"].get("max_vertices")
    if len(inputs) != 1 or len(outputs) != 1 or not max_vertices:
        generator.refuse_lowering(
            "stage: a geometry stage names one input primitive (layout(points|lines|... ) in), one output primitive "
            "and max_vertices (layout(points|line_strip|triangle_strip, max_vertices = N) out)"
        )
        return [], None
    mode, count = _INPUT_PRIMITIVES[inputs[0]]
    modes = [mode, _OUTPUT_PRIMITIVES[outputs[0]], f"output_vertices = {max_vertices}"]
    invocations = entries["in"].get("invocations")
    if invocations:
        modes.append(f"invocations = {invocations}")
    return modes, count


def _interpolation(generator, node):
    """The rust-gpu attributes of ``node``'s interpolation, or ``None`` when rust-gpu has no attribute for it."""
    names = set(_names(getattr(node, "qualifiers", None))) | set(
        _names(getattr(node, "attributes", None))
    )
    for mode in _UNSUPPORTED_INTERPOLATION:
        if mode in names:
            generator.refuse_lowering(
                f"stage: geometry value '{node.name}' is {mode}; rust-gpu has no attribute for that interpolation"
            )
            return None
    for attribute in getattr(node, "attributes", None) or []:
        if str(getattr(attribute, "name", "")).lower() == "stream":
            arguments = getattr(attribute, "arguments", None) or []
            if not arguments or _literal(arguments[0]) != 0:
                generator.refuse_lowering(
                    f"stage: geometry output '{node.name}' is in a vertex stream other than 0; rust-gpu cannot "
                    "decorate an output with its stream"
                )
                return None
    return ["flat"] if names & {"flat", "nointerpolation"} else []


def _declared_values(generator, ast, stage, count):
    """``(node, direction, CrossGL type, location span)`` of each module-scope or stage-level input and output."""
    candidates = list(getattr(ast, "global_variables", None) or [])
    candidates += list(getattr(stage, "local_variables", None) or [])
    values = []
    for node in candidates:
        direction = _direction(node)
        if direction is None:
            continue
        var_type = getattr(node, "var_type", None)
        if direction == "in":
            if not isinstance(var_type, ArrayType):
                generator.refuse_lowering(
                    f"stage: geometry input '{node.name}' is not an array of one value per input vertex"
                )
                continue
            size = getattr(var_type, "size", None)
            if size is not None and _literal(size) != count:
                generator.refuse_lowering(
                    f"stage: geometry input '{node.name}' has {_literal(size)} elements; the input primitive has "
                    f"{count} vertices"
                )
                continue
            element = generator.convert_type_node_to_string(var_type.element_type)
            values.append(
                (
                    node,
                    direction,
                    f"{element}[{count}]",
                    location_span(var_type.element_type),
                )
            )
            continue
        type_name = generator.rust_variable_source_type(node)
        values.append((node, direction, type_name, location_span(var_type)))
    return values


def _builtin_record(name, builtin, type_name, kind):
    node = VariableNode(name, _crossgl_type(type_name))
    if kind == STAGE_OUTPUT_KIND:
        type_name = workgroup_type_name(type_name)
    return {"node": node, "type_name": type_name, "kind": kind, "attribute": builtin}


def _crossgl_type(type_name):
    if type_name == "vec4":
        return VectorType(PrimitiveType("float"), 4)
    return PrimitiveType(type_name)


def plan_rust_geometry_interface(generator, ast, stage_names) -> None:
    """Record the entry-parameter records of ``ast``'s geometry interface on ``generator``, refusing what it cannot.

    ``generator.rust_geometry_interface`` maps each interface value's name to its record (``node``, ``type_name``,
    ``kind`` and the ``attribute`` inside ``#[spirv(...)]``); ``generator.rust_geometry_modes`` lists the execution
    modes of the stage attribute.
    """
    generator.rust_geometry_interface = {}
    generator.rust_geometry_modes = []
    if GEOMETRY_STAGE not in stage_names:
        return
    stage = _geometry_stage(ast)
    if stage is None:
        generator.refuse_lowering(
            "stage: a geometry entry point is written for rust-gpu from a geometry stage block"
        )
        return
    modes, count = _layout_modes(generator, stage)
    generator.rust_geometry_modes = modes
    if count is None:
        return
    values = _declared_values(generator, ast, stage, count)
    if values and stage_names & _OTHER_GRAPHICS_STAGES:
        generator.refuse_lowering(
            "stage interface: module-scope inputs and outputs of a program with a geometry stage and another "
            "graphics stage belong to no single rust-gpu entry point"
        )
        return
    interface = {}
    for direction, kind in (("in", STAGE_INPUT_KIND), ("out", STAGE_OUTPUT_KIND)):
        group = [value for value in values if value[1] == direction]
        locations = allocate_locations(
            [(explicit_location(node), span) for node, _d, _t, span in group]
        )
        for (node, _direction_name, type_name, _span), location in zip(
            group, locations
        ):
            interpolation = _interpolation(generator, node)
            if interpolation is None:
                continue
            if kind == STAGE_OUTPUT_KIND:
                type_name = workgroup_type_name(type_name)
            interface[node.name] = {
                "node": node,
                "type_name": type_name,
                "kind": kind,
                "attribute": ", ".join(interpolation + [f"location = {location}"]),
            }
    referenced = {
        value.name
        for value in walk_ast_values(ast)
        if isinstance(value, IdentifierNode)
    }
    for builtins, kind in (
        (_INPUT_BUILTINS, STAGE_INPUT_KIND),
        (_OUTPUT_BUILTINS, STAGE_OUTPUT_KIND),
    ):
        for name in sorted(builtins):
            if name in referenced and name not in interface:
                builtin, type_name = builtins[name]
                interface[name] = _builtin_record(name, builtin, type_name, kind)
    generator.rust_geometry_interface = interface


def rust_geometry_stage_attribute(generator) -> str:
    return f"#[spirv(geometry({', '.join(generator.rust_geometry_modes)}))]\n"


def refuse_rust_geometry_entry(generator, func, param_list, return_type) -> None:
    """Refuse a geometry entry point in HLSL's form, which takes its primitive and streams as parameters."""
    if param_list or generator.map_type(return_type) != "()":
        generator.refuse_lowering(
            f"stage: geometry entry point '{func.name}' takes parameters or returns a value; rust-gpu output "
            "writes geometry stages in GLSL's form (stage layouts, module-scope inputs and outputs, EmitVertex)"
        )


def generate_rust_geometry_call(generator, name, args):
    """The ``spirv_std::arch`` call of a geometry built-in call, or ``None`` when ``name`` is not one."""
    if name not in _CALLS and name not in _STREAM_CALLS:
        return None
    if generator.is_user_defined_function(name):
        return None
    if GEOMETRY_STAGE not in getattr(generator, "rust_gpu_program_stages", ()):
        return generator.refuse_lowering(
            f"function: '{name}' is a geometry-stage function", "()"
        )
    if name in _CALLS:
        if args:
            return None
        return f"unsafe {{ spirv_std::arch::{_CALLS[name]}() }}"
    stream = _literal(args[0]) if len(args) == 1 else None
    if not isinstance(stream, int):
        return generator.refuse_lowering(
            f"function: '{name}' needs a constant stream number; rust-gpu takes the stream as a const generic",
            "()",
        )
    return f"unsafe {{ spirv_std::arch::{_STREAM_CALLS[name]}::<{stream}>() }}"
