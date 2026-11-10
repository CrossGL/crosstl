"""Slang entry points for stage inputs and outputs that CrossGL declares as module-scope ``in`` and ``out`` variables.

CrossGL keeps GLSL's ``layout(location = N) out vec4 color;`` as ``out vec4 color @location(N);``. The GLSL import
does so for a fragment shader with more than one colour output, and written CrossGL can declare any stage's inputs and
outputs that way. Slang has no such global: as in HLSL, a module-scope variable without ``static`` is a uniform
parameter. The Slang output therefore read the inputs from a constant buffer, and a write to an output did not compile
(slangc E30011, left of '=' is not an l-value).

Slang declares stage inputs and outputs as entry-point parameters. Each such variable becomes a ``static`` global, so
the entry point and the functions it calls keep reading and writing it by name, and an entry point gains one ``in`` or
``out`` parameter per variable that it or a function it calls names. The parameter is placed like any ``@location``
varying (slang_varying_locations.py); a variable without a location takes the lowest free one of its direction. The
body first copies each input parameter into its global. A ``defer`` block copies each output global into its
parameter when the entry point returns, on every path and after a returned value is computed, as SPIRV-Cross's HLSL
output does at the end of its entry wrapper.
"""

from __future__ import annotations

from ..ast import (
    AttributeNode,
    IdentifierNode,
    LiteralNode,
    ParameterNode,
    PrimitiveType,
)
from .slang_varying_locations import varying_location

__all__ = [
    "STAGE_IO_LOCATION_ANNOTATION",
    "STAGE_IO_PARAMETER_ANNOTATION",
    "collect_stage_io_globals",
    "stage_io_global_declaration",
    "stage_io_output_names",
    "stage_io_parameters",
    "stage_io_prologue",
]

STAGE_IO_PARAMETER_ANNOTATION = "slang_stage_io_global"
STAGE_IO_LOCATION_ANNOTATION = "slang_stage_io_location"

_DIRECTIONS = ("in", "out")
_LOWERED_STAGES = {"vertex", "fragment"}
_STORAGE_QUALIFIERS = {
    "uniform",
    "buffer",
    "shared",
    "groupshared",
    "workgroup",
    "threadgroup",
    "const",
    "constant",
    "static",
}


def _qualifiers(node):
    return [
        str(qualifier).lower() for qualifier in getattr(node, "qualifiers", None) or []
    ]


_DIRECTION_ATTRIBUTES = {"in": "in", "input": "in", "out": "out", "output": "out"}


def _direction(node):
    """``in`` or ``out`` for a module-scope stage input or output, else None.

    The direction is a qualifier (``out vec4 color``) or, from the SPIR-V import, an ``@input`` or ``@output``
    attribute.
    """
    qualifiers = set(_qualifiers(node))
    if qualifiers & _STORAGE_QUALIFIERS:
        return None
    directions = {direction for direction in _DIRECTIONS if direction in qualifiers}
    directions.update(
        _DIRECTION_ATTRIBUTES[name]
        for name in (
            str(getattr(attribute, "name", "")).lower()
            for attribute in getattr(node, "attributes", None) or []
        )
        if name in _DIRECTION_ATTRIBUTES
    )
    if len(directions) != 1:
        return None
    name = str(getattr(node, "name", "") or "")
    return next(iter(directions)) if name else None


def _is_builtin_input(variable, direction):
    """A ``gl_`` input, which an entry point reads through its system-value parameters instead."""
    return direction == "in" and str(getattr(variable, "name", "")).startswith("gl_")


def _named_identifiers(codegen, root):
    return {
        node.name
        for node in codegen.walk_ast(root)
        if isinstance(node, IdentifierNode)
        and isinstance(getattr(node, "name", None), str)
    }


def collect_stage_io_globals(codegen, ast):
    """Map each entry point's id to the ``(variable, direction)`` stage variables it uses, and their ids.

    The candidates are the shader's module-scope variables and every stage block's variables; an entry point uses one
    when its body, or the body of a function it reaches by name, names it. Only vertex and fragment entry points are
    lowered; a variable that an entry point of another stage uses keeps its former declaration everywhere.
    """
    candidates = [
        (variable, _direction(variable))
        for variable in getattr(ast, "global_variables", None) or []
    ]
    stage_candidates = {}
    functions_by_name = {}
    for function in getattr(ast, "functions", None) or []:
        functions_by_name.setdefault(getattr(function, "name", None), []).append(
            function
        )
    stages = getattr(ast, "stages", None) or {}
    for stage_type, stage in stages.items():
        stage_candidates[id(stage)] = [
            (variable, _direction(variable))
            for variable in getattr(stage, "local_variables", None) or []
        ]
        for function in getattr(stage, "local_functions", None) or []:
            functions_by_name.setdefault(getattr(function, "name", None), []).append(
                function
            )

    if not any(
        direction is not None
        for _variable, direction in candidates
        + [item for items in stage_candidates.values() for item in items]
    ):
        return {}, set()

    def reached_names(entry):
        names = set()
        pending = [entry]
        visited = set()
        while pending:
            function = pending.pop()
            if id(function) in visited:
                continue
            visited.add(id(function))
            named = _named_identifiers(codegen, getattr(function, "body", None))
            names |= named
            for name in named:
                pending.extend(functions_by_name.get(name, ()))
        return names

    entries = []
    for function in getattr(ast, "functions", None) or []:
        stage_name = codegen.get_function_stage(function)
        if stage_name:
            entries.append((function, stage_name, candidates))
    for stage_type, stage in stages.items():
        entry = getattr(stage, "entry_point", None)
        if entry is not None:
            entries.append(
                (
                    entry,
                    codegen.get_stage_name(stage_type),
                    candidates + stage_candidates.get(id(stage), []),
                )
            )

    used = {}
    refused_ids = set()
    for entry, stage_name, scope in entries:
        names = reached_names(entry)
        parameter_names = {
            getattr(parameter, "name", None)
            for parameter in getattr(entry, "parameters", None)
            or getattr(entry, "params", None)
            or []
        }
        selected = [
            (variable, direction)
            for variable, direction in scope
            if direction is not None
            and variable.name in names
            and variable.name not in parameter_names
        ]
        if not selected:
            continue
        if codegen.slang_shader_stage_name(stage_name) not in _LOWERED_STAGES:
            # Other stages take arrays and primitive streams that these parameters do not model.
            refused_ids.update(id(variable) for variable, _direction in selected)
            continue
        used[id(entry)] = selected
    used = {
        entry_id: selected
        for entry_id, selected in used.items()
        if not any(id(variable) in refused_ids for variable, _direction in selected)
    }
    # Unused stage variables become static too: as uniforms they would be shader parameters (slangc E39019).
    variable_ids = {
        id(variable)
        for variable, direction in candidates
        + [item for items in stage_candidates.values() for item in items]
        if direction is not None and id(variable) not in refused_ids
    }
    return used, variable_ids


def stage_io_global_declaration(codegen, node):
    """The ``static`` declaration of a stage input or output variable, or none for a built-in input."""
    if _is_builtin_input(node, _direction(node)):
        return ""
    vtype = codegen.variable_declaration_type(node)
    codegen.register_variable_type(node.name, vtype, node)
    declaration = codegen.format_declaration(vtype, node.name, node)
    initial_value = getattr(node, "initial_value", getattr(node, "value", None))
    if initial_value is not None:
        value = codegen.generate_expression_with_expected(
            initial_value, codegen.initializer_expected_type(vtype)
        )
        return f"static {declaration} = {value};\n"
    return f"static {declaration};\n"


def _unique_name(base, used_names):
    name = base
    suffix = 1
    while name in used_names:
        name = f"{base}_{suffix}"
        suffix += 1
    used_names.add(name)
    return name


def _location_attribute(index):
    return AttributeNode("location", [LiteralNode(index, PrimitiveType("int"))])


def stage_io_parameters(codegen, entry, fragment_stage):
    """The entry parameters of ``entry``'s stage variables, each annotated with the variable it carries."""
    selected = codegen.slang_stage_io_globals.get(id(entry))
    if not selected:
        return []
    used_names = set(codegen.user_symbol_names)
    used_locations = {direction: set() for direction in _DIRECTIONS}
    selected = [
        (variable, direction)
        for variable, direction in selected
        if not _is_builtin_input(variable, direction)
    ]
    for variable, direction in selected:
        location = varying_location(variable)
        if location is not None:
            used_locations[direction].add(location)

    parameters = []
    unnumbered = []
    for variable, direction in selected:
        attributes = [
            attribute
            for attribute in getattr(variable, "attributes", None) or []
            if str(getattr(attribute, "name", "")).lower() == "location"
        ]
        builtin = _builtin_output_attribute(codegen, variable)
        if builtin is not None:
            # A built-in output such as gl_Position keeps its system value and takes no location, unless Slang
            # spells it with a user semantic (gl_PointSize is PSIZE), which is placed after the others.
            attributes = [builtin]
        elif varying_location(variable) is None:
            location = 0
            while location in used_locations[direction]:
                location += 1
            used_locations[direction].add(location)
            attributes = [_location_attribute(location)]
        interpolation = [
            name
            for name in _qualifiers(variable)
            + [
                str(getattr(attribute, "name", "")).lower()
                for attribute in variable.attributes or []
            ]
            if codegen.slang_interpolation_qualifier(name)
        ]
        parameter = ParameterNode(
            name=_unique_name(f"crossgl_{direction}_{variable.name}", used_names),
            param_type=getattr(variable, "var_type", getattr(variable, "vtype", None)),
            attributes=attributes,
            qualifiers=[direction] + (interpolation if direction == "in" else []),
            source_location=getattr(variable, "source_location", None),
        )
        parameter.add_annotation(STAGE_IO_PARAMETER_ANNOTATION, variable)
        if builtin is not None:
            mapped = str(codegen.map_semantic(builtin.name) or "")
            if not mapped.upper().startswith("SV_"):
                unnumbered.append((parameter, direction))
        elif fragment_stage and direction == "out":
            codegen.slang_fragment_output_location_ids.add(id(parameter))
        parameters.append(parameter)
    for parameter, direction in unnumbered:
        location = 0
        while location in used_locations[direction]:
            location += 1
        used_locations[direction].add(location)
        parameter.add_annotation(STAGE_IO_LOCATION_ANNOTATION, location)
    return parameters


def _builtin_output_attribute(codegen, variable):
    """The attribute naming the built-in that a ``gl_`` output variable writes, or None for a user varying."""
    name = str(getattr(variable, "name", "") or "")
    if not name.startswith("gl_"):
        return None
    for attribute in getattr(variable, "attributes", None) or []:
        if str(getattr(attribute, "name", "")) == name or str(
            getattr(attribute, "name", "")
        ).startswith("gl_"):
            return attribute
    return AttributeNode(name)


def stage_io_output_names(parameters):
    """Names of the output variables that ``parameters`` carry, which no other output lowering may claim."""
    names = set()
    for parameter in parameters:
        variable = parameter.get_annotation(STAGE_IO_PARAMETER_ANNOTATION)
        if variable is not None and "out" in _qualifiers(parameter):
            names.add(variable.name)
    return names


def stage_io_prologue(codegen, parameters, indent):
    """Copy-in statements and the ``defer`` copy-out block that open an entry point with ``parameters``."""
    copies_in = []
    copies_out = []
    for parameter in parameters:
        variable = parameter.get_annotation(STAGE_IO_PARAMETER_ANNOTATION)
        if variable is None:
            continue
        name = codegen.generate_expression(IdentifierNode(variable.name))
        if "out" in _qualifiers(parameter):
            copies_out.append(f"{parameter.name} = {name};")
        else:
            copies_in.append(f"{name} = {parameter.name};")
    lines = [f"{indent}{line}\n" for line in copies_in]
    if copies_out:
        lines.append(f"{indent}defer\n{indent}{{\n")
        lines.extend(f"{indent}    {line}\n" for line in copies_out)
        lines.append(f"{indent}}}\n")
    return "".join(lines)
