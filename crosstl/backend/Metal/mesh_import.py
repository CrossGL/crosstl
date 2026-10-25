"""Metal ``[[mesh]]`` functions imported as CrossGL mesh stages.

A Metal mesh function writes its geometry through a parameter of type ``metal::mesh<V, P, NV, NP, topology>``: it
stores vertex ``i`` with ``set_vertex(i, v)``, primitive ``i`` with ``set_primitive(i, p)``, one 8-bit vertex index
with ``set_index(i, value)`` and the number of primitives with ``set_primitive_count(n)``. A mesh function may also
read the object stage's payload through an ``object_data`` reference marked ``[[payload]]``. The importer used to
print such a function verbatim, as a plain helper that still called those methods, so no target received a mesh
stage.

CrossGL declares a mesh stage's outputs as ``@vertices``, ``@primitives`` and ``@indices`` out arrays and sizes them
with ``SetMeshOutputCounts(vertices, primitives)``, the HLSL form, which HLSL, GLSL, SPIR-V and Metal all lower. The
importer rewrites each Metal method onto that form:

* ``set_vertex(i, v)`` and ``set_primitive(i, p)`` store into the vertex and primitive arrays.
* ``set_primitive_count(n)`` becomes ``SetMeshOutputCounts(NV, n)``. HLSL requires the call once, in uniform control
  flow, before any output is written; the importer accepts the Metal call only as a single top-level statement of
  the function body that no output store or ``return`` precedes, and refuses every other placement. Metal leaves
  the vertex count implicit, so the declared maximum ``NV`` is used; vertices that no index names are never
  rasterized.
* ``set_index(i, value)`` writes one component of a primitive's index vector, which DXC rejects for mesh index
  outputs (they must be written as whole vectors). For line and triangle meshes the importer stages the 8-bit
  indices in threadgroup memory, then, after a threadgroup barrier at the end of the function, one invocation copies
  every primitive's indices into the index array as whole vectors. Point meshes have one index per primitive and
  store it directly. Because every invocation must reach that barrier, a mesh function that stages indices and
  returns early is refused.

Other uses of the mesh object, ``set_indices``, a missing ``set_primitive_count`` and a mesh type whose arguments
are not constant are refused through :meth:`refuse_import`. Object (``[[object]]``) functions are refused as well
until their payload and grid-size dispatch are imported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .MetalAst import MethodCallNode, ReturnNode, VariableNode

__all__ = ["MetalMeshImport", "MetalMeshOutput"]

_MESH_TYPE = re.compile(r"^(?:::)?(?:metal\s*::\s*)?mesh\s*<(?P<args>.*)>$", re.DOTALL)
_TOPOLOGY = {"point": 1, "line": 2, "triangle": 3}
_INDEX_TYPES = {1: "uint", 2: "uvec2", 3: "uvec3"}
_OUTPUT_METHODS = frozenset({"set_vertex", "set_primitive", "set_index", "set_indices"})
_MESH_METHODS = _OUTPUT_METHODS | {"set_primitive_count"}


def _split_template_arguments(text):
    arguments = []
    depth = 0
    start = 0
    for index, character in enumerate(text):
        if character in "<([":
            depth += 1
        elif character in ">)]":
            depth -= 1
        elif character == "," and depth == 0:
            arguments.append(text[start:index].strip())
            start = index + 1
    arguments.append(text[start:].strip())
    return arguments


def _walk(node, seen=None):
    """Every AST node under ``node``, depth first; strings and numbers are leaves."""
    if seen is None:
        seen = set()
    if node is None or isinstance(node, (str, int, float, bool)):
        return
    if isinstance(node, (list, tuple)):
        for item in node:
            yield from _walk(item, seen)
        return
    if isinstance(node, dict):
        for item in node.values():
            yield from _walk(item, seen)
        return
    if id(node) in seen:
        return
    seen.add(id(node))
    yield node
    for value in getattr(node, "__dict__", {}).values():
        yield from _walk(value, seen)


@dataclass
class MetalMeshOutput:
    """The CrossGL form of one Metal mesh function's ``metal::mesh`` parameter."""

    function: object
    parameter: object
    vertex_type: str
    primitive_type: str | None
    max_vertices: int
    max_primitives: int
    topology: str
    payload_parameters: set = field(default_factory=set)
    vertices: str = ""
    primitives: str = ""
    indices: str = ""
    count: str = ""
    staging: str = ""
    invocation: str = ""
    added_invocation: bool = False
    count_statement: object = None

    @property
    def index_width(self):
        return _TOPOLOGY[self.topology]


class MetalMeshImport:
    """Mixin for :class:`MetalToCrossGLConverter`: Metal mesh functions as CrossGL mesh stages."""

    current_metal_mesh_output = None

    def is_metal_mesh_object(self, node):
        output = self.current_metal_mesh_output
        return (
            output is not None
            and isinstance(node, VariableNode)
            and node.name == output.parameter.name
        )

    def metal_mesh_type_arguments(self, metal_type):
        """``(V, P, NV, NP, topology)`` for a ``metal::mesh`` type, or a string explaining why it is not one."""
        resolved = str(self.resolve_type_alias(str(metal_type)) or "").strip()
        match = _MESH_TYPE.match(resolved)
        if match is None:
            return None
        arguments = _split_template_arguments(match.group("args"))
        if len(arguments) != 5:
            return f"metal::mesh type {resolved} does not have five template arguments"
        vertex_type, primitive_type, max_vertices, max_primitives, topology = arguments
        extents = []
        for label, text in (("vertex", max_vertices), ("primitive", max_primitives)):
            value = self.evaluate_concrete_array_extent(text)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                return f"metal::mesh {label} count '{text}' is not a positive integer constant"
            extents.append(value)
        topology_name = re.sub(r"\s+", "", topology).rsplit("::", 1)[-1]
        if topology_name not in _TOPOLOGY:
            return f"metal::mesh topology '{topology}' is not point, line or triangle"
        primitive = (
            None
            if re.sub(r"\s+", "", primitive_type) in {"void", "metal::void"}
            else primitive_type
        )
        return vertex_type, primitive, extents[0], extents[1], topology_name

    def plan_metal_mesh_function(self, function):
        """The :class:`MetalMeshOutput` of a ``[[mesh]]`` function, or ``None`` after recording its refusal."""
        name = function.name
        outputs = []
        for parameter in function.params:
            arguments = self.metal_mesh_type_arguments(getattr(parameter, "vtype", ""))
            if isinstance(arguments, str):
                self.refuse_import(f"Metal mesh function '{name}': {arguments}")
                return None
            if arguments is not None:
                outputs.append((parameter, arguments))
        if len(outputs) != 1:
            self.refuse_import(
                f"Metal mesh function '{name}' must take exactly one metal::mesh parameter, found {len(outputs)}"
            )
            return None
        parameter, (vertex, primitive, max_vertices, max_primitives, topology) = (
            outputs[0]
        )
        output = MetalMeshOutput(
            function=function,
            parameter=parameter,
            vertex_type=vertex,
            primitive_type=primitive,
            max_vertices=max_vertices,
            max_primitives=max_primitives,
            topology=topology,
        )
        for candidate in function.params:
            if any(
                getattr(attribute, "name", None) == "payload"
                for attribute in getattr(candidate, "attributes", []) or []
            ):
                output.payload_parameters.add(id(candidate))
        if not self.validate_metal_mesh_calls(output):
            return None
        return output

    def validate_metal_mesh_calls(self, output):
        """Check where the function calls the mesh methods; record a refusal and return ``False`` if it cannot."""
        name = output.function.name
        body = list(getattr(output.function, "body", None) or [])
        mesh_name = output.parameter.name
        calls = [
            node
            for node in _walk(body)
            if isinstance(node, MethodCallNode)
            and isinstance(node.object, VariableNode)
            and node.object.name == mesh_name
        ]
        unsupported = sorted({call.method for call in calls} - _MESH_METHODS)
        if unsupported:
            self.refuse_import(
                f"Metal mesh function '{name}' calls metal::mesh::{', '.join(unsupported)}, which has no CrossGL form"
            )
            return False
        if any(call.method == "set_indices" for call in calls):
            self.refuse_import(
                f"Metal mesh function '{name}' calls metal::mesh::set_indices; import writes one index at a time "
                "with set_index"
            )
            return False
        counts = [call for call in calls if call.method == "set_primitive_count"]
        top_level = [
            statement for statement in body if any(statement is call for call in counts)
        ]
        if len(counts) != 1 or len(top_level) != 1:
            self.refuse_import(
                f"Metal mesh function '{name}' must call set_primitive_count exactly once, as a top-level "
                "statement, for CrossGL's SetMeshOutputCounts"
            )
            return False
        position = next(i for i, statement in enumerate(body) if statement is counts[0])
        for statement in body[:position]:
            for node in _walk(statement):
                if isinstance(node, ReturnNode) or (
                    isinstance(node, MethodCallNode)
                    and isinstance(node.object, VariableNode)
                    and node.object.name == mesh_name
                    and node.method in _OUTPUT_METHODS
                ):
                    self.refuse_import(
                        f"Metal mesh function '{name}' writes an output or returns before set_primitive_count; "
                        "SetMeshOutputCounts must precede every output in uniform control flow"
                    )
                    return False
        output.count_statement = counts[0]
        stages_indices = output.index_width > 1 and any(
            call.method == "set_index" for call in calls
        )
        if stages_indices and any(isinstance(node, ReturnNode) for node in _walk(body)):
            self.refuse_import(
                f"Metal mesh function '{name}' returns early, but its staged set_index writes are copied to the "
                "index array after a barrier that every invocation must reach"
            )
            return False
        if stages_indices:
            output.staging = "pending"
        return True

    def reserve_metal_mesh_names(self, output):
        base = output.parameter.name
        output.vertices = self.reserve_generated_identifier(f"{base}_vertices")
        if output.primitive_type is not None:
            output.primitives = self.reserve_generated_identifier(f"{base}_primitives")
        output.indices = self.reserve_generated_identifier(f"{base}_indices")
        output.count = self.reserve_generated_identifier(f"{base}_primitive_count")
        if output.staging:
            output.staging = self.reserve_generated_identifier(f"{base}_index_values")
            for parameter in output.function.params:
                if any(
                    getattr(attribute, "name", None) == "thread_index_in_threadgroup"
                    for attribute in getattr(parameter, "attributes", []) or []
                ):
                    output.invocation = self.render_identifier(parameter.name)
                    break
            else:
                output.invocation = self.reserve_generated_identifier(
                    f"{base}_invocation"
                )
                output.added_invocation = True

    def generate_metal_mesh_stage(self, function, **options):
        """A ``mesh { ... }`` stage block for one ``[[mesh]]`` function, or ``""`` when it is refused."""
        output = self.plan_metal_mesh_function(function)
        if output is None:
            return ""
        self.reserve_metal_mesh_names(output)
        previous = self.current_metal_mesh_output
        self.current_metal_mesh_output = output
        try:
            text = self.generate_function(
                function, stage_entry=function.name != "main", **options
            )
        finally:
            self.current_metal_mesh_output = previous
        body_start = text.find("{\n")
        body = text[body_start:] if body_start >= 0 else text
        if re.search(rf"\b{re.escape(output.parameter.name)}\b", body):
            self.refuse_import(
                f"Metal mesh function '{function.name}' uses its metal::mesh parameter "
                f"'{output.parameter.name}' other than in statement-level set_* calls"
            )
            return ""
        return (
            "    // Mesh Shader\n"
            "    mesh {\n" + self.kernel_workgroup_size_attribute() + text + "    }\n\n"
        )

    def metal_mesh_parameter_declaration(self, var):
        """The CrossGL declarations replacing a mesh function's ``metal::mesh`` or ``[[payload]]`` parameter."""
        output = self.current_metal_mesh_output
        if output is None or self.current_function is not output.function:
            return None
        if var is output.parameter:
            declarations = [
                f"out {self.map_type(output.vertex_type)} "
                f"{output.vertices}[{output.max_vertices}] @vertices"
            ]
            if output.primitive_type is not None:
                declarations.append(
                    f"out {self.map_type(output.primitive_type)} "
                    f"{output.primitives}[{output.max_primitives}] @primitives"
                )
            declarations.append(
                f"out {_INDEX_TYPES[output.index_width]} "
                f"{output.indices}[{output.max_primitives}] @indices"
            )
            if output.added_invocation:
                declarations.append(
                    f"uint {output.invocation} @gl_LocalInvocationIndex"
                )
            return ", ".join(declarations)
        if id(var) in output.payload_parameters:
            raw_type = str(getattr(var, "vtype", "")).strip().rstrip("&").strip()
            return f"in {self.map_type(raw_type)} {self.render_identifier(var.name)} @mesh_payload"
        return None

    def metal_mesh_function_annotation(self, func):
        output = self.current_metal_mesh_output
        if output is None or output.function is not func:
            return ""
        return f"@outputtopology({output.topology})"

    def metal_mesh_body_prologue(self, func, indent):
        output = self.current_metal_mesh_output
        if output is None or output.function is not func or not output.staging:
            return ""
        size = output.max_primitives * output.index_width
        return f"{'    ' * indent}groupshared uint {output.staging}[{size}];\n"

    def metal_mesh_body_epilogue(self, func, indent):
        output = self.current_metal_mesh_output
        if output is None or output.function is not func or not output.staging:
            return ""
        pad = "    " * indent
        width = output.index_width
        primitive = self.reserve_generated_identifier("primitive")
        components = ", ".join(
            f"{output.staging}[{width}u * {primitive} + {offset}u]"
            for offset in range(width)
        )
        return (
            f"{pad}workgroupBarrier();\n"
            f"{pad}if ({output.invocation} == 0u) {{\n"
            f"{pad}    for (uint {primitive} = 0u; {primitive} < {output.count}; {primitive}++) {{\n"
            f"{pad}        {output.indices}[{primitive}] = {_INDEX_TYPES[width]}({components});\n"
            f"{pad}    }}\n"
            f"{pad}}}\n"
        )

    def generate_metal_mesh_statement(self, stmt, indent, is_main=False):
        """The CrossGL statements for one statement-level ``metal::mesh`` method call, or ``None``."""
        if not isinstance(stmt, MethodCallNode) or not self.is_metal_mesh_object(
            stmt.object
        ):
            return None
        output = self.current_metal_mesh_output
        method = stmt.method
        expected = 1 if method == "set_primitive_count" else 2
        if len(stmt.args) != expected:
            self.refuse_import(
                f"metal::mesh::{method} takes {expected} arguments, got {len(stmt.args)}"
            )
            return None
        args = [self.generate_expression(arg, is_main) for arg in stmt.args]
        pad = "    " * indent
        if method == "set_primitive_count":
            return (
                f"uint {output.count} = uint({args[0]});\n"
                f"{pad}SetMeshOutputCounts({output.max_vertices}u, {output.count});\n"
            )
        if method == "set_vertex":
            return f"{output.vertices}[{args[0]}] = {args[1]};\n"
        if method == "set_primitive":
            if output.primitive_type is None:
                self.refuse_import(
                    "metal::mesh::set_primitive on a mesh whose primitive type is void"
                )
                return None
            return f"{output.primitives}[{args[0]}] = {args[1]};\n"
        if method == "set_index":
            # set_index takes a uchar, so the stored index is the value modulo 256.
            if output.staging:
                return f"{output.staging}[{args[0]}] = uint({args[1]}) & 255u;\n"
            return f"{output.indices}[{args[0]}] = uint({args[1]}) & 255u;\n"
        return None
