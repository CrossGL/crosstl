"""Vertex and fragment entry points for rust-gpu: stage inputs and outputs as entry-point parameters.

rust-gpu (0.11) declares the interface of a vertex or fragment shader as the parameters of its entry point
(``#[spirv(vertex)]``, ``#[spirv(fragment)]``): an input is a parameter taken by value and an output a ``&mut``
parameter, each either a user varying with ``#[spirv(location = N)]`` or a built-in with its attribute
(``#[spirv(position)]``, ``#[spirv(frag_coord)]``; rustc_codegen_spirv/src/symbols.rs, ``BUILTINS``). A CrossGL entry
point takes its inputs as a structure or as values with semantics, and returns its outputs as a structure or as one
value with a semantic. The generator writes:

- one parameter for each input value, a structure input flattened member by member in declaration order, and at the
  start of the body a local of the CrossGL structure built from those parameters, so the body reads ``input.uv`` as
  the source does;
- one ``&mut`` parameter for each output value, and in place of every ``return value;`` a write of each member of
  the value through its parameter followed by ``return;``.

Locations follow :mod:`.stage_io_semantic_locations`, as on every other target, so vertex outputs and fragment inputs
written for different targets still match; a fragment output's ``SV_TargetN``, ``gl_FragColorN`` or ``COLORn`` names
colour attachment ``N``. Integer and double-precision fragment inputs, and integer vertex outputs, carry
``#[spirv(flat)]``, which Vulkan requires of the fragment inputs (VUID-StandaloneSpirv-Flat-04744). ``gl_InstanceID``
and ``SV_InstanceID`` exclude the base instance, so they are ``instance_index - base_instance``. A fragment entry
that writes ``gl_FragDepth`` declares ``depth_replacing`` (and the conservative-depth mode its attributes name), one
that writes the stencil reference ``stencil_ref_replacing_ext``, and one that asks for early fragment tests
``early_fragment_tests``.

Refused, with the value named: interpolation modes rust-gpu has no attribute for (``noperspective``, ``centroid``,
``sample``), ``bool`` and 8- or 16-bit varyings, structure members of an interface structure, built-ins this
table does not name or that the stage cannot read or write, and ``out`` parameters of an entry point other than the
fragment colour output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..ast import ArrayNode
from .rust_gpu_interface import COMPUTE_SEMANTIC_BUILTINS
from .stage_io_semantic_locations import allocate_locations, semantic_location
from .stage_utils import requests_early_fragment_tests

__all__ = ["GRAPHICS_STAGES", "RustGpuStageIoMixin"]

GRAPHICS_STAGES = frozenset({"vertex", "fragment"})

_IN, _OUT = "in", "out"
_INDEX = ("int", "uint")


def _entry(attribute, types, derived=None):
    return {"attribute": attribute, "types": types, "derived": derived}


# The rust-gpu built-in of each CrossGL built-in value, keyed by ``(stage, direction)`` and the lower-cased name. Each
# entry lists the CrossGL types the value may be declared with; ``derived`` names a second built-in the value is
# computed from.
_BUILTINS = {
    ("vertex", _IN): {
        "gl_vertexid": _entry("vertex_index", _INDEX),
        "gl_vertexindex": _entry("vertex_index", _INDEX),
        "sv_vertexid": _entry("vertex_index", _INDEX),
        "gl_relativevertexid": _entry("vertex_index", _INDEX, "base_vertex"),
        "gl_instanceindex": _entry("instance_index", _INDEX),
        "gl_instanceid": _entry("instance_index", _INDEX, "base_instance"),
        "sv_instanceid": _entry("instance_index", _INDEX, "base_instance"),
        "gl_basevertex": _entry("base_vertex", _INDEX),
        "sv_startvertexlocation": _entry("base_vertex", _INDEX),
        "gl_baseinstance": _entry("base_instance", _INDEX),
        "sv_startinstancelocation": _entry("base_instance", _INDEX),
        "gl_drawid": _entry("draw_index", _INDEX),
        "gl_viewindex": _entry("view_index", _INDEX),
        "sv_viewid": _entry("view_index", _INDEX),
    },
    ("vertex", _OUT): {
        "gl_position": _entry("position", ("vec4",)),
        "sv_position": _entry("position", ("vec4",)),
        "gl_pointsize": _entry("point_size", ("float",)),
        "gl_clipdistance": _entry("clip_distance", ("float",)),
        "sv_clipdistance": _entry("clip_distance", ("float",)),
        "gl_culldistance": _entry("cull_distance", ("float",)),
        "sv_culldistance": _entry("cull_distance", ("float",)),
    },
    ("fragment", _IN): {
        "gl_fragcoord": _entry("frag_coord", ("vec4",)),
        "gl_position": _entry("frag_coord", ("vec4",)),
        "sv_position": _entry("frag_coord", ("vec4",)),
        "gl_frontfacing": _entry("front_facing", ("bool",)),
        "sv_isfrontface": _entry("front_facing", ("bool",)),
        "gl_pointcoord": _entry("point_coord", ("vec2",)),
        "gl_primitiveid": _entry("primitive_id", _INDEX),
        "sv_primitiveid": _entry("primitive_id", _INDEX),
        "gl_sampleid": _entry("sample_id", _INDEX),
        "sv_sampleindex": _entry("sample_id", _INDEX),
        "gl_sampleposition": _entry("sample_position", ("vec2",)),
        "gl_helperinvocation": _entry("helper_invocation", ("bool",)),
        "gl_layer": _entry("layer", _INDEX),
        "sv_rendertargetarrayindex": _entry("layer", _INDEX),
        "gl_viewportindex": _entry("viewport_index", _INDEX),
        "sv_viewportarrayindex": _entry("viewport_index", _INDEX),
        "gl_viewindex": _entry("view_index", _INDEX),
        "sv_viewid": _entry("view_index", _INDEX),
        "gl_barycoordext": _entry("bary_coord", ("vec3",)),
        "sv_barycentrics": _entry("bary_coord", ("vec3",)),
        "gl_barycoordnoperspext": _entry("bary_coord_no_persp", ("vec3",)),
    },
    ("fragment", _OUT): {
        "gl_fragdepth": _entry("frag_depth", ("float",)),
        "sv_depth": _entry("frag_depth", ("float",)),
        "sv_depthgreaterequal": _entry("frag_depth", ("float",)),
        "sv_depthlessequal": _entry("frag_depth", ("float",)),
        "gl_fragstencilrefext": _entry("frag_stencil_ref_ext", _INDEX),
        "sv_stencilref": _entry("frag_stencil_ref_ext", _INDEX),
    },
}
# Conservative depth: the attribute or semantic of a depth output, as the fragment execution mode it asks for.
_DEPTH_MODES = {
    "depth_greater": "depth_greater",
    "depth_less": "depth_less",
    "depth_unchanged": "depth_unchanged",
    "sv_depthgreaterequal": "depth_greater",
    "sv_depthlessequal": "depth_less",
}
_DISTANCE_BUILTINS = frozenset({"clip_distance", "cull_distance"})
_RESERVED_PREFIXES = ("gl_", "sv_")
_FRAGMENT_TARGET = re.compile(r"(?:sv_target|gl_fragcolor|out_color)(\d*)")
_FRAGMENT_COLOR = re.compile(r"color(\d*)")
_LOCATION_SEMANTIC = re.compile(r"location\s*\(\s*(\d+)\s*\)")
_FLAT = frozenset({"flat", "nointerpolation"})
_DEFAULT_INTERPOLATION = frozenset({"smooth", "linear", "perspective"})
_UNSUPPORTED_INTERPOLATION = frozenset({"noperspective", "centroid", "sample"})
_VARYING_SCALARS = frozenset({"f32", "f64", "i32", "u32"})
# Attributes of an interface value that qualify it rather than name what it is.
_QUALIFIER_ATTRIBUTES = (
    _FLAT
    | _DEFAULT_INTERPOLATION
    | _UNSUPPORTED_INTERPOLATION
    | {"invariant", "location", "component", "precise", "depth_any"}
    | {"depth_greater", "depth_less", "depth_unchanged"}
)


@dataclass
class _Value:
    """One interface value: a parameter of the entry point."""

    name: str
    type_name: str
    semantic: object
    node: object
    builtin: dict | None = None
    location: int | None = None
    path: str = ""  # ``.member`` of the CrossGL value, or empty for the whole value.
    derived_name: str | None = None  # the parameter of the second built-in a derived value subtracts


@dataclass
class StageInterface:
    """The rust-gpu parameters of a vertex or fragment entry point and the statements around its body."""

    params: list = field(default_factory=list)
    prologue: list = field(default_factory=list)
    outputs: list = field(default_factory=list)  # ``(write statement template, ...)``
    modes: list = field(default_factory=list)


def _attribute_names(node):
    return {
        str(getattr(attr, "name", attr)).lower()
        for attr in getattr(node, "attributes", None) or []
    }


def _key(semantic):
    return re.sub(r"[^a-z0-9_]+", "", str(semantic).lower()) if semantic else ""


class RustGpuStageIoMixin:
    """Vertex and fragment entry-point interfaces of :class:`RustCodeGen`."""

    #: The output writes of the entry point being emitted: ``(reference name, member path, kind)``.
    current_rust_stage_outputs: tuple | list = ()

    def rust_gpu_derivatives_allowed(self):
        """Whether the function being emitted runs in fragment shaders only."""
        if self.current_shader_type is not None:
            return self.current_shader_type == "fragment"
        return getattr(self, "rust_gpu_program_stages", frozenset()) == {"fragment"}

    def collect_rust_gpu_struct_nodes(self, ast):
        structs = {}
        for struct in getattr(ast, "structs", None) or []:
            structs[getattr(struct, "name", None)] = struct
        for stage in (getattr(ast, "stages", None) or {}).values():
            for struct in getattr(stage, "local_structs", None) or []:
                structs[getattr(struct, "name", None)] = struct
        self.rust_gpu_struct_nodes = structs
        self.rust_program_type_names = frozenset(
            getattr(node, "name", None)
            for attribute in ("structs", "enums", "traits", "impls")
            for node in getattr(ast, attribute, None) or []
        ) | frozenset(structs)
        from .rust_gpu_subset import program_stage_names

        self.rust_gpu_program_stages = frozenset(program_stage_names(ast))

    def rust_gpu_stage_attribute(self, shader_type, interface):
        modes = ", ".join(interface.modes)
        return f"#[spirv({shader_type}({modes}))]\n" if modes else f"#[spirv({shader_type})]\n"

    # -- planning -------------------------------------------------------------------------------------------

    def plan_rust_gpu_compute_interface(self, func, param_list, return_type):
        """The built-in input parameters of compute entry point ``func``.

        A compute shader has no user inputs or outputs (Vulkan allows only built-ins in its Input storage class), so
        a parameter that is not a compute built-in, and a returned value, are refused. A parameter declared as a
        part or another integer type of its built-in (``uint id @ SV_DispatchThreadID``) is converted once at the
        start of the body.
        """
        interface = StageInterface()
        used = self.rust_stage_io_used_names(func, param_list)
        if self.map_type(return_type) != "()":
            self.refuse_lowering(
                f"stage: compute entry point '{func.name}' returns a {return_type}; compute shaders write their "
                "results to buffers"
            )
        for param in param_list:
            semantic = self.rust_stage_io_semantic(param)
            key = self.rust_stage_parameter_semantic_key(semantic) if semantic else None
            builtin = COMPUTE_SEMANTIC_BUILTINS.get(key)
            if builtin is None:
                self.refuse_lowering(
                    f"stage: compute entry parameter '{param.name}' is not a compute built-in input"
                )
                continue
            attribute, builtin_type = builtin
            type_name = self.function_parameter_type(param)
            name = self.rust_identifier(param.name)
            native = self.rust_stage_io_unique(f"{name}_{attribute}", used)
            value = self.rust_compute_builtin_value(native, builtin_type, type_name)
            if value is None:
                self.refuse_lowering(
                    f"stage: compute entry parameter '{param.name}' is declared {type_name}; the built-in "
                    f"{semantic} is a {builtin_type}"
                )
                continue
            interface.params.append(f"#[spirv({attribute})] {native}: {self.map_type(builtin_type)}")
            interface.prologue.append(
                f"{self.rust_stage_io_let(param.name)} {name}: {self.map_type(type_name)} = {value};"
            )
        return interface

    def rust_compute_builtin_value(self, native, builtin_type, type_name):
        """``native`` (a ``uint`` or ``uvec3`` built-in) as CrossGL type ``type_name``, or ``None``."""
        target = self.map_type(type_name)
        if target == self.map_type(builtin_type):
            return native
        if builtin_type == "uint":
            return f"({native} as i32)" if target == "i32" else None
        conversions = {
            "u32": f"{native}.x",
            "i32": f"({native}.x as i32)",
            "Vec2<u32>": f"{native}.truncate()",
            "Vec2<i32>": f"{native}.truncate().as_ivec2()",
            "Vec3<i32>": f"{native}.as_ivec3()",
        }
        return conversions.get(target)

    def plan_rust_gpu_stage_interface(
        self, func, shader_type, param_list, return_type, return_semantic, stage_node
    ):
        """The :class:`StageInterface` of entry point ``func``; every value it cannot lower is refused."""
        interface = StageInterface()
        used = self.rust_stage_io_used_names(func, param_list)
        inputs = []
        for param in param_list:
            qualifiers = {str(q).lower() for q in getattr(param, "qualifiers", None) or []}
            if qualifiers & {"out", "inout"}:
                self.refuse_lowering(
                    f"stage: {shader_type} entry parameter '{param.name}' is an output parameter; rust-gpu output "
                    "is written for returned values only"
                )
                continue
            inputs.append(self.rust_stage_io_param_values(param, shader_type, used, interface))
        flat_inputs = [value for values in inputs for value in values]
        self.assign_rust_stage_io_locations(flat_inputs, shader_type, _IN)
        for value in flat_inputs:
            interface.params.append(self.rust_stage_io_parameter(value, shader_type, _IN, used))
        outputs = self.rust_stage_io_return_values(
            func, shader_type, return_type, return_semantic, used
        )
        self.assign_rust_stage_io_locations(outputs, shader_type, _OUT)
        for value in outputs:
            interface.params.append(self.rust_stage_io_parameter(value, shader_type, _OUT, used))
            interface.outputs.append((value.name, value.path, value.builtin, value.type_name))
        if shader_type == "fragment":
            interface.modes = self.rust_stage_io_fragment_modes(func, outputs, stage_node)
        return interface

    def rust_stage_io_used_names(self, func, param_list):
        from .rust_resource_bindings import referenced_identifier_names

        used = set(referenced_identifier_names(getattr(func, "body", None)))
        used.update(getattr(param, "name", "") for param in param_list)
        used.update(self.user_function_names)
        used.update(self.variable_types)
        return used

    def rust_stage_io_unique(self, candidate, used):
        name = self.rust_unique_local_identifier(candidate, used)
        used.add(name)
        return name

    def rust_stage_io_members(self, type_name):
        """``(name, CrossGL type, node)`` of each member of interface structure ``type_name``, else ``None``."""
        struct = getattr(self, "rust_gpu_struct_nodes", {}).get(str(type_name))
        if struct is None:
            return None
        members = []
        for member in getattr(struct, "members", None) or []:
            if isinstance(member, ArrayNode):
                element = self.convert_type_node_to_string(
                    getattr(member, "element_type", getattr(member, "vtype", "float"))
                )
                size = self.format_array_size(getattr(member, "size", None))
                members.append((member.name, f"{element}[{size}]" if size else f"{element}[]", member))
                continue
            member_type = getattr(member, "member_type", getattr(member, "vtype", None))
            if member_type is None:
                continue
            members.append((member.name, self.convert_type_node_to_string(member_type), member))
        return members

    def rust_stage_io_param_values(self, param, shader_type, used, interface):
        """The interface values of entry parameter ``param``, and the prologue that rebuilds a structure."""
        type_name = self.function_parameter_type(param)
        name = self.rust_identifier(param.name)
        members = self.rust_stage_io_members(type_name)
        if members is None:
            value = _Value(name, type_name, self.rust_stage_io_semantic(param), param)
            self.classify_rust_stage_io_input(value, shader_type, used)
            if value.builtin is not None or name in self.current_mutated_names:
                # The parameter takes a fresh name; a local of the source name holds the CrossGL value.
                value.name = self.rust_stage_io_unique(f"{name}_in", used)
                interface.prologue.append(
                    f"{self.rust_stage_io_let(param.name)} {name}: {self.map_type(type_name)} = "
                    f"{self.rust_stage_io_input_value(value)};"
                )
            return [value]
        values = []
        fields = []
        for member_name, member_type, member_node in members:
            if self.rust_stage_io_members(member_type) is not None:
                self.refuse_lowering(
                    f"stage: {shader_type} input '{param.name}.{member_name}' is a structure; rust-gpu interface "
                    "values are scalars, vectors and arrays"
                )
                continue
            value = _Value(
                self.rust_stage_io_unique(f"{name}_{member_name}", used),
                member_type,
                self.rust_stage_io_semantic(member_node),
                member_node,
            )
            self.classify_rust_stage_io_input(value, shader_type, used)
            values.append(value)
            fields.append(f"{self.rust_identifier(member_name)}: {self.rust_stage_io_input_value(value)}")
        struct_name = self.rust_type_identifier(str(type_name))
        interface.prologue.append(
            f"{self.rust_stage_io_let(param.name)} {name}: {struct_name} = {struct_name} {{ "
            f"{', '.join(fields)} }};"
        )
        return values

    def classify_rust_stage_io_input(self, value, shader_type, used):
        self.classify_rust_stage_io_value(value, shader_type, _IN)
        if value.builtin is not None and value.builtin["derived"]:
            value.derived_name = self.rust_stage_io_unique(f"{value.name}_base", used)

    def rust_stage_io_semantic(self, node):
        """The semantic of an interface value: its ``semantic``, else its first attribute that is not a qualifier."""
        semantic = getattr(node, "semantic", None)
        if semantic:
            return semantic
        for attr in getattr(node, "attributes", None) or []:
            name = getattr(attr, "name", attr)
            if str(name).lower() not in _QUALIFIER_ATTRIBUTES:
                return name
        return None

    def rust_stage_io_let(self, source_name):
        return "let mut" if source_name in self.current_mutated_names else "let"

    def rust_stage_io_return_values(self, func, shader_type, return_type, semantic, used):
        if self.map_type(return_type) == "()":
            return []
        members = self.rust_stage_io_members(return_type)
        if members is None:
            if semantic is None and shader_type == "vertex":
                self.refuse_lowering(
                    f"stage: the vertex entry point '{func.name}' returns a {return_type} without a semantic"
                )
                return []
            value = _Value(self.rust_stage_io_unique("out_value", used), return_type, semantic, func)
            self.classify_rust_stage_io_value(value, shader_type, _OUT)
            return [value]
        values = []
        for member_name, member_type, member_node in members:
            if self.rust_stage_io_members(member_type) is not None:
                self.refuse_lowering(
                    f"stage: {shader_type} output '{return_type}.{member_name}' is a structure; rust-gpu interface "
                    "values are scalars, vectors and arrays"
                )
                continue
            value = _Value(
                self.rust_stage_io_unique(f"out_{member_name}", used),
                member_type,
                self.rust_stage_io_semantic(member_node),
                member_node,
                path=f".{self.rust_identifier(member_name)}",
            )
            self.classify_rust_stage_io_value(value, shader_type, _OUT)
            values.append(value)
        return values

    # -- classification -------------------------------------------------------------------------------------

    def classify_rust_stage_io_value(self, value, shader_type, direction):
        """Set ``value.builtin``, or check that it is a user varying rust-gpu can declare."""
        key = _key(value.semantic)
        builtin = _BUILTINS[(shader_type, direction)].get(key)
        if builtin is None and key.startswith(_RESERVED_PREFIXES) and not self.is_rust_stage_io_user_semantic(
            key, shader_type, direction
        ):
            self.refuse_lowering(
                f"stage: built-in {value.semantic} is not a rust-gpu {shader_type} "
                f"{'input' if direction == _IN else 'output'}"
            )
            return
        if builtin is not None:
            base, _size = self.rust_stage_io_array_parts(value.type_name)
            if base not in builtin["types"] or (
                _size is not None and builtin["attribute"] not in _DISTANCE_BUILTINS
            ):
                self.refuse_lowering(
                    f"stage: built-in {value.semantic} is declared {value.type_name}; rust-gpu delivers it as "
                    f"{' or '.join(builtin['types'])}"
                )
                return
            value.builtin = builtin
            return
        self.check_rust_stage_io_varying(value, shader_type, direction)

    def is_rust_stage_io_user_semantic(self, key, shader_type, direction):
        """Whether a reserved-looking semantic names a fragment colour attachment, a user varying."""
        return shader_type == "fragment" and direction == _OUT and _FRAGMENT_TARGET.fullmatch(key) is not None

    def rust_stage_io_array_parts(self, type_name):
        text = str(type_name)
        match = re.fullmatch(r"\s*([^\[\]]+?)\s*\[\s*(\d*)\s*\]\s*", text)
        if match is None:
            return text.strip(), None
        return match.group(1), int(match.group(2)) if match.group(2) else 0

    def rust_stage_io_scalar(self, type_name):
        """``(Rust scalar, vector size or 0, location span)`` of a varying type, or ``None`` for another type."""
        matrix = self.matrix_type_info(type_name)
        if matrix is not None:
            columns, rows = matrix["columns"], matrix["rows"]
            scalar = "f64" if matrix["component_type"] == "double" else "f32"
            # A matrix takes one location per column (Vulkan 1.3 section 15.1.4, "Location Assignment").
            return scalar, 0, columns * (2 if scalar == "f64" and rows > 2 else 1)
        info = self.vector_type_info(type_name)
        if info is not None:
            scalar = self.normalize_scalar_type(info["component_type"])
            size = info["size"]
            return scalar, size, 2 if scalar == "f64" and size > 2 else 1
        scalar = self.normalize_scalar_type(type_name)
        if scalar is None or self.map_type(type_name) != scalar:
            return None
        return scalar, 0, 1

    def check_rust_stage_io_varying(self, value, shader_type, direction):
        base, size = self.rust_stage_io_array_parts(value.type_name)
        shape = self.rust_stage_io_scalar(base)
        what = f"{shader_type} {'input' if direction == _IN else 'output'} '{value.name}'"
        if size == 0:
            self.refuse_lowering(f"stage: {what} is a runtime-sized array; rust-gpu interface arrays have a size")
        elif shape is None or shape[0] not in _VARYING_SCALARS:
            self.refuse_lowering(
                f"stage: {what} has type {value.type_name}; rust-gpu user varyings are 32- or 64-bit scalars "
                "and vectors, or arrays of them"
            )
        attributes = _attribute_names(value.node)
        unsupported = sorted(attributes & _UNSUPPORTED_INTERPOLATION)
        if unsupported:
            self.refuse_lowering(
                f"stage: {what} asks for {unsupported[0]} interpolation, which rust-gpu has no attribute for"
            )

    def assign_rust_stage_io_locations(self, values, shader_type, direction):
        varyings = [value for value in values if value.builtin is None]
        entries = []
        for value in varyings:
            base, size = self.rust_stage_io_array_parts(value.type_name)
            shape = self.rust_stage_io_scalar(base)
            span = (shape[2] if shape else 1) * max(size or 1, 1)
            entries.append((self.rust_stage_io_fixed_location(value, shader_type, direction), span))
        for value, location in zip(varyings, allocate_locations(entries)):
            value.location = location

    def rust_stage_io_fixed_location(self, value, shader_type, direction):
        for attr in getattr(value.node, "attributes", None) or []:
            if str(getattr(attr, "name", "")).lower() != "location":
                continue
            arguments = getattr(attr, "arguments", None) or getattr(attr, "args", None) or []
            if arguments:
                location = self.literal_int_value(arguments[0])
                if location is None:
                    try:
                        location = int(str(getattr(arguments[0], "value", arguments[0])), 0)
                    except ValueError:
                        location = None
                if location is not None:
                    return location
        key = _key(value.semantic)
        explicit = _LOCATION_SEMANTIC.fullmatch(str(value.semantic or "").lower())
        if explicit is not None:
            return int(explicit.group(1))
        if shader_type == "fragment" and direction == _OUT:
            target = _FRAGMENT_TARGET.fullmatch(key) or _FRAGMENT_COLOR.fullmatch(key)
            if target is not None:
                return int(target.group(1) or 0)
        return semantic_location(value.semantic)

    # -- rendering ------------------------------------------------------------------------------------------

    def rust_stage_io_rust_type(self, type_name):
        base, size = self.rust_stage_io_array_parts(type_name)
        if size:
            return f"[{self.map_type(base)}; {size}]"
        return self.map_type(type_name)

    def rust_stage_io_parameter(self, value, shader_type, direction, used):
        attributes = []
        rust_type = self.rust_stage_io_rust_type(value.type_name)
        derived_param = ""
        if value.builtin is not None:
            attributes.append(value.builtin["attribute"])
            if value.builtin["attribute"] in _DISTANCE_BUILTINS:
                _base, size = self.rust_stage_io_array_parts(value.type_name)
                rust_type = f"[f32; {size or 1}]"
            if value.derived_name:
                derived_param = (
                    f", #[spirv({value.builtin['derived']})] {value.derived_name}: {rust_type}"
                )
            if "invariant" in _attribute_names(value.node) and direction == _OUT:
                attributes.append("invariant")
            if shader_type == "fragment" and direction == _IN and value.builtin["types"] == _INDEX:
                # VUID-StandaloneSpirv-Flat-04744 covers built-in integer inputs as well.
                attributes.append("flat")
        else:
            attributes.append(f"location = {value.location}")
            if self.rust_stage_io_needs_flat(value, shader_type, direction):
                attributes.append("flat")
        prefix = f"#[spirv({', '.join(attributes)})] "
        reference = "&mut " if direction == _OUT else ""
        return f"{prefix}{value.name}: {reference}{rust_type}{derived_param}"

    def rust_stage_io_needs_flat(self, value, shader_type, direction):
        base, _size = self.rust_stage_io_array_parts(value.type_name)
        shape = self.rust_stage_io_scalar(base)
        integral = shape is not None and shape[0] in {"i32", "u32"}
        double = shape is not None and shape[0] == "f64"
        if shader_type == "fragment" and direction == _IN:
            return integral or double or bool(_attribute_names(value.node) & _FLAT)
        if shader_type == "vertex" and direction == _OUT:
            return integral or bool(_attribute_names(value.node) & _FLAT)
        return False

    def rust_stage_io_input_value(self, value):
        """The expression that reads ``value`` as its CrossGL type."""
        if value.derived_name:
            return f"{value.name}.wrapping_sub({value.derived_name})"
        return value.name

    def rust_stage_io_fragment_modes(self, func, outputs, stage_node):
        modes = []
        if requests_early_fragment_tests(func, stage_node):
            modes.append("early_fragment_tests")
        for value in outputs:
            attribute = value.builtin["attribute"] if value.builtin else None
            if attribute == "frag_depth":
                modes.append("depth_replacing")
                names = _attribute_names(value.node) | {_key(value.semantic)}
                modes.extend(sorted({_DEPTH_MODES[name] for name in names if name in _DEPTH_MODES}))
            elif attribute == "frag_stencil_ref_ext":
                modes.append("stencil_ref_replacing_ext")
        return modes

    # -- returns --------------------------------------------------------------------------------------------

    def generate_rust_stage_output_return(self, indent_str, return_expr):
        """``return value;`` of a vertex or fragment entry point: each output written through its parameter."""
        if return_expr is None:
            return f"{indent_str}return;\n"
        temporary = self.rust_output_return_temporary()
        lines = [f"{indent_str}let {temporary} = {return_expr};\n"]
        for name, path, builtin, _type_name in self.current_rust_stage_outputs:
            source = f"{temporary}{path}"
            _base, size = self.rust_stage_io_array_parts(_type_name)
            if builtin is not None and builtin["attribute"] in _DISTANCE_BUILTINS and size is None:
                lines.append(f"{indent_str}*{name} = [{source}];\n")
            else:
                lines.append(f"{indent_str}*{name} = {source};\n")
        lines.append(f"{indent_str}return;\n")
        return "".join(lines)
