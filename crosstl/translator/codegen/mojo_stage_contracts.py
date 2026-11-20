"""Stage contracts of the Mojo generator: geometry, tessellation and mesh attributes, parameter roles, their validation, and stage-local declarations."""

from ..ast import (
    ArrayNode,
    IdentifierNode,
    MeshOpNode,
    StructNode,
)
from ..refusals import MojoRefusal
from .mojo_tables import MOJO_SUPPORTED_STAGE_TYPES


class MojoStageContractMixin:
    """Stage-contract members of ``MojoCodeGen``."""

    def function_stage_qualifier(self, func):
        if hasattr(func, "qualifiers") and func.qualifiers:
            qualifier = func.qualifiers[0] if func.qualifiers else None
        else:
            qualifier = getattr(func, "qualifier", None)
        if qualifier:
            value = getattr(qualifier, "value", None)
            return value if isinstance(value, str) else str(qualifier)

        for attr in getattr(func, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name is None:
                continue
            normalized = str(attr_name).strip().lower().replace("-", "_")
            if normalized in MOJO_SUPPORTED_STAGE_TYPES:
                return normalized
        return None

    def stage_type_name(self, stage_type):
        value = getattr(stage_type, "value", None)
        if isinstance(value, str):
            return value
        return str(stage_type).split(".")[-1].lower()

    def validate_stage_type(self, stage_name):
        if stage_name not in MOJO_SUPPORTED_STAGE_TYPES:
            supported = ", ".join(sorted(MOJO_SUPPORTED_STAGE_TYPES))
            raise MojoRefusal(
                f"Unsupported {stage_name} shader stage for Mojo codegen; "
                f"supported compile-smoke stages are {supported}"
            )

    def geometry_stage_attribute_arguments(self, func, attribute_name):
        requested = str(attribute_name).strip().lower().replace("-", "_")
        for attr in getattr(func, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name is None:
                continue
            normalized = str(attr_name).strip().lower().replace("-", "_")
            if normalized == requested:
                return getattr(attr, "arguments", getattr(attr, "args", [])) or []
        return []

    def mojo_stage_attribute_value(self, func, attribute_name, stage_name):
        arguments = self.geometry_stage_attribute_arguments(func, attribute_name)
        if not arguments:
            return None
        if len(arguments) != 1:
            raise MojoRefusal(
                f"Mojo {stage_name} stage {attribute_name} requires exactly one "
                "argument"
            )
        return self.attribute_value_to_string(arguments[0])

    def geometry_maxvertexcount(self, func):
        arguments = self.geometry_stage_attribute_arguments(func, "maxvertexcount")
        if not arguments:
            raise MojoRefusal("Mojo geometry stage requires maxvertexcount attribute")
        if len(arguments) != 1:
            raise MojoRefusal(
                "Mojo geometry stage maxvertexcount requires exactly one argument"
            )
        value_text = self.attribute_value_to_string(arguments[0])
        value = self.literal_int_value(arguments[0])
        if value is not None and value <= 0:
            raise MojoRefusal(
                f"Mojo geometry stage maxvertexcount ({value}) must be positive"
            )
        return value_text

    def geometry_parameter_qualifiers(self, parameter):
        qualifiers = []
        allowed_qualifiers = {
            "const",
            "in",
            "out",
            "inout",
            "point",
            "line",
            "triangle",
            "lineadj",
            "triangleadj",
        }
        for qualifier in getattr(parameter, "qualifiers", []) or []:
            normalized = str(qualifier).lower()
            if normalized in allowed_qualifiers:
                qualifiers.append(normalized)

        for attr in getattr(parameter, "attributes", []) or []:
            if getattr(attr, "name", None) != "primitive":
                continue
            arguments = getattr(attr, "arguments", getattr(attr, "args", [])) or []
            if not arguments:
                continue
            primitive = self.attribute_value_to_string(arguments[0])
            normalized = str(primitive).lower()
            if normalized in allowed_qualifiers:
                qualifiers.append(normalized)
        return qualifiers

    def geometry_input_primitive_qualifier(self, parameter):
        primitive_qualifiers = {
            "point",
            "line",
            "triangle",
            "lineadj",
            "triangleadj",
        }
        for qualifier in self.geometry_parameter_qualifiers(parameter):
            if qualifier in primitive_qualifiers:
                return qualifier
        return None

    def geometry_parameter_is_array(self, parameter, param_type):
        raw_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        if (
            hasattr(raw_type, "element_type")
            and str(type(raw_type)).find("ArrayType") != -1
        ):
            return True
        return bool(param_type and "[" in str(param_type) and "]" in str(param_type))

    def geometry_parameter_array_count(self, parameter, param_type):
        raw_type = getattr(parameter, "param_type", getattr(parameter, "vtype", None))
        if (
            hasattr(raw_type, "element_type")
            and str(type(raw_type)).find("ArrayType") != -1
        ):
            size = getattr(raw_type, "size", None)
            if size is None:
                return None
            if isinstance(size, int):
                return size
            return self.literal_int_value(size)

        if not param_type or "[" not in str(param_type) or "]" not in str(param_type):
            return None
        _base_type, array_size = self.parse_array_type_name(param_type)
        return array_size

    def validate_geometry_input_primitive_arity(self, param_infos):
        expected_counts = {
            "point": 1,
            "line": 2,
            "triangle": 3,
            "lineadj": 4,
            "triangleadj": 6,
        }

        for parameter, param_type in param_infos:
            primitive = self.geometry_input_primitive_qualifier(parameter)
            if primitive is None:
                continue

            expected_count = expected_counts[primitive]
            if not self.geometry_parameter_is_array(parameter, param_type):
                raise MojoRefusal(
                    "Mojo geometry stage "
                    f"{primitive} input primitive parameter '{parameter.name}' "
                    f"must be an array with {expected_count} element(s)"
                )

            array_count = self.geometry_parameter_array_count(parameter, param_type)
            if array_count is None:
                continue
            if array_count != expected_count:
                raise MojoRefusal(
                    "Mojo geometry stage "
                    f"{primitive} input primitive parameter '{parameter.name}' "
                    f"must have {expected_count} element(s), got {array_count}"
                )

    def geometry_stream_info(self, type_name):
        if type_name is None:
            return None

        stream_names = {"PointStream", "LineStream", "TriangleStream"}
        type_text = self.type_name_string(type_name)
        if not type_text or "<" not in type_text or not type_text.endswith(">"):
            return None

        stream_name, generic_arg = type_text.split("<", 1)
        stream_name = stream_name.strip()
        if stream_name not in stream_names:
            return None

        output_type = generic_arg[:-1].strip()
        if not output_type:
            return None
        return stream_name, output_type

    def geometry_stream_mapped_type(self, type_name):
        stream_info = self.geometry_stream_info(type_name)
        if stream_info is None:
            return None
        stream_name, output_type = stream_info
        self.required_geometry_stream_helpers.add(stream_name)
        return f"CrossGLMojo{stream_name}[{self.map_type(output_type)}]"

    def tessellation_patch_type_info(self, type_name):
        type_text = self.type_name_string(type_name)
        generic = self.parse_generic_type_name(type_text)
        if generic is None:
            return None
        patch_name, generic_args = generic
        if patch_name not in {"InputPatch", "OutputPatch"}:
            return None
        if len(generic_args) != 2:
            raise MojoRefusal(
                "Mojo tessellation patch type "
                f"'{type_text}' must use {patch_name}<T, N>"
            )
        element_type, point_count = generic_args
        if not element_type or not point_count:
            return None
        return patch_name, element_type, point_count

    def tessellation_patch_mapped_type(self, type_name):
        patch_info = self.tessellation_patch_type_info(type_name)
        if patch_info is None:
            return None
        patch_name, element_type, point_count = patch_info
        self.required_tessellation_patch_helpers.add(patch_name)
        return f"CrossGLMojo{patch_name}[{self.map_type(element_type)}, {point_count}]"

    def validate_tessellation_patch_signatures(self, param_infos, stage_name):
        for parameter, param_type in param_infos:
            patch_info = self.tessellation_patch_type_info(param_type)
            if patch_info is None:
                continue
            patch_name, _element_type, point_count = patch_info
            value = self.literal_int_value(point_count)
            if value is not None and value <= 0:
                raise MojoRefusal(
                    "Mojo "
                    f"{stage_name} stage {patch_name} parameter "
                    f"'{parameter.name}' control point count ({value}) "
                    "must be positive"
                )

    def validate_tessellation_output_patch_control_points(self, func, param_infos):
        output_points = self.literal_int_value(
            self.tessellation_output_control_points(func)
        )
        if output_points is None:
            return

        for parameter, param_type in param_infos:
            patch_info = self.tessellation_patch_type_info(param_type)
            if patch_info is None:
                continue
            patch_name, _element_type, point_count = patch_info
            if patch_name != "OutputPatch":
                continue
            patch_points = self.literal_int_value(point_count)
            if patch_points is None or patch_points == output_points:
                continue
            raise MojoRefusal(
                "Mojo tessellation_control stage OutputPatch parameter "
                f"'{parameter.name}' control point count ({patch_points}) "
                "must match outputcontrolpoints "
                f"({output_points})"
            )

    def validate_geometry_stage(self, func, param_infos):
        self.geometry_maxvertexcount(func)

        if not any(
            self.geometry_stream_info(param_type) for _, param_type in param_infos
        ):
            raise MojoRefusal(
                "Mojo geometry stage parameters must include a PointStream, "
                "LineStream, or TriangleStream output parameter"
            )

        if not any(
            self.geometry_input_primitive_qualifier(param) for param, _ in param_infos
        ):
            raise MojoRefusal(
                "Mojo geometry stage parameters must include an input primitive "
                "parameter qualified as point, line, triangle, lineadj, or triangleadj"
            )

        self.validate_geometry_input_primitive_arity(param_infos)

    def generate_geometry_stage_comments(self, func, param_infos):
        maxvertexcount = self.geometry_maxvertexcount(func)
        input_descriptions = []
        stream_descriptions = []

        for parameter, param_type in param_infos:
            primitive = self.geometry_input_primitive_qualifier(parameter)
            if primitive is not None:
                input_descriptions.append(f"{primitive}:{parameter.name}")

            stream_info = self.geometry_stream_info(param_type)
            if stream_info is not None:
                stream_name, output_type = stream_info
                stream_descriptions.append(
                    f"{stream_name}:{parameter.name}->{output_type}"
                )

        lines = [f"# CrossGL geometry stage: maxvertexcount={maxvertexcount}\n"]
        if input_descriptions:
            lines.append(
                f"# CrossGL geometry input primitive: {', '.join(input_descriptions)}\n"
            )
        if stream_descriptions:
            lines.append(
                f"# CrossGL geometry output stream: {', '.join(stream_descriptions)}\n"
            )
        return "".join(lines)

    def tessellation_output_control_points(self, func):
        arguments = self.geometry_stage_attribute_arguments(func, "outputcontrolpoints")
        if not arguments:
            raise MojoRefusal(
                "Mojo tessellation_control stage requires outputcontrolpoints attribute"
            )
        if len(arguments) != 1:
            raise MojoRefusal(
                "Mojo tessellation_control stage outputcontrolpoints requires "
                "exactly one argument"
            )
        value_text = self.attribute_value_to_string(arguments[0])
        value = self.literal_int_value(arguments[0])
        if value is not None and value <= 0:
            raise MojoRefusal(
                "Mojo tessellation_control stage outputcontrolpoints "
                f"({value}) must be positive"
            )
        return value_text

    def canonical_tessellation_domain(self, value):
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return {
            "tri": "tri",
            "triangle": "tri",
            "triangles": "tri",
            "quad": "quad",
            "quads": "quad",
            "isoline": "isoline",
            "isolines": "isoline",
        }.get(normalized)

    def tessellation_domain(self, func, stage_name, required=False):
        value = self.mojo_stage_attribute_value(func, "domain", stage_name)
        if value is None:
            if required:
                raise MojoRefusal(f"Mojo {stage_name} stage requires domain attribute")
            return None
        canonical = self.canonical_tessellation_domain(value)
        if canonical is None:
            raise MojoRefusal(
                f"Mojo {stage_name} stage domain '{value}' must be tri, quad, "
                "or isoline"
            )
        return canonical

    def tessellation_partitioning(self, func, stage_name):
        value = self.mojo_stage_attribute_value(func, "partitioning", stage_name)
        if value is None:
            return None
        normalized = str(value).strip().lower()
        partitioning_aliases = {
            "integer": "integer",
            "equal": "integer",
            "equal_spacing": "integer",
            "fractional_even": "fractional_even",
            "fractional_even_spacing": "fractional_even",
            "fractional_odd": "fractional_odd",
            "fractional_odd_spacing": "fractional_odd",
            "pow2": "pow2",
        }
        canonical = partitioning_aliases.get(normalized)
        if canonical is None:
            valid = ", ".join(["fractional_even", "fractional_odd", "integer", "pow2"])
            raise MojoRefusal(
                f"Mojo {stage_name} stage partitioning '{value}' must be one of: "
                f"{valid}"
            )
        return canonical

    def validate_tessellation_stage(self, func, stage_name, param_infos):
        self.validate_tessellation_patch_signatures(param_infos, stage_name)
        if stage_name == "tessellation_control":
            self.tessellation_output_control_points(func)
            self.tessellation_domain(func, stage_name)
            self.validate_tessellation_output_patch_control_points(func, param_infos)
        elif stage_name == "tessellation_evaluation":
            self.tessellation_domain(func, stage_name, required=True)
        self.tessellation_partitioning(func, stage_name)

    def generate_tessellation_patch_parameter_comments(self, param_infos):
        patch_descriptions = []
        for parameter, param_type in param_infos:
            patch_info = self.tessellation_patch_type_info(param_type)
            if patch_info is None:
                continue
            patch_name, element_type, point_count = patch_info
            patch_descriptions.append(
                f"{patch_name}:{parameter.name}->{element_type}[{point_count}]"
            )
        if not patch_descriptions:
            return ""
        return (
            f"# CrossGL tessellation patch parameter: {', '.join(patch_descriptions)}\n"
        )

    def generate_tessellation_stage_comments(self, func, stage_name, param_infos):
        lines = []
        if stage_name == "tessellation_control":
            output_points = self.tessellation_output_control_points(func)
            lines.append(
                "# CrossGL tessellation control stage: "
                f"outputcontrolpoints={output_points}\n"
            )
        else:
            domain = self.tessellation_domain(func, stage_name, required=True)
            lines.append(f"# CrossGL tessellation evaluation stage: domain={domain}\n")

        domain = self.tessellation_domain(
            func, stage_name, required=stage_name == "tessellation_evaluation"
        )
        if stage_name == "tessellation_control" and domain is not None:
            lines.append(f"# CrossGL tessellation domain: {domain}\n")

        partitioning = self.tessellation_partitioning(func, stage_name)
        if partitioning is not None:
            lines.append(f"# CrossGL tessellation partitioning: {partitioning}\n")

        output_topology = self.mojo_stage_attribute_value(
            func, "outputtopology", stage_name
        )
        if output_topology is not None:
            lines.append(f"# CrossGL tessellation output topology: {output_topology}\n")

        patch_constant_function = self.mojo_stage_attribute_value(
            func, "patchconstantfunc", stage_name
        )
        if patch_constant_function is not None:
            lines.append(
                "# CrossGL tessellation patch constant function: "
                f"{patch_constant_function}\n"
            )

        patch_comments = self.generate_tessellation_patch_parameter_comments(
            param_infos
        )
        if patch_comments:
            lines.append(patch_comments)

        return "".join(lines)

    def mesh_stage_attribute_arguments(self, func, *attribute_names):
        for attribute_name in attribute_names:
            arguments = self.geometry_stage_attribute_arguments(func, attribute_name)
            if arguments:
                return arguments
        return []

    def mesh_stage_attribute_value(self, func, stage_name, *attribute_names):
        arguments = self.mesh_stage_attribute_arguments(func, *attribute_names)
        if not arguments:
            return None
        display_name = attribute_names[0]
        if len(arguments) != 1:
            raise MojoRefusal(
                f"Mojo {stage_name} stage {display_name} requires exactly one argument"
            )
        return self.attribute_value_to_string(arguments[0])

    def mesh_stage_positive_attribute(self, func, stage_name, *attribute_names):
        arguments = self.mesh_stage_attribute_arguments(func, *attribute_names)
        if not arguments:
            return None
        display_name = attribute_names[0]
        if len(arguments) != 1:
            raise MojoRefusal(
                f"Mojo {stage_name} stage {display_name} requires exactly one argument"
            )
        value_text = self.attribute_value_to_string(arguments[0])
        value = self.literal_int_value(arguments[0])
        if value is not None and value <= 0:
            raise MojoRefusal(
                f"Mojo {stage_name} stage {display_name} ({value}) must be positive"
            )
        return value_text

    def mesh_stage_numthreads(self, func, stage_name):
        arguments = self.mesh_stage_attribute_arguments(func, "numthreads")
        if not arguments:
            return None
        if len(arguments) > 3:
            raise MojoRefusal(
                f"Mojo {stage_name} stage numthreads requires at most three arguments"
            )
        values = [self.attribute_value_to_string(arg) for arg in arguments]
        for index, argument in enumerate(arguments):
            value = self.literal_int_value(argument)
            if value is not None and value <= 0:
                raise MojoRefusal(
                    f"Mojo {stage_name} stage numthreads value {index + 1} "
                    "must be positive"
                )
        while len(values) < 3:
            values.append("1")
        return tuple(values[:3])

    def mesh_stage_layout_local_size(self, stage_node):
        if stage_node is None:
            return None
        sizes = {"local_size_x": None, "local_size_y": None, "local_size_z": None}
        for layout in getattr(stage_node, "layout_qualifiers", []) or []:
            for entry in getattr(layout, "entries", []) or []:
                name = getattr(entry, "name", None)
                if name not in sizes:
                    continue
                arguments = (
                    getattr(entry, "arguments", getattr(entry, "args", [])) or []
                )
                if len(arguments) != 1:
                    continue
                sizes[name] = self.attribute_value_to_string(arguments[0])
        if all(value is None for value in sizes.values()):
            return None
        return (
            sizes["local_size_x"] or "1",
            sizes["local_size_y"] or "1",
            sizes["local_size_z"] or "1",
        )

    def canonical_mesh_output_topology(self, value):
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return {
            "point": "point",
            "points": "point",
            "line": "line",
            "lines": "line",
            "triangle": "triangle",
            "triangles": "triangle",
            "triangle_cw": "triangle_cw",
            "triangle_ccw": "triangle_ccw",
        }.get(normalized)

    def mesh_output_topology(self, func, stage_name, required=False):
        value = self.mesh_stage_attribute_value(func, stage_name, "outputtopology")
        if value is None:
            if required:
                raise MojoRefusal(
                    f"Mojo {stage_name} stage requires outputtopology attribute"
                )
            return None
        canonical = self.canonical_mesh_output_topology(value)
        if canonical is None:
            raise MojoRefusal(
                f"Mojo {stage_name} stage outputtopology '{value}' must be point, "
                "line, triangle, triangle_cw, or triangle_ccw"
            )
        return canonical

    def mesh_parameter_role(self, parameter):
        role_aliases = {
            "mesh_payload": "mesh_payload",
            "meshpayload": "mesh_payload",
            "payload": "mesh_payload",
            "vertices": "vertices",
            "vertex": "vertices",
            "indices": "indices",
            "index": "indices",
            "primitives": "primitives",
            "primitive": "primitives",
        }
        for attr in getattr(parameter, "attributes", []) or []:
            name = getattr(attr, "name", None)
            if name is None:
                continue
            normalized = str(name).strip().lower().replace("-", "_")
            role = role_aliases.get(normalized)
            if role is not None:
                return role
        return None

    def mesh_parameter_array_count(self, parameter, param_type):
        return self.geometry_parameter_array_count(parameter, param_type)

    def mesh_parameter_base_type(self, param_type):
        if self.is_array_type_name(param_type):
            base_type, _size = self.parse_array_type_name(param_type)
            return base_type
        return param_type

    def validate_mesh_parameter_roles(self, stage_name, param_infos):
        seen_roles = {}
        for parameter, param_type in param_infos:
            role = self.mesh_parameter_role(parameter)
            if role is None:
                continue
            seen_roles.setdefault(role, []).append(parameter.name)

            if role == "mesh_payload" and stage_name != "mesh":
                raise MojoRefusal(
                    "Mojo mesh payload parameters are only valid on mesh stages"
                )
            if role in {"vertices", "indices", "primitives"}:
                if stage_name != "mesh":
                    raise MojoRefusal(
                        f"Mojo mesh output parameter role {role} is only valid "
                        "on mesh stages"
                    )
                if not self.geometry_parameter_is_array(parameter, param_type):
                    raise MojoRefusal(
                        f"Mojo mesh {role} parameter '{parameter.name}' must be "
                        "an array"
                    )

        for role, names in seen_roles.items():
            if role == "mesh_payload" and len(names) > 1:
                raise MojoRefusal(
                    "Mojo mesh stage accepts at most one mesh_payload parameter"
                )

    def validate_mesh_output_parameter_bounds(self, func, stage_name, param_infos):
        if stage_name != "mesh":
            return
        max_vertices = self.literal_int_value(
            self.mesh_stage_positive_attribute(
                func, stage_name, "maxvertices", "max_vertices"
            )
        )
        max_primitives = self.literal_int_value(
            self.mesh_stage_positive_attribute(
                func, stage_name, "maxprimitives", "max_primitives"
            )
        )
        for parameter, param_type in param_infos:
            role = self.mesh_parameter_role(parameter)
            if role not in {"vertices", "indices", "primitives"}:
                continue
            array_count = self.mesh_parameter_array_count(parameter, param_type)
            if array_count is None:
                continue
            if role == "vertices" and max_vertices is not None:
                if array_count > max_vertices:
                    raise MojoRefusal(
                        f"Mojo mesh vertices parameter '{parameter.name}' count "
                        f"({array_count}) exceeds maxvertices ({max_vertices})"
                    )
            if role in {"indices", "primitives"} and max_primitives is not None:
                if array_count > max_primitives:
                    raise MojoRefusal(
                        f"Mojo mesh {role} parameter '{parameter.name}' count "
                        f"({array_count}) exceeds maxprimitives ({max_primitives})"
                    )

    def walk_mesh_ops(self, node, visited=None):
        if visited is None:
            visited = set()
        if node is None:
            return
        if isinstance(node, (str, int, float, bool)):
            return
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        if isinstance(node, MeshOpNode):
            yield node
            for argument in getattr(node, "arguments", []) or []:
                yield from self.walk_mesh_ops(argument, visited)
            return

        if isinstance(node, list):
            for item in node:
                yield from self.walk_mesh_ops(item, visited)
            return

        for attribute_name in (
            "statements",
            "body",
            "expression",
            "value",
            "left",
            "right",
            "condition",
            "then_branch",
            "else_branch",
            "if_body",
            "args",
            "arguments",
        ):
            if hasattr(node, attribute_name):
                yield from self.walk_mesh_ops(getattr(node, attribute_name), visited)

    def validate_mesh_op_stage(self, operation, stage_name):
        if operation == "SetMeshOutputCounts":
            if stage_name != "mesh":
                raise MojoRefusal(
                    f"Mojo {stage_name or 'function'} stage cannot call "
                    "SetMeshOutputCounts; SetMeshOutputCounts is only valid in "
                    "mesh stages"
                )
            return
        if operation == "DispatchMesh":
            if stage_name not in {"task", "amplification", "object"}:
                raise MojoRefusal(
                    f"Mojo {stage_name or 'function'} stage cannot call "
                    "DispatchMesh; DispatchMesh is only valid in task, "
                    "amplification, or object stages"
                )
            return
        raise MojoRefusal(f"Unsupported Mojo mesh intrinsic {operation}")

    def validate_mesh_op_arguments(self, operation, arguments, stage_name):
        if operation == "SetMeshOutputCounts":
            if len(arguments) != 2:
                raise MojoRefusal(
                    "Mojo mesh SetMeshOutputCounts requires exactly two arguments"
                )
            labels = ("vertex count", "primitive count")
        elif operation == "DispatchMesh":
            if len(arguments) not in {3, 4}:
                raise MojoRefusal(
                    "Mojo mesh DispatchMesh requires three group counts and an "
                    "optional payload argument"
                )
            labels = ("x count", "y count", "z count")
        else:
            return

        for label, argument in zip(labels, arguments):
            if not self.mesh_count_argument_is_integer_scalar(argument):
                raise MojoRefusal(
                    f"Mojo {stage_name} {operation} {label} argument must be an "
                    "integer scalar"
                )
            value = self.literal_int_value(argument)
            if value is not None and value < 0:
                raise MojoRefusal(
                    f"Mojo {stage_name} {operation} {label} argument must be "
                    "non-negative"
                )

    def mesh_count_argument_is_integer_scalar(self, argument):
        if isinstance(argument, bool) or isinstance(argument, float):
            return False
        if isinstance(argument, int) and not isinstance(argument, bool):
            return True
        value = self.literal_int_value(argument)
        if value is not None:
            return True
        argument_type = self.expression_result_type(argument)
        if argument_type is None:
            return True
        return self.is_scalar_integer_type(argument_type)

    def validate_mesh_stage(self, func, stage_name, param_infos):
        self.mesh_stage_numthreads(func, stage_name)
        if stage_name == "mesh":
            self.mesh_output_topology(func, stage_name, required=True)
            self.mesh_stage_positive_attribute(
                func, stage_name, "maxvertices", "max_vertices"
            )
            self.mesh_stage_positive_attribute(
                func, stage_name, "maxprimitives", "max_primitives"
            )
        self.validate_mesh_parameter_roles(stage_name, param_infos)
        self.validate_mesh_output_parameter_bounds(func, stage_name, param_infos)

        for mesh_op in self.walk_mesh_ops(getattr(func, "body", [])):
            operation = getattr(mesh_op, "operation", None)
            arguments = list(getattr(mesh_op, "arguments", []) or [])
            self.validate_mesh_op_stage(operation, stage_name)
            self.validate_mesh_op_arguments(operation, arguments, stage_name)

    def generate_mesh_parameter_role_comments(self, param_infos):
        descriptions = []
        for parameter, param_type in param_infos:
            role = self.mesh_parameter_role(parameter)
            if role is None:
                continue
            base_type = self.mesh_parameter_base_type(param_type)
            array_count = self.mesh_parameter_array_count(parameter, param_type)
            suffix = f"[{array_count}]" if array_count is not None else ""
            descriptions.append(f"{role}:{parameter.name}->{base_type}{suffix}")
        if not descriptions:
            return ""
        return f"# CrossGL mesh parameter role: {', '.join(descriptions)}\n"

    def generate_mesh_stage_comments(
        self, func, stage_name, param_infos, stage_node=None
    ):
        lines = [f"# CrossGL mesh/task stage: stage={stage_name}\n"]
        numthreads = self.mesh_stage_numthreads(func, stage_name)
        if numthreads is not None:
            lines.append(f"# CrossGL mesh/task numthreads: {', '.join(numthreads)}\n")

        local_size = self.mesh_stage_layout_local_size(stage_node)
        if local_size is not None:
            lines.append(f"# CrossGL mesh/task local size: {', '.join(local_size)}\n")

        if stage_name == "mesh":
            topology = self.mesh_output_topology(func, stage_name, required=True)
            lines.append(f"# CrossGL mesh output topology: {topology}\n")

        max_vertices = self.mesh_stage_positive_attribute(
            func, stage_name, "maxvertices", "max_vertices"
        )
        if max_vertices is not None:
            lines.append(f"# CrossGL mesh max vertices: {max_vertices}\n")

        max_primitives = self.mesh_stage_positive_attribute(
            func, stage_name, "maxprimitives", "max_primitives"
        )
        if max_primitives is not None:
            lines.append(f"# CrossGL mesh max primitives: {max_primitives}\n")

        role_comments = self.generate_mesh_parameter_role_comments(param_infos)
        if role_comments:
            lines.append(role_comments)
        return "".join(lines)

    def generate_stage_local_declarations(
        self,
        stage,
        emitted_stage_local_variables,
        emitted_stage_local_structs,
    ):
        code = ""
        for node in getattr(stage, "local_structs", []) or []:
            if (
                isinstance(node, StructNode)
                and node.name not in emitted_stage_local_structs
            ):
                code += self.generate_struct(node)
                emitted_stage_local_structs.add(node.name)

        for node in getattr(stage, "local_cbuffers", []) or []:
            if (
                isinstance(node, StructNode)
                and node.name not in emitted_stage_local_structs
            ):
                code += self.generate_resource_metadata_comment(
                    node, getattr(node, "name", None), kind="cbuffer"
                )
                code += self.generate_struct(node)
                code += self.generate_cbuffer_member_placeholders(node)
                emitted_stage_local_structs.add(node.name)

        for node in getattr(stage, "local_variables", []) or []:
            code += self.generate_stage_local_variable_declaration(
                node, emitted_stage_local_variables
            )
        return code

    def generate_stage_local_variable_declaration(
        self, node, emitted_stage_local_variables
    ):
        if self.is_mojo_shared_variable(node):
            if isinstance(node, ArrayNode):
                variable_type = self.array_declared_type(node)
            else:
                variable_type = self.variable_declared_type(node) or "float"
            self.register_variable_type(getattr(node, "name", None), variable_type)
            return ""

        if isinstance(node, ArrayNode):
            variable_type = self.array_declared_type(node)
            self.register_variable_type(node.name, variable_type)
            key = (node.name, variable_type)
            if key in emitted_stage_local_variables:
                return ""
            emitted_stage_local_variables.add(key)
            return self.generate_array_declaration(node)

        variable_type = self.variable_declared_type(node) or "float"
        self.register_variable_type(getattr(node, "name", None), variable_type)
        self.register_resource_access_metadata(node, variable_type)
        key = (getattr(node, "name", None), variable_type)
        if key in emitted_stage_local_variables:
            return ""
        emitted_stage_local_variables.add(key)

        if getattr(node, "initial_value", None) is not None:
            self.validate_expression_target_shape(
                node.initial_value, variable_type, f"stage declaration {node.name}"
            )
            init_expr = self.generate_expression(
                node.initial_value, variable_type, f"stage declaration {node.name}"
            )
            self.validate_scalar_target_conversion(
                node.initial_value, variable_type, f"stage declaration {node.name}"
            )
            var_name = self.mojo_identifier(node.name)
            return (
                f"{self.generate_resource_metadata_comment(node, variable_type)}"
                f"var {var_name}: {self.map_type(variable_type)} = {init_expr}\n"
            )

        resource_comment = self.generate_resource_metadata_comment(node, variable_type)
        var_name = self.mojo_identifier(node.name)
        if self.is_array_type_name(variable_type):
            return (
                f"{resource_comment}var {var_name} = "
                f"{self.array_initial_value_for_type(variable_type)}\n"
            )
        if self.is_struct_type_name(variable_type):
            return f"{resource_comment}var {var_name} = {self.zero_value_for_type(variable_type)}\n"
        if self.is_resource_type_name(variable_type):
            mapped_type = self.map_type(variable_type)
            return (
                f"{resource_comment}var {var_name}: {mapped_type} = "
                f"{self.zero_value_for_type(variable_type)}\n"
            )
        return (
            f"var {var_name}: {self.map_type(variable_type)} = "
            f"{self.zero_value_for_type(variable_type)}\n"
        )

    def is_mojo_shared_variable(self, node):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        return bool(
            qualifiers
            & {"groupshared", "shared", "threadgroup", "thread_group", "workgroup"}
        )

    def generate_function_local_shared_declarations(self, stage_node, indent):
        if stage_node is None:
            return ""
        code = ""
        for node in getattr(stage_node, "local_variables", []) or []:
            if not self.is_mojo_shared_variable(node):
                continue
            code += self.generate_statement(node, indent)
        return code

    def stage_local_functions_to_emit(self, stage):
        functions = list(getattr(stage, "local_functions", []) or [])
        implemented_signatures = {
            self.function_signature_key(func)
            for func in functions
            if not self.is_function_prototype(func)
        }
        emitted = []
        seen = set()
        for func in functions:
            signature = self.function_signature_key(func)
            if signature in seen:
                continue
            if self.is_function_prototype(func) and signature in implemented_signatures:
                continue
            emitted.append(func)
            seen.add(signature)
        return emitted

    def stage_local_extra_params(self, stage):
        entry_point = getattr(stage, "entry_point", None)
        if entry_point is None:
            return {}
        entry_params = []
        for param in (
            getattr(entry_point, "parameters", getattr(entry_point, "params", [])) or []
        ):
            param_name = getattr(param, "name", None)
            if not param_name:
                continue
            entry_params.append((param_name, self.function_parameter_type_name(param)))

        extras = {}
        if not entry_params:
            return extras

        for func in self.stage_local_functions_to_emit(stage):
            func_param_names = {
                getattr(param, "name", None)
                for param in (
                    getattr(func, "parameters", getattr(func, "params", [])) or []
                )
            }
            captures = [
                (name, type_name)
                for name, type_name in entry_params
                if name not in func_param_names
                and self.node_references_identifier(getattr(func, "body", None), name)
            ]
            if captures:
                extras[getattr(func, "name", None)] = captures
        return extras

    def node_references_identifier(self, node, name):
        if node is None:
            return False
        if isinstance(node, IdentifierNode):
            return getattr(node, "name", None) == name
        if isinstance(node, str):
            return node == name
        if isinstance(node, (int, float, bool)):
            return False
        if isinstance(node, (list, tuple)):
            return any(self.node_references_identifier(item, name) for item in node)
        if not hasattr(node, "__dict__"):
            return False

        for attr, value in node.__dict__.items():
            if attr in {"annotations", "parent", "source_location"}:
                continue
            if attr == "name" and not isinstance(node, IdentifierNode):
                continue
            if self.node_references_identifier(value, name):
                return True
        return False

    def is_function_prototype(self, func):
        return not self.body_statements(getattr(func, "body", None))

    def function_signature_key(self, func):
        params = []
        for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
            if hasattr(param, "param_type"):
                param_type = self.convert_type_node_to_string(param.param_type)
            elif hasattr(param, "vtype"):
                param_type = self.type_name(param.vtype)
            else:
                param_type = "float"
            params.append(param_type)
        return (getattr(func, "name", None), tuple(params))
