"""Geometry, tessellation, mesh and ray-tracing stage contracts of the Rust generator: stage attributes, parameter roles and their validation."""

from ..ast import MeshOpNode
from ..refusals import RustRefusal
from .rust_exact_lookups import RUST_RAY_STAGE_METADATA_NAMES
from .stage_utils import (
    normalize_stage_name,
    stage_matches,
)


class RustStageContractMixin:
    """Stage-contract members of ``RustCodeGen``."""

    def unsupported_stage_types(self):
        return set()

    def has_geometry_stage(self, ast_node, target_stage=None):
        """Return whether the AST contains a geometry stage to emit."""
        if not stage_matches(target_stage, "geometry"):
            return False

        for stage_type in getattr(ast_node, "stages", {}) or {}:
            if normalize_stage_name(stage_type) == "geometry":
                return True

        for func in getattr(ast_node, "functions", []) or []:
            qualifiers = list(getattr(func, "qualifiers", []) or [])
            qualifier = getattr(func, "qualifier", None)
            if qualifier:
                qualifiers.append(qualifier)
            if any(
                normalize_stage_name(qualifier) == "geometry"
                for qualifier in qualifiers
            ):
                return True

        return False

    def has_tessellation_stage(self, ast_node, target_stage=None):
        """Return whether the AST contains a tessellation stage to emit."""
        tessellation_stages = {"tessellation_control", "tessellation_evaluation"}
        if not any(
            stage_matches(target_stage, stage_name)
            for stage_name in tessellation_stages
        ):
            return False

        for stage_type in getattr(ast_node, "stages", {}) or {}:
            if normalize_stage_name(stage_type) in tessellation_stages:
                return True

        for func in getattr(ast_node, "functions", []) or []:
            qualifiers = list(getattr(func, "qualifiers", []) or [])
            qualifier = getattr(func, "qualifier", None)
            if qualifier:
                qualifiers.append(qualifier)
            if any(
                normalize_stage_name(qualifier) in tessellation_stages
                for qualifier in qualifiers
            ):
                return True

        return False

    def generate_rust_tessellation_patch_helpers(self):
        """Emit compile-oriented placeholders for CrossGL tessellation patches."""
        return (
            "pub struct CglRustInputPatch<T, const N: usize> {\n"
            "    control_points: [T; N],\n"
            "}\n\n"
            "impl<T: Default, const N: usize> Default for "
            "CglRustInputPatch<T, N> {\n"
            "    fn default() -> Self {\n"
            "        Self { control_points: core::array::from_fn(|_| "
            "T::default()) }\n"
            "    }\n"
            "}\n\n"
            "impl<T, const N: usize> core::ops::Index<usize> for "
            "CglRustInputPatch<T, N> {\n"
            "    type Output = T;\n\n"
            "    fn index(&self, index: usize) -> &Self::Output {\n"
            "        &self.control_points[index]\n"
            "    }\n"
            "}\n\n"
            "impl<T, const N: usize> core::ops::IndexMut<usize> for "
            "CglRustInputPatch<T, N> {\n"
            "    fn index_mut(&mut self, index: usize) -> &mut Self::Output {\n"
            "        &mut self.control_points[index]\n"
            "    }\n"
            "}\n\n"
            "pub struct CglRustOutputPatch<T, const N: usize> {\n"
            "    control_points: [T; N],\n"
            "}\n\n"
            "impl<T: Default, const N: usize> Default for "
            "CglRustOutputPatch<T, N> {\n"
            "    fn default() -> Self {\n"
            "        Self { control_points: core::array::from_fn(|_| "
            "T::default()) }\n"
            "    }\n"
            "}\n\n"
            "impl<T, const N: usize> core::ops::Index<usize> for "
            "CglRustOutputPatch<T, N> {\n"
            "    type Output = T;\n\n"
            "    fn index(&self, index: usize) -> &Self::Output {\n"
            "        &self.control_points[index]\n"
            "    }\n"
            "}\n\n"
            "impl<T, const N: usize> core::ops::IndexMut<usize> for "
            "CglRustOutputPatch<T, N> {\n"
            "    fn index_mut(&mut self, index: usize) -> &mut Self::Output {\n"
            "        &mut self.control_points[index]\n"
            "    }\n"
            "}\n\n"
        )

    def rust_ray_stage_names(self):
        return {
            "ray_any_hit",
            "ray_callable",
            "ray_closest_hit",
            "ray_generation",
            "ray_intersection",
            "ray_miss",
        }

    def rust_ray_stage_metadata_name(self, shader_type):
        name = RUST_RAY_STAGE_METADATA_NAMES.get(shader_type)
        if name is None:
            return self.refuse_lowering(
                f"ray-tracing stage {shader_type}: not a CrossGL ray-tracing stage",
                str(shader_type),
            )
        return name

    def rust_mesh_stage_names(self):
        return {"amplification", "mesh", "object", "task"}

    def rust_mesh_stage_attribute_arguments(self, func, *attribute_names):
        for attribute_name in attribute_names:
            arguments = self.rust_stage_attribute_arguments(func, attribute_name)
            if arguments:
                return arguments
        return []

    def rust_mesh_stage_attribute_value(self, func, stage_name, *attribute_names):
        arguments = self.rust_mesh_stage_attribute_arguments(func, *attribute_names)
        if not arguments:
            return None
        display_name = attribute_names[0]
        if len(arguments) != 1:
            raise RustRefusal(
                f"Rust {stage_name} stage {display_name} requires exactly one argument"
            )
        return self.attribute_argument_text(arguments[0])

    def rust_mesh_stage_positive_attribute(self, func, stage_name, *attribute_names):
        arguments = self.rust_mesh_stage_attribute_arguments(func, *attribute_names)
        if not arguments:
            return None
        display_name = attribute_names[0]
        if len(arguments) != 1:
            raise RustRefusal(
                f"Rust {stage_name} stage {display_name} requires exactly one argument"
            )
        value_text = self.attribute_argument_text(arguments[0])
        value = self.literal_int_value(arguments[0])
        if value is not None and value <= 0:
            raise RustRefusal(
                f"Rust {stage_name} stage {display_name} ({value}) must be positive"
            )
        return value_text

    def rust_mesh_stage_numthreads(self, func, stage_name):
        arguments = self.rust_mesh_stage_attribute_arguments(func, "numthreads")
        if not arguments:
            return None
        if len(arguments) > 3:
            raise RustRefusal(
                f"Rust {stage_name} stage numthreads requires at most three arguments"
            )
        values = [self.attribute_argument_text(arg) for arg in arguments]
        for index, argument in enumerate(arguments):
            value = self.literal_int_value(argument)
            if value is not None and value <= 0:
                raise RustRefusal(
                    f"Rust {stage_name} stage numthreads value {index + 1} "
                    "must be positive"
                )
        while len(values) < 3:
            values.append("1")
        return tuple(values[:3])

    def rust_mesh_stage_layout_local_size(self, stage_node, stage_name):
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
                    raise RustRefusal(
                        f"Rust {stage_name} stage {name} requires exactly one argument"
                    )
                value = self.literal_int_value(arguments[0])
                if value is not None and value <= 0:
                    raise RustRefusal(
                        f"Rust {stage_name} stage {name} must be positive"
                    )
                sizes[name] = self.attribute_argument_text(arguments[0])
        if all(value is None for value in sizes.values()):
            return None
        return (
            sizes["local_size_x"] or "1",
            sizes["local_size_y"] or "1",
            sizes["local_size_z"] or "1",
        )

    def canonical_rust_mesh_output_topology(self, value):
        if value is None:
            return None
        normalized = str(value).strip().strip("\"'").lower()
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

    def rust_mesh_output_topology(self, func, stage_name, required=False):
        value = self.rust_mesh_stage_attribute_value(func, stage_name, "outputtopology")
        if value is None:
            if required:
                raise RustRefusal(
                    f"Rust {stage_name} stage requires outputtopology attribute"
                )
            return None
        topology = self.canonical_rust_mesh_output_topology(value)
        if topology is None:
            raise RustRefusal(
                f"Rust {stage_name} stage outputtopology '{value}' must be point, "
                "line, triangle, triangle_cw, or triangle_ccw"
            )
        return topology

    def rust_mesh_parameter_role(self, parameter):
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

    def rust_mesh_parameter_base_type(self, param_type):
        if self.is_array_type_name(param_type):
            base_type, _sizes = self.c_array_type_parts(param_type)
            return base_type
        return param_type

    def rust_mesh_param_infos(self, parameters):
        return [
            (parameter, self.function_parameter_type(parameter))
            for parameter in parameters
        ]

    def validate_rust_mesh_parameter_roles(self, stage_name, param_infos):
        seen_roles = {}
        for parameter, param_type in param_infos:
            role = self.rust_mesh_parameter_role(parameter)
            if role is None:
                continue
            seen_roles.setdefault(role, []).append(parameter.name)

            if role == "mesh_payload" and stage_name != "mesh":
                raise RustRefusal(
                    "Rust mesh payload parameters are only valid on mesh stages"
                )
            if role in {"vertices", "indices", "primitives"}:
                if stage_name != "mesh":
                    raise RustRefusal(
                        f"Rust mesh output parameter role {role} is only valid "
                        "on mesh stages"
                    )
                if not self.rust_parameter_is_array(parameter):
                    raise RustRefusal(
                        f"Rust mesh {role} parameter '{parameter.name}' must be "
                        "an array"
                    )

        if len(seen_roles.get("mesh_payload", [])) > 1:
            raise RustRefusal(
                "Rust mesh stage accepts at most one mesh_payload parameter"
            )

    def validate_rust_mesh_output_parameter_bounds(self, func, stage_name, param_infos):
        if stage_name != "mesh":
            return
        max_vertices = self.literal_int_value(
            self.rust_mesh_stage_positive_attribute(
                func, stage_name, "maxvertices", "max_vertices"
            )
        )
        max_primitives = self.literal_int_value(
            self.rust_mesh_stage_positive_attribute(
                func, stage_name, "maxprimitives", "max_primitives"
            )
        )
        for parameter, _param_type in param_infos:
            role = self.rust_mesh_parameter_role(parameter)
            if role not in {"vertices", "indices", "primitives"}:
                continue
            array_count = self.rust_parameter_array_count(parameter)
            if array_count is None:
                continue
            if (
                role == "vertices"
                and max_vertices is not None
                and array_count > max_vertices
            ):
                raise RustRefusal(
                    f"Rust mesh vertices parameter '{parameter.name}' count "
                    f"({array_count}) exceeds maxvertices ({max_vertices})"
                )
            if (
                role in {"indices", "primitives"}
                and max_primitives is not None
                and array_count > max_primitives
            ):
                raise RustRefusal(
                    f"Rust mesh {role} parameter '{parameter.name}' count "
                    f"({array_count}) exceeds maxprimitives ({max_primitives})"
                )

    def walk_rust_mesh_ops(self, node, visited=None):
        if visited is None:
            visited = set()
        if node is None or isinstance(node, (str, int, float, bool)):
            return
        if isinstance(node, list):
            for item in node:
                yield from self.walk_rust_mesh_ops(item, visited)
            return
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        if isinstance(node, MeshOpNode):
            yield node
            for argument in getattr(node, "arguments", []) or []:
                yield from self.walk_rust_mesh_ops(argument, visited)
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
                yield from self.walk_rust_mesh_ops(
                    getattr(node, attribute_name), visited
                )

    def validate_rust_mesh_op_stage(self, operation, stage_name):
        if operation == "SetMeshOutputCounts":
            if stage_name != "mesh":
                raise RustRefusal(
                    f"Rust {stage_name or 'function'} stage cannot call "
                    "SetMeshOutputCounts; SetMeshOutputCounts is only valid in "
                    "mesh stages"
                )
            return
        if operation == "DispatchMesh":
            if stage_name not in {"task", "amplification", "object"}:
                raise RustRefusal(
                    f"Rust {stage_name or 'function'} stage cannot call "
                    "DispatchMesh; DispatchMesh is only valid in task, "
                    "amplification, or object stages"
                )
            return
        raise RustRefusal(f"Unsupported Rust mesh intrinsic {operation}")

    def rust_mesh_count_argument_is_integer_scalar(self, argument):
        if isinstance(argument, bool) or isinstance(argument, float):
            return False
        if isinstance(argument, int) and not isinstance(argument, bool):
            return True
        if self.literal_int_value(argument) is not None:
            return True
        argument_type = self.expression_result_type(argument)
        if argument_type is None:
            return True
        return self.normalize_scalar_type(argument_type) in {
            "i16",
            "u16",
            "i32",
            "u32",
            "i64",
            "u64",
        }

    def validate_rust_mesh_op_arguments(self, operation, arguments, stage_name):
        if operation == "SetMeshOutputCounts":
            if len(arguments) != 2:
                raise RustRefusal(
                    "Rust mesh SetMeshOutputCounts requires exactly two arguments"
                )
            labels = ("vertex count", "primitive count")
        elif operation == "DispatchMesh":
            if len(arguments) not in {3, 4}:
                raise RustRefusal(
                    "Rust mesh DispatchMesh requires three group counts and an "
                    "optional payload argument"
                )
            labels = ("x count", "y count", "z count")
        else:
            return

        for label, argument in zip(labels, arguments):
            if not self.rust_mesh_count_argument_is_integer_scalar(argument):
                raise RustRefusal(
                    f"Rust {stage_name} {operation} {label} argument must be an "
                    "integer scalar"
                )
            value = self.literal_int_value(argument)
            if value is not None and value < 0:
                raise RustRefusal(
                    f"Rust {stage_name} {operation} {label} argument must be "
                    "non-negative"
                )

    def validate_rust_mesh_stage(self, func, parameters, shader_type, stage_node=None):
        self.rust_mesh_stage_numthreads(func, shader_type)
        self.rust_mesh_stage_layout_local_size(stage_node, shader_type)
        param_infos = self.rust_mesh_param_infos(parameters)
        if shader_type == "mesh":
            self.rust_mesh_output_topology(func, shader_type, required=True)
            self.rust_mesh_stage_positive_attribute(
                func, shader_type, "maxvertices", "max_vertices"
            )
            self.rust_mesh_stage_positive_attribute(
                func, shader_type, "maxprimitives", "max_primitives"
            )
        self.validate_rust_mesh_parameter_roles(shader_type, param_infos)
        self.validate_rust_mesh_output_parameter_bounds(func, shader_type, param_infos)

        for mesh_op in self.walk_rust_mesh_ops(getattr(func, "body", [])):
            operation = getattr(mesh_op, "operation", None)
            arguments = list(getattr(mesh_op, "arguments", []) or [])
            self.validate_rust_mesh_op_stage(operation, shader_type)
            self.validate_rust_mesh_op_arguments(operation, arguments, shader_type)

    def generate_rust_mesh_parameter_role_comments(self, param_infos):
        descriptions = []
        for parameter, param_type in param_infos:
            role = self.rust_mesh_parameter_role(parameter)
            if role is None:
                continue
            base_type = self.rust_mesh_parameter_base_type(param_type)
            array_count = self.rust_parameter_array_count(parameter)
            suffix = f"[{array_count}]" if array_count is not None else ""
            descriptions.append(f"{role}:{parameter.name}->{base_type}{suffix}")
        if not descriptions:
            return ""
        return f"// CrossGL mesh parameter role: {', '.join(descriptions)}\n"

    def rust_geometry_maxvertexcount(self, func):
        arguments = self.rust_stage_attribute_arguments(func, "maxvertexcount")
        if not arguments:
            raise RustRefusal("Rust geometry stage requires maxvertexcount attribute")
        if len(arguments) != 1:
            raise RustRefusal(
                "Rust geometry stage maxvertexcount requires exactly one argument"
            )
        value_text = self.attribute_argument_text(arguments[0])
        value = self.literal_int_value(arguments[0])
        if value is not None and value <= 0:
            raise RustRefusal(
                f"Rust geometry stage maxvertexcount ({value}) must be positive"
            )
        return value_text

    def parameter_raw_type(self, parameter):
        if hasattr(parameter, "param_type"):
            return parameter.param_type
        if hasattr(parameter, "vtype"):
            return parameter.vtype
        return "void"

    def rust_parameter_qualifiers(self, parameter):
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
            arguments = self.attribute_arguments(attr)
            if not arguments:
                continue
            primitive = self.attribute_argument_text(arguments[0])
            normalized = str(primitive).lower()
            if normalized in allowed_qualifiers:
                qualifiers.append(normalized)
        return qualifiers

    def rust_geometry_input_primitive_qualifier(self, parameter):
        primitive_qualifiers = {
            "point",
            "line",
            "triangle",
            "lineadj",
            "triangleadj",
        }
        for qualifier in self.rust_parameter_qualifiers(parameter):
            if qualifier in primitive_qualifiers:
                return qualifier
        return None

    def rust_parameter_is_array(self, parameter):
        raw_type = self.parameter_raw_type(parameter)
        if getattr(raw_type, "__class__", None).__name__ == "ArrayType":
            return True
        type_name = self.type_name_string(raw_type)
        return bool(type_name and "[" in type_name and "]" in type_name)

    def rust_parameter_array_count(self, parameter):
        raw_type = self.parameter_raw_type(parameter)
        if getattr(raw_type, "__class__", None).__name__ == "ArrayType":
            size = getattr(raw_type, "size", None)
            if size is None:
                return None
            return self.literal_int_value(size)

        type_name = self.type_name_string(raw_type)
        if not type_name or "[" not in type_name or "]" not in type_name:
            return None
        _base_type, sizes = self.c_array_type_parts(type_name)
        return sizes[0] if sizes else None

    def validate_rust_geometry_input_primitive_arity(self, parameters):
        expected_counts = {
            "point": 1,
            "line": 2,
            "triangle": 3,
            "lineadj": 4,
            "triangleadj": 6,
        }

        for parameter in parameters:
            primitive = self.rust_geometry_input_primitive_qualifier(parameter)
            if primitive is None:
                continue

            expected_count = expected_counts[primitive]
            if not self.rust_parameter_is_array(parameter):
                raise RustRefusal(
                    "Rust geometry stage "
                    f"{primitive} input primitive parameter '{parameter.name}' "
                    f"must be an array with {expected_count} element(s)"
                )

            array_count = self.rust_parameter_array_count(parameter)
            if array_count is None:
                continue
            if array_count != expected_count:
                raise RustRefusal(
                    "Rust geometry stage "
                    f"{primitive} input primitive parameter '{parameter.name}' "
                    f"must have {expected_count} element(s), got {array_count}"
                )

    def validate_rust_geometry_stage(self, func, parameters):
        self.rust_geometry_maxvertexcount(func)

        if not any(
            self.rust_geometry_stream_info(self.parameter_raw_type(param))
            for param in parameters
        ):
            raise RustRefusal(
                "Rust geometry stage parameters must include a PointStream, "
                "LineStream, or TriangleStream output parameter"
            )

        if not any(
            self.rust_geometry_input_primitive_qualifier(param) for param in parameters
        ):
            raise RustRefusal(
                "Rust geometry stage parameters must include an input primitive "
                "parameter qualified as point, line, triangle, lineadj, or triangleadj"
            )

        self.validate_rust_geometry_input_primitive_arity(parameters)

    def rust_geometry_stream_info(self, type_name):
        if type_name is None:
            return None

        stream_names = {"PointStream", "LineStream", "TriangleStream"}
        if hasattr(type_name, "pointee_type") or hasattr(type_name, "referenced_type"):
            return None

        name = getattr(type_name, "name", None)
        generic_args = getattr(type_name, "generic_args", []) or []
        if name in stream_names and generic_args:
            return name, self.map_type(generic_args[0])

        type_text = self.type_name_string(type_name)
        if not type_text or "<" not in type_text or not type_text.endswith(">"):
            return None

        base_name, args_text = type_text.split("<", 1)
        base_name = base_name.strip()
        if base_name not in stream_names:
            return None

        generic_args = self.split_top_level_generic_args(args_text[:-1])
        if len(generic_args) != 1:
            return None
        output_type = generic_args[0].strip()
        if not output_type:
            return None
        return base_name, self.map_type(output_type)

    def rust_stage_entry_function_name(self, stage_name, func):
        """Return the deterministic Rust entry name for special stages."""
        name = getattr(func, "name", None)
        if (
            normalize_stage_name(stage_name)
            in {
                "amplification",
                "mesh",
                "object",
                "task",
                "tessellation_control",
                "tessellation_evaluation",
            }
            and name == "main"
        ):
            return f"{normalize_stage_name(stage_name)}_{name}"
        return name

    def rust_tessellation_stage_attribute_value(self, func, attribute_name, stage_name):
        arguments = self.rust_stage_attribute_arguments(func, attribute_name)
        if not arguments:
            return None
        if len(arguments) != 1:
            raise RustRefusal(
                f"Rust {stage_name} stage {attribute_name} requires exactly one "
                "argument"
            )
        return self.attribute_argument_text(arguments[0])

    def rust_tessellation_output_control_points(self, func):
        attribute_name = "outputcontrolpoints"
        arguments = self.rust_stage_attribute_arguments(func, attribute_name)
        if not arguments:
            arguments = self.rust_stage_attribute_arguments(func, "vertices")
        if not arguments:
            raise RustRefusal(
                "Rust tessellation_control stage requires outputcontrolpoints attribute"
            )
        if len(arguments) != 1:
            raise RustRefusal(
                "Rust tessellation_control stage outputcontrolpoints requires "
                "exactly one argument"
            )

        value_text = self.attribute_argument_text(arguments[0])
        value = self.literal_int_value(arguments[0])
        if value is None:
            raise RustRefusal(
                "Rust tessellation_control stage outputcontrolpoints "
                f"'{value_text}' must be an integer literal"
            )
        if value <= 0:
            raise RustRefusal(
                "Rust tessellation_control stage outputcontrolpoints "
                f"({value}) must be positive"
            )
        if value > 32:
            raise RustRefusal(
                "Rust tessellation_control stage outputcontrolpoints "
                f"({value}) must be at most 32"
            )
        return value_text, value

    def canonical_rust_tessellation_domain(self, value):
        if value is None:
            return None
        normalized = str(value).strip().strip("\"'").lower()
        return {
            "tri": "tri",
            "triangle": "tri",
            "triangles": "tri",
            "quad": "quad",
            "quads": "quad",
            "isoline": "isoline",
            "isolines": "isoline",
        }.get(normalized)

    def rust_tessellation_domain(self, func, stage_name, required=False):
        value = self.rust_tessellation_stage_attribute_value(func, "domain", stage_name)
        if value is None:
            if required:
                raise RustRefusal(f"Rust {stage_name} stage requires domain attribute")
            return None

        domain = self.canonical_rust_tessellation_domain(value)
        if domain is None:
            raise RustRefusal(
                f"Rust {stage_name} stage domain '{value}' must be one of: "
                "isoline, quad, tri"
            )
        return domain

    def rust_tessellation_partitioning(self, func, stage_name, required=False):
        value = self.rust_tessellation_stage_attribute_value(
            func, "partitioning", stage_name
        )
        if value is None:
            if required:
                raise RustRefusal(
                    f"Rust {stage_name} stage requires partitioning attribute"
                )
            return None

        normalized = str(value).strip().strip("\"'").lower()
        partitioning = {
            "integer": "integer",
            "equal": "integer",
            "equal_spacing": "integer",
            "fractional_even": "fractional_even",
            "fractional_even_spacing": "fractional_even",
            "fractional_odd": "fractional_odd",
            "fractional_odd_spacing": "fractional_odd",
            "pow2": "pow2",
        }.get(normalized)
        if partitioning is None:
            raise RustRefusal(
                f"Rust {stage_name} stage partitioning '{value}' must be one of: "
                "fractional_even, fractional_odd, integer, pow2"
            )
        return partitioning

    def rust_tessellation_patch_info(self, type_name):
        base_name, args = self.generic_type_parts(type_name)
        patch_kinds = {
            "InputPatch": "InputPatch",
            "OutputPatch": "OutputPatch",
            "CglRustInputPatch": "InputPatch",
            "CglRustOutputPatch": "OutputPatch",
        }
        kind = patch_kinds.get(base_name)
        if kind is None:
            return None

        info = {
            "kind": kind,
            "valid": len(args) == 2,
            "element_type": None,
            "mapped_element_type": None,
            "count_text": None,
            "count_value": None,
        }
        if len(args) >= 1:
            element_type = args[0].strip()
            if element_type:
                info["element_type"] = element_type
                info["mapped_element_type"] = self.map_type(element_type)
        if len(args) >= 2:
            count_text = args[1].strip()
            info["count_text"] = count_text
            info["count_value"] = self.literal_int_value(count_text)
        return info

    def rust_tessellation_patch_parameters(self, parameters, patch_type):
        for parameter in parameters:
            info = self.rust_tessellation_patch_info(self.parameter_raw_type(parameter))
            if info is not None and info["kind"] == patch_type:
                yield parameter, info

    def validate_rust_tessellation_patch_parameter_signature(
        self, parameter, info, stage_name
    ):
        patch_type = info["kind"]
        parameter_name = getattr(parameter, "name", None)
        name_clause = (
            f" parameter '{parameter_name}'" if parameter_name else " parameter"
        )
        type_name = self.type_name_string(self.parameter_raw_type(parameter))

        if not info["valid"]:
            raise RustRefusal(
                f"Rust {stage_name} stage {patch_type}{name_clause} must use "
                f"{patch_type}<T, N>; found '{type_name or patch_type}'"
            )
        if info["element_type"] is None:
            raise RustRefusal(
                f"Rust {stage_name} stage {patch_type}{name_clause} must include "
                "an element type"
            )

        control_points = info["count_value"]
        if control_points is None:
            raise RustRefusal(
                f"Rust {stage_name} stage {patch_type}{name_clause} control "
                f"point count '{info['count_text']}' must be an integer literal"
            )
        if control_points <= 0:
            raise RustRefusal(
                f"Rust {stage_name} stage {patch_type}{name_clause} control "
                f"point count ({control_points}) must be positive"
            )
        if control_points > 32:
            raise RustRefusal(
                f"Rust {stage_name} stage {patch_type}{name_clause} control "
                "point count "
                f"({control_points}) must be at most 32"
            )

    def validate_rust_tessellation_patch_parameter_signatures(
        self, parameters, patch_type, stage_name
    ):
        for parameter, info in self.rust_tessellation_patch_parameters(
            parameters, patch_type
        ):
            self.validate_rust_tessellation_patch_parameter_signature(
                parameter,
                info,
                stage_name,
            )

    def validate_rust_tessellation_control_stage(self, func, parameters):
        stage_name = "tessellation_control"
        self.rust_tessellation_domain(func, stage_name, required=True)
        self.rust_tessellation_partitioning(func, stage_name, required=True)
        self.rust_tessellation_output_control_points(func)

        input_patch_parameters = list(
            self.rust_tessellation_patch_parameters(parameters, "InputPatch")
        )
        if not input_patch_parameters:
            raise RustRefusal(
                "Rust tessellation_control stage parameters must include an "
                "InputPatch<T, N> parameter"
            )
        self.validate_rust_tessellation_patch_parameter_signatures(
            parameters,
            "InputPatch",
            stage_name,
        )

    def validate_rust_tessellation_evaluation_stage(self, func, parameters):
        stage_name = "tessellation_evaluation"
        self.rust_tessellation_domain(func, stage_name, required=True)
        self.rust_tessellation_partitioning(func, stage_name)

        output_patch_parameters = list(
            self.rust_tessellation_patch_parameters(parameters, "OutputPatch")
        )
        if not output_patch_parameters:
            raise RustRefusal(
                "Rust tessellation_evaluation stage parameters must include an "
                "OutputPatch<T, N> parameter"
            )
        self.validate_rust_tessellation_patch_parameter_signatures(
            parameters,
            "OutputPatch",
            stage_name,
        )

    def validate_rust_tessellation_stage(self, func, parameters, shader_type):
        if shader_type == "tessellation_control":
            self.validate_rust_tessellation_control_stage(func, parameters)
        elif shader_type == "tessellation_evaluation":
            self.validate_rust_tessellation_evaluation_stage(func, parameters)

    def generate_rust_tessellation_stage_comments(self, func, parameters, shader_type):
        metadata = []
        domain = self.rust_tessellation_domain(
            func,
            shader_type,
            required=shader_type == "tessellation_evaluation",
        )
        if domain is not None:
            metadata.append(f"domain={domain}")
        partitioning = self.rust_tessellation_partitioning(func, shader_type)
        if partitioning is not None:
            metadata.append(f"partitioning={partitioning}")
        output_topology = self.rust_tessellation_stage_attribute_value(
            func,
            "outputtopology",
            shader_type,
        )
        if output_topology is not None:
            metadata.append(f"outputtopology={output_topology}")
        if shader_type == "tessellation_control":
            output_control_points, _value = (
                self.rust_tessellation_output_control_points(func)
            )
            metadata.append(f"outputcontrolpoints={output_control_points}")
        patch_constant_function = self.rust_tessellation_stage_attribute_value(
            func,
            "patchconstantfunc",
            shader_type,
        )
        if patch_constant_function is not None:
            metadata.append(f"patchconstantfunc={patch_constant_function}")

        lines = [
            f"// CrossGL {shader_type} stage"
            f"{': ' + ', '.join(metadata) if metadata else ''}\n"
        ]

        patch_descriptions = []
        for parameter in parameters:
            info = self.rust_tessellation_patch_info(self.parameter_raw_type(parameter))
            if info is None or not info["valid"]:
                continue
            patch_descriptions.append(
                f"{info['kind']}:{parameter.name}->{info['mapped_element_type']}"
                f"[{info['count_text']}]"
            )
        if patch_descriptions:
            lines.append(
                "// CrossGL tessellation patch parameters: "
                f"{', '.join(patch_descriptions)}\n"
            )
        return "".join(lines)

    def rust_tessellation_patch_mapped_type(self, type_name):
        info = self.rust_tessellation_patch_info(type_name)
        if info is None or not info["valid"]:
            return None
        helper_name = {
            "InputPatch": "CglRustInputPatch",
            "OutputPatch": "CglRustOutputPatch",
        }[info["kind"]]
        return f"{helper_name}<{info['mapped_element_type']}, {info['count_text']}>"

    def validate_supported_stage_types(self, ast, target_stage=None):
        unsupported_stages = set()
        for stage_type in getattr(ast, "stages", {}) or {}:
            stage_name = normalize_stage_name(stage_type)
            if stage_name in self.unsupported_stage_types() and stage_matches(
                target_stage, stage_name
            ):
                unsupported_stages.add(stage_name)

        for func in getattr(ast, "functions", []) or []:
            for qualifier in getattr(func, "qualifiers", []) or []:
                stage_name = normalize_stage_name(qualifier)
                if stage_name in self.unsupported_stage_types() and stage_matches(
                    target_stage, stage_name
                ):
                    unsupported_stages.add(stage_name)

        if unsupported_stages:
            stage_list = ", ".join(sorted(unsupported_stages))
            raise RustRefusal(
                f"Rust output does not support stage type(s): {stage_list}"
            )

    def stage_entry_name_counts(self, stages):
        """Count entry-point names across shader stages."""
        counts = {}
        for stage in stages.values():
            entry_point = getattr(stage, "entry_point", None)
            name = getattr(entry_point, "name", None)
            if name:
                counts[name] = counts.get(name, 0) + 1
        return counts

    def stage_entry_function_name(self, stage_name, entry_point, name_counts):
        """Avoid duplicate Rust symbols when multiple stages use ``main``."""
        name = getattr(entry_point, "name", None)
        if not name:
            return name
        if (
            normalize_stage_name(stage_name)
            in {
                "amplification",
                "mesh",
                "object",
                "task",
                "tessellation_control",
                "tessellation_evaluation",
            }
            and name == "main"
        ):
            return f"{normalize_stage_name(stage_name)}_{name}"
        if name_counts.get(name, 0) > 1:
            return f"{stage_name}_{name}"
        return name
