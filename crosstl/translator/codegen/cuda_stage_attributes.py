"""Stage attributes of the CUDA generator: geometry, tessellation and mesh/task contracts and their validation."""

from ..refusals import CudaRefusal
from .array_utils import parse_array_type
from .stage_utils import normalize_stage_name


class CudaStageAttributeMixin:
    """Stage attribute members of ``CudaCodeGen``."""

    def function_stage_name(self, node):
        if self.current_stage_name:
            return self.current_stage_name
        for qualifier_name in self.function_stage_qualifier_names(node):
            return qualifier_name
        return None

    def function_stage_qualifier_names(self, node):
        """Return normalized stage qualifiers from legacy qualifiers and attributes."""
        supported_stage_names = {
            "compute",
            "fragment",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "vertex",
        } | self.cuda_mesh_task_stage_names()
        qualifiers = list(getattr(node, "qualifiers", []) or [])
        qualifier = getattr(node, "qualifier", None)
        if qualifier:
            qualifiers.append(qualifier)

        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name:
                qualifiers.append(attr_name)

        return [
            qualifier_name
            for qualifier_name in (
                normalize_stage_name(qualifier) for qualifier in qualifiers
            )
            if qualifier_name in supported_stage_names
        ]

    def cuda_stage_attribute_arguments(self, func, attribute_name):
        requested = str(attribute_name).strip().lower().replace("-", "_")
        for attr in getattr(func, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name is None:
                continue
            normalized = str(attr_name).strip().lower().replace("-", "_")
            if normalized == requested:
                return self.attribute_arguments(attr)
        return []

    def cuda_geometry_maxvertexcount(self, func):
        arguments = self.cuda_stage_attribute_arguments(func, "maxvertexcount")
        if not arguments:
            raise CudaRefusal("CUDA geometry stage requires maxvertexcount attribute")
        if len(arguments) != 1:
            raise CudaRefusal(
                "CUDA geometry stage maxvertexcount requires exactly one argument"
            )
        value_text = self.attribute_value_to_string(arguments[0])
        value = self.literal_int_value(arguments[0])
        if value is not None and value <= 0:
            raise CudaRefusal(
                f"CUDA geometry stage maxvertexcount ({value}) must be positive"
            )
        return value_text

    def cuda_stage_attribute_value(self, func, attribute_name, stage_name):
        arguments = self.cuda_stage_attribute_arguments(func, attribute_name)
        if not arguments:
            return None
        if len(arguments) != 1:
            raise CudaRefusal(
                f"CUDA {stage_name} stage {attribute_name} requires exactly one "
                "argument"
            )
        return self.attribute_value_to_string(arguments[0])

    def cuda_tessellation_output_control_points(self, func):
        arguments = self.cuda_stage_attribute_arguments(func, "outputcontrolpoints")
        if not arguments:
            raise CudaRefusal(
                "CUDA tessellation_control stage requires outputcontrolpoints "
                "attribute"
            )
        if len(arguments) != 1:
            raise CudaRefusal(
                "CUDA tessellation_control stage outputcontrolpoints requires "
                "exactly one argument"
            )
        value_text = self.attribute_value_to_string(arguments[0])
        value = self.literal_int_value(arguments[0])
        if value is not None and value <= 0:
            raise CudaRefusal(
                "CUDA tessellation_control stage outputcontrolpoints "
                f"({value}) must be positive"
            )
        return value_text

    def canonical_cuda_tessellation_domain(self, value):
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

    def cuda_tessellation_domain(self, func, stage_name, required=False):
        value = self.cuda_stage_attribute_value(func, "domain", stage_name)
        if value is None:
            if required:
                raise CudaRefusal(f"CUDA {stage_name} stage requires domain attribute")
            return None
        canonical = self.canonical_cuda_tessellation_domain(value)
        if canonical is None:
            raise CudaRefusal(
                f"CUDA {stage_name} stage domain '{value}' must be tri, quad, "
                "or isoline"
            )
        return canonical

    def cuda_tessellation_partitioning(self, func, stage_name):
        value = self.cuda_stage_attribute_value(func, "partitioning", stage_name)
        if value is None:
            return None
        normalized = str(value).strip().lower()
        valid_partitioning = {
            "integer",
            "fractional_even",
            "fractional_odd",
            "pow2",
        }
        if normalized not in valid_partitioning:
            valid = ", ".join(sorted(valid_partitioning))
            raise CudaRefusal(
                f"CUDA {stage_name} stage partitioning '{value}' must be one of: "
                f"{valid}"
            )
        return normalized

    def validate_cuda_tessellation_stage(self, func, stage_name):
        if stage_name == "tessellation_control":
            self.cuda_tessellation_output_control_points(func)
            self.cuda_tessellation_domain(func, stage_name)
        elif stage_name == "tessellation_evaluation":
            self.cuda_tessellation_domain(func, stage_name, required=True)
        self.cuda_tessellation_partitioning(func, stage_name)

    def generate_cuda_tessellation_stage_comments(self, func, stage_name):
        lines = []
        if stage_name == "tessellation_control":
            output_points = self.cuda_tessellation_output_control_points(func)
            lines.append(
                "// CrossGL tessellation control stage: "
                f"outputcontrolpoints={output_points}\n"
            )
        else:
            domain = self.cuda_tessellation_domain(func, stage_name, required=True)
            lines.append(
                "// CrossGL tessellation evaluation stage: " f"domain={domain}\n"
            )

        domain = self.cuda_tessellation_domain(
            func, stage_name, required=stage_name == "tessellation_evaluation"
        )
        if stage_name == "tessellation_control" and domain is not None:
            lines.append(f"// CrossGL tessellation domain: {domain}\n")

        partitioning = self.cuda_tessellation_partitioning(func, stage_name)
        if partitioning is not None:
            lines.append(f"// CrossGL tessellation partitioning: {partitioning}\n")

        output_topology = self.cuda_stage_attribute_value(
            func, "outputtopology", stage_name
        )
        if output_topology is not None:
            lines.append(
                "// CrossGL tessellation output topology: " f"{output_topology}\n"
            )

        patch_constant_function = self.cuda_stage_attribute_value(
            func, "patchconstantfunc", stage_name
        )
        if patch_constant_function is not None:
            lines.append(
                "// CrossGL tessellation patch constant function: "
                f"{patch_constant_function}\n"
            )

        return "".join(lines)

    def cuda_mesh_task_stage_label(self, stage_name):
        return {
            "amplification": "amplification/task",
            "mesh": "mesh",
            "object": "object/task",
            "task": "task",
        }[
            stage_name
        ]  # only called for cuda_mesh_task_stage_names(), which this table covers

    def cuda_mesh_task_positive_attribute(self, func, attribute_name, stage_name):
        value = self.cuda_stage_attribute_value(func, attribute_name, stage_name)
        if value is None:
            return None
        value_int = self.literal_int_value(
            self.cuda_stage_attribute_arguments(func, attribute_name)[0]
        )
        if value_int is not None and value_int <= 0:
            raise CudaRefusal(
                f"CUDA {stage_name} stage {attribute_name} ({value_int}) "
                "must be positive"
            )
        return value

    def cuda_mesh_task_numthreads(self, func, stage_name):
        for attribute_name in ("numthreads", "local_size", "workgroup_size"):
            arguments = self.cuda_stage_attribute_arguments(func, attribute_name)
            if not arguments:
                continue
            if len(arguments) != 3:
                raise CudaRefusal(
                    f"CUDA {stage_name} stage {attribute_name} requires exactly "
                    "three arguments"
                )
            values = []
            for axis, argument in zip("xyz", arguments):
                value = self.literal_int_value(argument)
                if value is not None and value <= 0:
                    raise CudaRefusal(
                        f"CUDA {stage_name} stage {attribute_name} {axis} "
                        f"dimension ({value}) must be positive"
                    )
                values.append(self.attribute_value_to_string(argument))
            return ", ".join(values)
        return None

    def cuda_mesh_task_output_topology(self, func, stage_name):
        value = self.cuda_stage_attribute_value(func, "outputtopology", stage_name)
        if value is None:
            return None
        normalized = str(value).strip().lower()
        topology_aliases = {
            "point": "point",
            "points": "point",
            "line": "line",
            "lines": "line",
            "triangle": "triangle",
            "triangles": "triangle",
        }
        topology = topology_aliases.get(normalized)
        if topology is None:
            raise CudaRefusal(
                f"CUDA {stage_name} stage outputtopology '{value}' must be "
                "point, line, or triangle"
            )
        return topology

    def validate_cuda_mesh_task_stage(self, func, stage_name):
        self.cuda_mesh_task_numthreads(func, stage_name)
        if stage_name == "mesh":
            self.cuda_mesh_task_output_topology(func, stage_name)
            self.cuda_mesh_task_positive_attribute(func, "max_vertices", stage_name)
            self.cuda_mesh_task_positive_attribute(func, "max_primitives", stage_name)

    def generate_cuda_mesh_task_stage_comments(self, func, stage_name):
        label = self.cuda_mesh_task_stage_label(stage_name)
        lines = [f"// CrossGL {label} stage\n"]

        numthreads = self.cuda_mesh_task_numthreads(func, stage_name)
        if numthreads is not None:
            lines.append(f"// CrossGL mesh/task numthreads: {numthreads}\n")

        if stage_name == "mesh":
            topology = self.cuda_mesh_task_output_topology(func, stage_name)
            if topology is not None:
                lines.append(f"// CrossGL mesh output topology: {topology}\n")
            for attribute_name in ("max_vertices", "max_primitives"):
                value = self.cuda_mesh_task_positive_attribute(
                    func, attribute_name, stage_name
                )
                if value is not None:
                    lines.append(f"// CrossGL mesh {attribute_name}: {value}\n")

        return "".join(lines)

    def parameter_raw_type(self, parameter):
        if hasattr(parameter, "param_type"):
            return parameter.param_type
        if hasattr(parameter, "vtype"):
            return parameter.vtype
        return "void"

    def cuda_parameter_qualifiers(self, parameter):
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
            primitive = self.attribute_value_to_string(arguments[0])
            normalized = str(primitive).lower()
            if normalized in allowed_qualifiers:
                qualifiers.append(normalized)
        return qualifiers

    def cuda_geometry_input_primitive_qualifier(self, parameter):
        primitive_qualifiers = {
            "point",
            "line",
            "triangle",
            "lineadj",
            "triangleadj",
        }
        for qualifier in self.cuda_parameter_qualifiers(parameter):
            if qualifier in primitive_qualifiers:
                return qualifier
        return None

    def cuda_parameter_is_array(self, parameter):
        raw_type = self.parameter_raw_type(parameter)
        if (
            hasattr(raw_type, "element_type")
            and str(type(raw_type)).find("ArrayType") != -1
        ):
            return True
        type_name = self.type_name_string(raw_type)
        return bool(type_name and "[" in type_name and "]" in type_name)

    def cuda_parameter_array_count(self, parameter):
        raw_type = self.parameter_raw_type(parameter)
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

        type_name = self.type_name_string(raw_type)
        if not type_name or "[" not in type_name or "]" not in type_name:
            return None
        _base_type, array_size = parse_array_type(type_name)
        return array_size

    def validate_cuda_geometry_input_primitive_arity(self, parameters):
        expected_counts = {
            "point": 1,
            "line": 2,
            "triangle": 3,
            "lineadj": 4,
            "triangleadj": 6,
        }

        for parameter in parameters:
            primitive = self.cuda_geometry_input_primitive_qualifier(parameter)
            if primitive is None:
                continue

            expected_count = expected_counts[primitive]
            if not self.cuda_parameter_is_array(parameter):
                raise CudaRefusal(
                    "CUDA geometry stage "
                    f"{primitive} input primitive parameter '{parameter.name}' "
                    f"must be an array with {expected_count} element(s)"
                )

            array_count = self.cuda_parameter_array_count(parameter)
            if array_count is None:
                continue
            if array_count != expected_count:
                raise CudaRefusal(
                    "CUDA geometry stage "
                    f"{primitive} input primitive parameter '{parameter.name}' "
                    f"must have {expected_count} element(s), got {array_count}"
                )

    def validate_cuda_geometry_stage(self, func, parameters):
        self.cuda_geometry_maxvertexcount(func)

        if not any(
            self.cuda_geometry_stream_info(self.parameter_raw_type(param))
            for param in parameters
        ):
            raise CudaRefusal(
                "CUDA geometry stage parameters must include a PointStream, "
                "LineStream, or TriangleStream output parameter"
            )

        if not any(
            self.cuda_geometry_input_primitive_qualifier(param) for param in parameters
        ):
            raise CudaRefusal(
                "CUDA geometry stage parameters must include an input primitive "
                "parameter qualified as point, line, triangle, lineadj, or triangleadj"
            )

        self.validate_cuda_geometry_input_primitive_arity(parameters)

    def generate_cuda_geometry_stage_comments(self, func, parameters):
        maxvertexcount = self.cuda_geometry_maxvertexcount(func)
        input_descriptions = []
        stream_descriptions = []

        for parameter in parameters:
            primitive = self.cuda_geometry_input_primitive_qualifier(parameter)
            if primitive is not None:
                input_descriptions.append(f"{primitive}:{parameter.name}")

            stream_info = self.cuda_geometry_stream_info(
                self.parameter_raw_type(parameter)
            )
            if stream_info is not None:
                stream_name, output_type = stream_info
                stream_descriptions.append(
                    f"{stream_name}:{parameter.name}->{output_type}"
                )

        lines = [f"// CrossGL geometry stage: maxvertexcount={maxvertexcount}\n"]
        if input_descriptions:
            lines.append(
                "// CrossGL geometry input primitive: "
                f"{', '.join(input_descriptions)}\n"
            )
        if stream_descriptions:
            lines.append(
                "// CrossGL geometry output stream: "
                f"{', '.join(stream_descriptions)}\n"
            )
        return "".join(lines)
