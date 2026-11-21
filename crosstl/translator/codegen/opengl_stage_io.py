"""Stage inputs and outputs in the GLSL generator: interface deduplication, entry storage bindings, semantics, parameter outputs, layout reservation and vertex input members."""

import re

from ..ast import (
    PointerType,
    ReferenceType,
    VariableNode,
)
from ..refusals import OpenGLRefusal
from .array_utils import (
    format_c_style_array_declaration,
    split_array_type_suffix,
)
from .clip_position_defaults import default_clip_position_member
from .glsl_refusals import patch_parameter_refusal_reason
from .glsl_spirv_varying_locations import (
    spirv_entry_input_locations,
    spirv_entry_output_locations,
    spirv_location_prefixed,
)
from .glsl_stage_builtin_members import note_vertex_builtin_output
from .glsl_stage_distances import distance_private_output_declaration
from .glsl_vertex_inputs import vertex_parameter_layouts
from .image_access_contracts import normalized_image_access
from .interface_conflicts import InterfaceLocationConflictError
from .stage_utils import (
    function_stage_name,
    normalize_stage_name,
    should_emit_qualified_function,
    stage_matches,
)


class OpenGLStageIoMixin:
    """Stage io members of ``GLSLCodeGen``."""

    def deduplicate_stage_interface_declarations(self, nodes):
        declarations = []
        declarations_by_name = {}
        for node in nodes:
            name = self.resource_node_name(node)
            if not name:
                declarations.append(node)
                continue

            vtype, _, array_suffix, _ = self.resource_declaration_shape(node)
            signature = (
                self.canonical_glsl_compile_time_type_text(self.map_type(vtype)),
                self.canonical_glsl_compile_time_type_text(array_suffix),
                self.glsl_variable_layout_prefix(node),
                self.glsl_variable_qualifier_prefix(node),
            )
            existing = declarations_by_name.get(name)
            if existing is None:
                declarations_by_name[name] = signature
                declarations.append(node)
                continue
            if existing != signature:
                raise OpenGLRefusal(
                    "Conflicting OpenGL stage interface declaration for "
                    f"'{name}': {signature} differs from {existing}"
                )
        return declarations

    def deduplicate_resource_declaration_nodes(self, nodes):
        declarations = []
        declarations_by_name = {}
        for node in nodes:
            name = self.resource_node_name(node)
            identity = self.resource_declaration_identity(node)
            if not name or identity is None:
                declarations.append(node)
                continue

            binding = self.explicit_resource_binding_index(node)
            existing = declarations_by_name.get(name)
            if existing is None:
                declarations_by_name[name] = {
                    "identity": identity,
                    "binding": binding,
                    "index": len(declarations),
                }
                declarations.append(node)
                continue

            if existing["identity"] != identity:
                raise OpenGLRefusal(
                    "Conflicting OpenGL resource declaration for "
                    f"'{name}': {identity} differs from {existing['identity']}"
                )

            existing_binding = existing["binding"]
            if (
                existing_binding is not None
                and binding is not None
                and existing_binding != binding
            ):
                raise OpenGLRefusal(
                    "Conflicting OpenGL resource binding for "
                    f"'{name}': binding {binding} differs from "
                    f"existing binding {existing_binding}"
                )
            if existing_binding is None and binding is not None:
                declarations[existing["index"]] = node
                existing["binding"] = binding

        return declarations

    def stage_entry_constant_struct_parameter_type(self, param):
        qualifiers = {str(q).lower() for q in getattr(param, "qualifiers", []) or []}
        if "constant" not in qualifiers:
            return None
        if self.explicit_resource_binding_index(param) is None:
            return None

        raw_type = self.resource_node_type(param)
        if isinstance(raw_type, ReferenceType):
            referenced_type = raw_type.referenced_type
            type_name = self.type_name_string(referenced_type)
        elif isinstance(raw_type, PointerType):
            type_name = self.type_name_string(raw_type.pointee_type)
        else:
            type_name = self.type_name_string(raw_type)
            type_name = self.strip_metal_parameter_indirection(type_name)

        if type_name in self.structs_by_name:
            return type_name
        return None

    def strip_metal_parameter_indirection(self, type_name):
        type_name = str(type_name or "").strip()
        while type_name.endswith(("*", "&")):
            type_name = type_name[:-1].strip()
        while True:
            stripped = re.sub(
                r"^(?:const|constant|device|thread|threadgroup)\s+",
                "",
                type_name,
                count=1,
            ).strip()
            if stripped == type_name:
                return stripped
            type_name = stripped

    def stage_entry_constant_value_parameter_type(self, param):
        resource_qualifiers = self.resource_parameter_qualifiers(param)
        if "uniform" in resource_qualifiers:
            raw_type = self.resource_node_type(param)
            if isinstance(raw_type, (PointerType, ReferenceType)):
                return None
            type_name = self.type_name_string(raw_type)
            if not type_name or type_name in self.structs_by_name:
                return None
            if self.is_sampler_type(type_name):
                return None
            if self.is_structured_buffer_type(
                type_name
            ) or self.is_constant_buffer_type(type_name):
                return None
            mapped_type = self.map_resource_type_with_format(
                self.resource_base_type(type_name), param
            )
            if self.is_opaque_resource_type(mapped_type):
                return None
            return type_name

        qualifiers = {str(q).lower() for q in getattr(param, "qualifiers", []) or []}
        if "constant" not in qualifiers:
            return None
        if self.explicit_resource_binding_index(param) is None:
            return None
        if self.stage_entry_constant_struct_parameter_type(param) is not None:
            return None

        raw_type = self.resource_node_type(param)
        if isinstance(raw_type, PointerType):
            return None
        if isinstance(raw_type, ReferenceType):
            raw_type = raw_type.referenced_type

        type_name = self.strip_metal_parameter_indirection(
            self.type_name_string(raw_type)
        )
        if not type_name or type_name in self.structs_by_name:
            return None
        if self.is_sampler_type(type_name):
            return None
        if self.is_structured_buffer_type(type_name) or self.is_constant_buffer_type(
            type_name
        ):
            return None
        mapped_type = self.map_resource_type_with_format(
            self.resource_base_type(type_name), param
        )
        if self.is_opaque_resource_type(mapped_type):
            return None
        return type_name

    def stage_entry_compute_value_parameter_type(self, param):
        if self.semantic_from_node(param) is not None:
            return None
        if self.is_stage_entry_resource_parameter(param):
            return None

        raw_type = self.resource_node_type(param)
        if isinstance(raw_type, (PointerType, ReferenceType)):
            return None
        type_name = self.type_name_string(raw_type)
        if not type_name or type_name in self.structs_by_name:
            return None
        if "[" in str(type_name) and "]" in str(type_name):
            return None
        if self.is_sampler_type(type_name):
            return None
        if self.is_structured_buffer_type(type_name) or self.is_constant_buffer_type(
            type_name
        ):
            return None
        mapped_type = self.map_resource_type_with_format(
            self.resource_base_type(type_name), param
        )
        if self.is_opaque_resource_type(mapped_type):
            return None
        return type_name

    def stage_entry_metal_scalar_reference_element_type(self, param):
        if self.explicit_resource_binding_index(param) is None:
            return None

        raw_type = self.resource_node_type(param)
        if not isinstance(raw_type, ReferenceType):
            return None
        address_space = str(getattr(raw_type, "address_space", None) or "").lower()
        if not address_space:
            qualifiers = {
                str(q).lower() for q in getattr(param, "qualifiers", []) or []
            }
            address_space = next(
                (
                    candidate
                    for candidate in ("device", "global", "storage")
                    if candidate in qualifiers
                ),
                "",
            )
        if address_space not in {"device", "global", "storage"}:
            return None

        type_name = self.type_name_string(raw_type.referenced_type)
        if not type_name or type_name in self.structs_by_name:
            return None
        return type_name

    def stage_entry_metal_pointer_element_type(self, param):
        if self.explicit_resource_binding_index(param) is None:
            return None

        scalar_reference_type = self.stage_entry_metal_scalar_reference_element_type(
            param
        )
        if scalar_reference_type is not None:
            return scalar_reference_type

        qualifiers = {str(q).lower() for q in getattr(param, "qualifiers", []) or []}
        if not qualifiers.intersection({"constant", "device"}):
            return None

        raw_type = self.resource_node_type(param)
        if isinstance(raw_type, PointerType):
            return self.type_name_string(raw_type.pointee_type)

        type_name = self.type_name_string(raw_type)
        if not type_name:
            return None
        if not str(type_name).strip().endswith("*"):
            return None
        return self.strip_metal_parameter_indirection(type_name)

    def stage_entry_metal_pointer_structured_buffer_type(self, param):
        element_type = self.stage_entry_metal_pointer_element_type(param)
        if element_type is None:
            return None
        if element_type in self.structs_by_name:
            return None

        qualifiers = {str(q).lower() for q in getattr(param, "qualifiers", []) or []}
        raw_type = self.resource_node_type(param)
        read_only_reference = isinstance(raw_type, ReferenceType) and not getattr(
            raw_type, "is_mutable", True
        )
        buffer_type = (
            "StructuredBuffer"
            if qualifiers.intersection({"constant", "const", "readonly"})
            or read_only_reference
            else "RWStructuredBuffer"
        )
        return f"{buffer_type}<{element_type}>"

    def glsl_structured_buffer_root_binding(self, node, root_name):
        if not root_name:
            return None
        resource_type, array_size, array_suffix, resource_count = (
            self.resource_declaration_shape(node)
        )
        if (
            array_size is not None
            or array_suffix
            or resource_count != 1
            or self.structured_buffer_type_name(resource_type)
            not in {"StructuredBuffer", "RWStructuredBuffer"}
        ):
            return None
        source_element_type = self.structured_buffer_source_element_type(resource_type)
        if source_element_type is None:
            return None
        element_type = self.structured_buffer_element_type(resource_type)
        access = self.structured_buffer_effective_access(resource_type, node)
        if access is None:
            access = "read_write"
        return {
            "kind": "storage-pointer",
            "domain": "storage-buffer",
            "root": root_name,
            "expression": root_name,
            "offset": 0,
            "element_type": self.map_type(element_type),
            "source_element_type": source_element_type,
            "type": resource_type,
            "access": access,
            "specializable": True,
            "dynamic_parameters": [],
            "resource_root": True,
        }

    def glsl_stage_entry_storage_pointer_binding(self, param, root_name):
        return self.glsl_structured_buffer_root_binding(param, root_name)

    def collect_glsl_storage_pointer_resource_bindings(self, declarations):
        bindings = {}
        for declaration in declarations or []:
            source_name = self.resource_node_name(declaration)
            if not source_name:
                continue
            emitted_name = self.glsl_global_identifier_name(source_name)
            binding = self.glsl_stage_entry_storage_pointer_binding(
                declaration, emitted_name
            )
            if binding is not None:
                bindings[source_name] = dict(binding)
                bindings[emitted_name] = dict(binding)
        return bindings

    def resource_parameter_qualifiers(self, param):
        return {
            str(qualifier).lower()
            for qualifier in getattr(param, "resource_qualifiers", []) or []
        }

    def stage_entry_storage_array_structured_buffer_type(self, param):
        resource_qualifiers = self.resource_parameter_qualifiers(param)
        if "storage" not in resource_qualifiers:
            return None

        raw_type = self.resource_node_type(param)
        if not (
            hasattr(raw_type, "element_type")
            and str(type(raw_type)).find("ArrayType") != -1
        ):
            return None

        element_type = self.convert_type_node_to_string(raw_type.element_type)
        access = None
        for qualifier in resource_qualifiers:
            normalized = normalized_image_access(qualifier)
            if normalized is not None:
                access = normalized
                break
        buffer_type = "StructuredBuffer" if access == "read" else "RWStructuredBuffer"
        return f"{buffer_type}<{element_type}>"

    def stage_entry_resource_parameter_type(self, param):
        constant_struct_type = self.stage_entry_constant_struct_parameter_type(param)
        if constant_struct_type is not None:
            return constant_struct_type
        storage_array_type = self.stage_entry_storage_array_structured_buffer_type(
            param
        )
        if storage_array_type is not None:
            return storage_array_type
        return self.stage_entry_metal_pointer_structured_buffer_type(param)

    def stage_entry_parameter_expression_type(self, param, raw_type):
        resource_type = self.stage_entry_resource_parameter_type(param)
        if resource_type is None:
            if self.is_constant_buffer_type(raw_type):
                return self.constant_buffer_element_type(raw_type)
            return self.type_name_string(raw_type)
        if self.is_structured_buffer_type(resource_type):
            element_type = self.structured_buffer_element_type(resource_type)
            if self.stage_entry_metal_scalar_reference_element_type(param) is not None:
                return element_type
            return f"{element_type}[]"
        if self.is_constant_buffer_type(resource_type):
            return self.constant_buffer_element_type(resource_type)
        return self.type_name_string(resource_type)

    def is_stage_entry_resource_parameter(self, param):
        raw_type = self.resource_node_type(param)
        if self.is_sampler_type(raw_type):
            return False
        if self.stage_entry_constant_struct_parameter_type(param) is not None:
            return True
        if self.is_constant_buffer_type(raw_type):
            return True
        resource_type = self.stage_entry_resource_parameter_type(param)
        if resource_type is not None:
            return True
        if self.is_glsl_buffer_block_variable(param, raw_type):
            return True
        if self.is_structured_buffer_type(raw_type):
            return True
        mapped_type = self.map_resource_type_with_format(
            self.resource_base_type(raw_type), param
        )
        return self.is_opaque_resource_type(mapped_type)

    def is_stage_entry_value_parameter(self, param):
        raw_type = self.resource_node_type(param)
        if self.is_sampler_type(raw_type):
            return False
        if self.is_stage_entry_resource_parameter(param):
            return False
        if self.stage_entry_constant_value_parameter_type(param) is not None:
            return False
        type_name = self.type_node_name(raw_type)
        return type_name not in self.structs_by_name

    def is_stage_builtin_semantic(self, semantic):
        mapped_semantic = self.map_semantic(semantic)
        return mapped_semantic.startswith("gl_") or semantic == "threads_per_grid"

    def stage_io_semantic_from_node(self, node):
        semantic = self.semantic_from_node(node)
        if semantic is not None:
            return semantic
        if id(node) in self.default_clip_position_member_ids:
            return "gl_Position"
        name = getattr(node, "name", None)
        mapped_name = self.semantic_map.get(name)
        if isinstance(mapped_name, str) and mapped_name.startswith("gl_"):
            return name
        return None

    def glsl_default_clip_position_member_ids(self):
        """Members that carry a vertex stage's clip position without a semantic (``clip_position_defaults``)."""
        member_ids = set()
        for struct_name in sorted(self.vertex_output_struct_names):
            struct = self.structs_by_name.get(struct_name)
            member = default_clip_position_member(
                getattr(struct, "members", None),
                has_semantic=lambda node: self.semantic_from_node(node) is not None
                or self.stage_io_semantic_from_node(node) is not None,
                declares_position=lambda node: self.map_semantic(
                    self.stage_io_semantic_from_node(node)
                )
                == "gl_Position",
                is_float4=lambda node: self.member_type_name(node) == "vec4",
            )
            if member is not None:
                member_ids.add(id(member))
        return member_ids

    def parameter_has_mapped_semantic(
        self, parameter, expected_semantic, stage_name=None
    ):
        semantic = self.semantic_from_node(parameter)
        if semantic is None:
            return False
        mapped = self.map_stage_input_semantic(semantic, stage_name)
        expected = self.map_stage_input_semantic(expected_semantic, stage_name)
        return mapped.lower() == expected.lower()

    def validate_exact_mapped_semantic_type(
        self, parameters, stage_name, semantic, expected_type, expected_description
    ):
        for parameter in parameters:
            if not self.parameter_has_mapped_semantic(parameter, semantic, stage_name):
                continue

            mapped_type = self.map_type(self.resource_node_type(parameter))
            base_type, array_suffix = split_array_type_suffix(str(mapped_type))
            if array_suffix or base_type != expected_type:
                if self.stage_builtin_parameter_local_alias_declaration(
                    parameter, stage_name
                ):
                    continue
                raise OpenGLRefusal(
                    f"OpenGL {stage_name} stage {semantic} "
                    f"parameter '{parameter.name}' "
                    f"must be {expected_description}"
                )

    def validate_compute_builtin_parameter_types(self, parameters):
        for semantic in (
            "gl_GlobalInvocationID",
            "gl_LocalInvocationID",
            "gl_WorkGroupID",
            "gl_WorkGroupSize",
            "gl_NumWorkGroups",
            "threads_per_grid",
        ):
            self.validate_exact_mapped_semantic_type(
                parameters, "compute", semantic, "uvec3", "uvec3"
            )
        for semantic in (
            "gl_LocalInvocationIndex",
            "gl_QuadGroupID",
            "gl_QuadGroupInvocationID",
        ):
            self.validate_exact_mapped_semantic_type(
                parameters,
                "compute",
                semantic,
                "uint",
                "scalar uint",
            )

    def validate_graphics_builtin_parameter_types(self, parameters, stage_name):
        for semantic in ("gl_VertexID", "gl_InstanceID", "gl_PrimitiveID"):
            self.validate_exact_mapped_semantic_type(
                parameters, stage_name, semantic, "int", "scalar int"
            )
        self.validate_exact_mapped_semantic_type(
            parameters, stage_name, "gl_FrontFacing", "bool", "scalar bool"
        )
        self.validate_exact_mapped_semantic_type(
            parameters, stage_name, "gl_FragCoord", "vec4", "vec4"
        )
        self.validate_exact_mapped_semantic_type(
            parameters, stage_name, "gl_PointCoord", "vec2", "vec2"
        )
        self.validate_exact_mapped_semantic_type(
            parameters, stage_name, "gl_SampleID", "int", "scalar int"
        )
        self.validate_exact_mapped_semantic_type(
            parameters, stage_name, "gl_SamplePosition", "vec2", "vec2"
        )
        self.validate_exact_mapped_semantic_type(
            parameters, stage_name, "gl_SampleMaskIn", "int", "scalar int"
        )

    def validate_function_return_semantic(self, func, stage_name):
        semantic = self.function_return_semantic(func)
        if semantic is None:
            return
        mapped_semantic = self.map_semantic(semantic)
        return_type = self.function_return_type(func)
        return_base_type, array_suffix = split_array_type_suffix(str(return_type))
        if not array_suffix and return_base_type == "void":
            function_name = getattr(func, "name", "<anonymous>")
            raise OpenGLRefusal(
                f"OpenGL {stage_name} function '{function_name}' cannot use "
                f"return semantic '{semantic}' with void return type"
            )

        self.validate_function_return_semantic_stage(stage_name, semantic)

        invalid_return_semantics = {
            "gl_VertexID",
            "gl_InstanceID",
            "gl_PrimitiveID",
            "gl_FrontFacing",
            "gl_FragCoord",
            "gl_PointCoord",
            "gl_SampleID",
            "gl_SamplePosition",
            "gl_SampleMaskIn",
            "gl_GlobalInvocationID",
            "gl_LocalInvocationID",
            "gl_WorkGroupID",
            "gl_LocalInvocationIndex",
            "gl_WorkGroupSize",
            "gl_NumWorkGroups",
        }
        if mapped_semantic in invalid_return_semantics:
            raise OpenGLRefusal(
                f"OpenGL {stage_name} semantic '{semantic}' cannot be used as "
                "a function return semantic"
            )

    def validate_function_return_semantic_stage(self, stage_name, semantic):
        mapped_semantic = self.map_semantic(semantic)
        semantic_text = str(semantic)
        is_fragment_output = mapped_semantic in {
            "gl_FragDepth",
            "gl_FragStencilRefARB",
            "gl_SampleMask",
        } or semantic_text.startswith("gl_FragColor")
        is_vertex_output = self.is_vertex_builtin_output(mapped_semantic)

        if stage_name == "vertex" and is_fragment_output:
            raise OpenGLRefusal(
                f"OpenGL vertex stage function return cannot use fragment output "
                f"semantic '{semantic}'"
            )
        if stage_name == "fragment" and is_vertex_output:
            raise OpenGLRefusal(
                f"OpenGL fragment stage function return cannot use vertex output "
                f"semantic '{semantic}'"
            )
        if stage_name == "compute" and (is_fragment_output or is_vertex_output):
            raise OpenGLRefusal(
                f"OpenGL compute stage function return cannot use graphics output "
                f"semantic '{semantic}'"
            )

    def generate_stage_parameter_input_declarations(self, ast, target_stage=None):
        declarations = []
        graphics_stages = {
            "vertex",
            "fragment",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "mesh",
            "task",
        }

        def is_input(param, stage_name):
            if stage_name == "mesh" and self.is_mesh_output_parameter(param):
                return False
            if self.is_graphics_stage_output_parameter(param, stage_name):
                return False
            if not self.is_stage_entry_value_parameter(param):
                return False
            return not self.is_stage_builtin_semantic(self.semantic_from_node(param))

        def add_parameter(param, stage_name, layouts=None, spirv_locations=None):
            if not is_input(param, stage_name):
                return
            patch_refusal = patch_parameter_refusal_reason(
                param.name, self.type_name_string(self.resource_node_type(param))
            )
            if patch_refusal is not None:
                self.refuse_lowering(patch_refusal)
                return

            param_type = self.map_type(self.resource_node_type(param))
            input_name = self.glsl_module_identifier_name(param.name)
            layout = (layouts or {}).get(id(param)) or self.stage_io_layout_for_node(
                param
            )
            self.reserve_stage_io_layout(
                self.stage_io_used_locations,
                stage_name,
                "input",
                layout,
                param_type,
                input_name,
            )
            prefix = self.stage_io_declaration_prefix(
                param, "in", layout=layout, stage_name=stage_name
            )
            if self.requires_flat_stage_input(stage_name, param_type):
                prefix = self.with_flat_stage_input_qualifier(prefix)
            parameter_declaration = format_c_style_array_declaration(
                param_type, input_name
            )
            declaration = spirv_location_prefixed(
                self,
                f"{prefix} {parameter_declaration};",
                (spirv_locations or {}).get(id(param)),
            )
            if self.reserve_stage_io_declaration(
                stage_name, "input", input_name, declaration
            ):
                if target_stage is None:
                    declarations.append(
                        self.wrap_stage_guard(f"{declaration}\n", stage_name)
                    )
                else:
                    declarations.append(declaration)

        def vertex_input_parameter_layouts(parameters, stage_name):
            if stage_name != "vertex":
                return {}
            return vertex_parameter_layouts(
                self, parameters, lambda param: is_input(param, stage_name)
            )

        def spirv_input_locations(parameters, stage_name):
            return spirv_entry_input_locations(
                self, parameters, stage_name, lambda param: is_input(param, stage_name)
            )

        for func in getattr(ast, "functions", []) or []:
            qualifier_name = function_stage_name(func)
            if qualifier_name not in graphics_stages:
                continue
            if not should_emit_qualified_function(target_stage, qualifier_name):
                continue
            parameters = getattr(func, "parameters", getattr(func, "params", [])) or []
            layouts = vertex_input_parameter_layouts(parameters, qualifier_name)
            locations = spirv_input_locations(parameters, qualifier_name)
            for param in parameters:
                add_parameter(param, qualifier_name, layouts, locations)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if stage_name not in graphics_stages:
                continue
            if not stage_matches(target_stage, stage_name):
                continue
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is None:
                continue
            parameters = (
                getattr(entry_point, "parameters", getattr(entry_point, "params", []))
                or []
            )
            layouts = vertex_input_parameter_layouts(parameters, stage_name)
            locations = spirv_input_locations(parameters, stage_name)
            for param in parameters:
                add_parameter(param, stage_name, layouts, locations)

        if target_stage is None:
            return "".join(declarations)
        return "\n".join(declarations) + ("\n" if declarations else "")

    def generate_stage_parameter_output_declarations(self, ast, target_stage=None):
        declarations = []
        graphics_stages = {"vertex", "fragment"}

        def is_vertex_user_output(param):
            # A vertex output parameter whose semantic is not a built-in (stage_output_parameter_name).
            if not self.is_graphics_stage_output_parameter(param, "vertex"):
                return False
            if not self.is_stage_entry_value_parameter(param):
                return False
            semantic = self.map_semantic(self.semantic_from_node(param))
            return not self.is_vertex_stage_builtin_output(semantic)

        def spirv_output_locations(func, parameters, stage_name):
            if stage_name != "vertex":
                return {}
            return spirv_entry_output_locations(
                self,
                func,
                [param for param in parameters if is_vertex_user_output(param)],
                stage_name,
            )

        def add_parameter(func, param, stage_name, spirv_locations=None):
            if not self.is_graphics_stage_output_parameter(param, stage_name):
                return
            if not self.is_stage_entry_value_parameter(param):
                return

            param_type = self.map_type(self.resource_node_type(param))
            output_name = self.stage_output_parameter_name(func, param, stage_name)
            self.stage_parameter_output_aliases.setdefault(id(func), {})[
                param.name
            ] = output_name

            if self.is_fragment_builtin_output_target(output_name):
                return
            if self.is_vertex_builtin_output(output_name):
                return

            layout = self.stage_output_parameter_layout(param, stage_name)
            if layout:
                self.reserve_stage_io_layout(
                    self.stage_io_used_locations,
                    stage_name,
                    "output",
                    layout,
                    param_type,
                    output_name,
                )
            prefix = self.stage_io_declaration_prefix(param, "out", layout)
            prefix = self.stage_io_prefix_with_required_flat(prefix, param_type)
            parameter_declaration = format_c_style_array_declaration(
                param_type, output_name
            )
            declaration = spirv_location_prefixed(
                self,
                f"{prefix} {parameter_declaration};",
                (spirv_locations or {}).get(id(param)),
            )
            if self.reserve_stage_io_declaration(
                stage_name, "output", output_name, declaration
            ):
                if target_stage is None:
                    declarations.append(
                        self.wrap_stage_guard(f"{declaration}\n", stage_name)
                    )
                else:
                    declarations.append(declaration)

        for func in getattr(ast, "functions", []) or []:
            qualifier_name = function_stage_name(func)
            if qualifier_name not in graphics_stages:
                continue
            if not should_emit_qualified_function(target_stage, qualifier_name):
                continue
            parameters = getattr(func, "parameters", getattr(func, "params", [])) or []
            locations = spirv_output_locations(func, parameters, qualifier_name)
            for param in parameters:
                add_parameter(func, param, qualifier_name, locations)

        for stage_type, stage in getattr(ast, "stages", {}).items():
            stage_name = normalize_stage_name(stage_type)
            if stage_name not in graphics_stages:
                continue
            if not stage_matches(target_stage, stage_name):
                continue
            entry_point = getattr(stage, "entry_point", None)
            if entry_point is None:
                continue
            parameters = (
                getattr(entry_point, "parameters", getattr(entry_point, "params", []))
                or []
            )
            locations = spirv_output_locations(entry_point, parameters, stage_name)
            for param in parameters:
                add_parameter(entry_point, param, stage_name, locations)

        if target_stage is None:
            return "".join(declarations)
        return "\n".join(declarations) + ("\n" if declarations else "")

    def stage_io_layout_location(self, layout):
        value = self.stage_io_layout_value(layout, "location")
        return int(value) if value is not None and value.isdigit() else None

    def stage_io_layout_value(self, layout, name):
        if not layout.startswith("layout("):
            return None
        start = layout.find("(")
        end = layout.rfind(")")
        if start == -1 or end == -1 or end <= start:
            return None

        for item in layout[start + 1 : end].split(","):
            key, separator, value = item.partition("=")
            if separator and key.strip() == name:
                return value.strip()
        return None

    def stage_io_layout_int_value(self, layout, name):
        value = self.stage_io_layout_value(layout, name)
        if value is None:
            return None
        return int(value) if value.isdigit() else None

    def stage_io_layout_component(self, layout):
        return self.stage_io_layout_int_value(layout, "component")

    def stage_io_layout_index(self, layout):
        index = self.stage_io_layout_int_value(layout, "index")
        return 0 if index is None else index

    def stage_io_location_count(self, mapped_type):
        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        array_count = self.stage_io_array_location_count(array_suffix)
        matrix_count = 1

        matrix_size = ""
        if base_type.startswith("dmat"):
            matrix_size = base_type[4:]
        elif base_type.startswith("mat"):
            matrix_size = base_type[3:]

        if matrix_size:
            columns = matrix_size.split("x", 1)[0]
            if columns.isdigit():
                matrix_count = int(columns)

        return max(matrix_count * array_count, 1)

    def stage_io_component_count(self, mapped_type):
        base_type, _ = split_array_type_suffix(str(mapped_type))
        normalized = str(base_type)
        if normalized in {
            "float",
            "int",
            "uint",
            "bool",
            "half",
            "short",
            "ushort",
        }:
            return 1
        if normalized == "double":
            return 2

        vector_prefix_widths = {
            "vec": 1,
            "ivec": 1,
            "uvec": 1,
            "bvec": 1,
            "dvec": 2,
        }
        for prefix, component_width in vector_prefix_widths.items():
            if normalized.startswith(prefix):
                size = normalized[len(prefix) :]
                if size.isdigit():
                    return int(size) * component_width
        return 4

    def stage_io_array_location_count(self, array_suffix):
        if not array_suffix:
            return 1

        count = 1
        index = 0
        while index < len(array_suffix):
            if array_suffix[index] != "[":
                return count
            close_index = array_suffix.find("]", index + 1)
            if close_index == -1:
                return count
            size = array_suffix[index + 1 : close_index].strip()
            if not size:
                index = close_index + 1
                continue
            try:
                dimension = int(size, 10)
            except ValueError:
                return count
            if dimension <= 0:
                return count
            count *= dimension
            index = close_index + 1
        return count

    def reserve_stage_io_layout(
        self, used_locations, stage_name, direction, layout, mapped_type, name
    ):
        start = self.stage_io_layout_location(layout)
        if start is None:
            return

        count = self.stage_io_location_count(mapped_type)
        end = start + count - 1
        component = self.stage_io_layout_component(layout)
        index = self.stage_io_layout_index(layout)
        if component is not None or self.stage_io_layout_has(layout, {"index"}):
            self.reserve_stage_io_component_layout(
                used_locations,
                stage_name,
                direction,
                start,
                end,
                component,
                index,
                mapped_type,
                name,
            )
            return

        ranges = used_locations.setdefault((stage_name, direction), [])
        for used_start, used_end, used_name in ranges:
            if start <= used_end and used_start <= end:
                if used_start == start and used_end == end and used_name == name:
                    return
                raise InterfaceLocationConflictError(
                    "opengl",
                    f"Conflicting OpenGL {stage_name} {direction} location "
                    f"for '{name}': {self.stage_io_location_label(start, end)} "
                    f"overlaps '{used_name}' "
                    f"{self.stage_io_location_label(used_start, used_end)}",
                )
        for (
            used_start,
            used_end,
            used_component_start,
            used_component_end,
            used_index,
            used_name,
        ) in self.stage_io_used_components.get((stage_name, direction), []):
            if used_index != 0:
                continue
            if start <= used_end and used_start <= end:
                raise InterfaceLocationConflictError(
                    "opengl",
                    f"Conflicting OpenGL {stage_name} {direction} location "
                    f"for '{name}': {self.stage_io_location_label(start, end)} "
                    f"overlaps '{used_name}' "
                    f"{self.stage_io_component_location_label(used_start, used_end, used_component_start, used_component_end)}",
                )
        ranges.append((start, end, name))

    def reserve_stage_io_component_layout(
        self,
        used_locations,
        stage_name,
        direction,
        start,
        end,
        component,
        index,
        mapped_type,
        name,
    ):
        component_start = 0 if component is None else component
        component_count = (
            4 if component is None else self.stage_io_component_count(mapped_type)
        )
        component_end = component_start + component_count - 1

        for used_start, used_end, used_name in used_locations.get(
            (stage_name, direction), []
        ):
            if start <= used_end and used_start <= end:
                if index != 0:
                    continue
                if used_start == start and used_end == end and used_name == name:
                    return
                raise InterfaceLocationConflictError(
                    "opengl",
                    f"Conflicting OpenGL {stage_name} {direction} location "
                    f"for '{name}': {self.stage_io_component_location_label(start, end, component_start, component_end)} "
                    f"overlaps '{used_name}' "
                    f"{self.stage_io_location_label(used_start, used_end)}",
                )

        entries = self.stage_io_used_components.setdefault((stage_name, direction), [])
        for (
            used_start,
            used_end,
            used_component_start,
            used_component_end,
            used_index,
            used_name,
        ) in entries:
            if start <= used_end and used_start <= end and index == used_index:
                if (
                    used_start == start
                    and used_end == end
                    and used_component_start == component_start
                    and used_component_end == component_end
                    and used_name == name
                ):
                    return
                if (
                    component_start <= used_component_end
                    and used_component_start <= component_end
                ):
                    raise InterfaceLocationConflictError(
                        "opengl",
                        f"Conflicting OpenGL {stage_name} {direction} location "
                        f"for '{name}': {self.stage_io_component_location_label(start, end, component_start, component_end)} "
                        f"overlaps '{used_name}' "
                        f"{self.stage_io_component_location_label(used_start, used_end, used_component_start, used_component_end)}",
                    )
        entries.append((start, end, component_start, component_end, index, name))

    def reserve_stage_io_declaration(self, stage_name, direction, name, declaration):
        key = (stage_name, direction, name)
        existing = self.stage_io_declarations.get(key)
        if existing is None:
            self.stage_io_declarations[key] = declaration
            return True
        if existing == declaration:
            return False
        raise InterfaceLocationConflictError(
            "opengl",
            f"Conflicting OpenGL {stage_name} {direction} declaration for "
            f"'{name}': {declaration} differs from {existing}",
        )

    def stage_io_declared_names(self, stage_name, direction):
        return {
            name
            for stage, used_direction, name in self.stage_io_declarations
            if stage == stage_name and used_direction == direction
        }

    def stage_io_location_label(self, start, end):
        if start == end:
            return f"location {start}"
        return f"locations {start}-{end}"

    def stage_io_component_label(self, start, end):
        if start == end:
            return f"component {start}"
        return f"components {start}-{end}"

    def stage_io_component_location_label(
        self, start, end, component_start, component_end
    ):
        return (
            f"{self.stage_io_location_label(start, end)} "
            f"{self.stage_io_component_label(component_start, component_end)}"
        )

    def struct_member_names(self, struct_names):
        names = set()
        for struct_name in struct_names:
            struct = self.structs_by_name.get(struct_name)
            for member in getattr(struct, "members", []) or []:
                names.add(member.name)
        return names

    def fragment_input_declaration_names(self, struct_names):
        names = set()
        for struct_name in struct_names:
            struct = self.structs_by_name.get(struct_name)
            for member in getattr(struct, "members", []) or []:
                input_name = self.fragment_input_member_name(member, struct_name)
                if input_name is not None:
                    names.add(input_name)
        return names

    def vertex_output_declaration_names(self):
        names = set()
        for struct_name in self.vertex_output_struct_names:
            struct = self.structs_by_name.get(struct_name)
            for member in getattr(struct, "members", []) or []:
                output_name = self.vertex_output_member_name(member)
                if self.is_vertex_stage_builtin_output(output_name):
                    continue
                if distance_private_output_declaration(self, member) is None:
                    names.add(output_name)
        return names

    def stage_value_parameter_input_names(self, ast, stage_name):
        names = set()
        for func in self.stage_functions(ast, stage_name):
            for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
                if not self.is_stage_entry_value_parameter(param):
                    continue
                if self.is_graphics_stage_output_parameter(param, stage_name):
                    continue
                if self.is_stage_builtin_semantic(self.semantic_from_node(param)):
                    continue
                names.add(param.name)
        return names

    def parameter_qualifier_names(self, node=None):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        qualifiers.update(
            str(getattr(attribute, "name", "")).lower()
            for attribute in getattr(node, "attributes", []) or []
        )
        qualifiers.discard("")
        return qualifiers

    def is_graphics_stage_output_parameter(self, param, stage_name):
        if normalize_stage_name(stage_name) not in {"vertex", "fragment"}:
            return False
        return bool(self.parameter_qualifier_names(param) & {"out", "inout"})

    def stage_output_parameter_name(self, func, param, stage_name):
        semantic = self.semantic_from_node(param)
        mapped_semantic = self.map_semantic(semantic)
        if normalize_stage_name(stage_name) == "vertex":
            if self.is_vertex_stage_builtin_output(mapped_semantic):
                note_vertex_builtin_output(self, mapped_semantic)
                return mapped_semantic
            reserved = set(self.vertex_input_member_names)
            reserved.update(self.stage_io_declared_names("vertex", "output"))
            return self.stage_io_name_avoiding_reserved(param.name, reserved)

        if self.is_fragment_builtin_output_target(mapped_semantic):
            return mapped_semantic
        if str(semantic or "").startswith("gl_FragColor"):
            return self.fragment_output_name(
                semantic, self.function_local_variable_names(func)
            )
        reserved = set(self.fragment_input_member_names)
        reserved.update(self.stage_io_declared_names("fragment", "output"))
        reserved.update(self.function_local_variable_names(func))
        return self.stage_io_name_avoiding_reserved(param.name, reserved)

    def stage_output_parameter_layout(self, param, stage_name):
        if normalize_stage_name(stage_name) != "fragment":
            return self.stage_io_layout_for_node(param)

        mapped_semantic = self.map_semantic(self.semantic_from_node(param))
        if mapped_semantic.startswith("layout("):
            return mapped_semantic
        layout = self.stage_io_layout_for_node(param)
        if layout.startswith("layout("):
            return layout
        return "layout(location = 0)"

    def stage_io_name_avoiding_reserved(self, preferred_name, reserved_names):
        if preferred_name in self.stage_io_reserved_value_names:
            return self.glsl_generated_module_identifier(
                ("stage-io", "reserved", str(preferred_name)),
                f"out_{self.glsl_sanitized_identifier_base(preferred_name)}",
            )
        emitted_name = self.glsl_module_identifier_name(preferred_name)
        emitted_reserved_names = {
            self.glsl_module_identifier_name(name)
            for name in reserved_names
            if name
            and name not in self.GLSL_BUILTIN_INTERFACE_BLOCK_NAMES
            and not str(name).startswith("gl_")
        }
        emitted_reserved_names.update(
            name for name in reserved_names if str(name).startswith("gl_")
        )
        if emitted_name not in emitted_reserved_names:
            return emitted_name

        return self.glsl_generated_module_identifier(
            ("stage-io", "reserved", str(preferred_name)),
            f"out_{emitted_name}",
        )

    def function_local_variable_names(self, func):
        body = getattr(func, "body", [])
        return {
            node.name
            for node in self.walk_ast(body)
            if isinstance(node, VariableNode) and getattr(node, "name", None)
        }

    def fragment_stage_local_variable_names(self, ast):
        names = set()
        for func in self.stage_functions(ast, "fragment"):
            names.update(self.function_local_variable_names(func))
        return names

    def fragment_output_member_maps(self, ast):
        maps = {}
        reserved_names = set(self.fragment_input_member_names)
        if self.current_target_stage is None:
            reserved_names.update(self.combined_vertex_output_member_names)
        reserved_names.update(self.fragment_stage_local_variable_names(ast))
        for struct_name, struct in self.structs_by_name.items():
            if struct_name not in self.fragment_output_struct_names:
                continue

            member_map = {}
            for member in getattr(struct, "members", []) or []:
                output_name = self.fragment_output_member_name(
                    member, reserved_names=reserved_names
                )
                member_map[member.name] = output_name
                if not self.is_fragment_builtin_output_target(output_name):
                    reserved_names.add(output_name)
            maps[struct_name] = member_map
        return maps

    def next_available_stage_io_layout(self, stage_name, direction, mapped_type):
        count = self.stage_io_location_count(mapped_type)
        ranges = sorted(self.stage_io_used_locations.get((stage_name, direction), []))
        ranges.extend(
            sorted(
                (start, end, name)
                for start, end, _, _, _, name in self.stage_io_used_components.get(
                    (stage_name, direction), []
                )
            )
        )
        location = 0

        while True:
            end = location + count - 1
            overlap = next(
                (
                    (used_start, used_end)
                    for used_start, used_end, _ in ranges
                    if location <= used_end and used_start <= end
                ),
                None,
            )
            if overlap is None:
                return f"layout(location = {location})"
            location = overlap[1] + 1

    def fragment_output_member_layout_maps_for_outputs(self):
        maps = {}

        for struct_name, struct in self.structs_by_name.items():
            if struct_name not in self.fragment_output_struct_names:
                continue

            member_layouts = {}
            for member in getattr(struct, "members", []) or []:
                output_name = self.fragment_output_member_name(member, struct_name)
                layout = self.stage_io_layout_for_node(member)
                if self.is_fragment_builtin_output_target(
                    layout
                ) or self.is_fragment_builtin_output_target(output_name):
                    member_layouts[member.name] = output_name
                    continue
                if not layout.startswith("layout("):
                    continue

                self.reserve_stage_io_layout(
                    self.stage_io_used_locations,
                    "fragment",
                    "output",
                    layout,
                    self.member_type_name(member),
                    output_name,
                )
                member_layouts[member.name] = layout

            maps[struct_name] = member_layouts

        for struct_name, struct in self.structs_by_name.items():
            if struct_name not in self.fragment_output_struct_names:
                continue

            member_layouts = maps.setdefault(struct_name, {})
            for member in getattr(struct, "members", []) or []:
                if member.name in member_layouts:
                    continue

                output_name = self.fragment_output_member_name(member, struct_name)
                layout = self.next_available_stage_io_layout(
                    "fragment", "output", self.member_type_name(member)
                )
                self.reserve_stage_io_layout(
                    self.stage_io_used_locations,
                    "fragment",
                    "output",
                    layout,
                    self.member_type_name(member),
                    output_name,
                )
                member_layouts[member.name] = layout

        return maps

    def type_node_name(self, type_node):
        if type_node is None:
            return None
        referenced_type = getattr(type_node, "referenced_type", None)
        if referenced_type is not None:
            return self.type_node_name(referenced_type)
        pointee_type = getattr(type_node, "pointee_type", None)
        if pointee_type is not None:
            return self.type_node_name(pointee_type)
        if hasattr(type_node, "name"):
            return type_node.name
        type_name = str(type_node)
        for suffix in ("&", "*"):
            if type_name.endswith(suffix):
                return type_name[: -len(suffix)].strip()
        return type_name

    def vertex_input_member_emitted_name(self, member):
        """A vertex input that would redeclare a global takes another name; its location still links it."""
        if member.name not in self.stage_io_reserved_value_names:
            return None
        return self.glsl_generated_module_identifier(
            ("stage-io", "vertex-input", member.name),
            f"in_{member.name}",
        )
