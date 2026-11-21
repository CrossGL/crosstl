"""Stage interfaces of the Metal generator: wave-lane context arguments, vertex and fragment built-in parameters, kernel derivatives, reachable functions and call overloads, SPIR-V stage interface layouts and semantics, vertex stage inputs, stage output parameters and returns, and return semantic wrappers."""

import re

from ..ast import (
    ArrayNode,
    FunctionCallNode,
    PointerType,
    ReferenceType,
)
from ..refusals import MetalRefusal
from .array_utils import (
    format_c_style_array_declaration,
    split_array_type_suffix,
)
from .fragment_output_targets import frag_data_attachment_index
from .metal_buffer_arrays import expand_entry_buffer_array
from .metal_entry_return import (
    metal_entry_return_attribute,
    metal_entry_return_semantic_key,
)
from .metal_exact_lookups import (
    metal_semantic_key,
    metal_spirv_builtin_semantic,
)
from .metal_fragment_outputs import metal_fragment_output_color_semantic
from .metal_invariant_position import metal_invariant_attribute_suffix
from .metal_raster_order_groups import metal_raster_order_group_attribute
from .metal_signed_builtins import metal_signed_builtin_parameter_name
from .metal_varyings import (
    is_metal_varying_builtin,
    metal_varying_attribute,
)


class MetalStageInterfaceMixin:
    """Stage interface members of ``MetalCodeGen``."""

    def required_metal_wave_lane_context_arguments(self, func_name):
        dependencies = self.function_metal_wave_lane_dependencies.get(func_name, set())
        if not dependencies:
            return []
        args = []
        if "WaveGetLaneIndex" in dependencies:
            if self.current_metal_wave_lane_index_parameter is None:
                return None
            args.append(self.current_metal_wave_lane_index_parameter)
        if "WaveGetLaneCount" in dependencies:
            if self.current_metal_wave_lane_count_parameter is None:
                return None
            args.append(self.current_metal_wave_lane_count_parameter)
        return args

    def metal_wave_lane_helper_call_diagnostic(self, func_name):
        dependencies = self.function_metal_wave_lane_dependencies.get(func_name, set())
        if not dependencies:
            return None
        missing_requirements = []
        if (
            "WaveGetLaneIndex" in dependencies
            and self.current_metal_wave_lane_index_parameter is None
        ):
            missing_requirements.append("thread_index_in_simdgroup")
        if (
            "WaveGetLaneCount" in dependencies
            and self.current_metal_wave_lane_count_parameter is None
        ):
            missing_requirements.append("threads_per_simdgroup")
        if not missing_requirements:
            return None
        requirement = " and ".join(missing_requirements)
        self.refuse_lowering(
            f"wave helper call: function '{func_name}' requires compute-stage "
            f"{requirement} value"
        )
        return_type = self.function_return_types.get(func_name, "void")
        if self.map_type(return_type) == "void":
            return "(void)0"
        return self.diagnostic_zero_value_for_type(return_type)

    def metal_vertex_builtin_parameter_specs(self):
        return [
            ("gl_VertexIndex", "uint", "vertex_id"),
            # Vulkan's InstanceIndex counts from the base instance, as MSL's instance_id does.
            ("gl_InstanceIndex", "uint", "instance_id"),
            ("gl_ViewIndex", "uint", "amplification_id"),
        ]

    def metal_fragment_builtin_parameter_specs(self):
        return [
            ("gl_FragCoord", "float4", "position"),
            ("gl_PointCoord", "float2", "point_coord"),
        ]

    def explicit_graphics_stage_builtin_parameters(self, parameters, stage_name):
        if stage_name == "vertex":
            builtin_specs = self.metal_vertex_builtin_parameter_specs()
        elif stage_name == "fragment":
            builtin_specs = self.metal_fragment_builtin_parameter_specs()
        else:
            return {}
        stage_parameters = {}
        builtin_attributes = {
            attribute for _builtin_name, _param_type, attribute in builtin_specs
        }
        for parameter in parameters or []:
            semantic = self.semantic_from_node(parameter)
            metal_semantic = self.canonical_metal_semantic(semantic)
            if metal_semantic in builtin_attributes:
                stage_parameters[metal_semantic] = getattr(parameter, "name", None)
        if stage_name == "fragment":
            # A [[stage_in]] structure member carrying a built-in, such as HLSL's SV_Position member, already
            # delivers it; a second [[position]] parameter would not compile.
            for parameter in parameters or []:
                param_type = getattr(parameter, "param_type", None)
                struct = self.structs_by_name.get(
                    str(getattr(param_type, "name", param_type))
                )
                for member in getattr(struct, "members", None) or []:
                    attribute = self.canonical_metal_semantic(
                        self.semantic_from_node(member)
                    )
                    if attribute in builtin_attributes:
                        stage_parameters.setdefault(
                            attribute, f"{parameter.name}.{member.name}"
                        )
        return stage_parameters

    def required_metal_vertex_builtin_parameters(
        self, func, reserved_names=None, explicit_stage_builtins=None
    ):
        used_names = self.used_metal_vertex_builtin_names(getattr(func, "body", []))
        reserved_names = set(reserved_names or ())
        explicit_stage_builtins = explicit_stage_builtins or {}
        required_parameters = []
        for (
            builtin_name,
            param_type,
            attribute,
        ) in self.metal_vertex_builtin_parameter_specs():
            if builtin_name not in used_names or attribute in explicit_stage_builtins:
                continue
            base_name = {
                "gl_ViewIndex": "_crossglViewIndex",
                "gl_InstanceIndex": "_crossglInstanceID",
            }.get(builtin_name, "_crossglVertexID")
            name = self.unique_metal_generated_name(base_name, reserved_names)
            reserved_names.add(name)
            required_parameters.append((builtin_name, name, param_type, attribute))
        return required_parameters

    def used_metal_vertex_builtin_names(self, body):
        builtin_names = {
            builtin
            for builtin, _type, _attribute in (
                self.metal_vertex_builtin_parameter_specs()
            )
        }
        used_names = set()
        for node in self.iter_ast_nodes(body):
            class_name = node.__class__.__name__
            if "Identifier" not in class_name:
                continue
            name = getattr(node, "name", "")
            base_name = name.split(".", 1)[0]
            if base_name in builtin_names:
                used_names.add(base_name)
        return used_names

    def required_metal_fragment_builtin_parameters(
        self, func, reserved_names=None, explicit_stage_builtins=None
    ):
        used_names = self.used_metal_fragment_builtin_names(getattr(func, "body", []))
        reserved_names = set(reserved_names or ())
        explicit_stage_builtins = explicit_stage_builtins or {}
        required_parameters = []
        for (
            builtin_name,
            param_type,
            attribute,
        ) in self.metal_fragment_builtin_parameter_specs():
            if builtin_name not in used_names or attribute in explicit_stage_builtins:
                continue
            # gl_FragCoord -> _crossglFragCoord, gl_PointCoord -> _crossglPointCoord.
            name = self.unique_metal_generated_name(
                "_crossgl" + builtin_name[len("gl_") :], reserved_names
            )
            reserved_names.add(name)
            required_parameters.append((builtin_name, name, param_type, attribute))
        return required_parameters

    def used_metal_fragment_builtin_names(self, body):
        builtin_names = {
            builtin_name
            for builtin_name, _type, _attribute in (
                self.metal_fragment_builtin_parameter_specs()
            )
        }
        used_names = set()
        for node in self.iter_ast_nodes(body):
            class_name = node.__class__.__name__
            if "Identifier" not in class_name:
                continue
            name = getattr(node, "name", "")
            base_name = name.split(".", 1)[0]
            if base_name in builtin_names:
                used_names.add(base_name)
        return used_names

    def metal_kernel_derivative_call_name(self, func):
        """Return the first derivative built-in that ``func`` calls, or ``None``."""
        derivative_names = {
            *self.METAL_DERIVATIVE_FUNCTION_ALIASES,
            *self.METAL_DERIVATIVE_FUNCTION_ALIASES.values(),
            "fwidth",
        }
        for node in self.metal_render_walks.body_nodes_of_type(func, FunctionCallNode):
            name = self.function_call_name(node)
            if name in derivative_names and name not in self.user_function_names:
                return name
        return None

    # The Metal function kind each non-fragment stage is written as. None of
    # them has derivatives.
    METAL_DERIVATIVE_FREE_FUNCTION_KINDS = {
        "compute": "kernel",
        "ray_generation": "kernel",
        "mesh": "mesh",
        "object": "object",
        "task": "object",
        "amplification": "object",
    }

    def metal_reachable_functions(self, root_func):
        """``root_func`` and every user function a call reaches from it, in breadth-first order.

        A call reaches the overloads of its name that accept its argument count; among several, the one whose
        parameter types match the known argument types, or all of them when the types do not decide (#3168).
        """
        reachable = []
        pending = [root_func]
        visited = set()
        while pending:
            func = pending.pop(0)
            if func is None or id(func) in visited:
                continue
            visited.add(id(func))
            reachable.append(func)
            for called_func in self.metal_called_user_function_overloads(func):
                if id(called_func) not in visited:
                    pending.append(called_func)
        return reachable

    def metal_called_user_function_overloads(self, func):
        """The user overloads that the calls in ``func`` can select, in a deterministic order."""
        selected = []
        seen = set()
        for node in self.metal_render_walks.body_nodes_of_type(func, FunctionCallNode):
            name = self.function_call_name(node)
            if name not in self.user_function_names or name == getattr(
                func, "name", None
            ):
                continue
            arguments = list(
                getattr(node, "args", getattr(node, "arguments", [])) or []
            )
            for candidate in self.metal_call_overload_candidates(name, arguments):
                if id(candidate) not in seen:
                    seen.add(id(candidate))
                    selected.append(candidate)
        return selected

    def metal_call_overload_candidates(self, name, arguments):
        overloads = list(self.function_overloads_by_name.get(name, []))
        if not overloads:
            single = (self.functions_by_name or {}).get(name)
            return [single] if single is not None else []
        by_arity = [
            overload
            for overload in overloads
            if self.metal_user_function_accepts_argument_count(overload, len(arguments))
        ] or overloads
        if len(by_arity) == 1:
            return by_arity
        argument_types = [
            self.map_type(argument_type) if argument_type is not None else None
            for argument_type in (
                self.expression_result_type(argument) for argument in arguments
            )
        ]
        by_type = [
            overload
            for overload in by_arity
            if self.metal_overload_matches_argument_types(overload, argument_types)
        ]
        return by_type if len(by_type) == 1 else by_arity

    def metal_overload_matches_argument_types(self, overload, argument_types):
        parameters = list(
            getattr(overload, "parameters", getattr(overload, "params", [])) or []
        )
        for parameter, argument_type in zip(parameters, argument_types):
            if argument_type is None:
                return False
            parameter_type = getattr(
                parameter, "param_type", getattr(parameter, "vtype", None)
            )
            if parameter_type is None or self.map_type(parameter_type) != argument_type:
                return False
        return True

    def function_uses_metal_fragment_invocation_density(self, func):
        for node in self.iter_ast_nodes(getattr(func, "body", []) or []):
            class_name = node.__class__.__name__
            if "Identifier" not in class_name and class_name != "VariableNode":
                continue
            name = getattr(node, "name", "")
            if name.split(".", 1)[0] == "gl_FragSizeEXT":
                return True
        return False

    def validate_graphics_builtin_parameter_types(self, parameters, stage_name):
        expected_types = {
            "vertex_id": "uint",
            "instance_id": "uint",
            "base_vertex": "uint",
            "base_instance": "uint",
            "primitive_id": "uint",
            "front_facing": "bool",
            "render_target_array_index": "uint",
            "viewport_array_index": "uint",
            "amplification_id": "uint",
            "position": "float4",
            "point_coord": "float2",
            "sample_id": "uint",
            "sample_mask": "uint",
        }
        for parameter in parameters or []:
            if self.is_graphics_stage_output_parameter(parameter, stage_name):
                continue
            semantic = self.semantic_from_node(parameter)
            if stage_name == "fragment" and str(semantic) == "gl_SampleMask":
                raise MetalRefusal(
                    "Metal fragment stage gl_SampleMask parameter is output-only; "
                    "use gl_SampleMaskIn or sample_mask for coverage-mask input"
                )
            metal_semantic = self.canonical_metal_semantic(semantic)
            expected_type = expected_types.get(metal_semantic)
            if expected_type is None:
                continue
            actual_type = self.map_type(self.parameter_raw_type(parameter))
            if self.metal_graphics_builtin_parameter_lowering(
                self.parameter_raw_type(parameter),
                actual_type,
                parameter,
                stage_name,
            ):
                continue
            if actual_type != expected_type:
                name = getattr(parameter, "name", "<anonymous>")
                raise MetalRefusal(
                    f"Metal {stage_name} semantic '{semantic}' maps to "
                    f"'{metal_semantic}' and requires parameter '{name}' to "
                    f"have type {expected_type}, got {actual_type}"
                )

    def is_graphics_stage_output_parameter(self, parameter, stage_name):
        if stage_name not in {"vertex", "fragment"}:
            return False
        if self.is_metal_mesh_output_parameter(stage_name, parameter):
            return False
        if not (self.parameter_qualifier_names(parameter) & {"out", "inout"}):
            return False
        return not self.is_resource_parameter_type(self.parameter_raw_type(parameter))

    def is_spirv_stage_interface_layout(self, node):
        return (
            getattr(node, "layout_type", None) in {"IN", "OUT"}
            and getattr(node, "data_type", None) is not None
            and getattr(node, "variable_name", None) is not None
        )

    def spirv_stage_input_layouts(self, shader_type):
        if shader_type not in {"vertex", "fragment"}:
            return []
        return [
            node
            for node in getattr(self, "metal_spirv_interface_variables", []) or []
            if getattr(node, "layout_type", None) == "IN"
        ]

    def spirv_stage_output_layouts(self, shader_type):
        if shader_type not in {"vertex", "fragment"}:
            return []
        return [
            node
            for node in getattr(self, "metal_spirv_interface_variables", []) or []
            if getattr(node, "layout_type", None) == "OUT"
            and self.spirv_stage_output_semantic(node) is not None
        ]

    def spirv_stage_layout_qualifier_value(self, node, qualifier_name):
        for name, value in getattr(node, "qualifiers", []) or []:
            if str(name).lower() == qualifier_name:
                return value
        return None

    def spirv_stage_input_semantic(self, node):
        location = self.spirv_stage_layout_qualifier_value(node, "location")
        if location is not None:
            return f"attribute({location})"
        builtin = self.spirv_stage_layout_qualifier_value(node, "builtin")
        return self.spirv_builtin_semantic(builtin)

    def spirv_stage_output_semantic(self, node):
        builtin = self.spirv_stage_layout_qualifier_value(node, "builtin")
        if builtin is not None:
            return self.spirv_builtin_semantic(builtin)
        location = self.spirv_stage_layout_qualifier_value(node, "location")
        if location is not None:
            return f"color({location})"
        return None

    def spirv_stage_input_attribute(self, node):
        semantic = self.spirv_stage_input_semantic(node)
        return self.map_semantic(semantic) if semantic is not None else None

    def spirv_stage_output_attribute(self, node):
        semantic = self.spirv_stage_output_semantic(node)
        return self.map_semantic(semantic) if semantic is not None else ""

    def spirv_builtin_semantic(self, builtin):
        return metal_spirv_builtin_semantic(self, builtin)

    def metal_graphics_builtin_parameter_lowering(
        self,
        raw_param_type,
        mapped_param_type,
        parameter,
        stage_name,
        reserved_names=None,
    ):
        metal_semantic = self.canonical_metal_semantic(
            self.semantic_from_node(parameter)
        )
        base_name = metal_signed_builtin_parameter_name(
            stage_name, metal_semantic, mapped_param_type
        )
        if base_name is None:
            return None
        return {
            "type": "uint",
            "name": self.unique_metal_generated_name(base_name, reserved_names or ()),
            "attribute": metal_semantic,
            "source_type": raw_param_type,
        }

    def is_plain_metal_vertex_stage_input_parameter(
        self, raw_param_type, mapped_param_type, parameter, shader_type
    ):
        if shader_type != "vertex":
            return False
        if self.is_graphics_stage_output_parameter(parameter, shader_type):
            return False
        if (
            self.semantic_from_node(parameter) is not None
            and self.metal_vertex_input_location_semantic(parameter) is None
        ):
            return False
        if self.is_resource_parameter_type(raw_param_type):
            return False
        if isinstance(raw_param_type, (PointerType, ReferenceType)):
            return False
        if self.metal_parameter_user_struct_type(parameter) is not None:
            return False

        mapped_type = self.type_name_string(mapped_param_type)
        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        if array_suffix:
            return False
        if self.metal_matrix_dimensions(base_type) is not None:
            return False
        if base_type in self.structs_by_name:
            return False
        return self.is_metal_scalar_or_vector_type_name(base_type)

    def is_plain_metal_fragment_stage_input_parameter(
        self, raw_param_type, mapped_param_type, parameter, shader_type
    ):
        if shader_type != "fragment":
            return False
        if self.is_graphics_stage_output_parameter(parameter, shader_type):
            return False
        semantic = self.semantic_from_node(parameter)
        if semantic is not None and is_metal_varying_builtin(
            self.canonical_metal_semantic(semantic)
        ):
            return False
        if self.is_metal_resource_or_buffer_parameter(
            raw_param_type, parameter, shader_type
        ):
            return False
        if isinstance(raw_param_type, (PointerType, ReferenceType)):
            return False
        if self.metal_parameter_user_struct_type(parameter) is not None:
            return False

        mapped_type = self.type_name_string(mapped_param_type)
        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        if array_suffix:
            return False
        if self.metal_matrix_dimensions(base_type) is not None:
            return False
        if base_type in self.structs_by_name:
            return False
        return self.is_metal_scalar_or_vector_type_name(base_type)

    def is_metal_scalar_or_vector_type_name(self, type_name):
        type_name = str(type_name).strip()
        return (
            re.fullmatch(
                r"(?:packed_)?(?:half|float|double|bool|char|uchar|short|ushort|"
                r"int|uint|long|ulong)(?:[234])?",
                type_name,
            )
            is not None
        )

    def unique_vertex_stage_input_struct_name(self, function_name):
        base_name = f"{function_name}_Input"
        if base_name[:1].isdigit():
            base_name = f"Generated_{base_name}"
        used_names = set(self.structs_by_name)
        used_names.update(self.generated_return_wrapper_struct_names)
        used_names.update(getattr(self, "metal_generated_struct_names", set()))
        candidate = base_name
        suffix = 2
        while candidate in used_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        self.generated_return_wrapper_struct_names.add(candidate)
        return candidate

    def generate_metal_vertex_stage_input_parameter_struct(
        self, struct_name, parameters
    ):
        code = f"struct {struct_name} {{\n"
        used_attributes = {
            attribute_index
            for parameter in parameters
            for attribute_index in [
                self.metal_attribute_index_from_attribute(parameter.get("attribute"))
            ]
            if attribute_index is not None
        }
        next_attribute = 0
        for parameter in parameters:
            attribute = parameter.get("attribute")
            attribute_index = self.metal_attribute_index_from_attribute(attribute)
            if (
                attribute_index is None and not parameter.get("preserve_user_attribute")
            ) and (
                attribute is None
                or not attribute.strip()
                or (
                    self.is_metal_user_stage_io_semantic(parameter.get("semantic"))
                    and not parameter.get("preserve_user_attribute")
                )
            ):
                while next_attribute in used_attributes:
                    next_attribute += 1
                attribute = f" [[attribute({next_attribute})]]"
                used_attributes.add(next_attribute)
                next_attribute += 1
            if "declaration" in parameter:
                code += f"    {parameter['declaration']}{attribute or ''};\n"
            else:
                field_decl = format_c_style_array_declaration(
                    parameter["mapped_type"], parameter["name"]
                )
                code += f"    {field_decl}{attribute or ''};\n"
        code += "};\n\n"
        return code

    def generate_metal_vertex_stage_input_alias_declarations(
        self, input_parameter_name, parameters
    ):
        declarations = []
        for parameter in parameters:
            declaration = parameter.get("declaration")
            if declaration is None:
                declaration = format_c_style_array_declaration(
                    parameter["mapped_type"],
                    self.metal_local_identifier_name(parameter["name"]),
                )
            declarations.append(
                f"{declaration} = {input_parameter_name}.{parameter['name']};"
            )
        return declarations

    def metal_stage_output_parameter_attribute(self, parameter, stage_name):
        if stage_name == "fragment":
            color = metal_fragment_output_color_semantic(self, parameter)
            if color is not None:
                return f" [[{color}]]"
        semantic = self.semantic_from_node(parameter)
        if semantic is None and stage_name == "fragment":
            semantic = "gl_FragColor"
        if stage_name == "vertex":
            varying = metal_varying_attribute(self, parameter, semantic)
            if varying is not None:
                return varying
        return self.map_semantic(semantic)

    def generate_metal_stage_output_parameter_struct(self, struct_name, parameters):
        code = f"struct {struct_name} {{\n"
        for parameter in parameters:
            field_decl = format_c_style_array_declaration(
                parameter["mapped_type"], parameter["name"]
            )
            code += f"    {field_decl}{parameter['attribute']};\n"
        code += "};\n\n"
        return code

    def generate_metal_stage_output_parameter_locals(self, parameters):
        code = ""
        for parameter in parameters:
            if parameter.get("return_value"):
                continue
            default_value = self.metal_default_value_expression(parameter["raw_type"])
            declaration = format_c_style_array_declaration(
                parameter["mapped_type"],
                self.metal_local_identifier_name(parameter["name"]),
            )
            code += f"    {declaration} = {default_value};\n"
        return code

    def generate_metal_stage_output_parameter_return(
        self, struct_name, parameters, reserved_names
    ):
        if not parameters or not struct_name:
            return ""
        state = self.metal_stage_output_return_state(
            struct_name, parameters, reserved_names
        )
        return self.generate_metal_stage_output_return(state, "    ")

    def metal_stage_output_return_state(self, struct_name, parameters, reserved_names):
        """What a ``return`` of an entry point with stage output parameters builds."""
        return {
            "struct_name": struct_name,
            "parameters": parameters,
            "result_name": self.unique_metal_generated_name(
                "_crossglOutput", reserved_names
            ),
        }

    def generate_metal_stage_output_return(self, state, indent_str, value=None):
        """Lines that fill the stage output structure from the output parameters and return it.

        ``value`` is the generated return value of an entry point that also returns a value; it is stored
        first, so the output parameters are read after it has been evaluated.
        """
        result_name = state["result_name"]
        code = f"{indent_str}{state['struct_name']} {result_name};\n"
        for parameter in state["parameters"]:
            if parameter.get("return_value"):
                if value is not None:
                    code += (
                        f"{indent_str}{result_name}.{parameter['name']} = {value};\n"
                    )
        for parameter in state["parameters"]:
            if parameter.get("return_value"):
                continue
            local_name = self.metal_local_identifier_name(parameter["name"])
            code += f"{indent_str}{result_name}.{parameter['name']} = {local_name};\n"
        code += f"{indent_str}return {result_name};\n"
        return code

    def generate_metal_stage_output_early_return(self, state, indent_str, value=None):
        """A ``return`` inside the body of an entry point with stage output parameters.

        The braces scope the result variable, so several returns and the closing one do not collide.
        """
        inner = self.generate_metal_stage_output_return(state, indent_str + "  ", value)
        return f"{indent_str}{{\n{inner}{indent_str}}}\n"

    def metal_stage_return_value_output(
        self, func, shader_type, raw_return_type, return_type, parameters
    ):
        """The member of the stage output structure that holds the entry point's return value.

        HLSL entry points can return one output and write others through ``out`` parameters
        (``float4 main(out float depth : SV_Depth) : SV_Target``). MSL returns every stage output in one
        structure, so the return value becomes a member with the return semantic's attribute.
        """
        function_name = getattr(func, "name", "<anonymous>")
        if str(raw_return_type) in self.structs_by_name:
            raise MetalRefusal(
                f"Metal {shader_type} function '{function_name}' cannot combine "
                "output parameters with a structure return type; MSL stage outputs "
                "are the members of one returned structure"
            )
        semantic = self.function_return_semantic(func)
        attribute = None
        if shader_type == "fragment":
            location = frag_data_attachment_index(func)
            if location is not None:
                attribute = f" [[color({location})]]"
        if attribute is None and semantic is not None and shader_type == "vertex":
            attribute = metal_varying_attribute(self, func, semantic)
        if attribute is None and semantic is not None:
            attribute = self.map_semantic(semantic)
        if not attribute:
            attribute = " [[position]]" if shader_type == "vertex" else " [[color(0)]]"
        used = {parameter["name"] for parameter in parameters}
        name = "_crossglReturn"
        suffix = 2
        while name in used:
            name = f"_crossglReturn{suffix}"
            suffix += 1
        return {
            "name": name,
            "raw_type": raw_return_type,
            "mapped_type": return_type,
            "semantic": semantic,
            "attribute": attribute,
            "return_value": True,
        }

    def struct_member_builtin_semantic_type(self, metal_semantic):
        fixed_types = {
            "vertex_id": "uint",
            "instance_id": "uint",
            "primitive_id": "uint",
            "front_facing": "bool",
            "render_target_array_index": "uint",
            "viewport_array_index": "uint",
            "amplification_id": "uint",
            "position": "float4",
            "point_coord": "float2",
            "point_size": "float",
            "depth(any)": "float",
            "sample_id": "uint",
            "sample_mask": "uint",
            # A fragment function's stencil reference output (MSL 3.2 section 5.2.3.4).
            "stencil": "uint",
        }
        if metal_semantic is None:
            return None
        if metal_semantic.startswith("color("):
            return "float4"
        return fixed_types.get(metal_semantic)

    def struct_member_raw_type(self, member):
        if isinstance(member, ArrayNode):
            element_type = getattr(
                member, "element_type", getattr(member, "vtype", "float")
            )
            raw_type = self.type_name_string(element_type) or "float"
            if getattr(member, "size", None) is None:
                return f"{raw_type}[]"
            return f"{raw_type}[{member.size}]"
        if hasattr(member, "member_type"):
            return self.type_name_string(member.member_type)
        if hasattr(member, "vtype"):
            return self.type_name_string(member.vtype)
        return "float"

    def function_return_semantic_wrapper(
        self, func, raw_return_type, mapped_return_type, stage_name, function_name
    ):
        semantic = self.function_return_semantic(func)
        metal_semantic = metal_entry_return_semantic_key(
            self, func, semantic, stage_name
        )
        # MSL has no attribute on a return value, so an invariant position is returned through a structure.
        invariant_attr = metal_invariant_attribute_suffix(self, func, semantic)
        if not invariant_attr and not self.function_return_semantic_requires_wrapper(
            semantic, metal_semantic
        ):
            return None
        expected_type = self.function_return_builtin_semantic_type(
            semantic, metal_semantic
        )
        if expected_type is None or expected_type == "invalid":
            return None
        if mapped_return_type != expected_type:
            return None
        struct_name = self.unique_return_wrapper_struct_name(function_name)
        return {
            "struct_name": struct_name,
            "field_type": mapped_return_type,
            "field_name": self.return_wrapper_field_name(semantic, metal_semantic),
            "source_type": raw_return_type,
            "semantic_attr": (
                metal_entry_return_attribute(self, semantic, metal_semantic)
                + invariant_attr
            ),
        }

    def function_return_semantic_requires_wrapper(self, semantic, metal_semantic):
        if semantic is None:
            return False
        semantic = str(semantic)
        if semantic == "gl_FragDepth":
            return True
        if semantic == "gl_FragStencilRefEXT":
            return True
        if metal_semantic == "sample_mask":
            return True
        return bool(
            metal_semantic
            and metal_semantic.startswith("color(")
            and metal_semantic != "color(0)"
        )

    def unique_return_wrapper_struct_name(self, function_name):
        base_name = f"{function_name}_Return"
        if base_name[:1].isdigit():
            base_name = f"Generated_{base_name}"
        used_names = set(self.structs_by_name)
        used_names.update(self.generated_return_wrapper_struct_names)
        candidate = base_name
        suffix = 2
        while candidate in used_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        self.generated_return_wrapper_struct_names.add(candidate)
        return candidate

    def return_wrapper_field_name(self, semantic, metal_semantic):
        semantic = str(semantic)
        if semantic == "gl_FragDepth":
            return "depth"
        if semantic == "gl_PointSize":
            return "pointSize"
        if metal_semantic == "sample_mask":
            return "sampleMask"
        if metal_semantic and metal_semantic.startswith("color("):
            return "color"
        return "value"

    def generate_return_wrapper_struct(self, wrapper):
        return (
            f"struct {wrapper['struct_name']} {{\n"
            f"    {wrapper['field_type']} {wrapper['field_name']}"
            f"{wrapper['semantic_attr']};\n"
            "};\n\n"
        )

    def validate_function_return_semantic_stage(
        self, stage_name, semantic, metal_semantic
    ):
        if semantic is None:
            return

        semantic_text = str(semantic)
        is_fragment_output = semantic_text in {
            "gl_FragDepth",
            "gl_FragStencilRefEXT",
        } or bool(metal_semantic and metal_semantic.startswith("color("))
        is_fragment_output = is_fragment_output or metal_semantic in {
            "sample_mask",
            "stencil",
        }
        is_vertex_output = semantic_text == "gl_Position"

        if stage_name == "vertex" and is_fragment_output:
            raise MetalRefusal(
                f"Metal vertex stage function return cannot use fragment output "
                f"semantic '{semantic}'"
            )
        if stage_name == "fragment" and is_vertex_output:
            raise MetalRefusal(
                f"Metal fragment stage function return cannot use vertex output "
                f"semantic '{semantic}'"
            )

    def function_return_builtin_semantic_type(self, semantic, metal_semantic):
        if semantic is None:
            return None
        semantic = str(semantic)
        invalid_return_semantics = {
            "gl_VertexID",
            "gl_InstanceID",
            "gl_PrimitiveID",
            "gl_IsFrontFace",
            "gl_PointSize",
            "gl_FragCoord",
            "gl_FrontFacing",
            "gl_PointCoord",
            "gl_SampleID",
            "gl_SampleMaskIn",
            "gl_GlobalInvocationID",
            "gl_LocalInvocationID",
            "gl_WorkGroupID",
            "gl_LocalInvocationIndex",
            "gl_WorkGroupSize",
            "gl_NumWorkGroups",
        }
        if semantic in invalid_return_semantics:
            return "invalid"
        output_types = {
            "gl_Position": "float4",
            "gl_FragDepth": "float",
            "gl_SampleMask": "uint",
            "gl_FragStencilRefEXT": "uint",
        }
        if semantic in output_types:
            return output_types[semantic]
        if metal_semantic and metal_semantic.startswith("color("):
            return "float4"
        if metal_semantic == "sample_mask":
            return "uint"
        return None

    def canonical_metal_semantic(self, semantic):
        """The key a semantic is classified by; see :func:`.metal_exact_lookups.metal_semantic_key`."""
        return metal_semantic_key(self, semantic)

    def append_global_resource_parameters(
        self,
        params_str,
        func_name=None,
        func=None,
        filter_writable_textures=False,
        shader_type=None,
    ):
        resource_params = []

        def ordered(variable, declaration):
            return metal_raster_order_group_attribute(
                variable, shader_type, declaration
            )

        dependencies = None
        entry_point_dependencies = (
            self.entry_point_global_resource_dependencies(func)
            if func is not None
            else None
        )
        texture_dependencies = (
            entry_point_dependencies if filter_writable_textures else None
        )
        sampler_dependencies = (
            entry_point_dependencies
            if func is not None
            else (
                self.function_global_resource_dependencies.get(func_name, set())
                if func_name
                else None
            )
        )
        cbuffer_dependencies = None
        if self.cbuffer_variables:
            for cbuffer in self.cbuffer_variables:
                cbuffer_name = getattr(cbuffer, "name", None)
                if (
                    cbuffer_dependencies is not None
                    and cbuffer_name not in cbuffer_dependencies
                ):
                    continue
                binding = self.cbuffer_binding_indices.get(id(cbuffer), 0)
                parameter_name = self.cbuffer_parameter_name(cbuffer)
                resource_params.append(
                    f"constant {cbuffer.name}& {parameter_name} [[buffer({binding})]]"
                )
        if self.metal_buffer_resource_variables:
            for (
                buffer_variable,
                i,
                buffer_type,
                array_size,
                address_space,
            ) in self.metal_buffer_resource_variables:
                buffer_name = getattr(buffer_variable, "name", None)
                if dependencies is not None and buffer_name not in dependencies:
                    continue
                declaration = self.format_metal_buffer_resource_parameter(
                    buffer_type,
                    buffer_name,
                    array_size,
                    address_space,
                    buffer_variable,
                )
                resource_params.append(
                    f"{declaration} [[buffer({i})]]"
                    f"{ordered(buffer_variable, declaration)}"
                )
        if self.texture_variables:
            for (
                texture_variable,
                i,
                texture_type,
                array_size,
            ) in self.texture_variables:
                texture_name = getattr(texture_variable, "name", None)
                if (
                    texture_dependencies is not None
                    and texture_name not in texture_dependencies
                    and self.metal_texture_parameter_type_is_writable(texture_type)
                ):
                    continue
                declaration = self.format_resource_parameter(
                    self.metal_coherent_texture_type(texture_type, None, texture_name),
                    texture_name,
                    array_size,
                )
                resource_params.append(
                    f"{declaration} [[texture({i})]]"
                    f"{ordered(texture_variable, declaration)}"
                )
        if self.acceleration_structure_variables:
            for (
                acceleration_structure_variable,
                i,
                acceleration_structure_type,
                array_size,
            ) in self.acceleration_structure_variables:
                acceleration_structure_name = getattr(
                    acceleration_structure_variable, "name", None
                )
                if (
                    dependencies is not None
                    and acceleration_structure_name not in dependencies
                ):
                    continue
                declaration = self.format_resource_parameter(
                    acceleration_structure_type,
                    acceleration_structure_name,
                    array_size,
                )
                resource_params.append(f"{declaration} [[buffer({i})]]")
        if self.visible_function_table_variables:
            for (
                visible_function_table_variable,
                i,
                visible_function_table_type,
                array_size,
            ) in self.visible_function_table_variables:
                visible_function_table_name = getattr(
                    visible_function_table_variable, "name", None
                )
                if (
                    dependencies is not None
                    and visible_function_table_name not in dependencies
                ):
                    continue
                declaration = self.format_visible_function_table_parameter(
                    visible_function_table_type, visible_function_table_name, array_size
                )
                resource_params.append(f"{declaration} [[buffer({i})]]")
        if self.intersection_function_table_variables:
            for (
                intersection_function_table_variable,
                i,
                intersection_function_table_type,
                array_size,
            ) in self.intersection_function_table_variables:
                intersection_function_table_name = getattr(
                    intersection_function_table_variable, "name", None
                )
                if (
                    dependencies is not None
                    and intersection_function_table_name not in dependencies
                ):
                    continue
                declaration = self.format_intersection_function_table_parameter(
                    intersection_function_table_type,
                    intersection_function_table_name,
                    array_size,
                )
                resource_params.append(f"{declaration} [[buffer({i})]]")
        if self.structured_buffer_variables:
            for (
                buffer_variable,
                i,
                buffer_type,
                array_size,
            ) in self.structured_buffer_variables:
                buffer_name = getattr(buffer_variable, "name", None)
                if dependencies is not None and buffer_name not in dependencies:
                    continue
                declaration = self.format_structured_buffer_parameter(
                    buffer_type, buffer_name, array_size, buffer_variable
                )
                resource_params.extend(
                    parameter + ordered(buffer_variable, declaration)
                    for parameter in expand_entry_buffer_array(
                        self,
                        declaration,
                        i,
                        self.metal_entry_buffer_array_declarations,
                    )
                )
        if self.structured_buffer_length_variables:
            for (
                buffer_variable,
                i,
                _buffer_type,
                array_size,
            ) in self.structured_buffer_length_variables:
                buffer_name = getattr(buffer_variable, "name", None)
                if dependencies is not None and buffer_name not in dependencies:
                    continue
                declaration = self.format_structured_buffer_length_parameter(
                    buffer_name, array_size
                )
                resource_params.extend(
                    expand_entry_buffer_array(
                        self,
                        declaration,
                        i,
                        self.metal_entry_buffer_array_declarations,
                    )
                )
        if self.structured_buffer_counter_variables:
            for (
                buffer_variable,
                i,
                _buffer_type,
                array_size,
            ) in self.structured_buffer_counter_variables:
                buffer_name = getattr(buffer_variable, "name", None)
                if dependencies is not None and buffer_name not in dependencies:
                    continue
                declaration = self.format_structured_buffer_counter_parameter(
                    buffer_name, array_size
                )
                resource_params.extend(
                    expand_entry_buffer_array(
                        self,
                        declaration,
                        i,
                        self.metal_entry_buffer_array_declarations,
                    )
                )
        if self.glsl_buffer_block_variables:
            for (
                buffer_variable,
                i,
                block,
                array_size,
            ) in self.glsl_buffer_block_variables:
                buffer_name = getattr(buffer_variable, "name", None)
                if dependencies is not None and buffer_name not in dependencies:
                    continue
                declaration = self.format_glsl_buffer_block_parameter(
                    block, buffer_name, array_size
                )
                resource_params.extend(
                    parameter + ordered(buffer_variable, declaration)
                    for parameter in expand_entry_buffer_array(
                        self,
                        declaration,
                        i,
                        self.metal_entry_buffer_array_declarations,
                    )
                )
        if self.sampler_variables:
            for sampler_variable, i, array_size in self.sampler_variables:
                sampler_name = getattr(sampler_variable, "name", None)
                if (
                    sampler_dependencies is not None
                    and sampler_name not in sampler_dependencies
                ):
                    continue
                declaration = self.format_resource_parameter(
                    "sampler", sampler_name, array_size
                )
                resource_params.append(f"{declaration} [[sampler({i})]]")
        if not resource_params:
            return params_str
        if params_str:
            return f"{params_str}, {', '.join(resource_params)}"
        return ", ".join(resource_params)

    def entry_point_global_resource_dependencies(self, func):
        dependencies = set(self.direct_global_resource_dependencies(func))
        for called_name in self.called_user_function_names(func):
            dependencies.update(
                self.function_global_resource_dependencies.get(called_name, set())
            )
        return dependencies

    def entry_point_cbuffer_dependencies(self, func):
        dependencies = set(self.direct_cbuffer_dependencies(func))
        for called_name in self.called_user_function_names(func):
            dependencies.update(
                self.function_cbuffer_dependencies.get(called_name, set())
            )
        return dependencies

    def metal_texture_parameter_type_is_writable(self, texture_type):
        texture_type = str(texture_type or "")
        return "access::write" in texture_type or "access::read_write" in texture_type

    def global_resource_parameter_names(self):
        names = set()
        for texture_variable, _, _, _ in self.texture_variables:
            if getattr(texture_variable, "name", None):
                names.add(texture_variable.name)
        for (
            acceleration_structure_variable,
            _,
            _,
            _,
        ) in self.acceleration_structure_variables:
            if getattr(acceleration_structure_variable, "name", None):
                names.add(acceleration_structure_variable.name)
        for (
            visible_function_table_variable,
            _,
            _,
            _,
        ) in self.visible_function_table_variables:
            if getattr(visible_function_table_variable, "name", None):
                names.add(visible_function_table_variable.name)
        for (
            intersection_function_table_variable,
            _,
            _,
            _,
        ) in self.intersection_function_table_variables:
            if getattr(intersection_function_table_variable, "name", None):
                names.add(intersection_function_table_variable.name)
        for buffer_variable, _, _, _ in self.structured_buffer_variables:
            if getattr(buffer_variable, "name", None):
                names.add(buffer_variable.name)
        for buffer_variable, _, _, _, _ in self.metal_buffer_resource_variables:
            if getattr(buffer_variable, "name", None):
                names.add(buffer_variable.name)
        for buffer_variable, _, _, _ in self.structured_buffer_length_variables:
            if getattr(buffer_variable, "name", None):
                names.add(
                    self.structured_buffer_length_parameter_name(buffer_variable.name)
                )
        for buffer_variable, _, _, _ in self.structured_buffer_counter_variables:
            if getattr(buffer_variable, "name", None):
                names.add(
                    self.structured_buffer_counter_parameter_name(buffer_variable.name)
                )
        for buffer_variable, _, _, _ in self.glsl_buffer_block_variables:
            if getattr(buffer_variable, "name", None):
                names.add(buffer_variable.name)
        for sampler_variable, _, _ in self.sampler_variables:
            if getattr(sampler_variable, "name", None):
                names.add(sampler_variable.name)
        return names

    def append_required_cbuffer_parameters(self, params_str, func_name):
        cbuffer_params = []
        for cbuffer in self.required_function_cbuffers(func_name):
            parameter_name = self.cbuffer_parameter_name(cbuffer)
            cbuffer_params.append(f"constant {cbuffer.name}& {parameter_name}")
        if not cbuffer_params:
            return params_str
        if params_str:
            return f"{params_str}, {', '.join(cbuffer_params)}"
        return ", ".join(cbuffer_params)

    def append_required_stage_parameter_parameters(self, params_str, func_name):
        stage_params = []
        for parameter in self.required_function_stage_parameters(func_name):
            raw_type = self.parameter_raw_type(parameter)
            mapped_type = self.map_type(raw_type)
            stage_params.append(
                format_c_style_array_declaration(mapped_type, parameter.name)
            )
        if not stage_params:
            return params_str
        if params_str:
            return f"{params_str}, {', '.join(stage_params)}"
        return ", ".join(stage_params)

    def append_required_stage_output_parameter(self, params_str, func_name):
        output_type = self.required_function_stage_output_type(func_name)
        if output_type is None:
            return params_str
        declaration = f"thread {self.map_type(output_type)}& output"
        if params_str:
            return f"{params_str}, {declaration}"
        return declaration
