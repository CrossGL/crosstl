"""CrossGL-to-WebGL GLSL ES code generator."""

from copy import copy

from ..ast import (
    AssignmentNode,
    BinaryOpNode,
    FunctionCallNode,
    StageMap,
    TernaryOpNode,
    WaveOpNode,
)
from .array_utils import (
    format_c_style_array_declaration,
    split_array_type_suffix,
)
from .GLSL_codegen import GLSLCodeGen
from .stage_utils import STAGE_QUALIFIER_NAMES, normalize_stage_name


class WebGLCodeGen(GLSLCodeGen):
    """Generate WebGL 2.0 compatible GLSL ES output from CrossGL ASTs."""

    BUILTIN_INTERFACE_BLOCK_NAMES = {"gl_PerVertex"}
    UNSUPPORTED_VERTEX_OUTPUT_BUILTINS = {"gl_ClipDistance", "gl_CullDistance"}
    UNSUPPORTED_STAGE_NAMES = (
        {
            "compute",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
        }
        | GLSLCodeGen.MESH_STAGE_NAMES
        | GLSLCodeGen.RAY_STAGE_NAMES
    )
    DEFAULT_PRECISION_LINES = (
        ("float", "precision highp float;"),
        ("int", "precision highp int;"),
    )
    UNSUPPORTED_INTERPOLATION_QUALIFIERS = {"noperspective", "sample"}
    SUPPORTED_STAGE_NAMES = {"fragment", "vertex"}
    STORAGE_IMAGE_INTRINSIC_NAMES = {
        "imageLoad",
        "imageStore",
        "imageSize",
        "imageSamples",
        "imageAtomicAdd",
        "imageAtomicMin",
        "imageAtomicMax",
        "imageAtomicAnd",
        "imageAtomicOr",
        "imageAtomicXor",
        "imageAtomicExchange",
        "imageAtomicCompSwap",
    }
    ATOMIC_INTRINSIC_NAMES = GLSLCodeGen.GLSL_MEMORY_ATOMIC_FUNCTIONS | {
        "atomicCounterIncrement",
        "atomicCounterDecrement",
        "atomicCounter",
        "atomicCounterAdd",
    }
    SYNCHRONIZATION_INTRINSIC_NAMES = {
        "barrier",
        "workgroupBarrier",
        "memoryBarrier",
        "memoryBarrierAtomicCounter",
        "memoryBarrierBuffer",
        "memoryBarrierImage",
        "memoryBarrierShared",
        "groupMemoryBarrier",
        "GroupMemoryBarrier",
        "GroupMemoryBarrierWithGroupSync",
        "DeviceMemoryBarrier",
        "DeviceMemoryBarrierWithGroupSync",
        "AllMemoryBarrier",
        "AllMemoryBarrierWithGroupSync",
    }
    GLSL_ES_310_TEXTURE_INTRINSIC_NAMES = {
        "textureGather",
        "textureGatherOffset",
        "textureGatherOffsets",
        "textureGatherCompare",
        "textureGatherCompareOffset",
        "textureGatherCompareOffsets",
    }
    GLSL_DESKTOP_TEXTURE_QUERY_INTRINSIC_NAMES = {
        "textureQueryLevels",
        "textureQueryLod",
    }
    UNSUPPORTED_SAMPLED_RESOURCE_TYPES = {
        "sampler1D",
        "sampler1DArray",
        "sampler2DMS",
        "sampler2DMSArray",
        "sampler2DRect",
        "samplerBuffer",
        "samplerCubeArray",
        "samplerCubeArrayShadow",
        "isampler1D",
        "isampler1DArray",
        "isampler2DMS",
        "isampler2DMSArray",
        "isampler2DRect",
        "isamplerBuffer",
        "isamplerCubeArray",
        "usampler1D",
        "usampler1DArray",
        "usampler2DMS",
        "usampler2DMSArray",
        "usampler2DRect",
        "usamplerBuffer",
        "usamplerCubeArray",
    }
    UNSUPPORTED_FLOAT64_TYPES = {
        "double",
        "dvec2",
        "dvec3",
        "dvec4",
        "dmat2",
        "dmat3",
        "dmat4",
        "dmat2x2",
        "dmat2x3",
        "dmat2x4",
        "dmat3x2",
        "dmat3x3",
        "dmat3x4",
        "dmat4x2",
        "dmat4x3",
        "dmat4x4",
    }
    UNSUPPORTED_OPAQUE_RESOURCE_TYPES = {
        "atomic_uint": "atomic counter",
    }
    BUILTIN_OUTPUT_TYPES = {
        "gl_Position": ("vec4", "vec4"),
        "gl_PointSize": ("float", "scalar float"),
        "gl_ClipDistance": ("float", "scalar float"),
        "gl_CullDistance": ("float", "scalar float"),
        "gl_FragDepth": ("float", "scalar float"),
        "gl_FragStencilRefARB": ("int", "scalar int"),
        "gl_SampleMask": ("int", "scalar int"),
    }

    def default_glsl_version_line(self, ast, target_stage=None):
        return "#version 300 es"

    def map_type(self, vtype):
        mapped_type = super().map_type(vtype)
        base_type = mapped_type.split("[", 1)[0].rstrip("*&")
        if base_type in self.UNSUPPORTED_FLOAT64_TYPES:
            raise ValueError(
                "WebGL target does not support 64-bit floating-point type "
                f"'{base_type}'"
            )
        return mapped_type

    def map_image_base_type_with_format(self, vtype, node=None):
        if self.is_storage_image_type(vtype):
            raise ValueError(
                "WebGL target does not support storage image resource "
                f"'{self.resource_node_name(node, '<unnamed>')}' "
                f"({self.type_name_string(self.resource_base_type(vtype))})"
            )
        mapped_type = super().map_image_base_type_with_format(vtype, node)
        return self.sampled_image_type(mapped_type)

    def glsl_resource_binding_layouts_supported(self, version_line):
        return False

    def generate_expression(self, expr, is_main=False):
        replacement = getattr(self, "_webgl_expression_replacements", {}).get(id(expr))
        if replacement is not None:
            return replacement
        if isinstance(expr, WaveOpNode):
            self.validate_webgl_wave_support(expr.operation)
        if isinstance(expr, FunctionCallNode):
            self.validate_webgl_function_call_support(self.function_call_name(expr))
        return super().generate_expression(expr, is_main=is_main)

    def validate_webgl_wave_support(self, operation):
        raise ValueError(
            "WebGL target does not support wave/subgroup intrinsic " f"'{operation}'"
        )

    def validate_webgl_function_call_support(self, func_name):
        if func_name in self.STORAGE_IMAGE_INTRINSIC_NAMES:
            raise ValueError(
                "WebGL target does not support storage image intrinsic "
                f"'{func_name}'"
            )
        if func_name in self.GLSL_ES_310_TEXTURE_INTRINSIC_NAMES:
            raise ValueError(
                "WebGL target requires GLSL ES 3.00 and does not support "
                f"texture gather intrinsic '{func_name}'"
            )
        if func_name in self.GLSL_DESKTOP_TEXTURE_QUERY_INTRINSIC_NAMES:
            raise ValueError(
                "WebGL target requires GLSL ES 3.00 and does not support "
                f"texture query intrinsic '{func_name}'"
            )
        if func_name in self.ATOMIC_INTRINSIC_NAMES:
            raise ValueError(
                f"WebGL target does not support atomic operation '{func_name}'"
            )
        if self.is_webgl_synchronization_intrinsic(func_name):
            raise ValueError(
                "WebGL target does not support synchronization intrinsic "
                f"'{func_name}'"
            )

    def is_webgl_synchronization_intrinsic(self, func_name):
        if func_name in self.function_return_types:
            return False
        return func_name in self.SYNCHRONIZATION_INTRINSIC_NAMES

    def validate_function_return_semantic(self, func, stage_name):
        super().validate_function_return_semantic(func, stage_name)
        semantic = self.function_return_semantic(func)
        if semantic is None:
            return
        self.validate_webgl_builtin_output_type(
            stage_name,
            semantic,
            self.function_return_type(func),
            f"function '{getattr(func, 'name', '<anonymous>')}' return",
        )

    def stage_output_member_map(self, func, shader_type):
        member_map = super().stage_output_member_map(func, shader_type)
        if member_map:
            self.validate_webgl_stage_output_struct_members(func, shader_type)
        return member_map

    def validate_webgl_stage_output_struct_members(self, func, stage_name):
        struct_name = self.type_node_name(getattr(func, "return_type", None))
        struct = self.structs_by_name.get(struct_name)
        if struct is None:
            return
        for member in getattr(struct, "members", []) or []:
            semantic = self.semantic_from_node(member)
            if semantic is None:
                continue
            self.validate_webgl_builtin_output_type(
                stage_name,
                semantic,
                self.member_type_name(member),
                f"struct '{struct_name}' member '{member.name}'",
            )

    def validate_webgl_builtin_output_type(
        self, stage_name, semantic, mapped_type, source
    ):
        mapped_semantic = self.map_semantic(semantic)
        expected = self.BUILTIN_OUTPUT_TYPES.get(mapped_semantic)
        if expected is None:
            return
        expected_type, expected_description = expected
        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        if array_suffix or base_type != expected_type:
            raise ValueError(
                f"WebGL {stage_name} stage {source} semantic '{semantic}' "
                f"must be {expected_description}"
            )

    def generate_glsl_interface_block_declaration(self, node):
        if self.is_webgl_builtin_interface_block(node):
            return ""
        return super().generate_glsl_interface_block_declaration(node)

    def is_webgl_builtin_interface_block(self, node):
        if not self.is_glsl_interface_block_struct(node):
            return False
        return (
            self.glsl_interface_block_name(node) in self.BUILTIN_INTERFACE_BLOCK_NAMES
        )

    def glsl_dynamic_resource_call_dispatch_info(self, expr):
        dispatch = super().glsl_dynamic_resource_call_dispatch_info(expr)
        if dispatch is not None:
            return dispatch
        if not isinstance(expr, FunctionCallNode):
            return None

        func_name = self.function_call_name(expr)
        if not func_name:
            return None
        args = list(getattr(expr, "arguments", getattr(expr, "args", [])) or [])
        dynamic_info = self.webgl_dynamic_sampler_call_info(func_name, args)
        if dynamic_info is None:
            return None

        cases = []
        direct_texture_call = dynamic_info.get("direct_texture_call", False)
        for index in range(dynamic_info["array_size"]):
            static_args = list(args)
            static_args[dynamic_info["arg_index"]] = (
                self.glsl_static_array_access_argument(dynamic_info, index)
            )
            if direct_texture_call:
                rendered_call = self.webgl_render_static_dynamic_sampler_call(
                    expr,
                    static_args,
                )
            else:
                rendered_args = ", ".join(
                    self.generate_function_call_arguments(func_name, static_args)
                )
                rendered_call = f"{func_name}({rendered_args})"
            cases.append((index, rendered_call))

        return {
            "index_expr": dynamic_info["index_expr"],
            "cases": cases,
            "return_type": self.expression_result_type(expr),
        }

    def webgl_dynamic_sampler_call_info(self, func_name, args):
        direct_info = self.webgl_direct_dynamic_sampler_array_call_info(func_name, args)
        if direct_info is not None:
            return direct_info
        return self.webgl_dynamic_sampler_array_call_info(func_name, args)

    def webgl_direct_dynamic_sampler_array_call_info(self, func_name, args):
        if (
            not args
            or func_name in self.STORAGE_IMAGE_INTRINSIC_NAMES
            or func_name not in self.texture_resource_operation_names()
        ):
            return None

        dynamic_info = self.glsl_dynamic_resource_array_access_info(
            args[0],
            self.current_resource_aliases,
        )
        if dynamic_info is None:
            return None

        texture_type = self.texture_declared_resource_type(dynamic_info["array_expr"])
        if texture_type is None:
            texture_type = self.expression_result_type(args[0])
        if not self.is_sampled_texture_type(texture_type):
            return None

        return {
            "arg_index": 0,
            "direct_texture_call": True,
            **dynamic_info,
        }

    def webgl_render_static_dynamic_sampler_call(self, expr, static_args):
        static_call = copy(expr)
        static_call.arguments = list(static_args)
        static_call.args = static_call.arguments
        return self.generate_expression(static_call)

    def webgl_dynamic_sampler_array_call_info(self, func_name, args):
        callee = self.function_definitions.get(func_name)
        if callee is None:
            return None

        params = list(getattr(callee, "parameters", getattr(callee, "params", [])))
        dynamic_arg = None
        for index, (param, arg) in enumerate(zip(params, args or [])):
            param_type = self.type_name_string(
                getattr(param, "param_type", getattr(param, "vtype", None))
            )
            if not self.is_sampled_texture_type(param_type):
                continue
            dynamic_info = self.glsl_dynamic_resource_array_access_info(
                arg,
                self.current_resource_aliases,
            )
            if dynamic_info is None:
                continue
            if dynamic_arg is not None:
                return None
            dynamic_arg = {
                "arg_index": index,
                **dynamic_info,
            }

        return dynamic_arg

    def webgl_nested_dynamic_sampler_expression_info(self, expr):
        if not self.webgl_dynamic_sampler_expression_can_be_lifted(expr):
            return None
        matches = []
        for node in self.walk_ast(expr):
            if node is expr or not isinstance(node, FunctionCallNode):
                continue
            func_name = self.function_call_name(node)
            if not func_name:
                continue
            args = list(getattr(node, "arguments", getattr(node, "args", [])) or [])
            dynamic_info = self.webgl_dynamic_sampler_call_info(func_name, args)
            if dynamic_info is None:
                continue
            dispatch = self.glsl_dynamic_resource_call_dispatch_info(node)
            if dispatch is None:
                continue
            matches.append(
                {
                    "call": node,
                    "dispatch": dispatch,
                    "return_type": (
                        dispatch.get("return_type")
                        or self.expression_result_type(node)
                        or "float"
                    ),
                }
            )
        if len(matches) != 1:
            return None
        return matches[0]

    def webgl_unliftable_dynamic_sampler_expression_info(self, expr):
        if self.webgl_dynamic_sampler_expression_can_be_lifted(expr):
            return None
        for node in self.walk_ast(expr):
            if not isinstance(node, FunctionCallNode):
                continue
            func_name = self.function_call_name(node)
            if not func_name:
                continue
            args = list(getattr(node, "arguments", getattr(node, "args", [])) or [])
            dynamic_info = self.webgl_dynamic_sampler_call_info(func_name, args)
            if dynamic_info is None:
                continue
            return {
                "call": node,
                "return_type": (
                    self.expression_result_type(expr)
                    or self.expression_result_type(node)
                    or "float"
                ),
            }
        return None

    def webgl_dynamic_sampler_expression_can_be_lifted(self, expr):
        for node in self.walk_ast(expr):
            if isinstance(node, TernaryOpNode):
                return False
            if isinstance(node, BinaryOpNode):
                op = self.map_operator(getattr(node, "op", None))
                if op in {"&&", "||"}:
                    return False
        return True

    def webgl_unliftable_dynamic_sampler_value(self, value_type):
        zero_value = self.zero_value_expression(value_type)
        return (
            "/* unsupported WebGL dynamic sampler array expression: "
            "dynamic sampler arrays cannot be lifted from ternary or "
            f"short-circuit expressions */ {zero_value}"
        )

    def webgl_unique_dynamic_sampler_temp_name(self):
        used_names = set(self.local_variable_types)
        used_names.update(self.current_identifier_aliases.values())
        used_names.update(self.current_stage_declared_names())
        base_name = "crossgl_dynamic_sampler_value"
        name = base_name
        suffix = 1
        while name in used_names:
            suffix += 1
            name = f"{base_name}_{suffix}"
        return name

    def webgl_render_expression_with_replacement(
        self,
        expr,
        call,
        replacement,
        expected_type,
    ):
        previous_replacements = getattr(self, "_webgl_expression_replacements", {})
        self._webgl_expression_replacements = {
            **previous_replacements,
            id(call): replacement,
        }
        try:
            return self.generate_expression_with_expected(expr, expected_type)
        finally:
            self._webgl_expression_replacements = previous_replacements

    def webgl_generate_nested_dynamic_sampler_prefix(self, nested_info, indent):
        temp_name = self.webgl_unique_dynamic_sampler_temp_name()
        temp_type = nested_info["return_type"]
        self.local_variable_types[temp_name] = temp_type
        indent_str = "    " * indent
        declaration = format_c_style_array_declaration(
            self.map_type(temp_type),
            temp_name,
        )
        code = f"{indent_str}{declaration};\n"
        code += self.generate_glsl_dynamic_resource_switch_statement(
            nested_info["dispatch"],
            indent,
            lambda call: f"{temp_name} = {call};",
            f"{temp_name} = {self.zero_value_expression(temp_type)};",
        )
        return temp_name, code

    def generate_glsl_dynamic_resource_call_assignment_statement(
        self,
        expr,
        target,
        target_type,
        indent,
    ):
        direct_statement = (
            super().generate_glsl_dynamic_resource_call_assignment_statement(
                expr,
                target,
                target_type,
                indent,
            )
        )
        if direct_statement is not None:
            return direct_statement

        nested_info = self.webgl_nested_dynamic_sampler_expression_info(expr)
        if nested_info is None:
            unliftable_info = self.webgl_unliftable_dynamic_sampler_expression_info(
                expr
            )
            if unliftable_info is None:
                return None
            indent_str = "    " * indent
            fallback = self.webgl_unliftable_dynamic_sampler_value(target_type)
            return f"{indent_str}{target} = {fallback};\n"

        temp_name, code = self.webgl_generate_nested_dynamic_sampler_prefix(
            nested_info,
            indent,
        )
        rendered_expr = self.webgl_render_expression_with_replacement(
            expr,
            nested_info["call"],
            temp_name,
            target_type,
        )
        indent_str = "    " * indent
        code += f"{indent_str}{target} = {rendered_expr};\n"
        return code

    def generate_glsl_dynamic_resource_assignment_node(self, stmt, indent):
        direct_statement = super().generate_glsl_dynamic_resource_assignment_node(
            stmt,
            indent,
        )
        if direct_statement is not None:
            return direct_statement

        left_node = getattr(stmt, "target", getattr(stmt, "left", None))
        right_node = getattr(stmt, "value", getattr(stmt, "right", None))
        op = self.map_operator(getattr(stmt, "operator", getattr(stmt, "op", "=")))
        if op != "=":
            return None
        nested_info = self.webgl_nested_dynamic_sampler_expression_info(right_node)
        if nested_info is None:
            return None
        self.validate_glsl_buffer_block_assignment_target(left_node, op)
        expected_type = self.glsl_tessellation_factor_assignment_expected_type(
            left_node
        )
        if expected_type is not None:
            self.validate_glsl_tessellation_factor_assignment_value(
                left_node,
                right_node,
            )
        else:
            expected_type = self.expression_result_type(left_node)
        left = self.generate_glsl_buffer_block_mutation_target(left_node)
        return self.generate_glsl_dynamic_resource_call_assignment_statement(
            right_node,
            left,
            expected_type,
            indent,
        )

    def generate_glsl_dynamic_resource_call_expression_statement(self, expr, indent):
        direct_statement = (
            super().generate_glsl_dynamic_resource_call_expression_statement(
                expr,
                indent,
            )
        )
        if direct_statement is not None:
            return direct_statement

        nested_info = self.webgl_nested_dynamic_sampler_expression_info(expr)
        if nested_info is None:
            unliftable_info = self.webgl_unliftable_dynamic_sampler_expression_info(
                expr
            )
            if unliftable_info is None:
                return None
            indent_str = "    " * indent
            fallback = self.webgl_unliftable_dynamic_sampler_value(
                unliftable_info["return_type"]
            )
            return f"{indent_str}{fallback};\n"

        temp_name, code = self.webgl_generate_nested_dynamic_sampler_prefix(
            nested_info,
            indent,
        )
        rendered_expr = self.webgl_render_expression_with_replacement(
            expr,
            nested_info["call"],
            temp_name,
            None,
        )
        indent_str = "    " * indent
        code += f"{indent_str}{rendered_expr};\n"
        return code

    def generate_glsl_dynamic_resource_call_return_statement(self, expr, indent):
        if self.current_stage_output is not None:
            direct_dispatch = self.glsl_dynamic_resource_call_dispatch_info(expr)
            if direct_dispatch is not None:
                indent_str = "    " * indent
                return_type = (
                    direct_dispatch.get("return_type")
                    or self.expression_result_type(expr)
                    or self.current_function_return_type
                )
                return (
                    self.generate_glsl_dynamic_resource_switch_statement(
                        direct_dispatch,
                        indent,
                        lambda call: f"{self.current_stage_output['name']} = {call};",
                        (
                            f"{self.current_stage_output['name']} = "
                            f"{self.zero_value_expression(return_type)};"
                        ),
                    )
                    + f"{indent_str}return;\n"
                )

            nested_info = self.webgl_nested_dynamic_sampler_expression_info(expr)
            if nested_info is not None:
                temp_name, code = self.webgl_generate_nested_dynamic_sampler_prefix(
                    nested_info,
                    indent,
                )
                rendered_expr = self.webgl_render_expression_with_replacement(
                    expr,
                    nested_info["call"],
                    temp_name,
                    self.current_function_return_type,
                )
                indent_str = "    " * indent
                code += f"{indent_str}{self.current_stage_output['name']} = {rendered_expr};\n"
                code += f"{indent_str}return;\n"
                return code

            unliftable_info = self.webgl_unliftable_dynamic_sampler_expression_info(
                expr
            )
            if unliftable_info is not None:
                indent_str = "    " * indent
                fallback = self.webgl_unliftable_dynamic_sampler_value(
                    unliftable_info["return_type"]
                )
                return (
                    f"{indent_str}{self.current_stage_output['name']} = {fallback};\n"
                    f"{indent_str}return;\n"
                )

        direct_statement = super().generate_glsl_dynamic_resource_call_return_statement(
            expr,
            indent,
        )
        if direct_statement is not None:
            return direct_statement

        nested_info = self.webgl_nested_dynamic_sampler_expression_info(expr)
        if nested_info is None:
            unliftable_info = self.webgl_unliftable_dynamic_sampler_expression_info(
                expr
            )
            if unliftable_info is None:
                return None
            indent_str = "    " * indent
            fallback = self.webgl_unliftable_dynamic_sampler_value(
                unliftable_info["return_type"]
            )
            return f"{indent_str}return {fallback};\n"

        temp_name, code = self.webgl_generate_nested_dynamic_sampler_prefix(
            nested_info,
            indent,
        )
        rendered_expr = self.webgl_render_expression_with_replacement(
            expr,
            nested_info["call"],
            temp_name,
            self.current_function_return_type,
        )
        indent_str = "    " * indent
        code += f"{indent_str}return {rendered_expr};\n"
        return code

    def should_emit_stage_io_layout(self, stage_name, direction):
        normalized_stage = normalize_stage_name(stage_name)
        if (
            normalized_stage == "fragment"
            and direction == "in"
            or normalized_stage == "vertex"
            and direction == "out"
        ):
            return False
        return super().should_emit_stage_io_layout(stage_name, direction)

    def generate_program(self, ast, target_stage=None):
        target_stage = normalize_stage_name(target_stage)
        self.validate_webgl_stage_support(ast, target_stage)
        supported_ast = self.webgl_supported_stage_ast(ast, target_stage)
        self.validate_webgl_builtin_support(supported_ast)
        self.validate_webgl_resource_support(supported_ast)
        self.validate_webgl_interpolation_qualifiers(supported_ast)
        codegen_ast = ast if target_stage is not None else supported_ast
        code = super().generate_program(
            codegen_ast,
            target_stage=target_stage,
        )
        return self._with_default_precision(code)

    def combined_fragment_input_member_name(self, input_name):
        # Guarded WebGL sections are compiled per stage, so varyings must link by name.
        return input_name

    def validate_webgl_resource_support(self, ast):
        structs_by_name = self.webgl_structs_by_name(ast)
        for node in self.walk_ast(ast):
            self.validate_webgl_node_resource_support(node)
            self.validate_webgl_block_resource_members(node, structs_by_name)

    def webgl_structs_by_name(self, ast):
        structs_by_name = {}
        for node in self.walk_ast(ast):
            if not self.is_struct_declaration_node(node):
                continue
            node_name = getattr(node, "name", None)
            if node_name:
                structs_by_name.setdefault(str(node_name), node)
        return structs_by_name

    def validate_webgl_block_resource_members(self, node, structs_by_name):
        if getattr(node, "is_cbuffer", False):
            self.validate_webgl_container_members(
                node,
                "constant buffer",
                self.resource_node_name(node, "<unnamed>"),
            )
            return

        if self.is_glsl_interface_block_struct(node):
            if self.is_webgl_builtin_interface_block(node):
                return
            self.validate_webgl_container_members(
                node,
                "interface block",
                self.glsl_interface_block_name(node),
            )
            return

        node_type = self.resource_node_type(node)
        if self.is_constant_buffer_type(node_type):
            struct_name = self.constant_buffer_element_type(node_type)
            struct = structs_by_name.get(str(struct_name))
            if struct is not None:
                self.validate_webgl_container_members(
                    struct,
                    "constant buffer",
                    self.resource_node_name(node, "<unnamed>"),
                )
            return

        if self.is_webgl_layout_bound_struct_uniform(node, node_type, structs_by_name):
            struct_name = str(self.resource_base_type(node_type))
            self.validate_webgl_container_members(
                structs_by_name[struct_name],
                "uniform block",
                self.resource_node_name(node, "<unnamed>"),
            )

    def is_webgl_layout_bound_struct_uniform(self, node, node_type, structs_by_name):
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        if "uniform" not in qualifiers:
            return False
        if self.explicit_resource_binding_index(node) is None:
            return False
        return str(self.resource_base_type(node_type)) in structs_by_name

    def validate_webgl_container_members(self, struct, container_kind, container_name):
        for member in getattr(struct, "members", []) or []:
            member_type = self.webgl_member_type(member)
            resource_kind, diagnostic_type = self.webgl_opaque_member_resource(
                member_type
            )
            if resource_kind is None:
                continue
            member_name = self.resource_node_name(member, "<unnamed>")
            raise ValueError(
                f"WebGL target does not support {resource_kind} resource member "
                f"'{member_name}' in {container_kind} '{container_name}' "
                f"({diagnostic_type})"
            )

    def webgl_member_type(self, member):
        if hasattr(member, "member_type"):
            return member.member_type
        if hasattr(member, "element_type"):
            return member.element_type
        return getattr(member, "vtype", "float")

    def webgl_opaque_member_resource(self, member_type):
        base_type = self.resource_base_type(member_type)
        if self.is_storage_image_type(base_type):
            return (
                "storage image",
                self.map_type(base_type),
            )

        sampled_type = self.sampled_image_type(base_type)
        sampled_base_type = self.map_type(self.resource_base_type(sampled_type))
        if self.is_webgl_sampled_resource_type(sampled_base_type):
            return "sampled", sampled_base_type

        mapped_base_type = self.map_type(base_type)
        if mapped_base_type in self.UNSUPPORTED_OPAQUE_RESOURCE_TYPES:
            return (
                self.UNSUPPORTED_OPAQUE_RESOURCE_TYPES[mapped_base_type],
                mapped_base_type,
            )
        if self.is_opaque_resource_type(mapped_base_type):
            return "opaque", mapped_base_type
        return None, None

    def validate_webgl_interpolation_qualifiers(self, ast):
        for node in self.walk_ast(ast):
            node_name = self.resource_node_name(node, "<unnamed>")
            for qualifier in self.webgl_node_unsupported_interpolation_qualifiers(node):
                raise ValueError(
                    "WebGL target does not support interpolation qualifier "
                    f"'{qualifier}' on '{node_name}'"
                )

    def webgl_node_unsupported_interpolation_qualifiers(self, node):
        qualifiers = [
            str(qualifier) for qualifier in getattr(node, "qualifiers", []) or []
        ]
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name:
                qualifiers.append(str(attr_name))

        unsupported = []
        for qualifier in qualifiers:
            normalized = qualifier.lower()
            if normalized.startswith("glsl_"):
                normalized = normalized[len("glsl_") :]
            normalized = normalized.replace("-", "_")
            if normalized in self.UNSUPPORTED_INTERPOLATION_QUALIFIERS:
                unsupported.append(normalized)
        return unsupported

    def validate_webgl_builtin_support(self, ast):
        builtin_names = self.webgl_unsupported_builtin_output_names(ast)
        for node in self.walk_ast(ast):
            if not isinstance(node, AssignmentNode):
                continue
            target = getattr(node, "target", getattr(node, "left", None))
            unsupported_builtin = self.webgl_referenced_unsupported_builtin(
                target, builtin_names
            )
            if unsupported_builtin is None:
                continue
            raise ValueError(
                "WebGL target does not support vertex built-in output "
                f"'{unsupported_builtin}'"
            )

    def webgl_unsupported_builtin_output_names(self, ast):
        names = {
            builtin: builtin for builtin in self.UNSUPPORTED_VERTEX_OUTPUT_BUILTINS
        }
        for node in self.walk_ast(ast):
            node_name = getattr(node, "name", None)
            if not node_name:
                continue
            builtin_name = self.webgl_unsupported_builtin_attribute_name(node)
            if builtin_name is not None:
                names[str(node_name)] = builtin_name
        return names

    def webgl_unsupported_builtin_attribute_name(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name in self.UNSUPPORTED_VERTEX_OUTPUT_BUILTINS:
                return attr_name
        semantic = getattr(node, "semantic", None)
        if semantic in self.UNSUPPORTED_VERTEX_OUTPUT_BUILTINS:
            return semantic
        return None

    def webgl_referenced_unsupported_builtin(self, node, builtin_names):
        if node is None:
            return None
        for child in self.walk_ast(node):
            name = getattr(child, "name", None)
            if name in self.UNSUPPORTED_VERTEX_OUTPUT_BUILTINS:
                return name
            if name in builtin_names:
                return builtin_names[name]
        return None

    def validate_webgl_node_resource_support(self, node):
        if isinstance(node, FunctionCallNode):
            self.validate_webgl_function_call_support(self.function_call_name(node))
            return

        node_type = self.resource_node_type(node)
        pointer_buffer_type = self.pointer_buffer_structured_type(node, node_type)
        if pointer_buffer_type is not None:
            node_type = pointer_buffer_type

        if self.is_webgl_glsl_buffer_block_node(node):
            raise ValueError(
                "WebGL target does not support GLSL buffer block resource "
                f"'{self.resource_node_name(node, '<unnamed>')}'"
            )
        if self.is_structured_buffer_type(node_type):
            raise ValueError(
                "WebGL target does not support storage buffer resource "
                f"'{self.resource_node_name(node, '<unnamed>')}' "
                f"({self.structured_buffer_type_name(node_type)})"
            )
        if self.is_storage_image_type(node_type):
            raise ValueError(
                "WebGL target does not support storage image resource "
                f"'{self.resource_node_name(node, '<unnamed>')}' "
                f"({self.type_name_string(self.resource_base_type(node_type))})"
            )
        mapped_base_type = self.map_type(self.resource_base_type(node_type))
        if mapped_base_type in self.UNSUPPORTED_OPAQUE_RESOURCE_TYPES:
            resource_kind = self.UNSUPPORTED_OPAQUE_RESOURCE_TYPES[mapped_base_type]
            raise ValueError(
                f"WebGL target does not support {resource_kind} resource "
                f"'{self.resource_node_name(node, '<unnamed>')}' "
                f"({mapped_base_type})"
            )
        sampled_type = self.sampled_image_type(node_type)
        sampled_base_type = self.map_type(self.resource_base_type(sampled_type))
        if sampled_base_type in self.UNSUPPORTED_SAMPLED_RESOURCE_TYPES:
            raise ValueError(
                "WebGL target does not support sampled resource "
                f"'{self.resource_node_name(node, '<unnamed>')}' "
                f"({sampled_base_type})"
            )
        memory_qualifiers = self.resource_memory_qualifiers(node)
        if memory_qualifiers and self.is_webgl_sampled_resource_type(sampled_base_type):
            raise ValueError(
                "WebGL target does not support resource memory qualifier(s) "
                f"'{memory_qualifiers}' on sampled resource "
                f"'{self.resource_node_name(node, '<unnamed>')}'"
            )

    def is_webgl_sampled_resource_type(self, type_name):
        return str(type_name).startswith(("sampler", "isampler", "usampler"))

    def is_webgl_glsl_buffer_block_node(self, node):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        if "buffer" in qualifiers:
            return True
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if attr_name and str(attr_name).lower() == "glsl_buffer_block":
                return True
        return False

    def structured_buffer_block_declaration(
        self, vtype, name, binding, array_size=None, node=None
    ):
        raise ValueError(
            "WebGL target does not support storage buffer resource "
            f"'{name}' ({self.structured_buffer_type_name(vtype)})"
        )

    def glsl_buffer_block_declaration(
        self, node, vtype, name, binding, array_suffix=""
    ):
        raise ValueError(
            f"WebGL target does not support GLSL buffer block resource '{name}'"
        )

    def validate_webgl_stage_support(self, ast, target_stage=None):
        normalized_target_stage = normalize_stage_name(target_stage)
        if normalized_target_stage:
            stages = {normalized_target_stage}
        else:
            stages = self.webgl_stage_names(ast)

        unsupported = sorted(stages & self.UNSUPPORTED_STAGE_NAMES)
        if unsupported:
            supported = stages & self.SUPPORTED_STAGE_NAMES
            if supported:
                return
            raise ValueError(
                "WebGL target does not support shader stage(s): "
                + ", ".join(unsupported)
            )

    def webgl_stage_names(self, ast):
        stages = set()
        for func in getattr(ast, "functions", []) or []:
            stage_name = self.webgl_function_stage_name(func)
            if stage_name:
                stages.add(stage_name)
        for stage_type in getattr(ast, "stages", {}) or {}:
            stage_name = normalize_stage_name(stage_type)
            if stage_name:
                stages.add(stage_name)
        return stages

    def webgl_supported_stage_ast(self, ast, target_stage=None):
        target_stage = normalize_stage_name(target_stage)
        filtered = copy(ast)
        filtered.functions = [
            func
            for func in getattr(ast, "functions", []) or []
            if self.should_emit_webgl_function(func, target_stage)
        ]
        filtered.stages = StageMap()
        for stage_type, stage in (getattr(ast, "stages", {}) or {}).items():
            if self.should_emit_webgl_stage(stage_type, target_stage):
                filtered.stages.append(stage_type, stage)
        return filtered

    def should_emit_webgl_function(self, func, target_stage=None):
        stage_name = self.webgl_function_stage_name(func)
        if not stage_name:
            return True
        if stage_name in self.UNSUPPORTED_STAGE_NAMES:
            return False
        if target_stage is not None:
            return stage_name == target_stage
        return stage_name in self.SUPPORTED_STAGE_NAMES

    def should_emit_webgl_stage(self, stage_type, target_stage=None):
        stage_name = normalize_stage_name(stage_type)
        if stage_name in self.UNSUPPORTED_STAGE_NAMES:
            return False
        if target_stage is not None:
            return stage_name == target_stage
        return stage_name in self.SUPPORTED_STAGE_NAMES

    def webgl_function_stage_name(self, func):
        qualifiers = list(getattr(func, "qualifiers", []) or [])
        qualifier = getattr(func, "qualifier", None)
        if qualifier:
            qualifiers.append(qualifier)
        for entry in qualifiers:
            stage_name = normalize_stage_name(entry)
            if stage_name in STAGE_QUALIFIER_NAMES:
                return stage_name
        for attr in getattr(func, "attributes", []) or []:
            stage_name = normalize_stage_name(getattr(attr, "name", ""))
            if stage_name in STAGE_QUALIFIER_NAMES:
                return stage_name
        return None

    def _with_default_precision(self, code):
        lines = code.splitlines()
        if not lines:
            return code

        existing_precision = {
            "float": any(
                line.strip().startswith("precision ")
                and line.strip().endswith(" float;")
                for line in lines
            ),
            "int": any(
                line.strip().startswith("precision ") and line.strip().endswith(" int;")
                for line in lines
            ),
        }
        precision_lines = [
            line
            for scalar_kind, line in self.DEFAULT_PRECISION_LINES
            if not existing_precision[scalar_kind]
        ]
        if not precision_lines:
            return code

        insert_at = next(
            (
                index + 1
                for index, line in enumerate(lines)
                if line.startswith("#version")
            ),
            0,
        )
        while insert_at < len(lines) and lines[insert_at].startswith("#extension"):
            insert_at += 1
        lines[insert_at:insert_at] = precision_lines
        return "\n".join(lines) + ("\n" if code.endswith("\n") else "")
