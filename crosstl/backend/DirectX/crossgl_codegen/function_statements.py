"""Functions and call statements in the HLSL to CrossGL converter: function declarations and bodies, stage struct parameters, local structs, clip, sincos and GetDimensions statements, sizeof and long binary expressions."""

import re

from ...common_ast import (
    ArrayAccessNode,
    BreakNode,
    CastNode,
    ContinueNode,
)
from ..DirectxAst import (
    AssignmentNode,
    BinaryOpNode,
    DoWhileNode,
    ForNode,
    FunctionCallNode,
    IfNode,
    MemberAccessNode,
    ReturnNode,
    ScopedBlockNode,
    StructNode,
    SwitchNode,
    TernaryOpNode,
    TypeAliasNode,
    UnaryOpNode,
    VariableNode,
    VectorConstructorNode,
    WhileNode,
)
from ..discarded_values import (
    VALUE_STATEMENT_TYPES,
    discarded_cast_operand,
)
from ..function_templates import (
    explicit_specialization_annotation,
    template_function_generic_prefix,
    template_type_parameter_names,
)
from ..struct_methods import method_local_names
from ..tiled_status_reads import (
    begin_status_texel_function,
    end_status_texel_function,
    open_status_texel_scope,
    take_status_texel_declarations,
)
from ..vk_locations import VK_LOCATION_ATTRIBUTE, check_function_return_vk_location
from ..work_graph_import import NODE_STAGE, crossgl_node_entry_attributes


class FunctionStatementMixin:
    """Function statement members of ``HLSLToCrossGLConverter``."""

    def generate_function(
        self,
        func,
        indent=1,
        skip_attribute_names=None,
        entry_name=None,
        stage_entry=False,
    ):
        """Render one HLSL function node as a CrossGL function block."""
        attributes = getattr(func, "attributes", []) or []
        if getattr(func, "qualifier", None) == NODE_STAGE:
            attributes = crossgl_node_entry_attributes(attributes)
        early_tests = any(
            str(getattr(attribute, "name", "")).lower() == "earlydepthstencil"
            for attribute in attributes
        )
        code = ""
        if early_tests and getattr(func, "qualifier", None) == "fragment":
            # [earlydepthstencil] is CrossGL's early_fragment_tests stage layout, as MSL's
            # [[early_fragment_tests]] imports; it is not a semantic of the function.
            code += "    " * indent + "layout(early_fragment_tests) in;\n"
        check_function_return_vk_location(self, func)
        code += self.format_attributes(
            attributes,
            indent,
            set(skip_attribute_names or ())
            | {VK_LOCATION_ATTRIBUTE}
            | ({"earlydepthstencil"} if early_tests else set()),
        )
        precise_as_attribute = (
            self.function_precise_return_needs_attribute(func)
            or self.precise_arithmetic_enabled()
        )
        if precise_as_attribute:
            code += "    " * indent + "@ precise\n"
        previous_template_type_names = getattr(self, "current_template_type_names", ())
        self.current_template_type_names = template_type_parameter_names(func)
        code += "    " * indent + template_function_generic_prefix(self, func)
        previous_variable_types = self.current_variable_types
        previous_resource_array_dims = self.current_resource_array_dims
        previous_identifier_renames = self.current_identifier_renames
        previous_stage_struct_parameter_member_aliases = (
            self.current_stage_struct_parameter_member_aliases
        )
        self.current_variable_types = dict(self.global_variable_types)
        self.current_resource_array_dims = dict(self.global_resource_array_dims)
        qualifier = getattr(func, "qualifier", None)
        emitted_params, member_aliases = self.flatten_stage_struct_parameters(
            func.params, qualifier
        )
        previous_struct_method = getattr(self, "current_struct_method", None)
        previous_struct_method_locals = getattr(
            self, "current_struct_method_locals", set()
        )
        struct_method = self.struct_method_for_function(func)
        self.current_struct_method = struct_method
        self.current_struct_method_locals = set()
        if struct_method is not None:
            self.current_struct_method_locals = method_local_names(
                func, self.iter_ast_children
            )
            if not struct_method.static:
                emitted_params = [
                    self.struct_method_receiver_parameter(struct_method),
                    *emitted_params,
                ]
        self.current_stage_struct_parameter_member_aliases = member_aliases
        self.current_identifier_renames = self.function_parameter_renames(
            emitted_params
        )
        for param in emitted_params:
            self.record_variable_type(param)
        params = ", ".join(
            self.format_parameter(p, qualifier, index)
            for index, p in enumerate(emitted_params)
        )
        semantic = self.map_semantic(func.semantic)
        semantic = f" {semantic}" if semantic else ""
        stage_entry_attribute = " @ stage_entry" if stage_entry else ""
        stage_entry_attribute += explicit_specialization_annotation(self, func)
        function_name = getattr(self, "specialization_identifiers", {}).get(
            id(func)
        ) or self.render_function_identifier(entry_name or func.name)
        precise_prefix = (
            "" if precise_as_attribute else self.format_precise_qualifier_prefix(func)
        )
        return_type = (
            f"{precise_prefix}{self.map_type(func.return_type)}"
            f"{self.format_array_suffixes(func)}"
        )
        code += (
            f"{return_type} "
            f"{function_name}({params}){semantic}{stage_entry_attribute} {{\n"
        )
        previous_return_type = self.current_function_return_type
        previous_source_function = getattr(self, "current_source_function", None)
        self.current_function_return_type = func.return_type
        self.current_source_function = func
        previous_texels = begin_status_texel_function(self)
        code += self.generate_function_body(func.body, indent=indent + 1)
        end_status_texel_function(self, previous_texels)
        self.current_function_return_type = previous_return_type
        self.current_source_function = previous_source_function
        code += "    " * indent + "}\n\n"
        self.current_variable_types = previous_variable_types
        self.current_resource_array_dims = previous_resource_array_dims
        self.current_identifier_renames = previous_identifier_renames
        self.current_template_type_names = previous_template_type_names
        self.current_struct_method = previous_struct_method
        self.current_struct_method_locals = previous_struct_method_locals
        self.current_stage_struct_parameter_member_aliases = (
            previous_stage_struct_parameter_member_aliases
        )
        return code

    def flatten_stage_struct_parameters(self, params, function_qualifier=None):
        """Expose HLSL struct in/out entry parameters as scalar stage fields."""
        stage_name = str(function_qualifier or "").lower()
        if stage_name not in {"vertex", "fragment"}:
            return params, {}
        if not self.has_stage_struct_output_parameter(params):
            return params, {}

        flattened_params = []
        member_aliases = {}
        used_names = {getattr(param, "name", None) for param in params or []}
        for param_index, param in enumerate(params or []):
            struct_type = self.raw_type_base(getattr(param, "vtype", None))
            struct_node = self.structs_by_name.get(struct_type)
            if struct_node is None:
                flattened_params.append(param)
                continue

            direction_qualifiers = [
                qualifier
                for qualifier in getattr(param, "qualifiers", []) or []
                if str(qualifier).lower() in {"in", "out", "inout"}
            ]
            if not direction_qualifiers:
                direction_qualifiers = ["in"]

            param_name = getattr(param, "name", "") or f"_param{param_index}"
            for member in getattr(struct_node, "members", []) or []:
                if not isinstance(member, VariableNode):
                    continue
                alias_name = self.unique_flattened_stage_parameter_name(
                    param_name, member.name, used_names
                )
                used_names.add(alias_name)
                flattened = VariableNode(
                    getattr(member, "vtype", None),
                    alias_name,
                    qualifiers=direction_qualifiers
                    + [
                        qualifier
                        for qualifier in getattr(member, "qualifiers", []) or []
                        if str(qualifier).lower()
                        not in {"in", "out", "inout", "static", "const"}
                    ],
                    attributes=list(getattr(member, "attributes", []) or []),
                    is_const=getattr(member, "is_const", False),
                    semantic=getattr(member, "semantic", None),
                )
                flattened.array_sizes = list(getattr(member, "array_sizes", []) or [])
                member_aliases[(param_name, member.name)] = alias_name
                flattened_params.append(flattened)

        return flattened_params, member_aliases

    def has_stage_struct_output_parameter(self, params):
        for param in params or []:
            type_name = self.raw_type_base(getattr(param, "vtype", None))
            if type_name not in self.structs_by_name:
                continue
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(param, "qualifiers", []) or []
            }
            if qualifiers & {"out", "inout"}:
                return True
        return False

    def unique_flattened_stage_parameter_name(
        self, param_name, member_name, used_names
    ):
        base = f"{param_name}_{member_name}"
        candidate = base
        suffix = 2
        while candidate in used_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def generate_function_body(self, body, indent=0, is_main=False, *, new_scope=True):
        previous_types = self.current_variable_types
        previous_dims = self.current_resource_array_dims
        if new_scope:
            self.current_variable_types = dict(previous_types)
            self.current_resource_array_dims = dict(previous_dims)
        code = ""
        for stmt in body:
            stmt = discarded_cast_operand(stmt)
            if stmt is None or isinstance(stmt, (bool, int, float)):
                continue  # nothing to evaluate, as in ``(void)x;`` or ``1;``
            open_status_texel_scope(self)
            try:
                statement = self.generate_body_statement(stmt, indent, is_main)
            finally:
                declarations = take_status_texel_declarations(self)
            for declaration in declarations:
                code += "    " * indent + declaration + "\n"
            code += statement
        self.current_variable_types = previous_types
        self.current_resource_array_dims = previous_dims
        return code

    def generate_body_statement(self, stmt, indent=0, is_main=False):
        """Render one statement of a function body, indented by ``indent`` levels."""
        code = ""
        if isinstance(stmt, FunctionCallNode):
            return self.generate_function_call_statement(stmt, indent, is_main)
        if isinstance(stmt, VariableNode):
            code += self.format_matrix_layout_attributes(stmt, indent)
        code += "    " * indent
        if isinstance(stmt, VariableNode):
            array_suffix = self.format_array_suffixes(stmt, is_main)
            qualifier_prefix = self.format_local_storage_qualifier_prefix(
                stmt
            ) + self.format_precise_qualifier_prefix(stmt)
            if stmt.value is not None:
                value = self.generate_expression(
                    stmt.value, is_main, expected_type=stmt.vtype
                )
                value = self.truncated_initializer_value(stmt, value)
                self.record_variable_type(stmt)
                code += (
                    f"{qualifier_prefix}{self.map_variable_type(stmt)} "
                    f"{self.render_identifier(stmt.name)}{array_suffix} = "
                    f"{value};\n"
                )
            else:
                self.record_variable_type(stmt)
                code += (
                    f"{qualifier_prefix}{self.map_variable_type(stmt)} "
                    f"{self.render_identifier(stmt.name)}{array_suffix};\n"
                )
        elif isinstance(stmt, AssignmentNode):
            texel_update = self.generate_storage_image_component_update(
                stmt, indent, is_main
            )
            if texel_update is not None:
                code += texel_update
            else:
                code += self.generate_assignment(stmt, is_main) + ";\n"

        elif isinstance(stmt, BinaryOpNode):
            code += f"{self.generate_expression(stmt, is_main)};\n"
        elif isinstance(stmt, UnaryOpNode):
            code += f"{self.generate_expression(stmt, is_main)};\n"
        elif isinstance(stmt, ReturnNode):
            if stmt.value is None:
                code += "return;\n"
            elif not is_main:
                value = self.generate_expression(
                    stmt.value,
                    is_main,
                    expected_type=self.current_function_return_type,
                )
                code += f"return {value};\n"
        elif isinstance(stmt, ForNode):
            code += self.generate_for_loop(stmt, indent, is_main)
        elif isinstance(stmt, WhileNode):
            code += self.generate_while_loop(stmt, indent, is_main)
        elif isinstance(stmt, DoWhileNode):
            code += self.generate_do_while_loop(stmt, indent, is_main)
        elif isinstance(stmt, IfNode):
            code += self.generate_if_statement(stmt, indent, is_main)
        elif isinstance(stmt, SwitchNode):
            code += self.generate_switch_statement(stmt, indent, is_main)
        elif isinstance(stmt, BreakNode):
            code += "break;\n"
        elif isinstance(stmt, ContinueNode):
            code += "continue;\n"
        elif isinstance(stmt, ScopedBlockNode):
            code += "{\n"
            code += self.generate_function_body(stmt.statements, indent + 1, is_main)
            code += "    " * indent + "}\n"
        elif isinstance(stmt, str):
            code += f"{stmt};\n"
        elif isinstance(stmt, TypeAliasNode):
            code += f"{self.generate_type_alias_declaration(stmt)}\n"
        elif isinstance(stmt, StructNode):
            code += self.generate_local_struct(stmt, indent, is_main)
        elif isinstance(stmt, VALUE_STATEMENT_TYPES):
            code += f"{self.generate_expression(stmt, is_main)};\n"
        else:
            code += self.refuse_import(
                f"statement {type(stmt).__name__} has no CrossGL form"
            )
        return code

    def generate_local_struct(self, stmt, indent, is_main=False):
        """Write a structure declared in a function body as a CrossGL block-scoped structure.

        Declarators after its closing brace (``struct S { ... } a, b;``) follow it as declarations.
        """
        unsupported = [
            label
            for label, field in (
                ("member functions", "methods"),
                ("base classes", "base_classes"),
            )
            if getattr(stmt, field, None)
        ]
        if unsupported:
            return self.refuse_import(
                f"DirectX structure '{stmt.name}' declared in a function body has "
                f"{' and '.join(unsupported)}, which a CrossGL block-scoped structure "
                "does not carry"
            )
        saved_indentation = self.indentation
        self.indentation = indent
        try:
            text = self.visit_StructNode(stmt)
        finally:
            self.indentation = saved_indentation
        # The semicolon keeps a following statement from reading as a declarator.
        code = text.rstrip("\n") + ";\n"
        declarations = list(getattr(stmt, "variable_declarations", None) or [])
        if declarations:
            code += self.generate_function_body(
                declarations, indent, is_main, new_scope=False
            )
        return code

    def generate_function_call_statement(self, stmt, indent=0, is_main=False):
        clip_statement = self.generate_clip_statement(stmt, indent, is_main)
        if clip_statement is not None:
            return clip_statement
        sincos_statement = self.generate_sincos_statement(stmt, indent, is_main)
        if sincos_statement is not None:
            return sincos_statement
        asuint_double_statement = self.generate_asuint_double_statement(
            stmt, indent, is_main
        )
        if asuint_double_statement is not None:
            return asuint_double_statement
        lowered_sequence = self.generate_function_call_statement_sequence(stmt, indent)
        if lowered_sequence is not None:
            return lowered_sequence
        lowered_get_dimensions = self.generate_get_dimensions_statement(
            stmt, indent, is_main
        )
        if lowered_get_dimensions is not None:
            return lowered_get_dimensions
        return "    " * indent + f"{self.generate_expression(stmt, is_main)};\n"

    def generate_clip_statement(self, stmt, indent=0, is_main=False):
        if (
            not isinstance(stmt.name, str)
            or self.normalize_hlsl_intrinsic_name(stmt.name) != "clip"
            or not stmt.args
        ):
            return None
        condition = self.generate_clip_condition(stmt.args[0], is_main)
        indent_text = "    " * indent
        return (
            f"{indent_text}if ({condition}) {{\n"
            f"{indent_text}    discard;\n"
            f"{indent_text}}}\n"
        )

    def generate_sincos_statement(self, stmt, indent=0, is_main=False):
        if (
            not isinstance(stmt.name, str)
            or self.normalize_hlsl_intrinsic_name(stmt.name) != "sincos"
        ):
            return None
        if len(stmt.args) != 3:
            return None
        value = self.generate_expression(stmt.args[0], is_main)
        sine = f"sin({value})"
        cosine = f"cos({value})"
        if "sincos" not in self.source_function_names:
            sine = self.trigonometric_call("sin", stmt.args[0], value)
            cosine = self.trigonometric_call("cos", stmt.args[0], value)
        sine_target = self.generate_expression(stmt.args[1], is_main)
        cosine_target = self.generate_expression(stmt.args[2], is_main)
        indent_text = "    " * indent
        return (
            f"{indent_text}{sine_target} = {sine};\n"
            f"{indent_text}{cosine_target} = {cosine};\n"
        )

    def generate_asuint_double_statement(self, stmt, indent=0, is_main=False):
        if (
            not isinstance(stmt.name, str)
            or self.normalize_hlsl_intrinsic_name(stmt.name) != "asuint"
        ):
            return None
        if len(stmt.args) != 3:
            return None
        if self.raw_type_base(self.expression_value_raw_type(stmt.args[0])) != "double":
            return None
        value = self.generate_expression(stmt.args[0], is_main)
        low_target = self.generate_expression(stmt.args[1], is_main)
        high_target = self.generate_expression(stmt.args[2], is_main)
        indent_text = "    " * indent
        return (
            f"{indent_text}{low_target} = unpackDouble2x32({value}).x;\n"
            f"{indent_text}{high_target} = unpackDouble2x32({value}).y;\n"
        )

    def generate_clip_condition(self, operand_expr, is_main=False):
        operand = self.generate_expression(operand_expr, is_main)
        operand = self.maybe_parenthesize(operand_expr, operand)
        component_count = self.clip_operand_component_count(operand_expr)
        if component_count and component_count > 1:
            return f"any(lessThan({operand}, vec{component_count}(0.0)))"
        return f"{operand} < 0.0"

    def clip_operand_component_count(self, operand_expr):
        raw_type = self.expression_raw_type(operand_expr)
        if raw_type is None:
            return None
        mapped_type = self.map_type(raw_type)
        for prefix in ("vec", "ivec", "uvec", "bvec", "dvec", "f16vec"):
            if mapped_type.startswith(prefix):
                suffix = mapped_type[len(prefix) :]
                if suffix.isdigit():
                    return int(suffix)
        return None

    def generate_function_call_statement_sequence(self, stmt, indent=0):
        if not isinstance(stmt.name, str):
            return None
        sequence = self.function_statement_sequences.get(
            self.normalize_hlsl_intrinsic_name(stmt.name)
        )
        if sequence is None or stmt.args:
            return None
        indent_text = "    " * indent
        return "".join(f"{indent_text}{func_name}();\n" for func_name in sequence)

    def generate_get_dimensions_statement(self, stmt, indent=0, is_main=False):
        if not isinstance(stmt.name, MemberAccessNode):
            return None
        member = self.templated_method_base(stmt.name.member)
        if member != "GetDimensions":
            return None

        resource_type = self.expression_raw_type(stmt.name.object)
        resource_base = self.raw_type_base(resource_type)
        if not resource_base:
            return None

        texel_buffer_dimensions = self.generate_texel_buffer_dimensions(
            stmt, indent, is_main
        )
        if texel_buffer_dimensions is not None:
            return texel_buffer_dimensions

        obj = self.generate_expression(stmt.name.object, is_main)
        rendered_args = [self.generate_expression(arg, is_main) for arg in stmt.args]
        indent_text = "    " * indent

        if self.is_buffer_resource_type(resource_type):
            return (
                indent_text
                + f"buffer_dimensions({', '.join([obj, *rendered_args])});\n"
            )

        assignments = self.get_dimensions_assignments(
            resource_base, obj, stmt.args, rendered_args, is_main
        )
        if assignments is None:
            descriptor = self.resource_method_descriptor(
                member, len(stmt.args), resource_type
            )
            function = descriptor["function"] if descriptor else "texture_dimensions"
            self.refuse_import(
                f"DirectX GetDimensions overload for {resource_base} with "
                f"{len(stmt.args)} arguments: no CrossGL dimension query fills them"
            )
            return indent_text + f"{function}({', '.join([obj, *rendered_args])});\n"
        return "".join(indent_text + assignment + ";\n" for assignment in assignments)

    def get_dimensions_assignments(
        self, resource_base, obj, raw_args, rendered_args, is_main=False
    ):
        layout = self.get_dimensions_layout(resource_base, obj, rendered_args)
        if layout is None:
            return None

        size_expr = layout["size_expr"]
        component_suffixes = layout["component_suffixes"]
        assignments = []
        for target, suffix, arg_index in zip(
            layout["dimension_targets"],
            component_suffixes,
            layout["dimension_indices"],
        ):
            value = f"{size_expr}{suffix}"
            assignments.append(
                f"{target} = {self.cast_get_dimensions_value(value, raw_args[arg_index])}"
            )

        if layout.get("levels_target") is not None:
            value = f"textureQueryLevels({obj})"
            assignments.append(
                f"{layout['levels_target']} = "
                f"{self.cast_get_dimensions_value(value, raw_args[layout['levels_index']])}"
            )
        if layout.get("samples_target") is not None:
            value = f"textureSamples({obj})"
            assignments.append(
                f"{layout['samples_target']} = "
                f"{self.cast_get_dimensions_value(value, raw_args[layout['samples_index']])}"
            )
        return assignments

    def get_dimensions_layout(self, resource_base, obj, rendered_args):
        args_count = len(rendered_args)
        if args_count == 0:
            return None

        texture_dimensions = {
            "Texture1D": 1,
            "Texture1DArray": 2,
            "Texture2D": 2,
            "Texture2DArray": 3,
            "Texture3D": 3,
            "TextureCube": 2,
            "TextureCubeArray": 3,
            "FeedbackTexture2D": 2,
            "FeedbackTexture2DArray": 3,
        }
        image_dimensions = {
            "RWTexture1D": 1,
            "RWTexture1DArray": 2,
            "RWTexture2D": 2,
            "RWTexture2DArray": 3,
            "RWTexture3D": 3,
            "RWTextureCube": 2,
            "RWTextureCubeArray": 3,
            "RasterizerOrderedTexture1D": 1,
            "RasterizerOrderedTexture1DArray": 2,
            "RasterizerOrderedTexture2D": 2,
            "RasterizerOrderedTexture2DArray": 3,
            "RasterizerOrderedTexture3D": 3,
        }
        multisample_dimensions = {
            "Texture2DMS": 2,
            "Texture2DMSArray": 3,
            "RWTexture2DMS": 2,
            "RWTexture2DMSArray": 3,
            "RasterizerOrderedTexture2DMS": 2,
            "RasterizerOrderedTexture2DMSArray": 3,
        }

        if resource_base in multisample_dimensions:
            dimensions = multisample_dimensions[resource_base]
            if args_count not in {dimensions, dimensions + 1}:
                return None
            size_function = (
                "imageSize"
                if resource_base.startswith(("RWTexture", "RasterizerOrderedTexture"))
                else "textureSize"
            )
            return {
                "size_expr": f"{size_function}({obj})",
                "dimension_targets": rendered_args[:dimensions],
                "dimension_indices": list(range(dimensions)),
                "component_suffixes": self.dimension_component_suffixes(dimensions),
                "samples_target": (
                    rendered_args[dimensions] if args_count == dimensions + 1 else None
                ),
                "samples_index": dimensions if args_count == dimensions + 1 else None,
            }

        if resource_base in image_dimensions:
            dimensions = image_dimensions[resource_base]
            if args_count != dimensions:
                return None
            return {
                "size_expr": f"imageSize({obj})",
                "dimension_targets": rendered_args,
                "dimension_indices": list(range(dimensions)),
                "component_suffixes": self.dimension_component_suffixes(dimensions),
            }

        dimensions = texture_dimensions.get(resource_base)
        if dimensions is None:
            return None

        lod = "0"
        out_start = 0
        levels_index = None
        if args_count == dimensions:
            pass
        elif args_count == dimensions + 1:
            levels_index = dimensions
        elif args_count == dimensions + 2:
            lod = rendered_args[0]
            out_start = 1
            levels_index = dimensions + 1
        else:
            return None

        dimension_targets = rendered_args[out_start : out_start + dimensions]
        if len(dimension_targets) != dimensions:
            return None
        return {
            "size_expr": f"textureSize({obj}, {lod})",
            "dimension_targets": dimension_targets,
            "dimension_indices": list(range(out_start, out_start + dimensions)),
            "component_suffixes": self.dimension_component_suffixes(dimensions),
            "levels_target": (
                rendered_args[levels_index] if levels_index is not None else None
            ),
            "levels_index": levels_index,
        }

    def dimension_component_suffixes(self, dimensions):
        if dimensions == 1:
            return [""]
        return [f".{component}" for component in ("x", "y", "z")[:dimensions]]

    def cast_get_dimensions_value(self, value, target_arg):
        if self.get_dimensions_target_scalar_type(target_arg) == "uint":
            return f"uint({value})"
        return value

    def get_dimensions_target_scalar_type(self, target_arg):
        if target_arg is None:
            return None

        mapped_type = self.map_type(self.expression_raw_type(target_arg))
        if mapped_type == "uint":
            return "uint"

        if not isinstance(target_arg, MemberAccessNode):
            return mapped_type

        if target_arg.member not in {"x", "y", "z", "w", "r", "g", "b", "a"}:
            return mapped_type

        if mapped_type in {"uvec2", "uvec3", "uvec4"}:
            return "uint"
        return mapped_type

    def generate_sizeof_expression(self, args):
        if len(args) != 1:
            return None

        size = self.sizeof_operand(args[0])
        if size is None:
            return None
        return str(size)

    def sizeof_operand(self, operand):
        if isinstance(operand, ArrayAccessNode):
            element_size = self.sizeof_operand(operand.array)
            element_count = self.sizeof_array_element_count(operand.index)
            if element_size is None or element_count is None:
                return None
            return element_size * element_count

        if isinstance(operand, CastNode):
            return self.sizeof_type_name(operand.target_type)

        if isinstance(operand, VectorConstructorNode):
            return self.sizeof_type_name(operand.type_name)

        raw_type = self.expression_raw_type(operand)
        if raw_type is not None:
            size = self.sizeof_type_name(raw_type)
            if size is not None:
                return size

        if isinstance(operand, str):
            return self.sizeof_type_name(operand)

        return None

    def sizeof_array_element_count(self, expr):
        if isinstance(expr, int) and not isinstance(expr, bool):
            return expr
        return None

    def sizeof_type_name(self, type_name):
        if not type_name:
            return None

        type_name = self.canonical_composite_type(
            self.strip_sizeof_type_qualifiers(str(type_name).strip())
        )

        if "<" in type_name and type_name.endswith(">"):
            base, generic_args = type_name.split("<", 1)
            args = self.split_generic_arguments(generic_args[:-1].strip())
            if base == "vector" and args:
                components = (
                    self.parse_template_dimension(args[1]) if len(args) > 1 else 4
                )
                scalar_size = self.sizeof_scalar_type(args[0])
                if scalar_size is None or components is None:
                    return None
                return scalar_size * components
            if base == "matrix" and args:
                rows = self.parse_template_dimension(args[1]) if len(args) > 1 else 4
                cols = self.parse_template_dimension(args[2]) if len(args) > 2 else 4
                scalar_size = self.sizeof_scalar_type(args[0])
                if scalar_size is None or rows is None or cols is None:
                    return None
                return scalar_size * rows * cols
            return None

        matrix_match = re.fullmatch(
            r"(float|half|fixed|double|min16float|min10float|float16_t|int|uint|bool|"
            r"min16int|min12int|int16_t|min16uint|uint16_t|int64_t|uint64_t)"
            r"([1-4])x([1-4])",
            type_name,
        )
        if matrix_match:
            scalar_type, rows_text, cols_text = matrix_match.groups()
            scalar_size = self.sizeof_scalar_type(scalar_type)
            if scalar_size is None:
                return None
            return scalar_size * int(rows_text) * int(cols_text)

        vector_match = re.fullmatch(
            r"(float|half|fixed|double|min16float|min10float|float16_t|int|uint|bool|"
            r"min16int|min12int|int16_t|min16uint|uint16_t|int64_t|uint64_t)"
            r"([1-4])",
            type_name,
        )
        if vector_match:
            scalar_type, width_text = vector_match.groups()
            scalar_size = self.sizeof_scalar_type(scalar_type)
            if scalar_size is None:
                return None
            return scalar_size * int(width_text)

        scalar_size = self.sizeof_scalar_type(type_name)
        if scalar_size is not None:
            return scalar_size

        return None

    def strip_sizeof_type_qualifiers(self, type_name):
        qualifiers = {
            "const",
            "row_major",
            "column_major",
            "precise",
            "snorm",
            "unorm",
        }
        parts = str(type_name).split()
        while parts and parts[0].lower() in qualifiers:
            parts.pop(0)
        return " ".join(parts)

    def sizeof_scalar_type(self, type_name):
        scalar_sizes = {
            "bool": 4,
            "int": 4,
            "uint": 4,
            "dword": 4,
            "float": 4,
            "float32_t": 4,
            "fixed": 4,
            "int32_t": 4,
            "uint32_t": 4,
            "half": 2,
            "min16float": 2,
            "min10float": 2,
            "float16_t": 2,
            "min16int": 2,
            "min12int": 2,
            "int16_t": 2,
            "min16uint": 2,
            "uint16_t": 2,
            "double": 8,
            "int64_t": 8,
            "uint64_t": 8,
        }
        return scalar_sizes.get(self.canonical_composite_type(str(type_name).strip()))

    def format_float(self, value: float) -> str:
        text = format(value, ".10f")
        text = text.rstrip("0").rstrip(".")
        if text in ("", "-0"):
            text = "0"
        if "." not in text and "e" not in text.lower():
            text += ".0"
        return text

    def maybe_parenthesize(self, expr, rendered: str) -> str:
        if isinstance(expr, (BinaryOpNode, TernaryOpNode, AssignmentNode)):
            return f"({rendered})"
        return rendered

    def binary_operator_text(self, op):
        return op.value if hasattr(op, "value") else str(op)

    def generate_flat_long_binary_expression(self, expr, is_main=False):
        op = self.binary_operator_text(expr.op)
        if op != "+":
            return None

        parts = []
        current = expr
        while (
            isinstance(current, BinaryOpNode)
            and self.binary_operator_text(current.op) == op
        ):
            parts.append(current.right)
            current = current.left

        parts.append(current)
        if len(parts) < 16:
            return None

        parts.reverse()
        result = self.generate_expression(parts[0], is_main)
        result = self.maybe_parenthesize(parts[0], result)
        result_type = self.expression_value_raw_type(parts[0])
        for part in parts[1:]:
            rendered = self.generate_expression(part, is_main)
            rendered = self.maybe_parenthesize(part, rendered)
            part_type = self.expression_value_raw_type(part)
            common = self.wide_integer_common_type(result_type, part_type, op)
            if common is not None:
                result = self.convert_integer_operand(result, result_type, common)
                rendered = self.convert_integer_operand(rendered, part_type, common)
            result = f"{result} {op} {rendered}"
            result_type = self.integer_expression_raw_type(result_type, part_type, op)
        return result
