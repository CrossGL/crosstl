"""Steps of ``GLSLToCrossGLConverter.generate_shader``: resetting the per-shader state, recording the built-in stage outputs the entry writes, rendering the structs, stage interfaces, uniforms, constants and globals, and rendering the stage block with its functions."""

from .block_member_names import (
    instance_block_member_names,
    rename_shadowing_locals,
)
from .OpenglAst import (
    LayoutNode,
    ReturnNode,
    VariableNode,
)


class GLSLShaderSectionMixin:
    """Shader-rendering members of ``GLSLToCrossGLConverter.generate_shader``."""

    def reset_glsl_shader_state(
        self,
        node,
    ):
        """Reset the per-shader tables and normalise the stage variables before rendering."""
        self.uniform_vars = []
        self.inputs = []
        self.outputs = []
        self.local_vars = []
        self.flattened_uniform_block_instances = {}
        self.flattened_uniform_block_member_renames = {}
        rename_shadowing_locals(node, instance_block_member_names(node))
        self.emitted_cbuffer_member_names = set()
        self.function_name_renames = self.collect_function_name_renames(
            getattr(node, "functions", []) or []
        )
        self.user_function_arities = {
            (
                getattr(function, "name", None),
                len(getattr(function, "params", []) or []),
            )
            for function in getattr(node, "functions", []) or []
            if getattr(function, "name", None)
        }
        self.glsl_function_return_types = {}
        for function in getattr(node, "functions", []) or []:
            if getattr(function, "name", None):
                self.glsl_function_return_types.setdefault(function.name, set()).add(
                    str(getattr(function, "return_type", "") or "").strip()
                )
        self.prepare_structured_buffers(node)
        node.io_variables = self.deduplicate_builtin_stage_io_variables(
            getattr(node, "io_variables", []) or []
        )

        for var in node.io_variables:
            if isinstance(var, (LayoutNode, VariableNode)):
                if self._is_input_var(var):
                    self.inputs.append(var)
                if self._is_output_var(var):
                    self.outputs.append(var)

    def collect_glsl_stage_builtin_writes(
        self,
        node,
    ):
        """Record the built-in stage outputs and fragment results the entry point writes, and the uniforms it declares."""
        builtin_redeclaration_qualifiers = self.qualifier_only_builtin_qualifiers(node)
        self.apply_builtin_redeclaration_qualifiers(
            [*self.inputs, *self.outputs], builtin_redeclaration_qualifiers
        )

        fragment_writes_depth = self.fragment_main_writes_name(node, "gl_FragDepth")
        fragment_writes_sample_mask = self.fragment_main_writes_name(
            node, "gl_SampleMask"
        )
        vertex_writes_point_size = self.main_writes_name(
            node, "gl_PointSize", shader_type="vertex"
        )
        vertex_builtin_output_writes = {}
        if self.shader_type == "vertex":
            for builtin_name in self.VERTEX_BUILTIN_OUTPUT_TYPES:
                vertex_builtin_output_writes[builtin_name] = self.main_writes_name(
                    node, builtin_name, shader_type="vertex"
                )

        # Preserve source-declared/written vertex built-ins without inventing outputs
        # that the source never assigns. DirectX rejects an unwritten SV_Position
        # member when this CrossGL is lowered to HLSL.
        if self.shader_type == "vertex":
            output_names = {
                var.name for var in self.outputs if isinstance(var, VariableNode)
            }
            if "gl_Position" not in output_names and (
                "gl_Position" in builtin_redeclaration_qualifiers
                or vertex_builtin_output_writes.get("gl_Position", False)
            ):
                builtin = VariableNode(
                    "vec4",
                    "gl_Position",
                    qualifiers=self.builtin_output_qualifiers(
                        "gl_Position", builtin_redeclaration_qualifiers
                    ),
                    semantic="gl_Position",
                )
                self.outputs.append(builtin)
                output_names.add("gl_Position")

            for name, vtype in self.VERTEX_BUILTIN_OUTPUT_TYPES.items():
                if name == "gl_Position" or (
                    name not in builtin_redeclaration_qualifiers
                    and not (name == "gl_PointSize" and vertex_writes_point_size)
                    and not vertex_builtin_output_writes.get(name, False)
                ):
                    continue
                if name in output_names:
                    continue
                is_array = name in self.VERTEX_BUILTIN_ARRAY_OUTPUTS
                self.outputs.append(
                    VariableNode(
                        vtype,
                        name,
                        qualifiers=self.builtin_output_qualifiers(
                            name, builtin_redeclaration_qualifiers
                        ),
                        array_sizes=[None] if is_array else [],
                        is_array=is_array,
                        semantic=name,
                    )
                )
                output_names.add(name)

        # Ensure fragment outputs include gl_FragColor if no outputs declared
        fragment_entry_return = None
        if self.shader_type == "fragment":
            fragment_entry_return = next(
                (
                    function
                    for function in getattr(node, "functions", []) or []
                    if getattr(function, "name", None) == "main"
                    and self.is_fragment_explicit_entry_return(function)
                ),
                None,
            )

        if (
            self.shader_type == "fragment"
            and not self.outputs
            and not fragment_writes_depth
            and not fragment_writes_sample_mask
            and fragment_entry_return is None
        ):
            builtin = VariableNode(
                "vec4", "gl_FragColor", qualifiers=["out"], semantic="gl_FragColor"
            )
            self.outputs.append(builtin)
        fragment_uses_direct_outputs = self.fragment_uses_direct_output_declarations(
            fragment_writes_depth, fragment_writes_sample_mask
        )

        for uniform in node.uniforms:
            if self._is_unsupported_extension_resource_type(uniform.vtype):
                continue
            self.uniform_vars.append(uniform)
        return fragment_uses_direct_outputs

    def render_glsl_shader_interface(
        self,
        node,
    ):
        """Open the shader and render the preprocessor lines, structs and stage interface structs."""
        result = ""
        preprocessor = getattr(node, "preprocessor", []) or []
        if preprocessor:
            for line in preprocessor:
                result += f"{line}\n"
            result += "\n"
        result += "shader main {\n"

        # Generate struct definitions
        crossgl_structs = self.structs_for_crossgl_output(node.structs)
        for struct in crossgl_structs:
            if struct.name in self.converted_ssbo_struct_names:
                continue
            if self.is_push_constant_interface_block_struct(struct) or (
                self.is_descriptor_set_uniform_block_struct(struct)
                and not self.is_arrayed_descriptor_set_uniform_block_struct(struct)
            ):
                continue
            result += self.indent_str + self.generate_struct(struct) + "\n\n"

        # Generate input struct if needed
        if self.inputs and self.shader_type in (
            "vertex",
            "fragment",
        ):
            result += self.indent_str + f"struct {self.stage_struct_name()}Input {{\n"
            self.increase_indent()
            for input_var in self.inputs:
                result += self.indent() + self.generate_stage_struct_member(input_var)
            self.decrease_indent()
            result += self.indent_str + "};\n\n"

        # Generate output struct for vertex-like stages
        if self.outputs and self.shader_type == "vertex":
            result += self.indent_str + f"struct {self.stage_struct_name()}Output {{\n"
            self.increase_indent()
            for output_var in self.outputs:
                result += self.indent() + self.generate_stage_struct_member(output_var)
            self.decrease_indent()
            result += self.indent_str + "};\n\n"
        return result, crossgl_structs

    def render_glsl_shader_globals(
        self,
        node,
        crossgl_structs,
        fragment_uses_direct_outputs,
        result,
    ):
        """Render the uniforms, constants, buffer blocks, globals and direct stage outputs."""
        if self.uniform_vars:
            resource_uniforms = [
                u for u in self.uniform_vars if self._is_resource_type(u.vtype)
            ]
            subroutine_uniforms = [
                u for u in self.uniform_vars if self.is_subroutine_qualified(u)
            ]
            resource_uniforms = [
                u for u in resource_uniforms if not self.is_subroutine_qualified(u)
            ]
            data_uniforms = [
                u
                for u in self.uniform_vars
                if not self._is_resource_type(u.vtype)
                and not self.is_subroutine_qualified(u)
            ]
            push_constant_blocks = {}
            descriptor_set_uniform_blocks = {}
            arrayed_descriptor_set_uniform_blocks = []
            ordinary_data_uniforms = []
            for uniform in data_uniforms:
                if self.is_push_constant_uniform(uniform):
                    block_name = self.push_constant_block_name(uniform)
                    push_constant_blocks.setdefault(block_name, []).append(uniform)
                elif self.is_arrayed_descriptor_set_uniform_block(uniform):
                    arrayed_descriptor_set_uniform_blocks.append(uniform)
                else:
                    block_name = self.descriptor_set_uniform_block_name(uniform)
                    if block_name:
                        descriptor_set_uniform_blocks.setdefault(block_name, []).append(
                            uniform
                        )
                    else:
                        ordinary_data_uniforms.append(uniform)

            for uniform in resource_uniforms:
                var_type = self.convert_type(uniform.vtype)
                var_name = uniform.name
                attributes = self.image_resource_attribute_suffix(
                    uniform
                ) + self.precision_qualifier_attribute_suffix(uniform)
                array_suffix = self.array_suffix(uniform)
                result += (
                    self.indent_str
                    + f"{var_type} {var_name}{array_suffix}{attributes};\n"
                )

            for uniform in subroutine_uniforms:
                result += (
                    self.indent_str
                    + self.generate_variable_declaration(uniform)
                    + ";\n"
                )

            for block_name, uniforms in push_constant_blocks.items():
                result += self.indent_str + self.generate_push_constant_block(
                    block_name, uniforms
                )

            for block_name, uniforms in descriptor_set_uniform_blocks.items():
                result += self.indent_str + self.generate_descriptor_set_uniform_block(
                    block_name, uniforms
                )

            for uniform in arrayed_descriptor_set_uniform_blocks:
                result += (
                    self.indent_str
                    + self.generate_arrayed_descriptor_set_uniform_block(uniform)
                )

            if ordinary_data_uniforms:
                ordinary_block_name = self.fallback_uniform_block_name(
                    node,
                    crossgl_structs,
                    {
                        *push_constant_blocks,
                        *descriptor_set_uniform_blocks,
                        *(
                            self.uniform_block_name(uniform)
                            for uniform in arrayed_descriptor_set_uniform_blocks
                        ),
                    },
                )
                result += self.indent_str + f"cbuffer {ordinary_block_name} {{\n"
                self.increase_indent()
                for uniform in ordinary_data_uniforms:
                    var_type = self.convert_type(uniform.vtype)
                    var_name = uniform.name
                    array_suffix = self.array_suffix(uniform)
                    attributes = self.vulkan_memory_model_qualifier_attribute_suffix(
                        uniform
                    )
                    result += (
                        self.indent()
                        + f"{var_type} {var_name}{array_suffix}{attributes};\n"
                    )
                self.decrease_indent()
                result += self.indent_str + "};\n"

            result += "\n"

        # Generate global constants
        for const_var in getattr(node, "constant", []) or []:
            result += (
                self.indent_str
                + self.generate_variable_declaration(const_var, array_on_type=True)
                + ";\n"
            )
        if getattr(node, "constant", []):
            result += "\n"

        # Generate global variables
        for ssbo_struct in self.anonymous_ssbo_block_structs:
            result += (
                self.indent_str
                + self.anonymous_ssbo_block_declaration(ssbo_struct)
                + "\n"
            )

        for global_var in getattr(node, "global_variables", []) or []:
            if self.is_qualifier_only_builtin_declaration(global_var):
                continue
            if self.is_buffer_reference_forward_declaration(global_var):
                continue
            if self.is_anonymous_ssbo_member_variable(global_var):
                continue
            builtin_specialization = self.builtin_specialization_constant_declaration(
                global_var
            )
            if builtin_specialization is not None:
                result += self.indent_str + builtin_specialization + ";\n"
                continue
            structured_buffer_decl = self.structured_buffer_declaration(global_var)
            if structured_buffer_decl is not None:
                result += self.indent_str + structured_buffer_decl + ";\n"
                continue
            ssbo_block_decl = self.ssbo_block_declaration(global_var)
            if ssbo_block_decl is not None:
                result += self.indent_str + ssbo_block_decl + ";\n"
                continue
            result += (
                self.indent_str
                + self.private_global_prefix(global_var)
                + self.generate_variable_declaration(global_var)
                + ";\n"
            )
        if getattr(node, "global_variables", []):
            result += "\n"

        if fragment_uses_direct_outputs and self.outputs:
            for output_var in self.outputs:
                result += (
                    self.indent_str
                    + self.generate_variable_declaration(
                        output_var,
                        array_before_attributes=True,
                    )
                    + ";\n"
                )
            result += "\n"

        if self.shader_type in self.NON_STRUCT_STAGE_TYPES and (
            self.inputs or self.outputs
        ):
            for interface_var in [*self.inputs, *self.outputs]:
                if self.is_graphics_interface_block_variable(interface_var):
                    continue
                result += (
                    self.indent_str
                    + self.generate_variable_declaration(interface_var)
                    + ";\n"
                )
            result += "\n"
        return result

    def render_glsl_stage_block(
        self,
        node,
        fragment_uses_direct_outputs,
        result,
    ):
        """Render the stage block: layouts, helper functions and the entry point."""
        result += self.indent_str + f"{self.shader_type} {{\n"

        layouts = getattr(node, "layouts", []) or []
        if layouts:
            self.increase_indent()
            for layout in layouts:
                result += self.indent() + f"{self.format_layout(layout)};\n"
            result += "\n"
            self.decrease_indent()

        main_function = None
        other_functions = []

        for function in node.functions:
            if function.name == "main":
                main_function = function
            else:
                other_functions.append(function)

        shadertoy_main_image = None
        if main_function is None and self.shader_type == "fragment":
            shadertoy_main_image = self.find_shadertoy_main_image(other_functions)

        defined_function_signatures = {
            self.function_signature(function)
            for function in other_functions
            if not self.is_function_prototype(function)
        }

        # Generate auxiliary functions first
        for function in other_functions:
            if (
                self.is_function_prototype(function)
                and self.function_signature(function) in defined_function_signatures
            ):
                continue
            self.increase_indent()
            result += self.indent() + self.generate_function(function) + "\n\n"
            self.decrease_indent()

        # Generate the main function if it exists
        if main_function or shadertoy_main_image:
            self.increase_indent()
            input_parameter = (
                f"{self.stage_struct_name()}Input input" if self.inputs else ""
            )

            # Determine function signature based on shader type
            if self.shader_type == "vertex":
                result += (
                    self.indent()
                    + f"{self.stage_struct_name()}Output main({input_parameter})"
                )
            elif self.shader_type == "fragment":
                if self.is_fragment_explicit_entry_return(main_function):
                    return_type = self.convert_type(main_function.return_type)
                    semantic = self.semantic_attribute_suffix(
                        getattr(main_function, "semantic", None)
                    )
                    result += (
                        self.indent()
                        + f"{return_type} main({input_parameter})"
                        + semantic
                    )
                elif fragment_uses_direct_outputs:
                    result += self.indent() + f"void main({input_parameter})"
                else:
                    output_type = "vec4"
                    output_attributes = ""
                    if self.outputs:
                        output_var = self.outputs[0]
                        output_type = self.convert_type(output_var.vtype)
                        output_attributes = self.fragment_return_attribute_suffix(
                            output_var, 0
                        )
                    result += (
                        self.indent()
                        + f"{output_type} main({input_parameter})"
                        + output_attributes
                    )
            elif self.shader_type in self.NON_STRUCT_STAGE_TYPES:
                result += self.indent() + "void main()"
            else:
                result += (
                    self.indent()
                    + f"{self.stage_struct_name()}Output main({input_parameter})"
                )

            result += " {\n"

            self.increase_indent()

            # For vertex shaders, create the output struct
            if self.shader_type == "vertex":
                result += self.indent() + f"{self.stage_struct_name()}Output output;\n"

            # For fragment shaders, declare a local output if assignments are used
            if (
                self.shader_type == "fragment"
                and self.outputs
                and not fragment_uses_direct_outputs
                and shadertoy_main_image is None
            ):
                output_type = self.convert_type(self.outputs[0].vtype)
                output_name = self.outputs[0].name
                result += self.indent() + f"{output_type} {output_name};\n"

            # Generate statements for the main function
            if shadertoy_main_image is not None:
                for statement in self.generate_shadertoy_main_image_entrypoint(
                    shadertoy_main_image, fragment_uses_direct_outputs
                ):
                    result += self.indent() + statement + "\n"
            else:
                # GLSL's void main() may leave early with `return;`; the entry now returns its outputs.
                self.entry_bare_return_value = None
                if self.shader_type == "vertex":
                    self.entry_bare_return_value = "output"
                elif (
                    self.shader_type == "fragment"
                    and self.outputs
                    and not fragment_uses_direct_outputs
                    and not self.is_fragment_explicit_entry_return(main_function)
                ):
                    self.entry_bare_return_value = self.outputs[0].name
                # The entry's locals get a scope of their own, as other functions' do, so that a local that
                # shadows a member of an anonymous buffer block is not read as that member.
                self.push_variable_type_scope()
                try:
                    for statement in self.generate_statement_sequence(
                        main_function.body
                    ):
                        result += self.indent() + statement + "\n"
                finally:
                    self.pop_variable_type_scope()
                    self.entry_bare_return_value = None

                if self.shader_type in ("vertex",) and not any(
                    isinstance(stmt, ReturnNode) for stmt in main_function.body
                ):
                    result += self.indent() + "return output;\n"

                if (
                    self.shader_type == "fragment"
                    and not fragment_uses_direct_outputs
                    and not self.is_fragment_explicit_entry_return(main_function)
                    and not any(
                        isinstance(stmt, ReturnNode) for stmt in main_function.body
                    )
                ):
                    output_name = (
                        self.outputs[0].name if self.outputs else "gl_FragColor"
                    )
                    result += self.indent() + f"return {output_name};\n"

            self.decrease_indent()
            result += self.indent() + "}\n"
            self.decrease_indent()
        return result
