"""GLSL stage-interface declarations of program structs for ``GLSLCodeGen.declare_glsl_program_struct``."""


class OpenGLStructDeclarationMixin:
    """Stage-interface struct declaration members of ``GLSLCodeGen``."""

    def declare_glsl_vertex_output_interface_struct(
        self,
        structs_emission,
        node,
    ):
        """Declare the legacy ``VSOutput`` struct as vertex outputs and fragment inputs."""
        emitted_io = False
        if (
            node.name in self.vertex_output_struct_names
            and structs_emission.emit_vertex_io
        ):
            structs_emission.stage_io_code += self.wrap_stage_guard(
                self.generate_vertex_output_declarations(node),
                "vertex",
            )
            emitted_io = True
        if (
            node.name in self.fragment_input_struct_names
            and structs_emission.emit_fragment_io
        ):
            structs_emission.stage_io_code += self.wrap_stage_guard(
                self.generate_fragment_input_declarations(node),
                "fragment",
            )
            emitted_io = True
        if (
            node.name in self.fragment_output_struct_names
            and structs_emission.emit_fragment_io
        ):
            structs_emission.stage_io_code += self.wrap_stage_guard(
                self.generate_fragment_output_declarations(node),
                "fragment",
            )
            emitted_io = True
        if not emitted_io:
            if structs_emission.emit_vertex_io and structs_emission.legacy_vertex_names:
                structs_emission.stage_io_code += self.wrap_stage_guard(
                    self.generate_legacy_output_declarations(node),
                    "vertex",
                )
                if self.stage_io_struct_referenced_outside_stage_entries(
                    structs_emission.ast, node.name
                ):
                    structs_emission.data_struct_nodes.append(node)
            else:
                structs_emission.data_struct_nodes.append(node)
        elif self.should_emit_flattened_stage_io_struct(
            structs_emission.ast,
            node.name,
            emitted_io,
            structs_emission.emit_graphics_io,
        ):
            structs_emission.data_struct_nodes.append(node)

    def declare_glsl_vertex_output_struct(
        self,
        structs_emission,
        node,
    ):
        """Declare a vertex stage's returned struct as vertex outputs."""
        emitted_io = False
        if structs_emission.emit_vertex_io:
            structs_emission.stage_io_code += self.wrap_stage_guard(
                self.generate_vertex_output_declarations(node),
                "vertex",
            )
            emitted_io = True
        if (
            node.name in self.fragment_input_struct_names
            and structs_emission.emit_fragment_io
        ):
            structs_emission.stage_io_code += self.wrap_stage_guard(
                self.generate_fragment_input_declarations(node),
                "fragment",
            )
            emitted_io = True
        if (
            node.name in self.fragment_output_struct_names
            and structs_emission.emit_fragment_io
        ):
            structs_emission.stage_io_code += self.wrap_stage_guard(
                self.generate_fragment_output_declarations(node),
                "fragment",
            )
            emitted_io = True
        if self.should_emit_flattened_stage_io_struct(
            structs_emission.ast,
            node.name,
            emitted_io,
            structs_emission.emit_graphics_io,
        ):
            structs_emission.data_struct_nodes.append(node)

    def declare_glsl_fragment_input_struct(
        self,
        structs_emission,
        node,
    ):
        """Declare a fragment stage's parameter struct as fragment inputs."""
        if structs_emission.emit_fragment_io:
            structs_emission.stage_io_code += self.wrap_stage_guard(
                self.generate_fragment_input_declarations(node),
                "fragment",
            )
            if node.name in self.fragment_output_struct_names:
                structs_emission.stage_io_code += self.wrap_stage_guard(
                    self.generate_fragment_output_declarations(node),
                    "fragment",
                )
        if node.name in self.fragment_output_struct_names:
            structs_emission.data_struct_nodes.append(node)
        elif not structs_emission.emit_graphics_io or (
            structs_emission.emit_fragment_io
            and self.stage_io_struct_referenced_outside_stage_entries(
                structs_emission.ast, node.name
            )
        ):
            structs_emission.data_struct_nodes.append(node)
