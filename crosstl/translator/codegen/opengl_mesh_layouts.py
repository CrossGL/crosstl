"""Mesh layouts in the GLSL generator: stage entry constant blocks, mesh output layouts and limits, stage attribute arguments, topology and task payload parameters."""

import re
from types import SimpleNamespace

from ..ast import (
    ArrayNode,
    StructNode,
)
from ..refusals import OpenGLRefusal
from ..resource_identity import resource_identity_marker
from ..validation import (
    collect_cbuffer_declaration_name_conflicts,
    collect_cbuffer_member_global_conflicts,
    collect_duplicate_cbuffer_member_names,
    collect_duplicate_cbuffer_names,
)
from .array_utils import format_c_style_array_declaration
from .glsl_exact_lookups import glsl_mesh_intrinsic_call
from .glsl_fragment_interlock import fragment_interlock_layout
from .glsl_host_workgroup_size import (
    glsl_host_workgroup_layout,
    glsl_program_host_workgroup_size_ids,
)
from .glsl_tessellation_modes import merge_control_stage_modes
from .host_workgroup_size_ids import is_host_sized_workgroup
from .opengl_values import (
    GLSL_SPIRV_GUARD,
    OpenGLWorkgroupSizeError,
)
from .push_constant_blocks import glsl_push_constant_cbuffer
from .specialized_workgroup_size import workgroup_size_specialization_ids
from .stage_utils import (
    collect_stage_local_structs,
    compute_local_size,
)


class OpenGLMeshLayoutMixin:
    """Mesh layout members of ``GLSLCodeGen``."""

    def unique_stage_entry_constant_value_block_name(self, block_name, used_names):
        return self.unique_stage_entry_constant_value_identifier(
            block_name,
            used_names,
            "Entry_Value_Args",
        )

    def stage_entry_constant_value_parameter_member_name(
        self,
        block_name,
        parameter_name,
        used_names,
        parameter_member_name_counts=None,
        force_block_qualified=False,
    ):
        source_parameter_name = parameter_name or "value"
        parameter_name = self.glsl_sanitized_identifier_base(
            source_parameter_name,
            fallback="value",
        )
        if (
            not force_block_qualified
            and parameter_name
            and not self.glsl_identifier_requires_escape(parameter_name)
            and parameter_name not in used_names
            and (parameter_member_name_counts or {}).get(source_parameter_name, 0) == 1
        ):
            return parameter_name
        block_name = self.glsl_sanitized_identifier_base(
            block_name,
            fallback="Entry_Value_Args",
        )
        base_name = "_".join(
            part for part in (block_name, parameter_name or "value") if part
        )
        return self.unique_stage_entry_constant_value_identifier(
            base_name,
            used_names,
            "Entry_Value",
        )

    def stage_entry_constant_value_parameter_block_name(self, func, parameter_name):
        function_name = self.glsl_sanitized_identifier_base(
            getattr(func, "name", None),
            fallback="entry",
        )
        parameter_name = self.glsl_sanitized_identifier_base(
            parameter_name,
            fallback="value",
        )
        block_name = "_".join(
            part for part in (function_name, parameter_name, "Args") if part
        )
        if not block_name:
            block_name = "Entry_Value_Args"
        return block_name

    @staticmethod
    def glsl_uniform_member_orientation(member):
        """``layout(row_major) `` for a block member whose matrices are stored row by row (GLSL 4.60 section
        4.4.5), which sets the bytes the host writes."""
        names = {str(q).lower() for q in getattr(member, "qualifiers", None) or []}
        names.update(
            str(getattr(attribute, "name", "")).lower()
            for attribute in getattr(member, "attributes", None) or []
        )
        if "row_major" in names and "column_major" not in names:
            return "layout(row_major) "
        return ""

    def generate_cbuffers(self, ast, target_stage=None):
        code = ""
        cbuffers = self.glsl_cbuffer_nodes(ast, target_stage)
        for cbuffer in cbuffers:
            for member in getattr(cbuffer, "members", []) or []:
                self.union_storage.validate_buffer(self.member_type_name(member))
        duplicate_names = collect_duplicate_cbuffer_names(cbuffers)
        if duplicate_names:
            names = ", ".join(sorted(duplicate_names))
            raise OpenGLRefusal(f"Duplicate cbuffer name(s) in OpenGL output: {names}")

        cbuffer_ast = SimpleNamespace(
            cbuffers=cbuffers,
            structs=list(getattr(ast, "structs", []) or [])
            + collect_stage_local_structs(ast, target_stage),
            global_variables=getattr(ast, "global_variables", []) or [],
            constants=getattr(ast, "constants", []) or [],
        )

        declaration_conflicts = collect_cbuffer_declaration_name_conflicts(cbuffer_ast)
        if declaration_conflicts:
            names = ", ".join(sorted(declaration_conflicts))
            raise OpenGLRefusal(
                f"Cbuffer name(s) conflict with existing OpenGL declaration(s): {names}"
            )

        duplicate_members = collect_duplicate_cbuffer_member_names(cbuffers)
        if duplicate_members:
            names = ", ".join(sorted(duplicate_members))
            raise OpenGLRefusal(
                f"Ambiguous cbuffer member name(s) in OpenGL output: {names}"
            )

        global_member_conflicts = collect_cbuffer_member_global_conflicts(cbuffer_ast)
        if global_member_conflicts:
            names = ", ".join(sorted(global_member_conflicts))
            raise OpenGLRefusal(
                "Cbuffer member name(s) conflict with OpenGL global declaration(s): "
                f"{names}"
            )
        cbuffer_binding_cursors = {}
        used_cbuffer_bindings = {}
        for node in cbuffers:
            explicit_binding = self.explicit_resource_binding_index(node)
            if explicit_binding is None:
                continue
            if self.should_remap_stage_entry_cbuffer_binding_conflict(
                used_cbuffer_bindings,
                explicit_binding,
                node.name,
            ):
                continue
            self.reserve_resource_binding_range(
                used_cbuffer_bindings,
                "OpenGL",
                "uniform buffer binding",
                explicit_binding,
                1,
                node.name,
            )
        for node in cbuffers:
            explicit_binding = self.explicit_resource_binding_index(node)
            if explicit_binding is not None:
                if self.should_remap_stage_entry_cbuffer_binding_conflict(
                    used_cbuffer_bindings,
                    explicit_binding,
                    node.name,
                ):
                    resource_binding = self.next_available_resource_binding_from(
                        used_cbuffer_bindings,
                        "uniform buffer binding",
                        explicit_binding,
                        1,
                    )
                else:
                    resource_binding = explicit_binding
            else:
                resource_binding = self.next_available_resource_binding(
                    used_cbuffer_bindings,
                    cbuffer_binding_cursors,
                    "uniform buffer binding",
                    1,
                )
            self.reserve_resource_binding_range(
                used_cbuffer_bindings,
                "OpenGL",
                "uniform buffer binding",
                resource_binding,
                1,
                node.name,
            )
            self.advance_resource_binding(
                cbuffer_binding_cursors,
                "uniform buffer binding",
                resource_binding,
                1,
            )
            layout = self.glsl_resource_layout_prefix(
                "std140", binding=resource_binding, node=node
            )
            block_name = self.glsl_module_identifier_name(node.name)
            member_names = {
                self.glsl_global_identifier_name(member.name)
                for member in getattr(node, "members", None) or []
            }
            if block_name in member_names:
                # HLSL's ``cbuffer ubo { UBO ubo; }`` names the buffer like its member; a GLSL block name shares
                # the global namespace with the members of an unnamed block, so the block takes another name.
                block_name = self.glsl_generated_module_identifier(
                    ("cbuffer-block", node.name), f"{block_name}_block"
                )
            self.record_glsl_resource_binding(
                node,
                node.name,
                block_name,
                "constant-buffer",
                "uniform buffer binding",
                resource_binding,
            )
            code += resource_identity_marker(node, block_name)
            push_constant = glsl_push_constant_cbuffer(self, node, layout, block_name)
            if push_constant is not None:
                code += push_constant
                continue
            register_block = self.glsl_hlsl_cbuffer_block(node, layout, block_name)
            if register_block is not None:
                code += register_block
                continue
            if isinstance(node, StructNode):
                code += f"{layout} uniform {block_name} {{\n"
                members = getattr(node, "members", [])
                for member in members:
                    member_name = self.glsl_global_identifier_name(member.name)
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        orientation = self.glsl_uniform_member_orientation(member)
                        if member.size:
                            code += f"    {orientation}{self.map_type(element_type)} {member_name}[{member.size}];\n"
                        else:
                            # Dynamic arrays in uniform blocks need special handling in GLSL
                            code += f"    {orientation}{self.map_type(element_type)} {member_name}[];\n"
                    else:
                        if hasattr(member, "member_type"):
                            member_type = self.map_type(member.member_type)
                        else:
                            member_type = self.map_type(
                                getattr(member, "vtype", "float")
                            )
                        declaration = format_c_style_array_declaration(
                            member_type, member_name
                        )
                        orientation = self.glsl_uniform_member_orientation(member)
                        code += f"    {orientation}{declaration};\n"
                code += "};\n"
            elif hasattr(node, "name") and hasattr(
                node, "members"
            ):  # CbufferNode handling
                code += f"{layout} uniform {block_name} {{\n"
                for member in node.members:
                    member_name = self.glsl_global_identifier_name(member.name)
                    if isinstance(member, ArrayNode):
                        element_type = getattr(
                            member, "element_type", getattr(member, "vtype", "float")
                        )
                        orientation = self.glsl_uniform_member_orientation(member)
                        if member.size:
                            code += f"    {orientation}{self.map_type(element_type)} {member_name}[{member.size}];\n"
                        else:
                            # Dynamic arrays in uniform blocks need special handling in GLSL
                            code += f"    {orientation}{self.map_type(element_type)} {member_name}[];\n"
                    else:
                        if hasattr(member, "member_type"):
                            member_type = self.map_type(member.member_type)
                        else:
                            member_type = self.map_type(
                                getattr(member, "vtype", "float")
                            )
                        declaration = format_c_style_array_declaration(
                            member_type, member_name
                        )
                        orientation = self.glsl_uniform_member_orientation(member)
                        code += f"    {orientation}{declaration};\n"
                code += "};\n"
        return code

    def generate_compute_layout(self, execution_config=None):
        if is_host_sized_workgroup(execution_config):
            return glsl_host_workgroup_layout(
                glsl_program_host_workgroup_size_ids(self), GLSL_SPIRV_GUARD
            )
        x, y, z = self.validate_glsl_compute_workgroup_size(
            compute_local_size(execution_config)
        )
        fixed = f"local_size_x = {x}, local_size_y = {y}, local_size_z = {z}"
        ids = workgroup_size_specialization_ids(execution_config, "OpenGL")
        specialized = "".join(
            f", local_size_{axis}_id = {spec_id}"
            for axis, spec_id in zip("xyz", ids)
            if spec_id is not None
        )
        if not specialized:
            return f"layout({fixed}) in;\n"
        # local_size_*_id exists only when the shader is compiled to SPIR-V (GLSL 4.60 section 4.4.1.4,
        # GL_KHR_vulkan_glsl); a driver compiling the source keeps the defaults, as for layout(constant_id).
        return (
            f"{GLSL_SPIRV_GUARD}layout({fixed}{specialized}) in;\n"
            f"#else\nlayout({fixed}) in;\n#endif\n"
        )

    def validate_glsl_compute_workgroup_size(self, values):
        resolved = []
        for value in values:
            text = str(value).strip()
            if not re.fullmatch(r"[+-]?[0-9]+", text):
                return tuple(values)
            resolved.append(int(text))
        if any(value <= 0 for value in resolved):
            raise OpenGLWorkgroupSizeError(
                "OpenGL compute workgroup dimensions must be positive integers",
                workgroup_size=resolved,
                reason="non-positive-dimension",
            )
        if resolved[0] > 1024 or resolved[1] > 1024 or resolved[2] > 64:
            raise OpenGLWorkgroupSizeError(
                "OpenGL compute workgroup dimensions exceed portable limits "
                "(x <= 1024, y <= 1024, z <= 64)",
                workgroup_size=resolved,
                reason="dimension-limit-exceeded",
            )
        if resolved[0] * resolved[1] * resolved[2] > 1024:
            raise OpenGLWorkgroupSizeError(
                "OpenGL compute workgroup size must not exceed 1024 invocations",
                workgroup_size=resolved,
                reason="invocation-count-limit-exceeded",
            )
        return tuple(resolved)

    def generate_stage_layout(self, shader_type, func, stage_layout_qualifiers=None):
        if shader_type == "fragment":
            return self.generate_fragment_stage_layout(func, stage_layout_qualifiers)
        if shader_type == "geometry":
            return self.generate_geometry_stage_layout(func, stage_layout_qualifiers)
        if shader_type == "tessellation_control":
            return self.generate_tessellation_control_layout(
                func, stage_layout_qualifiers
            )
        if shader_type == "tessellation_evaluation":
            return self.generate_tessellation_evaluation_layout(
                func, stage_layout_qualifiers
            )
        if shader_type == "mesh":
            return self.generate_mesh_stage_layout(func, stage_layout_qualifiers)
        return ""

    def generate_fragment_stage_layout(self, func, stage_layout_qualifiers=None):
        code = ""
        if self.glsl_stage_bare_attribute(
            func, {"early_fragment_tests"}, stage_layout_qualifiers, "in"
        ):
            code += "layout(early_fragment_tests) in;\n"
        code += fragment_interlock_layout(self, func, stage_layout_qualifiers)

        blend_support = self.glsl_fragment_blend_support_layout_parts(
            func, stage_layout_qualifiers
        )
        if blend_support:
            code += f"layout({', '.join(blend_support)}) out;\n"
        return code

    def glsl_fragment_blend_support_layout_parts(
        self, func, stage_layout_qualifiers=None
    ):
        parts = []
        seen = set()

        def add_if_blend_support(attr):
            attr_name = self.glsl_stage_control_attribute_name(attr)
            if attr_name not in self.GLSL_BLEND_SUPPORT_LAYOUT_NAMES:
                return
            if getattr(attr, "arguments", []) or attr_name in seen:
                return
            seen.add(attr_name)
            parts.append(attr_name)

        for attr in getattr(func, "attributes", []) or []:
            add_if_blend_support(attr)
        for attr_name in getattr(self, "fragment_blend_support_layout_parts", []):
            if attr_name not in seen:
                seen.add(attr_name)
                parts.append(attr_name)
        for attr in self.glsl_stage_layout_entries(stage_layout_qualifiers, "out"):
            add_if_blend_support(attr)

        return parts

    def glsl_fragment_output_blend_support_layout_parts(self, nodes):
        parts = []
        seen = set()

        for node in nodes:
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(node, "qualifiers", []) or []
            }
            if "out" not in qualifiers:
                continue
            for attr in getattr(node, "attributes", []) or []:
                attr_name = self.glsl_stage_control_attribute_name(attr)
                if attr_name not in self.GLSL_BLEND_SUPPORT_LAYOUT_NAMES:
                    continue
                if getattr(attr, "arguments", []) or attr_name in seen:
                    continue
                seen.add(attr_name)
                parts.append(attr_name)

        return parts

    def generate_geometry_stage_layout(self, func, stage_layout_qualifiers=None):
        input_bare_names = {
            "points",
            "lines",
            "lines_adjacency",
            "triangles",
            "triangles_adjacency",
        }
        output_bare_names = {"points", "line_strip", "triangle_strip"}
        if self.glsl_stage_has_attribute(
            func, "inputtopology", stage_layout_qualifiers, "in"
        ) or self.glsl_stage_has_bare_attribute(
            func,
            input_bare_names - {"points"},
            stage_layout_qualifiers,
            "in",
        ):
            input_bare_names.remove("points")
        if self.glsl_stage_has_attribute(
            func, "outputtopology", stage_layout_qualifiers, "out"
        ) or self.glsl_stage_has_bare_attribute(
            func,
            output_bare_names - {"points"},
            stage_layout_qualifiers,
            "out",
        ):
            output_bare_names.remove("points")

        input_primitive = self.glsl_stage_consistent_layout_value(
            "geometry",
            "input topology",
            func,
            explicit_attribute_name="inputtopology",
            bare_names=input_bare_names,
            value_mapper=self.glsl_geometry_input_topology,
            stage_layout_qualifiers=stage_layout_qualifiers,
            direction="in",
        )
        output_primitive = self.glsl_stage_consistent_layout_value(
            "geometry",
            "output topology",
            func,
            explicit_attribute_name="outputtopology",
            bare_names=output_bare_names,
            value_mapper=self.glsl_geometry_output_topology,
            stage_layout_qualifiers=stage_layout_qualifiers,
            direction="out",
        )
        max_vertices = self.glsl_stage_positive_int_layout_argument(
            "geometry", func, "max_vertices", stage_layout_qualifiers, "out"
        )
        if max_vertices is None:
            max_vertices = self.glsl_stage_positive_int_layout_argument(
                "geometry",
                func,
                "maxvertexcount",
                stage_layout_qualifiers,
                "out",
                layout_name="max_vertices",
            )
        invocations = self.glsl_stage_positive_int_layout_argument(
            "geometry", func, "invocations", stage_layout_qualifiers, "in"
        )

        code = ""
        input_layout = []
        if input_primitive:
            input_layout.append(input_primitive)
        if invocations is not None:
            input_layout.append(f"invocations = {invocations}")
        if input_layout:
            code += f"layout({', '.join(input_layout)}) in;\n"

        output_layout = []
        if output_primitive:
            output_layout.append(output_primitive)
        output_layout.extend(
            self.glsl_stage_stream_layout_parts(
                stage_layout_qualifiers, "out", skip_first=True
            )
        )
        if max_vertices is not None:
            output_layout.append(f"max_vertices = {max_vertices}")
        if output_layout:
            code += f"layout({', '.join(output_layout)}) out;\n"
        for stream_layout in self.glsl_stage_stream_layout_parts(
            stage_layout_qualifiers, "out", skip_first=False
        )[1:]:
            code += f"layout({stream_layout}) out;\n"
        return code

    def glsl_stage_stream_layout_parts(
        self, stage_layout_qualifiers=None, direction=None, skip_first=False
    ):
        stream_parts = []
        for attr in self.glsl_stage_attributes(
            None, "stream", stage_layout_qualifiers, direction
        ):
            arguments = getattr(attr, "arguments", []) or []
            if len(arguments) != 1:
                continue
            stream_parts.append(
                f"stream = {self.glsl_attribute_argument_source(arguments[0])}"
            )
        if skip_first:
            return stream_parts[:1]
        return stream_parts

    def glsl_stage_positive_int_layout_argument(
        self,
        stage_name,
        func,
        attribute_name,
        stage_layout_qualifiers=None,
        direction=None,
        layout_name=None,
    ):
        arguments = self.glsl_stage_attribute_arguments(
            func, attribute_name, stage_layout_qualifiers, direction
        )
        if not arguments:
            return None
        if len(arguments) != 1:
            raise OpenGLRefusal(
                f"GLSL stage attribute {attribute_name} requires exactly one argument"
            )

        argument = arguments[0]
        literal = self.literal_int_value(argument, self.literal_int_constants)
        if not isinstance(literal, int) or isinstance(literal, bool) or literal <= 0:
            layout_name = layout_name or attribute_name
            raise OpenGLRefusal(
                f"GLSL {stage_name} {layout_name} layout requires a positive "
                "integer constant, got "
                f"{self.glsl_attribute_argument_source(argument)}"
            )
        return self.generate_expression(argument)

    def glsl_attribute_argument_source(self, argument):
        try:
            return self.generate_expression(argument)
        except Exception:
            return self.attribute_value_to_string(argument)

    def generate_tessellation_control_layout(self, func, stage_layout_qualifiers=None):
        self.validate_glsl_tessellation_patchconstantfunc(func, stage_layout_qualifiers)
        vertices = self.glsl_stage_positive_int_layout_argument(
            "tessellation_control",
            func,
            "vertices",
            stage_layout_qualifiers,
            "out",
        )
        if vertices is None:
            vertices = self.glsl_stage_positive_int_layout_argument(
                "tessellation_control",
                func,
                "outputcontrolpoints",
                stage_layout_qualifiers,
                "out",
                layout_name="vertices",
            )
        if vertices is None:
            return ""
        return f"layout(vertices = {vertices}) out;\n"

    def validate_glsl_tessellation_patchconstantfunc(
        self, func, stage_layout_qualifiers=None
    ):
        for attr in self.glsl_stage_attributes(
            func, "patchconstantfunc", stage_layout_qualifiers
        ):
            arguments = getattr(attr, "arguments", []) or []
            if len(arguments) != 1:
                raise OpenGLRefusal(
                    "GLSL tessellation_control stage patchconstantfunc "
                    "requires exactly one function name"
                )
            patch_name = self.attribute_value_to_string(arguments[0])
            self.refuse_lowering(
                "GLSL tessellation_control stage patchconstantfunc "
                f"'{patch_name}' is unsupported; write tessellation factors "
                "directly to gl_TessLevelOuter and gl_TessLevelInner in the "
                "tessellation_control entry point"
            )

    def refuse_glsl_hlsl_patch_parameters(self, func, stage_name):
        """Refuse HLSL ``InputPatch``/``OutputPatch`` entry parameters, which GLSL has no declaration for."""
        for param in (
            getattr(func, "parameters", None) or getattr(func, "params", None) or []
        ):
            param_type = getattr(param, "param_type", None) or getattr(
                param, "vtype", None
            )
            type_name = str(self.type_node_name(param_type) or "")
            if type_name.split("<", 1)[0].strip().lower() in {
                "inputpatch",
                "outputpatch",
            }:
                self.refuse_lowering(
                    f"GLSL {stage_name} stage cannot declare HLSL control-point patch parameter "
                    f"'{getattr(param, 'name', '<unnamed>')}' of type '{type_name}'"
                )

    def generate_tessellation_evaluation_layout(
        self, func, stage_layout_qualifiers=None
    ):
        self.refuse_glsl_hlsl_patch_parameters(func, "tessellation_evaluation")
        layout_parts = []
        mapped_domain = self.glsl_stage_consistent_layout_value(
            "tessellation",
            "domain",
            func,
            explicit_attribute_name="domain",
            bare_names={"triangles", "quads", "isolines"},
            value_mapper=self.glsl_tessellation_domain,
            stage_layout_qualifiers=stage_layout_qualifiers,
            direction="in",
        )

        partitioning = self.glsl_stage_consistent_layout_value(
            "tessellation",
            "partitioning",
            func,
            explicit_attribute_name="partitioning",
            bare_names={
                "equal_spacing",
                "fractional_even_spacing",
                "fractional_odd_spacing",
            },
            value_mapper=self.glsl_tessellation_partitioning,
            stage_layout_qualifiers=stage_layout_qualifiers,
            direction="in",
        )

        winding = self.glsl_stage_consistent_layout_value(
            "tessellation",
            "winding",
            func,
            bare_names={"cw", "ccw"},
            stage_layout_qualifiers=stage_layout_qualifiers,
            direction="in",
        )
        output_topology = self.glsl_single_stage_attribute_argument(
            func, "outputtopology", stage_layout_qualifiers
        )
        topology_winding, topology_point_mode = (
            self.glsl_tessellation_output_topology_layout(
                output_topology, mapped_domain
            )
        )
        if topology_winding:
            if winding and winding != topology_winding:
                raise OpenGLRefusal(
                    "Conflicting GLSL tessellation outputtopology "
                    f"{output_topology} with winding {winding}"
                )
            winding = topology_winding

        point_mode = bool(
            self.glsl_stage_bare_attribute(
                func, {"point_mode"}, stage_layout_qualifiers, "in"
            )
            or topology_point_mode
        )
        mapped_domain, partitioning, winding, point_mode = merge_control_stage_modes(
            self, func, mapped_domain, partitioning, winding, point_mode
        )
        layout_parts.extend(
            part for part in (mapped_domain, partitioning, winding) if part
        )
        if point_mode:
            layout_parts.append("point_mode")

        if not layout_parts:
            return ""
        return f"layout({', '.join(layout_parts)}) in;\n"

    def generate_mesh_stage_layout(self, func, stage_layout_qualifiers=None):
        output_primitive = self.glsl_mesh_stage_output_topology_name(
            func, stage_layout_qualifiers
        )

        max_vertices = self.glsl_stage_consistent_positive_int_layout_argument(
            "mesh",
            func,
            ("max_vertices", "maxvertexcount"),
            stage_layout_qualifiers,
            "out",
            layout_name="max_vertices",
        )

        max_primitives = self.glsl_stage_consistent_positive_int_layout_argument(
            "mesh",
            func,
            ("max_primitives", "maxprimitivecount"),
            stage_layout_qualifiers,
            "out",
            layout_name="max_primitives",
        )

        declared = self.glsl_mesh_output_array_limits(func)
        if max_vertices is None:
            max_vertices = declared.get("vertices")
        if max_primitives is None:
            max_primitives = declared.get("primitives")

        layout_parts = []
        if output_primitive:
            layout_parts.append(output_primitive)
        if max_vertices is not None:
            layout_parts.append(f"max_vertices = {max_vertices}")
        if max_primitives is not None:
            layout_parts.append(f"max_primitives = {max_primitives}")

        if not layout_parts:
            return ""
        return f"layout({', '.join(layout_parts)}) out;\n"

    def glsl_mesh_output_array_limits(self, func):
        """The vertex and primitive limits that a mesh entry's output arrays declare.

        GL_EXT_mesh_shader requires ``max_vertices`` and ``max_primitives`` in the output layout; without the
        attributes, the ``@vertices``, ``@primitives`` and ``@indices`` array sizes give them, as they do in HLSL.
        """
        limits = {}
        for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
            if not self.is_mesh_output_parameter(param):
                continue
            count = self.mesh_output_parameter_count_value(param)
            if count is None:
                continue
            role = self.mesh_output_parameter_role(param)
            key = "vertices" if role == "vertices" else "primitives"
            limits[key] = max(count, limits.get(key, 0))
        return limits

    def glsl_mesh_stage_output_topology_name(self, func, stage_layout_qualifiers=None):
        return self.glsl_stage_consistent_layout_value(
            "mesh",
            "output topology",
            func,
            explicit_attribute_name="outputtopology",
            bare_names={"points", "lines", "triangles"},
            value_mapper=self.glsl_mesh_output_topology,
            stage_layout_qualifiers=stage_layout_qualifiers,
            direction="out",
        )

    def glsl_stage_bare_attribute(
        self, func, names, stage_layout_qualifiers=None, direction=None
    ):
        names = set(names)
        for attr in getattr(func, "attributes", []) or []:
            attr_name = self.glsl_stage_control_attribute_name(attr)
            if attr_name in names:
                if getattr(attr, "arguments", []):
                    raise OpenGLRefusal(
                        f"GLSL stage attribute {attr_name} does not accept arguments"
                    )
                return attr_name
        for attr in self.glsl_stage_layout_entries(stage_layout_qualifiers, direction):
            attr_name = self.glsl_stage_control_attribute_name(attr)
            if attr_name in names:
                if getattr(attr, "arguments", []):
                    raise OpenGLRefusal(
                        f"GLSL stage attribute {attr_name} does not accept arguments"
                    )
                return attr_name
        return None

    def glsl_stage_consistent_layout_value(
        self,
        stage_name,
        layout_name,
        func,
        explicit_attribute_name=None,
        bare_names=None,
        value_mapper=None,
        stage_layout_qualifiers=None,
        direction=None,
    ):
        bare_names = set(bare_names or ())
        choices = []
        for attr in self.glsl_stage_control_attributes(
            func, stage_layout_qualifiers, direction
        ):
            attr_name = self.glsl_stage_control_attribute_name(attr)
            if explicit_attribute_name is not None and (
                attr_name == explicit_attribute_name
            ):
                arguments = getattr(attr, "arguments", []) or []
                if len(arguments) != 1:
                    raise OpenGLRefusal(
                        "GLSL stage attribute "
                        f"{explicit_attribute_name} requires exactly one argument"
                    )
                value = self.attribute_value_to_string(arguments[0])
                source = f"{attr_name} {value}"
            elif attr_name in bare_names:
                if getattr(attr, "arguments", []):
                    raise OpenGLRefusal(
                        f"GLSL stage attribute {attr_name} does not accept arguments"
                    )
                value = attr_name
                source = attr_name
            else:
                continue

            mapped_value = value_mapper(value) if value_mapper else value
            choices.append((source, mapped_value))

        if not choices:
            return None

        first_source, first_mapped = choices[0]
        for source, mapped_value in choices[1:]:
            if mapped_value != first_mapped:
                raise OpenGLRefusal(
                    f"Conflicting GLSL {stage_name} {layout_name} layout "
                    f"{first_source} with {source}"
                )
        return first_mapped

    def glsl_stage_consistent_positive_int_layout_argument(
        self,
        stage_name,
        func,
        attribute_names,
        stage_layout_qualifiers=None,
        direction=None,
        layout_name=None,
    ):
        layout_name = layout_name or attribute_names[0]
        choices = []
        for attribute_name in attribute_names:
            arguments = self.glsl_stage_attribute_arguments(
                func, attribute_name, stage_layout_qualifiers, direction
            )
            if not arguments:
                continue
            if len(arguments) != 1:
                raise OpenGLRefusal(
                    f"GLSL stage attribute {attribute_name} requires exactly one argument"
                )

            argument = arguments[0]
            literal = self.literal_int_value(argument, self.literal_int_constants)
            source = self.glsl_attribute_argument_source(argument)
            if (
                not isinstance(literal, int)
                or isinstance(literal, bool)
                or literal <= 0
            ):
                raise OpenGLRefusal(
                    f"GLSL {stage_name} {layout_name} layout requires a positive "
                    f"integer constant, got {source}"
                )
            choices.append(
                (
                    f"{attribute_name} {source}",
                    literal,
                    self.generate_expression(argument),
                )
            )

        if not choices:
            return None

        first_source, first_literal, first_expression = choices[0]
        for source, literal, _expression in choices[1:]:
            if literal != first_literal:
                raise OpenGLRefusal(
                    f"Conflicting GLSL {stage_name} {layout_name} layout "
                    f"{first_source} with {source}"
                )
        return first_expression

    def glsl_stage_control_attributes(
        self, func, stage_layout_qualifiers=None, direction=None
    ):
        yield from getattr(func, "attributes", []) or []
        yield from self.glsl_stage_layout_entries(stage_layout_qualifiers, direction)

    def glsl_stage_has_attribute(
        self, func, attribute_name, stage_layout_qualifiers=None, direction=None
    ):
        return any(
            True
            for _ in self.glsl_stage_attributes(
                func, attribute_name, stage_layout_qualifiers, direction
            )
        )

    def glsl_stage_has_bare_attribute(
        self, func, names, stage_layout_qualifiers=None, direction=None
    ):
        names = set(names)
        for attr in self.glsl_stage_control_attributes(
            func, stage_layout_qualifiers, direction
        ):
            attr_name = self.glsl_stage_control_attribute_name(attr)
            if attr_name in names and not getattr(attr, "arguments", []):
                return True
        return False

    def glsl_single_stage_attribute_argument(
        self, func, attribute_name, stage_layout_qualifiers=None, direction=None
    ):
        arguments = self.glsl_stage_attribute_arguments(
            func, attribute_name, stage_layout_qualifiers, direction
        )
        if not arguments:
            return None
        if len(arguments) != 1:
            raise OpenGLRefusal(
                f"GLSL stage attribute {attribute_name} requires exactly one argument"
            )
        return self.attribute_value_to_string(arguments[0])

    def glsl_stage_attribute_arguments(
        self, func, attribute_name, stage_layout_qualifiers=None, direction=None
    ):
        for attr in self.glsl_stage_attributes(
            func, attribute_name, stage_layout_qualifiers, direction
        ):
            return getattr(attr, "arguments", []) or []
        return []

    def glsl_stage_attributes(
        self, func, attribute_name, stage_layout_qualifiers=None, direction=None
    ):
        for attr in getattr(func, "attributes", []) or []:
            if self.glsl_stage_control_attribute_name(attr) == attribute_name:
                yield attr
        for attr in self.glsl_stage_layout_entries(stage_layout_qualifiers, direction):
            if self.glsl_stage_control_attribute_name(attr) == attribute_name:
                yield attr

    def glsl_stage_layout_entries(self, stage_layout_qualifiers, direction=None):
        if direction is not None:
            direction = str(direction).lower()
        for layout in stage_layout_qualifiers or []:
            layout_direction = getattr(layout, "direction", None)
            if layout_direction is not None:
                layout_direction = str(layout_direction).lower()
            if direction is not None and layout_direction != direction:
                continue
            yield from getattr(layout, "entries", []) or []

    def glsl_geometry_output_topology(self, topology):
        topology_name = str(topology).strip().strip('"').lower()
        topology_map = {
            "point": "points",
            "points": "points",
            "line": "line_strip",
            "lines": "line_strip",
            "line_strip": "line_strip",
            "triangle": "triangle_strip",
            "triangles": "triangle_strip",
            "triangle_strip": "triangle_strip",
        }
        mapped = topology_map.get(topology_name)
        if mapped is None:
            raise OpenGLRefusal(
                f"GLSL geometry output topology cannot be lowered: {topology}"
            )
        return mapped

    def glsl_geometry_input_topology(self, topology):
        topology_name = str(topology).strip().strip('"').lower()
        topology_map = {
            "point": "points",
            "points": "points",
            "line": "lines",
            "lines": "lines",
            "line_adjacency": "lines_adjacency",
            "lines_adjacency": "lines_adjacency",
            "triangle": "triangles",
            "triangles": "triangles",
            "triangle_adjacency": "triangles_adjacency",
            "triangles_adjacency": "triangles_adjacency",
        }
        mapped = topology_map.get(topology_name)
        if mapped is None:
            raise OpenGLRefusal(
                f"GLSL geometry input topology cannot be lowered: {topology}"
            )
        return mapped

    def glsl_mesh_output_topology(self, topology):
        topology_name = str(topology).strip().strip('"').lower()
        topology_map = {
            "point": "points",
            "points": "points",
            "line": "lines",
            "lines": "lines",
            "triangle": "triangles",
            "triangles": "triangles",
        }
        mapped = topology_map.get(topology_name)
        if mapped is None:
            raise OpenGLRefusal(
                f"GLSL mesh output topology cannot be lowered: {topology}"
            )
        return mapped

    def map_mesh_intrinsic(self, operation, args):
        return glsl_mesh_intrinsic_call(self, operation, args)

    def is_task_payload_shared_variable(self, node):
        return (
            "taskPayloadSharedEXT" in self.glsl_variable_qualifier_prefix(node).split()
        )

    def glsl_task_payload_shared_variable_infos(self, nodes):
        infos = []
        for node in nodes:
            if not self.is_task_payload_shared_variable(node):
                continue
            name = self.resource_node_name(node)
            if not name:
                continue
            vtype, _, array_suffix, _ = self.resource_declaration_shape(node)
            infos.append(
                {
                    "name": name,
                    "type": self.map_type(vtype),
                    "raw_type": self.type_name_string(vtype),
                    "array_suffix": array_suffix,
                }
            )
        return infos

    def glsl_dispatch_mesh_payload_target(self, payload_expr):
        if not self.task_payload_shared_variables:
            raise OpenGLRefusal(
                "GLSL DispatchMesh payload argument requires "
                "taskPayloadSharedEXT storage"
            )

        payload_name = self.expression_name(payload_expr)
        for info in self.task_payload_shared_variables:
            if payload_name == info["name"]:
                return info

        payload_type = self.expression_result_type(payload_expr)
        mapped_payload_type = self.map_type(payload_type) if payload_type else None
        if mapped_payload_type:
            matches = [
                info
                for info in self.task_payload_shared_variables
                if info["type"] == mapped_payload_type
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                names = ", ".join(sorted(info["name"] for info in matches))
                raise OpenGLRefusal(
                    "Ambiguous GLSL DispatchMesh payload target for "
                    f"{mapped_payload_type}: {names}"
                )
            expected_types = ", ".join(
                sorted({info["type"] for info in self.task_payload_shared_variables})
            )
            raise OpenGLRefusal(
                "GLSL DispatchMesh payload argument type "
                f"{mapped_payload_type} does not match taskPayloadSharedEXT "
                f"payload type {expected_types}"
            )

        if len(self.task_payload_shared_variables) == 1:
            return self.task_payload_shared_variables[0]

        names = ", ".join(
            sorted(info["name"] for info in self.task_payload_shared_variables)
        )
        raise OpenGLRefusal(f"Ambiguous GLSL DispatchMesh payload target: {names}")

    def mesh_output_parameter_role(self, node):
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            normalized = str(attr_name).lower().replace("-", "_")
            if normalized.startswith("glsl_"):
                normalized = normalized[len("glsl_") :]
            if normalized in {"vertices", "vertex", "mesh_vertices"}:
                return "vertices"
            if normalized in {"indices", "index", "primitive_indices"}:
                return "indices"
            if normalized in {"primitives", "primitive", "mesh_primitives"}:
                return "primitives"
        return None

    def is_mesh_output_parameter(self, node):
        if self.mesh_output_parameter_role(node) is None:
            return False
        qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", []) or []}
        return bool(qualifiers & {"out", "inout"})

    def mesh_output_parameter_array_shape(self, node):
        node_type = self.resource_node_type(node)
        if (
            hasattr(node_type, "element_type")
            and str(type(node_type)).find("ArrayType") != -1
        ):
            return node_type.element_type, getattr(node_type, "size", None)
        return node_type, None

    def mesh_output_parameter_element_type(self, node):
        element_type, _ = self.mesh_output_parameter_array_shape(node)
        return self.type_name_string(element_type)

    def mesh_output_parameter_count(self, node):
        _, size = self.mesh_output_parameter_array_shape(node)
        return self.generate_expression(size) if size is not None else None

    def mesh_output_parameter_count_value(self, node):
        _, size = self.mesh_output_parameter_array_shape(node)
        if size is None:
            return None
        value = self.literal_int_value(size)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None

    def mesh_primitive_index_builtin(self, topology):
        return {
            "points": "gl_PrimitivePointIndicesEXT",
            "lines": "gl_PrimitiveLineIndicesEXT",
            "triangles": "gl_PrimitiveTriangleIndicesEXT",
        }.get(topology)

    def glsl_stage_attribute_literal_int(
        self, func, attribute_name, stage_layout_qualifiers=None, direction=None
    ):
        arguments = self.glsl_stage_attribute_arguments(
            func, attribute_name, stage_layout_qualifiers, direction
        )
        if not arguments:
            return None
        if len(arguments) != 1:
            raise OpenGLRefusal(
                f"GLSL stage attribute {attribute_name} requires exactly one argument"
            )
        value = self.literal_int_value(arguments[0])
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None

    def glsl_mesh_output_count_limits(self, func, stage_layout_qualifiers=None):
        max_vertices = self.glsl_stage_attribute_literal_int(
            func, "max_vertices", stage_layout_qualifiers, "out"
        )
        if max_vertices is None:
            max_vertices = self.glsl_stage_attribute_literal_int(
                func, "maxvertexcount", stage_layout_qualifiers, "out"
            )

        max_primitives = self.glsl_stage_attribute_literal_int(
            func, "max_primitives", stage_layout_qualifiers, "out"
        )
        if max_primitives is None:
            max_primitives = self.glsl_stage_attribute_literal_int(
                func, "maxprimitivecount", stage_layout_qualifiers, "out"
            )

        return {
            "vertices": max_vertices,
            "indices": max_primitives,
            "primitives": max_primitives,
        }

    def mesh_output_role_count_attribute(self, role):
        return "max_vertices" if role == "vertices" else "max_primitives"

    def validate_mesh_output_parameter_shape(
        self, node, role, element_type, count, count_value, count_limit, topology
    ):
        if count is None:
            raise OpenGLRefusal(
                "GLSL mesh output parameter "
                f"@{role} '{node.name}' requires an array with explicit size"
            )

        if count_value is not None and count_limit is not None:
            if count_value != count_limit:
                attribute_name = self.mesh_output_role_count_attribute(role)
                raise OpenGLRefusal(
                    "GLSL mesh output parameter "
                    f"@{role} '{node.name}' array size {count_value} must match "
                    f"{attribute_name} {count_limit}"
                )

        if role in {"vertices", "primitives"}:
            if element_type not in self.structs_by_name:
                raise OpenGLRefusal(
                    "GLSL mesh output parameter "
                    f"@{role} '{node.name}' requires a struct element type, "
                    f"got {element_type}"
                )
            return

        expected_index_type = self.GLSL_MESH_PRIMITIVE_INDEX_TYPES.get(topology)
        if expected_index_type is not None and element_type != expected_index_type:
            raise OpenGLRefusal(
                "GLSL mesh output parameter "
                f"@indices '{node.name}' for {topology} topology requires "
                f"{expected_index_type} elements, got {element_type}"
            )

    def mesh_output_parameter_infos(
        self, func, shader_type, stage_layout_qualifiers=None
    ):
        if shader_type != "mesh":
            return {}

        topology = self.glsl_mesh_stage_output_topology_name(
            func, stage_layout_qualifiers
        )
        count_limits = self.glsl_mesh_output_count_limits(func, stage_layout_qualifiers)
        infos = {}
        for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
            if not self.is_mesh_output_parameter(param):
                continue

            role = self.mesh_output_parameter_role(param)
            element_type = self.mesh_output_parameter_element_type(param)
            count = self.mesh_output_parameter_count(param)
            count_value = self.mesh_output_parameter_count_value(param)
            self.validate_mesh_output_parameter_shape(
                param,
                role,
                element_type,
                count,
                count_value,
                count_limits.get(role),
                topology,
            )
            info = {
                "name": param.name,
                "role": role,
                "element_type": element_type,
                "count": count,
                "count_value": count_value,
                "topology": topology,
                "index_builtin": self.mesh_primitive_index_builtin(topology),
                "members": {},
            }

            struct = self.structs_by_name.get(element_type)
            if struct is not None:
                for member in getattr(struct, "members", []) or []:
                    mapped_semantic = self.map_semantic(self.semantic_from_node(member))
                    info["members"][member.name] = {
                        "node": member,
                        "type": self.member_type_name(member),
                        "output_name": member.name,
                        "semantic": mapped_semantic,
                    }

            infos[param.name] = info
        return infos

    def generate_glsl_mesh_payload_parameter_declaration(self, func):
        """Declare a mesh entry's payload parameter as the ``taskPayloadSharedEXT`` block it reads.

        GL_EXT_mesh_shader passes the task stage's ``EmitMeshTasksEXT`` payload to every mesh workgroup through a
        ``taskPayloadSharedEXT`` variable; the entry body keeps reading it under the parameter's name.
        """
        payloads = [
            param
            for param in getattr(func, "parameters", getattr(func, "params", [])) or []
            if self.is_glsl_mesh_payload_parameter(param)
        ]
        if not payloads:
            return ""
        if len(payloads) > 1:
            raise OpenGLRefusal(
                "GLSL mesh entry declares more than one payload parameter: "
                + ", ".join(param.name for param in payloads)
            )
        param = payloads[0]
        param_type = self.map_type(
            self.type_name_string(
                getattr(param, "param_type", getattr(param, "vtype", None))
            )
        )
        declaration = format_c_style_array_declaration(param_type, param.name)
        return f"taskPayloadSharedEXT {declaration};\n"

    def is_glsl_mesh_payload_parameter(self, param):
        names = {
            str(getattr(attribute, "name", "")).lower()
            for attribute in getattr(param, "attributes", []) or []
        }
        names.update(
            str(qualifier).lower()
            for qualifier in getattr(param, "qualifiers", []) or []
        )
        return bool(names & {"mesh_payload", "meshpayload", "payload"})
