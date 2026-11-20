"""Stage members in the GLSL generator: input declarations, fragment and vertex output members, built-in aliases and scalar array view bindings."""

import re

from ..ast import (
    ArrayNode,
    BlockNode,
    ConstructorNode,
    DoWhileNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    IfNode,
    LoopNode,
    RangeNode,
    SwitchNode,
    UnaryOpNode,
    VariableNode,
    WhileNode,
)
from ..refusals import OpenGLRefusal
from .array_utils import (
    format_c_style_array_declaration,
    split_array_type_suffix,
)
from .glsl_draw_parameters import (
    DRAW_PARAMETERS,
    vertex_builtin_spelling,
)
from .glsl_instance_index import INSTANCE_INDEX
from .glsl_interface_block_io import (
    block_declared_stage_io,
    block_member_location_type,
)
from .glsl_mesh_locations import mesh_fragment_input_layouts
from .glsl_nested_stage_io import refuse_nested_stage_builtins
from .glsl_stage_builtin_members import (
    fragment_member_builtin_alias,
    fragment_parameter_builtin_expression,
    is_vertex_layer_output,
    note_vertex_builtin_output,
    refuse_unmapped_vertex_output,
)
from .glsl_stage_distances import (
    distance_input_alias,
    distance_output_epilogue,
    distance_output_name,
    distance_private_output_declaration,
)
from .glsl_stage_output_values import builtin_output_value
from .glsl_vertex_inputs import (
    vertex_input_builtin,
    vertex_input_layouts,
)
from .opengl_values import GLSL_CONVERTED_FRAGMENT_BUILTIN_TYPES
from .stage_distance_layout import distance_semantic


class OpenGLStageMemberMixin:
    """Stage member members of ``GLSLCodeGen``."""

    def generate_stage_input_declarations(self, node):
        code = ""
        members = getattr(node, "members", []) or []
        layouts = vertex_input_layouts(self, members)
        for member in members:
            if member.name not in layouts:
                # A built-in value; its reads read the built-in.
                continue
            member_type = self.member_type_name(member)
            member_name = self.vertex_input_member_emitted_name(
                member
            ) or self.glsl_module_identifier_name(member.name)
            layout = layouts[member.name]
            self.reserve_stage_io_layout(
                self.stage_io_used_locations,
                "vertex",
                "input",
                layout,
                member_type,
                member_name,
            )
            prefix = self.stage_io_declaration_prefix(
                member, "in", layout=layout, stage_name="vertex"
            )
            member_decl = format_c_style_array_declaration(member_type, member_name)
            declaration = f"{prefix} {member_decl};"
            if self.reserve_stage_io_declaration(
                "vertex", "input", member_name, declaration
            ):
                code += f"{declaration}\n"
        return code

    def generate_legacy_output_declarations(self, node):
        code = ""
        for member in getattr(node, "members", []) or []:
            member_name = self.glsl_module_identifier_name(member.name)
            member_decl = format_c_style_array_declaration(
                self.member_type_name(member), member_name
            )
            declaration = f"out {member_decl};"
            if self.reserve_stage_io_declaration(
                "vertex", "output", member_name, declaration
            ):
                code += f"{declaration}\n"
        return code

    def generate_vertex_output_declarations(self, node):
        code = ""
        for member in getattr(node, "members", []) or []:
            output_name = self.vertex_output_member_name(member)
            if self.is_vertex_stage_builtin_output(output_name):
                continue
            private_distance = distance_private_output_declaration(self, member)
            if private_distance is not None:
                code += private_distance
                continue
            refuse_nested_stage_builtins(self, member, "vertex", "output")

            layout = self.stage_io_layout_for_node(member)
            block = block_declared_stage_io(self, "out", member, output_name)
            self.reserve_stage_io_layout(
                self.stage_io_used_locations,
                "vertex",
                "output",
                layout,
                (
                    self.member_type_name(member)
                    if block is None
                    else block_member_location_type(self, member, block)
                ),
                output_name,
            )
            if block is not None:
                # Declared by the output interface block itself.
                continue
            prefix = self.stage_io_declaration_prefix(
                member, "out", stage_name="vertex"
            )
            member_type = self.member_type_name(member)
            prefix = self.stage_io_prefix_with_required_flat(prefix, member_type)
            member_decl = format_c_style_array_declaration(member_type, output_name)
            declaration = f"{prefix} {member_decl};"
            if self.reserve_stage_io_declaration(
                "vertex", "output", output_name, declaration
            ):
                code += f"{declaration}\n"
        return code

    def generate_fragment_input_declarations(self, node):
        code = ""
        mesh_layouts = (
            mesh_fragment_input_layouts(
                self,
                [
                    member
                    for member in getattr(node, "members", []) or []
                    if self.fragment_input_member_name(member, node.name) is not None
                ],
            )
            if self.glsl_program_has_mesh_stage
            else {}
        )
        for member in getattr(node, "members", []) or []:
            input_name = self.fragment_input_member_name(member, node.name)
            if input_name is None:
                continue
            refuse_nested_stage_builtins(self, member, "fragment", "input")

            layout = mesh_layouts.get(member.name) or self.stage_io_layout_for_node(
                member
            )
            member_type = self.member_type_name(member)
            block = block_declared_stage_io(self, "in", member, input_name)
            self.reserve_stage_io_layout(
                self.stage_io_used_locations,
                "fragment",
                "input",
                layout,
                (
                    block_member_location_type(self, member, block)
                    if block is not None
                    else (
                        self.stage_io_per_vertex_element_type(member_type)
                        if self.is_pervertex_stage_io(member, "")
                        else member_type
                    )
                ),
                input_name,
            )
            if block is not None:
                # Declared by the input interface block itself.
                continue
            prefix = self.stage_io_declaration_prefix(
                member, "in", layout=layout, stage_name="fragment"
            )
            prefix = self.stage_io_prefix_with_required_flat(prefix, member_type)
            member_decl = format_c_style_array_declaration(member_type, input_name)
            declaration = f"{prefix} {member_decl};"
            if self.reserve_stage_io_declaration(
                "fragment", "input", input_name, declaration
            ):
                code += f"{declaration}\n"
        return code

    def generate_fragment_output_declarations(self, node):
        code = ""
        for member in getattr(node, "members", []) or []:
            member_type = self.member_type_name(member)
            output_name = self.fragment_output_member_name(member, node.name)
            if self.is_fragment_builtin_output_target(output_name):
                continue
            layout = self.fragment_output_member_layout_maps.get(node.name, {}).get(
                member.name
            )
            if layout is None:
                layout = self.map_semantic(self.semantic_from_node(member))
                if not layout.startswith("layout("):
                    layout = "layout(location = 0)"
            self.reserve_stage_io_layout(
                self.stage_io_used_locations,
                "fragment",
                "output",
                layout,
                member_type,
                output_name,
            )
            prefix = self.stage_io_declaration_prefix(member, "out", layout)
            member_decl = format_c_style_array_declaration(member_type, output_name)
            declaration = f"{prefix} {member_decl};"
            if self.reserve_stage_io_declaration(
                "fragment", "output", output_name, declaration
            ):
                code += f"{declaration}\n"
        return code

    def member_type_name(self, member):
        if hasattr(member, "member_type"):
            return self.map_type(member.member_type)
        return self.map_type(getattr(member, "vtype", "float"))

    def stage_io_prefix_with_required_flat(self, prefix, mapped_type):
        if not self.requires_flat_stage_io(mapped_type):
            return prefix
        parts = prefix.split()
        if "flat" in parts:
            return prefix
        for direction in ("in", "out"):
            if direction in parts:
                parts.insert(parts.index(direction), "flat")
                return " ".join(parts)
        return f"{prefix} flat" if prefix else "flat"

    def requires_flat_stage_io(self, mapped_type):
        base_type, _ = split_array_type_suffix(str(mapped_type))
        return base_type in {
            "int",
            "uint",
            "int64_t",
            "uint64_t",
            "bool",
            "ivec2",
            "ivec3",
            "ivec4",
            "uvec2",
            "uvec3",
            "uvec4",
            "i64vec2",
            "i64vec3",
            "i64vec4",
            "u64vec2",
            "u64vec3",
            "u64vec4",
            "bvec2",
            "bvec3",
            "bvec4",
        }

    def vertex_output_member_name(self, member):
        semantic = self.stage_io_semantic_from_node(member)
        mapped_semantic = self.map_semantic(semantic)
        if distance_semantic(mapped_semantic) is not None:
            return distance_output_name(self, member, mapped_semantic)
        if self.is_vertex_stage_builtin_output(mapped_semantic):
            note_vertex_builtin_output(self, mapped_semantic)
            return mapped_semantic
        refuse_unmapped_vertex_output(self, semantic, mapped_semantic)
        if (
            member.name in self.vertex_input_member_names
            or member.name in self.stage_io_reserved_value_names
        ):
            return self.glsl_generated_module_identifier(
                ("stage-io", "vertex-output", member.name),
                f"out_{member.name}",
            )
        return self.glsl_module_identifier_name(member.name)

    def fragment_output_member_name(
        self, member, struct_name=None, reserved_names=None
    ):
        if struct_name in self.fragment_output_member_name_maps:
            member_map = self.fragment_output_member_name_maps[struct_name]
            if member.name in member_map:
                return member_map[member.name]
            return self.refuse_lowering(
                f"fragment output '{member.name}' is not declared by structure '{struct_name}'",
                member.name,
            )

        mapped_semantic = self.map_semantic(self.stage_io_semantic_from_node(member))
        if mapped_semantic == "gl_FragDepth":
            return mapped_semantic
        if mapped_semantic == "gl_FragStencilRefARB":
            return mapped_semantic
        if mapped_semantic == "gl_SampleMask":
            return "gl_SampleMask[0]"

        if reserved_names is None:
            reserved_names = self.fragment_input_member_names
        return self.stage_io_name_avoiding_reserved(member.name, reserved_names)

    def is_fragment_builtin_output_target(self, target):
        return target in {
            "gl_FragDepth",
            "gl_FragStencilRefARB",
            "gl_SampleMask",
            "gl_SampleMask[0]",
        }

    def fragment_input_member_name(self, member, struct_name):
        if self.fragment_input_builtin_alias(member) is not None:
            return None
        if struct_name in self.vertex_output_struct_names:
            output_name = self.vertex_output_member_name(member)
            if self.is_vertex_stage_builtin_output(output_name):
                return None
            return self.combined_fragment_input_member_name(output_name)
        linked_name = self.linked_vertex_output_member_name(member.name)
        if linked_name is None and member.name in self.stage_io_reserved_value_names:
            linked_name = self.glsl_generated_module_identifier(
                ("stage-io", "vertex-output", member.name),
                f"out_{member.name}",
            )
        return self.combined_fragment_input_member_name(linked_name or member.name)

    def linked_vertex_output_member_name(self, member_name):
        """The emitted name of the vertex output a fragment input of another structure reads by name."""
        for struct_name in sorted(self.vertex_output_struct_names):
            struct = self.structs_by_name.get(struct_name)
            for member in getattr(struct, "members", []) or []:
                if member.name != member_name:
                    continue
                output_name = self.vertex_output_member_name(member)
                if self.is_vertex_stage_builtin_output(output_name):
                    return None
                return output_name
        return None

    def fragment_input_builtin_alias(self, member):
        semantic = self.stage_io_semantic_from_node(member)
        if semantic is None:
            return None
        distance = distance_input_alias(self, member)
        if distance is not None:
            return distance
        mapped_semantic = self.map_stage_input_semantic(semantic, "fragment")
        if mapped_semantic not in {
            "gl_PrimitiveID",
            "gl_FrontFacing",
            "gl_FragCoord",
            "gl_PointCoord",
            "gl_SampleID",
            "gl_SamplePosition",
            "gl_SampleMaskIn",
        }:
            return fragment_member_builtin_alias(self, semantic, mapped_semantic)
        return self.stage_input_builtin_alias(semantic, "fragment")

    def combined_fragment_input_member_name(self, input_name):
        # Guarded stage sections are compiled one stage at a time, and a varying without a location links to the
        # output of the same name (GLSL 4.60 section 4.3.4), so a fragment input keeps the vertex output's name.
        return self.glsl_module_identifier_name(input_name)

    def is_vertex_builtin_output(self, name):
        # A scalar distance output is named by its element, ``gl_ClipDistance[0]``.
        name = re.sub(r"\[\d+\]$", "", str(name or ""))
        return name in {
            "gl_Position",
            "gl_PointSize",
            "gl_ClipDistance",
            "gl_CullDistance",
        }

    def is_vertex_stage_builtin_output(self, name):
        # A vertex stage also writes the layer and viewport (glsl_stage_builtin_members); a mesh vertex does not.
        return self.is_vertex_builtin_output(name) or is_vertex_layer_output(name)

    def stage_input_member_maps(self, func, shader_type):
        if shader_type not in {"vertex", "fragment"}:
            return {}

        maps = {}
        for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
            if self.is_stage_entry_resource_parameter(param):
                continue
            type_name = self.type_node_name(getattr(param, "param_type", None))
            struct = self.structs_by_name.get(type_name)
            if struct is None:
                continue
            if shader_type == "fragment":
                member_map = {}
                for member in getattr(struct, "members", []) or []:
                    builtin_alias = self.fragment_input_builtin_alias(member)
                    if builtin_alias is not None:
                        member_map[member.name] = builtin_alias
                        continue
                    input_name = self.fragment_input_member_name(member, type_name)
                    if input_name is not None:
                        member_map[member.name] = input_name
                maps[param.name] = member_map
            else:
                maps[param.name] = {
                    member.name: (
                        vertex_input_builtin(self, member)
                        or self.vertex_input_member_emitted_name(member)
                        or self.glsl_struct_member_name(type_name, member.name)
                    )
                    for member in getattr(struct, "members", [])
                }
        return maps

    def apply_stage_context_entry_aliases(self, func, shader_type, stage_context):
        if shader_type is not None:
            return
        if stage_context not in {"vertex", "fragment"}:
            return

        entry_func = self.stage_entry_functions_by_stage.get(stage_context)
        if entry_func is None:
            return

        parameter_names = {
            param.name
            for param in getattr(func, "parameters", getattr(func, "params", [])) or []
            if getattr(param, "name", None)
        }
        for name, member_map in self.stage_input_member_maps(
            entry_func,
            stage_context,
        ).items():
            if name not in parameter_names:
                self.current_stage_inputs.setdefault(name, member_map)

        for name, alias in self.stage_parameter_aliases(
            entry_func,
            stage_context,
        ).items():
            if name not in parameter_names:
                self.current_stage_parameter_aliases.setdefault(name, alias)

        if "output" not in parameter_names:
            output_member_map = self.stage_output_member_map(entry_func, stage_context)
            if output_member_map:
                self.current_stage_outputs.setdefault("output", output_member_map)

    def stage_parameter_aliases(self, func, shader_type):
        if shader_type is None:
            return {}

        aliases = {}
        for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
            output_alias = self.stage_parameter_output_aliases.get(id(func), {}).get(
                param.name
            )
            if output_alias is not None:
                aliases[param.name] = output_alias
                continue
            semantic = self.semantic_from_node(param)
            if self.is_stage_builtin_semantic(semantic):
                if self.stage_builtin_parameter_local_alias_declaration(
                    param, shader_type
                ):
                    continue
                aliases[param.name] = self.stage_input_builtin_alias(
                    semantic, shader_type
                )
                continue
            if shader_type in {
                "vertex",
                "fragment",
                "geometry",
                "tessellation_control",
                "tessellation_evaluation",
                "mesh",
                "task",
            } and self.is_stage_entry_value_parameter(param):
                input_name = self.glsl_module_identifier_name(param.name)
                if input_name != param.name:
                    aliases[param.name] = input_name
        return aliases

    def stage_builtin_parameter_local_alias_declarations(self, func, shader_type):
        if shader_type is None:
            return []

        declarations = []
        for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
            declaration = self.stage_builtin_parameter_local_alias_declaration(
                param, shader_type
            )
            if declaration is not None:
                declarations.append(declaration)
        return declarations

    def stage_builtin_parameter_local_alias_declaration(self, param, shader_type):
        semantic = self.semantic_from_node(param)
        if semantic is None:
            return None

        builtin = self.map_stage_input_semantic(semantic, shader_type)
        mapped_type = self.map_type(self.resource_node_type(param))
        base_type, array_suffix = split_array_type_suffix(str(mapped_type))
        if array_suffix:
            return None

        expression = fragment_parameter_builtin_expression(
            self, builtin, base_type, shader_type
        ) or self.stage_builtin_parameter_local_alias_expression(builtin, base_type)
        if expression is None:
            return None
        parameter_name = self.glsl_local_identifier_name(param.name)
        return f"{base_type} {parameter_name} = {expression};"

    def stage_builtin_parameter_local_alias_expression(self, builtin, mapped_type):
        # Metal permits the positional/dimension compute builtins to be declared
        # as scalar `uint`/`int` or 2-component vectors; GLSL exposes them only as
        # uvec3. Emit a component-extracting local alias (e.g.
        # `uint lid = gl_LocalInvocationID.x;`) so MLX's 1-D reduction kernels
        # (softmax, rms_norm, layer_norm, logsumexp, ...) lower cleanly instead of
        # being rejected for not being uvec3.
        vector3_builtins = {
            "gl_GlobalInvocationID",
            "gl_LocalInvocationID",
            "gl_WorkGroupID",
            "gl_NumWorkGroups",
            "gl_WorkGroupSize",
            self.GLSL_THREADS_PER_GRID_EXPRESSION,
        }
        if builtin in vector3_builtins:
            if mapped_type == "uvec3":
                return None
            if mapped_type == "ivec3":
                return f"ivec3({builtin})"
            if mapped_type in {"uvec2", "ivec2"}:
                return f"{mapped_type}({builtin}.xy)"
            if mapped_type in {"uint", "int"}:
                return f"{mapped_type}({builtin}.x)"
            return None

        scalar_int_builtins = {
            "gl_VertexID",
            "gl_InstanceID",
            "gl_PrimitiveID",
            "gl_SampleID",
            "gl_SampleMaskIn",
            INSTANCE_INDEX,
            *DRAW_PARAMETERS,
        }
        if builtin in scalar_int_builtins:
            if mapped_type == "int":
                return None
            source = (
                "gl_SampleMaskIn[0]"
                if builtin == "gl_SampleMaskIn"
                else vertex_builtin_spelling(self, builtin)
            )
            if mapped_type == "uint":
                return f"uint({source})"
            return None

        # Built-ins GLSL declares as bool or int that HLSL delivers as uint: SV_InnerCoverage (gl_FullyCoveredEXT),
        # SV_ShadingRate, SV_RenderTargetArrayIndex and SV_ViewportArrayIndex.
        glsl_type = GLSL_CONVERTED_FRAGMENT_BUILTIN_TYPES.get(builtin)
        if glsl_type is not None and mapped_type in {"uint", "int"}:
            return None if mapped_type == glsl_type else f"{mapped_type}({builtin})"

        return None

    def stage_output_member_map(self, func, shader_type):
        if shader_type not in {"vertex", "fragment"}:
            return {}

        type_name = self.type_node_name(getattr(func, "return_type", None))
        struct = self.structs_by_name.get(type_name)
        if struct is None:
            return {}

        if shader_type == "fragment":
            return {
                member.name: self.fragment_output_member_name(member, type_name)
                for member in getattr(struct, "members", []) or []
            }
        return {
            member.name: self.vertex_output_member_name(member)
            for member in getattr(struct, "members", []) or []
        }

    def stage_output_member_maps(self, func, shader_type):
        member_map = self.stage_output_member_map(func, shader_type)
        if not member_map:
            return {}

        type_name = self.type_node_name(getattr(func, "return_type", None))
        maps = {}
        body = getattr(func, "body", [])
        statements = getattr(body, "statements", body if isinstance(body, list) else [])
        for stmt in statements:
            if not isinstance(stmt, VariableNode):
                continue
            if self.type_node_name(getattr(stmt, "var_type", None)) == type_name:
                maps[stmt.name] = member_map
        return maps

    def function_return_type(self, func):
        return_type = getattr(func, "return_type", None)
        if return_type is None:
            return "void"
        return self.map_type(return_type)

    def stage_return_output(self, func, shader_type):
        if shader_type == "fragment":
            return self.fragment_stage_output(func, shader_type)
        if shader_type == "vertex":
            return self.vertex_stage_output(func)
        return None

    def vertex_stage_output(self, func):
        output_type = self.function_return_type(func)
        source_output_type = self.glsl_source_type_identifier_name(output_type)
        if output_type == "void":
            return None

        semantic = self.function_return_semantic(func)
        mapped_semantic = self.map_semantic(semantic)
        if not mapped_semantic and output_type == "vec4":
            mapped_semantic = "gl_Position"
        elif not mapped_semantic:
            if source_output_type in self.structs_by_name or output_type not in {
                "float",
                "double",
                "int",
                "uint",
                "bool",
                "vec2",
                "vec3",
                "vec4",
                "dvec2",
                "dvec3",
                "dvec4",
                "ivec2",
                "ivec3",
                "ivec4",
                "uvec2",
                "uvec3",
                "uvec4",
                "bvec2",
                "bvec3",
                "bvec4",
                "mat2",
                "mat3",
                "mat4",
                "dmat2",
                "dmat3",
                "dmat4",
            }:
                return None
            raise OpenGLRefusal(
                "OpenGL vertex stage entry point with non-void return type "
                f"'{output_type}' requires an output semantic"
            )

        if self.is_vertex_builtin_output(mapped_semantic):
            return {
                "name": mapped_semantic,
                "declaration": "",
                "type": output_type,
            }

        layout = mapped_semantic
        if not layout.startswith("layout("):
            layout = "layout(location = 0)"
        reserved_names = set(self.vertex_input_member_names)
        reserved_names.update(self.stage_io_declared_names("vertex", "output"))
        reserved_names.update(self.function_local_variable_names(func))
        output_name = self.stage_io_name_avoiding_reserved(
            "vertexOutput", reserved_names
        )
        self.reserve_stage_io_layout(
            self.stage_io_used_locations,
            "vertex",
            "output",
            layout,
            output_type,
            output_name,
        )
        declaration = f"{layout} out {output_type} {output_name};"
        if not self.reserve_stage_io_declaration(
            "vertex", "output", output_name, declaration
        ):
            declaration = ""
        return {
            "name": output_name,
            "declaration": declaration,
            "type": output_type,
        }

    def fragment_stage_output(self, func, shader_type):
        if shader_type != "fragment":
            return None

        output_type = self.function_return_type(func)
        source_output_type = self.glsl_source_type_identifier_name(output_type)
        if output_type == "void":
            return None
        if source_output_type in self.structs_by_name:
            return None

        semantic = self.function_return_semantic(func) or "gl_FragColor"
        mapped_semantic = self.map_semantic(semantic)
        if mapped_semantic == "gl_FragDepth":
            return {
                "name": "gl_FragDepth",
                "declaration": "",
            }
        if mapped_semantic == "gl_FragStencilRefARB":
            return {
                "name": "gl_FragStencilRefARB",
                "declaration": "",
            }
        if mapped_semantic == "gl_SampleMask":
            return {
                "name": "gl_SampleMask[0]",
                "declaration": "",
            }

        layout = self.stage_io_layout_for_node(func)
        if not layout:
            layout = mapped_semantic
        if not layout.startswith("layout("):
            layout = "layout(location = 0)"

        local_names = self.function_local_variable_names(func)
        output_key = self.fragment_direct_output_key(
            mapped_semantic, layout, output_type
        )
        output_name = self.fragment_direct_output_name_maps.get(output_key)
        if output_name in local_names:
            output_name = None
        if output_name is None:
            output_name = self.fragment_output_name(semantic, local_names)
        self.reserve_stage_io_layout(
            self.stage_io_used_locations,
            "fragment",
            "output",
            layout,
            output_type,
            output_name,
        )
        prefix = self.stage_io_declaration_prefix(func, "out", layout)
        declaration = f"{prefix} {output_type} {output_name};"
        if not self.reserve_stage_io_declaration(
            "fragment", "output", output_name, declaration
        ):
            declaration = ""
        self.fragment_direct_output_name_maps.setdefault(output_key, output_name)
        return {
            "name": output_name,
            "declaration": declaration,
            "type": output_type,
        }

    def fragment_direct_output_key(self, mapped_semantic, layout, output_type):
        return (
            str(mapped_semantic or ""),
            str(layout or ""),
            str(output_type or ""),
        )

    def fragment_output_name(self, semantic, additional_reserved_names=None):
        if semantic and semantic.startswith("gl_FragColor"):
            suffix = semantic[len("gl_FragColor") :]
            output_name = f"fragColor{suffix}"
        else:
            output_name = "fragColor"

        reserved_names = set(self.fragment_input_member_names)
        for member_map in self.fragment_output_member_name_maps.values():
            for name in member_map.values():
                if not self.is_fragment_builtin_output_target(name):
                    reserved_names.add(name)
        reserved_names.update(self.stage_io_declared_names("fragment", "output"))
        reserved_names.update(additional_reserved_names or set())
        return self.stage_io_name_avoiding_reserved(output_name, reserved_names)

    def fragment_legacy_color_output_aliases(self, func, shader_type, stage_output):
        if shader_type != "fragment" or stage_output is None:
            return {}

        semantic = str(self.function_return_semantic(func) or "gl_FragColor")
        if not semantic.startswith("gl_FragColor"):
            return {}

        output_name = stage_output.get("name")
        if not output_name or output_name.startswith("gl_"):
            return {}

        local_names = self.function_local_variable_names(func)
        if semantic not in local_names:
            return {}
        return {semantic: output_name}

    def generate_stage_struct_constructor_return(self, expr, indent):
        if not self.current_stage_output_member_map:
            return None

        struct = self.structs_by_name.get(self.current_stage_return_type)
        if struct is None:
            return None

        members = getattr(struct, "members", []) or []
        field_values = self.stage_struct_constructor_field_values(expr, members)
        if field_values is None:
            return None

        indent_str = "    " * indent
        code = ""
        for member, value_expr in field_values:
            output_name = self.current_stage_output_member_map.get(member.name)
            if output_name is None:
                continue
            member_type = self.member_type_name(member)
            value = self.generate_expression_with_expected(value_expr, member_type)
            value = builtin_output_value(self, output_name, value, member_type)
            code += f"{indent_str}{output_name} = {value};\n"
        code += distance_output_epilogue(self, indent_str)
        code += f"{indent_str}return;\n"
        return code

    def stage_struct_constructor_field_values(self, expr, members):
        if isinstance(expr, FunctionCallNode):
            if self.function_call_name(expr) != self.current_stage_return_type:
                return None
            args = getattr(expr, "arguments", getattr(expr, "args", [])) or []
            if len(args) != len(members):
                return None
            return list(zip(members, args))

        if isinstance(expr, ConstructorNode):
            constructor_type = self.type_name_string(
                getattr(expr, "constructor_type", None)
            )
            if constructor_type != self.current_stage_return_type:
                return None
            args = list(getattr(expr, "arguments", []) or [])
            named_args = dict(getattr(expr, "named_arguments", {}) or {})
            if len(args) > len(members):
                return None

            values = []
            for index, member in enumerate(members):
                if index < len(args):
                    values.append((member, args[index]))
                    continue
                if member.name not in named_args:
                    return None
                values.append((member, named_args[member.name]))
            return values

        return None

    def generate_glsl_statement_code(self, code, indent=0):
        indent_str = "    " * indent
        lines = [line.rstrip() for line in str(code).splitlines() if line.strip()]
        result = ""
        for line in lines:
            terminator = "" if line.endswith((";", "}")) else ";"
            result += f"{indent_str}{line}{terminator}\n"
        return result

    def generate_glsl_return_value_statement(self, value, indent):
        """Render a return while preserving ``@precise`` without a return qualifier."""
        indent_str = "    " * indent
        return_type = self.current_glsl_precise_return_type
        if return_type is None:
            return f"{indent_str}return {value};\n"

        return_name = self.glsl_synthetic_local_identifier("crossglPreciseReturn")
        source_return_type = self.type_name_string(self.current_function_return_type)
        self.local_variable_types[return_name] = source_return_type
        self.local_variable_source_types[return_name] = source_return_type
        declaration = format_c_style_array_declaration(return_type, return_name)
        return (
            f"{indent_str}precise {declaration} = {value};\n"
            f"{indent_str}return {return_name};\n"
        )

    def glsl_scalar_array_view_lexical_bindings(self, function):
        """Bind bare address arguments to exact visible local declarations."""

        call_bindings = {}
        call_source_types = {}
        declarations = {}
        reference_bindings = {}

        def visible_source_types(visible):
            return {
                name: binding["source_type"]
                for name, binding in visible.items()
                if binding.get("source_type") is not None
            }

        def expression_source_type(value, visible):
            previous_source_types = self.local_variable_source_types
            previous_types = self.local_variable_types
            scoped_types = visible_source_types(visible)
            self.local_variable_source_types = {
                **previous_source_types,
                **scoped_types,
            }
            self.local_variable_types = {
                **previous_types,
                **scoped_types,
            }
            try:
                return self.glsl_source_expression_type(value)
            finally:
                self.local_variable_source_types = previous_source_types
                self.local_variable_types = previous_types

        def declaration_source_type(node, visible):
            raw_type = getattr(
                node,
                "param_type",
                getattr(node, "var_type", getattr(node, "vtype", None)),
            )
            type_name = self.type_name_string(raw_type)
            if type_name not in {None, "auto"}:
                return raw_type
            return expression_source_type(
                getattr(node, "initial_value", None),
                visible,
            )

        def for_in_binding_source_type(node, visible):
            iterable = getattr(node, "iterable", None)
            if isinstance(iterable, RangeNode):
                return "int"
            iterable_type = expression_source_type(iterable, visible)
            if self.is_scalar_integer_type(iterable_type):
                return "int"
            outer_array = self.glsl_outer_array_type(
                self.type_name_string(iterable_type)
            )
            if outer_array is not None:
                _extent, element_type = outer_array
                return element_type
            return getattr(
                node,
                "binding_type",
                getattr(node, "pattern_type", None),
            )

        def register(name, node, source_type, kind):
            binding = {
                "id": id(node),
                "kind": kind,
                "name": name,
                "node": node,
                "source_type": source_type,
            }
            declarations[id(node)] = binding
            return binding

        def visit_sequence(value, visible):
            if value is None:
                return
            statements = getattr(value, "statements", value)
            if not isinstance(statements, (list, tuple)):
                statements = [statements]
            for statement in statements:
                visit(statement, visible)

        def visit(node, visible):
            if node is None or isinstance(node, (str, int, float, bool)):
                return
            if isinstance(node, dict):
                for child in node.values():
                    visit(child, visible)
                return
            if isinstance(node, (list, tuple, set)):
                for child in node:
                    visit(child, visible)
                return

            if isinstance(node, IdentifierNode):
                binding = visible.get(getattr(node, "name", None))
                if binding is not None:
                    reference_bindings[id(node)] = binding["id"]
                return

            if isinstance(node, (VariableNode, ArrayNode)):
                name = getattr(node, "name", None)
                if name:
                    visible.pop(name, None)
                initializer_source_types = visible_source_types(visible)
                visit(getattr(node, "initial_value", None), visible)
                if name:
                    visible[name] = register(
                        name,
                        node,
                        declaration_source_type(node, visible),
                        "local",
                    )
                    visible[name]["initializer_source_types"] = initializer_source_types
                return

            if isinstance(node, FunctionCallNode):
                arguments = list(
                    getattr(node, "arguments", getattr(node, "args", [])) or []
                )
                bindings = {}
                for index, argument in enumerate(arguments):
                    if not (
                        isinstance(argument, UnaryOpNode)
                        and self.map_operator(argument.op) == "&"
                        and isinstance(argument.operand, (IdentifierNode, VariableNode))
                    ):
                        continue
                    name = getattr(argument.operand, "name", None)
                    binding = visible.get(name)
                    if binding is not None:
                        bindings[index] = binding["id"]
                call_bindings[id(node)] = bindings
                call_source_types[id(node)] = visible_source_types(visible)
                for argument in arguments:
                    visit(argument, visible)
                return

            if isinstance(node, BlockNode):
                visit_sequence(node, dict(visible))
                return

            if isinstance(node, IfNode):
                branch_visible = dict(visible)
                condition = getattr(
                    node, "condition", getattr(node, "if_condition", None)
                )
                visit(condition, branch_visible)
                visit_sequence(node.if_body, dict(branch_visible))
                for else_if_condition, else_if_body in zip(
                    getattr(node, "else_if_conditions", []) or [],
                    getattr(node, "else_if_bodies", []) or [],
                ):
                    else_if_visible = dict(visible)
                    visit(else_if_condition, else_if_visible)
                    visit_sequence(else_if_body, else_if_visible)
                visit_sequence(getattr(node, "else_body", None), dict(branch_visible))
                return

            if isinstance(node, ForNode):
                loop_visible = dict(visible)
                visit(getattr(node, "init", None), loop_visible)
                visit(getattr(node, "condition", None), loop_visible)
                visit_sequence(getattr(node, "body", None), dict(loop_visible))
                visit(getattr(node, "update", None), loop_visible)
                return

            if isinstance(node, ForInNode):
                loop_visible = dict(visible)
                visit(getattr(node, "iterable", None), loop_visible)
                pattern = getattr(node, "pattern", None)
                if pattern:
                    loop_visible.pop(pattern, None)
                    loop_visible[pattern] = register(
                        pattern,
                        node,
                        for_in_binding_source_type(node, visible),
                        "for-in",
                    )
                visit_sequence(getattr(node, "body", None), loop_visible)
                return

            if isinstance(node, (WhileNode, DoWhileNode, LoopNode)):
                loop_visible = dict(visible)
                visit(getattr(node, "condition", None), loop_visible)
                visit_sequence(getattr(node, "body", None), loop_visible)
                return

            if isinstance(node, SwitchNode):
                switch_visible = dict(visible)
                visit(getattr(node, "expression", None), switch_visible)
                for case in getattr(node, "cases", []) or []:
                    case_visible = dict(switch_visible)
                    visit(getattr(case, "value", None), case_visible)
                    visit_sequence(getattr(case, "statements", None), case_visible)
                default_case = getattr(node, "default_case", None)
                visit_sequence(
                    getattr(default_case, "statements", default_case),
                    dict(switch_visible),
                )
                return

            if hasattr(node, "child_nodes"):
                for child in node.child_nodes():
                    visit(child, visible)
                return
            if hasattr(node, "__dict__"):
                for field, child in vars(node).items():
                    if field not in {"annotations", "parent", "source_location"}:
                        visit(child, visible)

        visible = {}
        for parameter in (
            getattr(function, "parameters", getattr(function, "params", [])) or []
        ):
            name = getattr(parameter, "name", None)
            if name:
                visible[name] = register(
                    name,
                    parameter,
                    declaration_source_type(parameter, visible),
                    "parameter",
                )
        visit_sequence(getattr(function, "body", []), visible)
        return {
            "call_bindings": call_bindings,
            "call_source_types": call_source_types,
            "declarations": declarations,
            "reference_bindings": reference_bindings,
        }
