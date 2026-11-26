"""Entry-point parameters of the Slang generator: function bodies, hull outputs, patch parameters, implicit system values and built-in output returns."""

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    IdentifierNode,
    MemberAccessNode,
    ReturnNode,
    VariableNode,
)
from ..refusals import SlangRefusal
from .array_utils import split_array_type_suffix
from .ast_walk import walk_ast_values
from .slang_subgroup_builtins import (
    SLANG_SUBGROUP_WORKGROUP_BUILTINS,
    slang_subgroup_builtin_candidates,
)
from .slang_workgroup_size import SlangWorkgroupSizeConstants


class SlangStageParametersMixin:
    """Stage-parameter members of ``SlangCodeGen``."""

    def emit_slang_function_body(
        self,
        signature,
        body_statements,
        builtin_return_rewrite,
        hull_output_rewrite,
        fragment_output_rewrite,
        local_alias_declarations,
    ):
        """The braced body of a function whose ``signature`` :meth:`generate_function` has built."""
        result = f"{signature}\n{{\n"
        self.indent_level += 1

        if builtin_return_rewrite is not None and builtin_return_rewrite["mode"] in {
            "local",
            "struct",
        }:
            local_type = self.convert_type(builtin_return_rewrite["return_type_name"])
            local_name = builtin_return_rewrite["local_name"]
            result += f"{self.indent()}{local_type} {local_name};\n"
        if hull_output_rewrite is not None:
            local_type = self.convert_type(hull_output_rewrite["return_type_name"])
            local_name = hull_output_rewrite["local_name"]
            result += f"{self.indent()}{local_type} {local_name};\n"
        if fragment_output_rewrite is not None:
            local_type = self.convert_type(fragment_output_rewrite["return_type_name"])
            local_name = fragment_output_rewrite["local_name"]
            self.register_variable_type(
                local_name,
                fragment_output_rewrite["return_type_name"],
                fragment_output_rewrite["parameter"],
            )
            result += f"{self.indent()}{local_type} {local_name};\n"
        for declaration in local_alias_declarations:
            result += f"{self.indent()}{declaration}\n"

        # The precise captures are declared at entry, once every statement has named the ones it uses.
        captures_offset = len(result)
        for statement_index, stmt in enumerate(body_statements):
            result += (
                self.emit_function_body_statement(
                    stmt, statement_index, builtin_return_rewrite
                )
                + "\n"
            )

        if builtin_return_rewrite is not None and builtin_return_rewrite["mode"] in {
            "local",
            "struct",
        }:
            result += f"{self.indent()}return {builtin_return_rewrite['local_name']};\n"
        if hull_output_rewrite is not None:
            result += f"{self.indent()}return {hull_output_rewrite['local_name']};\n"
        if fragment_output_rewrite is not None and not self.statement_body_terminates(
            body_statements
        ):
            result += (
                f"{self.indent()}return " f"{fragment_output_rewrite['local_name']};\n"
            )

        result = (
            result[:captures_offset]
            + self.slang_precise_capture_declarations(self.indent())
            + result[captures_offset:]
        )
        self.indent_level -= 1
        return result + "}"

    def slang_workgroup_size_constants(self):
        constants = getattr(self, "workgroup_size_constants", None)
        if constants is None:
            constants = self.workgroup_size_constants = SlangWorkgroupSizeConstants(
                None
            )
        return constants

    def generate_compute_numthreads(self, execution_config=None):
        # Specialized and host-chosen sizes name module-scope constants (slang_workgroup_size.py).
        return self.slang_workgroup_size_constants().numthreads(execution_config)

    def validate_slang_return_semantic(
        self, shader_type, semantic, stage_role, ret_type_name
    ):
        if semantic is None:
            return

        shader_stage = self.slang_shader_stage_name(shader_type)
        if shader_stage == "hull":
            if stage_role == "patch_constant":
                raise SlangRefusal(
                    "Slang patch constant function returns must put semantics on "
                    "the returned struct members"
                )

            raise SlangRefusal(
                "Slang tessellation_control stage returns must put semantics on "
                "the output control-point struct members"
            )

        if self.convert_type(ret_type_name) == "void":
            if shader_type:
                raise SlangRefusal(
                    f"Slang {shader_type} stage void return cannot use return "
                    f"semantic {semantic}"
                )
            raise SlangRefusal(
                f"Slang void function return cannot use return semantic {semantic}"
            )

        self.validate_slang_output_semantic_stage(
            shader_type, semantic, "return semantic", stage_role=stage_role
        )
        self.validate_slang_builtin_semantic_type(
            semantic, ret_type_name, "return semantic"
        )

    def validate_slang_struct_return_semantics(
        self, shader_type, return_type_name, stage_role=None
    ):
        if shader_type is None:
            return

        base_type = self.type_name_string(return_type_name)
        if not base_type:
            return
        base_type = base_type.split("<", 1)[0].split("[", 1)[0].strip()
        struct_node = self.user_structs_by_name.get(base_type)
        if struct_node is None:
            return

        for member in getattr(struct_node, "members", []) or []:
            semantic = self.semantic_from_node(member)
            if semantic is None:
                continue
            member_name = getattr(member, "name", "<anonymous>")
            context = f"struct return semantic '{base_type}.{member_name}'"
            self.validate_slang_output_semantic_stage(
                shader_type, semantic, context, stage_role=stage_role
            )
            member_type = self.slang_tess_factor_member_type_name(member)
            self.validate_slang_builtin_semantic_type(semantic, member_type, context)

    def slang_stage_hull_output_rewrite(
        self, body_statements, shader_type, ret_type_name, semantic, param_list
    ):
        if self.slang_shader_stage_name(shader_type) != "hull":
            return None

        gl_out_assignments = []
        for node in self.walk_ast(body_statements):
            if not isinstance(node, AssignmentNode):
                continue
            index = self.slang_hull_output_access_index(node.left)
            if index is not None:
                gl_out_assignments.append((node, index))

        if not gl_out_assignments:
            if self.slang_stage_uses_gl_out(body_statements):
                raise SlangRefusal(
                    "Slang hull stage gl_out requires assignments to "
                    "gl_out[gl_InvocationID] or gl_out[gl_InvocationID].field"
                )
            return None

        if semantic is not None:
            raise SlangRefusal(
                "Slang hull stage gl_out outputs cannot use a return semantic"
            )
        if self.contains_return_statement(body_statements):
            raise SlangRefusal(
                "Slang hull stage gl_out outputs cannot be mixed with explicit returns"
            )

        for assignment, index in gl_out_assignments:
            if getattr(assignment, "operator", None) != "=":
                raise SlangRefusal(
                    "Slang hull stage gl_out outputs require simple assignment"
                )
            if not self.is_slang_hull_output_index(index, param_list):
                raise SlangRefusal(
                    "Slang hull stage gl_out writes must target "
                    "gl_out[gl_InvocationID] or the SV_OutputControlPointID parameter"
                )

        for index in self.slang_hull_output_access_indices(body_statements):
            if not self.is_slang_hull_output_index(index, param_list):
                raise SlangRefusal(
                    "Slang hull stage gl_out accesses must target "
                    "gl_out[gl_InvocationID] or the SV_OutputControlPointID parameter"
                )

        indexed_identifier_ids = self.slang_hull_output_indexed_identifier_ids(
            body_statements
        )
        for node in self.walk_ast(body_statements):
            if (
                isinstance(node, IdentifierNode)
                and node.name == "gl_out"
                and id(node) not in indexed_identifier_ids
            ):
                raise SlangRefusal(
                    "Slang hull stage gl_out must be indexed as gl_out[gl_InvocationID]"
                )

        output_type_name = None
        removed_param_names = set()
        if self.convert_type(ret_type_name) != "void":
            output_type_name = ret_type_name
        else:
            output_patch_params = []
            for param in param_list or []:
                element_type = self.slang_patch_parameter_element_type_name(
                    param, "OutputPatch"
                )
                if element_type:
                    output_patch_params.append((param, element_type))

            if len(output_patch_params) == 1:
                output_patch_param, output_type_name = output_patch_params[0]
                removed_param_names.add(output_patch_param.name)
            else:
                raise SlangRefusal(
                    "Slang hull stage gl_out requires a non-void return type or "
                    "exactly one explicit OutputPatch<..., N> parameter"
                )

        return {
            "return_type_name": output_type_name,
            "local_name": self.unique_slang_builtin_output_local_name(
                "gl_OutputControlPoint", body_statements
            ),
            "removed_param_names": removed_param_names,
        }

    def slang_filtered_stage_parameters(self, param_list, hull_output_rewrite):
        if hull_output_rewrite is None:
            return param_list

        removed_param_names = hull_output_rewrite["removed_param_names"]
        if not removed_param_names:
            return param_list
        if not param_list:
            return param_list

        if hasattr(param_list[0], "name"):
            return [
                param
                for param in param_list
                if getattr(param, "name", None) not in removed_param_names
            ]
        return [
            param
            for param in param_list
            if len(param) < 2 or param[1] not in removed_param_names
        ]

    def slang_stage_uses_gl_out(self, body_statements):
        return any(
            isinstance(node, IdentifierNode) and node.name == "gl_out"
            for node in self.walk_ast(body_statements)
        )

    def slang_hull_output_access_index(self, target):
        if isinstance(target, MemberAccessNode):
            return self.slang_hull_output_access_index(
                getattr(target, "object", getattr(target, "object_expr", None))
            )
        if isinstance(target, ArrayAccessNode):
            array = getattr(target, "array", getattr(target, "array_expr", None))
            if self.identifier_name(array) == "gl_out":
                return getattr(target, "index", getattr(target, "index_expr", None))
            return self.slang_hull_output_access_index(array)
        return None

    def slang_hull_output_access_indices(self, body_statements):
        indices = []
        for node in self.walk_ast(body_statements):
            if not isinstance(node, ArrayAccessNode):
                continue
            array = getattr(node, "array", getattr(node, "array_expr", None))
            if self.identifier_name(array) == "gl_out":
                indices.append(
                    getattr(node, "index", getattr(node, "index_expr", None))
                )
        return indices

    def slang_hull_output_indexed_identifier_ids(self, body_statements):
        identifier_ids = set()
        for node in self.walk_ast(body_statements):
            if not isinstance(node, ArrayAccessNode):
                continue
            array = getattr(node, "array", getattr(node, "array_expr", None))
            if isinstance(array, IdentifierNode) and array.name == "gl_out":
                identifier_ids.add(id(array))
        return identifier_ids

    def slang_hull_output_member_target(self, target):
        if not isinstance(target, MemberAccessNode):
            return None

        obj = getattr(target, "object", getattr(target, "object_expr", None))
        if not isinstance(obj, ArrayAccessNode):
            return None

        array = getattr(obj, "array", getattr(obj, "array_expr", None))
        if self.identifier_name(array) != "gl_out":
            return None

        index = getattr(obj, "index", getattr(obj, "index_expr", None))
        return target.member, index

    def is_slang_hull_output_index(self, index, param_list):
        index_name = self.identifier_name(index)
        if index_name == "gl_InvocationID":
            return True

        for param in param_list or []:
            param_name = getattr(param, "name", None)
            if param_name != index_name:
                continue
            semantic = self.semantic_from_node(param)
            if self.map_semantic(semantic, "tessellation_control") == (
                "SV_OutputControlPointID"
            ):
                return True
        return False

    def slang_patch_parameter_element_type_name(self, param, patch_type):
        shape = self.slang_patch_parameter_shape(param, patch_type)
        if shape is not None:
            return shape[0]
        return None

    def slang_patch_parameters(self, param_list, patch_type):
        patch_params = []
        for param in param_list or []:
            shape = self.slang_patch_parameter_shape(param, patch_type)
            if shape is not None:
                patch_params.append((param, shape))
        return patch_params

    def slang_patch_parameter_shape(self, param, patch_type):
        type_node = getattr(param, "param_type", None)
        if getattr(type_node, "name", None) == patch_type:
            generic_args = getattr(type_node, "generic_args", []) or []
            if generic_args:
                element_type = self.convert_type_node_to_string(generic_args[0])
                patch_size = None
                if len(generic_args) > 1:
                    patch_size = self.convert_type_node_to_string(generic_args[1])
                return element_type, patch_size

        type_name = self.slang_parameter_type_name(param)
        prefix = f"{patch_type}<"
        if not type_name.startswith(prefix) or not type_name.endswith(">"):
            return None

        args = type_name[len(prefix) : -1]
        generic_args = self.slang_generic_arguments(args)
        if not generic_args:
            return None
        element_type = generic_args[0]
        patch_size = generic_args[1] if len(generic_args) > 1 else None
        return element_type, patch_size

    def first_slang_generic_argument(self, args):
        generic_args = self.slang_generic_arguments(args)
        if generic_args:
            return generic_args[0]
        return args.strip()

    def slang_generic_arguments(self, args):
        arguments = []
        start = 0
        depth = 0
        for index, char in enumerate(args):
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            elif char == "," and depth == 0:
                arguments.append(args[start:index].strip())
                start = index + 1
        trailing_arg = args[start:].strip()
        if trailing_arg:
            arguments.append(trailing_arg)
        return arguments

    def slang_implicit_stage_parameters(
        self, body_statements, shader_type, param_list, stage_role=None
    ):
        candidates = self.slang_implicit_stage_parameter_candidates(
            shader_type, stage_role=stage_role
        )
        if not candidates:
            return []

        declared_names = {
            getattr(param, "name", None)
            for param in param_list or []
            if getattr(param, "name", None)
        }
        existing_semantics = {
            self.map_semantic(self.semantic_from_node(param), shader_type)
            for param in param_list or []
            if self.semantic_from_node(param)
        }
        for node in self.walk_ast(body_statements):
            if isinstance(node, VariableNode):
                declared_names.add(node.name)

        used_names = set()
        for node in self.walk_ast(body_statements):
            if isinstance(node, IdentifierNode) and node.name in candidates:
                used_names.add(node.name)

        implicit_params = []
        for name, (param_type, semantic) in candidates.items():
            mapped_semantic = self.map_semantic(semantic)
            if (
                name in used_names
                and name not in declared_names
                and mapped_semantic not in existing_semantics
            ):
                implicit_params.append((param_type, name, mapped_semantic))
        return implicit_params

    def slang_stage_system_value_aliases(
        self, body_statements, shader_type, param_list, stage_role=None
    ):
        candidates = self.slang_implicit_stage_parameter_candidates(
            shader_type, stage_role=stage_role
        )
        if not candidates:
            return {}

        declared_names = {
            getattr(param, "name", None)
            for param in param_list or []
            if getattr(param, "name", None)
        }
        semantic_parameters = {}
        for param in param_list or []:
            param_name = getattr(param, "name", None)
            semantic = self.semantic_from_node(param)
            if param_name and semantic:
                semantic_parameters[self.map_semantic(semantic, shader_type)] = (
                    param_name
                )

        for node in self.walk_ast(body_statements):
            if isinstance(node, VariableNode):
                declared_names.add(node.name)

        aliases = {}
        for node in self.walk_ast(body_statements):
            if not isinstance(node, IdentifierNode) or node.name not in candidates:
                continue
            if node.name in declared_names:
                continue

            _param_type, semantic = candidates[node.name]
            existing_param_name = semantic_parameters.get(
                self.map_semantic(semantic, shader_type)
            )
            if existing_param_name and existing_param_name != node.name:
                aliases[node.name] = existing_param_name
        return aliases

    def slang_compute_global_invocation_id_parameter_bridge(
        self, parameter, param_type, semantic, emitted_param_names, param_list
    ):
        if semantic is None:
            return None
        if self.map_semantic(semantic, "compute") != "SV_DispatchThreadID":
            return None

        base_type, array_suffix = split_array_type_suffix(str(param_type))
        if array_suffix or base_type not in {"uint", "uint2"}:
            return None

        source_name = getattr(parameter, "name", None)
        if not isinstance(source_name, str) or not source_name:
            return None

        used_names = set(emitted_param_names)
        used_names.update(
            getattr(param, "name", None)
            for param in param_list or []
            if getattr(param, "name", None)
        )
        used_names.update(self.variable_types)
        native_name = self.slang_unique_local_identifier(
            f"{source_name}_dispatchThreadID", used_names
        )
        declaration = f"uint3 {native_name} : SV_DispatchThreadID"
        if base_type == "uint":
            local_alias = f"uint {source_name} = {native_name}.x;"
        else:
            local_alias = f"uint2 {source_name} = {native_name}.xy;"
        return declaration, local_alias, native_name

    def slang_num_workgroups_parameter_alias(
        self, parameter, param_type, semantic, shader_type
    ):
        """Bind a ``gl_NumWorkGroups`` or ``gl_WorkGroupSize`` entry parameter from its Slang intrinsic.

        Slang has no system-value semantic for the dispatch size or the workgroup size, so the parameter becomes a
        local read from ``WorkgroupCount()`` or ``WorkgroupSize()`` (the entry's ``numthreads`` as an ``int3``) in a
        compute entry point and is refused in other entry points.
        """
        intrinsics = {
            "gl_numworkgroups": ("gl_NumWorkGroups", "WorkgroupCount()", "{}"),
            "gl_workgroupsize": ("gl_WorkGroupSize", "WorkgroupSize()", "uint3({})"),
        }
        if semantic is None or str(semantic).lower() not in intrinsics:
            return None
        builtin, intrinsic, value = intrinsics[str(semantic).lower()]
        name = getattr(parameter, "name", None)
        if shader_type is None:
            return None
        if shader_type != "compute":
            self.refuse_lowering(
                f"Slang has no system value for the {builtin} parameter '{name}' "
                f"of a {shader_type} entry point; {intrinsic} is read in compute "
                "entry points only"
            )
            return None
        base_type, array_suffix = split_array_type_suffix(str(param_type))
        if array_suffix or base_type != "uint3":
            self.refuse_lowering(
                f"Slang {builtin} parameter '{name}' must be a uvec3, not {param_type}"
            )
            return None
        return f"uint3 {name} = {value.format(intrinsic)};"

    def slang_helper_invocation_parameter_alias(
        self, parameter, param_type, semantic, shader_type
    ):
        """Bind a ``gl_HelperInvocation`` entry parameter from ``IsHelperLane()``.

        Slang has no system-value semantic for it, so the parameter becomes a local read from the
        ``IsHelperLane()`` intrinsic in a fragment entry point and is refused in other entry points.
        """
        if semantic is None or str(semantic).lower() != "gl_helperinvocation":
            return None
        name = getattr(parameter, "name", None)
        if shader_type is None:
            return None
        if self.slang_shader_stage_name(shader_type) != "fragment":
            self.refuse_lowering(
                f"Slang has no system value for the gl_HelperInvocation parameter '{name}' "
                f"of a {shader_type} entry point; IsHelperLane() is read in fragment "
                "entry points only"
            )
            return None
        if str(param_type) != "bool":
            self.refuse_lowering(
                f"Slang gl_HelperInvocation parameter '{name}' must be a bool, not {param_type}"
            )
            return None
        return f"bool {name} = IsHelperLane();"

    def slang_unique_local_identifier(self, candidate, used_names):
        while candidate in used_names:
            candidate = f"{candidate}_"
        return candidate

    def slang_stage_intrinsic_builtin_aliases(
        self, body_statements, shader_type, param_list
    ):
        candidates = self.slang_intrinsic_builtin_candidates(shader_type)

        declared_names = {
            getattr(param, "name", None)
            for param in param_list or []
            if getattr(param, "name", None)
        }
        identifiers = []
        for node in self.walk_ast(body_statements):
            if isinstance(node, VariableNode):
                declared_names.add(node.name)
            elif isinstance(node, IdentifierNode):
                identifiers.append(node)

        aliases = {}
        for node in identifiers:
            if node.name in declared_names:
                continue
            if node.name in candidates:
                return_type, intrinsic = candidates[node.name]
                aliases[node.name] = intrinsic
                self.register_variable_type(node.name, return_type)
            elif node.name in ("gl_NumWorkGroups", "gl_WorkGroupSize"):
                # Slang declares neither, and the WorkgroupCount() and WorkgroupSize()
                # aliases above are only made for a compute entry point.
                where = (
                    f"a {shader_type} entry point"
                    if shader_type
                    else "a helper function"
                )
                intrinsic = (
                    "WorkgroupCount()"
                    if node.name == "gl_NumWorkGroups"
                    else "WorkgroupSize()"
                )
                self.refuse_lowering(
                    f"Slang {node.name}: lowered to {intrinsic} only "
                    f"in a compute entry point, not in {where}"
                )
            elif node.name in SLANG_SUBGROUP_WORKGROUP_BUILTINS:
                # GL_KHR_shader_subgroup and Slang's WaveGetNumWaves()/WaveGetWaveIndex() are compute-class only.
                intrinsic = SLANG_SUBGROUP_WORKGROUP_BUILTINS[node.name][1]
                self.refuse_lowering(
                    f"Slang {node.name}: {intrinsic} exists only in compute, mesh "
                    f"and amplification entry points, not in a {shader_type} entry point"
                )
        return aliases

    def slang_stage_patch_input_aliases(self, body_statements, shader_type, param_list):
        shader_stage = self.slang_shader_stage_name(shader_type)
        patch_type = {
            "hull": "InputPatch",
            "domain": "OutputPatch",
        }.get(shader_stage)
        if patch_type is None:
            return {}

        declared_names = {
            getattr(param, "name", None)
            for param in param_list or []
            if getattr(param, "name", None)
        }
        for node in self.walk_ast(body_statements):
            if isinstance(node, VariableNode):
                declared_names.add(node.name)

        uses_gl_in = any(
            isinstance(node, IdentifierNode)
            and node.name == "gl_in"
            and node.name not in declared_names
            for node in self.walk_ast(body_statements)
        )
        if not uses_gl_in:
            return {}

        patch_params = [
            param
            for param in param_list or []
            if self.slang_parameter_type_name(param).startswith(f"{patch_type}<")
        ]
        if len(patch_params) == 1:
            patch_param = patch_params[0]
            self.register_variable_type(
                "gl_in", self.slang_parameter_type_name(patch_param), patch_param
            )
            return {"gl_in": patch_param.name}

        raise SlangRefusal(
            f"Slang {shader_stage} stage gl_in requires exactly one explicit "
            f"{patch_type}<..., N> parameter"
        )

    def slang_parameter_type_name(self, param):
        if hasattr(param, "param_type"):
            return self.convert_type_node_to_string(param.param_type)
        if hasattr(param, "vtype"):
            return str(param.vtype)
        return ""

    def slang_intrinsic_builtin_candidates(self, shader_type):
        shader_stage = self.slang_shader_stage_name(shader_type)
        candidates = slang_subgroup_builtin_candidates(shader_stage)
        candidates.update(self.slang_stage_intrinsic_builtin_candidates(shader_stage))
        return candidates

    def slang_stage_intrinsic_builtin_candidates(self, shader_stage):
        if shader_stage == "compute":
            # Slang has no SV_ system value for the dispatch size; it exposes
            # the number of workgroups through the WorkgroupCount() intrinsic.
            return {
                "gl_NumWorkGroups": ("uvec3", "WorkgroupCount()"),
                # Slang declares no gl_WorkGroupSize; WorkgroupSize() is the entry's numthreads as an int3.
                "gl_WorkGroupSize": ("uvec3", "uint3(WorkgroupSize())"),
            }
        if shader_stage not in self.slang_ray_stage_types():
            return {}

        return {
            "gl_HitKindEXT": ("uint", "HitKind()"),
            "gl_HitTEXT": ("float", "RayTCurrent()"),
            # In ray tracing stages gl_InstanceID is the instance's index and
            # gl_InstanceCustomIndexEXT its application-defined custom index.
            "gl_InstanceCustomIndexEXT": ("uint", "InstanceID()"),
            "gl_InstanceID": ("uint", "InstanceIndex()"),
            "gl_LaunchIDEXT": ("uvec3", "DispatchRaysIndex()"),
            "gl_LaunchSizeEXT": ("uvec3", "DispatchRaysDimensions()"),
            "gl_GeometryIndexEXT": ("uint", "GeometryIndex()"),
            "gl_ObjectRayDirectionEXT": ("vec3", "ObjectRayDirection()"),
            "gl_ObjectRayOriginEXT": ("vec3", "ObjectRayOrigin()"),
            "gl_PrimitiveID": ("uint", "PrimitiveIndex()"),
            "gl_RayTmaxEXT": ("float", "RayTCurrent()"),
            "gl_RayTminEXT": ("float", "RayTMin()"),
            "gl_WorldRayDirectionEXT": ("vec3", "WorldRayDirection()"),
            "gl_WorldRayOriginEXT": ("vec3", "WorldRayOrigin()"),
        }

    def slang_implicit_stage_parameter_candidates(self, shader_type, stage_role=None):
        shader_stage = self.slang_shader_stage_name(shader_type)
        thread_parameters = {
            "gl_WorkGroupID": ("uvec3", "SV_GroupID"),
            "gl_LocalInvocationID": ("uvec3", "SV_GroupThreadID"),
            "gl_GlobalInvocationID": ("uvec3", "SV_DispatchThreadID"),
            "gl_LocalInvocationIndex": ("uint", "SV_GroupIndex"),
        }
        if shader_stage in {"compute", "mesh", "amplification"}:
            return thread_parameters
        if shader_stage == "vertex":
            return {
                "gl_VertexID": ("uint", "SV_VertexID"),
                "gl_InstanceID": ("uint", "SV_InstanceID"),
                "gl_BaseVertex": ("int", "SV_StartVertexLocation"),
                "gl_BaseInstance": ("uint", "SV_StartInstanceLocation"),
                "gl_DrawID": ("uint", "SV_DrawIndex"),
            }
        if shader_stage == "fragment":
            return {
                "gl_FragCoord": ("vec4", "SV_Position"),
                "gl_PointCoord": ("vec2", "SV_PointCoord"),
                "gl_FrontFacing": ("bool", "SV_IsFrontFace"),
                "gl_PrimitiveID": ("uint", "SV_PrimitiveID"),
                "gl_SampleID": ("uint", "SV_SampleIndex"),
                "gl_SampleMaskIn": ("uint", "SV_Coverage"),
                "gl_Layer": ("uint", "SV_RenderTargetArrayIndex"),
                "gl_ViewportIndex": ("uint", "SV_ViewportArrayIndex"),
                # GL_EXT_fragment_shader_barycentric; the no-perspective one is written noperspective
                # (SLANG_IMPLICIT_PARAMETER_MODIFIERS), which slangc writes as BaryCoordNoPerspKHR.
                "gl_BaryCoordEXT": ("vec3", "SV_Barycentrics"),
                "gl_BaryCoordNoPerspEXT": ("vec3", "SV_Barycentrics"),
                # GL_EXT_fragment_shading_rate (slang_view_shading_rate.py).
                "gl_ShadingRateEXT": ("int", "SV_ShadingRate"),
            }
        if shader_stage == "geometry":
            return {
                "gl_PrimitiveIDIn": ("uint", "SV_PrimitiveID"),
                "gl_InvocationID": ("uint", "SV_GSInstanceID"),
            }
        if shader_stage == "hull":
            if stage_role == "patch_constant":
                return {
                    "gl_PrimitiveID": ("uint", "SV_PrimitiveID"),
                }
            return {
                "gl_InvocationID": ("uint", "SV_OutputControlPointID"),
                "gl_PrimitiveID": ("uint", "SV_PrimitiveID"),
            }
        if shader_stage == "domain":
            return {
                "gl_TessCoord": ("vec3", "SV_DomainLocation"),
                "gl_PrimitiveID": ("uint", "SV_PrimitiveID"),
            }
        return {}

    def walk_ast(self, root):
        return walk_ast_values(root)

    def slang_stage_builtin_return_rewrite(
        self, body_statements, shader_type, ret_type_name, semantic
    ):
        if semantic is not None:
            return None
        if self.convert_type(ret_type_name) != "void" or not body_statements:
            return None
        if self.contains_return_statement(body_statements):
            return None

        output_assignments = self.slang_stage_builtin_output_assignments(
            body_statements, shader_type
        )
        output_targets = self.unique_slang_builtin_output_targets(output_assignments)
        if not output_targets:
            return None

        if len(output_targets) > 1:
            return self.slang_stage_builtin_struct_return_rewrite(
                shader_type, body_statements, output_targets
            )

        target_name = output_targets[0]
        return_type_name = self.slang_stage_builtin_return_type(
            shader_type, target_name
        )
        if return_type_name is None:
            return None

        final_statement = body_statements[-1]
        assignment = self.assignment_from_statement(final_statement)
        if (
            len(output_assignments) == 1
            and assignment is output_assignments[0][0]
            and getattr(assignment, "operator", None) == "="
        ):
            return {
                "mode": "direct",
                "statement_index": len(body_statements) - 1,
                "return_type_name": return_type_name,
                "semantic": target_name,
                "value": getattr(
                    assignment, "right", getattr(assignment, "value", None)
                ),
            }

        return {
            "mode": "local",
            "return_type_name": return_type_name,
            "semantic": target_name,
            "target_name": target_name,
            "local_name": self.unique_slang_builtin_output_local_name(
                target_name, body_statements
            ),
        }

    def contains_return_statement(self, body_statements):
        return any(
            isinstance(node, ReturnNode) for node in self.walk_ast(body_statements)
        )

    def unique_slang_builtin_output_targets(self, output_assignments):
        targets = []
        seen = set()
        for _assignment, target_name in output_assignments:
            if target_name not in seen:
                seen.add(target_name)
                targets.append(target_name)
        return targets

    def slang_stage_builtin_struct_return_rewrite(
        self, shader_type, body_statements, output_targets
    ):
        struct_name = self.unique_slang_builtin_output_struct_name(shader_type)
        local_name = self.unique_slang_builtin_output_local_name(
            "gl_Output", body_statements
        )
        fields = []
        for target_name in output_targets:
            return_type_name = self.slang_stage_builtin_return_type(
                shader_type, target_name
            )
            if return_type_name is None:
                return None
            fields.append(
                {
                    "target_name": target_name,
                    "field_name": self.slang_builtin_output_field_name(target_name),
                    "return_type_name": return_type_name,
                    "semantic": target_name,
                }
            )

        return {
            "mode": "struct",
            "return_type_name": struct_name,
            "semantic": None,
            "local_name": local_name,
            "fields": fields,
        }

    def unique_slang_builtin_output_struct_name(self, shader_type):
        base_name = f"CGL{self.slang_stage_entry_function_name(shader_type)}Output"
        used_names = set(self.user_symbol_names)
        if base_name not in used_names:
            return base_name

        suffix = 1
        while f"{base_name}_{suffix}" in used_names:
            suffix += 1
        return f"{base_name}_{suffix}"

    def slang_builtin_output_field_name(self, target_name):
        target_suffix = (
            target_name[3:] if target_name.startswith("gl_") else target_name
        )
        return f"cgl_{target_suffix}"

    def slang_stage_builtin_return_aliases(self, rewrite):
        if rewrite is None:
            return {}
        if rewrite["mode"] == "local":
            return {rewrite["target_name"]: rewrite["local_name"]}
        if rewrite["mode"] == "struct":
            return {
                field["target_name"]: f"{rewrite['local_name']}.{field['field_name']}"
                for field in rewrite["fields"]
            }
        return {}

    def generate_slang_builtin_output_struct(self, rewrite):
        result = f"struct {rewrite['return_type_name']}\n{{\n"
        for field in rewrite["fields"]:
            field_type = self.convert_type(field["return_type_name"])
            semantic = self.semantic_suffix(field["semantic"])
            result += f"    {field_type} {field['field_name']}{semantic};\n"
        result += "};"
        return result

    def slang_stage_builtin_output_assignments(self, body_statements, shader_type):
        assignments = []
        for node in self.walk_ast(body_statements):
            if not isinstance(node, AssignmentNode):
                continue
            if getattr(node, "operator", None) != "=":
                continue

            target = getattr(node, "left", getattr(node, "target", None))
            target_name = self.slang_stage_builtin_output_target_name(target)
            if self.slang_stage_builtin_return_type(shader_type, target_name):
                assignments.append((node, target_name))
        return assignments

    def slang_stage_builtin_output_target_name(self, target):
        target_name = self.identifier_name(target)
        if target_name is not None:
            return target_name

        if isinstance(target, ArrayAccessNode):
            array_name = self.identifier_name(getattr(target, "array", None))
            index = self.literal_int_value(getattr(target, "index", None))
            if array_name == "gl_FragData":
                if index is None or index < 0:
                    raise SlangRefusal(
                        "Slang fragment output gl_FragData requires a "
                        "non-negative literal render-target index"
                    )
                return f"gl_FragColor{index}"
            if array_name == "gl_SampleMask":
                if index != 0:
                    raise SlangRefusal(
                        "Slang fragment output gl_SampleMask requires literal "
                        "index 0"
                    )
                return "gl_SampleMask"

        return None

    def unique_slang_builtin_output_local_name(self, target_name, body_statements):
        target_suffix = (
            target_name[3:] if target_name.startswith("gl_") else target_name
        )
        base_name = f"cgl_{target_suffix}"
        used_names = set()
        for node in self.walk_ast(body_statements):
            if hasattr(node, "name") and isinstance(getattr(node, "name"), str):
                used_names.add(node.name)
        if base_name not in used_names:
            return base_name

        suffix = 1
        while f"{base_name}_{suffix}" in used_names:
            suffix += 1
        return f"{base_name}_{suffix}"

    def slang_stage_builtin_return_type(self, shader_type, target_name):
        if target_name is None:
            return None
        if shader_type == "vertex":
            if target_name == "gl_Position":
                return "vec4"
            if target_name == "gl_PointSize":
                return "float"
            if target_name in {"gl_Layer", "gl_ViewportIndex"}:
                return "uint"
        if shader_type == "fragment":
            if target_name == "gl_FragDepth":
                return "float"
            if target_name == "gl_SampleMask":
                return "uint"
            if target_name == "gl_FragColor" or (
                target_name.startswith("gl_FragColor")
                and target_name[len("gl_FragColor") :].isdigit()
            ):
                return "vec4"
        return None
