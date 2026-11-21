"""Resource globals and diagnostics in the HLSL generator: textures and samplers, entry-only resources, byte address interlocked buffers, stage functions and storage variable types."""

import re
from copy import deepcopy

from ..ast import (
    ArrayAccessNode,
    FunctionCallNode,
    MemberAccessNode,
    NamedType,
    PointerType,
    ReferenceType,
    StructMemberNode,
    StructNode,
    VariableNode,
)
from ..refusals import DirectXRefusal
from ..validation import expression_debug_name
from .array_utils import split_array_type_suffix
from .builtin_variable_types import crossgl_builtin_variable_type
from .enum_utils import (
    generic_type_parts,
    sanitize_type_name,
)
from .hlsl_constant_matrices import hlsl_constant_matrix_member_type
from .image_access_contracts import (
    TEXTURE_COMPARE_INTRINSIC_NAMES,
    TEXTURE_GATHER_COMPARE_INTRINSIC_NAMES,
    TEXTURE_GATHER_INTRINSIC_NAMES,
    TEXTURE_IMPLICIT_SAMPLER_INTRINSIC_NAMES,
    TEXTURE_QUERY_LOD_INTRINSIC_NAMES,
    TEXTURE_SAMPLING_INTRINSIC_NAMES,
    is_projected_texture_basic_offset_operation,
    is_projected_texture_compare_operation,
    is_projected_texture_grad_offset_operation,
    is_projected_texture_lod_offset_operation,
    is_projected_texture_operation,
    is_texture_compare_grad_offset_operation,
    is_texture_compare_lod_offset_operation,
    is_texture_compare_non_projected_offset_operation,
    is_texture_compare_offset_operation,
    is_texture_gather_compare_offset_operation,
    is_texture_gather_compare_operation,
    is_texture_gather_offset_operation,
    is_texture_query_lod_operation,
    is_texture_sample_offset_operation,
)
from .stage_utils import (
    collect_stage_entry_records,
    collect_stage_local_variables,
    declaration_signature,
    deduplicate_named_declarations,
)


class DirectXResourceGlobalMixin:
    """Resource global members of ``HLSLCodeGen``."""

    def hlsl_byteaddress_interlocked_member_call_parts(self, expr):
        if not (
            isinstance(expr, FunctionCallNode)
            or (hasattr(expr, "__class__") and "FunctionCall" in str(expr.__class__))
        ):
            return None

        func_expr = getattr(expr, "function", getattr(expr, "name", None))
        if not (
            isinstance(func_expr, MemberAccessNode)
            or (
                hasattr(func_expr, "__class__")
                and "MemberAccess" in str(func_expr.__class__)
            )
        ):
            return None

        method_name = str(getattr(func_expr, "member", ""))
        if method_name not in self.hlsl_byteaddress_interlocked_operations():
            return None

        object_expr = getattr(
            func_expr, "object", getattr(func_expr, "object_expr", None)
        )
        resource_type = self.hlsl_buffer_helper_resource_type(object_expr)
        if resource_type is None:
            return None

        args = getattr(expr, "arguments", getattr(expr, "args", []))
        return method_name, func_expr, args

    def generate_hlsl_byteaddress_interlocked_statement_expression(self, expr):
        previous_allowed = self.allow_hlsl_byteaddress_interlocked_member_expression
        self.allow_hlsl_byteaddress_interlocked_member_expression = True
        try:
            return self.generate_expression(expr)
        finally:
            self.allow_hlsl_byteaddress_interlocked_member_expression = previous_allowed

    def hlsl_byteaddress_interlocked_arity_label(self, arities):
        if len(arities) == 1:
            return str(arities[0])
        return " or ".join(str(arity) for arity in arities)

    def hlsl_byteaddress_interlocked_error_prefix(self, method_name):
        return f"DirectX ByteAddressBuffer interlocked member '{method_name}'"

    def validate_hlsl_byteaddress_interlocked_uint_input(
        self, method_name, args, index, role
    ):
        if index >= len(args):
            return
        if self.hlsl_byteaddress_interlocked_uint_input_matches(args[index]):
            return
        kind = (
            self.scalar_expression_kind(args[index])
            or self.type_name_string(self.expression_result_type(args[index]))
            or "unknown"
        )
        raise DirectXRefusal(
            f"{self.hlsl_byteaddress_interlocked_error_prefix(method_name)} "
            f"{role} argument must be scalar uint, got {kind}"
        )

    def validate_hlsl_byteaddress_interlocked_original(
        self, method_name, original_arg, signed=False
    ):
        original_kind = self.scalar_expression_kind(original_arg)
        expected = "int or uint" if signed else "uint"
        if original_kind != "uint" and not (signed and original_kind == "int"):
            original_type = (
                self.type_name_string(self.expression_result_type(original_arg))
                or original_kind
                or expression_debug_name(original_arg)
            )
            raise DirectXRefusal(
                f"{self.hlsl_byteaddress_interlocked_error_prefix(method_name)} "
                f"original argument must be scalar {expected}, got {original_type}"
            )
        if not self.hlsl_typed_buffer_atomic_original_is_lvalue(original_arg):
            raise DirectXRefusal(
                f"{self.hlsl_byteaddress_interlocked_error_prefix(method_name)} "
                f"original argument must be an assignable scalar {expected} target"
            )

    def comparison_texture_function_names(self):
        return TEXTURE_COMPARE_INTRINSIC_NAMES | TEXTURE_GATHER_COMPARE_INTRINSIC_NAMES

    def regular_texture_function_names(self):
        return (
            TEXTURE_SAMPLING_INTRINSIC_NAMES
            | TEXTURE_GATHER_INTRINSIC_NAMES
            | TEXTURE_QUERY_LOD_INTRINSIC_NAMES
        )

    def collect_regular_sampler_parameters(self, root):
        regular_params = {}
        functions = self.collect_functions(root)
        global_resource_types = self.collect_global_texture_types(root)

        for func in functions:
            func_name = getattr(func, "name", None)
            if not func_name:
                continue
            sampler_params = {
                param.name
                for param in getattr(func, "parameters", getattr(func, "params", []))
                if self.is_sampler_type(
                    getattr(param, "param_type", getattr(param, "vtype", None))
                )
            }
            if not sampler_params:
                continue

            previous_local_variable_types = self.local_variable_types
            self.local_variable_types = self.function_scope_variable_types(func)
            try:
                for node in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    func_expr = getattr(node, "function", getattr(node, "name", None))
                    func_name_expr = self.expression_name(func_expr)
                    if func_name_expr not in self.regular_texture_function_names():
                        continue
                    args = getattr(node, "arguments", getattr(node, "args", []))
                    if not self.has_explicit_sampler_argument(
                        func_name_expr, args, sampler_params
                    ):
                        continue
                    texture_type = self.texture_argument_analysis_type(
                        args[0], global_resource_types
                    )
                    if self.texture_call_is_diagnostic_only(
                        func_name_expr, texture_type
                    ):
                        continue
                    sampler_name = self.expression_name(args[1])
                    if sampler_name in sampler_params:
                        regular_params.setdefault(func_name, set()).add(sampler_name)
            finally:
                self.local_variable_types = previous_local_variable_types

        changed = True
        while changed:
            changed = False
            for func in functions:
                func_name = getattr(func, "name", None)
                if not func_name:
                    continue

                sampler_param_names = {
                    param.name
                    for param in getattr(
                        func, "parameters", getattr(func, "params", [])
                    )
                    if self.is_sampler_type(
                        getattr(param, "param_type", getattr(param, "vtype", None))
                    )
                }
                if not sampler_param_names:
                    continue

                for node in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    func_expr = getattr(node, "function", getattr(node, "name", None))
                    callee_name = self.expression_name(func_expr)
                    required_indices = self.comparison_sampler_parameter_indices(
                        functions, regular_params, callee_name
                    )
                    if not required_indices:
                        continue

                    args = getattr(node, "arguments", getattr(node, "args", []))
                    for index in required_indices:
                        if index >= len(args):
                            continue
                        arg_name = self.expression_name(args[index])
                        if arg_name in sampler_param_names:
                            current = regular_params.setdefault(func_name, set())
                            if arg_name not in current:
                                current.add(arg_name)
                                changed = True

        return regular_params

    def collect_regular_sampler_arguments(self, root, regular_params):
        return self.collect_sampler_arguments(root, regular_params)

    def collect_sampler_arguments(self, root, sampler_params_by_function):
        sampler_names = set()
        functions = self.collect_functions(root)

        for func in functions:
            for node in self.walk_ast(getattr(func, "body", [])):
                if not isinstance(node, FunctionCallNode):
                    continue
                func_expr = getattr(node, "function", getattr(node, "name", None))
                callee_name = self.expression_name(func_expr)
                required_indices = self.comparison_sampler_parameter_indices(
                    functions, sampler_params_by_function, callee_name
                )
                if not required_indices:
                    continue

                args = getattr(node, "arguments", getattr(node, "args", []))
                for index in required_indices:
                    if index >= len(args):
                        continue
                    arg_name = self.expression_name(args[index])
                    if arg_name:
                        sampler_names.add(arg_name)

        return sampler_names

    def collect_explicit_sampler_resource_names(self, root, texture_funcs):
        sampler_names = set()
        global_sampler_names = self.collect_global_sampler_names(root)
        global_resource_types = self.collect_global_texture_types(root)
        previous_local_variable_types = self.local_variable_types
        try:
            for func in self.collect_functions(root):
                self.local_variable_types = self.function_scope_variable_types(func)
                sampler_scope = global_sampler_names | {
                    param.name
                    for param in getattr(
                        func, "parameters", getattr(func, "params", [])
                    )
                    if self.is_sampler_type(
                        getattr(param, "param_type", getattr(param, "vtype", None))
                    )
                }
                for node in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    func_expr = getattr(node, "function", getattr(node, "name", None))
                    func_name = self.expression_name(func_expr)
                    if func_name not in texture_funcs:
                        continue
                    args = getattr(node, "arguments", getattr(node, "args", []))
                    if not self.has_explicit_sampler_argument(
                        func_name, args, sampler_scope
                    ):
                        continue
                    texture_type = self.texture_argument_analysis_type(
                        args[0], global_resource_types
                    )
                    if self.texture_call_is_diagnostic_only(func_name, texture_type):
                        continue
                    sampler_name = self.expression_name(args[1])
                    if sampler_name:
                        sampler_names.add(sampler_name)
        finally:
            self.local_variable_types = previous_local_variable_types
        return sampler_names

    def collect_sampler_struct_members(self, root, sampler_params, texture_funcs):
        members = set()
        previous_local_variable_types = self.local_variable_types
        functions = self.collect_functions(root)
        global_resource_types = self.collect_global_texture_types(root)
        try:
            for func in functions:
                self.local_variable_types = self.function_scope_variable_types(func)
                for node in self.walk_ast(getattr(func, "body", [])):
                    if not isinstance(node, FunctionCallNode):
                        continue
                    func_expr = getattr(node, "function", getattr(node, "name", None))
                    func_name = self.expression_name(func_expr)
                    args = getattr(node, "arguments", getattr(node, "args", []))
                    if (
                        func_name in texture_funcs
                        and self.has_explicit_sampler_argument(func_name, args, set())
                    ):
                        texture_type = self.texture_argument_analysis_type(
                            args[0], global_resource_types
                        )
                        if self.texture_call_is_diagnostic_only(
                            func_name, texture_type
                        ):
                            continue
                        member_ref = self.sampler_struct_member_reference(args[1])
                        if member_ref is not None:
                            members.add(member_ref)

                    required_indices = self.comparison_sampler_parameter_indices(
                        functions, sampler_params, func_name
                    )
                    for index in required_indices:
                        if index >= len(args):
                            continue
                        member_ref = self.sampler_struct_member_reference(args[index])
                        if member_ref is not None:
                            members.add(member_ref)
        finally:
            self.local_variable_types = previous_local_variable_types
        return members

    def function_scope_variable_types(self, func):
        variable_types = {}
        for param in getattr(func, "parameters", getattr(func, "params", [])):
            param_type = getattr(param, "param_type", getattr(param, "vtype", None))
            variable_types[param.name] = self.type_name_string(param_type)

        for node in self.walk_ast(getattr(func, "body", [])):
            if not isinstance(node, VariableNode):
                continue
            name = getattr(node, "name", None)
            if not name:
                continue
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            variable_types[name] = self.type_name_string(vtype)

        return variable_types

    def collect_global_variable_types(self, global_vars):
        variable_types = {}
        for node in global_vars or []:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            if not name:
                continue

            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            if vtype is None:
                vtype = getattr(node, "param_type", None)
            type_name = (
                self.hlsl_struct_buffer_source_resource_type(node, vtype)
                or self.hlsl_struct_buffer_resource_types.get(name)
                or self.type_name_string(vtype)
            )
            if not type_name:
                continue

            if "[" not in str(type_name):
                array_size = self.hlsl_resource_array_size_expression(node, vtype)
                if array_size is not None:
                    type_name = f"{type_name}[{self.expression_to_string(array_size)}]"

            variable_types[name] = type_name
        return variable_types

    def collect_hlsl_struct_buffer_resource_types(self, nodes):
        resources = {}
        for node in nodes or []:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            resource_type = self.hlsl_struct_buffer_resource_type(node)
            if name and resource_type:
                resources[name] = resource_type
        return resources

    def hlsl_struct_buffer_source_resource_type(self, node, vtype=None):
        """Retain source element semantics for physically lowered buffers."""
        if node is None:
            return None
        annotated_source_type = self.hlsl_bfloat16_storage_resource_source_type(node)
        if annotated_source_type:
            type_name = self.type_name_string(vtype)
            generic_base, generic_args = generic_type_parts(type_name)
            if len(generic_args) == 1:
                return f"{generic_base}<{annotated_source_type}>"

        type_name = self.type_name_string(vtype)
        source_element = self.hlsl_typed_buffer_element_type(
            type_name, self.hlsl_half_storage.resource_types
        )
        if source_element is not None and self.hlsl_half_storage.lowered(
            source_element
        ):
            return type_name

        qualifiers = {str(value).lower() for value in getattr(node, "qualifiers", [])}
        if (
            "buffer" not in qualifiers
            or self.glsl_buffer_block_attribute(node) is not None
        ):
            return None

        if vtype is None:
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            if vtype is None:
                vtype = getattr(node, "param_type", None)
        pointee_type = self.hlsl_buffer_pointer_pointee_type(vtype)
        source_element_type = self.hlsl_bfloat16_scalar_type_name(pointee_type)
        if source_element_type is None and self.hlsl_half_storage.lowered(pointee_type):
            source_element_type = pointee_type
        if source_element_type is None:
            return None

        resource_name = (
            "StructuredBuffer" if "readonly" in qualifiers else "RWStructuredBuffer"
        )
        return f"{resource_name}<{source_element_type}>"

    def hlsl_struct_buffer_resource_type(self, node, vtype=None):
        if node is None:
            return None
        annotated_source_type = self.hlsl_bfloat16_storage_resource_source_type(node)
        if annotated_source_type:
            type_name = self.type_name_string(vtype)
            generic_base, generic_args = generic_type_parts(type_name)
            if len(generic_args) == 1 and generic_args[0] == "uint16_t":
                return type_name

        qualifiers = {str(value).lower() for value in getattr(node, "qualifiers", [])}
        if "buffer" not in qualifiers:
            return None
        if self.glsl_buffer_block_attribute(node) is not None:
            return None

        if vtype is None:
            vtype = getattr(node, "var_type", getattr(node, "vtype", None))
            if vtype is None:
                vtype = getattr(node, "param_type", None)

        pointee_type = self.hlsl_buffer_pointer_pointee_type(vtype)
        if pointee_type:
            resource_name = (
                "StructuredBuffer" if "readonly" in qualifiers else "RWStructuredBuffer"
            )
            storage_type = self.hlsl_structured_storage_type(
                pointee_type,
                operation=f"buffer '{getattr(node, 'name', '<anonymous>')}' element",
                source_location=getattr(node, "source_location", None),
            )
            return f"{resource_name}<{storage_type}>"

        type_name = self.type_name_string(vtype)
        if not type_name:
            return None
        base_type = self.resource_base_type(type_name)
        if base_type not in self.structs_by_name:
            return None
        if any(ch in str(type_name) for ch in "*&"):
            return None

        resource_name = (
            "StructuredBuffer" if "readonly" in qualifiers else "RWStructuredBuffer"
        )
        return f"{resource_name}<{self.hlsl_half_storage.physical_type(base_type)}>"

    def hlsl_bfloat16_storage_resource_source_type(self, node):
        annotations = getattr(node, "annotations", {}) or {}
        annotated_source_type = annotations.get(
            "directx.storage_source_type"
        ) or annotations.get("directx.bfloat16_storage_source_type")
        if annotated_source_type:
            return annotated_source_type
        return self.hlsl_bfloat16_storage_resource_source_types.get(id(node))

    def hlsl_buffer_pointer_pointee_type(self, vtype):
        if isinstance(vtype, PointerType):
            return self.hlsl_storage_buffer_element_type(
                self.type_name_string(vtype.pointee_type)
            )
        type_name = self.type_name_string(vtype)
        if not type_name:
            return None
        type_name = str(type_name).strip()
        if not type_name.endswith("*"):
            return None
        return self.hlsl_storage_buffer_element_type(type_name[:-1])

    def hlsl_storage_buffer_element_type(self, type_name):
        type_name = str(type_name or "").strip()
        if type_name.startswith("atomic<") and type_name.endswith(">"):
            return type_name[len("atomic<") : -1].strip() or None
        return type_name or None

    def variable_type_by_name(self, name):
        if not name:
            return None
        if name in self.local_variable_types:
            return self.local_variable_types[name]
        if name in self.global_variable_types:
            return self.global_variable_types[name]
        builtin_type = crossgl_builtin_variable_type(name)
        return self.map_type(builtin_type) if builtin_type is not None else None

    def sampler_struct_member_reference(self, expr):
        while isinstance(expr, ArrayAccessNode):
            expr = getattr(expr, "array", getattr(expr, "array_expr", None))
        if not isinstance(expr, MemberAccessNode):
            return None

        member_name = str(expr.member)
        object_type = self.type_name_string(self.expression_result_type(expr.object))
        if object_type:
            member_type = self.struct_member_types.get(object_type, {}).get(member_name)
            if member_type is not None and self.is_sampler_type(member_type):
                return object_type, member_name

        matching_structs = [
            struct_name
            for struct_name, members in self.struct_member_types.items()
            if member_name in members and self.is_sampler_type(members[member_name])
        ]
        if len(matching_structs) == 1:
            return matching_structs[0], member_name
        return None

    def collect_global_texture_names(self, root):
        texture_names = set()
        for node in self.global_resource_declaration_nodes(root):
            var_type = getattr(node, "var_type", getattr(node, "vtype", "float"))
            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            mapped_type = self.map_resource_type_with_format(var_type, node)
            if var_name and mapped_type.startswith("Texture"):
                texture_names.add(var_name)
        return texture_names

    def collect_global_texture_types(self, root):
        texture_types = {}
        for node in self.global_resource_declaration_nodes(root):
            var_type = getattr(node, "var_type", getattr(node, "vtype", "float"))
            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            mapped_type = self.map_resource_type_with_format(var_type, node)
            if var_name and (
                mapped_type.startswith("Texture")
                or self.is_hlsl_rw_texture_type(mapped_type)
            ):
                texture_types[var_name] = mapped_type
        return texture_types

    def collect_global_sampler_names(self, root):
        sampler_names = set()
        for node in self.global_resource_declaration_nodes(root):
            var_type = getattr(node, "var_type", getattr(node, "vtype", "float"))
            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            if var_name and self.is_sampler_type(var_type):
                sampler_names.add(var_name)
        return sampler_names

    def collect_global_mixed_sampler_names(self, root):
        global_sampler_names = self.collect_global_sampler_names(root)
        comparison_sampler_names = self.collect_explicit_sampler_resource_names(
            root, self.comparison_texture_function_names()
        ) | self.collect_comparison_sampler_arguments(
            root, self.comparison_sampler_parameters
        )
        regular_sampler_names = self.collect_explicit_sampler_resource_names(
            root, self.regular_texture_function_names()
        ) | self.collect_regular_sampler_arguments(
            root, self.regular_sampler_parameters
        )
        return (comparison_sampler_names & regular_sampler_names) & global_sampler_names

    def collect_global_resource_names(self, root):
        resource_names = set()
        for node in self.global_resource_declaration_nodes(root):
            var_type = getattr(node, "var_type", getattr(node, "vtype", "float"))
            var_name = getattr(node, "name", getattr(node, "variable_name", None))
            if var_name and self.is_resource_parameter_type(var_type):
                resource_names.add(var_name)
        return resource_names

    def global_resource_declaration_nodes(self, root, target_stage=None):
        current_nodes = getattr(self, "current_global_resource_declaration_nodes", None)
        if current_nodes is not None and target_stage is None:
            return current_nodes

        specialization_constant_ids = getattr(
            self, "directx_specialization_constant_global_ids", set()
        )
        global_vars = [
            node
            for node in getattr(root, "global_variables", []) or []
            if id(node) not in specialization_constant_ids
            and not getattr(node, "is_type_alias", False)
        ]
        stage_resource_vars = collect_stage_local_variables(
            root, target_stage, self.is_stage_local_resource_variable
        )
        entry_resource_params = self.hlsl_stage_entry_resource_parameter_globals(
            root,
            target_stage,
            global_vars + stage_resource_vars,
        )
        return deduplicate_named_declarations(
            global_vars + stage_resource_vars + entry_resource_params,
            "DirectX resource",
        )

    def is_stage_local_resource_variable(self, node):
        vtype = self.type_name_string(
            getattr(node, "var_type", getattr(node, "vtype", "float"))
        )
        qualifiers = {str(value).lower() for value in getattr(node, "qualifiers", [])}
        return (
            bool(qualifiers & {"uniform", "buffer"})
            or self.is_resource_parameter_type(vtype)
            or self.is_hlsl_readonly_buffer_type(vtype)
            or self.is_hlsl_uav_buffer_type(vtype)
            or self.is_hlsl_constant_buffer_type(vtype)
            or self.is_glsl_buffer_block_variable(node, vtype)
        )

    def hlsl_stage_entry_resource_parameter_globals(
        self, root, target_stage=None, base_declarations=None
    ):
        globals_ = []
        declarations_by_name = {}
        used_names = set()
        for node in base_declarations or []:
            name = getattr(node, "name", getattr(node, "variable_name", None))
            if not name:
                continue
            declarations_by_name.setdefault(
                name,
                (
                    declaration_signature(node),
                    self.hlsl_bfloat16_storage_resource_source_type(node),
                ),
            )
            used_names.add(name)

        entries = collect_stage_entry_records(
            root, target_stage, self.stage_entry_types()
        )
        for _, _stage_name, func in entries:
            for parameter in getattr(func, "parameters", getattr(func, "params", [])):
                proxy = self.hlsl_entry_resource_parameter_global(parameter, func)
                if proxy is not None:
                    original_name = getattr(parameter, "name", None)
                    emitted_name = getattr(proxy, "name", None)
                    signature = (
                        declaration_signature(proxy),
                        self.hlsl_bfloat16_storage_resource_source_type(proxy),
                    )
                    existing_signature = declarations_by_name.get(emitted_name)
                    if (
                        emitted_name
                        and existing_signature is not None
                        and existing_signature != signature
                    ):
                        emitted_name = self.hlsl_scoped_entry_resource_name(
                            func,
                            original_name,
                            used_names,
                        )
                        proxy.name = emitted_name
                    if emitted_name:
                        self.hlsl_promoted_entry_resource_parameter_names[
                            id(parameter)
                        ] = emitted_name
                        declarations_by_name.setdefault(emitted_name, signature)
                        used_names.add(emitted_name)
                    globals_.append(proxy)
        return globals_

    def hlsl_stage_entry_constant_cbuffers(
        self, root, target_stage=None, cbuffers=None, global_vars=None
    ):
        cbuffers_ = []
        used_names = set(self.collect_hlsl_program_identifier_names(root))
        used_names.update(
            getattr(cbuffer, "name", None)
            for cbuffer in cbuffers or []
            if getattr(cbuffer, "name", None)
        )
        used_names.update(
            getattr(node, "name", getattr(node, "variable_name", None))
            for node in global_vars or []
            if getattr(node, "name", getattr(node, "variable_name", None))
        )

        entries = collect_stage_entry_records(
            root, target_stage, self.stage_entry_types()
        )
        for _, _stage_name, func in entries:
            function_name = sanitize_type_name(getattr(func, "name", None) or "entry")
            for parameter in getattr(func, "parameters", getattr(func, "params", [])):
                member_type = self.hlsl_constant_parameter_member_type(parameter)
                if member_type is None:
                    continue

                parameter_name = getattr(parameter, "name", None) or "constant"
                parameter_name = sanitize_type_name(parameter_name)
                base_name = "_".join(
                    part for part in (function_name, parameter_name) if part
                )
                if not base_name:
                    base_name = "entry_constant"
                if base_name[0].isdigit():
                    base_name = f"entry_{base_name}"

                buffer_name = self.hlsl_unique_local_identifier(
                    f"{base_name}_Constants", used_names
                )
                used_names.add(buffer_name)
                member_name = self.hlsl_unique_local_identifier(base_name, used_names)
                used_names.add(member_name)

                cbuffer = StructNode(
                    name=buffer_name,
                    members=[
                        StructMemberNode(
                            member_name,
                            NamedType(member_type),
                        )
                    ],
                    attributes=deepcopy(
                        list(getattr(parameter, "attributes", []) or [])
                    ),
                )
                cbuffer.is_cbuffer = True
                if self.hlsl_half_storage.shape(member_type):
                    cbuffer.members[0].add_annotation("directx.half_storage", True)
                    self.hlsl_half_storage.constant_types[member_name] = member_type
                    self.hlsl_half_storage.resources[buffer_name] = {
                        "logicalElementType": "float16",
                        "encoding": "ieee754-binary16",
                    }
                cbuffer.add_annotation("directx.relocatable_binding", True)
                cbuffer.add_annotation(
                    "directx.binding_provenance",
                    self.directx_stage_entry_parameter_binding_provenance(
                        parameter, func
                    ),
                )
                self.hlsl_promoted_entry_constant_members[id(parameter)] = member_name
                cbuffers_.append(cbuffer)
        return cbuffers_

    def hlsl_constant_parameter_member_type(self, parameter):
        raw_type = self.hlsl_parameter_raw_type(parameter)
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "qualifiers", []) or []
        }
        if "constant" not in qualifiers:
            return None

        element_type = self.hlsl_constant_reference_element_type(raw_type)
        if element_type is None:
            return None
        if self.is_hlsl_bfloat16_type(element_type):
            return self.hlsl_bfloat16_scalar_type_name(element_type)

        mapped_type = self.map_type(element_type)
        if self.is_scalar_value_type(element_type) or self.is_vector_value_type(
            element_type
        ):
            return mapped_type
        return hlsl_constant_matrix_member_type(
            self, element_type, getattr(parameter, "name", None) or "constant"
        )

    def hlsl_constant_reference_element_type(self, vtype):
        if isinstance(vtype, ReferenceType):
            return self.type_name_string(vtype.referenced_type)

        type_name = self.type_name_string(vtype)
        if not type_name:
            return None
        type_name = str(type_name).strip()
        if not type_name.endswith("&"):
            return None

        type_name = type_name[:-1].strip()
        if type_name.startswith("metal::"):
            type_name = type_name.split("metal::", 1)[1].strip()
        metal_qualifiers = {
            "const",
            "constant",
            "device",
            "thread",
            "threadgroup",
            "volatile",
            "restrict",
        }
        parts = type_name.split()
        while parts and parts[0].lower() in metal_qualifiers:
            parts.pop(0)
        return " ".join(parts) if parts else None

    def hlsl_scoped_entry_resource_name(self, func, parameter_name, used_names):
        func_name = sanitize_type_name(getattr(func, "name", None) or "entry")
        parameter_name = sanitize_type_name(parameter_name or "resource")
        base_name = "_".join(part for part in (func_name, parameter_name) if part)
        if not base_name:
            base_name = "entry_resource"
        if base_name[0].isdigit():
            base_name = f"entry_{base_name}"

        candidate = base_name
        suffix = 2
        while (
            candidate in used_names
            or candidate in self.HLSL_RESERVED_LOCAL_IDENTIFIER_NAMES
        ):
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        return candidate

    def apply_hlsl_promoted_entry_resource_parameter_aliases(
        self,
        param_list,
        promoted_parameter_ids,
        func,
        *,
        mutable_resource_offsets=None,
        parameter_prologue_statements=None,
    ):
        mutable_resource_offsets = set(mutable_resource_offsets or ())
        if parameter_prologue_statements is None:
            parameter_prologue_statements = []
        for parameter in param_list:
            if id(parameter) not in promoted_parameter_ids:
                continue
            parameter_name = getattr(parameter, "name", None)
            if not parameter_name:
                continue
            constant_member_name = self.hlsl_promoted_entry_constant_members.get(
                id(parameter)
            )
            if constant_member_name is not None:
                self.local_variable_types[parameter_name] = self.type_name_string(
                    self.hlsl_parameter_raw_type(parameter)
                )
                self.current_identifier_aliases[parameter_name] = constant_member_name
                continue
            emitted_name = self.hlsl_promoted_entry_resource_parameter_names.get(
                id(parameter),
                parameter_name,
            )
            promoted_type = self.global_variable_types.get(emitted_name)
            if promoted_type is None:
                promoted_type = self.hlsl_entry_resource_parameter_global_type(
                    parameter,
                    func,
                )
            reference_type = self.hlsl_stage_entry_metal_buffer_reference_element_type(
                parameter
            )
            if reference_type is not None:
                self.local_variable_types[parameter_name] = reference_type
                self.local_variable_source_types[parameter_name] = reference_type
                emitted_identifier = self.hlsl_identifier_name(emitted_name)
                self.current_identifier_aliases[parameter_name] = (
                    f"{emitted_identifier}[0]"
                )
                continue
            if promoted_type:
                self.local_variable_types[parameter_name] = self.type_name_string(
                    promoted_type
                )
                self.local_variable_source_types[parameter_name] = (
                    self.type_name_string(self.hlsl_parameter_raw_type(parameter))
                )
            if emitted_name != parameter_name:
                self.current_identifier_aliases[parameter_name] = emitted_name
            if parameter_name not in mutable_resource_offsets:
                continue

            resource_type = self.directx_resource_declaration_type(promoted_type)
            resource_name = self.hlsl_resource_type_name(resource_type)
            if resource_name not in {
                "Buffer",
                "StructuredBuffer",
                "RWBuffer",
                "RWStructuredBuffer",
                "RasterizerOrderedBuffer",
                "RasterizerOrderedStructuredBuffer",
            }:
                continue
            element_type = self.hlsl_resource_pointer_element_type(resource_type)
            if element_type is None:
                continue

            used_names = set(self.current_identifier_reserved_names)
            used_names.update(self.local_variable_types)
            offset_name = self.hlsl_unique_local_identifier(
                f"{parameter_name}_offset",
                used_names,
            )
            self.current_identifier_reserved_names.add(offset_name)
            self.local_variable_types[offset_name] = "int64_t"
            access = (
                "read_write"
                if resource_name.startswith(("RW", "RasterizerOrdered"))
                else "read"
            )
            pointer_contract = self.hlsl_resource_pointer_parameter_contract(
                parameter, self.hlsl_function_declaration_name(func)
            )
            # A narrow source element keeps its logical type when helpers
            # advance the root's offset, as the unadvanced root does.
            logical_element_type = self.hlsl_storage_root_element_type(
                parameter_name, resource_type
            )
            if pointer_contract is None and logical_element_type is not None:
                pointer_contract = {
                    "address_space": "storage",
                    "access": access,
                    "element_type": logical_element_type,
                    "resource_type": resource_type,
                }
            if pointer_contract is not None:
                entry_binding = self.hlsl_resource_pointer_parameter_alias_binding(
                    pointer_contract,
                    root=self.hlsl_identifier_name(emitted_name),
                    offset=offset_name,
                    logical_storage=True,
                )
            else:
                entry_binding = {
                    "root": self.hlsl_identifier_name(emitted_name),
                    "offset": offset_name,
                    "element_type": element_type,
                    "source_element_type": element_type,
                    "resource_type": resource_type,
                    "address_space": "storage",
                    "access": access,
                }
            entry_binding.update(
                {
                    "kind": "resource-pointer",
                    "root_kind": "entry-resource",
                    "is_pointer_alias": True,
                }
            )
            self.current_hlsl_resource_pointer_aliases[parameter_name] = entry_binding
            parameter_prologue_statements.append(f"int64_t {offset_name} = int64_t(0);")

    def hlsl_stage_entry_metal_buffer_reference_element_type(self, parameter):
        if (
            self.explicit_resource_binding_index(
                parameter,
                {"binding", "buffer", "uav"},
                ("t", "u"),
            )
            is None
        ):
            return None

        raw_type = self.hlsl_parameter_raw_type(parameter)
        if not isinstance(raw_type, ReferenceType):
            return None
        address_space = str(getattr(raw_type, "address_space", None) or "").lower()
        if not address_space:
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(parameter, "qualifiers", []) or []
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

        element_type = self.type_name_string(raw_type.referenced_type)
        if (
            not element_type
            or element_type in self.structs_by_name
            or not (
                self.is_scalar_value_type(element_type)
                or self.is_vector_value_type(element_type)
            )
        ):
            return None
        return element_type

    def hlsl_entry_resource_parameter_global(self, parameter, func=None):
        mapped_type = self.hlsl_entry_resource_parameter_global_type(parameter, func)
        if mapped_type is None:
            return None

        annotations = dict(getattr(parameter, "annotations", {}) or {})
        proxy = VariableNode(
            name=parameter.name,
            var_type=mapped_type,
            attributes=list(getattr(parameter, "attributes", []) or []),
            qualifiers=list(getattr(parameter, "qualifiers", []) or []),
            annotations=annotations,
        )
        raw_type = self.hlsl_parameter_raw_type(parameter)
        source_element_type = self.hlsl_stage_entry_metal_buffer_reference_element_type(
            parameter
        )
        if source_element_type is None:
            source_element_type = self.hlsl_typed_buffer_element_type(
                raw_type,
                {"Buffer", "RWBuffer", "StructuredBuffer", "RWStructuredBuffer"},
            )
        if source_element_type is None:
            source_element_type = self.hlsl_pointer_element_type(raw_type)
        if source_element_type is None:
            source_element_type = self.hlsl_array_element_type_name(raw_type)
        source_bfloat_type = self.hlsl_bfloat16_scalar_type_name(source_element_type)
        if self.hlsl_half_storage.lowered(source_element_type):
            proxy.add_annotation("directx.storage_source_type", source_element_type)
        if source_bfloat_type is not None:
            proxy.add_annotation(
                "directx.bfloat16_storage_source_type", source_bfloat_type
            )
            self.hlsl_bfloat16_storage_resource_source_types[id(proxy)] = (
                source_bfloat_type
            )
        if self.directx_stage_entry_parameter_binding_allows_register_relocation(
            parameter
        ):
            proxy.add_annotation("directx.relocatable_binding", True)
            proxy.add_annotation(
                "directx.binding_provenance",
                self.directx_stage_entry_parameter_binding_provenance(parameter, func),
            )
        return proxy

    def directx_stage_entry_parameter_binding_allows_register_relocation(
        self, parameter
    ):
        attr_names = {
            str(getattr(attr, "name", "")).lower()
            for attr in getattr(parameter, "attributes", []) or []
        }
        return bool(
            "register" not in attr_names
            and attr_names & {"binding", "buffer", "sampler", "texture", "uav"}
        )

    def directx_stage_entry_parameter_binding_provenance(self, parameter, func):
        function_name = getattr(func, "name", None) or "<entry>"
        parameter_name = getattr(parameter, "name", None) or "<anonymous>"
        attribute_label = self.directx_resource_binding_attribute_label(parameter)
        if attribute_label:
            return (
                f"stage-entry parameter {function_name}.{parameter_name} "
                f"from {attribute_label}"
            )
        return f"stage-entry parameter {function_name}.{parameter_name}"

    def hlsl_entry_resource_parameter_global_type(self, parameter, func=None):
        reference_type = self.hlsl_stage_entry_metal_buffer_reference_element_type(
            parameter
        )
        if reference_type is not None:
            raw_type = self.hlsl_parameter_raw_type(parameter)
            qualifiers = {
                str(qualifier).lower()
                for qualifier in getattr(parameter, "qualifiers", []) or []
            }
            read_only = not getattr(raw_type, "is_mutable", True) or bool(
                qualifiers.intersection({"const", "constant", "read", "readonly"})
            )
            resource_name = "StructuredBuffer" if read_only else "RWStructuredBuffer"
            storage_type = self.hlsl_structured_storage_type(
                reference_type,
                operation=(
                    "stage-entry buffer reference "
                    f"'{getattr(parameter, 'name', '<anonymous>')}' storage"
                ),
                source_location=getattr(parameter, "source_location", None),
            )
            return f"{resource_name}<{storage_type}>"

        storage_array_type = self.hlsl_storage_array_parameter_resource_type(parameter)
        if storage_array_type is not None:
            return storage_array_type

        raw_type = self.hlsl_parameter_raw_type(parameter)
        mapped_type = self.map_resource_parameter_type_with_hint(
            raw_type,
            parameter,
            getattr(func, "name", None),
        )
        mapped_type = self.directx_resource_declaration_type(mapped_type)
        if self.is_hlsl_global_resource_type(mapped_type):
            return mapped_type
        return None

    def hlsl_storage_array_parameter_resource_type(self, parameter):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(parameter, "resource_qualifiers", []) or []
        }
        if "storage" not in qualifiers:
            return None

        raw_type = self.hlsl_parameter_raw_type(parameter)
        element_type = None
        if (
            hasattr(raw_type, "element_type")
            and str(type(raw_type)).find("ArrayType") != -1
        ):
            element_type = raw_type.element_type
        else:
            type_name = self.type_name_string(raw_type)
            if not type_name:
                return None
            base_type, array_suffix = split_array_type_suffix(type_name)
            if array_suffix:
                element_type = base_type
            else:
                base_name, generic_args = generic_type_parts(type_name)
                if base_name == "array" and generic_args:
                    element_type = generic_args[0]

        if element_type is None:
            return None

        resource_name = (
            "StructuredBuffer"
            if qualifiers.intersection({"const", "read", "readonly", "read_only"})
            else "RWStructuredBuffer"
        )
        storage_type = self.hlsl_structured_storage_type(
            element_type,
            operation=(
                f"storage parameter '{getattr(parameter, 'name', '<anonymous>')}' element"
            ),
            source_location=getattr(parameter, "source_location", None),
        )
        return f"{resource_name}<{storage_type}>"

    def is_hlsl_global_resource_type(self, mapped_type):
        mapped_type = str(mapped_type)
        # Read array suffixes from the end. A repeated suffix expression
        # re-segments a long malformed tail before it can reject the name.
        suffix_end = len(mapped_type)
        while suffix_end > 0 and mapped_type[suffix_end - 1] == "]":
            open_bracket = mapped_type.rfind("[", 0, suffix_end)
            if open_bracket < 0:
                break
            inner = mapped_type[open_bracket + 1 : suffix_end - 1]
            if re.fullmatch(r"\s*(?:\w+\s*)?", inner) is None:
                break
            suffix_end = open_bracket
            while suffix_end > 0 and mapped_type[suffix_end - 1].isspace():
                suffix_end -= 1
        sampler_type = mapped_type[:suffix_end]
        return (
            mapped_type.startswith("Texture")
            or self.is_hlsl_feedback_texture_type(mapped_type)
            or self.is_hlsl_rw_texture_type(mapped_type)
            or self.is_multisample_storage_image_resource_type(mapped_type)
            or self.is_hlsl_acceleration_structure_type(mapped_type)
            or self.is_hlsl_uav_buffer_type(mapped_type)
            or self.is_hlsl_constant_buffer_type(mapped_type)
            or self.is_hlsl_texture_buffer_type(mapped_type)
            or self.is_hlsl_readonly_buffer_type(mapped_type)
            or sampler_type in ["SamplerState", "SamplerComparisonState"]
        )

    def hlsl_requires_rwtexture_cube_alias(self, variables):
        for node in variables or []:
            vtype = self.type_name_string(
                getattr(node, "var_type", getattr(node, "vtype", "float"))
            )
            if self.resource_base_type(vtype) == "imageCube":
                return True
            mapped_type = self.map_resource_type_with_format(vtype, node)
            if str(mapped_type).startswith("RWTextureCube<"):
                return True
        return False

    def validate_global_resource_shadows(self, ast):
        return

    def validate_explicit_sampler_role_conflicts(self, ast):
        global_sampler_names = self.collect_global_sampler_names(ast)
        comparison_sampler_names = self.collect_explicit_sampler_resource_names(
            ast, self.comparison_texture_function_names()
        ) | self.collect_comparison_sampler_arguments(
            ast, self.comparison_sampler_parameters
        )
        regular_sampler_names = self.collect_explicit_sampler_resource_names(
            ast, self.regular_texture_function_names()
        ) | self.collect_regular_sampler_arguments(ast, self.regular_sampler_parameters)

        conflicts = [
            name
            for name in sorted(
                (comparison_sampler_names & regular_sampler_names)
                & global_sampler_names
            )
            if name not in self.global_mixed_sampler_names
        ]

        function_names = set(self.comparison_sampler_parameters) | set(
            self.regular_sampler_parameters
        )
        for func_name in sorted(function_names):
            for param_name in sorted(
                self.comparison_sampler_parameters.get(func_name, set())
                & self.regular_sampler_parameters.get(func_name, set())
            ):
                conflicts.append(f"{func_name}.{param_name}")

        for struct_name, member_name in sorted(
            self.comparison_sampler_struct_members & self.regular_sampler_struct_members
        ):
            conflicts.append(f"{struct_name}.{member_name}")

        if not conflicts:
            return

        names = ", ".join(conflicts)
        raise DirectXRefusal(
            "DirectX sampler(s) used for both regular sampling and shadow "
            f"comparison: {names}"
        )

    def is_projected_texture_function(self, func_name):
        return is_projected_texture_operation(func_name)

    def is_projected_texture_compare_function(self, func_name):
        return is_projected_texture_compare_operation(func_name)

    def projected_texture_call_is_diagnostic_only(self, func_name, texture_type):
        if not self.is_projected_texture_function(func_name):
            return False
        texture_type = self.sampled_texture_shape_type(texture_type)
        if texture_type == "TextureCubeArray":
            return True
        return texture_type == "TextureCube" and (
            is_projected_texture_basic_offset_operation(func_name)
            or is_projected_texture_lod_offset_operation(func_name)
            or is_projected_texture_grad_offset_operation(func_name)
        )

    def projected_texture_compare_call_is_diagnostic_only(
        self, func_name, texture_type
    ):
        if not self.is_projected_texture_compare_function(func_name):
            return False
        texture_type = self.sampled_texture_shape_type(texture_type)
        if texture_type == "TextureCubeArray":
            return True
        return texture_type == "TextureCube" and (
            is_texture_compare_offset_operation(func_name)
            or is_texture_compare_lod_offset_operation(func_name)
            or is_texture_compare_grad_offset_operation(func_name)
        )

    def diagnostic_texture_compare_sampler_parameter_is_comparison(
        self, func_name, texture_type
    ):
        if not self.is_projected_texture_compare_function(func_name):
            return False
        texture_type = self.sampled_texture_shape_type(texture_type)
        return texture_type in {"TextureCube", "TextureCubeArray"}

    def texture_gather_compare_offset_call_is_diagnostic_only(
        self, func_name, texture_type
    ):
        if (
            not is_texture_gather_compare_offset_operation(func_name)
            or texture_type is None
        ):
            return False
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return not self.texture_gather_compare_offset_supported(texture_type)

    def texture_gather_compare_call_is_diagnostic_only(self, func_name, texture_type):
        if not is_texture_gather_compare_operation(func_name) or texture_type is None:
            return False
        if self.texture_gather_compare_offset_call_is_diagnostic_only(
            func_name, texture_type
        ):
            return True
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return not self.texture_gather_supported(texture_type)

    def texture_gather_call_is_diagnostic_only(self, func_name, texture_type):
        if func_name != "textureGather" or texture_type is None:
            return False
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return not self.texture_gather_supported(texture_type)

    def texture_gather_offset_call_is_diagnostic_only(self, func_name, texture_type):
        if not is_texture_gather_offset_operation(func_name):
            return False
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return not self.texture_gather_offset_supported(texture_type)

    def texture_sample_offset_call_is_diagnostic_only(self, func_name, texture_type):
        if not is_texture_sample_offset_operation(func_name):
            return False
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return not self.texture_sample_offset_supported(texture_type)

    def texture_compare_offset_call_is_diagnostic_only(self, func_name, texture_type):
        if not is_texture_compare_non_projected_offset_operation(func_name):
            return False
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return not self.texture_compare_offset_supported(texture_type)

    def is_multisample_texture_resource_type(self, texture_type):
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return texture_type.startswith(("Texture2DMS<", "Texture2DMSArray<"))

    def is_multisample_storage_image_resource_type(self, texture_type):
        texture_type = self.resource_base_type(
            self.map_resource_type_with_format(texture_type)
        )
        return texture_type.startswith(("RWTexture2DMS<", "RWTexture2DMSArray<"))

    def directx_multisample_storage_image_srv_type(self, texture_type):
        """Return a legal HLSL SRV type for unsupported multisample images."""
        if texture_type is None:
            return None
        texture_text = str(texture_type)
        if "[" in texture_text and "]" in texture_text:
            base_type, array_suffix = split_array_type_suffix(texture_text)
        else:
            base_type, array_suffix = texture_text, ""
        base_type = self.resource_base_type(base_type)
        if "<" not in base_type or ">" not in base_type:
            return None
        component_type = base_type.split("<", 1)[1].split(">", 1)[0].strip()
        if base_type.startswith("RWTexture2DMSArray<"):
            return f"Texture2DMSArray<{component_type}>{array_suffix}"
        if base_type.startswith("RWTexture2DMS<"):
            return f"Texture2DMS<{component_type}>{array_suffix}"
        return None

    def directx_resource_declaration_type(self, texture_type):
        return (
            self.directx_multisample_storage_image_srv_type(texture_type)
            or texture_type
        )

    def multisample_texture_call_is_diagnostic_only(self, func_name, texture_type):
        if func_name not in TEXTURE_IMPLICIT_SAMPLER_INTRINSIC_NAMES:
            return False
        return self.is_multisample_texture_resource_type(texture_type)

    def texture_call_is_diagnostic_only(self, func_name, texture_type):
        if texture_type is None:
            return False
        if self.storage_image_texture_operation_expression(func_name, texture_type):
            return True
        if is_texture_query_lod_operation(
            func_name
        ) and self.is_storage_image_resource_type(texture_type):
            return True
        if self.projected_texture_call_is_diagnostic_only(func_name, texture_type):
            return True
        if self.projected_texture_compare_call_is_diagnostic_only(
            func_name, texture_type
        ):
            return True
        if self.texture_gather_compare_call_is_diagnostic_only(func_name, texture_type):
            return True
        if self.texture_gather_call_is_diagnostic_only(func_name, texture_type):
            return True
        if self.texture_gather_offset_call_is_diagnostic_only(func_name, texture_type):
            return True
        if self.texture_sample_offset_call_is_diagnostic_only(func_name, texture_type):
            return True
        if self.texture_compare_offset_call_is_diagnostic_only(func_name, texture_type):
            return True
        return self.multisample_texture_call_is_diagnostic_only(func_name, texture_type)

    def texture_argument_analysis_type(self, texture_arg, global_resource_types=None):
        texture_name = self.expression_name(texture_arg)
        if texture_name and global_resource_types:
            texture_type = global_resource_types.get(texture_name)
            if texture_type is not None:
                return texture_type

        arg_type = self.expression_result_type(texture_arg)
        if arg_type is not None and self.is_texture_or_image_type(arg_type):
            return self.map_resource_type_with_format(self.resource_base_type(arg_type))
        return None
