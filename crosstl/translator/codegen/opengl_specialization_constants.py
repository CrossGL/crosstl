"""Specialization constants in the GLSL generator: private pointer metadata of resource specializations, constant names and ids, cbuffers, stage entry constants and member semantics."""

import re
from copy import deepcopy

from ...glsl_builtins import GLSL_BUILTIN_INT_LIMITS
from ..ast import (
    ASTNode,
    AttributeNode,
    FunctionCallNode,
    IfNode,
    LiteralNode,
    PrimitiveType,
    ReferenceType,
    StructMemberNode,
    StructNode,
    UnaryOpNode,
)
from ..refusals import OpenGLRefusal
from ..resource_identity import RESOURCE_IDENTITY_ATTRIBUTE
from ..validation import expression_debug_name
from .array_utils import (
    evaluate_literal_int_expression,
    format_c_style_array_declaration,
)
from .enum_utils import sanitize_type_name
from .glsl_specialization_dependents import runtime_specialization_dependents
from .glsl_specialization_macros import glsl_specialization_macro_name
from .opengl_values import (
    GLSL_SPIRV_GUARD,
    OpenGLSpecializationConstantError,
    OpenGLStoragePointerError,
    OpenGLWorkgroupPointerError,
)
from .resource_arrays import is_private_pointer_parameter
from .stage_utils import (
    assign_stage_entry_names,
    collect_stage_entry_records,
    collect_stage_entry_reserved_function_names,
    collect_stage_local_cbuffers,
    collect_stage_local_structs,
    collect_stage_local_variables,
    function_stage_name,
)


class OpenGLSpecializationConstantMixin:
    """Specialization constant members of ``GLSLCodeGen``."""

    def register_glsl_resource_specialization_private_pointer_metadata(
        self,
        source_func,
        clone,
        target_name,
    ):
        source_key = self.glsl_function_declaration_name(source_func)
        self.function_private_pointer_local_arrays[target_name] = (
            self.glsl_private_pointer_local_arrays(clone)
        )
        source_parameters = self.function_private_pointer_parameters.get(source_key)
        if not source_parameters:
            return

        clone_parameters = list(
            getattr(clone, "parameters", getattr(clone, "params", [])) or []
        )
        clone_private_parameters = [
            parameter
            for parameter in clone_parameters
            if is_private_pointer_parameter(parameter)
        ]
        source_positions = {
            parameter.name: index for index, parameter in enumerate(source_parameters)
        }
        clone_parameter_names = {
            parameter.name for parameter in clone_private_parameters
        }
        if not clone_private_parameters or not clone_parameter_names.issubset(
            source_positions
        ):
            return

        def copy_parameter_map(metadata):
            return {
                name: value
                for name, value in metadata.get(source_key, {}).items()
                if name in clone_parameter_names
            }

        self.function_private_pointer_parameters[target_name] = clone_private_parameters
        self.function_private_pointer_parameter_indices[target_name] = {
            index: parameter.name
            for index, parameter in enumerate(clone_parameters)
            if is_private_pointer_parameter(parameter)
        }
        self.function_private_pointer_element_types[target_name] = copy_parameter_map(
            self.function_private_pointer_element_types
        )
        self.function_private_pointer_base_names[target_name] = copy_parameter_map(
            self.function_private_pointer_base_names
        )
        self.function_private_pointer_source_names[target_name] = (
            self.function_private_pointer_source_names.get(
                source_key,
                getattr(source_func, "name", target_name),
            )
        )
        self.function_private_pointer_read_only_parameters[target_name] = {
            name
            for name in self.function_private_pointer_read_only_parameters.get(
                source_key, set()
            )
            if name in clone_parameter_names
        }
        for recovery_key, recovery_contract in tuple(
            self.glsl_private_pointer_robust_zero_read_variants.items()
        ):
            function_key, parameter_name, extent = recovery_key
            if (
                function_key != source_key
                or parameter_name not in clone_parameter_names
            ):
                continue
            self.glsl_private_pointer_robust_zero_read_variants[
                (target_name, parameter_name, extent)
            ] = dict(recovery_contract)
        self.function_private_pointer_full_span_parameters[target_name] = {
            name
            for name in self.function_private_pointer_full_span_parameters.get(
                source_key, set()
            )
            if name in clone_parameter_names
        }
        self.function_private_pointer_scalar_access_violations[target_name] = {
            name
            for name in self.function_private_pointer_scalar_access_violations.get(
                source_key, set()
            )
            if name in clone_parameter_names
        }

        required_spans = copy_parameter_map(
            self.function_private_pointer_required_spans
        )
        self.function_private_pointer_required_spans[target_name] = required_spans
        self.glsl_collect_private_pointer_static_direct_spans(
            clone,
            target_name,
            required_spans,
        )

        source_variants = self.function_private_pointer_backing_variants.get(
            source_key, ()
        )
        if source_variants:
            self.function_private_pointer_backing_variants[target_name] = tuple(
                sorted(
                    {
                        tuple(
                            variant[source_positions[parameter.name]]
                            for parameter in clone_private_parameters
                        )
                        for variant in source_variants
                    },
                    key=self.glsl_private_pointer_variant_sort_key,
                )
            )
        self.function_private_pointer_array_size_hints[target_name] = (
            copy_parameter_map(self.function_private_pointer_array_size_hints)
        )

    def glsl_resource_specialization_name(self, func_name, bindings):
        suffix_parts = []
        for _, (param_name, binding) in sorted(bindings.items()):
            if binding.get("kind") in {
                "storage-pointer-unused",
                "workgroup-pointer-unused",
            }:
                suffix_parts.append(
                    "{}_unused_{}".format(
                        sanitize_type_name(param_name),
                        sanitize_type_name(binding["element_type"]),
                    )
                )
                continue
            if binding.get("kind") == "workgroup-pointer":
                suffix_parts.append(
                    "{}_{}_{}_{}".format(
                        sanitize_type_name(param_name),
                        sanitize_type_name(binding["root"]),
                        sanitize_type_name(binding["element_type"]),
                        binding["array_size"],
                    )
                )
                continue
            if binding.get("kind") == "storage-pointer-array":
                access_mode = binding.get("access") or "read"
                mutation = "mutable" if binding.get("offset_direction") else "fixed"
                suffix_parts.append(
                    "{}_{}_{}_{}_{}_{}".format(
                        sanitize_type_name(param_name),
                        sanitize_type_name(binding["root"]),
                        sanitize_type_name(binding["element_type"]),
                        binding["extent"],
                        sanitize_type_name(access_mode),
                        mutation,
                    )
                )
                continue
            if binding.get("kind") == "storage-pointer":
                view_key = self.glsl_storage_pointer_specialization_view_key(binding)
                view_suffix = ""
                if view_key[0] == "reinterpret":
                    view_suffix = "_view_{}_to_{}".format(
                        sanitize_type_name(view_key[1][0]),
                        sanitize_type_name(view_key[2][0]),
                    )
                elif view_key[0] == "byte-offset":
                    view_suffix = "_byte_offset"
                suffix_parts.append(
                    "{}_{}_{}{}".format(
                        sanitize_type_name(param_name),
                        sanitize_type_name(binding["root"]),
                        sanitize_type_name(binding["element_type"]),
                        view_suffix,
                    )
                )
                continue
            if binding.get("kind") == "fixed-array-storage":
                access_mode = binding["required_access"]
                if binding.get("access") not in {None, access_mode}:
                    access_mode = f"{access_mode}_{binding['access']}"
                suffix_parts.append(
                    "{}_{}_{}_{}_{}".format(
                        sanitize_type_name(param_name),
                        sanitize_type_name(binding["root"]),
                        sanitize_type_name(binding["element_type"]),
                        binding["fixed_extent"],
                        sanitize_type_name(access_mode),
                    )
                )
                continue
            suffix_parts.append(
                "{}_{}".format(
                    sanitize_type_name(param_name),
                    sanitize_type_name(binding["expression"]),
                )
            )
        suffix = "_".join(suffix_parts)
        return f"{sanitize_type_name(func_name)}__glsl_{suffix}"

    def glsl_reachable_ast_paths(self, value):
        """Yield stable structural paths for nodes that GLSL can emit.

        Compile-time ``if`` branches are pruned exactly as ``generate_if``
        prunes them. Transient analysis attributes and legacy type metadata do
        not participate in paths, so an equivalent deep copy or reconstructed
        function receives the same locations.
        """

        visited = set()
        excluded_fields = {
            "annotations",
            "parent",
            "source_location",
            "expression_type",
            "vtype",
            "semantic",
        }

        def visit(current, path):
            if current is None or isinstance(current, (str, int, float, bool)):
                return
            if isinstance(current, dict):
                for key in sorted(current, key=lambda item: repr(item)):
                    yield from visit(
                        current[key],
                        path + (("key", repr(key)),),
                    )
                return
            if isinstance(current, (list, tuple)):
                for index, child in enumerate(current):
                    yield from visit(child, path + (("index", index),))
                return
            if isinstance(current, set):
                for index, child in enumerate(sorted(current, key=repr)):
                    yield from visit(child, path + (("set-index", index),))
                return
            if not isinstance(current, ASTNode):
                return

            node_id = id(current)
            if node_id in visited:
                return
            visited.add(node_id)
            yield path, current

            if isinstance(current, IfNode):
                static_known, static_body = self.glsl_static_if_body(current)
                if static_known:
                    if static_body is None:
                        return
                    branches = [
                        current.if_body,
                        *(getattr(current, "else_if_bodies", []) or []),
                        getattr(current, "else_body", None),
                    ]
                    branch_index = next(
                        (
                            index
                            for index, branch in enumerate(branches)
                            if branch is static_body
                        ),
                        len(branches),
                    )
                    yield from visit(
                        static_body,
                        path + (("static-if-branch", branch_index),),
                    )
                    return

            for field, child in vars(current).items():
                if field in excluded_fields or field.startswith("_glsl_"):
                    continue
                yield from visit(child, path + (("field", field),))

        yield from visit(value, ())

    def walk_glsl_reachable_ast(self, value):
        """Yield nodes from branches that can contribute emitted GLSL."""

        for _path, node in self.glsl_reachable_ast_paths(value):
            yield node

    def glsl_resource_function_semantic_caller_identity(self, function):
        specialization_key = getattr(
            function,
            "_glsl_resource_specialization_key",
            None,
        )
        if specialization_key is not None:
            return ("resource-specialization", specialization_key)
        return (
            "function",
            getattr(function, "_generic_source_name", None)
            or getattr(function, "name", None),
            self.glsl_source_function_signature(function),
        )

    def glsl_resource_function_call_semantic_locations(self, function):
        """Index reachable calls by caller identity and structural path."""

        caller_identity = self.glsl_resource_function_semantic_caller_identity(function)
        return {
            id(node): (caller_identity, path)
            for path, node in self.glsl_reachable_ast_paths(
                getattr(function, "body", [])
            )
            if isinstance(node, FunctionCallNode)
        }

    @staticmethod
    def glsl_resource_function_call_semantic_identity(
        location,
        func_name,
        args,
    ):
        if location is None:
            return None
        argument_identity = tuple(
            (type(argument).__name__, expression_debug_name(argument))
            for argument in args or []
        )
        return (
            location,
            func_name,
            len(args or []),
            argument_identity,
        )

    def record_glsl_prepared_resource_function_call_plan(
        self,
        func_name,
        args,
        call_node,
        plan,
        *,
        semantic_call_locations=None,
    ):
        """Record one exact proved plan under transient and stable identities."""

        self.glsl_resource_function_call_plans[id(call_node)] = plan
        # Keep the fast path on the exact call and a stable structural path for
        # equivalent deep copies or reconstructed final-emission functions.
        call_node._glsl_resource_specialization_plan = plan
        location = (semantic_call_locations or {}).get(id(call_node))
        identity = self.glsl_resource_function_call_semantic_identity(
            location,
            func_name,
            args,
        )
        if identity is None:
            return
        self.glsl_resource_function_semantic_call_plans.setdefault(
            identity,
            set(),
        ).add(plan)

    def glsl_prepared_resource_function_call_specialization(
        self,
        func_name,
        args,
        call_node,
        *,
        caller_function=None,
    ):
        """Return an exact specialization proved for this prepared call node."""

        if call_node is None:
            return None
        plan = getattr(call_node, "_glsl_resource_specialization_plan", None)
        if plan is None:
            plan = self.glsl_resource_function_call_plans.get(id(call_node))
        if plan is None:
            semantic_locations = getattr(
                self,
                "current_glsl_resource_function_call_semantic_locations",
                {},
            )
            location = semantic_locations.get(id(call_node))
            if location is None and caller_function is not None:
                location = self.glsl_resource_function_call_semantic_locations(
                    caller_function
                ).get(id(call_node))
            identity = self.glsl_resource_function_call_semantic_identity(
                location,
                func_name,
                args,
            )
            semantic_plans = self.glsl_resource_function_semantic_call_plans.get(
                identity,
                set(),
            )
            # Multiple exact proof variants at one semantic location are not
            # interchangeable. Let ordinary validation resolve or reject them.
            if len(semantic_plans) == 1:
                plan = next(iter(semantic_plans))
        if not isinstance(plan, tuple) or len(plan) != 3:
            return None
        planned_name, planned_argument_count, key = plan
        if planned_name != func_name or planned_argument_count != len(args or []):
            return None
        specialization = self.glsl_resource_function_specializations.get(key)
        if specialization is None:
            return None
        if getattr(specialization, "_glsl_resource_specialization_key", None) != key:
            return None
        return specialization

    @staticmethod
    def glsl_workgroup_specialization_key_contract(key):
        if not isinstance(key, tuple) or len(key) != 3:
            return key, None
        key_parts = []
        intervals = None
        for part in key[2]:
            if (
                isinstance(part, tuple)
                and len(part) == 2
                and part[0] == "workgroup-proof-intervals"
            ):
                intervals = dict(part[1])
            else:
                key_parts.append(part)
        return (key[0], key[1], tuple(key_parts)), intervals

    @staticmethod
    def glsl_workgroup_proof_intervals_cover(candidate, requested):
        if candidate is None or requested is None:
            return False
        for name, candidate_interval in candidate.items():
            requested_interval = requested.get(name)
            if requested_interval is None:
                return False
            if (
                requested_interval[0] < candidate_interval[0]
                or requested_interval[1] > candidate_interval[1]
            ):
                return False
        return True

    def glsl_resource_function_call_specialization(
        self,
        func_name,
        args,
        *,
        lexical_source_types=None,
        call_node=None,
        caller_function=None,
    ):
        prepared = self.glsl_prepared_resource_function_call_specialization(
            func_name,
            args,
            call_node,
            caller_function=caller_function,
        )
        if prepared is not None:
            return prepared
        with self.glsl_lexical_source_type_scope(lexical_source_types):
            key, _ = self.glsl_resource_function_specialization_key(
                func_name,
                args,
                self.glsl_storage_pointer_aliases(
                    self.glsl_workgroup_pointer_aliases(self.current_resource_aliases)
                ),
                lexical_source_types=lexical_source_types,
            )
        if key is None:
            return None
        specialization = self.glsl_resource_function_specializations.get(key)
        if specialization is not None:
            return specialization

        requested_base, requested_intervals = (
            self.glsl_workgroup_specialization_key_contract(key)
        )
        if requested_intervals is None:
            return None
        compatible = []
        for (
            candidate_key,
            candidate,
        ) in self.glsl_resource_function_specializations.items():
            candidate_base, candidate_intervals = (
                self.glsl_workgroup_specialization_key_contract(candidate_key)
            )
            if candidate_base != requested_base:
                continue
            if self.glsl_workgroup_proof_intervals_cover(
                candidate_intervals,
                requested_intervals,
            ):
                compatible.append((candidate_intervals, candidate))
        if not compatible:
            return None
        compatible.sort(
            key=lambda item: (
                -len(item[0]),
                sum(upper - lower for lower, upper in item[0].values()),
                getattr(item[1], "name", ""),
            )
        )
        return compatible[0][1]

    def glsl_resource_specialized_call_arguments(self, specialized_func, args):
        bound_indices = getattr(specialized_func, "_glsl_resource_bound_indices", set())
        call_args = [
            arg for index, arg in enumerate(args or []) if index not in bound_indices
        ]
        call_args.extend(
            getattr(specialized_func, "_glsl_resource_dynamic_call_arguments", []) or []
        )
        aliases = self.glsl_storage_pointer_aliases(
            self.glsl_workgroup_pointer_aliases(self.current_resource_aliases)
        )
        for index in getattr(
            specialized_func, "_glsl_workgroup_pointer_bound_indices", ()
        ):
            if index >= len(args or []):
                continue
            binding = self.glsl_workgroup_pointer_binding(args[index], aliases)
            if binding is None:
                raise OpenGLWorkgroupPointerError(
                    "OpenGL workgroup pointer call argument has no concrete "
                    "shared-array backing object",
                    function_name=getattr(
                        specialized_func, "_glsl_resource_source_name", None
                    ),
                    reason="call-backing-unresolved",
                    source_location=getattr(args[index], "source_location", None),
                )
            offset = self.glsl_workgroup_pointer_offset_expression(binding)
            call_args.append(f"int({offset})")
        for index in getattr(
            specialized_func, "_glsl_storage_pointer_bound_indices", ()
        ):
            if index >= len(args or []):
                continue
            binding = self.glsl_storage_pointer_binding(args[index], aliases)
            if binding is None:
                raise OpenGLStoragePointerError(
                    "OpenGL storage pointer call argument has no concrete "
                    "storage-buffer backing object",
                    function_name=getattr(
                        specialized_func, "_glsl_resource_source_name", None
                    ),
                    reason="call-backing-unresolved",
                    source_location=getattr(args[index], "source_location", None),
                )
            offset = self.glsl_workgroup_pointer_offset_expression(binding)
            offset_direction = getattr(
                specialized_func,
                "_glsl_storage_pointer_offset_directions",
                {},
            ).get(index)
            if offset_direction is not None:
                if not self.glsl_storage_pointer_offset_lvalue(offset):
                    parameter_name = getattr(
                        specialized_func,
                        "_glsl_storage_pointer_parameter_names",
                        {},
                    ).get(index)
                    function_name = getattr(
                        specialized_func, "_glsl_resource_source_name", None
                    )
                    raise OpenGLStoragePointerError(
                        "OpenGL mutable storage pointer argument for "
                        f"'{function_name}.{parameter_name}' has no assignable "
                        "logical offset",
                        function_name=function_name,
                        parameter_name=parameter_name,
                        reason="mutable-offset-lvalue-unresolved",
                        source_location=getattr(args[index], "source_location", None),
                    )
                call_args.append(offset)
            else:
                call_args.append(f"int({offset})")
            if index in getattr(
                specialized_func,
                "_glsl_storage_pointer_byte_offset_indices",
                (),
            ):
                byte_offset = str(binding.get("byte_offset", "0"))
                call_args.append(f"int({byte_offset})")
        for index in getattr(
            specialized_func, "_glsl_storage_pointer_array_bound_indices", ()
        ):
            if index >= len(args or []):
                continue
            binding = self.glsl_storage_pointer_array_binding(args[index], aliases)
            parameter_name = getattr(
                specialized_func,
                "_glsl_storage_pointer_array_parameter_names",
                {},
            ).get(index)
            function_name = getattr(
                specialized_func, "_glsl_resource_source_name", None
            )
            if binding is None:
                raise OpenGLStoragePointerError(
                    "OpenGL storage pointer-array call argument has no concrete "
                    "same-backing storage-buffer view",
                    function_name=function_name,
                    parameter_name=parameter_name,
                    reason="pointer-array-call-backing-unresolved",
                    source_location=getattr(args[index], "source_location", None),
                )
            expected_binding = getattr(
                specialized_func,
                "_glsl_storage_pointer_aliases",
                {},
            ).get(parameter_name, {})
            if binding.get("extent") != expected_binding.get("extent"):
                raise OpenGLStoragePointerError(
                    "OpenGL storage pointer-array call extent does not match "
                    f"'{function_name}.{parameter_name}'",
                    function_name=function_name,
                    parameter_name=parameter_name,
                    reason="pointer-array-call-extent-mismatch",
                    source_location=getattr(args[index], "source_location", None),
                )
            offsets = list(binding.get("offset_expressions") or [])
            if len(offsets) != binding.get("extent"):
                raise OpenGLStoragePointerError(
                    "OpenGL storage pointer-array call has incomplete logical "
                    "offset materialization",
                    function_name=function_name,
                    parameter_name=parameter_name,
                    reason="pointer-array-call-offsets-incomplete",
                    source_location=getattr(args[index], "source_location", None),
                )
            offset_direction = getattr(
                specialized_func,
                "_glsl_storage_pointer_array_offset_directions",
                {},
            ).get(index)
            for offset in offsets:
                offset = self.glsl_workgroup_pointer_offset_expression(
                    {**binding, "offset": offset}
                )
                if offset_direction is not None:
                    if not self.glsl_storage_pointer_offset_lvalue(offset):
                        raise OpenGLStoragePointerError(
                            "OpenGL mutable storage pointer-array argument for "
                            f"'{function_name}.{parameter_name}' has a "
                            "non-assignable logical offset element",
                            function_name=function_name,
                            parameter_name=parameter_name,
                            reason="pointer-array-mutable-offset-lvalue-unresolved",
                            source_location=getattr(
                                args[index], "source_location", None
                            ),
                        )
                    call_args.append(offset)
                else:
                    call_args.append(f"int({offset})")
        return call_args

    def generate_constants(self, ast, constants=None):
        code = ""
        for node in (
            getattr(ast, "constants", []) or [] if constants is None else constants
        ):
            code += self.generate_glsl_constant_declaration(node)

        return f"{code}\n" if code else ""

    def generate_glsl_global_specialization_constants(self):
        code = "".join(
            self.generate_glsl_constant_declaration(node)
            for node in self.glsl_global_specialization_constants
        )
        return f"{code}\n" if code else ""

    def generate_glsl_constant_declaration(self, node):
        name = self.glsl_specialization_constant_name(node)
        if not name:
            return ""

        const_type = self.glsl_specialization_constant_type(node)
        mapped_type = self.map_type(const_type)
        declaration = format_c_style_array_declaration(mapped_type, name)
        constant_id = self.glsl_specialization_constant_id(node)
        value = self.glsl_specialization_constant_value(node)
        builtin_declaration = self.glsl_builtin_specialization_constant_declaration(
            node,
            name=name,
            const_type=const_type,
            constant_id=constant_id,
            value=value,
        )
        if builtin_declaration is not None:
            return builtin_declaration
        if constant_id is None:
            value_code = self.generate_constant_expression(value, const_type)
            if id(node) in self.glsl_runtime_specialization_dependent_ids:
                return (
                    f"{GLSL_SPIRV_GUARD}{declaration} = {value_code};\n"
                    f"#else\nconst {declaration} = {value_code};\n#endif\n"
                )
            return f"const {declaration} = {value_code};\n"

        self.validate_glsl_specialization_constant_type(node, const_type)
        layout = f"layout(constant_id = {constant_id}) "
        if value is None:
            value_code = self.zero_value_expression(const_type)
            macro = glsl_specialization_macro_name(name)
            metadata = (
                "/* CrossGL specialization metadata: "
                f"name={name}; constant_id={constant_id}; "
                "required_override=true; "
                f"encoding_default={value_code}; source_macro={macro} */\n"
            )
            # A driver compiling the source has no specialization, so the host supplies the value the way OpenGL
            # programs select variants: as a macro defined after #version, like MTLFunctionConstantValues for MSL.
            fallback = (
                f"#elif defined({macro})\n"
                f"const {declaration} = {mapped_type}({macro});\n"
                "#else\n"
                f'#error "specialization constant {name} has no default: define {macro} '
                'or compile this shader to SPIR-V and specialize it"\n'
            )
        else:
            value_code = self.generate_constant_expression(value, const_type)
            metadata = ""
            fallback = f"#else\nconst {declaration} = {value_code};\n"
        # constant_id exists only when the shader is compiled to SPIR-V (GLSL 4.60 section 4.4.5,
        # GL_ARB_gl_spirv, GL_KHR_vulkan_glsl); a driver compiling the source keeps the default value.
        return (
            f"{GLSL_SPIRV_GUARD}{metadata}{layout}const {declaration} = {value_code};\n"
            f"{fallback}#endif\n"
        )

    def glsl_builtin_specialization_constant_declaration(
        self,
        node,
        *,
        name,
        const_type,
        constant_id,
        value,
    ):
        default = GLSL_BUILTIN_INT_LIMITS.get(name)
        if default is None or constant_id is None:
            return None
        self.validate_glsl_specialization_constant_type(node, const_type)
        if self.map_type(const_type) != "int":
            raise OpenGLSpecializationConstantError(
                f"OpenGL built-in specialization constant '{name}' must use int",
                constant_name=name,
                specialization_id=constant_id,
                constant_type=self.type_name_string(const_type),
                reason="builtin-type-mismatch",
                source_location=self.glsl_specialization_constant_source_location(node),
            )
        if value is not None and evaluate_literal_int_expression(value) != default:
            raise OpenGLSpecializationConstantError(
                f"OpenGL built-in specialization constant '{name}' must retain "
                f"its implicit default {default}",
                constant_name=name,
                specialization_id=constant_id,
                constant_type=self.type_name_string(const_type),
                reason="builtin-default-mismatch",
                source_location=self.glsl_specialization_constant_source_location(node),
            )
        return (
            f"{GLSL_SPIRV_GUARD}layout(constant_id = {constant_id}) {name};\n#endif\n"
        )

    def glsl_constant_layout_prefix(self, node):
        constant_id = self.glsl_specialization_constant_id(node)
        if constant_id is None:
            return ""
        return f"layout(constant_id = {constant_id}) "

    def glsl_specialization_constant_attributes(self, node):
        attributes = []
        for attr in getattr(node, "attributes", []) or []:
            attr_name = getattr(attr, "name", None)
            if not attr_name:
                continue
            normalized = str(attr_name).lower().replace("-", "_")
            for prefix in ("glsl_", "metal_", "msl_"):
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix) :]
                    break
            if normalized in {"constant_id", "function_constant"}:
                attributes.append(attr)
        return attributes

    def glsl_specialization_constant_name(self, node):
        name = getattr(node, "name", getattr(node, "variable_name", None))
        if name in GLSL_BUILTIN_INT_LIMITS:
            return name
        if not self.glsl_module_used_identifier_names:
            return name
        return self.glsl_global_identifier_name(name)

    def glsl_specialization_constant_type(self, node):
        return getattr(
            node,
            "const_type",
            getattr(node, "var_type", getattr(node, "vtype", "float")),
        )

    def glsl_specialization_constant_value(self, node):
        if hasattr(node, "const_type"):
            return getattr(node, "value", None)
        return getattr(node, "initial_value", getattr(node, "value", None))

    def glsl_specialization_constant_source_location(self, node, attr=None):
        return getattr(attr, "source_location", None) or getattr(
            node, "source_location", None
        )

    def glsl_specialization_constant_integer_id(self, value):
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, UnaryOpNode) and getattr(value, "op", None) in {"+", "-"}:
            operand = self.glsl_specialization_constant_integer_id(value.operand)
            if operand is None:
                return None
            return operand if value.op == "+" else -operand

        value_text = self.attribute_value_to_string(value)
        if value_text is None:
            return None
        try:
            return int(str(value_text).strip(), 0)
        except ValueError:
            return None

    def glsl_specialization_constant_id(self, node):
        constant_id = None
        name = self.glsl_specialization_constant_name(node) or "<unnamed>"
        for attr in self.glsl_specialization_constant_attributes(node):
            arguments = getattr(attr, "arguments", getattr(attr, "args", [])) or []
            if len(arguments) != 1:
                raise OpenGLSpecializationConstantError(
                    f"OpenGL specialization constant '{name}' requires one "
                    "non-negative integer id",
                    constant_name=name,
                    reason="invalid-id",
                    source_location=self.glsl_specialization_constant_source_location(
                        node, attr
                    ),
                )
            value = self.glsl_specialization_constant_integer_id(arguments[0])
            if value is None:
                source = self.attribute_value_to_string(arguments[0])
                raise OpenGLSpecializationConstantError(
                    f"OpenGL specialization constant '{name}' requires a concrete "
                    f"integer id, got {source or '<missing>'}",
                    constant_name=name,
                    reason="invalid-id",
                    source_location=self.glsl_specialization_constant_source_location(
                        node, attr
                    ),
                )
            if value < 0:
                raise OpenGLSpecializationConstantError(
                    f"OpenGL specialization constant '{name}' requires a "
                    f"non-negative id, got {value}",
                    constant_name=name,
                    specialization_id=value,
                    reason="negative-id",
                    source_location=self.glsl_specialization_constant_source_location(
                        node, attr
                    ),
                )
            if value > 0x7FFFFFFF:
                raise OpenGLSpecializationConstantError(
                    f"OpenGL specialization constant '{name}' id {value} exceeds "
                    "the signed 32-bit GLSL constant_id range",
                    constant_name=name,
                    specialization_id=value,
                    reason="id-out-of-range",
                    source_location=self.glsl_specialization_constant_source_location(
                        node, attr
                    ),
                )
            if constant_id is not None and constant_id != value:
                raise OpenGLSpecializationConstantError(
                    "Conflicting OpenGL specialization constant ids for "
                    f"'{name}': {constant_id} differs from {value}",
                    constant_name=name,
                    specialization_id=value,
                    reason="conflicting-id",
                    source_location=self.glsl_specialization_constant_source_location(
                        node, attr
                    ),
                )
            constant_id = value
        return constant_id

    def validate_glsl_specialization_constant_type(self, node, const_type):
        name = self.glsl_specialization_constant_name(node) or "<unnamed>"
        source_type = self.type_name_string(const_type)
        mapped_type = self.map_type(const_type)
        normalized_source = str(source_type or "").lower().replace(" ", "")
        scalar_types = {
            "bool",
            "int",
            "uint",
            "float",
            "double",
            "int64_t",
            "uint64_t",
        }
        invalid_atomic = "atomic<" in normalized_source or normalized_source.startswith(
            "atomic_"
        )
        if (
            mapped_type in scalar_types
            and not isinstance(const_type, ReferenceType)
            and not invalid_atomic
        ):
            return

        raise OpenGLSpecializationConstantError(
            f"OpenGL specialization constant '{name}' cannot use non-scalar "
            f"type '{source_type}'",
            constant_name=name,
            specialization_id=self.glsl_specialization_constant_id(node),
            constant_type=source_type,
            reason="unsupported-type",
            source_location=self.glsl_specialization_constant_source_location(node),
        )

    def prepare_glsl_specialization_constants(self, ast):
        constants = list(getattr(ast, "constants", []) or [])
        globals_ = list(getattr(ast, "global_variables", []) or [])
        declarations = [
            node
            for node in constants + globals_
            if self.glsl_specialization_constant_attributes(node)
        ]
        used_ids = {}
        used_names = {}
        for node in declarations:
            constant_id = self.glsl_specialization_constant_id(node)
            const_type = self.glsl_specialization_constant_type(node)
            self.validate_glsl_specialization_constant_type(node, const_type)
            name = self.glsl_specialization_constant_name(node) or "<unnamed>"
            previous = used_names.get(name)
            if previous is not None and previous is not node:
                previous_id = self.glsl_specialization_constant_id(previous)
                raise OpenGLSpecializationConstantError(
                    f"Duplicate OpenGL specialization constant name '{name}' "
                    f"for ids {previous_id} and {constant_id}",
                    constant_name=name,
                    specialization_id=constant_id,
                    constant_type=self.type_name_string(const_type),
                    conflicting_name=name,
                    conflicting_id=previous_id,
                    reason="duplicate-name",
                    source_location=self.glsl_specialization_constant_source_location(
                        node
                    ),
                )
            previous = used_ids.get(constant_id)
            if previous is not None and previous is not node:
                previous_name = (
                    self.glsl_specialization_constant_name(previous) or "<unnamed>"
                )
                raise OpenGLSpecializationConstantError(
                    f"Duplicate OpenGL specialization constant id {constant_id} "
                    f"for '{previous_name}' and '{name}'",
                    constant_name=name,
                    specialization_id=constant_id,
                    constant_type=self.type_name_string(const_type),
                    conflicting_name=previous_name,
                    reason="duplicate-id",
                    source_location=self.glsl_specialization_constant_source_location(
                        node
                    ),
                )
            used_ids[constant_id] = node
            used_names[name] = node

        global_ids = {id(node) for node in globals_}
        self.glsl_global_specialization_constants = [
            node for node in declarations if id(node) in global_ids
        ]
        self.glsl_specialization_constant_declarations = declarations

    def prepare_glsl_runtime_specialization_dependents(self, ast, version_line):
        declarations = self.glsl_specialization_constant_declarations
        declaration_ids = {id(node) for node in declarations}
        self.glsl_runtime_specialization_dependent_ids = (
            runtime_specialization_dependents(
                [
                    node
                    for node in getattr(ast, "constants", []) or []
                    if id(node) not in declaration_ids
                ],
                declarations,
                lambda node: self.map_type(
                    self.glsl_specialization_constant_type(node)
                ),
                self.map_type,
            )
        )
        if not self.glsl_runtime_specialization_dependent_ids or not re.search(
            r"\bes\b", version_line
        ):
            return ""
        # GLSL ES 3.20 section 4.3 requires constant global initializers; the extension lifts that.
        return (
            f"{GLSL_SPIRV_GUARD}"
            "#extension GL_EXT_shader_non_constant_global_initializers : enable\n"
            "#endif\n"
        )

    def generate_constant_expression(self, expr, expected_type=None):
        previous = self.glsl_generating_global_initializer
        self.glsl_generating_global_initializer = True
        try:
            value_code = self.generate_expression_with_expected(expr, expected_type)
        finally:
            self.glsl_generating_global_initializer = previous
        if value_code == "True":
            return "true"
        if value_code == "False":
            return "false"
        return value_code

    def combined_stage_entry_types(self):
        return {
            "vertex",
            "fragment",
            "compute",
            "geometry",
            "tessellation_control",
            "tessellation_evaluation",
            "mesh",
            "task",
            "amplification",
            "object",
            "ray_generation",
            "ray_intersection",
            "ray_closest_hit",
            "ray_any_hit",
            "ray_miss",
            "ray_callable",
        }

    def combined_stage_entry_names(self, ast, target_stage=None):
        """Return distinct helper names for monolithic multi-stage output."""
        if target_stage is not None:
            return {}

        stage_entry_types = self.combined_stage_entry_types()
        entries = collect_stage_entry_records(ast, None, stage_entry_types)
        used_names = collect_stage_entry_reserved_function_names(
            ast, None, stage_entry_types
        )

        def target_names(assigned_names):
            return {
                entry_id: (
                    name
                    if name == "main"
                    else self.glsl_generated_module_identifier(
                        ("stage-wrapper", entry_id), name
                    )
                )
                for entry_id, name in assigned_names.items()
            }

        if (
            self.should_emit_stage_guards(ast, target_stage)
            and "main" not in used_names
        ):
            # Each stage's guarded section declares its entry as main. A stage has one entry here:
            # generate_program refuses a second one (refuse_repeated_stage_entries).
            return {entry_id: "main" for entry_id, _stage_name, _func in entries}

        return target_names(
            assign_stage_entry_names(
                entries,
                used_names,
                lambda stage_name, _func: f"{stage_name}_main",
                single_entry_default="main",
            )
        )

    def validate_stage_main_helper_conflict(self, ast, target_stage):
        if target_stage is None:
            return

        stage_entry_types = self.combined_stage_entry_types()
        has_global_main_helper = False
        for func in getattr(ast, "functions", []) or []:
            stage_name = function_stage_name(func)
            if stage_name in stage_entry_types:
                continue
            if getattr(func, "name", None) == "main":
                has_global_main_helper = True
                break

        if not has_global_main_helper:
            return

        if collect_stage_entry_records(ast, target_stage, stage_entry_types):
            raise OpenGLRefusal(
                "Global helper function 'main' conflicts with the GLSL "
                f"{target_stage} stage entry point"
            )

    def glsl_cbuffer_nodes(self, ast, target_stage=None):
        return (
            list(getattr(ast, "cbuffers", []) or [])
            + collect_stage_local_cbuffers(ast, target_stage)
            + self.collect_stage_entry_constant_value_parameter_cbuffers(
                ast, target_stage
            )
        )

    def collect_stage_entry_constant_value_parameter_cbuffers(
        self, ast, target_stage=None
    ):
        cbuffers = []
        existing_cbuffers = list(getattr(ast, "cbuffers", []) or [])
        existing_cbuffers += collect_stage_local_cbuffers(ast, target_stage)
        used_block_names = self.glsl_existing_cbuffer_block_names(
            ast, target_stage, existing_cbuffers
        )
        used_member_names = self.glsl_existing_cbuffer_member_names(existing_cbuffers)
        used_member_names.update(
            self.glsl_existing_cbuffer_global_member_names(ast, target_stage)
        )
        self.stage_entry_constant_value_parameter_aliases = {}
        self.stage_entry_constant_value_cbuffer_names = set()
        stage_entry_types = self.combined_stage_entry_types()
        stage_entry_records = list(
            collect_stage_entry_records(ast, target_stage, stage_entry_types)
        )
        parameter_member_name_counts = (
            self.collect_stage_entry_constant_value_parameter_member_name_counts(
                stage_entry_records
            )
        )
        for _entry_id, stage_name, func in stage_entry_records:
            for parameter in getattr(func, "parameters", getattr(func, "params", [])):
                explicit_value_type = self.stage_entry_constant_value_parameter_type(
                    parameter
                )
                compute_value_type = (
                    self.stage_entry_compute_value_parameter_type(parameter)
                    if explicit_value_type is None and stage_name == "compute"
                    else None
                )
                member_type = explicit_value_type or compute_value_type
                if member_type is None:
                    continue
                binding = self.explicit_resource_binding_index(parameter)
                parameter_name = getattr(parameter, "name", None)
                if not parameter_name:
                    continue
                attributes = [
                    deepcopy(attr)
                    for attr in getattr(parameter, "attributes", []) or []
                    if getattr(attr, "name", None) == RESOURCE_IDENTITY_ATTRIBUTE
                ]
                if binding is not None:
                    attributes.append(
                        AttributeNode(
                            "binding",
                            [LiteralNode(binding, PrimitiveType("int"))],
                        )
                    )
                block_name = self.unique_stage_entry_constant_value_block_name(
                    self.stage_entry_constant_value_parameter_block_name(
                        func, parameter_name
                    ),
                    used_block_names,
                )
                used_block_names.add(block_name)
                member_name = self.stage_entry_constant_value_parameter_member_name(
                    block_name,
                    parameter_name,
                    used_member_names,
                    parameter_member_name_counts,
                    force_block_qualified=compute_value_type is not None,
                )
                used_member_names.add(member_name)
                if member_name != parameter_name:
                    self.stage_entry_constant_value_parameter_aliases.setdefault(
                        id(func), {}
                    )[parameter_name] = member_name
                member = StructMemberNode(member_name, member_type)
                member.add_annotation("opengl.source_name", parameter_name)
                cbuffer = StructNode(
                    block_name,
                    [member],
                    attributes=attributes,
                )
                cbuffer.is_cbuffer = True
                cbuffer.is_stage_entry_constant_value_cbuffer = True
                self.stage_entry_constant_value_cbuffer_names.add(block_name)
                cbuffers.append(cbuffer)
        return cbuffers

    def stage_entry_constant_value_parameter_member_type(self, parameter, stage_name):
        member_type = self.stage_entry_constant_value_parameter_type(parameter)
        if member_type is None and stage_name == "compute":
            return self.stage_entry_compute_value_parameter_type(parameter)
        return member_type

    def collect_stage_entry_constant_value_parameter_member_name_counts(
        self, stage_entry_records
    ):
        counts = {}
        for _entry_id, stage_name, func in stage_entry_records:
            for parameter in getattr(func, "parameters", getattr(func, "params", [])):
                if (
                    self.stage_entry_constant_value_parameter_member_type(
                        parameter, stage_name
                    )
                    is None
                ):
                    continue
                parameter_name = getattr(parameter, "name", None)
                if not parameter_name:
                    continue
                member_name = sanitize_type_name(parameter_name)
                if not member_name:
                    continue
                counts[member_name] = counts.get(member_name, 0) + 1
        return counts

    def glsl_existing_cbuffer_block_names(self, ast, target_stage, cbuffers):
        names = {
            getattr(cbuffer, "name", None)
            for cbuffer in cbuffers
            if getattr(cbuffer, "name", None)
        }
        for node in (
            list(getattr(ast, "structs", []) or [])
            + collect_stage_local_structs(ast, target_stage)
            + list(getattr(ast, "global_variables", []) or [])
            + list(getattr(ast, "constants", []) or [])
        ):
            name = getattr(node, "name", None)
            if name:
                names.add(name)
        return names

    def glsl_existing_cbuffer_member_names(self, cbuffers):
        names = set()
        for cbuffer in cbuffers:
            for member in getattr(cbuffer, "members", []) or []:
                name = getattr(member, "name", None)
                if name:
                    names.add(name)
        return names

    def glsl_existing_cbuffer_global_member_names(self, ast, target_stage):
        names = set()
        for node in (
            list(getattr(ast, "global_variables", []) or [])
            + list(getattr(ast, "constants", []) or [])
            + collect_stage_local_variables(
                ast,
                target_stage,
                lambda _node: True,
            )
        ):
            name = getattr(node, "name", None)
            if name:
                names.add(name)
        return names

    def unique_stage_entry_constant_value_identifier(
        self, base_name, used_names, fallback
    ):
        candidate_base = self.glsl_sanitized_identifier_base(
            base_name,
            fallback=fallback or "EntryConstant",
        )
        return self.glsl_unique_identifier(candidate_base, used_names)
