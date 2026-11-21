"""Variables, memory spaces, kernel launches and the argument, unique_ptr, cooperative-group and type-alias scopes in the HIP to CrossGL converter."""

import re

from crosstl.translator.refusals import HipSourceRefusal

from ...CUDA.array_extents import integral_constant_declaration_value
from ...CUDA.local_memory_spaces import crossgl_zero_value
from ...device_globals import device_memory_declaration, is_device_memory
from ...namespace_constants import is_c_family_namespace_constant
from ..HipAst import (
    CastNode,
    DesignatedInitializerNode,
    FunctionCallNode,
    InitializerListNode,
    UnaryOpNode,
)


class HipDeclarationMixin:
    """Declaration members of ``HipToCrossGLConverter``."""

    def visit_VariableNode(self, node):
        raw_name = node.name
        raw_type = self.function_local_materialized_variable_type(node)
        record_definition = self.record_value_definition_from_type(raw_type)
        referenced_record_definition = self.record_definition_from_type(raw_type)
        if self.is_nonconst_record_reference_type(
            raw_type, referenced_record_definition
        ):
            if self.record_contains_tracked_union_storage(referenced_record_definition):
                feature = (
                    "union indirect object access through non-const record "
                    "reference alias/writeback"
                )
            else:
                feature = "non-const record reference alias/writeback"
            self.raise_record_semantic(
                (feature, referenced_record_definition),
                f"variable '{raw_name}'",
            )
        self.register_union_object_declaration(raw_name, record_definition, node)
        self.register_union_indirect_object(
            raw_name, raw_type, referenced_record_definition
        )
        if record_definition is None and getattr(node, "value", None) is not None:
            self.validate_record_value_against_target(
                node.value, raw_type, f"variable '{raw_name}'", copy_semantics=False
            )
        self.validate_record_variable_initialization(node, record_definition)
        self.register_union_variable_initializer(
            raw_name, record_definition, getattr(node, "value", None)
        )
        output_name = self.register_variable_declaration_name(raw_name)
        cooperative_group = self.cooperative_group_declaration_metadata(node)
        if cooperative_group is not None:
            group_kind = cooperative_group["kind"]
            self.register_cooperative_group_name(raw_name, cooperative_group)
            if group_kind == "thread_block":
                self.emit(
                    f"// cooperative_groups thread_block {output_name} maps to the current workgroup"
                )
            elif (
                group_kind == "thread_block_tile"
                and cooperative_group.get("tile_size")
                and cooperative_group.get("parent_kind") == "thread_block"
            ):
                self.emit(
                    f"// cooperative_groups thread_block_tile<{cooperative_group['tile_size']}> "
                    f"{output_name} maps to a tiled partition of the current workgroup"
                )
            else:
                self.refuse_import(
                    f"HIP cooperative_groups {group_kind} '{output_name}': CrossGL has no group of this kind"
                )
            return

        var_type = self.convert_hip_variable_type_to_crossgl(raw_type, raw_name)
        qualifiers = set(getattr(node, "qualifiers", []) or [])

        self.register_vector1_name(raw_name, raw_type)
        self.register_packed_argument_list(node)
        self.register_unique_ptr_name(raw_name, raw_type)
        self.register_variable_type(raw_name, var_type)
        if output_name != raw_name:
            self.register_unique_ptr_name(output_name, raw_type)
            self.register_variable_type(output_name, var_type)
        self.integer_constants.declare(
            raw_name, integral_constant_declaration_value(node, self.integer_constants)
        )
        if "__shared__" in qualifiers:
            if getattr(node, "is_dynamic_shared_memory", False):
                self.emit(
                    f"// HIP dynamic shared memory: {output_name} uses launch-time "
                    "shared memory size"
                )
            self.emit(f"var<workgroup> {output_name}: {var_type};")
            return

        if "__constant__" in qualifiers:
            if (
                not (hasattr(node, "value") and node.value)
                and self.function_body_depth > 0
                and "static" in qualifiers
            ):
                # A function-scope static __constant__ local has static
                # storage, so without an initializer every element is zero,
                # and device code cannot write constant memory. Emit it as a
                # function-scope constant of zeros: it must neither take a
                # resource binding nor be read uninitialized by a later
                # backend. A block-scope extern declaration names a
                # namespace-scope variable the host may have written, so it
                # is not one.
                self.emit(
                    f"let {output_name}: {var_type} = "
                    f"{crossgl_zero_value(var_type)};"
                )
                return
            binding = self.global_resource_binding_for(node)
            if hasattr(node, "value") and node.value:
                value = self.format_variable_initializer_value(node.value)
                self.emit(
                    f"@group(0) @binding({binding}) var<uniform> {output_name}: "
                    f"{var_type} = {value};"
                )
            else:
                self.emit(
                    f"@group(0) @binding({binding}) var<uniform> {output_name}: {var_type};"
                )
            return

        if (
            self.namespace_scope_variable
            and is_device_memory(node)
            and not self.is_namespace_scope_constant(node)
        ):
            declaration = device_memory_declaration(
                self,
                node,
                output_name,
                var_type,
                self.global_resource_binding_for(node),
                "HIP",
            )
            if declaration is not None:
                if "__managed__" in qualifiers:
                    self.emit(f"// HIP managed memory: {output_name}")
                self.emit(declaration)
            return

        dim3_default_value = self.format_hip_dim3_default_initializer(node)
        if dim3_default_value is not None:
            self.emit(f"var {output_name}: {var_type} = {dim3_default_value};")
            return

        if hasattr(node, "value") and node.value:
            dim3_brace_value = self.format_hip_dim3_brace_initializer(node)
            if dim3_brace_value is not None:
                self.emit(f"var {output_name}: {var_type} = {dim3_brace_value};")
                return

            runtime_status = self.format_hip_runtime_status_expression(node.value)
            if runtime_status is not None:
                comments, value = runtime_status
                for comment in comments:
                    self.emit(comment)
                self.emit(f"var {output_name}: {var_type} = {value};")
                return

            value = self.format_variable_initializer_value(node.value)
            if self.is_namespace_scope_constant(node):
                self.namespace_constant_names.add(raw_name)
                self.emit(f"const {var_type} {output_name} = {value};")
                return
            self.emit(f"var {output_name}: {var_type} = {value};")
        else:
            self.emit(f"var {output_name}: {var_type};")

    def is_namespace_scope_constant(self, node):
        """Return whether a namespace-scope declaration is a ``const``/``constexpr`` constant.

        Such a constant imports as a CrossGL ``const`` so every target can read it
        from device code; a mutable global would become a host variable.
        """
        return self.namespace_scope_variable and is_c_family_namespace_constant(
            node, self.namespace_constant_names
        )

    def format_variable_initializer_value(self, value):
        string_value = self.format_single_string_initializer(value)
        if string_value is not None:
            return string_value
        scalar_value = self.format_single_scalar_initializer(value)
        if scalar_value is not None:
            return scalar_value
        return self.visit(value)

    def format_single_scalar_initializer(self, value):
        if not isinstance(value, InitializerListNode) or len(value.elements) != 1:
            return None
        initializer = value.elements[0]
        if isinstance(initializer, DesignatedInitializerNode):
            return None
        if not isinstance(initializer, FunctionCallNode):
            return None
        if not isinstance(initializer.name, str) or not initializer.name.startswith(
            "::"
        ):
            return None
        return self.visit(initializer)

    def format_single_string_initializer(self, value):
        if not isinstance(value, InitializerListNode) or len(value.elements) != 1:
            return None

        element = value.elements[0]
        if not isinstance(element, str):
            return None

        return self.format_cpp_string_literal_expression(element)

    def visit_TextureAccessNode(self, node):
        texture_name = self.visit(node.texture_name)
        coordinates = node.coordinates
        if isinstance(coordinates, (list, tuple)):
            rendered_coordinates = [
                self.visit(coordinate) for coordinate in coordinates
            ]
            if len(rendered_coordinates) == 1:
                coordinate = rendered_coordinates[0]
            else:
                coordinate = self.format_vector_constructor(
                    f"vec{len(rendered_coordinates)}", rendered_coordinates
                )
        else:
            coordinate = self.visit(coordinates)
        return f"texture({texture_name}, {coordinate})"

    def visit_SharedMemoryNode(self, node):
        var_type = self.convert_hip_variable_type_to_crossgl(node.vtype, node.name)
        output_name = self.register_identifier_name(node.name)
        self.register_variable_type(node.name, var_type)
        if getattr(node, "is_dynamic_shared_memory", False):
            self.emit(
                f"// HIP dynamic shared memory: {output_name} uses launch-time "
                "shared memory size"
            )
        if node.size is not None:
            size = self.format_crossgl_array_extent(self.visit(node.size))
            self.emit(f"var<workgroup> {output_name}: array<{var_type}, {size}>;")
        else:
            self.emit(f"var<workgroup> {output_name}: {var_type};")

    def visit_ConstantMemoryNode(self, node):
        var_type = self.convert_hip_variable_type_to_crossgl(node.vtype, node.name)
        output_name = self.register_identifier_name(node.name)
        self.register_variable_type(node.name, var_type)
        binding = self.global_resource_binding_for(node)
        if node.value is not None:
            value = self.visit(node.value)
            self.emit(
                f"@group(0) @binding({binding}) var<uniform> {output_name}: "
                f"{var_type} = {value};"
            )
        else:
            self.emit(
                f"@group(0) @binding({binding}) var<uniform> {output_name}: {var_type};"
            )

    def visit_HipErrorHandlingNode(self, node):
        error_type = self.visit(node.error_type)
        error_expr = self.visit(node.error_expr)
        runtime_expression = self.format_hip_runtime_expression_call(
            FunctionCallNode(error_type, [node.error_expr]), [error_expr]
        )
        if runtime_expression is not None:
            return runtime_expression
        return (
            f"(/* HIP error status: {error_type}, expr: {error_expr} */ {error_expr})"
        )

    def visit_HipDevicePropertyNode(self, node):
        property_name = self.visit(node.property_name)
        if node.device_id is None:
            return f"(/* HIP device property: {property_name} */ 0)"
        device_id = self.visit(node.device_id)
        return f"(/* HIP device property: {property_name}, device: {device_id} */ 0)"

    def push_packed_argument_scope(self):
        self.packed_argument_scopes.append({})

    def pop_packed_argument_scope(self):
        if self.packed_argument_scopes:
            self.packed_argument_scopes.pop()

    def push_unique_ptr_scope(self):
        self.unique_ptr_scopes.append(set())

    def pop_unique_ptr_scope(self):
        if len(self.unique_ptr_scopes) > 1:
            self.unique_ptr_scopes.pop()

    def push_cooperative_group_scope(self):
        self.cooperative_group_scopes.append({})

    def pop_cooperative_group_scope(self):
        if len(self.cooperative_group_scopes) > 1:
            self.cooperative_group_scopes.pop()

    def register_cooperative_group_name(self, name, group_metadata):
        if name:
            self.cooperative_group_scopes[-1][name] = group_metadata

    def register_cooperative_group_parameter(self, param):
        if isinstance(param, dict):
            name = param.get("name", "")
            type_name = param.get("type", "")
        else:
            name = getattr(param, "name", "")
            type_name = getattr(param, "vtype", "")

        metadata = self.cooperative_group_metadata_from_type(type_name)
        if metadata is not None:
            self.register_cooperative_group_name(name, metadata)

    def lookup_cooperative_group_metadata(self, name):
        for scope in reversed(self.cooperative_group_scopes):
            if name in scope:
                group_metadata = scope[name]
                if isinstance(group_metadata, str):
                    return {"kind": group_metadata}
                return group_metadata
        return None

    def push_type_alias_scope(self):
        self.type_alias_scopes.append({})
        self.concrete_type_alias_scopes.append({})

    def pop_type_alias_scope(self):
        if len(self.type_alias_scopes) > 1:
            self.type_alias_scopes.pop()
            self.concrete_type_alias_scopes.pop()

    def register_type_alias(self, name, alias_type):
        self.type_alias_scopes[-1][name] = alias_type

    def register_concrete_type_alias(self, name, alias_type):
        self.concrete_type_alias_scopes[-1][name] = alias_type

    def resolve_type_alias(self, type_name):
        type_name = self.strip_type_qualifiers(type_name)
        for scope in reversed(self.type_alias_scopes):
            if type_name in scope:
                return scope[type_name]
        return type_name

    def is_concrete_crossgl_alias_type(self, type_name):
        if not isinstance(type_name, str):
            return False
        type_name = type_name.strip()
        if self.is_crossgl_scalar_type(type_name):
            return True
        if type_name in self.materialized_record_type_names():
            return True
        base_name, template_args = self.parse_cpp_template(type_name)
        if not template_args:
            return False
        concrete_bases = {
            "array",
            "atomic",
            "mat2",
            "mat3",
            "mat4",
            "pair",
            "ptr",
            "tuple",
            "vec2",
            "vec3",
            "vec4",
        }
        return base_name in concrete_bases or base_name.startswith(
            ("sampler", "texture")
        )

    def resolve_concrete_type_alias(self, type_name):
        if not isinstance(type_name, str):
            return type_name
        original = self.strip_type_qualifiers(type_name)
        current = original
        resolving = set()
        while current not in resolving:
            resolving.add(current)
            resolved = current
            for scope in reversed(self.concrete_type_alias_scopes):
                if current in scope:
                    resolved = scope[current]
                    break
            if resolved == current:
                return original
            resolved = str(resolved).strip()
            if self.is_concrete_crossgl_alias_type(resolved):
                return resolved
            current = resolved
        raise HipSourceRefusal(f"recursive HIP type alias: {type_name}")

    def resolve_record_member_type_alias(self, type_name):
        if not isinstance(type_name, str):
            return type_name

        aliases = {}
        for scope in self.type_alias_scopes:
            aliases.update(scope)
        if not aliases:
            return type_name

        patterns = {
            name: re.compile(rf"(?<![A-Za-z0-9_:]){re.escape(name)}(?![A-Za-z0-9_:])")
            for name in aliases
            if name
        }

        def expand(text, resolving):
            for name in sorted(patterns, key=lambda value: (-len(value), value)):
                if name in resolving or not patterns[name].search(text):
                    continue
                replacement = expand(str(aliases[name]), {*resolving, name})
                text = patterns[name].sub(lambda _match: replacement, text)
            return text

        return expand(type_name, set())

    def register_unique_ptr_parameter(self, param):
        if isinstance(param, dict):
            self.register_unique_ptr_name(param.get("name", ""), param.get("type", ""))
        else:
            self.register_unique_ptr_name(
                getattr(param, "name", ""), getattr(param, "vtype", "")
            )

    def register_unique_ptr_name(self, name, type_name):
        if self.is_unique_ptr_type_name(type_name):
            self.unique_ptr_scopes[-1].add(name)

    def is_unique_ptr_expression(self, expr):
        if not isinstance(expr, str):
            return False
        return any(expr in scope for scope in reversed(self.unique_ptr_scopes))

    def register_packed_argument_list(self, node):
        if not self.packed_argument_scopes:
            return
        if self.is_packed_argument_list(node):
            self.packed_argument_scopes[-1][node.name] = (
                self.get_initializer_list_elements(node.value)
            )

    def is_packed_argument_list(self, node):
        if self.get_initializer_list_elements(getattr(node, "value", None)) is None:
            return False

        compact_type = getattr(node, "vtype", "").replace(" ", "")
        return compact_type in {"void*[]", "void**"}

    def get_initializer_list_elements(self, value):
        if isinstance(value, InitializerListNode):
            return value.elements
        if isinstance(value, CastNode) and isinstance(
            value.expression, InitializerListNode
        ):
            return value.expression.elements
        return None

    def resolve_packed_launch_args(self, args):
        if len(args) != 1:
            return args

        compound_elements = self.get_packed_compound_literal_elements(args[0])
        if compound_elements is not None:
            return compound_elements

        packed_arg_name = self.get_packed_argument_name(args[0])
        if packed_arg_name is None:
            return args

        for scope in reversed(self.packed_argument_scopes):
            if packed_arg_name in scope:
                return scope[packed_arg_name]

        return args

    def get_packed_argument_name(self, arg):
        if isinstance(arg, str):
            return arg
        if isinstance(arg, CastNode):
            return self.get_packed_argument_name(arg.expression)
        return None

    def get_packed_compound_literal_elements(self, arg):
        if not isinstance(arg, CastNode):
            return None

        compact_type = arg.target_type.replace(" ", "")
        if compact_type not in {"void*[]", "void**"}:
            return None

        return self.get_initializer_list_elements(arg.expression)

    def format_kernel_launch_arg(self, arg):
        if isinstance(arg, UnaryOpNode) and arg.op == "&":
            return self.visit(arg.operand)
        return self.visit(arg)
