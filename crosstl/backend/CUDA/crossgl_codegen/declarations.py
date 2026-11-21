"""Declarations in the CUDA to CrossGL converter: shaders, preprocessor lines, structs, enums, functions, kernels, variables, scoped name registries, packed launch arguments and shared and constant memory."""

from ...device_globals import device_memory_declaration, is_device_memory
from ...namespace_constants import is_c_family_namespace_constant
from ..array_extents import declare_enumerators, integral_constant_declaration_value
from ..CudaAst import (
    CastNode,
    DesignatedInitializerNode,
    FunctionCallNode,
    InitializerListNode,
    UnaryOpNode,
)
from ..local_memory_spaces import crossgl_zero_value
from ..parameter_defaults import merge_declared_defaults
from ..reference_parameters import ReferenceParameters
from ..union_layout_import import anonymous_union_members


class CudaImportDeclarationMixin:
    """Declaration members of ``CudaToCrossGLConverter``."""

    def visit_ShaderNode(self, node):
        """Render a CUDA shader/program AST as a CrossGL shader block."""
        self.emit("// CUDA to CrossGL conversion")

        if hasattr(node, "includes") and node.includes:
            for include in node.includes:
                self.visit(include)
            self.emit("")

        if hasattr(node, "structs") and node.structs:
            for struct in node.structs:
                if self.is_generated_matrix_helper_struct(struct):
                    continue
                self.visit(struct)
                self.emit("")

        if hasattr(node, "typedefs") and node.typedefs:
            for alias in node.typedefs:
                self.visit(alias)
            self.emit("")

        if hasattr(node, "global_variables") and node.global_variables:
            for var in node.global_variables:
                self.namespace_scope_variable = True
                try:
                    self.visit(var)
                finally:
                    self.namespace_scope_variable = False
                self.emit("")

        self.reference_parameters = ReferenceParameters(node)
        merge_declared_defaults(getattr(node, "functions", None))
        if hasattr(node, "functions") and node.functions:
            for func in node.functions:
                if getattr(func, "body", None) is None:
                    continue
                if self.is_generated_matrix_helper_function(func):
                    continue
                self.emit(f"// Function: {func.name}")
                self.visit(func)
                self.emit("")

        if hasattr(node, "kernels") and node.kernels:
            for kernel in node.kernels:
                if getattr(kernel, "body", None) is None:
                    continue
                self.emit(f"// Kernel: {kernel.name}")
                self.visit_kernel_as_compute_shader(kernel)
                self.emit("")

    def visit_PreprocessorNode(self, node):
        if node.directive == "include":
            if "cuda_runtime.h" in node.content:
                self.emit("// CUDA runtime functionality built-in")
            elif "device_launch_parameters.h" in node.content:
                self.emit("// Device parameters built-in")
            else:
                self.emit(f"// {node.directive} {node.content}")
        else:
            self.emit(f"// {node.directive} {node.content}")

    def cuda_kernel_buffer_element_type(self, cuda_type):
        """The CrossGL element type of the buffer a kernel pointer parameter binds."""
        element_type = self.convert_cuda_pointer_element_type(cuda_type)
        return self.crossgl_safe_generic_type_argument(element_type)

    def visit_StructNode(self, node):
        struct_name = self.convert_cuda_record_name_to_crossgl(node.name)
        layouts = getattr(self, "emitted_buffer_layouts", None)
        if layouts is not None:
            layouts.struct(
                struct_name,
                getattr(node, "attributes", None),
                getattr(node, "is_union", False),
            )
        union = self.cuda_union_layout_record(node, struct_name)
        self.emit(f"struct {struct_name} {{")
        self.indent_level += 1

        for member in node.members:
            member_type = self.convert_cuda_struct_member_type_to_crossgl(
                node.name, member.vtype, member.name
            )
            member_name = self.sanitize_identifier_name(member.name)
            if layouts is not None:
                # CUDA stores a three-component vector member at its scalar alignment; see buffer_layout_import.
                layouts.member(struct_name, member_type, member_name, len(self.output))
            if union is not None:
                union.member(member_type, member_name, len(self.output), member)
            self.emit(f"{member_type} {member_name};")

        self.indent_level -= 1
        self.emit("};")

    def cuda_union_layout_record(self, node, struct_name):
        """The record that marks union ``node`` with its layout contract once the program is rendered.

        Returns ``None`` for a struct, and refuses a struct whose anonymous union members it would write one after
        another; see :mod:`crosstl.backend.CUDA.union_layout_import`.
        """
        contracts = getattr(self, "union_layout_contracts", None)
        if not getattr(node, "is_union", False):
            overlapping = anonymous_union_members(node)
            if overlapping:
                self.refuse_import(
                    f"CUDA struct '{struct_name}' has anonymous union members "
                    f"{', '.join(repr(name) for name in overlapping)}, which share storage that CrossGL does not "
                    "keep in a struct"
                )
            return None
        if contracts is None:
            return None
        return contracts.union(
            struct_name, getattr(node, "attributes", None), len(self.output)
        )

    def visit_EnumNode(self, node):
        # An enum named ``let`` is declared as ``let_``, the name its uses get.
        name = (
            self.sanitize_identifier_name(node.name) or self.next_anonymous_enum_name()
        )
        underlying = getattr(node, "underlying_type", None)
        suffix = (
            f" : {self.convert_cuda_type_to_crossgl(underlying)}" if underlying else ""
        )
        self.emit(f"enum {name}{suffix} {{")
        self.indent_level += 1

        # An enumerator is a constant an array extent can name (``float s[TILE * 2]``).
        declare_enumerators(self.integer_constants, node)
        members = getattr(node, "members", None) or getattr(node, "variants", [])
        for member in members:
            if isinstance(member, tuple):
                member_name, member_value = member
            else:
                member_name = getattr(member, "name", str(member))
                member_value = getattr(member, "value", None)

            if member_value is not None:
                value = self.visit(member_value)
                member_name = self.sanitize_enum_member_name(member_name)
                self.emit(f"{member_name} = {value},")
            else:
                self.emit(f"{self.sanitize_enum_member_name(member_name)},")

        self.indent_level -= 1
        self.emit("};")

    def sanitize_enum_member_name(self, name):
        if name == "object":
            return "object_"
        return self.sanitize_identifier_name(name)

    def next_anonymous_enum_name(self):
        name = f"anonymous_enum_{self.anonymous_enum_count}"
        self.anonymous_enum_count += 1
        return name

    def format_kernel_launch_comments(self, node):
        kernel_name = self.visit(node.kernel_name)
        config = [self.visit(node.blocks), self.visit(node.threads)]
        if node.shared_mem is not None:
            config.append(self.visit(node.shared_mem))
        if node.stream is not None:
            config.append(self.visit(node.stream))

        comments = [f"// Kernel launch: {kernel_name}<<<{', '.join(config)}>>>()"]
        if node.args:
            args = self.resolve_packed_launch_args(node.args)
            args_str = ", ".join([self.format_kernel_launch_arg(arg) for arg in args])
            comments.append(f"// Arguments: {args_str}")
        return comments

    def visit_VariableNode(self, node):
        self.register_variable_type(node.name, node.vtype)
        self.integer_constants.declare(
            node.name,
            integral_constant_declaration_value(node, self.integer_constants),
        )

        cuda_async_sync = self.cuda_async_sync_declaration_metadata(node)
        if cuda_async_sync is not None:
            self.register_cuda_async_sync_name(node.name, cuda_async_sync)
            self.emit(
                self.format_cuda_async_sync_declaration(node.name, cuda_async_sync)
            )
            return

        cooperative_group = self.cooperative_group_declaration_metadata(node)
        if cooperative_group is not None:
            group_kind = cooperative_group["kind"]
            self.register_cooperative_group_name(node.name, cooperative_group)
            if group_kind == "thread_block":
                self.emit(
                    f"// cooperative_groups thread_block {node.name} maps to the current workgroup"
                )
            elif (
                group_kind == "thread_block_tile"
                and cooperative_group.get("tile_size")
                and cooperative_group.get("parent_kind") == "thread_block"
            ):
                self.emit(
                    f"// cooperative_groups thread_block_tile<{cooperative_group['tile_size']}> "
                    f"{node.name} maps to a tiled partition of the current workgroup"
                )
            else:
                self.refuse_import(
                    f"CUDA cooperative_groups {group_kind} '{node.name}': CrossGL has no group of this kind"
                )
            return

        var_type = self.convert_cuda_variable_type_to_crossgl(node.vtype, node.name)
        self.register_vector1_name(node.name, node.vtype)
        output_name = self.register_identifier_name(node.name)

        self.register_packed_argument_list(node)
        self.register_unique_ptr_name(node.name, node.vtype)
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
                self.allocate_global_resource_binding(),
                "CUDA",
            )
            if declaration is not None:
                self.emit(declaration)
            return
        dim3_value = self.format_cuda_dim3_initializer(node)
        if dim3_value is not None:
            self.register_warp_mask_value(node.name, output_name, None)
            if self.is_namespace_scope_constant(node):
                self.namespace_constant_names.add(node.name)
                self.emit(f"const {var_type} {output_name} = {dim3_value};")
                return
            self.emit(f"var {output_name}: {var_type} = {dim3_value};")
            return
        if node.value:
            runtime_expression = self.format_cuda_runtime_expression(node.value)
            if runtime_expression is not None:
                comments, value = runtime_expression
                for comment in comments:
                    self.emit(comment)
                self.register_warp_mask_value(node.name, output_name, value)
                self.emit(f"var {output_name}: {var_type} = {value};")
                return

            value = self.format_variable_initializer_value(node.value)
            self.register_warp_mask_value(node.name, output_name, value)
            if self.is_namespace_scope_constant(node):
                self.namespace_constant_names.add(node.name)
                self.emit(f"const {var_type} {output_name} = {value};")
                return
            self.emit(f"var {output_name}: {var_type} = {value};")
        else:
            self.register_warp_mask_value(node.name, output_name, None)
            self.emit(f"var {output_name}: {var_type};")

    def format_cuda_dim3_initializer(self, node):
        """Return a ``vec3<u32>`` for a defaulted or braced ``dim3`` declaration.

        CUDA sets every ``dim3`` component that is not given to 1, so
        ``dim3 unit;`` is (1, 1, 1) and ``dim3 block = {16, 16};`` is (16, 16, 1).
        Return None for any other declaration.
        """
        if (
            self.strip_type_qualifiers(getattr(node, "vtype", "") or "").strip()
            != "dim3"
        ):
            return None
        value = getattr(node, "value", None)
        if value is None:
            args = []
        elif isinstance(value, InitializerListNode) and len(value.elements) <= 3:
            args = [self.visit(element) for element in value.elements]
        else:
            return None
        padded_args = list(args) + ["1"] * (3 - len(args))
        return f"vec3<u32>({', '.join(padded_args)})"

    def is_namespace_scope_constant(self, node):
        """Return whether a namespace-scope declaration is a ``const``/``constexpr`` constant.

        Such a constant imports as a CrossGL ``const`` so every target can read it
        from device code; a mutable global would become a host variable.
        """
        return self.namespace_scope_variable and is_c_family_namespace_constant(
            node, self.namespace_constant_names
        )

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

    def push_cuda_async_sync_scope(self):
        self.cuda_async_sync_scopes.append({})

    def pop_cuda_async_sync_scope(self):
        if len(self.cuda_async_sync_scopes) > 1:
            self.cuda_async_sync_scopes.pop()

    def push_variable_type_scope(self):
        self.variable_type_scopes.append({})
        self.integer_constants.push()

    def pop_variable_type_scope(self):
        if len(self.variable_type_scopes) > 1:
            self.variable_type_scopes.pop()
            self.integer_constants.pop()

    def push_warp_mask_value_scope(self):
        self.warp_mask_value_scopes.append({})

    def pop_warp_mask_value_scope(self):
        if len(self.warp_mask_value_scopes) > 1:
            self.warp_mask_value_scopes.pop()

    def register_warp_mask_value(self, raw_name, output_name, value):
        if not self.warp_mask_value_scopes:
            self.warp_mask_value_scopes.append({})

        names = {name for name in (raw_name, output_name) if name}
        if not names:
            return

        if self.is_full_or_active_warp_mask(value):
            resolved_value = self.resolve_warp_mask_expression(value)
            for name in names:
                self.warp_mask_value_scopes[-1][name] = resolved_value
            return

        for name in names:
            self.warp_mask_value_scopes[-1].pop(name, None)

    def format_variable_initializer_value(self, value):
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

    def resolve_warp_mask_expression(self, mask):
        text = str(mask).strip()
        seen = set()

        while text and text not in seen:
            seen.add(text)
            replacement = None
            for scope in reversed(self.warp_mask_value_scopes):
                if text in scope:
                    replacement = scope[text]
                    break
            if replacement is None:
                return text
            text = str(replacement).strip()

        return text

    def register_variable_type(self, name, type_name):
        if name:
            self.variable_type_scopes[-1][name] = type_name
            # A declaration hides an integral constant of its name from array extents.
            self.integer_constants.declare(name)

    def lookup_variable_type(self, name):
        for scope in reversed(self.variable_type_scopes):
            if name in scope:
                return scope[name]
        return None

    def register_cuda_async_sync_parameter(self, param):
        metadata = self.cuda_async_sync_metadata_from_type(param.vtype)
        if metadata is not None:
            self.register_cuda_async_sync_name(param.name, metadata)

    def register_cuda_async_sync_name(self, name, metadata):
        if name:
            self.cuda_async_sync_scopes[-1][name] = metadata

    def lookup_cuda_async_sync_metadata(self, name):
        for scope in reversed(self.cuda_async_sync_scopes):
            if name in scope:
                return scope[name]
        return None

    def register_cooperative_group_name(self, name, group_metadata):
        if name:
            self.cooperative_group_scopes[-1][name] = group_metadata

    def register_cooperative_group_parameter(self, param):
        metadata = self.cooperative_group_metadata_from_type(param.vtype)
        if metadata is not None:
            self.register_cooperative_group_name(param.name, metadata)

    def lookup_cooperative_group_name(self, name):
        metadata = self.lookup_cooperative_group_metadata(name)
        return metadata["kind"] if metadata is not None else None

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

    def pop_type_alias_scope(self):
        if len(self.type_alias_scopes) > 1:
            self.type_alias_scopes.pop()

    def push_vector1_name_scope(self):
        self.vector1_name_scopes.append({})

    def pop_vector1_name_scope(self):
        if len(self.vector1_name_scopes) > 1:
            self.vector1_name_scopes.pop()

    def register_vector1_name(self, name, type_name):
        if not name:
            return
        self.vector1_name_scopes[-1][name] = (
            self.cuda_vector1_scalar_type(type_name) is not None
        )

    def is_vector1_name(self, name):
        if not isinstance(name, str):
            return False
        for scope in reversed(self.vector1_name_scopes):
            if name in scope:
                return scope[name]
        return False

    def cuda_vector1_scalar_type(self, type_name):
        type_name = self.strip_type_qualifiers(type_name)
        if "*" in type_name or self.has_array_suffix(type_name):
            return None
        return self.VECTOR1_TYPE_MAPPING.get(type_name)

    def register_type_alias(self, name, alias_type):
        self.type_alias_scopes[-1][name] = alias_type

    def function_reference_parameters(self):
        """The :class:`ReferenceParameters` of the program being converted."""
        analysis = getattr(self, "reference_parameters", None)
        if analysis is None:
            analysis = ReferenceParameters(None)
            self.reference_parameters = analysis
        return analysis

    def resolve_type_alias(self, type_name):
        type_name = self.strip_type_qualifiers(type_name)
        for scope in reversed(self.type_alias_scopes):
            if type_name in scope:
                return scope[type_name]
        return type_name

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

    def visit_SharedMemoryNode(self, node):
        cuda_async_sync = self.cuda_async_sync_metadata_from_type(node.vtype)
        if cuda_async_sync is not None:
            self.register_cuda_async_sync_name(node.name, cuda_async_sync)
            self.emit(
                self.format_cuda_async_sync_declaration(node.name, cuda_async_sync)
            )
            return

        # Convert to workgroup memory in CrossGL
        var_type = self.convert_cuda_variable_type_to_crossgl(node.vtype, node.name)
        self.register_vector1_name(node.name, node.vtype)
        if getattr(node, "is_dynamic_shared_memory", False):
            self.emit(
                f"// CUDA dynamic shared memory: {node.name} uses launch-time "
                "shared memory size"
            )
        if node.size:
            size = self.format_crossgl_array_extent(self.visit(node.size))
            self.emit(f"var<workgroup> {node.name}: array<{var_type}, {size}>;")
        else:
            self.emit(f"var<workgroup> {node.name}: {var_type};")

    def visit_ConstantMemoryNode(self, node):
        # Convert to uniform buffer in CrossGL
        var_type = self.convert_cuda_variable_type_to_crossgl(node.vtype, node.name)
        self.register_vector1_name(node.name, node.vtype)
        if (
            node.value is None
            and self.function_body_depth > 0
            and "static" in node.qualifiers
        ):
            # A function-scope static __constant__ local has static storage,
            # so without an initializer every element is zero, and device
            # code cannot write constant memory. Emit it as a function-scope
            # constant of zeros: it must neither take a resource binding
            # (which would also collide with the kernel parameters) nor be
            # read uninitialized by a later backend. A block-scope extern
            # declaration names a namespace-scope variable the host may have
            # written, so it is not one.
            self.emit(f"let {node.name}: {var_type} = {crossgl_zero_value(var_type)};")
            return
        binding = self.allocate_global_resource_binding()
        if node.value:
            value = self.visit(node.value)
            self.emit(
                f"@group(0) @binding({binding}) var<uniform> {node.name}: {var_type} = {value};"
            )
        else:
            self.emit(
                f"@group(0) @binding({binding}) var<uniform> {node.name}: {var_type};"
            )

    def allocate_global_resource_binding(self):
        binding = self.global_resource_binding_count
        self.global_resource_binding_count += 1
        return binding
