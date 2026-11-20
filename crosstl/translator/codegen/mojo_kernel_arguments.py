"""Mojo kernel arguments for the buffers and uniform values a CrossGL program declares outside its functions.

Mojo has no global variables: Mojo 1.2 rejects a ``var`` at module scope. A Mojo GPU kernel therefore receives each
storage buffer as an ``UnsafePointer`` argument and each uniform value by value, the way MAX's own kernels take
theirs. A function that reads one of them, directly or through the functions it calls, receives it as an extra
argument that its callers pass on.

The GPU index built-ins need no argument: every device function can read Mojo's ``global_idx``, ``thread_idx``,
``block_idx``, ``block_dim`` and ``grid_dim``, so a function that reads one declares it as a local.

Any other module-scope variable, such as a private mutable global or a texture, has no Mojo spelling, and the
generator refuses the program instead of declaring it.
"""

from __future__ import annotations

from ..ast import (
    ArrayLiteralNode,
    ArrayNode,
    FunctionCallNode,
    IdentifierNode,
    VariableNode,
)
from ..refusals import MojoRefusal
from .mojo_tables import (
    MOJO_BUILTIN_PLACEHOLDERS,
    MOJO_DTYPE_INFO,
    MOJO_GPU_BUILTIN_ALIASES,
)

# The qualifiers of a module-scope declaration whose value the host supplies and the kernel only reads.
MOJO_UNIFORM_QUALIFIERS = frozenset({"uniform"})
# The Mojo value each GPU index alias of mojo_tables.MOJO_GPU_BUILTIN_ALIASES reads.
MOJO_GPU_INDEX_SOURCES = {
    "global_idx_uint": "global_idx",
    "thread_idx_uint": "thread_idx",
    "block_idx_uint": "block_idx",
    "grid_dim_uint": "grid_dim",
    "block_dim_uint": "block_dim",
}
# The built-in placeholders of mojo_tables.MOJO_BUILTIN_PLACEHOLDERS that a Mojo device function can compute.
MOJO_COMPUTED_BUILTIN_PLACEHOLDERS = {
    "local_invocation_index": (
        "UInt32(thread_idx.z * block_dim.y * block_dim.x + thread_idx.y * block_dim.x + thread_idx.x)"
    ),
}
_SKIPPED_ATTRIBUTES = frozenset({"parent", "annotations", "source_location"})
# The Mojo index a compute entry point's built-in parameter reads, by mojo_codegen's normalized semantic.
MOJO_ENTRY_SEMANTIC_SOURCES = {
    "global_invocation_id": "global_idx",
    "local_invocation_id": "thread_idx",
    "workgroup_id": "block_idx",
    "group_index": "local_invocation_index",
    "local_invocation_index": "local_invocation_index",
}


def mojo_referenced_names(node, names=None, visited=None):
    """Every identifier, built-in and called function name a CrossGL subtree mentions."""
    if names is None:
        names = set()
    if visited is None:
        visited = set()
    if node is None or isinstance(node, (int, float, bool)):
        return names
    if isinstance(node, str):
        names.add(node)
        return names
    if isinstance(node, (list, tuple, set)):
        for item in node:
            mojo_referenced_names(item, names, visited)
        return names
    if isinstance(node, dict):
        for item in node.values():
            mojo_referenced_names(item, names, visited)
        return names
    if id(node) in visited or not hasattr(node, "__dict__"):
        return names
    visited.add(id(node))
    if isinstance(node, IdentifierNode):
        names.add(node.name)
    if isinstance(node, FunctionCallNode) and isinstance(node.function, str):
        names.add(node.function)
    builtin_name = getattr(node, "builtin_name", None)
    if isinstance(builtin_name, str):
        names.add(builtin_name)
    for attribute, value in vars(node).items():
        if attribute in _SKIPPED_ATTRIBUTES or isinstance(value, str):
            continue
        mojo_referenced_names(value, names, visited)
    return names


def mojo_declared_names(node, names=None, visited=None):
    """The names of the variables a CrossGL subtree declares."""
    if names is None:
        names = set()
    if visited is None:
        visited = set()
    if isinstance(node, (list, tuple)):
        for item in node:
            mojo_declared_names(item, names, visited)
        return names
    if node is None or not hasattr(node, "__dict__") or id(node) in visited:
        return names
    visited.add(id(node))
    if isinstance(node, (VariableNode, ArrayNode)) and isinstance(
        getattr(node, "name", None), str
    ):
        names.add(node.name)
    for attribute, value in vars(node).items():
        if attribute not in _SKIPPED_ATTRIBUTES:
            mojo_declared_names(value, names, visited)
    return names


class MojoKernelArgumentMixin:
    """Kernel-argument members of ``MojoCodeGen``."""

    def reset_mojo_kernel_arguments(self):
        self.mojo_module_kernel_arguments = {}
        self.mojo_stage_kernel_arguments = {}

    def mojo_kernel_argument_type(self, node, variable_type):
        """The CrossGL type a module-scope declaration is passed to the kernel as, or None."""
        if self.is_mojo_shared_variable(node):
            return None
        if getattr(node, "initial_value", None) is not None:
            return None
        if self.parse_pointer_type_name(str(variable_type)) is not None:
            return variable_type
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        if "buffer" in qualifiers and self.is_array_type_name(variable_type):
            element_type, size = self.parse_array_type_name(variable_type)
            if size in (None, ""):
                # A storage buffer's runtime-sized array is the buffer itself: a pointer to its first element.
                return f"{element_type}*"
        if not qualifiers & MOJO_UNIFORM_QUALIFIERS:
            return None
        if self.is_resource_type_name(variable_type):
            return None
        return variable_type

    def declare_mojo_kernel_argument(self, name, variable_type, stage_scope=False):
        """Record a buffer or uniform value as a kernel argument instead of a module-scope variable."""
        arguments = (
            self.mojo_stage_kernel_arguments
            if stage_scope
            else self.mojo_module_kernel_arguments
        )
        arguments.setdefault(name, variable_type)
        self.register_variable_type(name, variable_type)
        return ""

    def refuse_mojo_module_variable(self, name, what="a variable", type_name=None):
        """Refuse a module-scope declaration Mojo cannot hold."""
        described = f"{name}: {self.type_name(type_name)}" if type_name else name
        raise MojoRefusal(
            f"Mojo has no global variables, so {what} declared outside a function ({described}) has no Mojo "
            "spelling; only storage buffers and uniform values, which become kernel arguments, can be declared "
            "there"
        )

    def is_mojo_constant_declaration(self, node):
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        return "const" in qualifiers or getattr(node, "is_mutable", True) is False

    def generate_module_variable_declaration(self, node, stage_scope=False):
        """Declare a module- or stage-scope variable as a kernel argument or a constant.

        Buffers and uniform values become kernel arguments and constants become ``comptime`` values. Every other
        declaration keeps the module-scope ``var`` the generator wrote before Mojo dropped global variables.
        """
        name = getattr(node, "name", None)
        if isinstance(node, ArrayNode):
            array_type = self.mojo_array_kernel_argument_type(node)
            if array_type is not None:
                return self.declare_mojo_kernel_argument(name, array_type, stage_scope)
            self.register_variable_type(name, self.array_declared_type(node))
            return self.generate_array_declaration(node)
        variable_type = self.variable_declared_type(node)
        initial_value = getattr(node, "initial_value", None)
        if initial_value is not None:
            variable_type = variable_type or self.expression_result_type(initial_value)
            if self.is_mojo_constant_declaration(node):
                return self.generate_module_constant_declaration(node, variable_type)
            return self.generate_legacy_initialized_declaration(node, variable_type)
        variable_type = variable_type or "float"
        self.register_resource_access_metadata(node, variable_type)
        argument_type = self.mojo_kernel_argument_type(node, variable_type)
        if argument_type is not None:
            # The binding comment tells the host which kernel argument a binding feeds.
            comment = self.generate_resource_metadata_comment(node, variable_type)
            return comment + self.declare_mojo_kernel_argument(
                name, argument_type, stage_scope
            )
        return self.generate_legacy_module_variable(node, variable_type, stage_scope)

    def generate_legacy_initialized_declaration(self, node, variable_type):
        self.register_variable_type(node.name, variable_type)
        context = f"global declaration {node.name}"
        if isinstance(node.initial_value, ArrayLiteralNode) and (
            variable_type is not None and self.is_array_type_name(variable_type)
        ):
            value = self.generate_array_literal_expression(
                node.initial_value, variable_type
            )
        else:
            self.validate_expression_target_shape(
                node.initial_value, variable_type, context
            )
            value = self.generate_expression(node.initial_value, variable_type, context)
            self.validate_scalar_target_conversion(
                node.initial_value, variable_type, context
            )
        name = self.mojo_identifier(node.name)
        if variable_type is None:
            return f"var {name} = {value}\n"
        return (
            f"{self.generate_resource_metadata_comment(node, variable_type)}"
            f"var {name}: {self.map_type(variable_type)} = {value}\n"
        )

    def generate_legacy_module_variable(self, node, variable_type, stage_scope):
        """The module-scope ``var`` still written for a texture, image, HLSL buffer object or private global."""
        self.register_variable_type(node.name, variable_type)
        resource_comment = self.generate_resource_metadata_comment(node, variable_type)
        name = self.mojo_identifier(node.name)
        if self.is_array_type_name(variable_type):
            return (
                f"{resource_comment}var {name} = "
                f"{self.array_initial_value_for_type(variable_type)}\n"
            )
        if self.is_struct_type_name(variable_type):
            return f"{resource_comment}var {name} = {self.zero_value_for_type(variable_type)}\n"
        if self.is_resource_type_name(variable_type):
            mapped_type = self.map_type(variable_type)
            init_value = (
                self.zero_value_for_type(variable_type)
                if stage_scope
                else self.resource_binding_initial_value(
                    node.name, mapped_type, variable_type
                )
            )
            return f"{resource_comment}var {name}: {mapped_type} = {init_value}\n"
        return (
            f"var {name}: {self.map_type(variable_type)} = "
            f"{self.zero_value_for_type(variable_type)}\n"
        )

    def generate_module_constant_declaration(self, node, variable_type):
        """A compile-time ``comptime`` value for a constant declared outside a function."""
        self.register_variable_type(node.name, variable_type)
        context = f"global declaration {node.name}"
        self.validate_expression_target_shape(
            node.initial_value, variable_type, context
        )
        value = self.generate_expression(node.initial_value, variable_type, context)
        self.validate_scalar_target_conversion(
            node.initial_value, variable_type, context
        )
        name = self.mojo_identifier(node.name)
        if variable_type is None:
            return f"comptime {name} = {value}\n"
        return f"comptime {name}: {self.map_type(variable_type)} = {value}\n"

    def mojo_stage_functions(self, ast):
        """Every function the program's stages declare, in stage order."""
        functions = []
        for stage in (getattr(ast, "stages", None) or {}).values():
            functions.extend(getattr(stage, "local_functions", None) or [])
        return functions

    def mojo_entry_kernel_arguments(self):
        """The arguments of the kernel being written: the module's buffers and uniforms, then the stage's."""
        arguments = dict(self.mojo_module_kernel_arguments)
        for name, type_name in self.mojo_stage_kernel_arguments.items():
            arguments.setdefault(name, type_name)
        return list(arguments.items())

    def mojo_gpu_index_local(self, name, type_name, source):
        """A local holding a GPU index read from one of Mojo's index values, in the CrossGL type."""
        local_name = self.mojo_identifier(name)
        if source == "local_invocation_index":
            value = MOJO_COMPUTED_BUILTIN_PLACEHOLDERS["local_invocation_index"]
            scalar = self.map_type(type_name)
            return f"var {local_name}: {scalar} = {scalar}({value})"
        vector_info = self.vector_type_info(type_name)
        if vector_info is None:
            scalar = self.map_type(type_name)
            return f"var {local_name}: {scalar} = {scalar}({source}.x)"
        dtype, width, storage_width, _pad = vector_info
        scalar = self.map_type(MOJO_DTYPE_INFO[dtype][0])
        components = [f"{scalar}({source}.{axis})" for axis in "xyz"[:width]]
        components += [f"{scalar}(0)"] * (storage_width - len(components))
        return (
            f"var {local_name} = SIMD[{dtype}, {storage_width}]"
            f"({', '.join(components)})"
        )

    def mojo_compute_entry_parameter_local(self, name, type_name, semantic):
        """The local a compute entry point's built-in parameter becomes, None for a host value, or a refusal.

        A Mojo kernel's arguments come from the host, so a parameter the CrossGL entry point receives from the
        pipeline is read from Mojo's GPU indices inside the kernel instead. A parameter without a semantic is a
        value the host passes, which stays a kernel argument.
        """
        if not semantic:
            return None
        source = MOJO_ENTRY_SEMANTIC_SOURCES.get(self.normalized_semantic_key(semantic))
        if source is None:
            raise MojoRefusal(
                f"Mojo compute entry parameter {name} has no GPU index source for semantic {semantic}; a Mojo "
                "kernel's arguments are the buffers and uniform values the host passes"
            )
        return self.mojo_gpu_index_local(name, type_name, source)

    def mojo_array_kernel_argument_type(self, node):
        """The CrossGL array type of a uniform array declaration, or None."""
        if not isinstance(node, ArrayNode):
            return None
        array_type = self.array_declared_type(node)
        return self.mojo_kernel_argument_type(node, array_type)

    def mojo_kernel_argument_captures(self, functions, arguments):
        """The kernel arguments each function needs, directly or through its calls, keyed by function name."""
        if not arguments:
            return {}
        direct = {}
        calls = {}
        for func in functions:
            name = getattr(func, "name", None)
            if not isinstance(name, str):
                continue
            parameters = {
                getattr(param, "name", None)
                for param in (
                    getattr(func, "parameters", getattr(func, "params", [])) or []
                )
            }
            body = getattr(func, "body", None)
            mentioned = mojo_referenced_names(body)
            local_names = parameters | mojo_declared_names(body)
            direct.setdefault(name, set()).update(
                argument
                for argument in arguments
                if argument in mentioned and argument not in local_names
            )
            calls.setdefault(name, set()).update(mentioned)
        captured = {name: set(found) for name, found in direct.items()}
        changed = True
        while changed:
            changed = False
            for name in sorted(captured):
                for callee in sorted(calls[name] & captured.keys()):
                    if callee == name:
                        continue
                    missing = captured[callee] - captured[name]
                    if missing:
                        captured[name].update(missing)
                        changed = True
        return {
            name: [
                (argument, arguments[argument])
                for argument in arguments
                if argument in found
            ]
            for name, found in sorted(captured.items())
            if found
        }

    def merge_mojo_extra_params(self, *mappings):
        """Merge function-name → extra-parameter lists, keeping the first occurrence of each name."""
        merged = {}
        for mapping in mappings:
            for function_name, extras in mapping.items():
                current = merged.setdefault(function_name, [])
                known = {name for name, _ in current}
                current.extend(
                    (name, type_name) for name, type_name in extras if name not in known
                )
        return merged

    def mojo_function_local_builtin_declarations(self, body, indent):
        """Locals for the GPU indices and computed built-ins a function body reads.

        Each local is registered as a variable of the function, so the expressions that read the built-in use it
        instead of asking for a module-scope placeholder.
        """
        mentioned = mojo_referenced_names(body)
        lines = []
        aliases = sorted(
            {
                MOJO_GPU_BUILTIN_ALIASES[name]
                for name in mentioned
                if name in MOJO_GPU_BUILTIN_ALIASES
            }
        )
        for alias_name, type_name in aliases:
            source = MOJO_GPU_INDEX_SOURCES.get(alias_name)
            vector_info = self.vector_type_info(type_name)
            if (
                source is None
                or vector_info is None
                or alias_name in self.variable_types
            ):
                continue
            dtype, width, storage_width, _pad = vector_info
            scalar = self.map_type(MOJO_DTYPE_INFO[dtype][0])
            components = [f"{scalar}({source}.{axis})" for axis in "xyz"[:width]]
            components += [f"{scalar}(0)"] * (storage_width - len(components))
            lines.append(
                f"var {alias_name} = SIMD[{dtype}, {storage_width}]"
                f"({', '.join(components)})"
            )
            self.register_variable_type(alias_name, type_name)
        placeholders = sorted(
            {
                MOJO_BUILTIN_PLACEHOLDERS[name]
                for name in mentioned
                if name in MOJO_BUILTIN_PLACEHOLDERS
            }
        )
        for placeholder, type_name in placeholders:
            value = MOJO_COMPUTED_BUILTIN_PLACEHOLDERS.get(placeholder)
            if value is None or placeholder in self.variable_types:
                continue
            lines.append(f"var {placeholder}: {self.map_type(type_name)} = {value}")
            self.register_variable_type(placeholder, type_name)
        return "".join(f"{'    ' * indent}{line}\n" for line in lines)
