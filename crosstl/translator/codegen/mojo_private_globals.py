"""Private variables and built-in inputs a CrossGL program declares outside its functions, for Mojo.

Mojo 1.2 has no global variables. A private variable declared outside a function (GLSL's plain global, HLSL's
``static`` global) has one copy per invocation that lives for the whole invocation, which is what a local of the
Mojo kernel is. The kernel therefore declares each one at the top of its body, with its initializer, and every
function that reads or writes it, directly or through the functions it calls, receives it as a ``mut`` argument,
which Mojo passes by reference, so a store in a helper is the kernel's store.

A stage input variable with a GPU index built-in (``uint3 id @input @gl_GlobalInvocationID;``) is not a variable
at all on Mojo: every function that reads it declares a local from Mojo's GPU indices, as for the built-in names.
"""

from __future__ import annotations

from ..ast import ArrayLiteralNode

# Qualifiers that keep a declaration private to one invocation. Any other storage or interface qualifier (in, out,
# buffer, uniform, a resource access) makes it host-visible or stage input/output, which these rules do not cover.
MOJO_PRIVATE_GLOBAL_QUALIFIERS = frozenset(
    {"static", "private", "thread", "precise", "highp", "mediump", "lowp"}
)
# The built-in semantics whose stage input variable reads one of Mojo's GPU indices.
MOJO_BUILTIN_INPUT_SOURCES = {
    "global_invocation_id": "global_idx",
    "local_invocation_id": "thread_idx",
    "workgroup_id": "block_idx",
    "group_index": "local_invocation_index",
    "local_invocation_index": "local_invocation_index",
}


class MojoPrivateGlobalMixin:
    """Private-global members of ``MojoCodeGen``."""

    def reset_mojo_private_globals(self):
        self.mojo_module_private_globals = {}
        self.mojo_stage_private_globals = {}
        self.mojo_module_builtin_inputs = {}
        self.mojo_stage_builtin_inputs = {}

    def reset_mojo_stage_private_globals(self):
        self.mojo_stage_private_globals = {}
        self.mojo_stage_builtin_inputs = {}

    def is_mojo_private_global(self, node):
        """Whether a declaration outside a function is a private variable of each invocation."""
        qualifiers = {
            str(qualifier).lower()
            for qualifier in getattr(node, "qualifiers", []) or []
        }
        if getattr(node, "resource_qualifiers", None):
            return False
        if self.mojo_builtin_input_source(node) is not None:
            return False
        if {
            str(getattr(a, "name", a)) for a in getattr(node, "attributes", []) or []
        } & {
            "input",
            "output",
        }:
            return False
        return qualifiers <= MOJO_PRIVATE_GLOBAL_QUALIFIERS

    def mojo_builtin_input_source(self, node):
        """The Mojo GPU index a stage input variable reads, or None."""
        semantic = self.node_semantic(node)
        if not semantic:
            return None
        return MOJO_BUILTIN_INPUT_SOURCES.get(self.normalized_semantic_key(semantic))

    def declare_mojo_builtin_input(self, node, variable_type, stage_scope):
        """Record a GPU index input variable; each function that reads it declares it as a local."""
        inputs = (
            self.mojo_stage_builtin_inputs
            if stage_scope
            else self.mojo_module_builtin_inputs
        )
        inputs.setdefault(
            node.name, (variable_type, self.mojo_builtin_input_source(node))
        )
        return ""

    def declare_mojo_private_global(self, node, variable_type, stage_scope):
        """Record a private variable the kernel declares and the functions using it take as ``mut``."""
        privates = (
            self.mojo_stage_private_globals
            if stage_scope
            else self.mojo_module_private_globals
        )
        privates.setdefault(
            node.name, (variable_type, getattr(node, "initial_value", None))
        )
        self.register_variable_type(node.name, variable_type)
        return ""

    def mojo_private_globals(self):
        """Every private variable the kernel being written declares, module scope first, in declaration order."""
        privates = dict(self.mojo_module_private_globals)
        for name, entry in self.mojo_stage_private_globals.items():
            privates.setdefault(name, entry)
        return privates

    def mojo_private_global_arguments(self):
        """The private variables as the ``mut`` arguments helper functions take them as, by name."""
        return {
            name: variable_type
            for name, (variable_type, _value) in self.mojo_private_globals().items()
        }

    def is_mojo_private_global_name(self, name):
        return name in self.mojo_module_private_globals or (
            name in self.mojo_stage_private_globals
        )

    def mojo_private_global_declarations(self, indent):
        """The kernel's locals for the private variables, each with its initializer or zero value."""
        code = ""
        for name, (variable_type, value) in self.mojo_private_globals().items():
            self.register_variable_type(name, variable_type)
            code += (
                f"{'    ' * indent}var {self.mojo_identifier(name)}: "
                f"{self.map_type(variable_type)} = "
                f"{self.mojo_private_global_value(name, variable_type, value)}\n"
            )
        return code

    def mojo_private_global_value(self, name, variable_type, value):
        if value is None:
            if self.is_array_type_name(variable_type):
                return self.array_initial_value_for_type(variable_type)
            return self.zero_value_for_type(variable_type)
        if isinstance(value, ArrayLiteralNode) and self.is_array_type_name(
            variable_type
        ):
            return self.generate_array_literal_expression(value, variable_type)
        context = f"global declaration {name}"
        self.validate_expression_target_shape(value, variable_type, context)
        generated = self.generate_expression(value, variable_type, context)
        self.validate_scalar_target_conversion(value, variable_type, context)
        return generated

    def mojo_builtin_input_declarations(self, mentioned, indent):
        """Locals for the GPU index input variables a function body reads."""
        inputs = dict(self.mojo_module_builtin_inputs)
        for name, entry in self.mojo_stage_builtin_inputs.items():
            inputs.setdefault(name, entry)
        code = ""
        for name in sorted(inputs):
            if name not in mentioned or name in self.variable_types:
                continue
            variable_type, source = inputs[name]
            code += (
                f"{'    ' * indent}"
                f"{self.mojo_gpu_index_local(name, variable_type, source)}\n"
            )
            self.register_variable_type(name, variable_type)
        return code
