"""CrossGL ``out`` and ``inout`` parameters in Rust output.

CrossGL follows GLSL 4.60 section 6.1.1: an ``out`` or ``inout`` argument is copied into the callee (``inout``
only), the callee works on its own copy, and the copy is written back to the caller's lvalue when the callee
returns. Rust passes every value parameter by move or copy, so a by-value ``mut x: T`` parameter silently drops the
write-back. The Rust emitter therefore declares such a parameter as ``x_ref: &mut T`` and keeps GLSL's copy
semantics inside the callee:

* the function body starts with ``let mut x: T = *x_ref;``, so the body reads and writes a local copy exactly as
  CrossGL describes, and
* every ``return`` (and the end of a body that falls off its last statement) first evaluates the returned value,
  then stores ``*x_ref = x;`` for each such parameter in declaration order, then returns.

An ``out`` parameter is copied in as well: GLSL leaves its initial value undefined, and the caller's value is one
such value. At the call site the argument becomes ``&mut place``. Rust accepts that only for a place expression
owned by the caller that no other argument of the call reads, so the emitter refuses, with a typed diagnostic,
an argument that is not a local variable path (a swizzle of several components, a constructor, a global), and a
call whose ``out``/``inout`` argument shares its root variable with another argument. Copy-in and copy-out would
give such a call a defined meaning; a mutable borrow cannot express it.
"""

from __future__ import annotations

from ..ast import ArrayAccessNode, IdentifierNode, MemberAccessNode, VariableNode
from .c_reference_parameters import is_output_parameter

__all__ = ["RustOutputParameterMixin"]

_PATH_NODES = (ArrayAccessNode, MemberAccessNode)


def _function_parameters(function):
    return list(getattr(function, "parameters", getattr(function, "params", [])) or [])


def _array_base(node):
    return getattr(node, "array", getattr(node, "array_expr", None))


def _member_base(node):
    return getattr(node, "object", getattr(node, "object_expr", None))


def _storage_root(expression):
    """The variable an lvalue path names storage in, or ``None`` when ``expression`` is not such a path."""
    current = expression
    while isinstance(current, _PATH_NODES):
        if isinstance(current, ArrayAccessNode):
            current = _array_base(current)
        else:
            current = _member_base(current)
    if isinstance(current, (IdentifierNode, VariableNode)):
        return current.name
    return None


def _referenced_names(node, found=None, seen=None):
    """Every identifier name read anywhere inside ``node`` (an expression tree)."""
    if found is None:
        found, seen = set(), set()
    if node is None or isinstance(node, (str, int, float, bool)):
        return found
    if id(node) in seen:
        return found
    seen.add(id(node))
    if isinstance(node, (list, tuple)):
        for item in node:
            _referenced_names(item, found, seen)
        return found
    if isinstance(node, dict):
        for item in node.values():
            _referenced_names(item, found, seen)
        return found
    if isinstance(node, (IdentifierNode, VariableNode)):
        found.add(node.name)
    for name, value in vars(node).items() if hasattr(node, "__dict__") else ():
        if name in {"parent", "type_info", "resolved_type"}:
            continue
        if isinstance(value, (list, tuple, dict)) or hasattr(value, "__dict__"):
            _referenced_names(value, found, seen)
    return found


class RustOutputParameterMixin:
    """Copy-in/copy-out lowering of CrossGL ``out``/``inout`` parameters for :class:`RustCodeGen`."""

    #: ``(reference name, local name)`` of each ``out``/``inout`` parameter of the function being emitted.
    current_rust_output_parameters: tuple | list = ()

    # -- callee ---------------------------------------------------------------------------------------------

    def rust_output_parameter_reference_name(self, func, parameter_name):
        """A parameter name for the ``&mut`` reference of ``parameter_name`` that no name in ``func`` uses."""
        taken = _referenced_names(getattr(func, "body", None))
        taken.update(
            getattr(parameter, "name", None) for parameter in _function_parameters(func)
        )
        taken.update(self.user_function_names)
        taken.update(self.variable_types)
        base = f"{parameter_name}_ref"
        candidate, suffix = base, 1
        while candidate in taken:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def rust_output_return_temporary(self):
        """A local name for a returned value that the copy-back statements cannot shadow."""
        taken = {local for _, local in self.current_rust_output_parameters}
        taken.update(reference for reference, _ in self.current_rust_output_parameters)
        taken.update(self.local_variable_names)
        taken.update(self.variable_types)
        candidate, suffix = "return_value", 1
        while candidate in taken:
            candidate = f"return_value_{suffix}"
            suffix += 1
        return candidate

    def rust_output_parameter_copy_back(self, indent_str):
        """``*x_ref = x;`` for every ``out``/``inout`` parameter of the current function, in order."""
        return "".join(
            f"{indent_str}*{reference} = {local};\n"
            for reference, local in self.current_rust_output_parameters
        )

    def generate_rust_output_return(self, indent_str, return_expr):
        """A ``return`` that writes the ``out``/``inout`` copies back after evaluating ``return_expr``."""
        copy_back = self.rust_output_parameter_copy_back(indent_str)
        if return_expr is None:
            return f"{copy_back}{indent_str}return;\n"
        temporary = self.rust_output_return_temporary()
        return (
            f"{indent_str}let {temporary} = {return_expr};\n"
            f"{copy_back}{indent_str}return {temporary};\n"
        )

    # -- caller ---------------------------------------------------------------------------------------------

    def rust_function_output_indices(self, func_name):
        """Positions of the ``out``/``inout`` parameters of user function ``func_name``."""
        function = self.user_function_nodes.get(func_name)
        if function is None:
            return []
        return [
            index
            for index, parameter in enumerate(_function_parameters(function))
            if is_output_parameter(parameter)
        ]

    def rust_output_argument_roots(self, func_name, args):
        """The caller bindings that a call to ``func_name`` borrows mutably for its ``out``/``inout`` arguments."""
        roots = set()
        args = list(args or [])
        for index in self.rust_function_output_indices(func_name):
            if index < len(args):
                root = _storage_root(args[index])
                if root is not None:
                    roots.add(root)
        return roots

    def rust_output_argument_refusal(self, func_name, args, index):
        """Why argument ``index`` of a call to ``func_name`` cannot be borrowed ``&mut``, or ``None``."""
        argument = args[index]
        position = index + 1
        root = _storage_root(argument)
        if root is None or root not in self.local_variable_names:
            return (
                f"argument {position} of {func_name} is passed to an out/inout parameter but is not a "
                "local variable, member or element that Rust can borrow mutably"
            )
        current = argument
        while isinstance(current, _PATH_NODES):
            if isinstance(current, MemberAccessNode):
                member = str(getattr(current, "member", ""))
                components = self.member_swizzle_components(current)
                if components is not None and len(components) > 1:
                    return (
                        f"argument {position} of {func_name} passes swizzle .{member} to an out/inout "
                        "parameter; a Rust swizzle is a new vector, not a place to borrow"
                    )
                current = _member_base(current)
            else:
                current = _array_base(current)
        for other_index, other in enumerate(args):
            if other_index == index:
                continue
            if root in _referenced_names(other):
                return (
                    f"call to {func_name} passes {root} to the out/inout parameter {position} and reads it "
                    f"in argument {other_index + 1}; Rust cannot borrow it mutably while it is read"
                )
        return None

    def apply_rust_output_arguments(self, func_name, args, generated_args):
        """Replace the generated ``out``/``inout`` arguments of a call with ``&mut`` borrows of the caller's places."""
        args = list(args or [])
        for index in self.rust_function_output_indices(func_name):
            if index >= len(args) or index >= len(generated_args):
                continue
            reason = self.rust_output_argument_refusal(func_name, args, index)
            if reason is not None:
                generated_args[index] = self.refuse_lowering(
                    reason, "Default::default()"
                )
                continue
            self.assignment_lhs_depth += 1
            try:
                place = self.generate_expression(args[index])
            finally:
                self.assignment_lhs_depth -= 1
            generated_args[index] = f"&mut {place}"
        return generated_args
