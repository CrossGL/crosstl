"""WGSL pointer locals: a CrossGL ``T* p = &r;`` in a block becomes ``let p = &r;``.

WGSL keeps a pointer in a function only in a ``let`` declaration: a pointer is not storable, so no ``var`` holds one
(WGSL, "Pointer Types" and "let Declarations"). The declaration infers ``ptr<AS, T, AM>`` from its address-of
initializer ("Address-Of Expression"), so the generator writes no address space or access mode. A ``let`` cannot be
assigned, so the pointer keeps its target for its whole scope. The generator writes the declaration when

* the initializer is ``&r`` and ``r`` designates memory: a ``var`` local, a parameter the function copies into a
  ``var`` (taking its address counts as a write, see ``written_parameter_names``) or takes by reference, a buffer
  pointer parameter, or a writable module-scope variable (storage, workgroup or private). ``r`` may index arrays
  and matrices and select structure members, but it may not name a vector component, which WGSL cannot take the
  address of;
* every later use of the pointer in its scope dereferences it (``*p``), and the scope declares no other ``p``;
* no write goes through it into a read-only storage buffer.

Anything else, such as an uninitialized pointer, re-pointing it, pointer arithmetic or indexing, comparing pointers
or passing one on, has no WGSL form here and is refused by name.
"""

from __future__ import annotations

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    PointerType,
    SwizzleNode,
    UnaryOpNode,
    VariableNode,
)
from ..refusals import WGSLRefusal
from .ast_walk import walk_ast_values

__all__ = [
    "addressable_module_names",
    "generate_pointer_local",
    "is_pointer_local_declaration",
]

_INCREMENT_OPERATORS = frozenset({"++", "--"})


def is_pointer_local_declaration(statement) -> bool:
    """Whether ``statement`` declares a local whose declared type is a pointer."""
    return isinstance(statement, VariableNode) and isinstance(
        getattr(statement, "var_type", None), PointerType
    )


def addressable_module_names(codegen, global_variables, stage_parameters):
    """Module-scope names WGSL declares as a writable ``var``, whose address a pointer local may take."""
    read_only = getattr(codegen, "_read_only_module_names", ())
    names = set()
    for variable in global_variables:
        name = getattr(variable, "name", "")
        if not name or name in read_only:
            continue
        var_type = getattr(variable, "var_type", None)
        if codegen.is_buffer_pointer_type(
            var_type, getattr(variable, "qualifiers", ())
        ):
            names.add(name)
            continue
        if var_type is None or isinstance(var_type, PointerType):
            continue
        try:
            opaque = codegen.is_opaque_resource_type(var_type)
        except ValueError:
            continue
        if not opaque:
            names.add(name)
    for parameter in stage_parameters:
        if codegen.structured_buffer_access(getattr(parameter, "param_type", None)):
            names.add(parameter.name)
    return names


def generate_pointer_local(codegen, statement, indent, following):
    """The ``let`` line for the pointer local ``statement``; ``following`` are the statements of its scope."""
    name = statement.name
    initializer = statement.initial_value
    if not (
        isinstance(initializer, UnaryOpNode)
        and str(initializer.operator) == "&"
        and not initializer.is_postfix
    ):
        raise WGSLRefusal(
            f"WGSL target cannot declare pointer local '{name}' without an "
            "address-of initializer: WGSL keeps a pointer only in a let "
            "declaration initialized with &variable"
        )
    root = _pointer_target_root(codegen, name, initializer.operand)
    _validate_pointer_uses(codegen, name, root, following)
    address = codegen.generate_expression(initializer)
    variable_name = codegen.declare_local_identifier(name)
    codegen.register_local_identifier(name)
    codegen.register_local_mutability(name, False)
    codegen.register_value_type(name, statement.var_type)
    return f"{'    ' * indent}let {variable_name} = {address};"


def _pointer_target_root(codegen, name, target):
    """The root variable name of ``target``, refusing a target WGSL cannot take the address of."""
    node = target
    while True:
        if isinstance(node, SwizzleNode):
            _refuse_vector_component(name)
        if isinstance(node, MemberAccessNode):
            if _is_vector(codegen, node.object_expr):
                _refuse_vector_component(name)
            node = node.object_expr
            continue
        if isinstance(node, ArrayAccessNode):
            if _is_vector(codegen, node.array_expr):
                _refuse_vector_component(name)
            node = node.array_expr
            continue
        break
    if not isinstance(node, IdentifierNode):
        raise WGSLRefusal(
            f"WGSL target cannot initialize pointer local '{name}' with the "
            "address of an expression that is not a variable"
        )
    target_type = codegen.expression_type(target)
    if isinstance(target_type, str) and target_type.startswith("atomic<"):
        raise WGSLRefusal(
            f"WGSL target cannot point pointer local '{name}' at an atomic "
            "value: WGSL reads and writes atomics only through atomic built-ins"
        )
    if not _is_addressable_root(codegen, node.name):
        raise WGSLRefusal(
            f"WGSL target cannot take the address of '{node.name}' for pointer "
            f"local '{name}': WGSL takes the address only of a variable, and "
            f"'{node.name}' is a let value, a parameter, a constant, a uniform "
            "or a resource handle"
        )
    return node.name


def _refuse_vector_component(name):
    raise WGSLRefusal(
        f"WGSL target cannot point pointer local '{name}' at a vector "
        "component: WGSL cannot take the address of a vector component"
    )


def _is_vector(codegen, expression):
    expression_type = codegen.expression_type(expression)
    if expression_type is None or isinstance(expression_type, PointerType):
        return False
    return codegen.vector_shape(expression_type) is not None


def _is_addressable_root(codegen, root_name):
    mutable = codegen.local_mutability(root_name)
    if mutable is not None:
        return mutable
    if codegen.is_pointer_identifier(root_name):
        return True
    if codegen.is_current_parameter_name(root_name):
        plan = codegen.current_parameter_plan()
        return root_name in plan["references"] or root_name in plan["copies"]
    if codegen.is_local_identifier(root_name):
        return False
    return root_name in getattr(codegen, "_addressable_module_names", ())


def _is_dereference(node, name):
    return (
        isinstance(node, UnaryOpNode)
        and str(node.operator) == "*"
        and not node.is_postfix
        and isinstance(node.operand, IdentifierNode)
        and node.operand.name == name
    )


def _dereferenced_root(expression, name):
    """The ``*name`` node an l-value designates part of, or ``None``."""
    while isinstance(expression, (MemberAccessNode, ArrayAccessNode, SwizzleNode)):
        if isinstance(expression, MemberAccessNode):
            expression = expression.object_expr
        elif isinstance(expression, ArrayAccessNode):
            expression = expression.array_expr
        else:
            expression = expression.vector_expr
    return expression if _is_dereference(expression, name) else None


def _validate_pointer_uses(codegen, name, root_name, following):
    identifiers = set()
    dereferenced = set()
    writes = False
    for node in walk_ast_values(list(following)):
        if isinstance(node, VariableNode) and node.name == name:
            raise WGSLRefusal(
                f"WGSL target cannot lower pointer local '{name}': a nested "
                "declaration in its scope reuses the name"
            )
        if isinstance(node, IdentifierNode) and node.name == name:
            identifiers.add(id(node))
        elif _is_dereference(node, name):
            dereferenced.add(id(node.operand))
        if isinstance(node, AssignmentNode):
            writes = writes or _dereferenced_root(node.target, name) is not None
        elif isinstance(node, UnaryOpNode) and str(node.operator) in (
            _INCREMENT_OPERATORS
        ):
            writes = writes or _dereferenced_root(node.operand, name) is not None
        elif isinstance(node, FunctionCallNode):
            for index in codegen.reference_argument_indices_by_name(node):
                if index < len(node.arguments):
                    argument = node.arguments[index]
                    writes = writes or _dereferenced_root(argument, name) is not None
    if identifiers - dereferenced:
        raise WGSLRefusal(
            f"WGSL target cannot lower pointer local '{name}': WGSL pointers "
            "are only dereferenced; re-pointing, pointer arithmetic or "
            "indexing, comparison and passing the pointer on have no WGSL form"
        )
    if writes and codegen.storage_access_mode(root_name) == "read":
        raise WGSLRefusal(
            f"WGSL target cannot write through pointer local '{name}': it "
            f"points into read-only storage buffer {root_name}"
        )
