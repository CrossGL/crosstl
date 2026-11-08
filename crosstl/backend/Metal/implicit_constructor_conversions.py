"""Implicit conversions through converting constructors outside declarations.

MSL is C++14: a non-explicit constructor of a class callable with one argument is a converting constructor
([class.conv.ctor]), and copy-initialization applies it wherever a value of another type initializes an object of
the class type. A declaration ``T t = v;`` and a ``return v;`` already lower through the importer's constructor
factory. This module applies the same conversion to the other copy-initialization contexts:

* a value argument (or a ``const`` reference argument) of a user function, lowered method or overloaded operator
  ([dcl.init]/15, [over.match.funcs]);
* the right operand of a simple assignment to a class object, which initializes the parameter of the implicitly
  declared copy assignment operator ([class.copy.assign], [over.ass]).

CrossGL structs have no constructors, so leaving ``take(v)`` or ``out[i] = v`` unconverted hands every generator a
type mismatch. A struct that declares its own ``operator=`` assigns through that operator instead of the converting
constructor; the importer does not lower member operators, so that assignment is refused.
"""

from __future__ import annotations

import re

from .MetalAst import InitializerListNode, VariableNode


class MetalAssignmentOperatorError(ValueError):
    """Raised when an assignment would run a user-declared Metal ``operator=``."""

    project_diagnostic_code = (
        "project.translate.metal-assignment-operator-unrepresentable"
    )
    missing_capabilities = ("metal.member-assignment-operator-lowering",)

    def __init__(self, owner, value_type, source_location=None):
        self.owner = owner
        self.value_type = value_type
        self.source_location = source_location
        super().__init__(
            f"Cannot assign a {value_type} value to Metal struct {owner}: the struct declares operator=, "
            "and CrossGL has no member assignment operators to run in place of the implicit conversion"
        )


def _class_target_type(codegen, target_type):
    """``target_type`` when it names a class with source constructors, else ``None``."""
    if target_type is None:
        return None
    if codegen.metal_constructor_contract(target_type) is None:
        return None
    return target_type


_CONDITIONAL_TYPE = re.compile(
    r"^(?:typename\s+)?(?:::\s*)?(?:metal\s*::\s*)?conditional(?:_t)?\s*<"
)


def _selected_value_type(codegen, value_type):
    """``value_type`` with a ``conditional_t`` selection replaced by the branch it names.

    A local declared through ``using loader_t = conditional_t<C, A, B>`` has the type of the selected branch, and
    the class target of a conversion is resolved the same way (``metal_constructor_contract``). A selection whose
    condition is not known yet names no known type, so no conversion is decided from it (``None``).
    """
    if value_type is None or not _CONDITIONAL_TYPE.match(value_type):
        return value_type
    selected = codegen.resolve_conditional_type(value_type, require_concrete=True)
    if selected is None:
        return None
    return codegen.metal_source_overload_value_type(selected)


def converting_constructor_value(
    codegen, target_type, value, is_main, *, assignment=False
):
    """Render ``value`` copy-initializing a ``target_type`` object, or ``None`` when no conversion applies.

    No conversion applies when ``target_type`` is not a class with source constructors, when ``value`` is a braced
    list (the callers' list-initialization paths own those) or when the value already has the class type. A
    constructor that cannot be selected leaves the value to the caller's existing path.
    """
    target_type = _class_target_type(codegen, target_type)
    if target_type is None or isinstance(value, InitializerListNode):
        return None
    value_type = _selected_value_type(
        codegen,
        codegen.metal_source_overload_value_type(codegen.expression_metal_type(value)),
    )
    if value_type is None:
        return None
    target_identity = codegen.metal_source_overload_type_identity(target_type)
    if target_identity is None:
        return None
    if codegen.metal_source_overload_type_identity(value_type) == target_identity:
        return None
    struct_node = codegen.metal_constructor_contract(target_type)[0]
    source_location = getattr(value, "source_location", None)
    if assignment and getattr(struct_node, "declares_assignment_operator", False):
        raise MetalAssignmentOperatorError(
            getattr(struct_node, "name", target_type), value_type, source_location
        )
    return codegen.generate_explicit_constructor_call(
        target_type,
        [value],
        is_main,
        source_location,
        allow_explicit=False,
    )


def _reference_is_const(codegen, parameter_type, parameter):
    words = set(re.findall(r"[A-Za-z_]\w*", str(parameter_type)))
    return bool(
        {"const", "constant"} & words
        or codegen.readonly_parameter(parameter, getattr(parameter, "qualifiers", None))
    )


def converting_constructor_argument(codegen, argument, parameter, is_main):
    """Render ``argument`` initializing ``parameter`` through a converting constructor, or ``None``.

    A value parameter and a ``const`` reference parameter accept the converted temporary; a non-``const``
    reference cannot bind one ([dcl.init.ref]), so it keeps the caller's rendering.
    """
    parameter_type = codegen.metal_source_overload_parameter_type(parameter)
    if parameter_type is None:
        return None
    referenced = codegen.reference_element_type(parameter_type)
    if referenced is not None:
        if not _reference_is_const(codegen, parameter_type, parameter):
            return None
        if codegen.pointer_element_type(referenced) is not None:
            return None
        parameter_type = referenced
    elif codegen.pointer_element_type(parameter_type) is not None:
        return None
    return converting_constructor_value(codegen, parameter_type, argument, is_main)


def convert_user_call_arguments(codegen, function_name, expression, rendered, is_main):
    """Fill ``rendered`` with converted arguments of a resolved user function call.

    ``rendered`` maps argument indices to text the caller already produced; those indices are left alone. Only a
    call whose overload set has a parameter of a class type with source constructors is resolved.
    """
    candidates = codegen.metal_user_function_overloads(
        function_name, source_offset=codegen.alias_source_offset(expression)
    )
    if not any(
        _class_parameter_candidate(codegen, parameter)
        for candidate in candidates
        for parameter in getattr(candidate, "params", None) or []
    ):
        return
    selected = codegen.selected_metal_callable(expression) or _sole_candidate(
        codegen, candidates, expression
    )
    if selected is None:
        return
    for index, parameter in enumerate(getattr(selected, "params", None) or []):
        if index in rendered or index >= len(expression.args):
            continue
        if index == 0 and getattr(parameter, "name", None) == "self":
            # The receiver of a lowered method is its implicit object parameter, which no user-defined
            # conversion initializes ([over.match.funcs]/5).
            continue
        converted = converting_constructor_argument(
            codegen, expression.args[index], parameter, is_main
        )
        if converted is not None:
            rendered[index] = converted


def _sole_candidate(codegen, candidates, expression):
    """The only non-template overload of the callee when its arity matches the call.

    The importer's overload ranking has no user-defined conversion sequences, so a call that needs one ranks no
    candidate; the call still names that single function, which is the one C++ selects when it is viable.
    """
    signatures = {codegen.metal_function_source_signature(item) for item in candidates}
    if len(signatures) != 1:
        return None
    candidate = candidates[0]
    if getattr(candidate, "template_parameters", None):
        return None
    if len(getattr(candidate, "params", None) or []) != len(expression.args):
        return None
    return candidate


def _class_parameter_candidate(codegen, parameter):
    parameter_type = codegen.metal_source_overload_parameter_type(parameter)
    if parameter_type is None:
        return False
    referenced = codegen.reference_element_type(parameter_type)
    return _class_target_type(codegen, referenced or parameter_type) is not None


def converted_assignment_value(codegen, node, is_main):
    """Render the right operand of ``class_object = value`` through a converting constructor, or ``None``.

    Declarations (``T t = v``) keep the initializer path, and compound assignments are operator calls, not
    copy-initialization.
    """
    if getattr(node, "operator", None) != "=":
        return None
    if isinstance(node.left, VariableNode) and getattr(node.left, "vtype", None):
        return None
    return converting_constructor_value(
        codegen,
        codegen.expression_metal_type(node.left),
        node.right,
        is_main,
        assignment=True,
    )
