"""Member functions of HIP structs and classes as CrossGL free functions.

CrossGL structs have data members only. The CUDA importer lowers each member function a program calls to a free
function ``<Struct>_<method>`` whose first parameter is the receiver (:mod:`crosstl.backend.CUDA.struct_methods`);
HIP is the same C++ dialect, so :func:`lower_hip_struct_methods` runs that pass over a HIP program.

The HIP parser keeps a whole program as one ``statements`` list and every member function among a record's
``members``, so the adapter gives the pass the view it expects:

* the named top-level structs, classes and unions with a body. The pass lowers the member functions of those that
  are not unions, have no base class and no template header; a derived record's member functions can read the
  members of its bases, which the pass does not track. The others are given to the pass with no member functions,
  so a call of one of theirs, such as ``Box<float>::get``, is refused instead of written through;
* the member functions it may lower: those with a plain name and a body that are not static, virtual, friend,
  explicit, templates, constructors, destructors or conversion operators;
* every member-function name of every record, so that a member call the pass cannot resolve to a lowered function
  is refused (:class:`~crosstl.backend.import_refusals.HipImportRefused`) rather than written through;
* the HIP node kinds a member function body may hold, and a receiver in the HIP parameter shape.

Each free function goes right after its record in ``statements``, which is where a member function's body is in
scope ([basic.scope.class]). The member functions stay among the record's members, so a call the pass does not
rewrite, for example in a function the view does not hold, is still refused by the record checks.
"""

from __future__ import annotations

import re

from ...CUDA.struct_methods import StructMethodDialect, lower_struct_methods
from ..HipAst import (
    ASTNode,
    AtomicOperationNode,
    EnumNode,
    FunctionNode,
    HipBuiltinNode,
    KernelNode,
    StructNode,
    TypeAliasNode,
    VariableNode,
)

__all__ = ["lower_hip_struct_methods"]

_PLAIN_NAME = re.compile(r"[A-Za-z_]\w*")
_SKIPPED_QUALIFIERS = frozenset({"static", "virtual", "friend", "explicit"})
_SPECIAL_MEMBER_FLAGS = (
    "is_record_constructor",
    "is_record_destructor",
    "is_conversion_operator",
    "is_using_declaration",
    "is_inherited_constructor",
)

HIP_STRUCT_METHODS = StructMethodDialect(
    language="HIP",
    node_fields={AtomicOperationNode: ("args",), HipBuiltinNode: ()},
    make_receiver=lambda type_text, name: {"type": type_text, "name": name},
)


class _ProgramView(ASTNode):
    """The shape of a parsed CUDA program, over the statements of a HIP program."""

    def __init__(self, statements, structs, method_names):
        self.statements = statements
        self.structs = structs
        self.struct_method_names = method_names
        self.functions = [
            statement
            for statement in statements
            if isinstance(statement, FunctionNode)
            and not isinstance(statement, KernelNode)
        ]
        self.kernels = [
            statement for statement in statements if isinstance(statement, KernelNode)
        ]
        self.typedefs = [
            statement
            for statement in statements
            if isinstance(statement, TypeAliasNode)
        ]
        self.global_variables = [
            statement for statement in statements if isinstance(statement, VariableNode)
        ]


def _is_member_function(member):
    return (
        isinstance(member, FunctionNode)
        and not isinstance(member, KernelNode)
        and isinstance(member.name, str)
        and _PLAIN_NAME.fullmatch(member.name) is not None
        and not any(getattr(member, flag, False) for flag in _SPECIAL_MEMBER_FLAGS)
    )


def _is_lowerable_member_function(member):
    return (
        _is_member_function(member)
        and getattr(member, "body", None) is not None
        and not getattr(member, "template_parameters", None)
        and not set(getattr(member, "qualifiers", None) or []) & _SKIPPED_QUALIFIERS
    )


def _is_record(statement):
    return (
        isinstance(statement, StructNode)
        and not isinstance(statement, EnumNode)
        and isinstance(statement.name, str)
        and _PLAIN_NAME.fullmatch(statement.name) is not None
        and getattr(statement, "has_body", True)
    )


def _lowers_member_functions(record):
    return not (
        getattr(record, "is_union", False)
        or getattr(record, "is_record_template", False)
        or getattr(record, "base_classes", None)
    )


def lower_hip_struct_methods(program, refuse):
    """Lower the called member functions of the HIP ``program`` in place; ``refuse(reason)`` records each refusal."""
    statements = getattr(program, "statements", None)
    if not isinstance(statements, list):
        return
    method_names = sorted(
        {
            member.name
            for statement in statements
            if isinstance(statement, StructNode)
            for member in getattr(statement, "members", None) or []
            if _is_member_function(member)
        }
    )
    if not method_names:
        return
    records, function_count = [], 0
    for statement in statements:
        if isinstance(statement, FunctionNode) and not isinstance(
            statement, KernelNode
        ):
            function_count += 1
        elif _is_record(statement) and statement.name not in {
            record.name for record in records
        }:
            # A record whose member functions stay unlowered is still given to the pass, with none, so that a
            # call of one is refused rather than kept as a member call.
            statement.methods = [
                member
                for member in statement.members or []
                if _lowers_member_functions(statement)
                and _is_lowerable_member_function(member)
            ]
            statement.function_index = function_count
            records.append(statement)
    view = _ProgramView(statements, records, method_names)
    try:
        inserted = lower_struct_methods(view, refuse, HIP_STRUCT_METHODS)
    finally:
        for record in records:
            del record.methods, record.function_index
    if not inserted:
        return
    lowered = []
    for statement in statements:
        lowered.append(statement)
        if any(statement is record for record in records):
            lowered.extend(inserted.get(statement.name, []))
    program.statements = lowered
