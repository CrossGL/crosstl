"""CrossGL emission of HLSL struct member functions.

CrossGL has no member functions, so each HLSL struct method becomes a free
function that takes its receiver as an explicit parameter. These members find
the method for a function, resolve the receiver and the fields and static
members visible inside the method body, rewrite member calls into calls of the
lowered function, and report which call arguments a callee writes.
"""

from ..common_ast import FunctionNode, MemberAccessNode, VariableNode
from .struct_methods import RECEIVER_PARAMETER


class HLSLStructMethodCallMixin:
    """HLSL struct-method lowering members of ``HLSLToCrossGLConverter``."""

    # Intrinsics that write through argument positions, for receiver analysis.
    HLSL_WRITING_INTRINSIC_ARGUMENTS = {
        "InterlockedAdd": {0, 2},
        "InterlockedAnd": {0, 2},
        "InterlockedOr": {0, 2},
        "InterlockedXor": {0, 2},
        "InterlockedMin": {0, 2},
        "InterlockedMax": {0, 2},
        "InterlockedExchange": {0, 2},
        "InterlockedCompareExchange": {0, 3},
        "InterlockedCompareStore": {0},
        "sincos": {1, 2},
        "modf": {1},
        "frexp": {1},
    }

    def struct_method_functions(self, table):
        """Free-function copies of member functions defined in struct bodies."""
        functions = []
        for owner_methods in table.values():
            for method in owner_methods.values():
                if not method.inline or method.definition is None:
                    continue
                definition = method.definition
                function = FunctionNode(
                    definition.return_type,
                    method.qualified_name,
                    list(getattr(definition, "params", []) or []),
                    definition.body,
                    qualifiers=list(getattr(definition, "qualifiers", []) or []),
                    attributes=list(getattr(definition, "attributes", []) or []),
                )
                function.qualifier = getattr(definition, "qualifier", None)
                function.semantic = getattr(definition, "semantic", None)
                function.array_sizes = list(
                    getattr(definition, "array_sizes", []) or []
                )
                function.is_prototype = False
                functions.append(function)
                method.definition = function
        return functions

    def call_writes_argument(self, call, index, functions):
        """Whether a call may write through its argument at ``index``."""
        name = getattr(call, "name", None)
        if not isinstance(name, str):
            return False
        written = self.HLSL_WRITING_INTRINSIC_ARGUMENTS.get(name)
        if written is not None and name not in self.source_function_names:
            return index in written
        for function in functions or []:
            if getattr(function, "name", None) != name:
                continue
            params = getattr(function, "params", []) or []
            if index < len(params) and {
                str(qualifier).lower()
                for qualifier in getattr(params[index], "qualifiers", []) or []
            } & {"out", "inout"}:
                return True
        return False

    def struct_method_for_function(self, func):
        name = getattr(func, "name", None)
        if not isinstance(name, str) or "::" not in name:
            return None
        owner, member = name.rsplit("::", 1)
        method = getattr(self, "struct_methods", {}).get(owner, {}).get(member)
        if method is None or method.definition is not func:
            return None
        return method

    def struct_method_receiver_parameter(self, method):
        qualifier = "inout" if method.mutates_receiver else "in"
        return VariableNode(method.owner, RECEIVER_PARAMETER, qualifiers=[qualifier])

    def current_struct_method_fields(self):
        method = getattr(self, "current_struct_method", None)
        if method is None or method.static:
            return set()
        return set(self.struct_member_types.get(method.owner, {})) - (
            self.struct_static_member_names(method.owner)
        )

    def struct_static_member_names(self, owner):
        struct = getattr(self, "structs_by_name", {}).get(owner)
        return {
            member.name
            for member in getattr(struct, "members", []) or []
            if isinstance(member, VariableNode)
            and any(
                str(qualifier).lower() == "static"
                for qualifier in getattr(member, "qualifiers", []) or []
            )
        }

    def struct_method_context_identifier(self, name):
        """Render ``this`` and unshadowed members inside a member function."""
        method = getattr(self, "current_struct_method", None)
        if method is None or not isinstance(name, str):
            return None
        if name in self.current_struct_method_locals:
            return None
        if name in self.struct_static_member_names(method.owner):
            return self.render_identifier(f"{method.owner}::{name}")
        if method.static:
            return None
        if name == "this":
            return RECEIVER_PARAMETER
        if name in self.current_struct_method_fields():
            return f"{RECEIVER_PARAMETER}.{self.render_identifier(name)}"
        return None

    def struct_method_context_name_type(self, name):
        method = getattr(self, "current_struct_method", None)
        if method is None or method.static or not isinstance(name, str):
            return None
        if name == "this":
            return method.owner
        if name in self.current_struct_method_locals:
            return None
        return self.struct_member_types.get(method.owner, {}).get(name)

    def struct_method_call_expression(self, expr, is_main):
        """Lower calls of struct member functions to free-function calls."""
        table = getattr(self, "struct_methods", None)
        if not table:
            return None
        name = expr.name
        receiver = None
        if isinstance(name, MemberAccessNode):
            member = name.member
            if not isinstance(member, str) or "<" in member:
                return None
            owner = self.raw_type_base(self.struct_method_receiver_type(name.object))
            method = table.get(owner, {}).get(member)
            if method is None:
                return None
            if not method.static:
                receiver = self.generate_expression(name.object, is_main)
        elif isinstance(name, str):
            current = getattr(self, "current_struct_method", None)
            if "::" in name:
                owner, member = name.rsplit("::", 1)
                method = table.get(owner, {}).get(member)
                if method is None or not method.static:
                    return None
            elif current is not None and name not in self.current_struct_method_locals:
                method = table.get(current.owner, {}).get(name)
                if method is None:
                    return None
                if not method.static:
                    if current.static:
                        raise ValueError(
                            f"Static member function {current.qualified_name} "
                            f"calls instance member function {method.qualified_name}"
                        )
                    receiver = RECEIVER_PARAMETER
            else:
                return None
        else:
            return None
        args = [self.generate_expression(arg, is_main) for arg in expr.args]
        if receiver is not None:
            args.insert(0, receiver)
        callee = self.render_function_identifier(method.qualified_name)
        return f"{callee}({', '.join(args)})"

    def struct_method_receiver_type(self, expr):
        context_type = (
            self.struct_method_context_name_type(expr)
            if isinstance(expr, str)
            else None
        )
        if context_type is not None:
            return context_type
        return self.expression_value_raw_type(expr)
