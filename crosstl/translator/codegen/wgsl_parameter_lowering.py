"""Lower written and by-reference function parameters for WGSL.

WGSL formal parameters are immutable values (WGSL section 11.1): assigning to
one, or applying ``++``/``--`` to it, is a shader-creation error. C-family
sources write to value parameters freely, and pass ``inout``/``out``
parameters by copy-in/copy-out.

* A value parameter that the body writes is received under a fresh name and
  copied into a ``var`` of the source name at the top of the body, so the
  writes stay local to the call, as in the source.
* An ``inout`` or ``out`` parameter of a helper function becomes a
  ``ptr<function, T>`` parameter. Reads and writes in the body go through
  ``(*p)``, and each call site passes ``&v``.

Pointer semantics equal copy-in/copy-out when the callee cannot reach the
argument any other way. Core WGSL already demands that: the argument of a
``ptr<function, T>`` parameter must be a whole function-scope variable or a
pointer parameter (WGSL section 11.3.1, the restriction that the
``unrestricted_pointer_parameters`` extension lifts), and two pointer
arguments with one root variable may not be passed together when either is
written (WGSL section 11.4, alias analysis). Every other argument form raises
``ValueError`` instead of being passed by value, which would drop the
callee's writes. So do ``inout``/``out`` parameters of entry points, because
WGSL entry-point parameters are pipeline inputs (WGSL section 13.3.1).
"""

from __future__ import annotations

import copy

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    ParameterNode,
    PointerType,
    SwizzleNode,
    UnaryOpNode,
    VariableNode,
)

REFERENCE_PARAMETER_QUALIFIERS = ("inout", "out")
INCREMENT_OPERATORS = frozenset({"++", "--"})


def reference_parameter_qualifier(parameter):
    """The ``inout``/``out`` qualifier of a parameter, or ``None``."""
    qualifiers = {
        str(qualifier).strip().lower()
        for qualifier in getattr(parameter, "qualifiers", None) or ()
    }
    for qualifier in REFERENCE_PARAMETER_QUALIFIERS:
        if qualifier in qualifiers:
            return qualifier
    return None


def lvalue_root(expr):
    """The variable an l-value expression designates part of, or ``None``."""
    while True:
        if isinstance(expr, IdentifierNode):
            return expr
        if isinstance(expr, MemberAccessNode):
            expr = expr.object_expr
        elif isinstance(expr, ArrayAccessNode):
            expr = expr.array_expr
        elif isinstance(expr, SwizzleNode):
            expr = expr.vector_expr
        else:
            return None


def _walk(node):
    walk = getattr(node, "walk", None)
    if walk is None:
        return ()
    return walk()


class WGSLParameterLoweringMixin:
    """Parameter plans for ``WGSLCodeGen``; see the module docstring."""

    def push_parameter_plan(self, function, entry_point=False):
        plan = self.build_parameter_plan(function, entry_point=entry_point)
        self._parameter_plan_stack.append(plan)
        return plan

    def pop_parameter_plan(self):
        self._parameter_plan_stack.pop()

    def current_parameter_plan(self):
        plans = self._parameter_plan_stack
        return plans[-1] if plans else None

    def build_parameter_plan(self, function, entry_point=False):
        parameters = [
            parameter
            for parameter in getattr(function, "parameters", None) or ()
            if isinstance(parameter, ParameterNode)
        ]
        function_name = getattr(function, "name", "") or "<anonymous>"
        body_nodes = list(_walk(getattr(function, "body", None)))
        references = {}
        for parameter in parameters:
            qualifier = reference_parameter_qualifier(parameter)
            if qualifier is None:
                continue
            self.validate_reference_parameter(
                function_name, parameter, qualifier, entry_point, body_nodes
            )
            references[parameter.name] = parameter
        written = self.written_parameter_names(body_nodes)
        copies = {}
        if written:
            used_names = self.parameter_plan_used_names(parameters, body_nodes)
            for parameter in parameters:
                if parameter.name in references or parameter.name not in written:
                    continue
                if not self.is_plain_value_parameter(parameter):
                    continue
                incoming = self.unique_wgsl_identifier(
                    f"{self.identifier_name(parameter.name)}_in", used_names
                )
                used_names.add(incoming)
                copies[parameter.name] = incoming
        return {
            "references": references,
            "copies": copies,
            "parameter_names": {parameter.name for parameter in parameters},
        }

    def validate_reference_parameter(
        self, function_name, parameter, qualifier, entry_point, body_nodes
    ):
        if entry_point:
            raise ValueError(
                f"WGSL target cannot lower {qualifier} parameter "
                f"{parameter.name} of entry point {function_name}: WGSL "
                "entry-point parameters are pipeline inputs (WGSL 13.3.1); "
                "return stage outputs from the entry point instead"
            )
        if not self.is_plain_value_parameter(parameter):
            raise ValueError(
                f"WGSL target cannot pass {qualifier} parameter "
                f"{parameter.name} of {function_name} by reference: only "
                "plain value types can become ptr<function, T> parameters"
            )
        for node in body_nodes:
            if isinstance(node, VariableNode) and node.name == parameter.name:
                raise ValueError(
                    f"WGSL target cannot lower {qualifier} parameter "
                    f"{parameter.name} of {function_name}: a local variable "
                    "in the body shadows it"
                )

    def written_parameter_names(self, body_nodes):
        """Names whose variable the body assigns, increments or passes by reference."""
        names = set()
        for node in body_nodes:
            if isinstance(node, AssignmentNode):
                root = lvalue_root(node.target)
            elif (
                isinstance(node, UnaryOpNode)
                and str(node.operator) in INCREMENT_OPERATORS
            ):
                root = lvalue_root(node.operand)
            elif isinstance(node, FunctionCallNode):
                for index in self.reference_argument_indices_by_name(node):
                    if index < len(node.arguments):
                        root = lvalue_root(node.arguments[index])
                        if root is not None:
                            names.add(root.name)
                continue
            else:
                continue
            if root is not None:
                names.add(root.name)
        return names

    def reference_argument_indices_by_name(self, call):
        """Argument positions that any overload of the callee takes by reference."""
        if not isinstance(call.function, IdentifierNode):
            return ()
        overloads = getattr(self, "_function_overloads_by_name", {}).get(
            call.function.name, ()
        )
        indices = set()
        for function in overloads:
            for index, parameter in enumerate(getattr(function, "parameters", ())):
                if reference_parameter_qualifier(parameter) is not None:
                    indices.add(index)
        return sorted(indices)

    def parameter_plan_used_names(self, parameters, body_nodes):
        used = set(getattr(self, "_type_identifier_names", {}).values())
        used.update(getattr(self, "_function_identifier_names", {}).values())
        used.update(getattr(self, "_module_identifier_names", {}).values())
        used.update(getattr(self, "_module_identifier_names", {}).keys())
        for parameter in parameters:
            used.add(parameter.name)
            used.add(self.identifier_name(parameter.name))
        for node in body_nodes:
            if isinstance(node, (IdentifierNode, VariableNode)):
                used.add(str(node.name))
                used.add(self.safe_wgsl_identifier(node.name))
        return used

    def is_plain_value_parameter(self, parameter):
        """Whether a parameter is a value WGSL can hold in a ``var``."""
        param_type = getattr(parameter, "param_type", None)
        if param_type is None or isinstance(param_type, PointerType):
            return False
        element_type = self.array_element_type(param_type) or param_type
        for candidate in (param_type, element_type):
            type_name = self.resource_type_name(candidate)
            if type_name is not None and self.is_resource_type_name(type_name):
                return False
            if self.sampled_texture_type(candidate) is not None:
                return False
            if self.is_sampler_type(candidate):
                return False
            if self.structured_buffer_element_type(candidate) is not None:
                return False
        if self.sampled_texture_array_info(param_type) is not None:
            return False
        if self.is_glsl_buffer_block_parameter(parameter):
            return False
        return not self.resource_member_parameter_declarations(
            parameter.name, param_type
        )

    def generate_planned_parameter(self, parameter, include_attributes):
        plan = self.current_parameter_plan()
        if plan is not None:
            if parameter.name in plan["references"]:
                return (
                    f"{self.identifier_name(parameter.name)}: ptr<function, "
                    f"{self.type_name_string(parameter.param_type)}>"
                )
            incoming = plan["copies"].get(parameter.name)
            if incoming is not None:
                renamed = copy.copy(parameter)
                renamed.name = incoming
                return self.generate_parameter(
                    renamed, include_attributes=include_attributes
                )
        return self.generate_parameter(parameter, include_attributes=include_attributes)

    def prepend_parameter_copies(self, body_text, function):
        plan = self.current_parameter_plan()
        if plan is None or not plan["copies"]:
            return body_text
        lines = body_text.split("\n")
        if not lines or not lines[0].rstrip().endswith("{"):
            raise ValueError(
                "WGSL target could not place parameter copies in the body of "
                f"{getattr(function, 'name', '<anonymous>')}"
            )
        copies = []
        for parameter in getattr(function, "parameters", None) or ():
            incoming = plan["copies"].get(getattr(parameter, "name", None))
            if incoming is None:
                continue
            copies.append(
                f"    var {self.identifier_name(parameter.name)}: "
                f"{self.type_name_string(parameter.param_type)} = {incoming};"
            )
        return "\n".join(lines[:1] + copies + lines[1:])

    def is_reference_parameter_name(self, name):
        plan = self.current_parameter_plan()
        return plan is not None and name in plan["references"]

    def reference_parameter_expression(self, name):
        return f"(*{self.identifier_name(name)})"

    def reference_argument_expression(self, argument, parameter, function):
        """``&v`` for a whole local variable, or a pointer parameter itself."""
        qualifier = reference_parameter_qualifier(parameter)
        callee = getattr(function, "name", "") or "<anonymous>"
        if isinstance(argument, IdentifierNode):
            name = argument.name
            if self.is_reference_parameter_name(name):
                return self.identifier_name(name)
            plan = self.current_parameter_plan()
            is_parameter = plan is not None and self.is_current_parameter_name(name)
            if self.is_local_identifier(name) and (
                not is_parameter or name in plan["copies"]
            ):
                return f"&{self.identifier_name(name)}"
            where = "a module-scope variable" if not is_parameter else "a parameter"
        else:
            where = "not a whole local variable"
        raise ValueError(
            f"WGSL target cannot pass {self.argument_display(argument)} to "
            f"{qualifier} parameter {parameter.name} of {callee}: the argument "
            f"is {where}, and a ptr<function, T> argument must be a whole "
            "function-scope variable (WGSL 11.3.1); copy it into a local "
            "variable around the call"
        )

    def is_current_parameter_name(self, name):
        plan = self.current_parameter_plan()
        return plan is not None and name in plan["parameter_names"]

    def argument_display(self, argument):
        root = lvalue_root(argument)
        if isinstance(argument, IdentifierNode):
            return f"'{argument.name}'"
        if root is not None:
            return f"a part of '{root.name}'"
        return "an expression"

    def validate_reference_argument_aliasing(self, function, arguments):
        """Reject two by-reference arguments rooted at one variable (WGSL 11.4)."""
        seen = {}
        callee = getattr(function, "name", "") or "<anonymous>"
        for parameter, argument in zip(
            getattr(function, "parameters", None) or (), arguments
        ):
            if reference_parameter_qualifier(parameter) is None:
                continue
            root = lvalue_root(argument)
            if root is None:
                continue
            if root.name in seen:
                raise ValueError(
                    f"WGSL target cannot pass '{root.name}' to both "
                    f"{seen[root.name]} and {parameter.name} of {callee}: "
                    "WGSL forbids aliased pointer arguments (WGSL 11.4)"
                )
            seen[root.name] = parameter.name
