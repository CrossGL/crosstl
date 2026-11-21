"""Function returns, record expression types and record call, operator and conversion validation in the HIP to CrossGL converter."""

import re

from ..HipAst import (
    ArrayAccessNode,
    CastNode,
    FunctionCallNode,
    FunctionNode,
    MemberAccessNode,
    ReturnNode,
    TernaryOpNode,
    UnaryOpNode,
)


class HipRecordCallMixin:
    """Record call members of ``HipToCrossGLConverter``."""

    def register_function_return_nodes(self, function):
        visited = set()

        def walk(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return
            marker = id(value)
            if marker in visited:
                return
            visited.add(marker)
            if isinstance(value, ReturnNode):
                self.return_type_by_node_id[id(value)] = getattr(
                    function, "return_type", "void"
                )
            if isinstance(value, FunctionNode) and value is not function:
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    walk(item)
                return
            if isinstance(value, dict):
                for item in value.values():
                    walk(item)
                return
            for child in getattr(value, "__dict__", {}).values():
                walk(child)

        walk(getattr(function, "body", None))

    def function_definition_candidates(self, name, argument_count):
        if not isinstance(name, str):
            return []
        names = [name, name[2:] if name.startswith("::") else name]
        base_name, template_args = self.parse_cpp_template(names[-1])
        if template_args:
            names.append(base_name)
        candidates = []
        seen = set()
        for candidate_name in names:
            for candidate in self.function_definitions.get(candidate_name, []):
                if id(candidate) in seen:
                    continue
                seen.add(id(candidate))
                if len(getattr(candidate, "params", None) or []) == argument_count:
                    candidates.append(candidate)
        return candidates

    def function_return_record_definition(self, name, argument_count):
        definitions = {
            id(definition): definition
            for function in self.function_definition_candidates(name, argument_count)
            for definition in [
                self.record_definition_from_type(getattr(function, "return_type", ""))
            ]
            if definition is not None
        }
        if len(definitions) == 1:
            return next(iter(definitions.values()))
        return None

    def infer_record_expression_definition(self, expression):
        if isinstance(expression, str):
            return self.record_definition_from_type(
                self.lookup_variable_type(expression) or ""
            )
        if isinstance(expression, MemberAccessNode):
            parent = self.infer_record_expression_definition(expression.object)
            member = self.record_storage_member(parent, expression.member)
            if member is None:
                return None
            member_type = self.resolve_record_member_type_alias(
                getattr(member, "vtype", "")
            )
            return self.record_definition_from_type(
                self.convert_hip_type_to_crossgl(member_type)
            )
        if isinstance(expression, ArrayAccessNode):
            return self.infer_record_expression_definition(expression.array)
        if isinstance(expression, CastNode):
            return self.record_definition_from_type(expression.target_type)
        if isinstance(expression, UnaryOpNode):
            return self.infer_record_expression_definition(expression.operand)
        if isinstance(expression, TernaryOpNode):
            left = self.infer_record_expression_definition(expression.true_expr)
            right = self.infer_record_expression_definition(expression.false_expr)
            return left if left is not None and left is right else None
        if isinstance(expression, FunctionCallNode) and isinstance(
            expression.name, str
        ):
            definition = self.record_definition_from_type(expression.name)
            if definition is not None:
                return definition
            return self.function_return_record_definition(
                expression.name, len(getattr(expression, "args", None) or [])
            )
        return None

    def indexed_expression_type(self, type_name):
        if not isinstance(type_name, str):
            return None
        type_name = self.resolve_record_member_type_alias(type_name).strip()
        type_name = self.resolve_concrete_type_alias(type_name)
        if self.has_array_suffix(type_name):
            return re.sub(r"\[[^\]]*\]", "", type_name, count=1).strip()
        base_name, template_args = self.parse_cpp_template(type_name)
        if base_name in {"array", "ptr"} and template_args:
            return template_args[0]
        base_type, pointer_depth = self.split_pointer_declarators(type_name)
        if pointer_depth:
            return base_type
        return None

    def expression_declared_type(self, expression):
        if isinstance(expression, str):
            return self.lookup_variable_type(expression)
        if isinstance(expression, MemberAccessNode):
            parent = self.infer_record_expression_definition(expression.object)
            found = self.record_storage_member_info(parent, expression.member)
            if found is None:
                return None
            owner, member = found
            return self.resolve_record_layout_member_type(
                owner, getattr(member, "vtype", "")
            )
        if isinstance(expression, ArrayAccessNode):
            return self.indexed_expression_type(
                self.expression_declared_type(expression.array)
            )
        if isinstance(expression, CastNode):
            return expression.target_type
        if isinstance(expression, UnaryOpNode):
            operand_type = self.expression_declared_type(expression.operand)
            if operand_type is None:
                return None
            if expression.op == "&":
                return f"ptr<{operand_type}>"
            if expression.op == "*":
                return self.indexed_expression_type(operand_type)
            return operand_type
        if isinstance(expression, TernaryOpNode):
            left = self.expression_declared_type(expression.true_expr)
            right = self.expression_declared_type(expression.false_expr)
            return left if left is not None and left == right else None
        if isinstance(expression, FunctionCallNode) and isinstance(
            expression.name, str
        ):
            candidates = self.function_definition_candidates(
                expression.name, len(getattr(expression, "args", None) or [])
            )
            return_types = {
                str(getattr(candidate, "return_type", ""))
                for candidate in candidates
                if getattr(candidate, "return_type", None)
            }
            if len(return_types) == 1:
                return next(iter(return_types))
            if self.record_definition_from_type(expression.name) is not None:
                return expression.name
        return None

    def direct_record_definition_from_type(self, type_name):
        if not isinstance(type_name, str):
            return None
        type_name = self.resolve_record_member_type_alias(type_name).strip()
        type_name = self.resolve_concrete_type_alias(type_name)
        if self.has_array_suffix(type_name):
            return None
        base_name, template_args = self.parse_cpp_template(type_name)
        if base_name in {"array", "ptr"} and template_args:
            return None
        base_type, pointer_depth = self.split_pointer_declarators(type_name)
        if pointer_depth:
            return None
        type_name = re.sub(r"\s*&{1,2}\s*", " ", type_name).strip()
        return self.record_definition_from_type(type_name)

    def infer_direct_record_expression_definition(self, expression):
        return self.direct_record_definition_from_type(
            self.expression_declared_type(expression)
        )

    def record_is_derived_from(self, source, target, visiting=None):
        if source is None or target is None or source is target:
            return False
        visiting = set(visiting or ())
        if id(source) in visiting:
            return False
        visiting.add(id(source))
        for base in self.record_base_definitions(source):
            if base is target or self.record_is_derived_from(base, target, visiting):
                return True
        return False

    def parameter_type(self, parameter):
        if isinstance(parameter, dict):
            return parameter.get("type", "")
        return getattr(parameter, "vtype", "")

    def parameter_is_by_value(self, parameter):
        raw_type = str(self.parameter_type(parameter))
        return not any(marker in raw_type for marker in ("*", "&", "["))

    def validate_record_function_arguments(self, call):
        if not isinstance(getattr(call, "name", None), str):
            return
        args = list(getattr(call, "args", None) or [])
        candidates = self.function_definition_candidates(call.name, len(args))
        if not candidates:
            return
        candidate_issues = []
        for candidate in candidates:
            issues = []
            for parameter, argument in zip(candidate.params, args):
                raw_parameter_type = self.parameter_type(parameter)
                target = self.record_definition_from_type(raw_parameter_type)
                source = self.infer_record_expression_definition(argument)
                # Record reference parameters import as inout or by value (ReferenceParameters).
                for definition in (source, target):
                    issue = self.record_layout_issue(definition)
                    if issue is not None:
                        issues.append(issue)
                if source is None:
                    continue
                if target is not None:
                    if self.record_contains_tracked_union_storage(
                        source
                    ) or self.record_contains_tracked_union_storage(target):
                        issues.append(
                            ("union function argument active storage", source)
                        )
                    elif source is not target and self.record_is_derived_from(
                        source, target
                    ):
                        issues.append(("derived-to-base argument conversion", source))
                    elif source is target and self.parameter_is_by_value(parameter):
                        issue = self.record_copy_constructor_issue(source)
                        if issue is not None:
                            issues.append(issue)
                else:
                    issue = self.record_conversion_operator_issue(source)
                    if issue is not None:
                        issues.append(issue)
            if not issues:
                return
            candidate_issues.append(issues[0])
        if candidate_issues:
            self.raise_record_semantic(candidate_issues[0], f"call '{call.name}'")

    def validate_record_reference_return_call(self, call):
        if not isinstance(getattr(call, "name", None), str):
            return
        args = list(getattr(call, "args", None) or [])
        candidates = self.function_definition_candidates(call.name, len(args))
        if not candidates:
            return
        issues = []
        for candidate in candidates:
            return_type = getattr(candidate, "return_type", "")
            definition = self.record_definition_from_type(return_type)
            if not self.is_nonconst_record_reference_type(return_type, definition):
                return
            issues.append(("non-const record reference return aliasing", definition))
        if issues:
            self.raise_record_semantic(issues[0], f"call '{call.name}'")

    def validate_record_representation_query(self, call):
        name = getattr(call, "name", None)
        if name not in {
            "sizeof",
            "alignof",
            "__alignof",
            "__alignof__",
            "__builtin_offsetof",
        }:
            return
        args = list(getattr(call, "args", None) or [])
        if not args:
            return
        definition = None
        first = args[0]
        if isinstance(first, str):
            definition = self.record_definition_from_type(first)
        if definition is None:
            definition = self.infer_record_expression_definition(first)
        if definition is not None:
            self.raise_record_semantic(
                (f"source ABI query '{name}'", definition),
                f"call '{name}'",
            )

    def format_fixed_scalar_representation_query(self, call):
        """Fold source scalar ABI queries that have a platform-stable result."""
        name = getattr(call, "name", None)
        if name not in {"sizeof", "alignof", "__alignof", "__alignof__"}:
            return None
        args = list(getattr(call, "args", None) or [])
        if len(args) != 1 or not isinstance(args[0], str):
            return None

        operand = args[0].strip()
        variable_type = self.lookup_variable_type(operand)
        if variable_type is not None:
            type_name = str(variable_type).strip()
            if type_name not in self.HIP_FIXED_SCALAR_EXPRESSION_TYPES:
                return None
        else:
            type_name = self.resolve_record_member_type_alias(operand).strip()
            type_name = self.strip_elaborated_record_type_keyword(
                self.strip_type_qualifiers(type_name)
            ).strip()
            type_name = self.canonical_cpp_scalar_type(type_name)

        byte_count = self.HIP_FIXED_SCALAR_REPRESENTATION_BYTES.get(type_name)
        return str(byte_count) if byte_count is not None else None

    def validate_record_constructor_call(self, call):
        if not isinstance(getattr(call, "name", None), str):
            return
        definition = self.record_definition_from_type(call.name)
        if definition is not None:
            args = list(getattr(call, "args", None) or [])
            if self.record_contains_tracked_union_storage(definition) and any(
                self.infer_record_expression_definition(argument) is not None
                for argument in args
            ):
                self.raise_record_semantic(
                    ("union-containing constructor value active storage", definition),
                    f"call '{call.name}'",
                )
            issue = self.record_direct_constructor_issue(definition)
            if issue is not None:
                self.raise_record_semantic(issue, f"call '{call.name}'")
            return
        args = list(getattr(call, "args", None) or [])
        if len(args) != 1:
            return
        target_type = self.convert_hip_type_to_crossgl(call.name)
        if self.is_crossgl_scalar_type(target_type):
            self.validate_record_conversion(
                args[0], call.name, f"function-style conversion '{call.name}'"
            )

    def validate_record_value_against_target(
        self, expression, target_type, context, copy_semantics
    ):
        source = self.infer_record_expression_definition(expression)
        if source is None:
            return
        target = self.record_definition_from_type(target_type)
        self.validate_record_layout_use(source, context)
        self.validate_record_layout_use(target, context)
        if target is None:
            issue = self.record_conversion_operator_issue(source)
            if issue is not None:
                self.raise_record_semantic(issue, context)
            return
        if source is not target and self.record_is_derived_from(source, target):
            self.raise_record_semantic(
                ("derived-to-base value conversion", source), context
            )
        if source is target and copy_semantics:
            if self.record_contains_tracked_union_storage(source):
                self.raise_record_semantic(
                    ("union-containing value flow active storage", source), context
                )
            issue = self.record_copy_constructor_issue(source)
            if issue is not None:
                self.raise_record_semantic(issue, context)

    def record_operator_issue(
        self, node, operator, visiting=None, require_record_semantics=True
    ):
        visiting = set(visiting or ())
        if node is None or id(node) in visiting:
            return None
        visiting.add(id(node))
        method_name = f"operator{operator}"
        if any(
            isinstance(member, FunctionNode)
            and getattr(member, "name", None) == method_name
            for member in getattr(node, "members", []) or []
        ):
            return (f"record overloaded operator '{operator}' use", node)
        for base in self.record_base_definitions(node):
            issue = self.record_operator_issue(
                base, operator, visiting, require_record_semantics
            )
            if issue is not None:
                return issue
        if not require_record_semantics:
            return None
        conversion_issue = self.record_conversion_operator_issue(node)
        if conversion_issue is not None:
            return conversion_issue
        return (f"record operator '{operator}' cannot be represented exactly", node)

    def validate_record_operator_operands(self, operator, operands):
        require_record_semantics = not (len(operands) == 1 and operator == "&")
        for operand in operands:
            definition = self.infer_direct_record_expression_definition(operand)
            issue = self.record_operator_issue(
                definition, operator, require_record_semantics=require_record_semantics
            )
            if issue is not None:
                self.raise_record_semantic(issue, f"operator expression '{operator}'")

    def validate_record_scalar_context(self, expression, context):
        definition = self.infer_direct_record_expression_definition(expression)
        if definition is None:
            return
        self.validate_record_layout_use(definition, context)
        issue = self.record_conversion_operator_issue(definition)
        if issue is None:
            issue = (
                "record scalar conversion cannot be represented exactly",
                definition,
            )
        self.raise_record_semantic(issue, context)

    def validate_record_subscript(self, expression):
        definition = self.infer_direct_record_expression_definition(expression.array)
        if definition is not None:
            issue = self.record_operator_issue(definition, "[]")
            self.raise_record_semantic(issue, "subscript expression")
        self.validate_record_scalar_context(
            expression.index, "subscript index expression"
        )

    def validate_record_call_operator(self, call):
        name = getattr(call, "name", None)
        if not isinstance(name, str):
            return
        definition = self.infer_direct_record_expression_definition(name)
        if definition is None:
            return
        issue = self.record_operator_issue(definition, "()")
        self.raise_record_semantic(issue, "call expression")

    def validate_record_assignment(self, assignment):
        operator = getattr(assignment, "operator", "=")
        context = f"assignment '{operator}'"
        if operator != "=":
            definitions = []
            for expression in (assignment.left, assignment.right):
                definition = self.infer_direct_record_expression_definition(expression)
                if definition is not None and all(
                    definition is not item for item in definitions
                ):
                    definitions.append(definition)
            for definition in definitions:
                issue = self.record_operator_issue(definition, operator)
                if issue is not None:
                    self.raise_record_semantic(issue, context)

        target = self.infer_record_expression_definition(assignment.left)
        source = self.infer_record_expression_definition(assignment.right)
        self.validate_record_layout_use(target, context)
        self.validate_record_layout_use(source, context)
        if source is None:
            return
        if target is None:
            issue = self.record_conversion_operator_issue(source)
            if issue is not None:
                self.raise_record_semantic(issue, context)
            return
        if source is not target and self.record_is_derived_from(source, target):
            self.raise_record_semantic(
                ("derived-to-base assignment conversion", source), context
            )
        if source is target:
            if self.record_contains_tracked_union_storage(target):
                self.raise_record_semantic(
                    ("union-containing assignment active storage", target), context
                )
            issue = self.record_assignment_operator_issue(target)
            if issue is not None:
                self.raise_record_semantic(issue, context)

    def validate_record_conversion(self, expression, target_type, context):
        source = self.infer_record_expression_definition(expression)
        if source is None:
            return
        target = self.record_definition_from_type(target_type)
        self.validate_record_layout_use(source, context)
        self.validate_record_layout_use(target, context)
        if target is not None:
            if source is not target and self.record_is_derived_from(source, target):
                self.raise_record_semantic(
                    ("derived-to-base explicit conversion", source), context
                )
            return
        issue = self.record_conversion_operator_issue(source)
        if issue is not None:
            self.raise_record_semantic(issue, context)

    def record_defines_method(self, node, method_name, visiting=None):
        if node is None:
            return False
        visiting = set(visiting or ())
        if id(node) in visiting:
            return False
        visiting.add(id(node))
        if any(
            isinstance(member, FunctionNode)
            and member.name == method_name
            and not getattr(member, "is_record_constructor", False)
            and not getattr(member, "is_record_destructor", False)
            and not getattr(member, "is_conversion_operator", False)
            and not getattr(member, "is_using_declaration", False)
            for member in getattr(node, "members", []) or []
        ):
            return True
        for base in self.record_base_definitions(node):
            if self.record_defines_method(base, method_name, visiting):
                return True
        return False

    def reject_unsupported_record_method_call(self, node):
        call_name = getattr(node, "name", None)
        if isinstance(call_name, MemberAccessNode):
            definition = self.infer_record_expression_definition(call_name.object)
            if self.record_defines_method(definition, call_name.member):
                self.raise_record_semantic(
                    (f"record method call '{call_name.member}'", definition),
                    "call expression",
                )
            return
        if not isinstance(call_name, str) or "::" not in call_name:
            return
        owner_name, method_name = call_name.rsplit("::", 1)
        definition = self.record_definitions.get(owner_name)
        if definition is None:
            definition = self.record_definition_from_type(owner_name)
        if self.record_defines_method(definition, method_name):
            self.raise_record_semantic(
                (f"record method call '{call_name}'", definition),
                "call expression",
            )

    def canonical_record_static_member_name(self, name):
        if not isinstance(name, str) or "::" not in name:
            return name
        owner_name, member_name = name.rsplit("::", 1)
        definition = self.record_definition_from_type(owner_name)
        if definition is None:
            return name
        owner_candidates = [
            self.record_qualified_names.get(id(definition)),
            self.convert_hip_record_name_to_crossgl(
                getattr(definition, "name", owner_name)
            ),
        ]
        for owner in owner_candidates:
            if not owner:
                continue
            candidate = f"{owner}::{member_name}"
            if candidate in self.unsupported_record_static_constants:
                return candidate
            if candidate in self.enum_value_aliases:
                return candidate
        return name
