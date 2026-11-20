"""Static integral folding and the promotion of pointer members in lowered structs."""

import ast
import operator
import re
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from ..constant_bit_builtins import (
    BIT_BUILTIN_WIDTHS,
    fold_bit_builtin,
)
from ..identifier_rewrites import (
    qualified_identifiers,
    replace_unqualified_identifiers,
    unqualified_identifiers,
)
from ..integral_casts import (
    fold_integral_cast,
    integral_cast_target,
    rewrite_integral_casts,
)
from .support import (
    IDENTIFIER_RE,
    _MetalStructDefinition,
    _MetalStructMethod,
    _PointerPromotionPlan,
)


class MetalPointerPromotionMixin:
    """Members of ``MetalPreprocessor``: static integral folding and the promotion of pointer members in lowered structs."""

    def _evaluate_static_integral_expression(
        self, expression: str
    ) -> Tuple[bool, Optional[int]]:
        # Translate C++ boolean spellings to Python solely for parsing, then
        # interpret a strict AST whitelist. The 32-bit intermediate bound avoids
        # folding expressions whose C++ overflow/promotion behavior is unclear.
        text = rewrite_integral_casts(
            self._replace_concrete_sizeof_expressions(
                self._strip_template_argument_comments(expression)
            )
        ).strip()
        if not text or len(text) > 512:
            return False, None
        if re.search(r"\b(?:if|else)\b", text):
            return False, None
        if self._static_integral_expression_has_ambiguous_precedence(text):
            return False, None
        text = self._rewrite_static_integral_conditional_expressions(text)
        if text is None:
            return False, None
        text = re.sub(r"\btrue\b", "True", text)
        text = re.sub(r"\bfalse\b", "False", text)
        text = text.replace("&&", " and ").replace("||", " or ")
        text = re.sub(r"!(?!=)", " not ", text)
        try:
            parsed = ast.parse(f"({text.strip()})", mode="eval")
        except (SyntaxError, ValueError):
            return False, None

        allowed_nodes = (
            ast.Expression,
            ast.Constant,
            ast.UnaryOp,
            ast.UAdd,
            ast.USub,
            ast.Not,
            ast.BinOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.LShift,
            ast.RShift,
            ast.BitAnd,
            ast.BitOr,
            ast.BitXor,
            ast.BoolOp,
            ast.And,
            ast.Or,
            ast.IfExp,
            ast.Compare,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
            ast.Name,
            ast.Load,
            ast.Call,
        )
        nodes = list(ast.walk(parsed))
        if len(nodes) > 128 or any(
            not isinstance(node, allowed_nodes) for node in nodes
        ):
            return False, None
        return self._evaluate_static_integral_node(parsed.body)

    def _static_integral_expression_has_ambiguous_precedence(
        self, expression: str, *, depth: int = 0
    ) -> bool:
        # Python groups bitwise operators above comparisons and `not` below
        # arithmetic, unlike C++. Reject only unparenthesized mixed groups;
        # explicit parentheses split the recursive checks and remain foldable.
        if depth > 32:
            return True
        question = self._top_level_conditional_question(expression)
        if question is not None:
            colon = self._matching_conditional_colon(expression, question)
            if colon is None:
                return True
            return any(
                self._static_integral_expression_has_ambiguous_precedence(
                    part, depth=depth + 1
                )
                for part in (
                    expression[:question],
                    expression[question + 1 : colon],
                    expression[colon + 1 :],
                )
            )

        top_level: List[str] = []
        cursor = 0
        while cursor < len(expression):
            if expression[cursor] in "\"'":
                _literal, consumed = self._read_string(expression, cursor)
                cursor += consumed
                continue
            if expression[cursor] == "(":
                close = self._find_matching_delimiter(expression, cursor, "(", ")")
                if (
                    close is None
                    or self._static_integral_expression_has_ambiguous_precedence(
                        expression[cursor + 1 : close], depth=depth + 1
                    )
                ):
                    return True
                cursor = close + 1
                continue
            operator_text = next(
                (
                    operator_text
                    for operator_text in (
                        "<<",
                        ">>",
                        "<=",
                        ">=",
                        "==",
                        "!=",
                        "&&",
                        "||",
                    )
                    if expression.startswith(operator_text, cursor)
                ),
                None,
            )
            if operator_text is not None:
                top_level.append(operator_text)
                cursor += len(operator_text)
                continue
            if expression[cursor] in "+-*/%<>&|^!":
                top_level.append(expression[cursor])
            cursor += 1

        logical_operands: List[List[str]] = [[]]
        for operator_text in top_level:
            if operator_text in {"&&", "||"}:
                logical_operands.append([])
            else:
                logical_operands[-1].append(operator_text)

        comparisons = {"<", "<=", ">", ">=", "==", "!="}
        bitwise = {"&", "|", "^"}
        for operand_operators in logical_operands:
            comparison_count = sum(
                operator_text in comparisons for operator_text in operand_operators
            )
            if comparison_count > 1:
                return True
            if comparisons.intersection(operand_operators) and bitwise.intersection(
                operand_operators
            ):
                return True
            if "!" in operand_operators and any(
                operator_text != "!" for operator_text in operand_operators
            ):
                return True
        return False

    def _rewrite_static_integral_conditional_expressions(
        self, expression: str, *, depth: int = 0
    ) -> Optional[str]:
        if depth > 32:
            return None
        question = self._top_level_conditional_question(expression)
        if question is not None:
            colon = self._matching_conditional_colon(expression, question)
            if colon is None:
                return None
            condition = self._rewrite_static_integral_conditional_expressions(
                expression[:question], depth=depth + 1
            )
            true_value = self._rewrite_static_integral_conditional_expressions(
                expression[question + 1 : colon], depth=depth + 1
            )
            false_value = self._rewrite_static_integral_conditional_expressions(
                expression[colon + 1 :], depth=depth + 1
            )
            if condition is None or true_value is None or false_value is None:
                return None
            if (
                not condition.strip()
                or not true_value.strip()
                or not false_value.strip()
            ):
                return None
            return f"(({true_value}) if ({condition}) else ({false_value}))"

        rewritten: List[str] = []
        cursor = 0
        while cursor < len(expression):
            if expression[cursor] in "\"'":
                literal, consumed = self._read_string(expression, cursor)
                rewritten.append(literal)
                cursor += consumed
                continue
            if expression[cursor] != "(":
                rewritten.append(expression[cursor])
                cursor += 1
                continue
            close = self._find_matching_delimiter(expression, cursor, "(", ")")
            if close is None:
                return None
            inner = self._rewrite_static_integral_conditional_expressions(
                expression[cursor + 1 : close], depth=depth + 1
            )
            if inner is None:
                return None
            rewritten.append(f"({inner})")
            cursor = close + 1
        return "".join(rewritten)

    def _top_level_conditional_question(self, expression: str) -> Optional[int]:
        depth = 0
        i = 0
        while i < len(expression):
            if expression[i] in "\"'":
                _literal, consumed = self._read_string(expression, i)
                i += consumed
                continue
            if expression[i] in "([{":
                depth += 1
            elif expression[i] in ")]}" and depth:
                depth -= 1
            elif expression[i] == "?" and depth == 0:
                return i
            i += 1
        return None

    def _matching_conditional_colon(
        self, expression: str, question: int
    ) -> Optional[int]:
        depth = 0
        nested_conditionals = 0
        i = question + 1
        while i < len(expression):
            if expression[i] in "\"'":
                _literal, consumed = self._read_string(expression, i)
                i += consumed
                continue
            token = expression[i]
            if token in "([{":
                depth += 1
            elif token in ")]}" and depth:
                depth -= 1
            elif depth == 0 and token == "?":
                nested_conditionals += 1
            elif depth == 0 and token == ":":
                if (i > 0 and expression[i - 1] == ":") or (
                    i + 1 < len(expression) and expression[i + 1] == ":"
                ):
                    i += 1
                    continue
                if nested_conditionals:
                    nested_conditionals -= 1
                else:
                    return i
            i += 1
        return None

    def _evaluate_static_integral_node(
        self, node: ast.AST
    ) -> Tuple[bool, Optional[int]]:
        if isinstance(node, ast.Constant):
            if type(node.value) not in {bool, int}:
                return False, None
            value = int(node.value)
            if not self._static_fold_value_is_bounded(value):
                return False, None
            return True, value

        if isinstance(node, ast.UnaryOp):
            valid, operand = self._evaluate_static_integral_node(node.operand)
            if not valid or operand is None:
                return False, None
            if isinstance(node.op, ast.UAdd):
                value = operand
            elif isinstance(node.op, ast.USub):
                value = -operand
            elif isinstance(node.op, ast.Not):
                return True, int(not operand)
            else:
                return False, None
            if not self._static_fold_value_is_bounded(value):
                return False, None
            return True, value

        if isinstance(node, ast.BinOp):
            left_valid, left = self._evaluate_static_integral_node(node.left)
            right_valid, right = self._evaluate_static_integral_node(node.right)
            if not left_valid or not right_valid or left is None or right is None:
                return False, None
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            elif isinstance(node.op, (ast.Div, ast.Mod)):
                if right == 0:
                    return False, None
                quotient = abs(left) // abs(right)
                if (left < 0) != (right < 0):
                    quotient = -quotient
                value = (
                    quotient
                    if isinstance(node.op, ast.Div)
                    else left - quotient * right
                )
            elif isinstance(node.op, (ast.LShift, ast.RShift)):
                if left < 0 or right < 0 or right >= 31:
                    return False, None
                value = (
                    left << right if isinstance(node.op, ast.LShift) else left >> right
                )
            elif isinstance(node.op, ast.BitAnd):
                if left < 0 or right < 0:
                    return False, None
                value = left & right
            elif isinstance(node.op, ast.BitOr):
                if left < 0 or right < 0:
                    return False, None
                value = left | right
            elif isinstance(node.op, ast.BitXor):
                if left < 0 or right < 0:
                    return False, None
                value = left ^ right
            else:
                return False, None
            if not self._static_fold_value_is_bounded(value):
                return False, None
            return True, value

        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                for item in node.values:
                    valid, value = self._evaluate_static_integral_node(item)
                    if not valid or value is None:
                        return False, None
                    if not value:
                        return True, 0
                return True, 1
            if isinstance(node.op, ast.Or):
                for item in node.values:
                    valid, value = self._evaluate_static_integral_node(item)
                    if not valid or value is None:
                        return False, None
                    if value:
                        return True, 1
                return True, 0
            return False, None

        if isinstance(node, ast.IfExp):
            valid, condition = self._evaluate_static_integral_node(node.test)
            if not valid or condition is None:
                return False, None
            body_valid, body = self._evaluate_static_integral_node(node.body)
            else_valid, else_value = self._evaluate_static_integral_node(node.orelse)
            if not body_valid or body is None or not else_valid or else_value is None:
                return False, None
            return True, body if condition else else_value

        if isinstance(node, ast.Compare):
            valid, left = self._evaluate_static_integral_node(node.left)
            if not valid or left is None:
                return False, None
            comparisons = {
                ast.Eq: operator.eq,
                ast.NotEq: operator.ne,
                ast.Lt: operator.lt,
                ast.LtE: operator.le,
                ast.Gt: operator.gt,
                ast.GtE: operator.ge,
            }
            for op_node, comparator in zip(node.ops, node.comparators):
                valid, right = self._evaluate_static_integral_node(comparator)
                comparison = comparisons.get(type(op_node))
                if not valid or right is None or comparison is None:
                    return False, None
                if not comparison(left, right):
                    return True, 0
                left = right
            return True, 1

        if isinstance(node, ast.Call):
            cast_target = (
                integral_cast_target(node.func.id)
                if isinstance(node.func, ast.Name)
                and len(node.args) == 1
                and not node.keywords
                else None
            )
            if cast_target is not None:
                valid, operand = self._evaluate_static_integral_node(node.args[0])
                if not valid or operand is None:
                    return False, None
                value = fold_integral_cast(cast_target, operand)
                return (False, None) if value is None else (True, value)
            # Otherwise only Clang's bit-counting builtins, with one concrete operand.
            if (
                not isinstance(node.func, ast.Name)
                or node.func.id not in BIT_BUILTIN_WIDTHS
                or len(node.args) != 1
                or node.keywords
            ):
                return False, None
            valid, operand = self._evaluate_static_integral_node(node.args[0])
            if not valid or operand is None:
                return False, None
            value = fold_bit_builtin(node.func.id, operand)
            if value is None:
                return False, None
            return True, value

        return False, None

    def _static_fold_value_is_bounded(self, value: int) -> bool:
        return self._STATIC_FOLD_MIN_INT <= value <= self._STATIC_FOLD_MAX_INT

    # ------------------------------------------------------------------ #
    # Pointer-member promotion (device/threadgroup pointer members).      #
    # ------------------------------------------------------------------ #

    def _promoted_pointer_param_name(self, member_name: str) -> str:
        # A fresh, collision-proof name for a promoted pointer parameter. Using a
        # prefix (rather than the raw member name) avoids clashing with a local
        # in the method body and sidesteps backends where `in`/`out` are reserved.
        return f"crosstl_ptr_{member_name}"

    def _promoted_pointer_offset_member_name(self, member_name: str) -> str:
        return f"crosstl_ptr_offset_{member_name}"

    def _pointer_initializer_root_and_offset(
        self,
        expression: str,
        pointer_parameter_names: Set[str],
    ) -> Optional[Tuple[str, str]]:
        """Split one derived pointer initializer into its base and scalar delta.

        Only additive expressions with exactly one constructor pointer parameter
        are accepted. Multiplication, casts, calls, or subtraction with the
        pointer on the right remain unpromoted so root-changing/reinterpreting
        expressions fail closed on the ordinary lowering path.
        """

        def analyze(value: str) -> Optional[Tuple[Optional[str], str]]:
            value = self._strip_enclosing_parens(value)
            pointer_references = [
                name
                for name in IDENTIFIER_RE.findall(value)
                if name in pointer_parameter_names
            ]
            if not pointer_references:
                return None, value
            if IDENTIFIER_RE.fullmatch(value) and value in pointer_parameter_names:
                return value, "0"

            split = self._split_top_level_binary_arithmetic(value)
            if split is None:
                return None
            left_text, operator_text, right_text = split
            if operator_text not in {"+", "-"}:
                return None
            left = analyze(left_text)
            right = analyze(right_text)
            if left is None or right is None:
                return None
            left_root, left_offset = left
            right_root, right_offset = right
            if left_root is not None and right_root is not None:
                return None
            if left_root is not None:
                if left_offset == "0":
                    offset = (
                        right_offset if operator_text == "+" else f"-({right_offset})"
                    )
                else:
                    offset = f"({left_offset}) {operator_text} ({right_offset})"
                return left_root, offset
            if right_root is not None:
                if operator_text != "+":
                    return None
                offset = (
                    left_offset
                    if right_offset == "0"
                    else f"({left_offset}) + ({right_offset})"
                )
                return right_root, offset
            return None

        result = analyze(expression)
        if result is None or result[0] is None:
            return None
        root, offset = result
        return root, offset

    def _promoted_pointer_offset_usage_is_supported(
        self,
        body: str,
        pointer_member_names: Set[str],
    ) -> bool:
        # Normalize explicit receiver spelling so the scanner can distinguish
        # this object's pointer members from `other.member` accesses.
        normalized = re.sub(r"\(\s*\*\s*this\s*\)\s*(?=\.|->)", "self", body)
        normalized = re.sub(r"\bthis\s*->\s*", "self.", normalized)
        normalized = re.sub(r"\bthis\s*\.\s*", "self.", normalized)
        for name in pointer_member_names:
            normalized = re.sub(
                rf"\bself\s*\.\s*{re.escape(name)}\b",
                name,
                normalized,
            )

        index = 0
        while index < len(normalized):
            if normalized[index] in "\"'":
                _literal, consumed = self._read_string(normalized, index)
                index += consumed
                continue
            if normalized.startswith("//", index):
                end = normalized.find("\n", index)
                index = len(normalized) if end == -1 else end + 1
                continue
            if normalized.startswith("/*", index):
                end = normalized.find("*/", index + 2)
                index = len(normalized) if end == -1 else end + 2
                continue
            if not (normalized[index].isalpha() or normalized[index] == "_"):
                index += 1
                continue
            identifier, consumed = self._read_identifier(normalized, index)
            end = index + consumed
            if (
                identifier in pointer_member_names
                and not self._is_member_identifier_context(normalized, index)
                and not self._identifier_is_declaration_or_call(normalized, index, end)
            ):
                cursor = end
                while cursor < len(normalized) and normalized[cursor].isspace():
                    cursor += 1
                suffix = normalized[cursor:]
                if suffix.startswith(("+=", "-=", "++", "--")):
                    pass
                elif suffix.startswith(
                    ("*=", "/=", "%=", "<<=", ">>=", "&=", "|=", "^=")
                ) or (suffix.startswith("=") and not suffix.startswith("==")):
                    return False

                previous = index - 1
                while previous >= 0 and normalized[previous].isspace():
                    previous -= 1
                if (
                    previous >= 0
                    and normalized[previous] == "&"
                    and not suffix.startswith("[")
                ):
                    return False
            index = end
        return True

    def _pointer_promotion_plan(
        self,
        struct: _MetalStructDefinition,
        code: Optional[str] = None,
    ) -> Optional[_PointerPromotionPlan]:
        # Decide whether `struct` can have its pointer members promoted out (see
        # _PointerPromotionPlan). Returns a plan, or None to leave the struct on
        # the ordinary self-passing lowering path. Every check is conservative:
        # any shape we are not certain we can rewrite faithfully bails to None so
        # a struct is never translated worse than it is today.
        data_members = struct.data_members
        if not data_members:
            return None
        pointer_members = [member for member in data_members if member.is_pointer]
        if not pointer_members:
            return None
        # A pointer member captured as an array (`device T* taps[4]`) or a
        # template/operator-call struct is out of scope.
        if any(member.is_array for member in pointer_members):
            return None
        if struct.has_operator_call:
            return None
        # Historical direct-pointer promotion remains strict. The derived-pointer
        # opt-in may discard template methods only when no call to any such method
        # survives outside the declaration (or from a concrete sibling method).
        # Used template methods stay on the ordinary lowering path, where their
        # existing call-site specialization machinery remains authoritative.
        if struct.template_methods and not self.promote_derived_pointer_members:
            return None
        # Exactly one constructor, which must source every pointer member from a
        # pointer parameter, so a construction argument is the pointer expression.
        if len(struct.constructors) != 1:
            return None
        constructor = struct.constructors[0]
        if constructor.span is None:
            return None
        pointer_member_names = {member.name for member in pointer_members}
        pointer_parameter_indices = [
            index
            for index, parameter_type in enumerate(constructor.param_types)
            if "*" in parameter_type
        ]
        pointer_parameter_names = {
            constructor.param_names[index]
            for index in pointer_parameter_indices
            if index < len(constructor.param_names)
        }
        pointer_ctor_arg_indices: List[int] = []
        pointer_offset_initializers: List[str] = []
        has_derived_initializer = False
        for member in pointer_members:
            init_expr = constructor.init_map.get(member.name)
            if init_expr is None:
                return None
            init_expr = init_expr.strip()
            if init_expr in pointer_parameter_names:
                root_name, offset = init_expr, "0"
            else:
                if not self.promote_derived_pointer_members:
                    return None
                derived = self._pointer_initializer_root_and_offset(
                    init_expr,
                    pointer_parameter_names,
                )
                if derived is None:
                    return None
                root_name, offset = derived
                has_derived_initializer = True
            index = constructor.param_names.index(root_name)
            if index >= len(constructor.param_types):
                return None
            pointer_ctor_arg_indices.append(index)
            pointer_offset_initializers.append(offset)

        # The constructor body must not reference any pointer parameter or pointer
        # member (both are about to disappear); otherwise dropping them is unsafe.
        pointer_param_names = {
            constructor.param_names[index] for index in pointer_ctor_arg_indices
        }
        body_identifiers = set(IDENTIFIER_RE.findall(constructor.body))
        if body_identifiers & pointer_param_names:
            return None
        if body_identifiers & pointer_member_names:
            return None
        # The remaining initializers stay in the rebuilt constructor. One that
        # reads a pointer parameter keeps that parameter. One that reads a
        # promoted member initialized directly from a parameter reads the
        # parameter instead; any other promoted member makes promotion unsafe.
        direct_member_parameters = {
            member.name: constructor.init_map[member.name].strip()
            for member in pointer_members
            if constructor.init_map[member.name].strip() in pointer_param_names
        }
        retained_parameter_names: Set[str] = set()
        for member_name, init_expr in constructor.init_map.items():
            if member_name in pointer_member_names:
                continue
            identifiers = set(unqualified_identifiers(init_expr))
            # ``this->src`` or ``other.src`` names a member the rewrite cannot
            # map to a parameter.
            if "this" in identifiers or (
                set(qualified_identifiers(init_expr)) & pointer_member_names
            ):
                return None
            for name in sorted(identifiers & pointer_member_names):
                if name not in direct_member_parameters:
                    return None
                retained_parameter_names.add(direct_member_parameters[name])
            retained_parameter_names.update(identifiers & pointer_param_names)
        retained_pointer_ctor_arg_indices = tuple(
            sorted(
                {
                    constructor.param_names.index(name)
                    for name in retained_parameter_names
                }
            )
        )
        # No method may declare a parameter or local that collides with a pointer
        # member name (the promoted parameter would shadow / be shadowed by it).
        for method in struct.methods:
            if method.is_static:
                continue
            names = set(method.parameter_names)
            names.update(self._local_variable_names(method.body))
            if names & pointer_member_names:
                return None

        offset_initializers: Optional[List[str]] = None
        if has_derived_initializer:
            if struct.template_methods and (
                code is None
                or self._promoted_struct_template_method_is_referenced(code, struct)
            ):
                return None
            generated_names = {
                self._promoted_pointer_offset_member_name(member.name)
                for member in pointer_members
            }
            occupied_names = set(struct.data_member_names) | set(
                constructor.param_names
            )
            if generated_names & occupied_names:
                return None
            if any(
                not self._promoted_pointer_offset_usage_is_supported(
                    method.body,
                    pointer_member_names,
                )
                for method in struct.methods
                if not method.is_static
            ):
                return None
            offset_initializers = pointer_offset_initializers

        return _PointerPromotionPlan(
            struct_name=struct.name,
            pointer_members=pointer_members,
            pointer_ctor_arg_indices=pointer_ctor_arg_indices,
            constructor=constructor,
            pointer_offset_initializers=offset_initializers,
            retained_pointer_ctor_arg_indices=retained_pointer_ctor_arg_indices,
            initializer_member_parameters={
                name: parameter
                for name, parameter in sorted(direct_member_parameters.items())
                if parameter in retained_parameter_names
            },
        )

    def _promoted_struct_template_method_is_referenced(
        self,
        code: str,
        struct: _MetalStructDefinition,
    ) -> bool:
        method_names = {method.name for method in struct.template_methods}
        if not method_names:
            return False

        outside = (
            code[: struct.span[0]]
            + (" " * (struct.span[1] - struct.span[0]))
            + code[struct.span[1] :]
        )
        masked_outside = self._mask_comments_and_literals(outside)
        for method_name in method_names:
            if re.search(
                rf"(?:\.|->)\s*{re.escape(method_name)}\b\s*"
                rf"(?:<[^;{{}}()]*>\s*)?\(",
                masked_outside,
            ):
                return True
        for method in struct.methods:
            masked_body = self._mask_comments_and_literals(method.body)
            for method_name in method_names:
                if re.search(
                    rf"\b{re.escape(method_name)}\b\s*" rf"(?:<[^;{{}}()]*>\s*)?\(",
                    masked_body,
                ):
                    return True
        return False

    def _render_promoted_struct(
        self,
        code: str,
        struct: _MetalStructDefinition,
        plan: _PointerPromotionPlan,
    ) -> Tuple[str, List[str]]:
        # Produce (data_only_struct_text, [free_function_text, ...]) for a struct
        # whose pointer members are promoted. Preserve unknown non-method body
        # text verbatim so nested aggregates, aliases, layout attributes, and
        # static assertions remain part of the owner's lexical type context.
        # Known data-member spans are still regenerated from their specialized
        # metadata: template materialization may have concretized their types,
        # extents, or initializers after the source text was captured. Promoted
        # pointers are removed/replaced, the constructor is rebuilt without
        # those pointers, and methods are hoisted. Reconstructing the entire body
        # solely from data-member metadata would silently discard nested byte-
        # view structs used by aggregate loads.
        struct_text = code[struct.span[0] : struct.span[1]]
        header_body_start = self._find_next_top_level_char(struct_text, 0, "{")
        header = struct_text[: header_body_start + 1]
        body_abs_start, body_abs_end = struct.body_span
        body_rel_start = body_abs_start - struct.span[0]
        body_rel_end = body_abs_end - struct.span[0]
        body = struct_text[body_rel_start:body_rel_end]

        replacements: List[Tuple[int, int, str]] = []
        for member in struct.data_members:
            if member.span is None:
                continue
            if member.is_pointer:
                replacement = ""
                if plan.pointer_offset_initializers is not None:
                    replacement = (
                        f"int {self._promoted_pointer_offset_member_name(member.name)};"
                    )
            else:
                replacement = f"{member.type_text} {member.name}{member.array_suffix}"
                if member.default is not None:
                    if member.initializer_style == "direct-list":
                        replacement += f"{{{member.default}}}"
                    else:
                        replacement += f" = {member.default}"
                replacement += ";"
            replacements.append(
                (
                    member.span[0] - body_abs_start,
                    member.span[1] - body_abs_start,
                    replacement,
                )
            )

        for method in [*struct.methods, *struct.template_methods]:
            rel_start = method.span[0] - body_abs_start
            rel_end = method.span[1] - body_abs_start
            tail = rel_end
            while tail < len(body) and body[tail].isspace():
                tail += 1
            if tail < len(body) and body[tail] == ";":
                rel_end = tail + 1
            replacements.append((rel_start, rel_end, ""))

        constructor = plan.constructor
        if constructor.span is not None:
            replacements.append(
                (
                    constructor.span[0] - body_abs_start,
                    constructor.span[1] - body_abs_start,
                    self._rebuild_promoted_constructor(plan),
                )
            )

        data_body = self._apply_text_replacements(body, replacements)
        data_body = self._collapse_blank_lines(data_body)
        data_only = (header + data_body + "}").rstrip()
        if not data_only.endswith(";"):
            data_only += ";"
        free_functions = [
            self._emit_promoted_free_function(struct, method, plan)
            for method in struct.methods
        ]
        return data_only, free_functions

    def _rebuild_promoted_constructor(self, plan: _PointerPromotionPlan) -> str:
        # Rebuild the constructor without its pointer parameters or their
        # initializer-list entries; the scalar parameters, remaining initializers
        # and the body (which computes scalar members) are preserved verbatim.
        constructor = plan.constructor
        drop_indices = plan.dropped_pointer_ctor_arg_indices
        kept_params: List[str] = []
        params = [
            param.strip()
            for param in self._split_top_level_commas(constructor.params_text)
        ]
        for index, param in enumerate(params):
            if not param:
                continue
            if index in drop_indices:
                continue
            kept_params.append(param)
        pointer_member_names = set(plan.pointer_member_names)
        offset_initializers = (
            dict(zip(plan.pointer_member_names, plan.pointer_offset_initializers))
            if plan.pointer_offset_initializers is not None
            else {}
        )
        kept_inits: List[str] = []
        if constructor.init_text:
            for item in self._split_top_level_commas(constructor.init_text):
                item = item.strip()
                if not item:
                    continue
                match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*[\({]", item)
                if match is not None and match.group(1) in pointer_member_names:
                    member_name = match.group(1)
                    if plan.pointer_offset_initializers is not None:
                        kept_inits.append(
                            f"{self._promoted_pointer_offset_member_name(member_name)}"
                            f"({offset_initializers[member_name]})"
                        )
                    continue
                kept_inits.append(
                    replace_unqualified_identifiers(
                        item, plan.initializer_member_parameters
                    )
                )
        rebuilt = f"{constructor.prefix}{', '.join(kept_params)})"
        if kept_inits:
            rebuilt += " : " + ", ".join(kept_inits)
        rebuilt += f" {{{constructor.body}}}"
        return rebuilt

    def _emit_promoted_free_function(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        plan: _PointerPromotionPlan,
    ) -> str:
        # Emit an instance helper with a reference receiver and promoted pointer
        # parameters, or the ordinary static form for a static method.
        return_type = self._canonicalize_struct_scoped_type(
            method.return_type, struct, None
        )
        rewritten_body = self._rewrite_promoted_method_body(struct, method, plan)
        rewritten_body = self._rewrite_promoted_internal_method_calls(
            struct,
            plan,
            rewritten_body,
        )
        rewritten_body = self._canonicalize_struct_scoped_local_declarations(
            rewritten_body, struct, None
        )
        rewritten_body = self._canonicalize_struct_scoped_cast_types(
            rewritten_body, struct, None
        )
        rewritten_body = self._canonicalize_struct_scoped_template_arguments(
            rewritten_body, struct, method, None
        )
        params = self._canonicalize_struct_scoped_parameters(
            method.parameters, struct, None
        )
        if method.is_static:
            new_params = params if params and params != "void" else ""
        else:
            parts = [self._instance_receiver_parameter(struct, method)]
            for member in plan.pointer_members:
                member_type = self._canonicalize_struct_scoped_type(
                    member.type_text, struct, None
                )
                parts.append(
                    f"{member_type} "
                    f"{self._promoted_pointer_param_name(member.name)}"
                )
            if params and params != "void":
                parts.append(params)
            new_params = ", ".join(parts)
        return (
            f"static inline {return_type} {method.free_name}({new_params}) "
            f"{{{rewritten_body}}}"
        )

    def _rewrite_promoted_internal_method_calls(
        self,
        struct: _MetalStructDefinition,
        plan: _PointerPromotionPlan,
        body: str,
    ) -> str:
        """Forward the implicit receiver and promoted pointers to sibling helpers."""
        siblings: Dict[str, List[_MetalStructMethod]] = {}
        for method in struct.methods:
            siblings.setdefault(method.name, []).append(method)

        result: List[str] = []
        index = 0
        while index < len(body):
            if body[index] in "\"'":
                literal, consumed = self._read_string(body, index)
                result.append(literal)
                index += consumed
                continue
            if body.startswith("//", index):
                end = body.find("\n", index)
                if end == -1:
                    result.append(body[index:])
                    break
                result.append(body[index:end])
                index = end
                continue
            if body.startswith("/*", index):
                end = body.find("*/", index + 2)
                if end == -1:
                    result.append(body[index:])
                    break
                result.append(body[index : end + 2])
                index = end + 2
                continue
            if not (body[index].isalpha() or body[index] == "_"):
                result.append(body[index])
                index += 1
                continue

            ident, consumed = self._read_identifier(body, index)
            ident_end = index + consumed
            rewrite = self._try_rewrite_promoted_internal_method_call(
                struct,
                plan,
                body,
                ident,
                index,
                ident_end,
                siblings,
            )
            if rewrite is not None:
                end, replacement = rewrite
                result.append(replacement)
                index = end
                continue
            result.append(ident)
            index = ident_end
        return "".join(result)

    def _try_rewrite_promoted_internal_method_call(
        self,
        struct: _MetalStructDefinition,
        plan: _PointerPromotionPlan,
        body: str,
        ident: str,
        ident_start: int,
        ident_end: int,
        siblings: Dict[str, List[_MetalStructMethod]],
    ) -> Optional[Tuple[int, str]]:
        method_name = ident
        method_name_end = ident_end

        if ident == "self":
            cursor = ident_end
            while cursor < len(body) and body[cursor].isspace():
                cursor += 1
            if body.startswith("->", cursor):
                cursor += 2
            elif cursor < len(body) and body[cursor] == ".":
                cursor += 1
            else:
                return None
            while cursor < len(body) and body[cursor].isspace():
                cursor += 1
            if cursor >= len(body) or not (
                body[cursor].isalpha() or body[cursor] == "_"
            ):
                return None
            method_name, consumed = self._read_identifier(body, cursor)
            method_name_end = cursor + consumed
        else:
            previous = ident_start - 1
            while previous >= 0 and body[previous].isspace():
                previous -= 1
            if previous >= 0 and (
                body[previous] == "."
                or (previous >= 1 and body[previous - 1 : previous + 1] in {"->", "::"})
            ):
                return None

        if method_name == "operator":
            empty_open = method_name_end
            while empty_open < len(body) and body[empty_open].isspace():
                empty_open += 1
            if empty_open >= len(body) or body[empty_open] != "(":
                return None
            empty_close = self._find_matching_delimiter(body, empty_open, "(", ")")
            if empty_close is None or body[empty_open + 1 : empty_close].strip():
                return None
            method_name = "operator()"
            method_name_end = empty_close + 1

        overloads = siblings.get(method_name, [])
        if len(overloads) != 1:
            return None
        call_suffix = self._member_template_call_suffix(body, method_name_end)
        if call_suffix is None:
            return None
        arg_open, explicit_template_arguments = call_suffix
        if explicit_template_arguments is not None:
            return None
        arg_close = self._find_matching_delimiter(body, arg_open, "(", ")")
        if arg_close is None:
            return None

        sibling = overloads[0]
        self._reject_reference_returning_method_call(
            body,
            struct,
            sibling,
            arg_open,
            arg_close,
        )
        raw_args = body[arg_open + 1 : arg_close].strip()
        forwarded: List[str] = []
        if not sibling.is_static:
            forwarded.append("self")
            forwarded.extend(
                self._promoted_pointer_param_name(member.name)
                for member in plan.pointer_members
            )
        if raw_args:
            forwarded.append(raw_args)
        return arg_close + 1, f"{sibling.free_name}({', '.join(forwarded)})"

    def _rewrite_promoted_method_body(
        self,
        struct: _MetalStructDefinition,
        method: _MetalStructMethod,
        plan: _PointerPromotionPlan,
    ) -> str:
        # Like _rewrite_method_body, but pointer members resolve to promoted base
        # pointer parameters. The derived-pointer opt-in additionally carries each
        # member's mutable state in a scalar offset field on `self`.
        body = method.body
        body = re.sub(r"\(\s*\*\s*this\s*\)\s*(?=\.|->)", "self", body)
        body = re.sub(r"\bthis\s*->\s*", "self.", body)
        body = re.sub(r"\bthis\s*\.\s*", "self.", body)
        body = re.sub(r"\bthis\b", "self", body)
        body = self._substitute_static_data_member_initializers(struct, method, body)
        if method.is_static:
            return body
        pointer_member_names = plan.pointer_member_names

        if plan.pointer_offset_initializers is None:
            # Preserve the historical direct-pointer promotion bytes exactly.
            for name in pointer_member_names:
                body = re.sub(
                    rf"\bself\s*\.\s*{re.escape(name)}\b",
                    self._promoted_pointer_param_name(name),
                    body,
                )
            shadowed = set(method.parameter_names)
            pointer_name_set = set(pointer_member_names)
            mapping: Dict[str, str] = {}
            for name in struct.data_member_names:
                if name in shadowed:
                    continue
                if name in pointer_name_set:
                    mapping[name] = self._promoted_pointer_param_name(name)
                else:
                    mapping[name] = f"self.{name}"
            if not mapping:
                return body
            return self._substitute_bare_member_references(
                body,
                mapping,
                local_shadows=self._local_variable_shadow_scopes(body),
            )

        body = self._rewrite_promoted_pointer_offset_mutations(body, plan)
        pointer_name_set = set(pointer_member_names)
        pointer_views = {
            name: "({} + self.{})".format(
                self._promoted_pointer_param_name(name),
                self._promoted_pointer_offset_member_name(name),
            )
            for name in pointer_member_names
        }
        for name, view in pointer_views.items():
            body = re.sub(
                rf"\bself\s*\.\s*{re.escape(name)}\b",
                view,
                body,
            )
        shadowed = set(method.parameter_names)
        mapping: Dict[str, str] = {}
        for name in struct.data_member_names:
            if name in shadowed:
                continue
            if name in pointer_name_set:
                mapping[name] = pointer_views[name]
            else:
                mapping[name] = f"self.{name}"
        return self._substitute_bare_member_references(
            body,
            mapping,
            local_shadows=self._local_variable_shadow_scopes(body),
        )

    def _rewrite_promoted_pointer_offset_mutations(
        self,
        body: str,
        plan: _PointerPromotionPlan,
    ) -> str:
        offset_names = {
            name: f"self.{self._promoted_pointer_offset_member_name(name)}"
            for name in plan.pointer_member_names
        }
        for name, offset_name in offset_names.items():
            body = re.sub(
                rf"(?P<operator>\+\+|--)(?P<space>\s*)"
                rf"\bself\s*\.\s*{re.escape(name)}\b",
                lambda match: (
                    f"{match.group('operator')}{match.group('space')}{offset_name}"
                ),
                body,
            )
            body = re.sub(
                rf"\bself\s*\.\s*{re.escape(name)}\b" rf"(?=\s*(?:\+=|-=|\+\+|--))",
                offset_name,
                body,
            )
        return self._substitute_bare_promoted_pointer_mutations(body, offset_names)

    def _substitute_bare_promoted_pointer_mutations(
        self,
        body: str,
        offset_names: Dict[str, str],
    ) -> str:
        result: List[str] = []
        index = 0
        while index < len(body):
            if body[index] in "\"'":
                literal, consumed = self._read_string(body, index)
                result.append(literal)
                index += consumed
                continue
            if body.startswith("//", index):
                end = body.find("\n", index)
                if end == -1:
                    result.append(body[index:])
                    break
                result.append(body[index:end])
                index = end
                continue
            if body.startswith("/*", index):
                end = body.find("*/", index + 2)
                if end == -1:
                    result.append(body[index:])
                    break
                result.append(body[index : end + 2])
                index = end + 2
                continue
            if not (body[index].isalpha() or body[index] == "_"):
                result.append(body[index])
                index += 1
                continue
            identifier, consumed = self._read_identifier(body, index)
            end = index + consumed
            replacement = offset_names.get(identifier)
            if (
                replacement is not None
                and not self._is_member_identifier_context(body, index)
                and not self._identifier_is_declaration_or_call(body, index, end)
            ):
                cursor = end
                while cursor < len(body) and body[cursor].isspace():
                    cursor += 1
                suffix_mutation = body[cursor:].startswith(("+=", "-=", "++", "--"))
                previous = index - 1
                while previous >= 0 and body[previous].isspace():
                    previous -= 1
                prefix_mutation = previous >= 1 and body[
                    previous - 1 : previous + 1
                ] in {"++", "--"}
                if suffix_mutation or prefix_mutation:
                    result.append(replacement)
                    index = end
                    continue
            result.append(identifier)
            index = end
        return "".join(result)

    def _substitute_bare_member_references(
        self,
        body: str,
        mapping: Dict[str, str],
        *,
        local_shadows: Optional[Dict[str, List[Tuple[int, int, int]]]] = None,
    ) -> str:
        # Walk identifiers and replace bare member references per `mapping`, while
        # leaving member accesses on OTHER objects (`obj.x`, `obj->x`), already
        # rewritten `self.x`, and function-name uses untouched. Generalizes
        # `_qualify_member_references` to an arbitrary member -> replacement map.
        result: List[str] = []
        i = 0
        n = len(body)
        while i < n:
            ch = body[i]
            if ch in "\"'":
                literal, consumed = self._read_string(body, i)
                result.append(literal)
                i += consumed
                continue
            if body.startswith("//", i):
                end = body.find("\n", i)
                if end == -1:
                    result.append(body[i:])
                    break
                result.append(body[i:end])
                i = end
                continue
            if body.startswith("/*", i):
                end = body.find("*/", i + 2)
                if end == -1:
                    result.append(body[i:])
                    break
                result.append(body[i : end + 2])
                i = end + 2
                continue
            if ch.isalpha() or ch == "_":
                ident, consumed = self._read_identifier(body, i)
                replacement = mapping.get(ident)
                if (
                    replacement is not None
                    and not any(
                        declaration_position <= i and scope_start <= i < scope_end
                        for declaration_position, scope_start, scope_end in (
                            local_shadows or {}
                        ).get(ident, ())
                    )
                    and not self._is_member_identifier_context(body, i)
                    and not self._identifier_is_declaration_or_call(
                        body, i, i + consumed
                    )
                ):
                    result.append(replacement)
                else:
                    result.append(ident)
                i += consumed
                continue
            result.append(ch)
            i += 1
        return "".join(result)

    def _rewrite_method_body(
        self, struct: _MetalStructDefinition, method: _MetalStructMethod
    ) -> str:
        # Rewrite member references inside a method body to `self.<member>`:
        #   - `this->x` / `this.x` / `(*this).x` -> `self.x`; bare `this` -> `self`
        #   - a bare identifier equal to a data-member name that is NOT a
        #     parameter and NOT a local variable declared in the body -> self.x
        # For a static method there is no `self`, so only the `this` forms (which
        # cannot legally appear) are normalized and bare members are left as-is.
        body = method.body
        # Normalize the various `this` spellings to a single `self` token first.
        body = re.sub(r"\(\s*\*\s*this\s*\)\s*(?=\.|->)", "self", body)
        body = re.sub(r"\bthis\s*->\s*", "self.", body)
        body = re.sub(r"\bthis\s*\.\s*", "self.", body)
        body = re.sub(r"\bthis\b", "self", body)
        body = self._substitute_static_data_member_initializers(struct, method, body)
        if method.is_static:
            return body

        shadowed = set(method.parameter_names)
        members = struct.data_member_names - shadowed
        if not members:
            return body
        return self._qualify_member_references(
            body,
            members,
            self._local_variable_shadow_scopes(body),
        )

    def _qualify_member_references(
        self,
        body: str,
        members: Set[str],
        local_shadows: Optional[Dict[str, List[Tuple[int, int, int]]]] = None,
    ) -> str:
        # Walk identifiers and prefix bare member references with `self.`, while
        # leaving member accesses on OTHER objects (`obj.x`, `obj->x`) and
        # already-qualified `self.x` untouched.
        result: List[str] = []
        i = 0
        n = len(body)
        while i < n:
            ch = body[i]
            if ch in "\"'":
                literal, consumed = self._read_string(body, i)
                result.append(literal)
                i += consumed
                continue
            if body.startswith("//", i):
                end = body.find("\n", i)
                if end == -1:
                    result.append(body[i:])
                    break
                result.append(body[i:end])
                i = end
                continue
            if body.startswith("/*", i):
                end = body.find("*/", i + 2)
                if end == -1:
                    result.append(body[i:])
                    break
                result.append(body[i : end + 2])
                i = end + 2
                continue
            if ch.isalpha() or ch == "_":
                ident, consumed = self._read_identifier(body, i)
                if (
                    ident in members
                    and not any(
                        declaration_position <= i and scope_start <= i < scope_end
                        for declaration_position, scope_start, scope_end in (
                            local_shadows or {}
                        ).get(ident, ())
                    )
                    and not self._is_member_identifier_context(body, i)
                    and not self._identifier_is_declaration_or_call(
                        body, i, i + consumed
                    )
                ):
                    result.append(f"self.{ident}")
                else:
                    result.append(ident)
                i += consumed
                continue
            result.append(ch)
            i += 1
        return "".join(result)
