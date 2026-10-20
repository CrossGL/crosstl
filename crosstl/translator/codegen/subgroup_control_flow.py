"""Converge proven subgroup control flow before software barrier analysis."""

import math
import re
from copy import deepcopy

from ..arithmetic_conversions import ArithmeticScalarKind, source_integer_shape
from ..ast import (
    ArrayAccessNode,
    ArrayLiteralNode,
    ArrayType,
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    BreakNode,
    CastNode,
    ConstructorNode,
    ContinueNode,
    CooperativeMatrixOpNode,
    DoWhileNode,
    ExpressionStatementNode,
    ForInNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    LoopNode,
    MemberAccessNode,
    NamedType,
    PointerType,
    PrimitiveType,
    ReferenceType,
    ReturnNode,
    StructMemberNode,
    StructNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
    VectorType,
    WaveOpNode,
    WhileNode,
)
from .loop_induction import (
    checked_induction_value,
    fixed_stride_loop_terminates,
    geometric_loop_terminates,
    walk_control_scopes,
)
from .subgroup_uniformity import subgroup_uniformity

_FLOATS = {
    "float",
    "float32_t",
    "half",
    "float16_t",
    # CrossGL spells binary16 as ``float16`` and ``f16``.
    "float16",
    "f16",
    "bfloat",
    "bfloat16",
    "bfloat16_t",
}
_SCALARS = {"bool", "int", "uint", "int32_t", "uint32_t"} | _FLOATS
_VOTES = {"WaveActiveAnyTrue", "WaveActiveAllTrue"}
_REDUCTIONS = {"WaveActiveSum", "WaveActiveProduct", "WaveActiveMin", "WaveActiveMax"}


def _type_name(value, aliases=None):
    """A type's spelling, resolved through ``aliases`` when they are given."""
    name = value if isinstance(value, str) else getattr(value, "name", None)
    seen = set()
    while aliases and name in aliases and name not in seen:
        seen.add(name)
        name = aliases[name]
    return name


def _type_aliases(nodes):
    """Program type aliases, such as ``typedef f16 float16_t;``, by alias name."""
    aliases = {}
    for node in nodes:
        if isinstance(node, VariableNode) and getattr(node, "is_type_alias", False):
            target = _type_name(node.var_type)
            if isinstance(target, str) and target != node.name:
                aliases[node.name] = target
    return aliases


def _statements(body):
    if isinstance(body, BlockNode):
        return list(body.statements)
    if isinstance(body, list):
        return list(body)
    return [] if body is None else [body]


def _operation(node):
    if isinstance(node, WaveOpNode):
        return node.operation
    if isinstance(node, FunctionCallNode):
        function = node.function
        return (
            function if isinstance(function, str) else getattr(function, "name", None)
        )
    return None


def _safe_value(
    node, names, map_operator, declared_functions, *, speculate=False, aliases=None
):
    def safe(child):
        return _safe_value(
            child,
            names,
            map_operator,
            declared_functions,
            speculate=speculate,
            aliases=aliases,
        )

    def safe_conversion(type_name):
        type_name = _type_name(type_name, aliases)
        return type_name in _SCALARS and (
            not speculate or type_name in {"bool"} | _FLOATS
        )

    if isinstance(node, LiteralNode):
        return True
    if isinstance(node, IdentifierNode):
        return node.name in names
    if isinstance(node, CastNode):
        return safe_conversion(_type_name(node.target_type)) and safe(node.expression)
    if isinstance(node, ConstructorNode):
        return (
            safe_conversion(_type_name(node.constructor_type))
            and not node.named_arguments
            and len(node.arguments) == 1
            and safe(node.arguments[0])
        )
    if isinstance(node, FunctionCallNode):
        return (
            _type_name(_operation(node), aliases) in _SCALARS
            and _operation(node) not in declared_functions
            and safe_conversion(_operation(node))
            and len(node.arguments) == 1
            and not getattr(node, "generic_args", None)
            and safe(node.arguments[0])
        )
    if isinstance(node, UnaryOpNode):
        return map_operator(node.operator) in {"!", "~", "+", "-"} and safe(
            node.operand
        )
    if isinstance(node, BinaryOpNode):
        return map_operator(node.operator) in {
            "+",
            "-",
            "*",
            "==",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
            "&",
            "|",
            "^",
        } and all(safe(part) for part in (node.left, node.right))
    return False


def _collective_value(node, declared_functions, aliases=None):
    """Look through pure floating conversions without removing their rounding."""
    while True:
        if (
            isinstance(node, CastNode)
            and _type_name(node.target_type, aliases) in _FLOATS
        ):
            node = node.expression
        elif (
            isinstance(node, ConstructorNode)
            and _type_name(node.constructor_type, aliases) in _FLOATS
            and not node.named_arguments
            and len(node.arguments) == 1
        ):
            node = node.arguments[0]
        elif (
            isinstance(node, FunctionCallNode)
            and _type_name(_operation(node), aliases) in _FLOATS
            and _operation(node) not in declared_functions
            and not getattr(node, "generic_args", None)
            and len(node.arguments) == 1
        ):
            node = node.arguments[0]
        else:
            return node


def _guarded_return_plan(
    function, declared_names, map_operator, declared_functions, aliases=None
):
    if _type_name(function.return_type, aliases) not in _SCALARS - {"bool"}:
        return None
    parameters = function.parameters or []
    # Unused receiver/resource parameters need not disqualify a pure method.
    # Only by-value scalar parameters may occur in speculated expressions.
    names = {
        parameter.name
        for parameter in parameters
        if _type_name(parameter.param_type, aliases) in _SCALARS
        and not set(parameter.qualifiers or []) & {"out", "inout", "volatile"}
    }
    names.update({"NAN", "INFINITY"} - declared_names)
    statements = _statements(function.body)
    alias = None
    if statements and isinstance(statements[0], VariableNode):
        alias = statements.pop(0)
        if _type_name(alias.var_type, aliases) != "bool" or alias.name in names:
            return None
    if not statements or not isinstance(statements[0], IfNode):
        return None
    branch = statements[0]
    if branch.else_if_conditions or branch.else_if_bodies:
        return None
    early = _statements(branch.then_branch)
    tail = (
        _statements(branch.else_branch)
        if len(statements) == 1
        else statements[1:] if branch.else_branch is None else []
    )
    if (
        len(early) != 1
        or len(tail) != 1
        or not isinstance(early[0], ReturnNode)
        or not isinstance(tail[0], ReturnNode)
    ):
        return None
    condition = branch.condition
    vote = condition
    if isinstance(vote, UnaryOpNode) and map_operator(vote.operator) == "!":
        vote = vote.operand
    if alias is not None:
        if not isinstance(vote, IdentifierNode) or vote.name != alias.name:
            return None
        vote = alias.initial_value
    reduction = _collective_value(tail[0].value, declared_functions, aliases)
    if _operation(vote) not in _VOTES or _operation(reduction) not in _REDUCTIONS:
        return None
    if _operation(vote) in declared_names or _operation(reduction) in declared_names:
        return None
    for call in (vote, reduction):
        if len(call.arguments) != 1 or not _safe_value(
            call.arguments[0],
            names,
            map_operator,
            declared_functions,
            speculate=call is reduction,
            aliases=aliases,
        ):
            return None
    if not _safe_value(
        early[0].value, names, map_operator, declared_functions, aliases=aliases
    ):
        return None
    return alias, condition, tail[0].value, early[0].value


def converge_subgroup_guarded_returns(ast, walk_ast, map_operator):
    """Copy and lower proven pure return patterns; never mutate the caller's AST.

    A vote is uniform within each logical subgroup, not across a workgroup.
    Computing both collectives before selection preserves active subgroup inputs
    while every invocation reaches the software barriers. Memory reads, unknown
    calls, mutation and potentially trapping arithmetic are not speculated.
    """
    nodes = list(walk_ast(ast))
    declared_functions = {node.name for node in nodes if isinstance(node, FunctionNode)}
    declared_names = {
        node.name
        for node in nodes
        if not isinstance(node, IdentifierNode)
        and isinstance(getattr(node, "name", None), str)
    }
    aliases = _type_aliases(nodes)
    plans = [
        (node, plan)
        for node in nodes
        if isinstance(node, FunctionNode)
        and (
            plan := _guarded_return_plan(
                node, declared_names, map_operator, declared_functions, aliases
            )
        )
        is not None
    ]
    if not plans:
        return ast
    reserved = {
        node.name for node in nodes if isinstance(getattr(node, "name", None), str)
    }
    memo = {}
    result = deepcopy(ast, memo)

    def fresh(base):
        name = base
        while name in reserved:
            name += "_"
        reserved.add(name)
        return name

    for function, plan in plans:
        alias, condition, reduction, early_value = deepcopy(plan, memo)
        function = memo[id(function)]
        guard_name = fresh("__crossgl_subgroup_guard")
        result_name = fresh("__crossgl_subgroup_result")
        location = getattr(condition, "source_location", None)
        statements = [alias] if alias is not None else []
        statements.extend(
            [
                VariableNode(
                    guard_name,
                    PrimitiveType("bool"),
                    condition,
                    source_location=location,
                ),
                VariableNode(
                    result_name,
                    function.return_type,
                    reduction,
                    source_location=getattr(reduction, "source_location", None),
                ),
                ReturnNode(
                    TernaryOpNode(
                        IdentifierNode(guard_name),
                        early_value,
                        IdentifierNode(result_name),
                    ),
                    source_location=location,
                ),
            ]
        )
        if isinstance(function.body, BlockNode):
            function.body.statements = statements
        else:
            function.body = statements
    return result


def _always_returns(statements):
    """Whether every path through ``statements`` ends in a return."""
    if not statements:
        return False
    last = statements[-1]
    if isinstance(last, ReturnNode):
        return True
    if isinstance(last, BlockNode):
        return _always_returns(_statements(last))
    if isinstance(last, IfNode) and last.else_branch is not None:
        return all(
            _always_returns(_statements(branch))
            for branch in (last.then_branch, *last.else_if_bodies, last.else_branch)
        )
    return False


_BITCASTS = {"asfloat", "asuint", "asint", "int", "uint"}


def _generated_scalar_helper(function, walk_ast):
    """A precise ``float(float)`` helper reading and writing only its locals."""
    parameters = function.parameters or []
    if (
        _type_name(function.return_type) != "float"
        or len(parameters) != 1
        or _type_name(parameters[0].param_type) != "float"
        or parameters[0].qualifiers
        or getattr(parameters[0].param_type, "qualifiers", None)
        or "precise" not in {attribute.name for attribute in function.attributes or []}
    ):
        return False
    nodes = list(walk_ast(function.body))
    names = {parameters[0].name} | {
        node.name for node in nodes if isinstance(node, VariableNode)
    }
    callees = set()
    for node in nodes:
        if isinstance(node, FunctionCallNode):
            if _operation(node) not in _BITCASTS or node.generic_args:
                return False
            callees.add(_operation(node))
        elif isinstance(node, VariableNode):
            if _type_name(node.var_type) not in _SCALARS or node.attributes:
                return False
        elif isinstance(node, (ArrayAccessNode, MemberAccessNode)):
            return False
    return all(
        node.name in names | callees
        for node in nodes
        if isinstance(node, IdentifierNode)
    )


def converge_subgroup_conditional_calls(ast, entries, walk_ast, map_operator):
    """Stage guarded inputs around pure subgroup wrappers in compute entries.

    Only a subgroup-uniform predicate may be lifted: every participating lane
    in an enabled subgroup must retain its original input. Disabled subgroups
    execute the same barrier schedule using private zero inputs. Their memory
    accesses and stores stay guarded, and their results are never published.
    """
    nodes = list(walk_ast(ast))
    functions = {}
    for node in nodes:
        if isinstance(node, FunctionNode):
            functions.setdefault(node.name, []).append(node)
    operations = (
        _REDUCTIONS
        | _VOTES
        | {
            "WavePrefixSum",
            "WavePrefixProduct",
            "WavePrefixInclusiveSum",
            "WavePrefixInclusiveProduct",
            "WaveShuffleDown",
            "WaveReadLaneAt",
            "WaveShuffleAndFillUp",
        }
    )
    summaries = {}
    # CrossGL spells binary16 as ``float16``. Wide integers wrap on both
    # software-subgroup targets, so staged zero inputs cannot trap.
    floats = _FLOATS | {"float16"}
    scalar_types = (
        _SCALARS
        | floats
        | {
            "char",
            "uchar",
            "short",
            "ushort",
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int8_t",
            "uint8_t",
            "int16_t",
            "uint16_t",
            "int64",
            "uint64",
            "int64_t",
            "uint64_t",
        }
    )
    # Bit-preserving reinterpretations used by MLX's two-word shuffle helpers.
    bitcast_widths = {
        "int64": 64,
        "uint64": 64,
        "int64_t": 64,
        "uint64_t": 64,
        "uvec2": 64,
        "uint2": 64,
    }

    def bitcast_target(node):
        operation = _operation(node)
        if (
            not isinstance(node, FunctionCallNode)
            or not isinstance(operation, str)
            or node.generic_args
            or len(node.arguments) != 1
        ):
            return None
        match = re.fullmatch(r"as_type\s*<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>", operation)
        return match.group(1) if match else None

    value_structs = {
        node.name: {
            member.name: _type_name(member.member_type) for member in node.members
        }
        for node in nodes
        if isinstance(node, StructNode)
        and node.members
        and not (node.generic_params or node.inheritance or node.attributes)
        and len({member.name for member in node.members}) == len(node.members)
        and all(
            isinstance(member, StructMemberNode)
            and _type_name(member.member_type) in scalar_types
            and not (
                member.attributes or member.resource_qualifiers or member.default_value
            )
            and not getattr(member.member_type, "qualifiers", [])
            for member in node.members
        )
    }
    value_types = scalar_types | set(value_structs)
    pure_float_arities = {
        "min": 2,
        "max": 2,
        "exp": 1,
        "log": 1,
        "cos": 1,
        "sin": 1,
        "sqrt": 1,
        "atan2": 2,
    }
    empty_types = {
        node.name
        for node in nodes
        if isinstance(node, StructNode)
        and not (
            node.members or node.generic_params or node.inheritance or node.attributes
        )
    }
    # The Metal frontend's software binary32 square root for precise::sqrt is
    # total and deterministic, so a staged zero input can neither trap nor
    # diverge; its normalization loop is what the proofs below cannot bound.
    # Only the generated precise signature over private scalars counts.
    trusted_helpers = {
        name
        for name, candidates in functions.items()
        if re.fullmatch(r"__crossgl_metal_precise_sqrt_float_*", name)
        and len(candidates) == 1
        and _generated_scalar_helper(candidates[0], walk_ast)
    }

    def trusted_call(node, types):
        return (
            isinstance(node, FunctionCallNode)
            and _operation(node) in trusted_helpers
            and not node.generic_args
            and len(node.arguments) == 1
            and value_type(node.arguments[0], types) in {"float", "float32_t"}
        )

    def pure_float_call(node, types):
        operation = _operation(node)
        argument_types = (
            {value_type(argument, types) for argument in node.arguments}
            if isinstance(node, FunctionCallNode)
            else set()
        )
        return (
            isinstance(node, FunctionCallNode)
            and operation in pure_float_arities
            and operation not in functions
            and not node.generic_args
            and len(node.arguments) == pure_float_arities[operation]
            and (
                argument_types <= {"float", "float32_t"}
                # binary16 calls stay binary16 only when every operand is.
                or len(argument_types) == 1
                and argument_types <= {"float16", "half"}
            )
        )

    def common_type(left, right):
        if left == right:
            return left
        if left in floats or right in floats:
            return "float"
        if {left, right} & {"uint64", "uint64_t"}:
            return "uint64"
        if {left, right} & {"int64", "int64_t"}:
            return "int64"
        if left in {"uint", "uint32_t"} or right in {"uint", "uint32_t"}:
            return "uint"
        return "int"

    def value_type(node, types):
        if isinstance(node, IdentifierNode):
            return types.get(node.name)
        if isinstance(node, MemberAccessNode):
            return value_structs.get(value_type(node.object_expr, types), {}).get(
                node.member
            )
        if isinstance(node, LiteralNode):
            return _type_name(node.literal_type)
        if isinstance(node, CastNode):
            return _type_name(node.target_type)
        if isinstance(node, ConstructorNode):
            return _type_name(node.constructor_type)
        if isinstance(node, UnaryOpNode):
            return (
                "bool"
                if map_operator(node.operator) == "!"
                else value_type(node.operand, types)
            )
        if isinstance(node, BinaryOpNode):
            if map_operator(node.operator) in {
                "==",
                "!=",
                "<",
                "<=",
                ">",
                ">=",
                "&&",
                "||",
            }:
                return "bool"
            left, right = value_type(node.left, types), value_type(node.right, types)
            return common_type(left, right)
        if isinstance(node, TernaryOpNode):
            left, right = value_type(node.true_expr, types), value_type(
                node.false_expr, types
            )
            if left == right:
                return left
            # Mixed arithmetic arms take the usual arithmetic conversion.
            return (
                common_type(left, right)
                if {left, right} <= scalar_types - {"bool"}
                else None
            )
        bitcast = bitcast_target(node)
        if bitcast is not None:
            return bitcast
        operation = _operation(node)
        if operation in functions:
            function = resolve_function(node, types)
            return _type_name(function.return_type) if function is not None else None
        if operation in value_types:
            return operation
        if pure_float_call(node, types):
            argument = value_type(node.arguments[0], types)
            return argument if argument in {"float16", "half"} else "float"
        if operation in {"isnan", "isinf"} | _VOTES:
            return "bool"
        if operation in {"asfloat", "asint", "asuint"}:
            return {"asfloat": "float", "asint": "int", "asuint": "uint"}[operation]
        if operation in operations and node.arguments:
            return value_type(node.arguments[0], types)
        return None

    def integer_constant(node):
        type_name = value_type(node, {})
        shape = source_integer_shape(type_name) if type_name else None
        if shape is None or shape[2] != 1 or shape[1] > 64:
            return None
        unsigned = shape[0] == ArithmeticScalarKind.UNSIGNED_INTEGER
        if isinstance(node, LiteralNode) and type(node.value) is int:
            value = node.value
        elif isinstance(node, UnaryOpNode) and map_operator(node.operator) in {
            "+",
            "-",
        }:
            value = integer_constant(node.operand)
            if value is None or unsigned and map_operator(node.operator) == "-":
                return None
            value *= -1 if map_operator(node.operator) == "-" else 1
        else:
            return None
        minimum = 0 if unsigned else -(1 << (shape[1] - 1))
        maximum = (1 << (shape[1] - (not unsigned))) - 1
        return value if minimum <= value <= maximum else None

    def exact_integer_constant(node):
        """Exact unbounded value of a literal integer expression, else None."""
        if isinstance(node, LiteralNode):
            return node.value if type(node.value) is int else None
        if isinstance(node, UnaryOpNode) and map_operator(node.operator) in {"+", "-"}:
            value = exact_integer_constant(node.operand)
            if value is None:
                return None
            return -value if map_operator(node.operator) == "-" else value
        if isinstance(node, BinaryOpNode) and map_operator(node.operator) in {
            "+",
            "-",
            "*",
        }:
            left = exact_integer_constant(node.left)
            right = exact_integer_constant(node.right)
            if left is None or right is None:
                return None
            operator = map_operator(node.operator)
            if operator == "+":
                return left + right
            return left - right if operator == "-" else left * right
        return None

    def integer_fits(type_name, value):
        shape = source_integer_shape(type_name) if type_name else None
        if shape is None or shape[2] != 1:
            return False
        unsigned = shape[0] == ArithmeticScalarKind.UNSIGNED_INTEGER
        minimum = 0 if unsigned else -(1 << (shape[1] - 1))
        maximum = (1 << (shape[1] - (not unsigned))) - 1
        return minimum <= value <= maximum

    def integral_float_literal(node):
        """Exact integer value of a finite integral floating literal, else None."""
        negative = False
        while isinstance(node, UnaryOpNode) and map_operator(node.operator) in {
            "+",
            "-",
        }:
            negative ^= map_operator(node.operator) == "-"
            node = node.operand
        if not isinstance(node, LiteralNode) or type(node.value) is not float:
            return None
        if not math.isfinite(node.value) or not node.value.is_integer():
            return None
        return -int(node.value) if negative else int(node.value)

    def converts_safely(target, value, types):
        source = value_type(value, types)
        if target in value_structs:
            return source == target
        constant = integer_constant(value)
        if constant is None:
            constant = exact_integer_constant(value)
        if constant is None and target in scalar_types - floats - {"bool"}:
            # MLX spells integer identities as static_cast<U>(1.0f); converting
            # an exactly integral literal that fits the target is exact.
            constant = integral_float_literal(value)
        if constant is not None and target in scalar_types:
            if target in floats | {"bool"}:
                return True
            shape = source_integer_shape(target)
            unsigned = shape[0] == ArithmeticScalarKind.UNSIGNED_INTEGER
            return (
                (0 if unsigned else -(1 << (shape[1] - 1)))
                <= constant
                <= (1 << (shape[1] - (not unsigned))) - 1
            )
        return (
            source in scalar_types
            and target in scalar_types
            and (target in floats | {"bool"} or source not in floats)
        )

    def resolve_function(call, types):
        candidates = [
            function
            for function in functions.get(_operation(call), ())
            if len(function.parameters) == len(call.arguments)
        ]
        if len(candidates) > 1:
            candidates = [
                function
                for function in candidates
                if all(
                    (
                        _type_name(parameter.param_type) in scalar_types
                        and value_type(argument, types) in scalar_types
                    )
                    or (
                        _type_name(parameter.param_type) in value_types | empty_types
                        and value_type(argument, types)
                        == _type_name(parameter.param_type)
                    )
                    for parameter, argument in zip(function.parameters, call.arguments)
                )
            ]
        if len(candidates) > 1:
            # Same-arity scalar overloads (MLX's 64-bit and bool shuffles) are
            # only distinguishable by exact parameter matches.
            def exact_matches(function):
                return sum(
                    value_type(argument, types) == _type_name(parameter.param_type)
                    for parameter, argument in zip(function.parameters, call.arguments)
                )

            best = max(exact_matches(function) for function in candidates)
            candidates = [
                function for function in candidates if exact_matches(function) == best
            ]
        return candidates[0] if len(candidates) == 1 else None

    def expression_inputs(node, parameters, types, visiting, initialized):
        if (
            isinstance(node, FunctionCallNode)
            and _operation(node) in value_structs
            and _operation(node) not in functions
            and not node.generic_args
        ):
            node = ConstructorNode(NamedType(_operation(node)), node.arguments)
        if isinstance(node, IdentifierNode) and node.name in parameters:
            return (
                (parameters[node.name], False)
                if types[node.name] in value_types and node.name in initialized
                else None
            )
        if isinstance(node, MemberAccessNode) and isinstance(
            node.object_expr, IdentifierNode
        ):
            name = node.object_expr.name
            if (
                name in parameters
                and value_type(node, types) in scalar_types
                and (name in initialized or (name, node.member) in initialized)
            ):
                return parameters[name], False
            return None
        if isinstance(node, LiteralNode):
            return frozenset(), False
        if integer_constant(node) is not None:
            return frozenset(), False
        if (
            isinstance(node, (UnaryOpNode, BinaryOpNode))
            and checked_induction_value(node, {}) is not None
        ):
            return frozenset(), False
        if isinstance(node, CastNode) and converts_safely(
            _type_name(node.target_type), node.expression, types
        ):
            return expression_inputs(
                node.expression, parameters, types, visiting, initialized
            )
        if (
            isinstance(node, ConstructorNode)
            and not node.named_arguments
            and len(node.arguments) == 1
            and converts_safely(
                _type_name(node.constructor_type), node.arguments[0], types
            )
        ):
            return expression_inputs(
                node.arguments[0], parameters, types, visiting, initialized
            )
        if (
            isinstance(node, ConstructorNode)
            and _type_name(node.constructor_type) in value_structs
        ):
            fields = value_structs[_type_name(node.constructor_type)]
            if (
                node.named_arguments
                or len(fields) != len(node.arguments)
                or not all(
                    converts_safely(field_type, argument, types)
                    for field_type, argument in zip(fields.values(), node.arguments)
                )
            ):
                return None
            proofs = [
                expression_inputs(argument, parameters, types, visiting, initialized)
                for argument in node.arguments
            ]
            if any(proof is None for proof in proofs):
                return None
            return frozenset().union(*(proof[0] for proof in proofs)), any(
                proof[1] for proof in proofs
            )
        bitcast = bitcast_target(node)
        if bitcast is not None:
            source = value_type(node.arguments[0], types)
            if (
                source is None
                or bitcast_widths.get(bitcast) is None
                or bitcast_widths.get(bitcast) != bitcast_widths.get(source)
            ):
                return None
            return expression_inputs(
                node.arguments[0], parameters, types, visiting, initialized
            )
        children = None
        if isinstance(node, UnaryOpNode):
            operator = map_operator(node.operator)
            operand_type = value_type(node.operand, types)
            if (
                operator in {"!", "+"}
                or operator == "-"
                and (
                    operand_type in scalar_types - {"bool"}
                    or checked_induction_value(node, {}) is not None
                )
                or operator == "~"
                and operand_type in scalar_types - floats
            ):
                children = [node.operand]
        elif isinstance(node, BinaryOpNode):
            operator = map_operator(node.operator)
            operands = {value_type(node.left, types), value_type(node.right, types)}
            # Integer +, - and * wrap on both targets; division can trap or
            # depend on zero-filled staged inputs, so it stays floating-only.
            if (
                operator in {
                    "==",
                    "!=",
                    "<",
                    "<=",
                    ">",
                    ">=",
                    "&&",
                    "||",
                    "&",
                    "|",
                    "^",
                }
                or operator in {"+", "-", "*"}
                and operands <= scalar_types
                and "bool" not in operands
                and (
                    operands & floats
                    or (constant := exact_integer_constant(node)) is None
                    or integer_fits(value_type(node, types), constant)
                )
                or operator == "/"
                and operands <= floats
            ):
                children = [node.left, node.right]
        elif isinstance(node, TernaryOpNode):
            children = [node.condition, node.true_expr, node.false_expr]
        if children is not None:
            proofs = [
                expression_inputs(child, parameters, types, visiting, initialized)
                for child in children
            ]
            if any(proof is None for proof in proofs):
                return None
            return frozenset().union(*(proof[0] for proof in proofs)), any(
                proof[1] for proof in proofs
            )
        operation = _operation(node)
        if pure_float_call(node, types) or trusted_call(node, types):
            proofs = [
                expression_inputs(argument, parameters, types, visiting, initialized)
                for argument in node.arguments
            ]
            if any(proof is None for proof in proofs):
                return None
            return frozenset().union(*(proof[0] for proof in proofs)), any(
                proof[1] for proof in proofs
            )
        if (
            isinstance(node, FunctionCallNode)
            and operation in scalar_types
            and operation not in functions
            and len(node.arguments) == 1
            and not node.generic_args
            and converts_safely(operation, node.arguments[0], types)
        ):
            return expression_inputs(
                node.arguments[0], parameters, types, visiting, initialized
            )
        if (
            isinstance(node, FunctionCallNode)
            and operation in {"isnan", "isinf", "asfloat", "asint", "asuint"}
            and operation not in functions
            and len(node.arguments) == 1
            and not node.generic_args
            and value_type(node.arguments[0], types)
            in {"float", "float32_t", "int", "int32_t", "uint", "uint32_t"}
            | (floats if operation in {"isnan", "isinf"} else set())
        ):
            return expression_inputs(
                node.arguments[0], parameters, types, visiting, initialized
            )
        if operation in operations and operation not in functions:
            required = range(len(node.arguments))
            collective = True
        elif isinstance(node, FunctionCallNode):
            proof = wrapper(node, visiting, allow_pure=True, types=types)
            if proof is None:
                return None
            _function, required, collective = proof
            if any(
                _type_name(parameter.param_type) in value_types
                and not converts_safely(
                    _type_name(parameter.param_type), argument, types
                )
                for parameter, argument in zip(_function.parameters, node.arguments)
            ):
                return None
        else:
            return None
        dependencies = set()
        for index, argument in enumerate(node.arguments):
            if index not in required:
                if not (
                    isinstance(argument, LiteralNode)
                    or isinstance(argument, IdentifierNode)
                    and argument.name in parameters
                ):
                    return None
                continue
            proof = expression_inputs(
                argument, parameters, types, visiting, initialized
            )
            if proof is None:
                return None
            if operation in operations and index == 0 and not proof[0]:
                return None
            dependencies.update(proof[0])
            collective |= proof[1]
        return frozenset(dependencies), collective

    def body_inputs(
        body,
        parameters,
        types,
        visiting,
        result_type,
        constants=None,
        controls=frozenset(),
        initialized=None,
    ):
        # Names mark whole values; (name, field) tracks partial local construction.
        initialized = set(parameters if initialized is None else initialized)
        parameters, types, constants = (
            dict(parameters),
            dict(types),
            dict(constants or {}),
        )
        dependencies, collective = set(), False
        for statement in _statements(body):
            member = None
            if isinstance(statement, ExpressionStatementNode):
                statement = statement.expression
            if isinstance(statement, (VariableNode, AssignmentNode, ReturnNode)):
                if isinstance(statement, VariableNode):
                    allowed_qualifiers = {"const", "thread"}
                    if _type_name(statement.var_type) in floats:
                        allowed_qualifiers.add("precise")
                    if (
                        statement.name in types
                        or statement.name in controls
                        or statement.attributes
                        or set(statement.qualifiers or []) - allowed_qualifiers
                        or set(getattr(statement.var_type, "qualifiers", []) or [])
                        - allowed_qualifiers
                    ):
                        return None
                    name, target, value = (
                        statement.name,
                        _type_name(statement.var_type),
                        statement.initial_value,
                    )
                    if value is None and target in value_structs:
                        parameters[name], types[name] = frozenset(), target
                        initialized.discard(name)
                        continue
                    if value is None and target in empty_types:
                        parameters[name], types[name] = frozenset(), target
                        initialized.add(name)
                        continue
                elif isinstance(statement, AssignmentNode):
                    destination = statement.target
                    if isinstance(destination, MemberAccessNode):
                        member = destination.member
                        destination = destination.object_expr
                    operator = map_operator(statement.operator)
                    compound = (
                        operator[:-1]
                        if member is None
                        and operator in {"+=", "-=", "*=", "&=", "|=", "^="}
                        else None
                    )
                    if (
                        not isinstance(destination, IdentifierNode)
                        or destination.name not in parameters
                        or destination.name in controls
                        or (operator != "=" and compound is None)
                    ):
                        return None
                    name = destination.name
                    target, value = types[name], statement.value
                    if compound is not None:
                        # `x op= v` reads the current whole value of `x`.
                        value = BinaryOpNode(IdentifierNode(name), compound, value)
                    if member is not None:
                        target = value_structs.get(target, {}).get(member)
                        if target is None:
                            return None
                else:
                    name, target, value = None, result_type, statement.value
                if isinstance(value, ArrayLiteralNode) and target in value_structs:
                    value = ConstructorNode(NamedType(target), value.elements)
                proof = expression_inputs(
                    value, parameters, types, visiting, initialized
                )
                if (
                    proof is None
                    or target is not None
                    and not converts_safely(target, value, types)
                ):
                    return None
                if name is not None:
                    if member is None:
                        parameters[name], types[name] = proof[0], target
                        initialized.add(name)
                    else:
                        parameters[name] |= proof[0]
                        initialized.add((name, member))
                        if all(
                            (name, field) in initialized
                            for field in value_structs[types[name]]
                        ):
                            initialized.add(name)
                    constants.pop(name, None)
                    if isinstance(statement, VariableNode) and "const" in set(
                        statement.qualifiers or []
                    ) | set(getattr(statement.var_type, "qualifiers", []) or []):
                        constant = checked_induction_value(value, constants)
                        if constant is not None:
                            constants[name] = constant
            elif isinstance(statement, (BlockNode, IfNode)):
                if isinstance(statement, BlockNode):
                    branches, conditions = [statement], []
                elif (
                    isinstance(statement.condition, LiteralNode)
                    and type(statement.condition.value) is bool
                    and not statement.else_if_conditions
                    and not statement.else_if_bodies
                ):
                    branches = [
                        (
                            statement.then_branch
                            if statement.condition.value
                            else statement.else_branch
                        )
                    ]
                    conditions = []
                else:
                    branches = [
                        statement.then_branch,
                        *statement.else_if_bodies,
                        statement.else_branch,
                    ]
                    conditions = [statement.condition, *statement.else_if_conditions]
                proofs = [
                    expression_inputs(
                        condition, parameters, types, visiting, initialized
                    )
                    for condition in conditions
                ]
                results = [
                    body_inputs(
                        branch,
                        parameters,
                        types,
                        visiting,
                        result_type,
                        constants,
                        controls,
                        initialized,
                    )
                    for branch in branches
                ]
                if any(proof is None for proof in [*proofs, *results]):
                    return None
                proofs += [result[:2] for result in results]
                if conditions and any(proof[1] for proof in proofs):
                    return None
                proof = frozenset().union(*(proof[0] for proof in proofs)), any(
                    proof[1] for proof in proofs
                )
                for name in parameters:
                    parameters[name] |= frozenset().union(
                        *(result[2][name] for result in results)
                    )
                initialized.intersection_update(*(result[4] for result in results))
                constants = {
                    name: value
                    for name, value in constants.items()
                    if all(result[3].get(name) == value for result in results)
                }
            elif isinstance(statement, ForNode):
                initial, condition, update = (
                    statement.init,
                    statement.condition,
                    statement.update,
                )
                if (
                    not isinstance(initial, VariableNode)
                    or initial.name in parameters
                    or initial.name in controls
                    or initial.attributes
                    or initial.qualifiers
                    or not isinstance(condition, BinaryOpNode)
                    or not isinstance(condition.left, IdentifierNode)
                    or condition.left.name != initial.name
                    or not isinstance(update, (AssignmentNode, UnaryOpNode))
                ):
                    return None
                target = (
                    update.target
                    if isinstance(update, AssignmentNode)
                    else update.operand
                )
                if (
                    not isinstance(target, IdentifierNode)
                    or target.name != initial.name
                ):
                    return None
                terminate = (
                    geometric_loop_terminates
                    if map_operator(update.operator) in {"*=", "<<="}
                    else fixed_stride_loop_terminates
                )
                if not terminate(
                    _type_name(initial.var_type),
                    initial.initial_value,
                    condition.right,
                    update,
                    constants,
                    map_operator(condition.operator),
                ):
                    return None
                if any(
                    isinstance(child, ReturnNode) for child in walk_ast(statement.body)
                ):
                    return None
                nested_parameters = {**parameters, initial.name: frozenset()}
                nested_types = {**types, initial.name: _type_name(initial.var_type)}
                loop_controls = (
                    controls
                    | {initial.name}
                    | {
                        child.name
                        for root in (condition.right, update)
                        for child in walk_ast(root)
                        if isinstance(child, IdentifierNode)
                    }
                )
                result = body_inputs(
                    statement.body,
                    nested_parameters,
                    nested_types,
                    visiting,
                    result_type,
                    constants,
                    loop_controls,
                    initialized | {initial.name},
                )
                if result is None:
                    return None
                proof = result[:2]
                for name in parameters:
                    parameters[name] |= result[2][name]
                initialized.intersection_update(result[4])
                constants = {
                    name: value
                    for name, value in constants.items()
                    if result[3].get(name) == value
                }
            else:
                return None
            dependencies.update(proof[0])
            collective |= proof[1]
        return frozenset(dependencies), collective, parameters, constants, initialized

    def wrapper(call, visiting=frozenset(), *, allow_pure=False, types=None):
        if getattr(call, "generic_args", None):
            return None
        function = resolve_function(call, types or {})
        if function is None:
            return None
        key = id(function)
        if key in visiting:
            return None
        if key not in summaries:
            statements = _statements(function.body)
            proof = None
            if (
                statements
                and _always_returns(statements)
                and _type_name(function.return_type) in value_types
                and not getattr(function, "generic_params", None)
                and not any(
                    set(parameter.qualifiers or []) & {"out", "volatile"}
                    or set(getattr(parameter.param_type, "qualifiers", []) or [])
                    & {"volatile"}
                    for parameter in function.parameters
                )
            ):
                proof = body_inputs(
                    statements,
                    {
                        parameter.name: frozenset({i})
                        for i, parameter in enumerate(function.parameters)
                    },
                    {
                        parameter.name: _type_name(parameter.param_type)
                        for parameter in function.parameters
                    },
                    visiting | {key},
                    _type_name(function.return_type),
                )
            if proof is not None:
                for index, parameter in enumerate(function.parameters):
                    if (
                        index not in proof[0]
                        and _type_name(parameter.param_type) in empty_types
                    ):
                        continue
                    if _type_name(parameter.param_type) not in value_types or (
                        set(parameter.qualifiers or [])
                        | set(getattr(parameter.param_type, "qualifiers", []) or [])
                    ) & {
                        "out",
                        "inout",
                        "volatile",
                        "device",
                        "constant",
                        "threadgroup",
                        "groupshared",
                    }:
                        proof = None
                        break
            summaries[key] = proof[:2] if proof is not None else None
        proof = summaries[key]
        if proof is None or not (allow_pure or proof[1]):
            return None
        return (function, *proof) if allow_pure else (function, proof[0])

    def root_name(node):
        while isinstance(node, (ArrayAccessNode, MemberAccessNode)):
            node = (
                node.array_expr
                if isinstance(node, ArrayAccessNode)
                else node.object_expr
            )
        if isinstance(node, UnaryOpNode) and map_operator(node.operator) == "&":
            return root_name(node.operand)
        return node.name if isinstance(node, IdentifierNode) else None

    declared = {
        node.name
        for node in nodes
        if not isinstance(node, IdentifierNode)
        and isinstance(getattr(node, "name", None), str)
    }
    plans = []
    for stage, entry, _config in entries:
        if stage != "compute":
            continue
        names = {"gl_SubgroupID"} - declared
        for parameter in entry.parameters:
            if any(
                attribute.name == "gl_SubgroupID" for attribute in parameter.attributes
            ):
                names.add(parameter.name)
        for node in walk_ast(entry.body):
            if isinstance(node, VariableNode):
                names.discard(node.name)
                if isinstance(node.var_type, (PointerType, ReferenceType)):
                    names.discard(root_name(node.initial_value))
            elif isinstance(node, AssignmentNode):
                names.discard(root_name(node.target))
            elif isinstance(node, UnaryOpNode) and map_operator(node.operator) in {
                "++",
                "--",
                "&",
            }:
                names.discard(root_name(node.operand))
            elif isinstance(node, FunctionCallNode):
                for argument in node.arguments:
                    names.discard(root_name(argument))
        if not names:
            continue
        for node in walk_ast(entry.body):
            if (
                not isinstance(node, IfNode)
                or node.else_branch is not None
                or node.else_if_conditions
                or node.else_if_bodies
                or not _safe_value(node.condition, names, map_operator, set(functions))
                or not any(
                    isinstance(child, IdentifierNode) and child.name in names
                    for child in walk_ast(node.condition)
                )
            ):
                continue
            body = _statements(node.then_branch)
            if not body or not isinstance(body[0], VariableNode):
                continue
            declaration = body[0]
            if (
                _type_name(declaration.var_type) not in value_types
                or declaration.attributes
                or set(declaration.qualifiers or []) - {"const", "thread"}
                or set(getattr(declaration.var_type, "qualifiers", []) or [])
                - {"const", "thread"}
            ):
                continue
            call = declaration.initial_value
            if not isinstance(call, FunctionCallNode):
                continue
            proof = wrapper(call)
            if proof is None:
                continue
            function, _required = proof
            staged = {
                index
                for index, parameter in enumerate(function.parameters)
                if _type_name(parameter.param_type) in value_types
            }
            if any(
                not isinstance(argument, IdentifierNode)
                for index, argument in enumerate(call.arguments)
                if index not in staged
            ):
                continue
            # Nested collectives in argument evaluation are not staged by this pass.
            if any(
                _operation(child) in operations
                or isinstance(child, FunctionCallNode)
                and wrapper(child) is not None
                for argument in call.arguments
                for child in walk_ast(argument)
            ):
                continue
            plans.append((node, function, staged))
    if not plans:
        return ast
    reserved = {
        node.name for node in nodes if isinstance(getattr(node, "name", None), str)
    }
    memo = {}
    result = deepcopy(ast, memo)

    def fresh(base):
        name = base
        while name in reserved:
            name += "_"
        reserved.add(name)
        return name

    def zero_value(type_):
        fields = value_structs.get(_type_name(type_))
        if fields:
            return ConstructorNode(
                type_,
                [
                    zero_value(PrimitiveType(field_type))
                    for field_type in fields.values()
                ],
            )
        return LiteralNode(False if _type_name(type_) == "bool" else 0, type_)

    for node, function, required in plans:
        branch = memo[id(node)]
        body = _statements(branch.then_branch)
        declaration = body[0]
        call = declaration.initial_value
        location = getattr(branch, "source_location", None)
        guard = fresh("__crossgl_subgroup_enabled")
        statements = [
            VariableNode(
                guard, PrimitiveType("bool"), branch.condition, source_location=location
            )
        ]
        inputs = []
        arguments = []
        for index, argument in enumerate(call.arguments):
            if index not in required:
                arguments.append(argument)
                continue
            name = fresh("__crossgl_subgroup_input")
            type_ = deepcopy(function.parameters[index].param_type)
            statements.append(
                VariableNode(name, type_, zero_value(type_), source_location=location)
            )
            inputs.append(
                AssignmentNode(IdentifierNode(name), argument, source_location=location)
            )
            arguments.append(IdentifierNode(name))
        call.arguments = arguments
        call.args = arguments
        statements.extend(
            [
                IfNode(
                    IdentifierNode(guard), BlockNode(inputs), source_location=location
                ),
                declaration,
                IfNode(
                    IdentifierNode(guard), BlockNode(body[1:]), source_location=location
                ),
            ]
        )
        replacement = BlockNode(statements, source_location=location)
        # Replace every legacy alias of the original branch using the same node.
        for parent in walk_ast(result):
            for key, value in list(getattr(parent, "__dict__", {}).items()):
                if value is branch:
                    setattr(parent, key, replacement)
                elif isinstance(value, list):
                    for index, child in enumerate(value):
                        if child is branch:
                            value[index] = replacement
    return result


_INT32 = {"int", "int32_t"}
_SPECULATIVE_OPERATORS = {
    "+",
    "-",
    "*",
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
    "&&",
    "||",
    "&",
    "|",
    "^",
    "<<",
    ">>",
}


def _call_name(node):
    function = node.function
    return function if isinstance(function, str) else getattr(function, "name", None)


def _qualifiers(declaration):
    return set(declaration.qualifiers or []) | set(
        getattr(declaration.var_type, "qualifiers", []) or []
    )


def _const_int(declaration):
    return (
        isinstance(declaration, VariableNode)
        and isinstance(declaration.var_type, PrimitiveType)
        and _type_name(declaration.var_type) in _INT32
        and "const" in _qualifiers(declaration)
        and declaration.initial_value is not None
        and not declaration.attributes
    )


def _int_literal(node):
    return (
        isinstance(node, LiteralNode)
        and type(node.value) is int
        and _type_name(node.literal_type) in _INT32
    )


def _effect_checker(nodes, walk_ast):
    """Return ``effects(root)``: the synchronizing operations ``root`` may run.

    ``"collective"`` marks subgroup and cooperative matrix operations and
    ``"barrier"`` every barrier. A call to a user function inherits whatever
    any function of that name reaches; an unnamed callee counts as both.
    """
    functions = {}
    for node in nodes:
        if isinstance(node, FunctionNode):
            functions.setdefault(node.name, []).append(node)
    summaries = {}

    def direct(node):
        if isinstance(node, (WaveOpNode, CooperativeMatrixOpNode)):
            return {"collective"}
        if not isinstance(node, FunctionCallNode):
            return set()
        name = _call_name(node)
        if not isinstance(name, str):
            return {"collective", "barrier"}
        if name in functions:
            return set().union(*(summary(function) for function in functions[name]))
        if "barrier" in name.lower():
            return {"barrier"}
        if name.startswith(("Wave", "Quad", "subgroup", "simd_", "quad_")):
            return {"collective"}
        return set()

    def summary(function):
        key = id(function)
        if key not in summaries:
            # A recursive cycle conservatively reaches everything.
            summaries[key] = {"collective", "barrier"}
            summaries[key] = effects(function.body)
        return summaries[key]

    def effects(root):
        return set().union(*(direct(child) for child in walk_ast(root)))

    return effects


def _speculative(node, map_operator):
    """Whether every invocation may evaluate ``node`` without side effects.

    Inactive invocations evaluate the headers of the loops that keep their
    collectives unguarded, so those headers may not call, read memory or
    write anything other than their counters.
    """
    if isinstance(node, (LiteralNode, IdentifierNode)):
        return True
    if isinstance(node, CastNode):
        return _speculative(node.expression, map_operator)
    if isinstance(node, MemberAccessNode):
        return isinstance(node.object_expr, IdentifierNode)
    if isinstance(node, UnaryOpNode):
        return map_operator(node.operator) in {"+", "-", "!", "~"} and _speculative(
            node.operand, map_operator
        )
    if isinstance(node, BinaryOpNode):
        return map_operator(node.operator) in _SPECULATIVE_OPERATORS and all(
            _speculative(part, map_operator) for part in (node.left, node.right)
        )
    return False


def _speculative_header(loop, map_operator):
    initializer, condition, update = loop.init, loop.condition, loop.update
    if (
        not isinstance(initializer, VariableNode)
        or isinstance(initializer.var_type, (PointerType, ReferenceType))
        or initializer.initial_value is None
        or not _speculative(initializer.initial_value, map_operator)
        or condition is None
        or not _speculative(condition, map_operator)
    ):
        return False
    if isinstance(update, UnaryOpNode):
        return (
            map_operator(update.operator) in {"++", "--"}
            and isinstance(update.operand, IdentifierNode)
            and update.operand.name == initializer.name
        )
    return (
        isinstance(update, AssignmentNode)
        and map_operator(update.operator) in {"+=", "-="}
        and isinstance(update.target, IdentifierNode)
        and update.target.name == initializer.name
        and _speculative(update.value, map_operator)
    )


def _guarded_segment(segment, later, active, walk_ast):
    """Guard collective-free ``segment``; keep names that ``later`` reads.

    A declaration that a later statement may name moves before the guard.
    Only uninitialized locals move: inactive invocations would otherwise skip
    their initialization, and an earlier use of the name in the segment would
    otherwise bind to the moved declaration instead of an outer one.
    """
    if not segment:
        return []
    named = {
        child.name for child in walk_ast(later) if isinstance(child, IdentifierNode)
    }
    hoisted, guarded = [], []
    for statement in segment:
        if isinstance(statement, VariableNode) and statement.name in named:
            if (
                statement.initial_value is not None
                or "const" in _qualifiers(statement)
                or statement.attributes
                or isinstance(statement.var_type, (PointerType, ReferenceType))
                or any(
                    isinstance(child, IdentifierNode) and child.name == statement.name
                    for child in walk_ast(guarded)
                )
            ):
                return None
            hoisted.append(statement)
        else:
            guarded.append(statement)
    if guarded:
        hoisted.append(IfNode(IdentifierNode(active), BlockNode(guarded)))
    return hoisted


def _guarded_statements(statements, active, effects, map_operator, walk_ast):
    """Predicate ``statements`` on ``active`` while every collective still runs.

    Without a loop that holds a collective, the statements form one guarded
    block, whose reductions the software subgroup lowering masks. A loop
    holding a collective instead runs in every invocation, with its body
    guarded in turn; its header must be speculative, and the code generator
    still proves its trip count uniform. Collective-free statements between
    such loops are guarded. Any other collective placement is rejected.
    """
    kinds = [effects(statement) for statement in statements]
    loops = [
        isinstance(statement, ForNode) and "collective" in kind
        for statement, kind in zip(statements, kinds)
    ]
    if not any(loops):
        return [IfNode(IdentifierNode(active), BlockNode(list(statements)))]
    result, segment = [], []
    for index, (statement, kind, loop) in enumerate(zip(statements, kinds, loops)):
        if not kind:
            segment.append(statement)
            continue
        if not loop or not _speculative_header(statement, map_operator):
            return None
        flushed = _guarded_segment(segment, statements[index:], active, walk_ast)
        body = _guarded_statements(
            _statements(statement.body), active, effects, map_operator, walk_ast
        )
        if flushed is None or body is None:
            return None
        result.extend(flushed)
        result.append(
            ForNode(
                statement.init,
                statement.condition,
                statement.update,
                BlockNode(body),
                source_location=getattr(statement, "source_location", None),
            )
        )
        segment = []
    flushed = _guarded_segment(segment, [], active, walk_ast)
    return None if flushed is None else result + flushed


def _unit_increment(update, counter, map_operator):
    if isinstance(update, UnaryOpNode):
        return (
            map_operator(update.operator) == "++"
            and isinstance(update.operand, IdentifierNode)
            and update.operand.name == counter
        )
    return (
        isinstance(update, AssignmentNode)
        and map_operator(update.operator) == "+="
        and isinstance(update.target, IdentifierNode)
        and update.target.name == counter
        and _int_literal(update.value)
        and update.value.value == 1
    )


def _capped_stride(value, start, map_operator, functions):
    """Return ``R`` from ``min(X, start + R)`` with either operand order."""
    if (
        not isinstance(value, FunctionCallNode)
        or _call_name(value) != "min"
        or "min" in functions
        or getattr(value, "generic_args", None)
        or len(value.arguments) != 2
    ):
        return None
    for argument in value.arguments:
        if (
            not isinstance(argument, BinaryOpNode)
            or map_operator(argument.operator) != "+"
        ):
            continue
        for first, second in (
            (argument.left, argument.right),
            (argument.right, argument.left),
        ):
            if isinstance(first, IdentifierNode) and first.name == start:
                if isinstance(second, IdentifierNode) or _int_literal(second):
                    return second
    return None


def _counter_written(body, counter, functions, map_operator, walk_ast):
    def root(node):
        while isinstance(node, (ArrayAccessNode, MemberAccessNode)):
            node = (
                node.array_expr
                if isinstance(node, ArrayAccessNode)
                else node.object_expr
            )
        return node.name if isinstance(node, IdentifierNode) else None

    for node, visible in walk_control_scopes(body, {counter}):
        if counter not in visible:
            continue
        if isinstance(node, AssignmentNode) and root(node.target) == counter:
            return True
        if (
            isinstance(node, UnaryOpNode)
            and map_operator(node.operator) in {"++", "--", "&"}
            and root(node.operand) == counter
        ):
            return True
        if (
            isinstance(node, VariableNode)
            and isinstance(node.var_type, (PointerType, ReferenceType))
            and any(
                isinstance(child, IdentifierNode) and child.name == counter
                for child in walk_ast(node.initial_value)
            )
        ):
            return True
        if (
            isinstance(node, FunctionCallNode)
            and _call_name(node) in functions
            and any(root(argument) == counter for argument in node.arguments)
        ):
            return True
    return False


def _bounded_loop_plan(statements, index, effects, functions, map_operator, walk_ast):
    loop = statements[index]
    initializer, condition = loop.init, loop.condition
    if (
        not isinstance(initializer, VariableNode)
        or not isinstance(initializer.var_type, PrimitiveType)
        or _type_name(initializer.var_type) not in _INT32
        or initializer.attributes
        or _qualifiers(initializer)
        or not isinstance(initializer.initial_value, IdentifierNode)
        or not isinstance(condition, BinaryOpNode)
        or map_operator(condition.operator) != "<"
        or not isinstance(condition.left, IdentifierNode)
        or condition.left.name != initializer.name
        or not isinstance(condition.right, IdentifierNode)
    ):
        return None
    counter = initializer.name
    start, bound = initializer.initial_value.name, condition.right.name
    if len({counter, start, bound}) != 3 or not _unit_increment(
        loop.update, counter, map_operator
    ):
        return None
    positions = {
        statement.name: position
        for position, statement in enumerate(statements[:index])
        if isinstance(statement, VariableNode)
    }
    if (
        start not in positions
        or bound not in positions
        or positions[start] > positions[bound]
        or not _const_int(statements[positions[start]])
        or not _const_int(statements[positions[bound]])
    ):
        return None
    stride = _capped_stride(
        statements[positions[bound]].initial_value, start, map_operator, functions
    )
    if stride is None:
        return None
    if isinstance(stride, IdentifierNode) and (
        stride.name in {counter, start, bound}
        or stride.name not in positions
        or positions[stride.name] > positions[bound]
        or not _const_int(statements[positions[stride.name]])
    ):
        return None
    kinds = effects(loop.body)
    if (
        "collective" not in kinds
        or "barrier" in kinds
        or any(
            isinstance(child, (BreakNode, ContinueNode, ReturnNode))
            for child in walk_ast(loop.body)
        )
        or any(
            isinstance(statement, VariableNode) and statement.name == counter
            for statement in _statements(loop.body)
        )
        or _counter_written(loop.body, counter, functions, map_operator, walk_ast)
    ):
        return None
    return {"counter": counter, "start": start, "bound": bound, "stride": stride}


def converge_subgroup_bounded_loops(ast, entries, walk_ast, map_operator):
    """Give per-subgroup key ranges a workgroup-uniform trip count.

    MLX's two-pass GQA attention gives each SIMD group its own key range::

        const int s1 = min(kend, s0 + sub);
        for (int t = s0; t < s1; t++) { ... simd_sum(score) ... }

    ``s0`` depends on the SIMD group, so the trip count differs between the
    groups of a workgroup and software subgroup barriers inside the loop
    would diverge. ``s1 <= s0 + R`` caps every trip count at ``R``, so the
    loop becomes::

        for (int step = 0; step < R; step++) {
            int t = s0 + step;
            bool active = t >= s0 && t < s1;
            if (active) { ... }  // collectives stay unguarded
        }

    Only ``const int`` locals of the loop's own statement list (or an integer
    literal for ``R``) may form the bounds, so they keep their values, and
    the body may not write ``t``, exit early or reach a barrier. For
    ``R >= 0`` this visits exactly the source's ``t = s0, ..., s1 - 1``: when
    ``s0 + R`` does not overflow, ``s1 - s0 <= R``; when it wraps upwards,
    ``s1 < s0``, the source loop is empty, and ``t >= s0`` rejects every
    wrapped ``t``. A negative ``R`` empties both loops unless ``s0 + R``
    wraps downwards, which source integer overflow would require. The code
    generator still proves ``R`` uniform and masks the guarded reductions;
    inactive invocations contribute identities and publish nothing.
    """
    nodes = list(walk_ast(ast))
    functions = {node.name for node in nodes if isinstance(node, FunctionNode)}
    effects = _effect_checker(nodes, walk_ast)
    candidates = []
    for stage, entry, _config in entries:
        if stage != "compute":
            continue
        lists = [entry.body] if isinstance(entry.body, list) else []
        lists += [
            node.statements
            for node in walk_ast(entry.body)
            if isinstance(node, BlockNode)
        ]
        for statements in lists:
            for index, statement in enumerate(statements):
                if not isinstance(statement, ForNode):
                    continue
                plan = _bounded_loop_plan(
                    statements, index, effects, functions, map_operator, walk_ast
                )
                if plan is not None and (
                    _guarded_statements(
                        _statements(statement.body),
                        "active",
                        effects,
                        map_operator,
                        walk_ast,
                    )
                    is not None
                ):
                    candidates.append((statement, plan))
    nested = {
        id(child)
        for loop, _plan in candidates
        for child in walk_ast(loop.body)
        if child is not loop
    }
    candidates = [(loop, plan) for loop, plan in candidates if id(loop) not in nested]
    if not candidates:
        return ast
    reserved = {
        node.name for node in nodes if isinstance(getattr(node, "name", None), str)
    }

    def fresh(base):
        name = base
        while name in reserved:
            name += "_"
        reserved.add(name)
        return name

    memo = {}
    result = deepcopy(ast, memo)
    for original, plan in candidates:
        loop = memo[id(original)]
        location = getattr(loop, "source_location", None)
        step = fresh("__crossgl_subgroup_step")
        active = fresh("__crossgl_subgroup_active")
        counter, start, bound = plan["counter"], plan["start"], plan["bound"]
        body = _guarded_statements(
            _statements(loop.body), active, effects, map_operator, walk_ast
        )
        if body is None:  # pragma: no cover - the copy has the planned shape
            continue
        loop.body = BlockNode(
            [
                VariableNode(
                    counter,
                    deepcopy(loop.init.var_type),
                    BinaryOpNode(IdentifierNode(start), "+", IdentifierNode(step)),
                    source_location=location,
                ),
                VariableNode(
                    active,
                    PrimitiveType("bool"),
                    BinaryOpNode(
                        BinaryOpNode(
                            IdentifierNode(counter), ">=", IdentifierNode(start)
                        ),
                        "&&",
                        BinaryOpNode(
                            IdentifierNode(counter), "<", IdentifierNode(bound)
                        ),
                    ),
                    source_location=location,
                ),
                *body,
            ],
            source_location=location,
        )
        loop.init = VariableNode(
            step,
            PrimitiveType("int"),
            LiteralNode(0, PrimitiveType("int")),
            source_location=location,
        )
        loop.condition = BinaryOpNode(
            IdentifierNode(step), "<", deepcopy(plan["stride"])
        )
        loop.update = UnaryOpNode("++", IdentifierNode(step), is_postfix=True)
    return result


_CONVERSIONS = re.compile(
    r"(?:bool|half|float|double|float(?:16|32|64)_t|u?int(?:8|16|32|64)?(?:_t)?)"
)
_WORKGROUP_BARRIERS = frozenset({"workgroupBarrier", "workgroupExecutionBarrier"})
# Updates a pure function may apply to its own value parameters and locals:
# none reads memory or can trap.
_PURE_UPDATE_OPERATORS = frozenset({"=", "+=", "-=", "*=", "&=", "|=", "^="})
_SHARED_QUALIFIERS = frozenset(
    {"threadgroup", "shared", "groupshared", "device", "constant", "static"}
)


def _literal_divisor(node, constants):
    """The nonzero integer other than -1 that ``node`` denotes, if any."""
    if isinstance(node, IdentifierNode) and node.name in constants:
        node = constants[node.name]
    if isinstance(node, CastNode):
        node = node.expression
    if (
        isinstance(node, LiteralNode)
        and type(node.value) is int
        and node.value not in {0, -1}
    ):
        return node.value
    return None


def _inactive_safe(node, constants, map_operator, private=frozenset()):
    """Whether an invocation that left early may evaluate ``node`` too.

    Only identifiers, literals, members of named values, scalar conversions,
    side-effect-free operators and integer division by a nonzero constant
    other than -1 qualify; ``private`` names private arrays and structs whose
    elements may also be read. Memory, calls and assignments never do.
    """

    def safe(child):
        return _inactive_safe(child, constants, map_operator, private)

    if isinstance(node, (LiteralNode, IdentifierNode)):
        return True
    if isinstance(node, CastNode):
        return safe(node.expression)
    if isinstance(node, MemberAccessNode):
        return isinstance(node.object_expr, IdentifierNode) or (
            _private_root(node, private) and safe(node.object_expr)
        )
    if isinstance(node, ArrayAccessNode):
        return _private_root(node, private) and all(
            safe(child) for child in (node.array_expr, node.index_expr)
        )
    if isinstance(node, UnaryOpNode):
        return map_operator(node.operator) in {"+", "-", "!", "~"} and safe(
            node.operand
        )
    if isinstance(node, BinaryOpNode):
        operator = map_operator(node.operator)
        if operator in {"/", "%"}:
            return _literal_divisor(node.right, constants) is not None and safe(
                node.left
            )
        return operator in _SPECULATIVE_OPERATORS and all(
            safe(part) for part in (node.left, node.right)
        )
    if isinstance(node, TernaryOpNode):
        return all(
            safe(part) for part in (node.condition, node.true_expr, node.false_expr)
        )
    if isinstance(node, FunctionCallNode):
        name = _call_name(node)
        if not isinstance(name, str):
            return False
        if name in constants.get("__pure__", ()):
            # A user function that only computes a value from its arguments.
            return not getattr(node, "generic_args", None) and all(
                safe(argument) for argument in node.arguments
            )
        if name in constants.get("__functions__", ()) or len(node.arguments) != 1:
            return False
        if _bitcast_name(name, node):
            # Metal's bit reinterpretation reads no memory and cannot trap.
            return safe(node.arguments[0])
        return (
            _CONVERSIONS.fullmatch(name) is not None
            and not getattr(node, "generic_args", None)
            and safe(node.arguments[0])
        )
    if isinstance(node, ConstructorNode):
        return (
            _CONVERSIONS.fullmatch(_type_name(node.constructor_type) or "") is not None
            and not node.named_arguments
            and len(node.arguments) == 1
            and safe(node.arguments[0])
        )
    return False


_AS_TYPE = re.compile(r"as_type<\s*[A-Za-z_][A-Za-z0-9_]*\s*>")


def _bitcast_name(name, node):
    """Whether a call is Metal's ``as_type<T>(x)``, however it was parsed."""
    if name == "as_type":
        return len(getattr(node, "generic_args", None) or []) == 1
    return _AS_TYPE.fullmatch(name) is not None and not getattr(
        node, "generic_args", None
    )


def _private_root(node, private):
    while isinstance(node, (ArrayAccessNode, MemberAccessNode)):
        node = (
            node.array_expr if isinstance(node, ArrayAccessNode) else node.object_expr
        )
    return isinstance(node, IdentifierNode) and node.name in private


def _private_locals(function, walk_ast):
    """Locals of ``function`` held per invocation, never shared or aliased."""
    names = {}
    for node in walk_ast(function.body):
        if not isinstance(node, VariableNode):
            continue
        if (
            isinstance(node.var_type, (PointerType, ReferenceType))
            or _qualifiers(node) & _SHARED_QUALIFIERS
            or node.attributes
            and any(attribute.name != "uniform_value" for attribute in node.attributes)
        ):
            names[node.name] = False
        else:
            names.setdefault(node.name, True)
    parameters = {parameter.name for parameter in function.parameters or []}
    return frozenset(
        name for name, private in names.items() if private and name not in parameters
    )


def _barrier_statement(statement, functions):
    expression = statement
    if isinstance(statement, ExpressionStatementNode):
        expression = statement.expression
    return (
        isinstance(expression, FunctionCallNode)
        and _call_name(expression) in _WORKGROUP_BARRIERS
        and _call_name(expression) not in functions
        and not expression.arguments
    )


_RELATIVE_SHUFFLES = frozenset(
    {
        "WaveShuffleDown",
        "WaveShuffleUp",
        "WaveShuffleXor",
        "simd_shuffle_down",
        "simd_shuffle_up",
        "simd_shuffle_xor",
    }
)


# Collectives whose result depends on which lanes take part. Invocations
# that left may run them only where every subgroup agrees on the predicate.
_UNIFORM_COLLECTIVES = (
    _RELATIVE_SHUFFLES
    | _REDUCTIONS
    | _VOTES
    | frozenset(
        {
            "WaveActiveBitAnd",
            "WaveActiveBitOr",
            "WaveActiveBitXor",
            "WaveReadLaneAt",
            "WaveReadLaneFirst",
            "WavePrefixSum",
            "WavePrefixProduct",
        }
    )
)


def _collective_call(
    node, functions, wrappers=frozenset(), operations=_RELATIVE_SHUFFLES
):
    """Whether ``node`` is a collective that invocations that left may run.

    A relative shuffle returns another lane's value, and a source lane that
    left gives an undefined value in Metal and HLSL alike, so the value it
    supplies after convergence refines the source. Sums, scans, votes,
    broadcasts and lane reads by index depend on which invocations are
    active; callers pass them in ``operations`` only where every subgroup
    agrees on the predicate (``_UNIFORM_COLLECTIVES``).
    """
    if isinstance(node, WaveOpNode):
        return node.operation in operations
    name = _call_name(node) if isinstance(node, FunctionCallNode) else None
    if isinstance(name, str) and name in wrappers:
        # A user overload set that only applies relative shuffles to its
        # arguments, as complex ``simd_shuffle_down`` shuffles both halves.
        return True
    return isinstance(name, str) and name not in functions and name in operations


def _inactive_collective_statement(
    statement,
    constants,
    private,
    functions,
    map_operator,
    operations=_RELATIVE_SHUFFLES,
):
    """``x op= collective(...)`` writing only private ``x`` from private values."""
    node = statement
    if isinstance(node, ExpressionStatementNode):
        node = node.expression
    if isinstance(node, VariableNode):
        target, value = None, node.initial_value
        if node.name not in private or value is None:
            return False
    elif isinstance(node, AssignmentNode):
        target, value = node.target, node.value
        if not _private_root(target, private) or not _inactive_safe(
            target, constants, map_operator, private
        ):
            return False
    else:
        return False

    def safe(expression):
        if _collective_call(
            expression,
            functions,
            constants.get("__collective__", frozenset()),
            operations,
        ):
            arguments = (
                expression.arguments
                if isinstance(expression, FunctionCallNode)
                else getattr(expression, "arguments", None) or []
            )
            return not getattr(expression, "generic_args", None) and all(
                safe(argument) for argument in arguments
            )
        if isinstance(expression, BinaryOpNode):
            operator = map_operator(expression.operator)
            if operator in {"/", "%"}:
                return False
            return operator in _SPECULATIVE_OPERATORS and all(
                safe(part) for part in (expression.left, expression.right)
            )
        if isinstance(expression, (CastNode,)):
            return safe(expression.expression)
        if (
            isinstance(expression, FunctionCallNode)
            and _call_name(expression) in constants.get("__pure__", ())
            and not getattr(expression, "generic_args", None)
        ):
            # A pure helper only combines its arguments, as the Metal
            # frontend's ``x = operator+=(x, simd_shuffle_down(x, n))``.
            return all(safe(argument) for argument in expression.arguments)
        return _inactive_safe(expression, constants, map_operator, private)

    return safe(value)


def _value_function_kinds(functions, walk_ast, map_operator):
    """Classify user overload sets that only compute values from arguments.

    A ``"pure"`` function takes value parameters, declares private value
    locals, assigns them or their members and ends in a return; every
    expression reads only those names, literals and conversions through
    side-effect-free operators and calls to pure functions. A
    ``"collective"`` one may also apply relative shuffles to such values,
    as the complex overload of ``simd_shuffle_down`` shuffles both halves.
    A name is classified only when each of its overloads qualifies, so any
    resolution of a call is safe for invocations that left early.
    """
    kinds = {}

    def classify(name, visiting):
        if name in kinds:
            return kinds[name]
        if name in visiting or name not in functions:
            return None
        visiting = visiting | {name}
        result = "pure"
        for function in functions[name]:
            kind = function_kind(function, visiting)
            if kind is None:
                result = None
                break
            if kind == "collective":
                result = "collective"
        kinds[name] = result
        return result

    def value_type(type_node):
        return (
            type_node is not None
            and not isinstance(type_node, (PointerType, ReferenceType))
            and _type_name(type_node) not in {None, "void"}
            and not set(getattr(type_node, "qualifiers", None) or [])
            & _SHARED_QUALIFIERS
        )

    def function_kind(function, visiting):
        if getattr(function, "generic_params", None) or not value_type(
            function.return_type
        ):
            return None
        names = set()
        for parameter in function.parameters or []:
            if (
                not value_type(parameter.param_type)
                or set(parameter.qualifiers or []) - {"in", "const", "thread"}
                or parameter.default_value is not None
            ):
                return None
            names.add(parameter.name)
        collective = False

        def safe(node):
            nonlocal collective
            if isinstance(node, (LiteralNode,)):
                return True
            if isinstance(node, IdentifierNode):
                return node.name in names
            if isinstance(node, MemberAccessNode):
                return safe(node.object_expr)
            if isinstance(node, CastNode):
                return safe(node.expression)
            if isinstance(node, UnaryOpNode):
                return map_operator(node.operator) in {"+", "-", "!", "~"} and safe(
                    node.operand
                )
            if isinstance(node, BinaryOpNode):
                operator = map_operator(node.operator)
                if operator in {"/", "%"}:
                    return _literal_divisor(node.right, {}) is not None and safe(
                        node.left
                    )
                return operator in _SPECULATIVE_OPERATORS and all(
                    safe(part) for part in (node.left, node.right)
                )
            if isinstance(node, TernaryOpNode):
                return all(
                    safe(part)
                    for part in (node.condition, node.true_expr, node.false_expr)
                )
            if isinstance(node, ConstructorNode):
                return not node.named_arguments and all(
                    safe(argument) for argument in node.arguments
                )
            if isinstance(node, WaveOpNode):
                if node.operation not in _RELATIVE_SHUFFLES:
                    return False
                collective = True
                return all(
                    safe(argument)
                    for argument in getattr(node, "arguments", None) or []
                )
            if not isinstance(node, FunctionCallNode):
                return False
            name = _call_name(node)
            if not isinstance(name, str) or not all(
                safe(argument) for argument in node.arguments
            ):
                return False
            if name in functions:
                kind = classify(name, visiting)
                collective = collective or kind == "collective"
                return kind is not None and not getattr(node, "generic_args", None)
            if _bitcast_name(name, node):
                return len(node.arguments) == 1
            if name.startswith(("Wave", "Quad", "subgroup", "simd_", "quad_")):
                # Only relative shuffles stay valid in invocations that left.
                collective = True
                return name in _RELATIVE_SHUFFLES and not getattr(
                    node, "generic_args", None
                )
            return (
                _CONVERSIONS.fullmatch(name) is not None
                and len(node.arguments) == 1
                and not getattr(node, "generic_args", None)
            )

        statements = _statements(function.body)
        if not statements or not isinstance(statements[-1], ReturnNode):
            return None
        for statement in statements:
            node = statement
            if isinstance(node, ExpressionStatementNode):
                node = node.expression
            if isinstance(node, VariableNode):
                if (
                    not value_type(node.var_type)
                    or node.attributes
                    or _qualifiers(node) & _SHARED_QUALIFIERS
                    or node.initial_value is not None
                    and not safe(node.initial_value)
                ):
                    return None
                names.add(node.name)
            elif isinstance(node, AssignmentNode):
                target = node.target
                while isinstance(target, MemberAccessNode):
                    target = target.object_expr
                operator = map_operator(node.operator)
                if (
                    operator not in _PURE_UPDATE_OPERATORS
                    or not isinstance(target, IdentifierNode)
                    or target.name not in names
                    or not safe(node.value)
                ):
                    return None
            elif isinstance(node, ReturnNode):
                if node is not statements[-1] or not safe(node.value):
                    return None
            else:
                return None
        return "collective" if collective else "pure"

    for name in functions:
        classify(name, frozenset())
    return {name: kind for name, kind in kinds.items() if kind is not None}


def _inactive_loop_header(loop, constants, map_operator):
    """A counted loop every invocation may run: a fresh scalar counter."""
    initializer, condition, update = loop.init, loop.condition, loop.update
    if (
        not isinstance(initializer, VariableNode)
        or not isinstance(initializer.var_type, (PrimitiveType, NamedType))
        or _CONVERSIONS.fullmatch(_type_name(initializer.var_type) or "") is None
        or initializer.initial_value is None
        or initializer.attributes
        or not _inactive_safe(initializer.initial_value, constants, map_operator)
        or condition is None
        or not _inactive_safe(condition, constants, map_operator)
    ):
        return False
    counter = initializer.name
    if isinstance(update, UnaryOpNode):
        return (
            map_operator(update.operator) in {"++", "--"}
            and isinstance(update.operand, IdentifierNode)
            and update.operand.name == counter
        )
    if isinstance(update, ExpressionStatementNode):
        update = update.expression
    return (
        isinstance(update, AssignmentNode)
        and map_operator(update.operator) in {"+=", "-=", "*=", ">>=", "<<="}
        and isinstance(update.target, IdentifierNode)
        and update.target.name == counter
        and _inactive_safe(update.value, constants, map_operator)
    )


def _escapes(statement, walk_ast):
    """Whether ``statement`` returns, or breaks or continues an outer loop."""
    loops = (ForNode, ForInNode, WhileNode, DoWhileNode, LoopNode)

    def visit(node, nested):
        if isinstance(node, ReturnNode):
            return True
        if isinstance(node, (BreakNode, ContinueNode)) and not nested:
            return True
        inner = nested or (isinstance(node, loops) and node is not statement)
        return any(
            visit(child, inner)
            for child in walk_ast(node)
            if child is not node and _direct_child(node, child)
        )

    return visit(statement, isinstance(statement, loops))


def _direct_child(parent, child):
    for value in vars(parent).values():
        if value is child:
            return True
        if isinstance(value, list) and any(item is child for item in value):
            return True
    return False


def _const_literals(statements):
    """Constant locals initialized with nonzero literals other than -1."""
    constants = {}
    for statement in statements:
        if (
            isinstance(statement, VariableNode)
            and (not statement.is_mutable or "const" in _qualifiers(statement))
            and _literal_divisor(statement.initial_value, {}) is not None
        ):
            constants[statement.name] = statement.initial_value
    return constants


def _written_roots(root, walk_ast):
    names = set()
    for node in walk_ast(root):
        target = None
        if isinstance(node, AssignmentNode):
            target = node.target
        elif isinstance(node, UnaryOpNode) and node.operator in {"++", "--"}:
            target = node.operand
        while isinstance(target, (ArrayAccessNode, MemberAccessNode)):
            target = (
                target.array_expr
                if isinstance(target, ArrayAccessNode)
                else target.object_expr
            )
        if isinstance(target, IdentifierNode):
            names.add(target.name)
    return names


_ZERO_SCALARS = re.compile(
    r"(?:bool|half|float|double|float(?:16|32|64)_t|u?int(?:8|16|32|64)?(?:_t)?)"
)
# CrossGL's own spellings of half and bfloat16 scalars, as MLX's masked
# GEMV's ``T out_scale{1};`` reaches the code generators. They start from
# ``T(0)``: the code generators may store them in another representation.
_ZERO_CONVERTED_SCALARS = frozenset(
    {"float16", "float32", "float64", "bfloat", "bfloat16", "bfloat16_t"}
)
_CONSTANT_QUALIFIERS = frozenset({"const", "constexpr"})


def _predicated_segment(segment, later, active, walk_ast):
    """Run ``segment`` under ``if (active)``, keeping names ``later`` reads.

    Like ``_guarded_segment``, but a scalar declared with a value becomes a
    zero-initialized (``false`` for ``bool``) declaration before the guard
    and an assignment inside it, so invocations that left read a defined
    zero. A constant scalar, as MLX's masked GEMV's ``const int
    mat_mask_step = mat_mask_strides[0];``, loses its ``const``: it is
    assigned once, under the guard, and the code generators never treat a
    value read from memory as a compile-time constant.
    """
    if not segment:
        return []
    named = {
        child.name for child in walk_ast(later) if isinstance(child, IdentifierNode)
    }
    hoisted, guarded = [], []
    for statement in segment:
        if not isinstance(statement, VariableNode) or statement.name not in named:
            guarded.append(statement)
            continue
        constant = bool(_qualifiers(statement) & _CONSTANT_QUALIFIERS) or (
            not statement.is_mutable
        )
        if (
            constant
            and statement.initial_value is None
            or statement.attributes
            or isinstance(statement.var_type, (PointerType, ReferenceType))
            or any(
                isinstance(child, IdentifierNode) and child.name == statement.name
                for child in walk_ast(guarded)
            )
        ):
            return None
        if statement.initial_value is None:
            hoisted.append(statement)
            continue
        type_name = _type_name(statement.var_type) or ""
        if not isinstance(statement.var_type, (PrimitiveType, NamedType)) or not (
            _ZERO_SCALARS.fullmatch(type_name) or type_name in _ZERO_CONVERTED_SCALARS
        ):
            return None
        location = getattr(statement, "source_location", None)
        declared_type = deepcopy(statement.var_type)
        if getattr(declared_type, "qualifiers", None):
            declared_type.qualifiers = [
                item
                for item in declared_type.qualifiers
                if item not in _CONSTANT_QUALIFIERS
            ]
        if type_name in _ZERO_CONVERTED_SCALARS:
            zero = FunctionCallNode(
                IdentifierNode(type_name), [LiteralNode(0, PrimitiveType("int"))]
            )
        else:
            zero = LiteralNode(
                False if type_name == "bool" else 0, deepcopy(declared_type)
            )
        hoisted.append(
            VariableNode(
                statement.name,
                declared_type,
                zero,
                qualifiers=[
                    item
                    for item in statement.qualifiers or []
                    if item not in _CONSTANT_QUALIFIERS
                ],
                source_location=location,
            )
        )
        guarded.append(
            ExpressionStatementNode(
                AssignmentNode(
                    IdentifierNode(statement.name), statement.initial_value, "="
                ),
                source_location=location,
            )
        )
    if guarded:
        hoisted.append(IfNode(IdentifierNode(active), BlockNode(guarded)))
    return hoisted


def _exit_branch(statement, context):
    """The statements before the ``return`` of ``if (c) { ...; return; }``.

    Only a branch without ``else`` whose last statement is a bare return
    qualifies, and the statements before it may neither leave, break, or
    continue nor synchronize. Returns ``None`` for any other statement.
    """
    if (
        not isinstance(statement, IfNode)
        or statement.else_branch is not None
        or getattr(statement, "else_if_conditions", None)
    ):
        return None
    body = _statements(statement.then_branch)
    if not body or not isinstance(body[-1], ReturnNode) or body[-1].value is not None:
        return None
    walk_ast = context["walk_ast"]
    prefix = body[:-1]
    if any(_escapes(item, walk_ast) for item in prefix) or (
        prefix and context["effects"](BlockNode(prefix)) & {"collective", "barrier"}
    ):
        return None
    return prefix


def _predicated_exit(statement, active, context):
    """Lower a later ``if (c) { S; return; }`` for still-active invocations.

    ``c`` is evaluated where ``active`` holds, as in the source; ``S`` runs
    where both hold, and those invocations then clear ``active`` so every
    later statement without a collective or barrier skips them.
    """
    prefix = _exit_branch(statement, context)
    condition = statement.condition
    location = getattr(statement, "source_location", None)
    taken = context["fresh"]("__crossgl_subgroup_exit")
    result = []
    if isinstance(condition, LiteralNode) and condition.value in {True, 1}:
        result.append(
            VariableNode(
                taken,
                PrimitiveType("bool"),
                IdentifierNode(active),
                source_location=location,
            )
        )
    elif _inactive_safe(
        condition, context["constants"], context["map_operator"], context["private"]
    ):
        result.append(
            VariableNode(
                taken,
                PrimitiveType("bool"),
                BinaryOpNode(IdentifierNode(active), "&&", condition),
                source_location=location,
            )
        )
    else:
        result += [
            VariableNode(
                taken,
                PrimitiveType("bool"),
                LiteralNode(False, PrimitiveType("bool")),
                source_location=location,
            ),
            IfNode(
                IdentifierNode(active),
                BlockNode(
                    [
                        ExpressionStatementNode(
                            AssignmentNode(IdentifierNode(taken), condition, "="),
                            source_location=location,
                        )
                    ]
                ),
            ),
        ]
    if prefix:
        result.append(IfNode(IdentifierNode(taken), BlockNode(prefix)))
    result.append(
        ExpressionStatementNode(
            AssignmentNode(
                IdentifierNode(active),
                BinaryOpNode(
                    IdentifierNode(active),
                    "&&",
                    UnaryOpNode("!", IdentifierNode(taken)),
                ),
                "=",
            ),
            source_location=location,
        )
    )
    return result


def _uniform_declaration_read_after_sync(statements, index, context):
    """Whether ``statements[index]`` is a ``uniform_value`` scalar to keep.

    Only a constant scalar whose attributes all mark it uniform and whose
    initializer is inactive-safe qualifies, and only when a later statement
    that synchronizes (or leaves) reads it: a predicated copy would then
    reach code every invocation runs. The attribute is not trusted; the code
    generators still prove every loop bound it feeds uniform.
    """
    statement = statements[index]
    if (
        not isinstance(statement, VariableNode)
        or statement.initial_value is None
        or not statement.attributes
        or any(attribute.name != "uniform_value" for attribute in statement.attributes)
        or not (
            _qualifiers(statement) & _CONSTANT_QUALIFIERS or not statement.is_mutable
        )
        or isinstance(statement.var_type, (PointerType, ReferenceType))
        or _qualifiers(statement) & _SHARED_QUALIFIERS
        or not _inactive_safe(
            statement.initial_value, context["constants"], context["map_operator"]
        )
    ):
        return False
    walk_ast = context["walk_ast"]

    def read(root):
        return {
            node.name for node in walk_ast(root) if isinstance(node, IdentifierNode)
        }

    for first in range(index + 1, len(statements)):
        later = statements[first]
        if (
            context["effects"](later) & {"collective", "barrier"}
            or _escapes(later, walk_ast)
            or _exit_branch(later, context) is not None
        ):
            break
    else:
        return False
    # MLX's ``n_iter = in_size / loop_stride`` also keeps ``in_size`` and
    # ``loop_stride``: later uniform declarations needed in turn.
    needed = read(BlockNode(statements[first:]))
    for later in reversed(statements[index + 1 : first]):
        if (
            isinstance(later, VariableNode)
            and later.name in needed
            and later.attributes
            and later.initial_value is not None
        ):
            needed |= read(later.initial_value)
    return statement.name in needed


def _early_exit_statements(statements, active, context):
    """Predicate ``statements`` after an early exit; ``None`` if unproven.

    Statements without collectives or barriers run under ``if (active)``.
    Everything else must be safe in invocations that left: barriers, loops
    with fresh counters and inactive-safe headers, branches on literal
    conditions, inactive-safe declarations and collective updates of private
    values. Inactive invocations thus publish nothing and only take part in
    collectives and barriers, as the code generators' masking requires.
    """
    effects, walk_ast, map_operator = (
        context["effects"],
        context["walk_ast"],
        context["map_operator"],
    )
    constants, private, functions = (
        context["constants"],
        context["private"],
        context["functions"],
    )
    result, segment = [], []

    def flush(later):
        nonlocal segment
        guarded = _predicated_segment(segment, later, active, walk_ast)
        segment = []
        if guarded is None:
            return False
        result.extend(guarded)
        return True

    for index, statement in enumerate(statements):
        if _exit_branch(statement, context) is not None:
            # A later ``if (c) { ...; return; }``: the invocations taking it
            # run its statements and leave, which clears their ``active``.
            if not flush(statements[index:]):
                return None
            if not _subgroup_uniform(statement.condition, context):
                context["uniform_state"]["active"] = False
            result.extend(_predicated_exit(statement, active, context))
            continue
        if _escapes(statement, walk_ast):
            if not isinstance(statement, IfNode) or (
                statement.else_branch is not None
                or getattr(statement, "else_if_conditions", None)
                or not isinstance(statement.condition, LiteralNode)
            ):
                return None
            if statement.condition.value in {False, 0}:
                continue  # never runs, so neither do its exits
            if statement.condition.value not in {True, 1}:
                return None
            # MLX's ``if (has_output_mask) { ...; if (!mask_out) { ...;
            # return; } }`` once its template constant is folded.
            if not flush(statements[index:]):
                return None
            block = _predicated_if(statement, active, context)
            if block is None:
                return None
            result.append(block)
            continue
        kinds = effects(statement) & {"collective", "barrier"}
        if _uniform_declaration_read_after_sync(statements, index, context):
            # MLX's ``const uniform<int> n_iter = in_size / loop_stride;``
            # bounds GEMV-T's barrier loop, which every invocation runs, so
            # every invocation computes it. It reads no memory and cannot
            # trap; predicating it would make the bound appear to diverge.
            if not flush(statements[index:]):
                return None
            result.append(statement)
            if _literal_divisor(statement.initial_value, {}) is not None:
                constants[statement.name] = statement.initial_value
            continue
        if (
            isinstance(statement, VariableNode)
            and statement.initial_value is not None
            and not isinstance(statement.var_type, ReferenceType)
            and (
                isinstance(statement.var_type, PointerType)
                or not _qualifiers(statement) & _SHARED_QUALIFIERS
            )
            and not statement.attributes
            and _inactive_safe(statement.initial_value, constants, map_operator)
        ):
            # Every invocation may compute it, so later code keeps its value.
            # A pointer declaration only offsets its base; nothing is read.
            if not flush(statements[index:]):
                return None
            result.append(statement)
            if not statement.is_mutable or "const" in _qualifiers(statement):
                if _literal_divisor(statement.initial_value, {}) is not None:
                    constants[statement.name] = statement.initial_value
            continue
        if not kinds:
            segment.append(statement)
            continue
        if not flush(statements[index:]):
            return None
        if _barrier_statement(statement, functions):
            result.append(statement)
        elif isinstance(statement, IfNode) and (
            isinstance(statement.condition, LiteralNode)
            and statement.condition.value in {False, 0}
            and not statement.else_branch
        ):
            result.append(statement)  # never runs
        elif (
            isinstance(statement, IfNode)
            and statement.else_branch is None
            and not getattr(statement, "else_if_conditions", None)
        ):
            block = _predicated_if(statement, active, context)
            if block is None:
                return None
            result.append(block)
        elif isinstance(statement, IfNode) and not getattr(
            statement, "else_if_conditions", None
        ):
            block = _predicated_if_else(statement, active, context)
            if block is None:
                return None
            result.append(block)
        elif isinstance(statement, ForNode):
            loop = _predicated_loop(statement, active, context)
            if loop is None:
                return None
            result.extend(loop)
        elif _inactive_collective_statement(
            statement, constants, private, functions, map_operator
        ) or (
            context.get("uniform_state", {}).get("active")
            and _inactive_collective_statement(
                statement,
                constants,
                private,
                functions,
                map_operator,
                _UNIFORM_COLLECTIVES,
            )
        ):
            guarded = _guarded_collective(statement, active, context)
            if guarded is None:
                return None
            result.extend(guarded)
        else:
            return None
    if not flush([]):
        return None
    return result


def _predicated_if(statement, active, context):
    """Predicate ``if (c) { ... }`` whose branch synchronizes; ``None`` if unproven.

    Every invocation reaches the branch's barriers and collectives; the rest
    of the branch runs where ``active`` (every invocation when ``None``) and
    the condition, evaluated once as before, both hold. A block keeps the
    branch's scope.
    """
    constants, private, map_operator = (
        context["constants"],
        context["private"],
        context["map_operator"],
    )
    condition = statement.condition
    location = getattr(statement, "source_location", None)
    prefix = []
    inner = active
    if not (isinstance(condition, LiteralNode) and condition.value in {True, 1}):
        inner = context["fresh"]("__crossgl_subgroup_taken")
        if active is None:
            # Every invocation evaluated the condition in the source too.
            prefix.append(
                VariableNode(
                    inner, PrimitiveType("bool"), condition, source_location=location
                )
            )
        elif _inactive_safe(condition, constants, map_operator, private):
            prefix.append(
                VariableNode(
                    inner,
                    PrimitiveType("bool"),
                    BinaryOpNode(IdentifierNode(active), "&&", condition),
                    source_location=location,
                )
            )
        else:
            prefix += [
                VariableNode(
                    inner,
                    PrimitiveType("bool"),
                    LiteralNode(False, PrimitiveType("bool")),
                    source_location=location,
                ),
                IfNode(
                    IdentifierNode(active),
                    BlockNode(
                        [
                            ExpressionStatementNode(
                                AssignmentNode(IdentifierNode(inner), condition, "="),
                                source_location=location,
                            )
                        ]
                    ),
                ),
            ]
    elif active is None:
        return None
    nested = dict(context, constants=dict(constants))
    if inner != active:
        nested["uniform_state"] = {
            "active": (
                active is not None
                and context.get("uniform_state", {}).get("active", False)
                and _subgroup_uniform(condition, context)
            )
        }
    body = _early_exit_statements(_statements(statement.then_branch), inner, nested)
    if body is None:
        return None
    return BlockNode(prefix + body, source_location=location)


def _guarded_collective(statement, active, context):
    """Keep a collective's update of a private value to ``active`` invocations.

    Every invocation runs the collective. Where ``active`` only marks the
    invocations that have not left, the others never read their private
    values again. A nested predicate, such as a branch's or a loop's
    ``running``, also clears in invocations that continue past it, so
    ``x op= value`` becomes ``T tmp = value; if (active) { x op= tmp; }``
    with ``T`` the declared type of ``x``. A declaration names a value
    only later predicated statements and collectives read; it stays.
    """
    node = statement
    if isinstance(node, ExpressionStatementNode):
        node = node.expression
    if active == context.get("exit_active") or isinstance(node, VariableNode):
        return [statement]
    value_type = _declared_type(node.target, context)
    if value_type is None:
        return None
    location = getattr(statement, "source_location", None)
    value = context["fresh"]("__crossgl_subgroup_value")
    return [
        VariableNode(value, deepcopy(value_type), node.value, source_location=location),
        IfNode(
            IdentifierNode(active),
            BlockNode(
                [
                    ExpressionStatementNode(
                        AssignmentNode(
                            deepcopy(node.target), IdentifierNode(value), node.operator
                        ),
                        source_location=location,
                    )
                ]
            ),
        ),
    ]


def _declared_type(target, context):
    """The scalar or vector type of a private name or of its array elements."""
    depth = 0
    while isinstance(target, ArrayAccessNode):
        target, depth = target.array_expr, depth + 1
    if not isinstance(target, IdentifierNode):
        return None
    declared = context.get("declared_types", {}).get(target.name)
    for _ in range(depth):
        if not isinstance(declared, ArrayType):
            return None
        declared = declared.element_type
    if isinstance(declared, ArrayType) or not isinstance(
        declared, (PrimitiveType, NamedType, VectorType)
    ):
        return None
    return declared


def _subgroup_uniform(expression, context):
    """Whether every invocation of a subgroup agrees on ``expression``."""
    uniformity = context.get("uniformity")
    return uniformity is not None and uniformity.expression(expression)


def _predicated_if_else(statement, active, context):
    """Keep ``if (c) { A } else { B }`` whose branches synchronize.

    Invocations that left evaluate ``c`` too, so it must be inactive-safe,
    and take one branch, where they run only its collectives and barriers;
    every other statement of ``A`` and ``B`` runs under ``active``. The code
    generators still prove ``c`` uniform before accepting the collectives.
    Nothing proves it uniform across the workgroup for a workgroup barrier,
    so a branch that synchronizes the workgroup stays unproven.
    """
    constants, map_operator = context["constants"], context["map_operator"]
    condition = statement.condition
    if "barrier" in context["effects"](statement):
        return None
    if not _inactive_safe(condition, constants, map_operator, context["private"]):
        return None
    uniform = context.get("uniform_state", {}).get(
        "active", False
    ) and _subgroup_uniform(condition, context)
    branches = []
    for branch in (statement.then_branch, statement.else_branch):
        nested = dict(
            context, constants=dict(constants), uniform_state={"active": uniform}
        )
        body = _early_exit_statements(_statements(branch), active, nested)
        if body is None:
            return None
        branches.append(BlockNode(body))
    return IfNode(
        condition,
        branches[0],
        branches[1],
        source_location=getattr(statement, "source_location", None),
    )


def _conjuncts(expression, map_operator):
    if (
        isinstance(expression, BinaryOpNode)
        and map_operator(expression.operator) == "&&"
    ):
        return _conjuncts(expression.left, map_operator) + _conjuncts(
            expression.right, map_operator
        )
    return [expression]


def _predicated_loop(statement, active, context):
    """Lower a counted loop around collectives or barriers; ``None`` if unproven.

    The loop's header must be one every invocation may run. A header
    ``i < n && p``, as MLX's quantized ``row < results && out_row + row <
    out_size``, leaves early only for some subgroups; it becomes::

        bool running = active;
        for (int i = 0; i < n; i++) {
            running = running && p;
            ...  // as for any statement list, predicated by ``running``
        }

    ``running`` holds exactly in the source's iterations: the counter is
    fresh and the body writes neither it nor anything ``p`` reads, and ``p``
    is inactive-safe. Only ``i < n`` then decides the trip count.
    """
    constants, map_operator, walk_ast = (
        context["constants"],
        context["map_operator"],
        context["walk_ast"],
    )
    counter = getattr(statement.init, "name", None)
    written = _written_roots(statement.body, walk_ast)
    location = getattr(statement, "source_location", None)
    if counter is None or counter in written:
        return None
    conjuncts = (
        _conjuncts(statement.condition, map_operator)
        if statement.condition is not None
        else [None]
    )
    head = ForNode(statement.init, conjuncts[0], statement.update, statement.body)
    if not _inactive_loop_header(head, constants, map_operator):
        return None
    nested = dict(context, constants=dict(constants))
    if len(conjuncts) == 1:
        body = _early_exit_statements(_statements(statement.body), active, nested)
        if body is None:
            return None
        return [
            ForNode(
                statement.init,
                statement.condition,
                statement.update,
                BlockNode(body),
                source_location=location,
            )
        ]
    rest = conjuncts[1]
    for conjunct in conjuncts[2:]:
        rest = BinaryOpNode(rest, "&&", conjunct)
    if not _inactive_safe(
        rest, constants, map_operator, context["private"]
    ) or _read_names(rest, walk_ast) & (written | {active}):
        return None
    running = context["fresh"]("__crossgl_subgroup_running")
    uniformity = context.get("uniformity")
    update = statement.update
    if isinstance(update, ExpressionStatementNode):
        update = update.expression
    step = update.value if isinstance(update, AssignmentNode) else None
    if uniformity is not None and all(
        uniformity.expression(part)
        for part in (statement.init.initial_value, step)
        if part is not None
    ):
        # Every invocation steps the counter through the same values.
        uniformity = uniformity.with_names({counter})
        nested["uniformity"] = uniformity
    nested["uniform_state"] = {
        "active": (
            context.get("uniform_state", {}).get("active", False)
            and uniformity is not None
            and uniformity.expression(rest)
        )
    }
    body = _early_exit_statements(_statements(statement.body), running, nested)
    if body is None:
        return None
    return [
        VariableNode(
            running,
            PrimitiveType("bool"),
            IdentifierNode(active),
            source_location=location,
        ),
        ForNode(
            statement.init,
            conjuncts[0],
            statement.update,
            BlockNode(
                [
                    ExpressionStatementNode(
                        AssignmentNode(
                            IdentifierNode(running),
                            BinaryOpNode(IdentifierNode(running), "&&", rest),
                            "=",
                        ),
                        source_location=location,
                    ),
                    *body,
                ]
            ),
            source_location=location,
        ),
    ]


def _early_exit_context(
    function,
    effects,
    walk_ast,
    map_operator,
    functions,
    kinds=None,
    fresh=None,
    uniformity=None,
):
    kinds = kinds or {}
    declared = {}
    for node in walk_ast(function.body):
        if isinstance(node, VariableNode):
            declared.setdefault(node.name, []).append(node.var_type)
    return {
        # Subgroup-uniform names of ``function``, and whether every subgroup
        # agrees on the current predicate (see ``subgroup_uniformity``).
        "uniformity": uniformity,
        "uniform_state": {"active": False},
        # The predicate of invocations that have not left; set by callers.
        "exit_active": None,
        "declared_types": {
            name: types[0]
            for name, types in declared.items()
            if all(repr(item) == repr(types[0]) for item in types)
        },
        "effects": effects,
        "walk_ast": walk_ast,
        "map_operator": map_operator,
        "constants": {
            "__functions__": frozenset(functions),
            "__pure__": frozenset(
                name for name, kind in kinds.items() if kind == "pure"
            ),
            "__collective__": frozenset(
                name for name, kind in kinds.items() if kind == "collective"
            ),
        },
        "private": _private_locals(function, walk_ast),
        "functions": frozenset(functions),
        "fresh": fresh or (lambda base: base),
    }


def _early_exit_plan(function, context):
    """Where ``function``'s exits start and whether the first is ``if (c) return;``.

    Returns ``(index, simple)`` for the first top-level statement holding a
    return, or ``None``. A ``simple`` exit, ``if (c) return;`` whose ``c``
    reads no memory, becomes ``bool active = !(c);``. Otherwise, as for MLX's
    masked GEMV-T whose only exit sits in ``if (has_output_mask) { ... }``,
    ``bool active = true;`` precedes that statement and
    ``_early_exit_statements`` lowers it like any later exit.
    """
    statements = _statements(function.body)
    for index, statement in enumerate(statements):
        if any(isinstance(node, ReturnNode) for node in context["walk_ast"](statement)):
            break
    else:
        return None
    simple = (
        isinstance(statement, IfNode)
        and statement.else_branch is None
        and not getattr(statement, "else_if_conditions", None)
        and len(_statements(statement.then_branch)) == 1
        and isinstance(_statements(statement.then_branch)[0], ReturnNode)
        and _statements(statement.then_branch)[0].value is None
        and _inactive_safe(
            statement.condition, context["constants"], context["map_operator"]
        )
    )
    rest = statements[index + 1 if simple else index :]
    if not context["effects"](BlockNode(rest)) & {"collective", "barrier"}:
        return None
    return index, simple


def converge_subgroup_early_exits(ast, entries, walk_ast, map_operator):
    """Predicate early returns that skip later collectives and barriers.

    MLX's GEMV leaves rows past the output before its SIMD-group reduction::

        if (out_row >= out_vec_size) return;
        ... loads ...; result += simd_shuffle_down(result, n); ... stores ...

    Apple GPUs run the reduction in the remaining lanes. Software subgroups
    need every invocation to reach their barriers, so a void function's first
    such return becomes ``bool active = !(condition);`` and every later
    statement without a collective or barrier runs under ``if (active)``.
    Invocations that would have left still run the collectives, barriers and
    the counted loops around them, whose headers, like the condition, may not
    read memory, call or trap; they write only private values that never
    leave the function. Later returns, breaks or continues around collectives
    and any other shape keep the source unchanged for the code generators to
    validate.
    """
    nodes = list(walk_ast(ast))
    functions = {}
    for node in nodes:
        if isinstance(node, FunctionNode):
            functions.setdefault(node.name, []).append(node)
    effects = _effect_checker(nodes, walk_ast)
    kinds = _value_function_kinds(functions, walk_ast, map_operator)
    uniformities = subgroup_uniformity(
        ast,
        entries,
        walk_ast,
        map_operator,
        frozenset(name for name, kind in kinds.items() if kind == "pure"),
    )
    reachable = set()
    pending = [entry for stage, entry, _config in entries if stage == "compute"]
    while pending:
        function = pending.pop()
        if id(function) in reachable:
            continue
        reachable.add(id(function))
        for node in walk_ast(function.body):
            if isinstance(node, FunctionCallNode) and _call_name(node) in functions:
                pending.extend(functions[_call_name(node)])
    plans = []
    for candidates in functions.values():
        for function in candidates:
            if (
                id(function) not in reachable
                or _type_name(function.return_type) != "void"
                or getattr(function, "generic_params", None)
            ):
                continue
            context = _early_exit_context(
                function,
                effects,
                walk_ast,
                map_operator,
                functions,
                kinds,
                uniformity=uniformities.get(id(function)),
            )
            plan = _early_exit_plan(function, context)
            if plan is None:
                continue
            index, simple = plan
            statements = _statements(function.body)
            context["uniform_state"]["active"] = not simple or _subgroup_uniform(
                statements[index].condition, context
            )
            context["exit_active"] = "__crossgl_subgroup_active"
            context["constants"].update(_const_literals(statements[:index]))
            rest = _early_exit_statements(
                statements[index + int(simple) :], "__crossgl_subgroup_active", context
            )
            if rest is not None:
                plans.append((function, index, simple))
    if not plans:
        return ast
    reserved = {
        node.name for node in nodes if isinstance(getattr(node, "name", None), str)
    }

    def fresh(base):
        name = base
        while name in reserved:
            name += "_"
        reserved.add(name)
        return name

    memo = {}
    result = deepcopy(ast, memo)
    copied_effects = _effect_checker(list(walk_ast(result)), walk_ast)
    for original, index, simple in plans:
        function = memo[id(original)]
        statements = _statements(function.body)
        exit_branch = statements[index]
        active = fresh("__crossgl_subgroup_active")
        context = _early_exit_context(
            function,
            copied_effects,
            walk_ast,
            map_operator,
            functions,
            kinds,
            fresh,
            uniformity=uniformities.get(id(original)),
        )
        context["uniform_state"]["active"] = not simple or _subgroup_uniform(
            exit_branch.condition, context
        )
        context["exit_active"] = active
        context["constants"].update(_const_literals(statements[:index]))
        rest = _early_exit_statements(
            statements[index + int(simple) :], active, context
        )
        if rest is None:  # pragma: no cover - the copy has the planned shape
            continue
        location = getattr(exit_branch, "source_location", None)
        body = [
            *statements[:index],
            VariableNode(
                active,
                PrimitiveType("bool"),
                (
                    UnaryOpNode("!", exit_branch.condition)
                    if simple
                    else LiteralNode(True, PrimitiveType("bool"))
                ),
                source_location=location,
            ),
            *rest,
        ]
        if isinstance(function.body, BlockNode):
            function.body.statements = body
        else:
            function.body = body
    return result


def has_subgroup_early_exits(ast, walk_ast):
    """Whether a void function returns from inside a top-level statement.

    MLX's GEMV has a top-level ``if (...) return;``; its masked variants
    also return from ``if (has_output_mask) { ... }``. Code generators keep a
    copy of such programs to retry them with ``converge_subgroup_early_exits``
    when the source's exits are unproven.
    """
    for node in walk_ast(ast):
        if not isinstance(node, FunctionNode) or _type_name(node.return_type) != "void":
            continue
        for statement in _statements(node.body):
            if isinstance(statement, IfNode) and any(
                isinstance(child, ReturnNode) for child in walk_ast(statement)
            ):
                return True
    return False


# System values every invocation of a workgroup shares; other entry
# parameters with a system semantic may differ between invocations.
_UNIFORM_SEMANTICS = frozenset(
    {
        "gl_WorkGroupID",
        "gl_NumWorkGroups",
        "gl_WorkGroupSize",
        "gl_NumSubgroups",
        "gl_SubgroupSize",
        "SV_GroupID",
    }
)


def _parameter_semantic(parameter):
    semantic = getattr(parameter, "semantic", None)
    if semantic:
        return str(semantic)
    for attribute in getattr(parameter, "attributes", None) or []:
        name = getattr(attribute, "name", None)
        if isinstance(name, str) and (
            name.startswith("gl_") or name.upper().startswith("SV_")
        ):
            return name
    return None


def _read_names(root, walk_ast):
    return {node.name for node in walk_ast(root) if isinstance(node, IdentifierNode)}


def _root_name(node):
    while isinstance(node, (ArrayAccessNode, MemberAccessNode)):
        node = (
            node.array_expr if isinstance(node, ArrayAccessNode) else node.object_expr
        )
    return node.name if isinstance(node, IdentifierNode) else None


def _varying_names(function, varying_parameters, walk_ast):
    """Names of ``function`` that may differ between a workgroup's invocations.

    The analysis over-approximates: a name is varying when a varying value
    reaches it through an initializer, assignment or call argument, or when a
    branch or loop whose controls read a varying name writes it.
    """
    names = {
        parameter.name
        for index, parameter in enumerate(function.parameters or [])
        if index in varying_parameters
    }
    loops = (ForNode, WhileNode, DoWhileNode)
    while True:
        count = len(names)
        for node in walk_ast(function.body):
            if isinstance(node, VariableNode):
                if node.initial_value is not None and (
                    _read_names(node.initial_value, walk_ast) & names
                ):
                    names.add(node.name)
            elif isinstance(node, AssignmentNode):
                root = _root_name(node.target)
                if (
                    root is not None
                    and (
                        _read_names(node.value, walk_ast)
                        | _read_names(node.target, walk_ast)
                    )
                    & names
                ):
                    names.add(root)
            elif isinstance(node, FunctionCallNode):
                if any(
                    _read_names(argument, walk_ast) & names
                    for argument in node.arguments
                ):
                    # Reference arguments may receive varying values.
                    names.update(
                        root
                        for root in map(_root_name, node.arguments)
                        if root is not None
                    )
            elif isinstance(node, IfNode) or isinstance(node, loops):
                controls = [
                    getattr(node, part, None)
                    for part in ("condition", "init", "update")
                ] + list(getattr(node, "else_if_conditions", None) or [])
                if any(
                    _read_names(control, walk_ast) & names
                    for control in controls
                    if control is not None
                ):
                    names |= _written_roots(node, walk_ast)
        if len(names) == count:
            return names


def _divergent_barrier_plan(function, varying, effects, walk_ast):
    """Top-level ``if (c) { ... barrier ... }`` statements whose ``c`` may vary."""
    if any(isinstance(node, ReturnNode) for node in walk_ast(function.body)):
        return None  # converge_subgroup_early_exits owns functions that exit
    indices = tuple(
        index
        for index, statement in enumerate(_statements(function.body))
        if isinstance(statement, IfNode)
        and statement.else_branch is None
        and not getattr(statement, "else_if_conditions", None)
        and "barrier" in effects(statement.then_branch)
        and _read_names(statement.condition, walk_ast) & varying
    )
    return indices or None


def converge_subgroup_divergent_barriers(ast, entries, walk_ast, map_operator):
    """Run the workgroup barriers of possibly divergent branches everywhere.

    MLX's GEMV-T accumulates only columns inside the output, with a barrier
    in its row loop::

        if (out_col < out_vec_size) {
            for (int i = 0; i < n_iter; ++i) {
                threadgroup_barrier(mem_flags::mem_none);
                ... loads ...
            }
        }

    Apple GPUs let the threads past the output skip that barrier. OpenGL and
    Direct3D need every invocation of the workgroup at each barrier, so a
    void function's top-level branch whose condition may differ between
    invocations becomes ``bool taken = condition;`` and only the branch's
    other statements run under ``if (taken)``. Its barriers, and the counted
    loops around them, run in every invocation; loop headers and other
    statements those invocations now evaluate must be safe as for
    ``converge_subgroup_early_exits``. Branches that may not converge keep
    the source unchanged.
    """
    nodes = list(walk_ast(ast))
    functions = {}
    for node in nodes:
        if isinstance(node, FunctionNode):
            functions.setdefault(node.name, []).append(node)
    compute_entries = [entry for stage, entry, _config in entries if stage == "compute"]
    if not compute_entries:
        return ast
    varying_parameters = {
        id(entry): {
            index
            for index, parameter in enumerate(entry.parameters or [])
            if _parameter_semantic(parameter) is not None
            and _parameter_semantic(parameter) not in _UNIFORM_SEMANTICS
        }
        for entry in compute_entries
    }
    reachable, pending = {}, list(compute_entries)
    queued = {id(entry) for entry in compute_entries}
    varying = {}
    while pending:
        function = pending.pop()
        queued.discard(id(function))
        reachable[id(function)] = function
        names = _varying_names(
            function, varying_parameters.get(id(function), ()), walk_ast
        )
        varying[id(function)] = names
        for call in walk_ast(function.body):
            if (
                not isinstance(call, FunctionCallNode)
                or _call_name(call) not in functions
            ):
                continue
            arguments = list(call.arguments)
            for callee in functions[_call_name(call)]:
                if len(callee.parameters or []) != len(arguments):
                    continue
                marked = varying_parameters.setdefault(id(callee), set())
                added = {
                    index
                    for index, argument in enumerate(arguments)
                    if _read_names(argument, walk_ast) & names
                } - marked
                if added or id(callee) not in reachable:
                    marked |= added
                    if id(callee) not in queued:
                        queued.add(id(callee))
                        pending.append(callee)
    effects = _effect_checker(nodes, walk_ast)
    kinds = _value_function_kinds(functions, walk_ast, map_operator)
    plans = []
    for key, function in reachable.items():
        if _type_name(function.return_type) != "void" or getattr(
            function, "generic_params", None
        ):
            continue
        indices = _divergent_barrier_plan(function, varying[key], effects, walk_ast)
        if indices is not None:
            plans.append((function, indices))
    if not plans:
        return ast
    reserved = {
        node.name for node in nodes if isinstance(getattr(node, "name", None), str)
    }

    def fresh(base):
        name = base
        while name in reserved:
            name += "_"
        reserved.add(name)
        return name

    memo = {}
    result = deepcopy(ast, memo)
    copied_effects = _effect_checker(list(walk_ast(result)), walk_ast)
    for original, indices in plans:
        function = memo[id(original)]
        statements = list(_statements(function.body))
        context = _early_exit_context(
            function, copied_effects, walk_ast, map_operator, functions, kinds, fresh
        )
        converged = list(statements)
        for index in indices:
            context["constants"].update(_const_literals(statements[:index]))
            block = _predicated_if(statements[index], None, context)
            if block is None:
                break
            converged[index] = block
        else:
            if isinstance(function.body, BlockNode):
                function.body.statements = converged
            else:
                function.body = converged
    return result
