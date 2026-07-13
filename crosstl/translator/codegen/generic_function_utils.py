"""Helpers for monomorphizing generic functions in C-style shader outputs."""

from copy import deepcopy

from ..ast import (
    AssignmentNode,
    BinaryOpNode,
    BlockNode,
    ConstructorNode,
    ExpressionStatementNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IdentifierNode,
    IfNode,
    LiteralNode,
    MatchNode,
    MemberAccessNode,
    PointerAccessNode,
    ReturnNode,
    VariableNode,
)
from .enum_utils import (
    generic_type_parts,
    sanitize_type_name,
    substitute_generic_type_name,
)
from .image_access_contracts import (
    IMAGE_ATOMIC_INTRINSIC_NAMES,
    RESOURCE_QUERY_SAMPLES_INTRINSIC_NAMES,
    RESOURCE_QUERY_SIZE_INTRINSIC_NAMES,
    TEXTURE_RESOURCE_INTRINSIC_NAMES,
)

BUFFER_RESOURCE_INTRINSIC_NAMES = frozenset(
    {
        "buffer_load",
        "buffer_load2",
        "buffer_load3",
        "buffer_load4",
        "buffer_store",
        "buffer_store2",
        "buffer_store3",
        "buffer_store4",
        "buffer_append",
        "buffer_consume",
        "buffer_dimensions",
        "buffer_increment_counter",
        "buffer_decrement_counter",
    }
)

IMAGE_RESOURCE_INTRINSIC_NAMES = frozenset(
    {
        "imageLoad",
        "imageStore",
        "imageSize",
        "imageSamples",
    }
) | frozenset(IMAGE_ATOMIC_INTRINSIC_NAMES)

RESERVED_GENERIC_FUNCTION_BUILTIN_NAMES = (
    BUFFER_RESOURCE_INTRINSIC_NAMES
    | IMAGE_RESOURCE_INTRINSIC_NAMES
    | frozenset(RESOURCE_QUERY_SAMPLES_INTRINSIC_NAMES)
    | frozenset(RESOURCE_QUERY_SIZE_INTRINSIC_NAMES)
    | frozenset(TEXTURE_RESOURCE_INTRINSIC_NAMES)
)


class GenericMemberCallSpecializationError(ValueError):
    """A target call reached code generation before member specialization."""

    project_diagnostic_code = "project.translate.generic-member-call-unresolved"
    missing_capabilities = ("generic.member-call-specialization",)

    def __init__(
        self,
        message,
        *,
        target_name,
        receiver,
        method_name,
        generic_arguments,
        source_location=None,
    ):
        super().__init__(message)
        self.target_name = target_name
        self.receiver = receiver
        self.method_name = method_name
        self.generic_arguments = tuple(generic_arguments)
        self.suggested_action = (
            "materialize the concrete member-template specialization before "
            f"{target_name} generation"
        )
        self.source_location = source_location


def reject_unresolved_generic_member_call(expr, target_name):
    """Reject a structured generic member call that was not materialized."""
    generic_args = list(getattr(expr, "generic_args", []) or [])
    if not generic_args:
        return

    function = getattr(expr, "function", getattr(expr, "name", None))
    if isinstance(function, MemberAccessNode):
        receiver_expr = getattr(
            function, "object_expr", getattr(function, "object", None)
        )
    elif isinstance(function, PointerAccessNode):
        receiver_expr = getattr(function, "pointer_expr", None)
    else:
        return

    receiver = _generic_member_expression_text(receiver_expr)
    method_name = str(getattr(function, "member", "") or "<unknown>")
    generic_arguments = [
        _generic_member_expression_text(argument) for argument in generic_args
    ]
    signature = f"{receiver}.{method_name}<{', '.join(generic_arguments)}>"
    raise GenericMemberCallSpecializationError(
        f"{target_name} generation cannot lower generic member call "
        f"'{signature}' because no concrete member specialization was "
        "materialized",
        target_name=target_name,
        receiver=receiver,
        method_name=method_name,
        generic_arguments=generic_arguments,
        source_location=getattr(expr, "source_location", None)
        or getattr(function, "source_location", None),
    )


def _generic_member_expression_text(value):
    if value is None:
        return "<unknown>"
    if isinstance(value, str):
        return value
    if isinstance(value, MemberAccessNode):
        owner = getattr(value, "object_expr", getattr(value, "object", None))
        return f"{_generic_member_expression_text(owner)}.{value.member}"
    if isinstance(value, PointerAccessNode):
        return f"{_generic_member_expression_text(value.pointer_expr)}->{value.member}"
    if isinstance(value, BinaryOpNode):
        return (
            f"{_generic_member_expression_text(value.left)} "
            f"{value.operator} {_generic_member_expression_text(value.right)}"
        )
    name = getattr(value, "name", None)
    if name is not None:
        nested_args = list(getattr(value, "generic_args", []) or [])
        if nested_args:
            rendered = ", ".join(
                _generic_member_expression_text(argument) for argument in nested_args
            )
            return f"{name}<{rendered}>"
        return str(name)
    literal = getattr(value, "value", None)
    if literal is not None:
        return str(literal)
    return str(value)


def prepare_generic_function_specializations(generator, functions):
    """Collect concrete generic function instantiations used by ``functions``."""
    definitions = {}
    definition_ordinals = {}
    for func in functions or []:
        func_name = getattr(func, "name", None)
        if (
            not func_name
            or not generic_function_parameters(func)
            or is_reserved_generic_function_builtin_name(func_name)
        ):
            continue
        definition_ordinals[id(func)] = len(definitions.setdefault(func_name, []))
        definitions[func_name].append(func)
    generator.generic_function_definitions = definitions
    generator.generic_function_definition_ordinals = definition_ordinals
    generator.generic_function_specializations = {}
    generator.generic_function_specialized_names = {}

    if not definitions:
        return {}

    pending = []
    for func in functions or []:
        if getattr(func, "name", None) in definitions:
            continue
        pending.extend(_collect_from_function(generator, func, {}))

    seen_analysis = set()
    while pending:
        key = pending.pop(0)
        clone = generator.generic_function_specializations.get(key)
        if clone is None or key in seen_analysis:
            continue
        seen_analysis.add(key)
        substitutions = dict(getattr(clone, "_generic_substitutions", {}) or {})
        pending.extend(_collect_from_function(generator, clone, substitutions))

    return generator.generic_function_specializations


def generic_function_parameters(func):
    return [
        getattr(param, "name", None)
        for param in getattr(func, "generic_params", []) or []
        if getattr(param, "name", None)
    ]


def generic_function_parameter_defaults(generator, func):
    defaults = {}
    for param in getattr(func, "generic_params", []) or []:
        name = getattr(param, "name", None)
        default_type = getattr(param, "default_type", None)
        if not name or default_type is None:
            continue
        rendered = generator.type_name_string(default_type)
        if rendered:
            defaults[name] = rendered
    return defaults


def iter_function_nodes(ast_node):
    """Yield every function node reachable from an AST object."""
    visited = set()

    def visit(value):
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return

        value_id = id(value)
        if value_id in visited:
            return
        visited.add(value_id)

        if isinstance(value, FunctionNode):
            yield value

        if isinstance(value, dict):
            children = value.values()
        elif isinstance(value, (list, tuple, set, frozenset)):
            children = value
        elif hasattr(value, "__dict__"):
            children = vars(value).values()
        else:
            return

        for child in children:
            yield from visit(child)

    yield from visit(ast_node)


def reject_unsupported_generic_functions(
    ast_node,
    target_name,
    specializations=None,
    referenced_generic_names=None,
):
    """Reject generic functions before emitting non-specialized target code."""
    specialized_source_names = {
        key[0] for key in specializations or {} if isinstance(key, tuple) and key
    }
    for func in iter_function_nodes(ast_node):
        generic_params = generic_function_parameters(func)
        if not generic_params:
            continue
        if getattr(func, "name", None) in specialized_source_names:
            continue
        if (
            referenced_generic_names is not None
            and getattr(func, "name", None) not in referenced_generic_names
        ):
            continue
        suffix = f" ({', '.join(generic_params)})" if generic_params else ""
        raise ValueError(
            f"{target_name} codegen does not support generic functions{suffix}; "
            f"specialize the function before {target_name} generation"
        )


def collect_unresolved_generic_function_call_names(generator, functions):
    """Return generic helpers called from emitted functions without a specialization."""
    definitions = getattr(generator, "generic_function_definitions", {}) or {}
    if not definitions:
        return []

    unresolved = []
    seen_names = set()
    visited = set()

    def visit(value):
        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return

        value_id = id(value)
        if value_id in visited:
            return
        visited.add(value_id)

        if isinstance(value, FunctionNode) and generic_function_parameters(value):
            return

        if isinstance(value, FunctionCallNode):
            func_name = _function_call_name(value)
            args = list(getattr(value, "arguments", getattr(value, "args", [])) or [])
            for arg in args:
                visit(arg)
            if (
                func_name in definitions
                and generic_function_call_name(generator, func_name, args) is None
                and func_name not in seen_names
            ):
                seen_names.add(func_name)
                unresolved.append(func_name)
            return

        if isinstance(value, dict):
            children = value.values()
        elif isinstance(value, (list, tuple, set, frozenset)):
            children = value
        elif hasattr(value, "__dict__"):
            children = vars(value).values()
        else:
            return

        for child in children:
            visit(child)

    for func in functions or []:
        if generic_function_parameters(func):
            continue
        visit(func)

    return unresolved


def generic_function_emission_list(generator, func):
    """Return concrete specializations to emit for ``func`` instead of the generic."""
    if not generic_function_parameters(func):
        return [func]

    specializations = getattr(generator, "generic_function_specializations", {}) or {}
    return [
        clone
        for clone in specializations.values()
        if getattr(clone, "_generic_source_function_id", None) == id(func)
    ]


def generic_function_call_name(generator, func_name, args):
    """Return a specialized callee name for a generic call, when inferable."""
    key = generic_function_call_key(generator, func_name, args)
    specialized_names = (
        getattr(generator, "generic_function_specialized_names", {}) or {}
    )
    if key is not None:
        return specialized_names.get(key)
    return _compatible_existing_specialized_call_name(generator, func_name, args)


def raise_unresolved_generic_function_call(generator, func_name, target_name):
    """Raise when a target cannot infer a concrete generic helper call."""
    candidates = (getattr(generator, "generic_function_definitions", {}) or {}).get(
        func_name,
        [],
    )
    if not candidates:
        return

    func = candidates[0]
    generic_params = generic_function_parameters(func)
    suffix = f" ({', '.join(generic_params)})" if generic_params else ""
    raise ValueError(
        f"{target_name} codegen cannot infer concrete template arguments for "
        f"generic function '{func_name}'{suffix}; specialize the function before "
        f"{target_name} generation"
    )


def generic_function_value_arguments(generator, func_name, args):
    """Return value arguments after any leading explicit generic type arguments."""
    definitions = getattr(generator, "generic_function_definitions", {}) or {}
    func = definitions.get(func_name)
    if func is None:
        return list(args or [])

    generic_params = generic_function_parameters(func)
    param_list = list(getattr(func, "parameters", getattr(func, "params", [])) or [])
    explicit_substitutions, value_args = _explicit_generic_call_parts(
        generator,
        generic_params,
        param_list,
        list(args or []),
    )
    if explicit_substitutions is None:
        return list(args or [])
    return value_args


def generic_function_call_key(generator, func_name, args):
    if is_reserved_generic_function_builtin_name(func_name):
        return None

    definitions = getattr(generator, "generic_function_definitions", {}) or {}
    candidates = definitions.get(func_name) or []
    if not candidates:
        return None

    matches = []
    for func in candidates:
        match = _generic_function_candidate_call_match(
            generator,
            func,
            func_name,
            args,
        )
        if match is None:
            continue
        matches.append(match)

    if not matches:
        return None

    matches.sort(key=lambda match: match["rank"], reverse=True)
    for match in matches:
        _ensure_generic_function_specialization(
            generator,
            match["function"],
            match["key"],
            match["substitutions"],
        )
    return matches[0]["key"]


def _compatible_existing_specialized_call_name(generator, func_name, args):
    specializations = getattr(generator, "generic_function_specializations", {}) or {}
    specialized_names = (
        getattr(generator, "generic_function_specialized_names", {}) or {}
    )
    matches = []
    for key, specialized_func in specializations.items():
        if not isinstance(key, tuple) or not key or key[0] != func_name:
            continue
        if _specialized_function_accepts_call(generator, specialized_func, args):
            specialized_name = specialized_names.get(key)
            if specialized_name:
                matches.append(specialized_name)
    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]
    return None


def _specialized_function_accepts_call(generator, func, args):
    param_list = list(getattr(func, "parameters", getattr(func, "params", [])) or [])
    if len(args or []) < len(param_list):
        return False

    saw_informative_argument = False
    saw_unknown_argument = False
    for param, arg in zip(param_list, args or []):
        expected_type = generator.type_name_string(
            getattr(param, "param_type", getattr(param, "vtype", None))
        )
        actual_type = _safe_call_argument_type_name(generator, arg)
        if not expected_type or not actual_type:
            saw_unknown_argument = True
            continue
        if actual_type == "auto":
            saw_unknown_argument = True
            continue
        saw_informative_argument = True
        if _call_argument_type_accepts_expected(generator, expected_type, actual_type):
            continue
        return False
    return saw_informative_argument or saw_unknown_argument


def _raw_generic_signature_type_name(generator, type_value):
    saved_substitutions = getattr(
        generator,
        "current_generic_function_substitutions",
        {},
    )
    generator.current_generic_function_substitutions = {}
    try:
        return generator.type_name_string(type_value)
    finally:
        generator.current_generic_function_substitutions = saved_substitutions


def _call_argument_type_accepts_expected(generator, expected_type, actual_type):
    expected_normalized = _normalize_signature_type_name(generator, expected_type)
    actual_normalized = _normalize_signature_type_name(generator, actual_type)
    if expected_normalized == actual_normalized:
        return True
    if _types_compatible(generator, expected_type, actual_type):
        return True

    expected_base, _expected_args = generic_type_parts(expected_type)
    actual_base, actual_args = generic_type_parts(actual_type)
    return expected_base == actual_base and not actual_args


def _generic_function_candidate_call_match(generator, func, func_name, args):
    generic_params = generic_function_parameters(func)
    if not generic_params:
        return None

    call_args = list(args or [])
    substitutions = {}
    generic_param_set = set(generic_params)
    param_list = list(getattr(func, "parameters", getattr(func, "params", [])) or [])
    explicit_substitutions, value_args = _explicit_generic_call_parts(
        generator,
        generic_params,
        param_list,
        call_args,
    )
    if explicit_substitutions is not None:
        substitutions.update(explicit_substitutions)
    else:
        value_args = call_args

    matched_params = len(substitutions)
    exact_matches = 0
    compatible_params = 0

    for param, arg in zip(param_list, value_args):
        expected_type = _raw_generic_signature_type_name(
            generator,
            getattr(param, "param_type", getattr(param, "vtype", None)),
        )
        actual_type = _safe_call_argument_type_name(generator, arg)
        if not actual_type:
            actual_type = getattr(
                generator,
                "current_generic_function_substitutions",
                {},
            ).get(expected_type)
        if not actual_type:
            continue

        before = dict(substitutions)
        _collect_type_parameter_bindings(
            expected_type,
            actual_type,
            substitutions,
            generic_param_set,
        )
        if substitutions != before:
            matched_params += 1

        concrete_expected = substitute_generic_type_name(
            expected_type,
            substitutions,
        )
        expected_normalized = _normalize_signature_type_name(
            generator,
            concrete_expected,
        )
        actual_normalized = _normalize_signature_type_name(generator, actual_type)
        if expected_normalized == actual_normalized:
            exact_matches += 1
        elif _types_compatible(generator, concrete_expected, actual_type):
            compatible_params += 1

    if any(param not in substitutions for param in generic_params):
        defaults = generic_function_parameter_defaults(generator, func)
        if defaults:
            for param in generic_params:
                if param in substitutions or param not in defaults:
                    continue
                substitutions[param] = substitute_generic_type_name(
                    defaults[param],
                    substitutions,
                )

    if any(param not in substitutions for param in generic_params):
        return None

    concrete_args = tuple(substitutions[param] for param in generic_params)
    ordinal = (
        getattr(generator, "generic_function_definition_ordinals", {}) or {}
    ).get(
        id(func),
        0,
    )
    overload_count = len(
        (getattr(generator, "generic_function_definitions", {}) or {}).get(
            func_name,
            [],
        )
    )
    key = (
        (func_name, concrete_args, ordinal)
        if overload_count > 1
        else (func_name, concrete_args)
    )
    args_count = len(value_args)
    arity_distance = abs(args_count - len(param_list))
    return {
        "function": func,
        "key": key,
        "substitutions": substitutions,
        "rank": (
            matched_params,
            exact_matches,
            compatible_params,
            -arity_distance,
            len(param_list),
            -ordinal,
        ),
    }


def _explicit_generic_call_parts(generator, generic_params, param_list, args):
    generic_count = len(generic_params)
    if generic_count == 0 or len(args) != len(param_list) + generic_count:
        return None, args

    substitutions = {}
    current_substitutions = getattr(
        generator,
        "current_generic_function_substitutions",
        {},
    )
    for param_name, arg in zip(generic_params, args[:generic_count]):
        type_name = _type_argument_name(generator, arg)
        if not type_name:
            return None, args
        substitutions[param_name] = substitute_generic_type_name(
            type_name,
            current_substitutions,
        )

    return substitutions, args[generic_count:]


def generate_numeric_trait_method_call(generator, func_expr, args):
    """Lower simple numeric trait-style methods to shader operators."""
    member = getattr(func_expr, "member", None)
    operator = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
    }.get(member)
    if operator is None or len(args or []) != 1:
        return None

    object_expr = getattr(func_expr, "object", getattr(func_expr, "object_expr", None))
    if object_expr is None:
        return None

    left = generator.generate_expression_with_expected(object_expr, None)
    right = generator.generate_expression_with_expected(args[0], None)
    return f"({left} {operator} {right})"


def numeric_trait_method_result_type(generator, call):
    """Infer the result type of simple numeric trait-style methods."""
    func_expr = getattr(call, "function", getattr(call, "name", None))
    member = getattr(func_expr, "member", None)
    if member not in {"add", "sub", "mul", "div"}:
        return None

    object_expr = getattr(func_expr, "object", getattr(func_expr, "object_expr", None))
    if object_expr is None:
        return None
    return _safe_expression_result_type(generator, object_expr)


def generate_static_generic_numeric_call(generator, func_name):
    """Lower ``T::zero()`` and ``T::one()`` inside concrete specializations."""
    if "::" not in str(func_name):
        return None

    type_param, method = str(func_name).split("::", 1)
    if method not in {"zero", "one"}:
        return None

    substitutions = getattr(generator, "current_generic_function_substitutions", {})
    concrete_type = substitutions.get(type_param)
    if not concrete_type:
        return None

    mapped_type = generator.map_type(concrete_type)
    mapped_type_key = str(mapped_type).lower()
    value = 1 if method == "one" else 0
    if (
        "float" in mapped_type_key
        or "half" in mapped_type_key
        or mapped_type_key == "double"
    ):
        return f"{value}.0"
    return str(value)


def _collect_from_function(generator, func, substitutions):
    previous_local_variable_types = getattr(generator, "local_variable_types", {})
    previous_expected_type = getattr(
        generator, "current_expression_expected_type", None
    )
    previous_generic_function_substitutions = getattr(
        generator,
        "current_generic_function_substitutions",
        {},
    )
    generator.local_variable_types = {}
    generator.current_expression_expected_type = None
    generator.current_generic_function_substitutions = substitutions
    try:
        for param in getattr(func, "parameters", getattr(func, "params", [])) or []:
            param_type = generator.type_name_string(
                getattr(param, "param_type", getattr(param, "vtype", None))
            )
            generator.local_variable_types[param.name] = substitute_generic_type_name(
                param_type,
                substitutions,
            )
        before = set(getattr(generator, "generic_function_specializations", {}) or {})
        _analyze_statement(generator, getattr(func, "body", None), substitutions)
        after = set(getattr(generator, "generic_function_specializations", {}) or {})
        return [key for key in after - before]
    finally:
        generator.local_variable_types = previous_local_variable_types
        generator.current_expression_expected_type = previous_expected_type
        generator.current_generic_function_substitutions = (
            previous_generic_function_substitutions
        )


def _analyze_statement(generator, stmt, substitutions):
    if stmt is None:
        return
    if isinstance(stmt, list):
        for child in stmt:
            _analyze_statement(generator, child, substitutions)
        return
    if isinstance(stmt, BlockNode):
        for child in getattr(stmt, "statements", []) or []:
            _analyze_statement(generator, child, substitutions)
        return
    if hasattr(stmt, "statements") and not isinstance(stmt, FunctionCallNode):
        for child in getattr(stmt, "statements", []) or []:
            _analyze_statement(generator, child, substitutions)
        return
    if isinstance(stmt, VariableNode):
        initial_value = getattr(stmt, "initial_value", None)
        _analyze_expression(generator, initial_value, substitutions)
        declared_type = generator.type_name_string(getattr(stmt, "var_type", None))
        if declared_type is None:
            declared_type = generator.type_name_string(
                _safe_expression_result_type(generator, initial_value)
            )
        if declared_type:
            generator.local_variable_types[stmt.name] = substitute_generic_type_name(
                declared_type,
                substitutions,
            )
        return
    if isinstance(stmt, AssignmentNode):
        _analyze_expression(generator, getattr(stmt, "value", None), substitutions)
        _analyze_expression(generator, getattr(stmt, "target", None), substitutions)
        return
    if isinstance(stmt, ReturnNode):
        _analyze_expression(generator, getattr(stmt, "value", None), substitutions)
        return
    if isinstance(stmt, ExpressionStatementNode):
        _analyze_expression(generator, getattr(stmt, "expression", None), substitutions)
        return
    if isinstance(stmt, IfNode):
        _analyze_expression(generator, getattr(stmt, "condition", None), substitutions)
        _analyze_statement(generator, getattr(stmt, "then_branch", None), substitutions)
        _analyze_statement(generator, getattr(stmt, "else_branch", None), substitutions)
        return
    if isinstance(stmt, ForNode):
        _analyze_statement(generator, getattr(stmt, "init", None), substitutions)
        _analyze_expression(generator, getattr(stmt, "condition", None), substitutions)
        _analyze_expression(generator, getattr(stmt, "update", None), substitutions)
        _analyze_statement(generator, getattr(stmt, "body", None), substitutions)
        return
    if isinstance(stmt, MatchNode):
        _analyze_expression(generator, getattr(stmt, "expression", None), substitutions)
        for arm in getattr(stmt, "arms", []) or []:
            _analyze_expression(generator, getattr(arm, "guard", None), substitutions)
            _analyze_statement(generator, getattr(arm, "body", None), substitutions)
        return

    _analyze_expression(generator, stmt, substitutions)


def _analyze_expression(generator, expr, substitutions):
    if expr is None or isinstance(expr, (str, int, float, bool)):
        return
    if isinstance(expr, FunctionCallNode):
        func_name = _function_call_name(expr)
        args = list(getattr(expr, "arguments", getattr(expr, "args", [])) or [])
        for arg in args:
            _analyze_expression(generator, arg, substitutions)
        if func_name is not None:
            generic_function_call_key(generator, func_name, args)
        return
    if isinstance(expr, ConstructorNode):
        for arg in getattr(expr, "arguments", []) or []:
            _analyze_expression(generator, arg, substitutions)
        for arg in (getattr(expr, "named_arguments", {}) or {}).values():
            _analyze_expression(generator, arg, substitutions)
        return
    if isinstance(expr, MatchNode):
        _analyze_expression(generator, getattr(expr, "expression", None), substitutions)
        for arm in getattr(expr, "arms", []) or []:
            _analyze_expression(generator, getattr(arm, "guard", None), substitutions)
            _analyze_statement(generator, getattr(arm, "body", None), substitutions)
        return
    if hasattr(expr, "__dict__"):
        for child in vars(expr).values():
            if isinstance(child, (str, int, float, bool)):
                continue
            _analyze_statement(generator, child, substitutions)


def _ensure_generic_function_specialization(generator, func, key, substitutions):
    specializations = generator.generic_function_specializations
    if key in specializations:
        return

    clone = deepcopy(func)
    overload_ordinal = key[2] if len(key) > 2 else None
    clone.name = generic_function_specialization_name(
        key[0],
        key[1],
        overload_ordinal,
    )
    clone.generic_params = []
    clone.return_type = substitute_generic_type_name(
        generator.type_name_string(getattr(func, "return_type", "void")),
        substitutions,
    )
    for param in getattr(clone, "parameters", getattr(clone, "params", [])) or []:
        param_type = generator.type_name_string(
            getattr(param, "param_type", getattr(param, "vtype", None))
        )
        concrete_type = substitute_generic_type_name(param_type, substitutions)
        if hasattr(param, "param_type"):
            param.param_type = concrete_type
        if hasattr(param, "vtype"):
            param.vtype = concrete_type
    clone._generic_source_name = key[0]
    clone._generic_source_function_id = id(func)
    clone._generic_substitutions = dict(substitutions)
    specializations[key] = clone
    generator.generic_function_specialized_names[key] = clone.name


def generic_function_specialization_name(
    func_name,
    concrete_args,
    overload_ordinal=None,
):
    suffix = "_".join(sanitize_type_name(arg) for arg in concrete_args)
    parts = [sanitize_type_name(func_name)]
    if suffix:
        parts.append(suffix)
    if overload_ordinal is not None:
        parts.append(f"overload{overload_ordinal}")
    return "_".join(parts)


def is_reserved_generic_function_builtin_name(func_name):
    return str(func_name) in RESERVED_GENERIC_FUNCTION_BUILTIN_NAMES


def _collect_type_parameter_bindings(
    expected_type,
    actual_type,
    substitutions,
    generic_params,
):
    if expected_type is None or actual_type is None:
        return

    expected_type = str(expected_type)
    actual_type = str(actual_type)
    expected_inner = _strip_reference_wrappers(expected_type)
    actual_inner = _strip_reference_wrappers(actual_type)
    if expected_inner != expected_type or actual_inner != actual_type:
        _collect_type_parameter_bindings(
            expected_inner,
            actual_inner,
            substitutions,
            generic_params,
        )
        return

    if expected_type in generic_params:
        substitutions.setdefault(expected_type, actual_type)
        return

    expected_pointee = _pointer_pointee_type(expected_type)
    actual_pointee = _pointer_pointee_type(actual_type)
    if expected_pointee is not None and actual_pointee is not None:
        _collect_type_parameter_bindings(
            expected_pointee,
            actual_pointee,
            substitutions,
            generic_params,
        )
        return

    expected_base, expected_args = generic_type_parts(expected_type)
    actual_base, actual_args = generic_type_parts(actual_type)
    if expected_base != actual_base or len(expected_args) != len(actual_args):
        return

    for expected_arg, actual_arg in zip(expected_args, actual_args):
        _collect_type_parameter_bindings(
            expected_arg,
            actual_arg,
            substitutions,
            generic_params,
        )


def _safe_expression_result_type(generator, expr):
    try:
        result_type = generator.expression_result_type(expr)
    except (AttributeError, TypeError, ValueError):
        result_type = None
    fallback_type = _fallback_expression_result_type(generator, expr)
    if fallback_type:
        return fallback_type
    if isinstance(expr, ConstructorNode):
        inferred_type = _infer_generic_constructor_type_name(generator, expr)
        if inferred_type:
            return inferred_type
    if result_type:
        return result_type
    return _infer_generic_constructor_type_name(generator, expr)


def _fallback_expression_result_type(generator, expr):
    if expr is None:
        return None

    if isinstance(expr, LiteralNode):
        return generator.type_name_string(getattr(expr, "literal_type", None))

    if isinstance(expr, IdentifierNode):
        name = getattr(expr, "name", None)
        if not name:
            return None
        return (getattr(generator, "local_variable_types", {}) or {}).get(name) or (
            getattr(generator, "variable_types", {}) or {}
        ).get(name)

    if isinstance(expr, ConstructorNode):
        return _constructor_result_type(generator, expr)

    if isinstance(expr, MemberAccessNode):
        return _member_access_result_type(generator, expr)

    if isinstance(expr, FunctionCallNode):
        return _function_call_result_type(generator, expr)

    if isinstance(expr, BinaryOpNode):
        return _binary_expression_result_type(generator, expr)

    return None


def _constructor_result_type(generator, expr):
    constructor_type = _constructor_type_name(generator, expr)
    if constructor_type in {"vec2", "vec3", "vec4", "mat2", "mat3", "mat4"}:
        return constructor_type
    if constructor_type in {"float2", "float3", "float4"}:
        return constructor_type
    return _infer_generic_constructor_type_name(generator, expr)


def _member_access_result_type(generator, expr):
    member = str(getattr(expr, "member", "") or "")
    if not member:
        return None

    object_expr = getattr(expr, "object", getattr(expr, "object_expr", None))
    object_type = generator.type_name_string(
        _safe_expression_result_type(generator, object_expr)
    )
    if not object_type:
        return None

    member_type = _struct_member_type_name(generator, object_type, member)
    if member_type:
        return member_type

    if set(member) <= set("xyzwrgba") and len(member) <= 4:
        component_type = _scalar_component_type_name(generator, object_type)
        if not component_type:
            return None
        if len(member) == 1:
            return component_type
        return _vector_type_name(component_type, len(member))

    return None


def _struct_member_type_name(generator, object_type, member):
    base_name, generic_args = generic_type_parts(object_type)
    struct_node = _generic_constructor_struct_node(generator, base_name)
    field_types = _generic_constructor_field_types(generator, base_name, struct_node)
    if not field_types:
        return None

    generic_params = _generic_constructor_struct_parameters(
        generator,
        base_name,
        struct_node,
    )
    substitutions = dict(zip(generic_params, generic_args))
    for field_name, field_type in field_types:
        if field_name != member:
            continue
        return substitute_generic_type_name(
            generator.type_name_string(field_type),
            substitutions,
        )
    return None


def _function_call_result_type(generator, expr):
    func_expr = getattr(expr, "function", getattr(expr, "name", None))
    member = getattr(func_expr, "member", None)
    if member in {"add", "sub", "mul", "div"}:
        object_expr = getattr(
            func_expr, "object", getattr(func_expr, "object_expr", None)
        )
        return generator.type_name_string(
            _safe_expression_result_type(generator, object_expr)
        )

    func_name = _function_call_name(expr)
    args = list(getattr(expr, "arguments", getattr(expr, "args", [])) or [])
    if func_name is None:
        return None
    func_name = str(func_name)

    if "::" in func_name:
        type_param, method = func_name.split("::", 1)
        if method in {"zero", "one"}:
            return (
                getattr(generator, "current_generic_function_substitutions", {}) or {}
            ).get(type_param)

    if func_name in {"vec2", "vec3", "vec4", "mat2", "mat3", "mat4"}:
        return func_name
    if func_name in {"float2", "float3", "float4"}:
        return func_name

    if func_name in {"normalize", "reflect", "cross"} and args:
        return generator.type_name_string(
            _safe_expression_result_type(generator, args[0])
        )

    if func_name in {"dot", "length", "distance"} and args:
        arg_type = generator.type_name_string(
            _safe_expression_result_type(generator, args[0])
        )
        return _scalar_component_type_name(generator, arg_type) or "float"

    if func_name in {"sqrt", "pow", "max", "min", "clamp", "abs"} and args:
        return generator.type_name_string(
            _safe_expression_result_type(generator, args[0])
        )

    return None


def _binary_expression_result_type(generator, expr):
    left_type = generator.type_name_string(
        _safe_expression_result_type(generator, getattr(expr, "left", None))
    )
    right_type = generator.type_name_string(
        _safe_expression_result_type(generator, getattr(expr, "right", None))
    )
    if _is_vector_type_name(left_type):
        return left_type
    if _is_vector_type_name(right_type):
        return right_type
    if left_type in {"double", "float"}:
        return left_type
    if right_type in {"double", "float"}:
        return right_type
    return left_type or right_type


def _scalar_component_type_name(generator, type_name):
    if not type_name:
        return None
    try:
        component_type = generator.scalar_component_type(type_name)
    except (AttributeError, TypeError, ValueError):
        component_type = None
    if component_type:
        return generator.type_name_string(component_type)

    type_text = str(type_name)
    if type_text.startswith("vec") and len(type_text) == 4 and type_text[-1].isdigit():
        return "float"
    if type_text.startswith("float") and type_text[-1:] in {"2", "3", "4"}:
        return "float"
    if type_text.startswith("double") and type_text[-1:] in {"2", "3", "4"}:
        return "double"
    if type_text.startswith("int") and type_text[-1:] in {"2", "3", "4"}:
        return "int"
    if type_text.startswith("uint") and type_text[-1:] in {"2", "3", "4"}:
        return "uint"
    return None


def _vector_type_name(component_type, size):
    if component_type == "float":
        return f"vec{size}"
    return f"{component_type}{size}"


def _is_vector_type_name(type_name):
    if not type_name:
        return False
    type_text = str(type_name)
    return (
        type_text in {"vec2", "vec3", "vec4"}
        or type_text.endswith(("2", "3", "4"))
        and type_text.startswith(("float", "double", "int", "uint"))
    )


def _infer_generic_constructor_type_name(generator, expr):
    if not isinstance(expr, ConstructorNode):
        return None

    constructor_type = _constructor_type_name(generator, expr)
    if not constructor_type:
        return None

    base_name, explicit_args = generic_type_parts(constructor_type)
    if explicit_args:
        return constructor_type

    struct_node = _generic_constructor_struct_node(generator, base_name)
    generic_params = _generic_constructor_struct_parameters(
        generator, base_name, struct_node
    )
    if not generic_params:
        return constructor_type

    field_types = _generic_constructor_field_types(generator, base_name, struct_node)
    if not field_types:
        return None

    substitutions = {}
    generic_param_set = set(generic_params)
    named_args = getattr(expr, "named_arguments", {}) or {}
    positional_args = list(getattr(expr, "arguments", []) or [])
    for index, (field_name, field_type) in enumerate(field_types):
        arg = named_args.get(field_name)
        if arg is None and index < len(positional_args):
            arg = positional_args[index]
        if arg is None:
            continue
        actual_type = generator.type_name_string(
            _safe_expression_result_type(generator, arg)
        )
        if not actual_type:
            continue
        _collect_type_parameter_bindings(
            generator.type_name_string(field_type),
            actual_type,
            substitutions,
            generic_param_set,
        )

    if any(param not in substitutions for param in generic_params):
        return None
    return f"{base_name}<{', '.join(substitutions[param] for param in generic_params)}>"


def _constructor_type_name(generator, expr):
    return generator.type_name_string(
        getattr(expr, "constructor_type", None)
        or getattr(expr, "vtype", None)
        or getattr(expr, "expression_type", None)
    )


def _generic_constructor_struct_node(generator, base_name):
    for attr_name in ("structs_by_name", "user_structs_by_name", "struct_declarations"):
        structs = getattr(generator, attr_name, {}) or {}
        struct_node = structs.get(base_name)
        if struct_node is not None:
            return struct_node
    return None


def _generic_constructor_struct_parameters(generator, base_name, struct_node):
    if struct_node is not None:
        return [
            getattr(param, "name", None)
            for param in getattr(struct_node, "generic_params", []) or []
            if getattr(param, "name", None)
        ]
    return list(
        (getattr(generator, "generic_struct_type_params", {}) or {}).get(base_name)
        or []
    )


def _generic_constructor_field_types(generator, base_name, struct_node):
    if struct_node is not None:
        fields = []
        for member in getattr(struct_node, "members", []) or []:
            field_name = getattr(member, "name", None)
            field_type = getattr(member, "member_type", getattr(member, "vtype", None))
            if field_name and field_type is not None:
                fields.append((field_name, field_type))
        return fields
    return list(
        ((getattr(generator, "struct_types", {}) or {}).get(base_name) or {}).items()
    )


def _safe_call_argument_type_name(generator, expr):
    fallback_type = generator.type_name_string(
        _fallback_expression_result_type(generator, expr)
    )
    try:
        call_argument_type_name = getattr(generator, "call_argument_type_name")
    except AttributeError:
        call_argument_type_name = None
    if call_argument_type_name is not None:
        try:
            actual_type = call_argument_type_name(expr)
        except (AttributeError, TypeError, ValueError):
            actual_type = None
        if actual_type:
            actual_type = generator.type_name_string(actual_type)
            if _is_more_specific_generic_type(fallback_type, actual_type):
                return fallback_type
            return actual_type

    if fallback_type:
        return fallback_type
    return generator.type_name_string(_safe_expression_result_type(generator, expr))


def _type_argument_name(generator, expr):
    if isinstance(expr, str):
        return expr
    if isinstance(expr, IdentifierNode):
        return expr.name
    return generator.type_name_string(expr)


def _is_more_specific_generic_type(candidate_type, current_type):
    if not candidate_type or not current_type:
        return False
    candidate_base, candidate_args = generic_type_parts(candidate_type)
    current_base, current_args = generic_type_parts(current_type)
    return candidate_base == current_base and candidate_args and not current_args


def _normalize_signature_type_name(generator, type_name):
    try:
        normalize = getattr(generator, "normalize_signature_type_name")
    except AttributeError:
        normalize = None
    if normalize is not None:
        return normalize(type_name)
    type_name = generator.type_name_string(type_name)
    return str(type_name).replace(" ", "") if type_name is not None else None


def _types_compatible(generator, expected_type, actual_type):
    try:
        compatible = getattr(generator, "storage_buffer_parameter_type_is_compatible")
    except AttributeError:
        compatible = None
    if compatible is not None:
        try:
            is_storage_buffer = getattr(
                generator, "is_storage_buffer_resource_type_name", None
            )
            if is_storage_buffer is not None and (
                is_storage_buffer(expected_type) or is_storage_buffer(actual_type)
            ):
                return compatible(expected_type, actual_type)
        except (AttributeError, TypeError, ValueError):
            pass

    try:
        scalar_or_vector = getattr(generator, "scalar_or_vector_type_compatible")
    except AttributeError:
        scalar_or_vector = None
    if scalar_or_vector is not None:
        try:
            return scalar_or_vector(expected_type, actual_type)
        except (AttributeError, TypeError, ValueError):
            return False
    return False


def _function_call_name(call):
    func_expr = getattr(call, "function", getattr(call, "name", None))
    return getattr(func_expr, "name", func_expr) if func_expr is not None else None


def _strip_reference_wrappers(type_text):
    type_text = str(type_text or "").strip()
    while type_text.startswith("&"):
        type_text = type_text[1:].strip()
    while type_text.endswith("&"):
        type_text = type_text[:-1].strip()
    return type_text


def _pointer_pointee_type(type_text):
    type_text = _strip_reference_wrappers(type_text)
    if not type_text.endswith("*"):
        return None
    pointee_type = type_text[:-1].strip()
    return pointee_type or None
