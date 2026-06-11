"""Helpers for monomorphizing generic functions in C-style shader outputs."""

from copy import deepcopy

from ..ast import (
    AssignmentNode,
    BlockNode,
    ConstructorNode,
    ExpressionStatementNode,
    ForNode,
    FunctionCallNode,
    FunctionNode,
    IfNode,
    MatchNode,
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


def reject_unsupported_generic_functions(ast_node, target_name):
    """Reject generic functions before emitting non-specialized target code."""
    for func in iter_function_nodes(ast_node):
        generic_params = generic_function_parameters(func)
        if not generic_params:
            continue
        suffix = f" ({', '.join(generic_params)})" if generic_params else ""
        raise ValueError(
            f"{target_name} codegen does not support generic functions{suffix}; "
            f"specialize the function before {target_name} generation"
        )


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
    if key is None:
        return None
    return (getattr(generator, "generic_function_specialized_names", {}) or {}).get(key)


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


def _generic_function_candidate_call_match(generator, func, func_name, args):
    generic_params = generic_function_parameters(func)
    if not generic_params:
        return None

    substitutions = {}
    generic_param_set = set(generic_params)
    param_list = list(getattr(func, "parameters", getattr(func, "params", [])) or [])
    matched_params = 0
    exact_matches = 0
    compatible_params = 0

    for param, arg in zip(param_list, args or []):
        expected_type = generator.type_name_string(
            getattr(param, "param_type", getattr(param, "vtype", None))
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
    args_count = len(args or [])
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
    value = 1 if method == "one" else 0
    if "float" in mapped_type or "half" in mapped_type or mapped_type == "double":
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
        generator.local_variable_types[stmt.name] = substitute_generic_type_name(
            declared_type or "float",
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
    if expected_type in generic_params:
        substitutions.setdefault(expected_type, actual_type)
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
        return generator.expression_result_type(expr)
    except (AttributeError, TypeError, ValueError):
        return None


def _safe_call_argument_type_name(generator, expr):
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
            return generator.type_name_string(actual_type)

    return generator.type_name_string(_safe_expression_result_type(generator, expr))


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
            expected_base = generator.array_base_type_name(str(expected_type))
            actual_base = generator.array_base_type_name(str(actual_type))
            if generator.structured_buffer_type_info(
                expected_base
            ) or generator.structured_buffer_type_info(actual_base):
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
