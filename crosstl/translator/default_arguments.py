"""Utilities for lowering defaulted function arguments in ASTs."""

from __future__ import annotations

import copy

from .ast import IdentifierNode


def lower_default_arguments(ast):
    """Append explicit default values to calls that omit trailing arguments."""
    if ast is None:
        return ast

    functions_by_name = {}
    for function in _iter_functions(ast):
        functions_by_name.setdefault(function.name, []).append(function)

    for node in _walk_ast(ast):
        if _node_class_name(node) != "FunctionCallNode":
            continue
        function_name = _function_call_name(node)
        if not function_name:
            continue
        arguments = list(getattr(node, "arguments", getattr(node, "args", [])) or [])
        target_function = _default_argument_target(
            functions_by_name.get(function_name, []), len(arguments)
        )
        if target_function is None:
            continue
        parameters = _function_parameters(target_function)
        for parameter in parameters[len(arguments) :]:
            arguments.append(_clone_default_value(_parameter_default_value(parameter)))
        if hasattr(node, "arguments"):
            node.arguments = arguments
        node.args = arguments

    return ast


def _clone_default_value(value):
    clone = copy.deepcopy(value)
    _clear_parent_links(clone)
    return clone


def _clear_parent_links(value):
    stack = [value]
    seen = set()
    while stack:
        node = stack.pop()
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)

        if hasattr(node, "parent"):
            node.parent = None

        child_nodes = getattr(node, "child_nodes", None)
        if callable(child_nodes):
            stack.extend(child_nodes())
            continue

        if isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, (list, tuple, set)):
            stack.extend(node)


def _walk_ast(root):
    walk = getattr(root, "walk", None)
    if callable(walk):
        yield from walk()
        return

    stack = [root]
    seen = set()
    while stack:
        value = stack.pop()
        value_id = id(value)
        if value_id in seen:
            continue
        seen.add(value_id)

        if isinstance(value, dict):
            stack.extend(value.values())
            continue
        if isinstance(value, (list, tuple, set)):
            stack.extend(value)
            continue
        if not hasattr(value, "__dict__"):
            continue

        yield value
        for field_name, field_value in vars(value).items():
            if field_name in {"annotations", "parent", "source_location"}:
                continue
            stack.append(field_value)


def _iter_functions(ast):
    for function in getattr(ast, "functions", []) or []:
        if _node_class_name(function) == "FunctionNode":
            yield function
    for stage in getattr(ast, "stages", {}).values():
        for function in getattr(stage, "functions", []) or []:
            if _node_class_name(function) == "FunctionNode":
                yield function


def _node_class_name(node):
    return node.__class__.__name__ if hasattr(node, "__class__") else ""


def _function_call_name(node):
    function = getattr(node, "function", getattr(node, "name", None))
    if isinstance(function, IdentifierNode):
        return function.name
    if isinstance(function, str):
        return function
    return None


def _default_argument_target(functions, argument_count):
    if any(
        len(_function_parameters(function)) == argument_count for function in functions
    ):
        return None

    candidates = []
    for function in functions:
        parameters = _function_parameters(function)
        if argument_count >= len(parameters):
            continue
        omitted = parameters[argument_count:]
        if all(
            _parameter_default_value(parameter) is not None for parameter in omitted
        ):
            candidates.append(function)
    return candidates[0] if len(candidates) == 1 else None


def _function_parameters(function):
    return list(getattr(function, "parameters", getattr(function, "params", [])) or [])


def _parameter_default_value(parameter):
    return getattr(parameter, "default_value", getattr(parameter, "value", None))
