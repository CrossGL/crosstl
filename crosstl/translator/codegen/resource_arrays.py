"""Helpers for collecting resource-array size hints from CrossGL AST nodes."""

from ..ast import ArrayAccessNode, FunctionCallNode


def split_array_suffix(array_suffix):
    dimensions = []
    offset = 0
    while offset < len(array_suffix):
        if array_suffix[offset] != "[":
            return None
        end = array_suffix.find("]", offset + 1)
        if end == -1:
            return None
        dimensions.append(array_suffix[offset + 1 : end])
        offset = end + 1
    return dimensions


def format_array_declarator(
    mapped_base,
    name,
    array_suffix,
    dynamic_array_as_pointer=True,
):
    if not dynamic_array_as_pointer or "[]" not in array_suffix:
        return f"{mapped_base} {name}{array_suffix}"

    dimensions = split_array_suffix(array_suffix)
    if not dimensions:
        return f"{mapped_base} {name}{array_suffix}"

    if dimensions[0] == "":
        trailing_dimensions = dimensions[1:]
        if not trailing_dimensions:
            return f"{mapped_base}* {name}"
        if all(dimension != "" for dimension in trailing_dimensions):
            trailing_suffix = "".join(f"[{dimension}]" for dimension in trailing_dimensions)
            return f"{mapped_base} (*{name}){trailing_suffix}"

    collapsed_suffix = "".join(
        f"[{dimension}]" for dimension in dimensions if dimension != ""
    )
    return f"{mapped_base}* {name}{collapsed_suffix}"


def collect_resource_array_size_hints(
    *,
    global_arrays,
    function_arrays,
    fixed_global_array_sizes=None,
    fixed_function_array_sizes=None,
    functions,
    walk_nodes,
    expression_name,
    literal_int_value,
    visible_literal_int_constants,
    function_call_name,
    initial_size,
    format_size,
):
    fixed_global_array_sizes = fixed_global_array_sizes or {}
    fixed_function_array_sizes = fixed_function_array_sizes or {}
    global_hints = {name: initial_size for name in global_arrays}
    function_hints = {
        func_name: {param_name: initial_size for param_name in params}
        for func_name, params in function_arrays.items()
    }
    fixed_requirements = {}
    functions_by_name = {
        getattr(func, "name", None): func
        for func in functions
    }
    functions_by_name = {
        name: func for name, func in functions_by_name.items() if name
    }

    def assert_not_larger_than_known_fixed(name, fixed_size, required_size):
        if required_size > fixed_size:
            raise ValueError(
                "Conflicting fixed resource array sizes for "
                f"'{name}': {fixed_size} and {required_size}"
            )

    for func_name, func in functions_by_name.items():
        visible_constants = visible_literal_int_constants(func)
        for node in walk_nodes(getattr(func, "body", [])) or ():
            if not isinstance(node, ArrayAccessNode):
                continue
            array_expr = getattr(node, "array", getattr(node, "array_expr", None))
            index_expr = getattr(node, "index", getattr(node, "index_expr", None))
            array_name = expression_name(array_expr)
            index = literal_int_value(index_expr, visible_constants)
            if array_name is None or index is None or index < 0:
                continue
            required_size = index + 1
            if array_name in global_hints:
                global_hints[array_name] = max(global_hints[array_name], required_size)
            if array_name in fixed_global_array_sizes:
                assert_not_larger_than_known_fixed(
                    array_name, fixed_global_array_sizes[array_name], required_size
                )
            if array_name in function_hints.get(func_name, {}):
                function_hints[func_name][array_name] = max(
                    function_hints[func_name][array_name], required_size
                )
            fixed_params = fixed_function_array_sizes.get(func_name, {})
            if array_name in fixed_params:
                assert_not_larger_than_known_fixed(
                    array_name, fixed_params[array_name], required_size
                )

    def register_fixed_requirement(scope_key, size, current_size):
        if size is None:
            return
        existing = fixed_requirements.get(scope_key)
        if existing is not None and existing != size:
            raise ValueError(
                "Conflicting fixed resource array sizes for "
                f"'{scope_key[-1]}': {existing} and {size}"
            )
        if current_size and current_size > size:
            raise ValueError(
                "Conflicting fixed resource array sizes for "
                f"'{scope_key[-1]}': {current_size} and {size}"
            )
        fixed_requirements[scope_key] = size

    def assert_not_larger_than_fixed(scope_key, required_size):
        fixed_size = fixed_requirements.get(scope_key)
        if fixed_size is not None and required_size > fixed_size:
            raise ValueError(
                "Conflicting fixed resource array sizes for "
                f"'{scope_key[-1]}': {fixed_size} and {required_size}"
            )

    changed = True
    while changed:
        changed = False
        for caller_name, func in functions_by_name.items():
            caller_param_hints = function_hints.get(caller_name, {})
            caller_fixed_hints = fixed_function_array_sizes.get(caller_name, {})
            for call in walk_nodes(getattr(func, "body", [])) or ():
                if not isinstance(call, FunctionCallNode):
                    continue
                callee_name = function_call_name(call)
                callee_param_hints = function_hints.get(callee_name, {})
                callee_fixed_hints = fixed_function_array_sizes.get(callee_name, {})
                if not callee_param_hints and not callee_fixed_hints:
                    continue
                callee = functions_by_name.get(callee_name)
                if callee is None:
                    continue
                callee_params = getattr(callee, "parameters", [])
                args = getattr(call, "arguments", getattr(call, "args", []))
                for index, arg in enumerate(args):
                    if index >= len(callee_params):
                        continue
                    callee_param_name = getattr(callee_params[index], "name", None)
                    required_size = (
                        callee_param_hints.get(callee_param_name)
                        if callee_param_name in callee_param_hints
                        else callee_fixed_hints.get(callee_param_name)
                    )
                    fixed_size = callee_fixed_hints.get(callee_param_name)
                    arg_name = expression_name(arg)
                    arg_size = None
                    arg_is_fixed = False
                    arg_scope_key = None
                    if arg_name in global_hints:
                        arg_scope_key = ("global", arg_name)
                        arg_size = global_hints[arg_name]
                    elif arg_name in fixed_global_array_sizes:
                        arg_size = fixed_global_array_sizes[arg_name]
                        arg_is_fixed = True
                        if fixed_size is not None and arg_size != fixed_size:
                            raise ValueError(
                                "Conflicting fixed resource array sizes for "
                                f"'{arg_name}': {arg_size} and {fixed_size}"
                            )
                    elif arg_name in caller_param_hints:
                        arg_scope_key = ("param", caller_name, arg_name)
                        arg_size = caller_param_hints[arg_name]
                    elif arg_name in caller_fixed_hints:
                        arg_size = caller_fixed_hints[arg_name]
                        arg_is_fixed = True
                        if fixed_size is not None and arg_size != fixed_size:
                            raise ValueError(
                                "Conflicting fixed resource array sizes for "
                                f"'{arg_name}': {arg_size} and {fixed_size}"
                            )
                    if arg_scope_key is not None:
                        register_fixed_requirement(
                            arg_scope_key, fixed_size, arg_size
                        )
                    if (
                        callee_param_name in callee_param_hints
                        and arg_size
                        and arg_size > callee_param_hints[callee_param_name]
                    ):
                        callee_param_hints[callee_param_name] = arg_size
                        changed = True
                    if not required_size:
                        continue
                    if (
                        arg_name in global_hints
                        and required_size > global_hints[arg_name]
                    ):
                        assert_not_larger_than_fixed(
                            ("global", arg_name), required_size
                        )
                        global_hints[arg_name] = required_size
                        changed = True
                    if (
                        arg_name in caller_param_hints
                        and required_size > caller_param_hints[arg_name]
                    ):
                        assert_not_larger_than_fixed(
                            ("param", caller_name, arg_name), required_size
                        )
                        caller_param_hints[arg_name] = required_size
                        changed = True
                    if arg_is_fixed and arg_size and required_size > arg_size:
                        raise ValueError(
                            "Conflicting fixed resource array sizes for "
                            f"'{arg_name}': {arg_size} and {required_size}"
                        )

    return (
        {name: format_size(size) for name, size in global_hints.items()},
        {
            func_name: {
                param_name: format_size(size)
                for param_name, size in param_hints.items()
            }
            for func_name, param_hints in function_hints.items()
        },
    )
