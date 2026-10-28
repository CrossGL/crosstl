"""Storage array parameters that receive views into a buffer.

A C++-style array parameter in a storage address space, such as Metal's
``constant const int64_t strides[2]``, is adjusted to a pointer to its element
([dcl.fct]/5), so a caller may pass any pointer into the storage, including the
view ``strides + 2``. HLSL passes a buffer to a function only whole, so such a
parameter must take the base-offset ABI of storage pointers. A parameter that
only ever receives whole buffers keeps the plain buffer parameter.

The analysis is a monotone fixed point over call edges, matched by callee name
and argument position. A parameter receives a view when a call passes it
anything except a whole buffer: pointer arithmetic, a local pointer, a storage
pointer parameter, or a storage array parameter that itself receives a view.
"""

from ..ast import ArrayType, FunctionCallNode, IdentifierNode, PointerType, VariableNode
from .ast_walk import walk_ast_values

STORAGE_ARRAY_ADDRESS_SPACES = ("constant", "device")


def parameter_type(parameter):
    return getattr(parameter, "param_type", getattr(parameter, "vtype", None))


def storage_array_address_space(parameter):
    """Return the address space of a storage array parameter, else ``None``.

    Stage-entry resources carry binding attributes and are buffers, not views.
    """

    value_type = parameter_type(parameter)
    if not isinstance(value_type, ArrayType) or isinstance(
        value_type.element_type, (ArrayType, PointerType)
    ):
        return None
    if getattr(parameter, "attributes", None):
        return None
    qualifiers = [
        str(qualifier).lower()
        for qualifier in getattr(parameter, "qualifiers", []) or []
    ]
    return next(
        (space for space in STORAGE_ARRAY_ADDRESS_SPACES if space in qualifiers),
        None,
    )


def storage_array_pointer_type(parameter):
    """Return the pointer type a storage array parameter is adjusted to."""

    address_space = storage_array_address_space(parameter)
    qualifiers = {
        str(qualifier).lower()
        for qualifier in getattr(parameter, "qualifiers", []) or []
    }
    return PointerType(
        parameter_type(parameter).element_type,
        is_mutable=address_space == "device" and "const" not in qualifiers,
        address_space=address_space,
        source_location=getattr(parameter, "source_location", None),
    )


def _parameters(function):
    return list(getattr(function, "parameters", getattr(function, "params", [])) or [])


def _call_arguments(call):
    return list(getattr(call, "arguments", getattr(call, "args", [])) or [])


def _local_names(body):
    return {
        node.name
        for node in walk_ast_values(body)
        if isinstance(node, VariableNode) and isinstance(node.name, str)
    }


def view_storage_array_parameters(functions, call_name):
    """Return the storage array parameters that receive views, keyed by ``id``.

    ``call_name`` maps a call node to the source name of its callee.
    """

    functions = list(functions or [])
    arrays_by_name = {}
    for function in functions:
        for index, parameter in enumerate(_parameters(function)):
            if storage_array_address_space(parameter) is not None:
                arrays_by_name.setdefault(function.name, {}).setdefault(
                    index, []
                ).append(parameter)
    if not arrays_by_name:
        return {}

    marked = {}
    edges = []

    def mark(parameters):
        for parameter in parameters:
            marked.setdefault(id(parameter), parameter)

    for function in functions:
        body = getattr(function, "body", None)
        if body is None:
            continue
        parameters = {parameter.name: parameter for parameter in _parameters(function)}
        locals_ = None
        for call in walk_ast_values(body):
            if not isinstance(call, FunctionCallNode):
                continue
            positions = arrays_by_name.get(call_name(call))
            if not positions:
                continue
            arguments = _call_arguments(call)
            for index, targets in sorted(positions.items()):
                if index >= len(arguments):
                    continue
                argument = arguments[index]
                if not isinstance(argument, IdentifierNode):
                    mark(targets)
                    continue
                source = parameters.get(argument.name)
                if source is not None:
                    if storage_array_address_space(source) is not None:
                        edges.append((source, targets))
                    elif isinstance(parameter_type(source), PointerType):
                        mark(targets)
                    continue
                if locals_ is None:
                    locals_ = _local_names(body)
                if argument.name in locals_:
                    mark(targets)

    changed = True
    while changed:
        changed = False
        for source, targets in edges:
            if id(source) in marked and any(
                id(target) not in marked for target in targets
            ):
                mark(targets)
                changed = True
    return marked
