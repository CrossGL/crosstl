"""Checked metadata bounds that helper parameters inherit from their calls.

Metadata preconditions bound an entry point's scalar inputs, and the loop
proofs of the software-subgroup lowerings read those bounds. A helper's scalar
parameter has no source identity of its own, but it holds whatever its calls
pass. When every reachable call passes either a bounded input of the caller or
an integer literal, and the argument has the parameter's own integer type, the
parameter lies within the hull of those bounds for as long as the helper never
reassigns or shadows it. A call that proves nothing for a parameter leaves it
unbounded, so a helper inherits only what all of its calls agree on.
"""

from ..arithmetic_conversions import ArithmeticScalarKind, source_integer_shape
from ..ast import LiteralNode, ReferenceType, VariableNode
from .loop_induction import metadata_bound_key


def _dtype(type_name):
    """The metadata ``dtype`` spelling of a scalar integer type, if it is one."""
    shape = source_integer_shape(type_name) if isinstance(type_name, str) else None
    if shape is None or shape[2] != 1 or shape[1] not in {8, 16, 32, 64}:
        return None
    unsigned = shape[0] == ArithmeticScalarKind.UNSIGNED_INTEGER
    return f"{'u' if unsigned else ''}int{shape[1]}"


def _representable(value, dtype):
    bits = int(dtype[dtype.index("int") + 3 :])
    if dtype.startswith("u"):
        return 0 <= value < (1 << bits)
    return -(1 << (bits - 1)) <= value < (1 << (bits - 1))


def call_parameter_bounds(parameters, arguments, bounds, type_name):
    """The bound records one call proves, keyed by parameter index.

    ``bounds`` are the caller's own records, keyed as ``metadata_bound_key``
    spells an argument; ``type_name`` spells a parameter's type.
    """
    result = {}
    for index, (parameter, argument) in enumerate(zip(parameters, arguments)):
        qualifiers = set(getattr(parameter, "qualifiers", ()) or ())
        if isinstance(parameter.param_type, ReferenceType) or qualifiers & {
            "out",
            "inout",
        }:
            continue
        dtype = _dtype(type_name(parameter.param_type))
        if dtype is None:
            continue
        if isinstance(argument, LiteralNode):
            value = argument.value
            if type(value) is int and _representable(value, dtype):
                result[index] = {"dtype": dtype, "minimum": value, "maximum": value}
            continue
        record = (bounds or {}).get(metadata_bound_key(argument))
        if record is not None and record.get("dtype") == dtype:
            result[index] = {
                "dtype": dtype,
                "minimum": record["minimum"],
                "maximum": record["maximum"],
            }
    return result


def join_parameter_bounds(previous, current):
    """Keep what every call proves, widened to cover each of them.

    ``previous`` is None before the first call.
    """
    if previous is None:
        return dict(current)
    return {
        index: {
            "dtype": record["dtype"],
            "minimum": min(record["minimum"], current[index]["minimum"]),
            "maximum": max(record["maximum"], current[index]["maximum"]),
        }
        for index, record in sorted(previous.items())
        if index in current and current[index]["dtype"] == record["dtype"]
    }


def inherited_metadata_bindings(function, inherited, mutable_names, walk_ast):
    """Inherited bounds, by name, of the parameters a helper never rebinds."""
    if not inherited:
        return {}
    invalid = set(mutable_names) | {
        node.name for node in walk_ast(function.body) if isinstance(node, VariableNode)
    }
    parameters = getattr(function, "parameters", ()) or ()
    names = [parameter.name for parameter in parameters]
    return {
        name: inherited[index]
        for index, name in enumerate(names)
        if index in inherited and name not in invalid and names.count(name) == 1
    }
