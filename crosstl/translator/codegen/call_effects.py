"""Builtin argument writes shared by collective control-flow proofs."""

_BUILTIN_ARGUMENT_WRITES = {
    ("round", 1): (),
    ("buffer_load", 2): (),
    ("buffer_store", 3): (0,),
}


def builtin_argument_write_indices(name, argument_count):
    """Return writable positions, or None when the call has no known contract.

    Callers must first exclude user-defined functions and member calls. Argument
    expressions still require traversal for nested calls, writes and escapes.
    No argument writes does not imply a uniform result, especially for loads.
    """
    return _BUILTIN_ARGUMENT_WRITES.get((name, argument_count))
