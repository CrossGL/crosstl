"""Names for generic function specializations that no program declaration uses.

Targets without generic functions emit one concrete clone per instantiation and name it after the generic and its
type arguments, so ``pick<float>`` becomes ``pick_float``. A program may already declare that name, for example a
concrete ``float pick_float(float, float)`` beside ``generic<T> T pick(T, T)``. Emitting the clone under the same name
would define the function twice, and every call of either function would name one definition. A clone whose
preferred name is taken therefore gets the first free ``<name>_<n>``, counting from 2 as the Metal importer's
specializations do.

The reservation covers every function and global the program declares and every name an earlier clone took.
Specializations are created in a deterministic order, so the chosen names are deterministic too.
"""

__all__ = ["reserve_program_names", "unique_specialization_name"]


def reserve_program_names(generator, functions, global_names=()):
    """Reserve the names of the program's functions and globals before any clone is named."""
    reserved = {
        name
        for name in (getattr(func, "name", None) for func in functions or [])
        if isinstance(name, str) and name
    }
    reserved.update(name for name in global_names or () if isinstance(name, str))
    generator.generic_function_reserved_names = reserved


def unique_specialization_name(generator, preferred):
    """Take ``preferred`` for a clone, or the first free ``<preferred>_<n>`` when the program uses it."""
    reserved = getattr(generator, "generic_function_reserved_names", None)
    if reserved is None:
        reserved = set()
        generator.generic_function_reserved_names = reserved
    name = preferred
    ordinal = 2
    while name in reserved:
        name = f"{preferred}_{ordinal}"
        ordinal += 1
    reserved.add(name)
    return name
