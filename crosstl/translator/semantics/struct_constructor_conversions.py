"""Structure constructor arguments converted to their member types.

CrossGL structure constructors follow GLSL: each argument sets one member, in order, and may be of a type that
converts implicitly to the member's type (GLSL 4.60 section 5.4.3, "Structure Constructors", and section 4.1.10,
"Implicit Conversions"). HLSL converts the arguments of its structure initializers the same way. ``Pair(x, k)``
with ``float`` members and an ``int`` ``k`` therefore sets ``b`` to ``float(k)``.

Most targets have no such conversion. Metal, CUDA and HIP build the structure with aggregate initialization,
``Pair{x, k}``, where converting a non-constant ``int`` to ``float`` is a narrowing conversion and an error (C++14
[dcl.init.list] paragraph 7). WGSL has no implicit conversion between concrete types (WGSL section 15.1), SPIR-V's
``OpCompositeConstruct`` takes constituents of the member types exactly, and GLSL ES has no implicit conversions.

:func:`convert_struct_constructor_arguments` makes the conversion explicit for every target: an argument of a
known scalar or vector type that differs from its member's type, with the same number of lanes, becomes a
conversion constructor of the member's type, ``Pair(x, float(k))``. Explicit conversions mean the same in every
target. Arguments of unknown type, structures, arrays and matrices, and calls whose argument count differs from
the member count, are left as they are.
"""

from __future__ import annotations

from .. import ast as A
from .hub_types import call_name, type_from_node
from .rewrite import call
from .scoped_walk import ScopedWalker, program_stages
from .struct_scalar_splats import struct_member_types

__all__ = ["convert_struct_constructor_arguments"]

_VECTOR_PREFIXES = {
    "float": "vec",
    "int": "ivec",
    "uint": "uvec",
    "double": "dvec",
    "bool": "bvec",
}


def convert_struct_constructor_arguments(program) -> int:
    """Convert structure constructor arguments to their members' types in place; returns how many were converted."""
    if not isinstance(program, A.ShaderNode):
        return 0
    structs = struct_member_types(program)
    if not structs:
        return 0
    walker = _ConversionWalker(program, structs)
    walker.run_program()
    for initializer in _program_initializers(program):
        walker.walk_expression(initializer)
    return walker.converted


def _program_initializers(program):
    """Initializers of program constants and globals, which no function body contains."""
    initializers = [
        getattr(constant, "value", None)
        for constant in getattr(program, "constants", None) or []
    ]
    initializers.extend(
        getattr(variable, "initial_value", None)
        for variable in getattr(program, "global_variables", None) or []
    )
    for stage in program_stages(program):
        initializers.extend(
            getattr(variable, "initial_value", None)
            for variable in getattr(stage, "local_variables", None) or []
        )
    return [initializer for initializer in initializers if initializer is not None]


class _ConversionWalker(ScopedWalker):
    def __init__(self, program, structs):
        super().__init__(program)
        self.structs = structs
        self.converted = 0

    def walk_expression(self, expr):
        """Walk a program-scope initializer with the program's declarations in scope."""
        self._walk(expr)

    def visit(self, node):
        if not isinstance(node, A.FunctionCallNode):
            return
        members = self.structs.get(call_name(node))
        arguments = node.arguments
        if not members or not isinstance(arguments, list):
            return
        if len(arguments) != len(members):
            return
        for index, (argument, (_member_name, member_type)) in enumerate(
            zip(arguments, members)
        ):
            converted = self._converted_argument(argument, member_type)
            if converted is not None:
                arguments[index] = converted
                self.converted += 1

    def _converted_argument(self, argument, member_type):
        member_hub = type_from_node(member_type)
        if member_hub is None:
            return None
        argument_hub = self.scope.infer(argument)
        if argument_hub is None or argument_hub == member_hub:
            return None
        if argument_hub.lanes != member_hub.lanes:
            return None
        spelling = _spelling(member_type)
        if spelling is None:
            return None
        conversion = call(spelling, [argument])
        location = getattr(argument, "source_location", None)
        if location is not None:
            conversion.source_location = location
        return conversion


def _spelling(type_node):
    """The CrossGL conversion constructor of a scalar or vector type, or ``None``."""
    if isinstance(type_node, A.PrimitiveType):
        return type_node.name
    if isinstance(type_node, A.NamedType):
        return None if type_node.generic_args else type_node.name
    if isinstance(type_node, A.VectorType):
        element = _spelling(type_node.element_type)
        prefix = _VECTOR_PREFIXES.get(element)
        size = type_node.size
        return f"{prefix}{size}" if prefix and str(size).isdigit() else None
    return None
