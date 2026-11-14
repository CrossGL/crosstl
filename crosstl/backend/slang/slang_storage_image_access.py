"""CrossGL lowering of Slang storage-texture subscripts and ``Interlocked*`` atomics.

Slang reads and writes a storage texture's texels through subscripts, as HLSL
does: ``image[coord]`` loads a texel, ``image[coord] = value`` stores one and
``InterlockedAdd(image[coord], value, original)`` updates one atomically.
CrossGL, like GLSL, has no addressable texels, so these members write the
image operations the HLSL importer writes:

- a load becomes ``imageLoad`` narrowed to the declared element type, because
  CrossGL image loads return four components;
- a store becomes ``imageStore`` of the value widened to four components with
  the texel fill (0, 0, 1); a compound store and an increment load the texel
  first, and a store to some components updates a loaded copy of the texel;
- ``Interlocked*`` on a texel becomes ``imageAtomic*``, on any other location
  (buffer elements, groupshared variables) ``atomic*``, and a
  ``RWByteAddressBuffer`` ``Interlocked*`` method ``atomic*(buffer, offset,
  ...)``; an ``original`` argument receives the returned value.

Every other texel write (one inside an expression, an atomic on float texels)
and every ``Interlocked*`` operation CrossGL has no atomic for is refused
rather than emitted verbatim.
"""

from __future__ import annotations

import re

from ..common_ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    CallNode,
    FunctionCallNode,
    MemberAccessNode,
    MethodCallNode,
    TernaryOpNode,
    UnaryOpNode,
    VariableNode,
)
from ..import_refusals import SlangImportRefused
from ..storage_image_texels import storage_image_element_type

__all__ = ["SlangStorageImageAccessMixin", "texel_element_shape"]

# Interlocked operation -> (location atomic, image atomic, value arguments, argument counts).
# The argument counts include the destination and the optional ``original`` output.
_INTERLOCKED = {
    "InterlockedAdd": ("atomicAdd", "imageAtomicAdd", 1, (2, 3)),
    "InterlockedAnd": ("atomicAnd", "imageAtomicAnd", 1, (2, 3)),
    "InterlockedOr": ("atomicOr", "imageAtomicOr", 1, (2, 3)),
    "InterlockedXor": ("atomicXor", "imageAtomicXor", 1, (2, 3)),
    "InterlockedMin": ("atomicMin", "imageAtomicMin", 1, (2, 3)),
    "InterlockedMax": ("atomicMax", "imageAtomicMax", 1, (2, 3)),
    "InterlockedExchange": ("atomicExchange", "imageAtomicExchange", 1, (2, 3)),
    "InterlockedCompareExchange": (
        "atomicCompareExchange",
        "imageAtomicCompSwap",
        2,
        (4,),
    ),
    "InterlockedCompareStore": (
        "atomicCompareExchange",
        "imageAtomicCompSwap",
        2,
        (3,),
    ),
}

_TEXEL_SCALAR = re.compile(
    r"(?:(?:unorm|snorm)\s*)?(?P<scalar>float|half|int|uint|dword)(?P<count>[1-4]?)"
)
_TEXEL_GENERIC = re.compile(
    r"(?:(?:unorm|snorm)\s*)?vector\s*<\s*(?P<scalar>float|half|int|uint|dword)"
    r"\s*,\s*(?P<count>[1-4])\s*>"
)
_SCALAR_FAMILY = {
    "float": "float",
    "half": "float",
    "int": "int",
    "uint": "uint",
    "dword": "uint",
}
_VECTOR_PREFIX = {"float": "vec", "int": "ivec", "uint": "uvec"}
_ZERO = {"float": "0.0", "int": "0", "uint": "0u"}
_ONE = {"float": "1.0", "int": "1", "uint": "1u"}
_NARROWING = {1: ".x", 2: ".xy", 3: ".xyz", 4: ""}
_INCREMENTS = {
    "POST_INCREMENT": ("+", "++"),
    "PRE_INCREMENT": ("+", "++"),
    "POST_DECREMENT": ("-", "--"),
    "PRE_DECREMENT": ("-", "--"),
}
_SWIZZLE_SETS = ("xyzw", "rgba")
_UNSIGNED_DECIMAL = re.compile(r"\d+")


def texel_element_shape(element_type):
    """``(scalar family, component count)`` of a storage texture element type.

    A missing element type is the four-component float default; ``None``
    means the element is not a scalar or vector of float, int or uint.
    """
    if element_type is None:
        return "float", 4
    text = " ".join(str(element_type).split())
    match = _TEXEL_SCALAR.fullmatch(text) or _TEXEL_GENERIC.fullmatch(text)
    if match is None:
        return None
    return _SCALAR_FAMILY[match.group("scalar")], int(match.group("count") or 1)


def _swizzle_within(member, count):
    """Whether ``member`` is a swizzle of the first ``count`` components."""
    if not member or len(member) > 4:
        return False
    return any(all(ch in names[:count] for ch in member) for names in _SWIZZLE_SETS)


def _parenthesized(expr, rendered):
    if isinstance(expr, (AssignmentNode, BinaryOpNode, TernaryOpNode, UnaryOpNode)):
        return f"({rendered})"
    return rendered


class SlangStorageImageAccessMixin:
    """Storage-texture texel and ``Interlocked*`` members of ``SlangToCrossGLConverter``."""

    # ------------------------------------------------------------ texels

    def storage_image_texel_resource(self, expr):
        """Storage texture type of the image ``expr`` subscripts to one texel, else ``None``."""
        if not isinstance(expr, ArrayAccessNode):
            return None
        owner = self.matrix_value_type(expr.array)
        if owner is None or owner[1]:
            return None
        if not self.is_storage_image_resource_type(owner[0]):
            return None
        return owner[0]

    def storage_image_texel_shape(self, access, resource_type):
        """``(scalar family, count)`` of the texels ``access`` reaches; refuses unsupported ones."""
        if self.is_multisample_resource_base(self.resource_type_base(resource_type)):
            self.refuse_storage_image_texel(
                access, "a multisampled storage texture has no texel subscript"
            )
        shape = texel_element_shape(storage_image_element_type(resource_type))
        if shape is None:
            self.refuse_storage_image_texel(
                access, f"CrossGL images have no texels of type '{resource_type}'"
            )
        return shape

    def refuse_storage_image_texel(self, access, reason):
        image = self.generate_expression(access.array)
        raise SlangImportRefused([f"Slang texel access on '{image}': {reason}"])

    def storage_image_access_parts(self, access, is_main):
        image = self.generate_expression(access.array, is_main)
        coord = self.generate_expression(access.index, is_main)
        return image, coord

    def generate_storage_image_texel_load(self, expr, is_main=False, narrow=True):
        """``imageLoad`` of a texel subscript, narrowed to the element type; ``None`` otherwise."""
        resource_type = self.storage_image_texel_resource(expr)
        if resource_type is None:
            return None
        _, count = self.storage_image_texel_shape(expr, resource_type)
        image, coord = self.storage_image_access_parts(expr, is_main)
        suffix = _NARROWING[count] if narrow else ""
        return f"imageLoad({image}, {coord}){suffix}"

    def generate_storage_image_texel_swizzle(self, expr, is_main=False):
        """``imageLoad(...).yz`` for a swizzle of a texel read within the element's components."""
        if not isinstance(expr, MemberAccessNode) or not isinstance(expr.member, str):
            return None
        resource_type = self.storage_image_texel_resource(expr.object)
        if resource_type is None:
            return None
        _, count = self.storage_image_texel_shape(expr.object, resource_type)
        if not _swizzle_within(expr.member, count):
            return None
        load = self.generate_storage_image_texel_load(expr.object, is_main, False)
        return f"{load}.{expr.member}"

    def storage_image_value_shape(self, expr):
        """``(scalar family, count)`` of a value, ``None`` when unknown."""
        resource_type = self.storage_image_texel_resource(expr)
        if resource_type is not None:
            return texel_element_shape(storage_image_element_type(resource_type))
        value_type = self.matrix_value_type(expr)
        if value_type is None or value_type[1]:
            return None
        return texel_element_shape(value_type[0])

    def widen_storage_image_texel(self, resource_type, value_shape, rendered):
        """``rendered`` widened to the four components an image store takes."""
        family, count = texel_element_shape(storage_image_element_type(resource_type))
        if count == 4:
            return rendered
        if value_shape is not None and value_shape[1] > count:
            rendered = f"({rendered}){_NARROWING[count]}"
            value_shape = (value_shape[0], count)
        if count > 1 and value_shape != (family, count):
            # Slang promotes a scalar to every component and converts the others.
            rendered = f"{_VECTOR_PREFIX[family]}{count}({rendered})"
        padding = ", ".join([_ZERO[family], _ZERO[family], _ONE[family]][count - 1 :])
        return f"{_VECTOR_PREFIX[family]}4({rendered}, {padding})"

    def storage_image_store_text(
        self, access, resource_type, value_shape, rendered, is_main
    ):
        image, coord = self.storage_image_access_parts(access, is_main)
        widened = self.widen_storage_image_texel(resource_type, value_shape, rendered)
        return f"imageStore({image}, {coord}, {widened})"

    def storage_image_update_store(
        self, access, resource_type, operator, operand, is_main
    ):
        """``imageStore(image, coord, imageLoad(image, coord) <op> operand)``, widened."""
        shape = self.storage_image_texel_shape(access, resource_type)
        self.require_side_effect_free_texel_coordinate(access, is_main)
        load = self.generate_storage_image_texel_load(access, is_main)
        return self.storage_image_store_text(
            access, resource_type, shape, f"{load} {operator} {operand}", is_main
        )

    def generate_storage_image_texel_store(self, node, is_main=False):
        """``imageStore`` for ``image[coord] = value`` and its compound forms; ``None`` otherwise."""
        if not isinstance(node, AssignmentNode):
            return None
        access = node.left
        resource_type = self.storage_image_texel_resource(access)
        if resource_type is None:
            return None
        self.storage_image_texel_shape(access, resource_type)
        right = self.generate_expression(node.right, is_main)
        if node.operator == "=":
            return self.storage_image_store_text(
                access,
                resource_type,
                self.storage_image_value_shape(node.right),
                right,
                is_main,
            )
        return self.storage_image_update_store(
            access,
            resource_type,
            node.operator[:-1],
            _parenthesized(node.right, right),
            is_main,
        )

    def require_side_effect_free_texel_coordinate(self, access, is_main):
        if not self.texel_expression_side_effect_free(access.index):
            coord = self.generate_expression(access.index, is_main)
            self.refuse_storage_image_texel(
                access,
                "updating a texel loads and stores it, so its coordinate "
                f"'{coord}' must have no side effects",
            )

    def texel_expression_side_effect_free(self, expr):
        """Whether evaluating ``expr`` twice is observably the same as once."""
        if expr is None or isinstance(expr, (str, int, float, bool, VariableNode)):
            return True
        if isinstance(expr, (AssignmentNode, MethodCallNode, CallNode)):
            return False
        if isinstance(expr, UnaryOpNode):
            return (
                expr.op not in _INCREMENTS
                and self.texel_expression_side_effect_free(expr.operand)
            )
        if isinstance(expr, FunctionCallNode):
            if not isinstance(expr.name, str) or expr.name in self.user_function_names:
                return False
            return all(self.texel_expression_side_effect_free(a) for a in expr.args)
        if isinstance(expr, BinaryOpNode):
            parts = (expr.left, expr.right)
        elif isinstance(expr, MemberAccessNode):
            parts = (expr.object,)
        elif isinstance(expr, ArrayAccessNode):
            parts = (expr.array, expr.index)
        elif isinstance(expr, TernaryOpNode):
            parts = (expr.condition, expr.true_expr, expr.false_expr)
        else:
            return False
        return all(self.texel_expression_side_effect_free(part) for part in parts)

    def generate_storage_image_statement(self, stmt, indent, is_main=False):
        """Newline-terminated text for a texel write statement, ``None`` for other statements."""
        if isinstance(stmt, UnaryOpNode) and stmt.op in _INCREMENTS:
            operator, spelled = _INCREMENTS[stmt.op]
            resource_type = self.storage_image_texel_resource(stmt.operand)
            if resource_type is not None:
                family, _ = self.storage_image_texel_shape(stmt.operand, resource_type)
                store = self.storage_image_update_store(
                    stmt.operand, resource_type, operator, _ONE[family], is_main
                )
                return f"{store};\n"
            return self.generate_storage_image_component_update(
                stmt.operand, spelled, None, indent, is_main
            )
        if not isinstance(stmt, AssignmentNode):
            return None
        store = self.generate_storage_image_texel_store(stmt, is_main)
        if store is not None:
            return f"{store};\n"
        return self.generate_storage_image_component_update(
            stmt.left, stmt.operator, stmt.right, indent, is_main
        )

    def generate_storage_image_component_update(
        self, target, operator, value, indent, is_main
    ):
        """Read-modify-write for ``image[coord].yz = value``; ``None`` for other targets.

        ``value`` is ``None`` for an increment or decrement, whose ``operator``
        is ``++`` or ``--``.
        """
        if not isinstance(target, MemberAccessNode) or not isinstance(
            target.member, str
        ):
            return None
        access = target.object
        resource_type = self.storage_image_texel_resource(access)
        if resource_type is None:
            return None
        family, count = self.storage_image_texel_shape(access, resource_type)
        component = target.member
        if not _swizzle_within(component, count) or len(set(component)) != len(
            component
        ):
            self.refuse_storage_image_texel(
                access, f"'.{component}' does not name distinct texel components"
            )
        rendered = None
        if value is not None:
            rendered = _parenthesized(value, self.generate_expression(value, is_main))
        if count == 1:
            # The texel's only component: the whole texel is stored.
            if value is None:
                store = self.storage_image_update_store(
                    access, resource_type, operator[0], _ONE[family], is_main
                )
            elif operator == "=":
                store = self.storage_image_store_text(
                    access,
                    resource_type,
                    self.storage_image_value_shape(value),
                    rendered,
                    is_main,
                )
            else:
                store = self.storage_image_update_store(
                    access, resource_type, operator[:-1], rendered, is_main
                )
            return f"{store};\n"
        self.require_side_effect_free_texel_coordinate(access, is_main)
        image, coord = self.storage_image_access_parts(access, is_main)
        vector = f"{_VECTOR_PREFIX[family]}4"
        temp = self.next_generated_temp_name("texel", vector)
        if value is None:
            update = f"{temp}.{component}{operator}"
        else:
            update = f"{temp}.{component} {operator} {rendered}"
        indent_text = "    " * indent
        return (
            f"{vector} {temp} = imageLoad({image}, {coord});\n"
            f"{indent_text}{update};\n"
            f"{indent_text}imageStore({image}, {coord}, {temp});\n"
        )

    def refuse_storage_image_texel_write(self, target):
        """Refuse a texel, or components of one, written where no statement can store it."""
        access = target.object if isinstance(target, MemberAccessNode) else target
        if self.storage_image_texel_resource(access) is not None:
            self.refuse_storage_image_texel(
                access, "a texel written inside an expression has no image operation"
            )

    # ------------------------------------------------------------ atomics

    def interlocked_operation(self, name, args, label):
        """``(location atomic, image atomic, value count)``; refuses unknown operations and arities."""
        operation = _INTERLOCKED.get(name)
        if operation is None:
            raise SlangImportRefused([f"Slang {label} has no CrossGL atomic operation"])
        location_atomic, image_atomic, value_count, counts = operation
        if len(args) not in counts:
            expected = " or ".join(str(count) for count in counts)
            raise SlangImportRefused(
                [f"Slang {label} takes {expected} arguments, not {len(args)}"]
            )
        return location_atomic, image_atomic, value_count

    def generate_interlocked_call(self, expr, function_name, is_main=False):
        """CrossGL atomic for an ``Interlocked*`` function call; ``None`` for other calls."""
        if not str(function_name).startswith("Interlocked"):
            return None
        if expr.name in self.user_function_names:
            return None
        args = list(expr.args or [])
        location_atomic, image_atomic, value_count = self.interlocked_operation(
            function_name, args, function_name
        )
        target = args[0]
        resource_type = self.storage_image_texel_resource(target)
        if resource_type is not None:
            family, count = self.storage_image_texel_shape(target, resource_type)
            if count != 1 or family == "float":
                self.refuse_storage_image_texel(
                    target,
                    f"{function_name} needs a texture of int or uint scalars, "
                    f"not '{resource_type}'",
                )
            operands = list(self.storage_image_access_parts(target, is_main))
            call_name = image_atomic
        else:
            self.refuse_storage_image_texel_write(target)
            shape = self.storage_image_value_shape(target)
            family = shape[0] if shape is not None else None
            if family == "float":
                raise SlangImportRefused(
                    [f"Slang {function_name} needs an int or uint location"]
                )
            operands = [self.generate_expression(target, is_main)]
            call_name = location_atomic
        return self.atomic_call_text(
            call_name, operands, args[1:], value_count, family, is_main
        )

    def generate_byte_address_interlocked_call(self, expr, obj, is_main=False):
        """``atomic*(buffer, offset, ...)`` for a byte-address ``Interlocked*`` method."""
        method = self.method_base_name(expr.method)
        if not str(method).startswith("Interlocked"):
            return None
        resource_type = self.buffer_resource_expression_type(expr.object)
        if (
            self.resource_type_base(resource_type)
            not in self.BYTE_ADDRESS_BUFFER_RESOURCE_TYPES
        ):
            return None
        args = list(expr.args or [])
        # The byte offset takes the destination's place among the arguments.
        location_atomic, _, value_count = self.interlocked_operation(
            method, args, f"{obj}.{expr.method}"
        )
        offset = self.generate_expression(args[0], is_main)
        return self.atomic_call_text(
            location_atomic, [obj, offset], args[1:], value_count, "uint", is_main
        )

    def atomic_call_text(self, name, operands, rest, value_count, family, is_main):
        """``name(operands, values)``, assigned to the ``original`` output when one follows."""
        values = [
            self.atomic_operand(value, family, is_main) for value in rest[:value_count]
        ]
        call = f"{name}({', '.join([*operands, *values])})"
        if len(rest) > value_count:
            return f"{self.generate_expression(rest[value_count], is_main)} = {call}"
        return call

    def atomic_operand(self, value, family, is_main):
        rendered = self.generate_expression(value, is_main)
        if family == "uint" and _UNSIGNED_DECIMAL.fullmatch(rendered):
            return f"{rendered}u"
        return rendered
