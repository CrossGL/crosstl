"""Mojo kernel arguments for HLSL buffer objects: structured and byte-address buffers become pointers.

Mojo has no global variables, so a buffer object declared outside a function cannot stay a module-scope ``var``.
A Mojo GPU kernel receives it the way it receives a CrossGL ``buffer T*``: as an ``UnsafePointer`` argument.

* ``StructuredBuffer<T>`` and ``RWStructuredBuffer<T>`` (and ``RasterizerOrderedStructuredBuffer<T>``) become an
  ``UnsafePointer[T]`` to the first element; ``b[i]`` and ``buffer_load(b, i)`` read element ``i``.
* ``ByteAddressBuffer`` and ``RWByteAddressBuffer`` become an ``UnsafePointer[UInt32]`` to the buffer's 32-bit
  words. ``buffer_load(b, a)`` reads word ``a >> 2``; ``buffer_load2/3/4`` and ``buffer_store/2/3/4`` read and
  write consecutive words through generated helpers, so the address is evaluated once. HLSL requires the address
  of these operations to be a multiple of four (``Load``/``Store`` of ``ByteAddressBuffer``).
* ``buffer_dimensions(b, n)`` (the HLSL ``GetDimensions`` query; ``buffer_strides`` already wrote any stride
  output as a constant) reads ``b_length``, an extra ``UInt32`` kernel argument written only for a buffer whose
  size the program queries. Like the ``b_length`` CUDA output declares, it holds the element count of a structured
  buffer and the size in bytes of a byte-address buffer.

The element layout must be the one the host wrote. Mojo lays out ``SIMD`` vectors, structures and arrays with their
natural alignment, which for scalars, vectors, structures and arrays is MSL's rule set (``T3`` sized and aligned
like ``T4``; structures aligned to their widest member). A buffer whose ``@buffer_layout`` (``hlsl_structured``,
``std430`` or ``metal``) places any scalar elsewhere, such as an HLSL ``StructuredBuffer<float3>`` with its 12-byte
stride, and an element holding a matrix or an MSL packed vector, are refused instead of read at the wrong offsets.

An append or consume buffer and a structured buffer's hidden counter have no Mojo counterpart, so
``buffer_append``, ``buffer_consume`` and the counter calls are refused, as are a buffer object passed to a function
(the old placeholder ``StructuredBuffer`` types never compiled) and atomics on a byte-address buffer.
"""

from __future__ import annotations

from ..ast import (
    BufferOpNode,
    FunctionCallNode,
    IdentifierNode,
    MemberAccessNode,
    NamedType,
    VariableNode,
)
from ..refusals import MojoRefusal
from ..semantics.structured_layouts import (
    StructuredLayoutUnsupportedError,
    layouts_agree,
)
from .buffer_member_layout import member_is_packed
from .mojo_tables import (
    MOJO_BUFFER_COUNTER_METHODS,
    MOJO_BUFFER_OP_ALIASES,
    MOJO_BYTE_ADDRESS_ATOMIC_METHODS,
    MOJO_MATRIX_TYPES,
)

__all__ = ["MojoBufferObjectMixin", "MOJO_POINTER_BUFFER_TYPES"]

# The canonical buffer object types (mojo_resource_types.canonical_buffer_resource_type) a kernel receives as a
# pointer argument.
MOJO_STRUCTURED_POINTER_BUFFER_TYPES = frozenset(
    {"StructuredBuffer", "RWStructuredBuffer"}
)
MOJO_BYTE_ADDRESS_POINTER_BUFFER_TYPES = frozenset(
    {"ByteAddressBuffer", "RWByteAddressBuffer"}
)
MOJO_POINTER_BUFFER_TYPES = (
    MOJO_STRUCTURED_POINTER_BUFFER_TYPES | MOJO_BYTE_ADDRESS_POINTER_BUFFER_TYPES
)
_WRITABLE_BUFFER_TYPES = frozenset({"RWStructuredBuffer", "RWByteAddressBuffer"})
_COUNTER_BUFFER_TYPES = frozenset({"AppendStructuredBuffer", "ConsumeStructuredBuffer"})
# CrossGL's buffer calls (the HLSL front end writes these for the buffer methods) and the words each moves.
_LOAD_CALLS = {
    "buffer_load": 1,
    "buffer_load2": 2,
    "buffer_load3": 3,
    "buffer_load4": 4,
}
_STORE_CALLS = {
    "buffer_store": 1,
    "buffer_store2": 2,
    "buffer_store3": 3,
    "buffer_store4": 4,
}
# The method spellings of the same operations, for a buffer written as ``b.Load(a)``.
_MEMBER_CALLS = {
    "Load": "buffer_load",
    "Load2": "buffer_load2",
    "Load3": "buffer_load3",
    "Load4": "buffer_load4",
    "Store": "buffer_store",
    "Store2": "buffer_store2",
    "Store3": "buffer_store3",
    "Store4": "buffer_store4",
    "GetDimensions": "buffer_dimensions",
    "IncrementCounter": "buffer_increment_counter",
    "DecrementCounter": "buffer_decrement_counter",
    "Append": "buffer_append",
    "Consume": "buffer_consume",
}
_COUNTER_CALLS = frozenset({"buffer_increment_counter", "buffer_decrement_counter"})
# Append and Consume belong to the append and consume buffers, which a Mojo kernel does not receive.
_QUEUE_METHODS = {"Append": "append", "Consume": "consume"}
_QUEUE_CALLS = {"buffer_append": "append", "buffer_consume": "consume"}
_DIMENSIONS_CALL = "buffer_dimensions"
# A byte-address atomic is spelled atomicOp(buffer, address, value...) in CrossGL.
_ATOMIC_PREFIX = "atomic"
_LENGTH_SUFFIX = "_length"
_NO_COUNTERS = "structured-buffer counters are not available in Mojo, whose kernels receive buffers as plain pointers"
_NO_BYTE_ADDRESS_ATOMICS = "byte-address atomics are not lowered for Mojo"
# @buffer_layout values and the rule names of structured_layouts.
_LAYOUT_RULES = {"hlsl_structured": "hlsl", "std430": "std430", "metal": "metal"}
# Mojo's natural layout of scalars, SIMD vectors, structures and arrays is the MSL rule set of structured_layouts.
_MOJO_LAYOUT_RULES = "metal"
_SKIPPED_ATTRIBUTES = frozenset({"parent", "annotations", "source_location"})


def _call_name(node):
    function = getattr(node, "function", None)
    if isinstance(function, str):
        return function
    if isinstance(function, IdentifierNode):
        return function.name
    if isinstance(function, MemberAccessNode):
        return _MEMBER_CALLS.get(function.member)
    return None


def _call_buffer_operand(node):
    function = getattr(node, "function", None)
    if isinstance(function, MemberAccessNode):
        return function.object_expr
    arguments = getattr(node, "arguments", None) or []
    return arguments[0] if arguments else None


def _walk(node, visited):
    if isinstance(node, (list, tuple)):
        for item in node:
            yield from _walk(item, visited)
        return
    if isinstance(node, dict):
        for item in node.values():
            yield from _walk(item, visited)
        return
    if node is None or not hasattr(node, "__dict__") or id(node) in visited:
        return
    visited.add(id(node))
    yield node
    for attribute, value in vars(node).items():
        if attribute not in _SKIPPED_ATTRIBUTES and not isinstance(value, str):
            yield from _walk(value, visited)


def mojo_buffer_length_queries(ast):
    """The names of the buffers whose size a ``buffer_dimensions`` (or ``GetDimensions``) call of ``ast`` reads."""
    names = set()
    if ast is None:
        return names
    for node in _walk(ast, set()):
        if isinstance(node, FunctionCallNode) and _call_name(node) == _DIMENSIONS_CALL:
            operand = _call_buffer_operand(node)
            if isinstance(operand, IdentifierNode):
                names.add(operand.name)
    return names


def _declared_layout(node):
    for attribute in getattr(node, "attributes", None) or []:
        if str(getattr(attribute, "name", "")).lower() != "buffer_layout":
            continue
        for argument in getattr(attribute, "arguments", None) or []:
            value = getattr(argument, "name", getattr(argument, "value", argument))
            return str(value).strip().strip('"').lower()
    return None


def _element_type_node(node):
    type_node = getattr(node, "var_type", None) or getattr(node, "vtype", None)
    if isinstance(type_node, NamedType) and len(type_node.generic_args) == 1:
        return type_node.generic_args[0]
    return None


class MojoBufferObjectMixin:
    """HLSL buffer-object members of ``MojoCodeGen``."""

    def reset_mojo_buffer_objects(self, ast=None):
        # name -> (canonical buffer type, element type or None for a byte-address buffer)
        self.mojo_buffer_objects = {}
        self.mojo_buffer_length_names = mojo_buffer_length_queries(ast)
        self.mojo_buffer_object_program = ast
        # kernel argument -> the extra arguments every function that reads it also receives
        self.mojo_kernel_argument_companions = {}
        # ("load" | "store", words)
        self.required_mojo_buffer_word_helpers = set()

    # Declarations

    def mojo_buffer_object_argument_type(self, node, variable_type):
        """The pointer type an HLSL buffer object declared outside a function is passed as, or None."""
        info = self.buffer_resource_info(variable_type)
        if info is None:
            return None
        buffer_type, element_type = info
        name = getattr(node, "name", None)
        if buffer_type in _COUNTER_BUFFER_TYPES:
            raise MojoRefusal(
                f"Mojo has no append or consume buffer: {buffer_type} {name} needs a hidden element counter, and "
                f"{_NO_COUNTERS}"
            )
        if buffer_type not in MOJO_POINTER_BUFFER_TYPES:
            return None
        if element_type is None:
            pointee = "uint"
        else:
            pointee = self.type_name(element_type)
            if self.is_array_type_name(pointee):
                raise MojoRefusal(
                    f"Mojo kernel argument {name}: a {buffer_type} of arrays ({pointee}) has no pointer form"
                )
            self.check_mojo_buffer_element_layout(node, buffer_type, pointee)
        self.mojo_buffer_objects[name] = (buffer_type, element_type)
        return f"{pointee}*"

    def check_mojo_buffer_element_layout(self, node, buffer_type, element_type):
        """Refuse a buffer whose declared element layout Mojo's natural layout does not reproduce."""
        layout = _declared_layout(node)
        if layout is None:
            return
        name = getattr(node, "name", None)
        rules = _LAYOUT_RULES.get(layout)
        if rules is None:
            raise MojoRefusal(
                f"Mojo kernel argument {name}: buffer layout {layout} of {buffer_type} has no Mojo counterpart"
            )
        if self.mojo_buffer_element_has_unmodelled_layout(element_type):
            raise MojoRefusal(
                f"Mojo kernel argument {name}: the {layout} layout of a {buffer_type} element holding a matrix "
                f"or a packed vector ({element_type}) is not Mojo's"
            )
        element_node = _element_type_node(node)
        try:
            agree = element_node is not None and layouts_agree(
                self.mojo_buffer_object_program,
                element_node,
                rules,
                _MOJO_LAYOUT_RULES,
                f"Mojo kernel argument {name}",
            )
        except StructuredLayoutUnsupportedError as error:
            raise MojoRefusal(
                f"Mojo kernel argument {name}: the {layout} layout of {buffer_type}<{element_type}> is not "
                f"known ({error})"
            ) from error
        if not agree:
            raise MojoRefusal(
                f"Mojo kernel argument {name}: {buffer_type}<{element_type}> uses the {layout} layout, which "
                "places its elements or members at other offsets than Mojo's natural layout (a 3-component "
                "vector takes 16 bytes and a vector is aligned to its size)"
            )

    def mojo_buffer_element_has_unmodelled_layout(self, type_name, seen=None):
        """Whether ``type_name`` holds a matrix or an MSL packed vector."""
        type_name = self.type_name(type_name)
        if type_name in MOJO_MATRIX_TYPES:
            return True
        if self.is_array_type_name(type_name):
            return self.mojo_buffer_element_has_unmodelled_layout(
                self.parse_array_type_name(type_name)[0], seen
            )
        struct_name, fields = self.struct_type_name_and_fields(type_name)
        if struct_name is None:
            return False
        seen = set() if seen is None else seen
        if struct_name in seen:
            return False
        seen.add(struct_name)
        struct_node = self.mojo_program_struct(struct_name)
        if struct_node is not None and any(
            member_is_packed(member)
            for member in getattr(struct_node, "members", None) or []
        ):
            return True
        return any(
            self.mojo_buffer_element_has_unmodelled_layout(field_type, seen)
            for field_type in fields.values()
        )

    def mojo_program_struct(self, name):
        for node in _walk(
            getattr(self.mojo_buffer_object_program, "structs", None), set()
        ):
            if getattr(node, "name", None) == name and hasattr(node, "members"):
                return node
        return None

    def mojo_buffer_length_argument(self, name):
        """The ``UInt32`` kernel argument holding the size of buffer ``name``, or None when nothing queries it."""
        if name not in getattr(self, "mojo_buffer_objects", {}):
            return None
        if name not in getattr(self, "mojo_buffer_length_names", set()):
            return None
        return f"{name}{_LENGTH_SUFFIX}"

    def declare_mojo_buffer_length_argument(self, name, arguments):
        """Add the size argument of buffer ``name`` beside it, and return its metadata comment."""
        length = self.mojo_buffer_length_argument(name)
        if length is None:
            return ""
        program_names = self.mojo_program_identifier_names()
        if length in program_names or length in arguments:
            raise MojoRefusal(
                f"Mojo kernel argument {length}, which holds the size of buffer {name}, would shadow a name the "
                "program declares"
            )
        arguments.setdefault(length, "uint")
        self.register_variable_type(length, "uint")
        self.mojo_kernel_argument_companions[name] = (length,)
        buffer_type, element_type = self.mojo_buffer_objects[name]
        unit = "bytes" if element_type is None else "elements"
        return f"# CrossGL buffer length: name={length} buffer={name} counts={unit}\n"

    def mojo_program_identifier_names(self):
        names = getattr(self, "_mojo_program_identifier_names", None)
        if names is None:
            names = set()
            for node in _walk(self.mojo_buffer_object_program, set()):
                for attribute in ("name", "identifier"):
                    value = getattr(node, attribute, None)
                    if isinstance(value, str):
                        names.add(value)
            self._mojo_program_identifier_names = names
        return names

    def refuse_mojo_buffer_object_type(self, buffer_type):
        """The refusal for an HLSL buffer object anywhere but a module- or stage-scope declaration."""
        raise MojoRefusal(
            f"Mojo receives an HLSL {buffer_type} only as a kernel argument declared outside a function: a "
            f"{buffer_type} parameter, local, array, structure member or typed buffer has no Mojo spelling"
        )

    # Calls

    def mojo_buffer_object_info(self, expr):
        """``(buffer type, element type)`` of a buffer object kernel argument ``expr`` names, or None."""
        if not isinstance(expr, (IdentifierNode, VariableNode)):
            return None
        info = getattr(self, "mojo_buffer_objects", {}).get(expr.name)
        if info is None:
            return None
        if self.parse_pointer_type_name(str(self.expression_result_type(expr))) is None:
            # A local of the same name hides the kernel argument.
            return None
        return info

    def mojo_buffer_object_call(self, expr):
        """``(CrossGL buffer call name, buffer type, element type)`` of a call or method call on a buffer object
        kernel argument, such as ``buffer_load(b, i)`` or ``b.Load(i)``, or None."""
        if isinstance(expr, BufferOpNode):
            operation = MOJO_BUFFER_OP_ALIASES.get(getattr(expr, "operation", ""))
            func_name = _MEMBER_CALLS.get(operation)
            operand = getattr(expr, "buffer_expr", None)
        elif isinstance(expr, FunctionCallNode):
            func_name = _call_name(expr)
            operand = _call_buffer_operand(expr)
        else:
            return None
        info = self.mojo_buffer_object_info(operand)
        if func_name is None or info is None:
            return None
        return (func_name, *info)

    def mojo_buffer_object_load_type(self, func_name, element_type):
        """The CrossGL type ``func_name`` reads from a buffer object holding ``element_type`` (None: words)."""
        width = _LOAD_CALLS.get(func_name)
        if width is None:
            return None
        if element_type is not None:
            return element_type if width == 1 else None
        return self.byte_address_load_result_type(width)

    def mojo_buffer_object_result_type(self, expr):
        """The CrossGL type a call on a buffer object kernel argument returns, or None for any other call."""
        call = self.mojo_buffer_object_call(expr)
        if call is None:
            return None
        func_name, _buffer_type, element_type = call
        if func_name in _STORE_CALLS or func_name == _DIMENSIONS_CALL:
            return "void"
        return self.mojo_buffer_object_load_type(func_name, element_type)

    def mojo_buffer_object_result_info(self, expr):
        """``(helper name, resource type, value type)`` of a load from a buffer object, for target-shape checks."""
        call = self.mojo_buffer_object_call(expr)
        if call is None:
            return None
        func_name, buffer_type, element_type = call
        value_type = self.mojo_buffer_object_load_type(func_name, element_type)
        if value_type is None:
            return None
        resource_type = buffer_type
        if element_type is not None:
            resource_type = f"{buffer_type}<{self.type_name(element_type)}>"
        return func_name, resource_type, value_type

    def generate_mojo_buffer_object_member_call(self, member, obj_expr, args):
        """A ``b.Load(a)``-style method call on a buffer object kernel argument, or None."""
        info = self.mojo_buffer_object_info(obj_expr)
        if info is None:
            return None
        if member in MOJO_BUFFER_COUNTER_METHODS:
            self.validate_resource_read_write_access(obj_expr, member)
            raise MojoRefusal(
                f"Unsupported Mojo buffer counter method {member} on {info[0]} {obj_expr.name}; "
                f"{_NO_COUNTERS}"
            )
        if member in MOJO_BYTE_ADDRESS_ATOMIC_METHODS and info[1] is None:
            self.validate_resource_read_write_access(obj_expr, member)
            raise MojoRefusal(
                f"Unsupported Mojo byte-address buffer atomic method {member} on {info[0]} {obj_expr.name}; "
                f"{_NO_BYTE_ADDRESS_ATOMICS}"
            )
        if member in _QUEUE_METHODS:
            raise MojoRefusal(
                f"Unsupported {member} of {info[0]} {obj_expr.name} for Mojo codegen; only an "
                f"{_QUEUE_METHODS[member]} buffer has it"
            )
        func_name = _MEMBER_CALLS.get(member)
        if func_name is None:
            return None
        return self.generate_mojo_buffer_object_call(func_name, [obj_expr, *args])

    def generate_mojo_buffer_object_call(self, func_name, args):
        """A CrossGL buffer call on a buffer object kernel argument, or None for any other call."""
        if not args or not isinstance(func_name, str):
            return None
        info = self.mojo_buffer_object_info(args[0])
        if info is None:
            return None
        buffer_type, element_type = info
        name = args[0].name
        if func_name in _QUEUE_CALLS:
            raise MojoRefusal(
                f"Unsupported {func_name} of {buffer_type} {name} for Mojo codegen; only an "
                f"{_QUEUE_CALLS[func_name]} buffer has it"
            )
        if func_name in _COUNTER_CALLS:
            raise MojoRefusal(
                f"Unsupported Mojo buffer counter call {func_name} on {buffer_type} {name}; {_NO_COUNTERS}"
            )
        if func_name in _LOAD_CALLS:
            return self.generate_mojo_buffer_object_load(
                name, buffer_type, element_type, _LOAD_CALLS[func_name], args
            )
        if func_name in _STORE_CALLS:
            return self.generate_mojo_buffer_object_store(
                name, buffer_type, element_type, _STORE_CALLS[func_name], args
            )
        if func_name == _DIMENSIONS_CALL:
            return self.generate_mojo_buffer_object_dimensions(name, buffer_type, args)
        if func_name.startswith(_ATOMIC_PREFIX) and element_type is None:
            raise MojoRefusal(
                f"Unsupported Mojo byte-address buffer atomic call {func_name} on {buffer_type} {name}; "
                f"{_NO_BYTE_ADDRESS_ATOMICS}"
            )
        return None

    def mojo_buffer_word_index(self, address):
        """The word index of byte address ``address``: HLSL byte-address operations take multiples of four."""
        return f"Int({self.generate_expression(address)}) >> 2"

    def generate_mojo_buffer_object_load(
        self, name, buffer_type, element_type, width, args
    ):
        if len(args) != 2:
            raise MojoRefusal(
                f"Unsupported buffer load of {buffer_type} {name} for Mojo codegen; expected one address "
                f"argument, got {len(args) - 1}"
            )
        self.validate_resource_read_access(args[0], "buffer_load")
        pointer = self.mojo_identifier(name)
        if element_type is not None:
            if width != 1:
                raise MojoRefusal(
                    f"Unsupported buffer_load{width} of {buffer_type} {name} for Mojo codegen; only a "
                    "byte-address buffer loads words"
                )
            return f"{pointer}[Int({self.generate_expression(args[1])})]"
        word = self.mojo_buffer_word_index(args[1])
        if width == 1:
            return f"{pointer}[{word}]"
        self.required_mojo_buffer_word_helpers.add(("load", width))
        return f"crosstl_buffer_load{width}({pointer}, {word})"

    def generate_mojo_buffer_object_store(
        self, name, buffer_type, element_type, width, args
    ):
        if buffer_type not in _WRITABLE_BUFFER_TYPES:
            store_name = "buffer_store" if width == 1 else f"buffer_store{width}"
            raise MojoRefusal(
                f"Unsupported {store_name} of read-only {buffer_type} {name} for Mojo codegen"
            )
        if element_type is not None:
            if width != 1 or len(args) != 3:
                raise MojoRefusal(
                    f"Unsupported buffer store of {buffer_type} {name} for Mojo codegen; a structured buffer "
                    f"stores one element at an index, got {width} word(s) and {len(args) - 1} argument(s)"
                )
            # buffer_store(b, i, v) writes element i, like b[i] = v.
            self.validate_resource_write_access(args[0], "buffer_store")
            context = f"buffer_store value of {name}"
            self.validate_expression_target_shape(args[2], element_type, context)
            value = self.generate_expression(args[2], element_type, context)
            index = self.generate_expression(args[1])
            return f"{self.mojo_identifier(name)}[Int({index})] = {value}"
        if len(args) != 3:
            raise MojoRefusal(
                f"Unsupported buffer store of {buffer_type} {name} for Mojo codegen; expected an address and "
                f"a value, got {len(args) - 1} argument(s)"
            )
        self.validate_resource_write_access(args[0], "buffer_store")
        value = self.mojo_buffer_word_value(name, args[2], width)
        word = self.mojo_buffer_word_index(args[1])
        self.required_mojo_buffer_word_helpers.add(("store", width))
        return f"crosstl_buffer_store{width}({self.mojo_identifier(name)}, {word}, {value})"

    def mojo_buffer_word_value(self, name, value, width):
        """``value`` as the ``UInt32`` words a byte-address store writes, converted like HLSL's ``uint`` argument."""
        value_type = self.type_name(self.expression_result_type(value))
        generated = self.generate_expression(value)
        if width == 1:
            if value_type is not None and not self.is_scalar_numeric_type(value_type):
                raise MojoRefusal(
                    f"Unsupported buffer_store to {name} for Mojo codegen; numeric scalar value required, got "
                    f"{value_type}"
                )
            return generated if value_type == "uint" else f"UInt32({generated})"
        vector_info = self.vector_type_info(value_type) if value_type else None
        if (
            vector_info is None
            or vector_info[1] != width
            or vector_info[0] == "DType.bool"
        ):
            raise MojoRefusal(
                f"Unsupported buffer_store{width} to {name} for Mojo codegen; numeric {width}-component vector "
                f"value required, got {value_type}"
            )
        if vector_info[0] == "DType.uint32":
            return generated
        return f"({generated}).cast[DType.uint32]()"

    def generate_mojo_buffer_object_dimensions(self, name, buffer_type, args):
        if len(args) != 2:
            raise MojoRefusal(
                f"Unsupported buffer_dimensions of {buffer_type} {name} for Mojo codegen; expected one count "
                f"output, got {len(args) - 1}"
            )
        length = self.mojo_buffer_length_argument(name)
        if length is None:
            raise MojoRefusal(
                f"Unsupported buffer_dimensions of {buffer_type} {name} for Mojo codegen; the buffer has no "
                "size argument"
            )
        target = args[1]
        target_type = self.type_name(self.expression_result_type(target))
        if target_type is None or not self.is_scalar_integer_type(target_type):
            raise MojoRefusal(
                "Unsupported GetDimensions out parameter for Mojo codegen; integer scalar target required: "
                f"{target_type}"
            )
        self.validate_resource_write_access(target, "GetDimensions")
        target_expr = self.generate_expression(target)
        return f"{target_expr} = {self.map_type(target_type)}({length})"

    def is_scalar_numeric_type(self, type_name):
        return self.is_scalar_integer_type(type_name) or type_name in {
            "float",
            "half",
            "double",
        }

    # Helpers

    def mojo_buffer_object_helper_source(self):
        """The word helpers the byte-address loads and stores of the program call."""
        if not getattr(self, "required_mojo_buffer_word_helpers", None):
            return ""
        code = "# CrossGL byte-address buffer words\n"
        pointer = "UnsafePointer[UInt32, MutAnyOrigin]"
        for kind, width in sorted(self.required_mojo_buffer_word_helpers):
            lanes = 4 if width == 3 else width
            vector = f"SIMD[DType.uint32, {lanes}]"
            if kind == "load":
                words = [
                    f"buffer[word + {lane}]" if lane else "buffer[word]"
                    for lane in range(width)
                ]
                words += ["UInt32(0)"] * (lanes - width)
                code += (
                    f"def crosstl_buffer_load{width}(buffer: {pointer}, word: Int) -> {vector}:\n"
                    f"    return {vector}({', '.join(words)})\n\n"
                )
                continue
            value_type = "UInt32" if width == 1 else vector
            code += f"def crosstl_buffer_store{width}(buffer: {pointer}, word: Int, value: {value_type}):\n"
            if width == 1:
                code += "    buffer[word] = value\n\n"
                continue
            for lane in range(width):
                offset = f"word + {lane}" if lane else "word"
                code += f"    buffer[{offset}] = value[{lane}]\n"
            code += "\n"
        return code
