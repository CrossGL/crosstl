"""Three-component byte-address loads and stores in Slang output.

slangc 2026.18.3 lowers ``Load3``, ``Store3``, ``Load<T3>`` and ``Store<T3>`` on a ``ByteAddressBuffer`` with a constant
offset that is a nonzero multiple of the vector's size to an element of a view laid out with the vector's
alignment instead. In SPIR-V output ``Store3(48, v)`` writes bytes 64..75: element ``48 / 12 = 4`` of a ``uint3``
view whose stride is 16. ``Store<uint16_t3>(12, v)`` writes bytes 16..21 the same way. Scalars and two- and
four-component vectors are unaffected, because their strides equal their sizes, and so are offsets that are not
constant.

CrossGL's GLSL buffer blocks put three-component vectors and matrix columns at such offsets (a std430 ``vec3`` at
byte 48, a ``mat3`` column at byte 96), and HLSL sources call ``Load3`` and ``Store3`` directly. The Slang emitter
therefore moves three components through a helper that moves the first two components and the third one
separately. A helper takes the offset and value once, so the access evaluates them once.

:func:`slang_byte_address_vector_call` also lowers CrossGL's ``buffer_load2``..``buffer_load4`` and
``buffer_store2``..``buffer_store4``, which the HLSL importer makes of ``LoadN`` and ``StoreN``; Slang output used to
print them unchanged, and slangc refused the undefined names.
"""

from __future__ import annotations

from ..refusals import SlangRefusal

__all__ = ["SlangThreeComponentAccess", "slang_byte_address_vector_call"]

_VECTOR_LOADS = {"buffer_load2": 2, "buffer_load3": 3, "buffer_load4": 4}
_VECTOR_STORES = {"buffer_store2": 2, "buffer_store3": 3, "buffer_store4": 4}

# Storage scalar -> byte size of one component.
_COMPONENT_SIZES = {
    "uint": 4,
    "int16_t": 2,
    "uint16_t": 2,
    "int64_t": 8,
    "uint64_t": 8,
}


def _load(buffer, storage, components, offset):
    if storage == "uint":
        method = "Load" if components == 1 else f"Load{components}"
        return f"{buffer}.{method}({offset})"
    vector = storage if components == 1 else f"{storage}{components}"
    return f"{buffer}.Load<{vector}>({offset})"


def _store(buffer, storage, components, offset, value):
    if storage == "uint":
        method = "Store" if components == 1 else f"Store{components}"
        return f"{buffer}.{method}({offset}, {value})"
    vector = storage if components == 1 else f"{storage}{components}"
    return f"{buffer}.Store<{vector}>({offset}, {value})"


class SlangThreeComponentAccess:
    """Helper calls that load or store three components of a byte-address buffer, and their declarations."""

    def __init__(self):
        # (kind, storage, buffer type) -> helper name.
        self.helpers = {}

    def _helper(self, kind, storage, buffer_types):
        if storage not in _COMPONENT_SIZES:
            raise SlangRefusal(f"no three-component byte-address helper for {storage}")
        name = f"__crossgl_{kind}3_{storage}"
        for buffer_type in buffer_types:
            self.helpers[(kind, storage, buffer_type)] = name
        return name

    def load(self, buffer, offset, *, storage="uint"):
        """``helper(buffer, offset)``: the three ``storage`` components at byte ``offset``.

        The helper is overloaded for both buffer types, so it applies to any byte-address receiver.
        """
        name = self._helper(
            "load", storage, ("ByteAddressBuffer", "RWByteAddressBuffer")
        )
        return f"{name}({buffer}, {offset})"

    def store(self, buffer, offset, value, *, storage="uint"):
        """``helper(buffer, offset, value)``: store three ``storage`` components at byte ``offset``."""
        name = self._helper("store", storage, ("RWByteAddressBuffer",))
        return f"{name}({buffer}, {offset}, {value})"

    def declarations(self):
        """The helpers the emitted code calls, in a stable order."""
        code = ""
        for (kind, storage, buffer_type), name in sorted(self.helpers.items()):
            size = _COMPONENT_SIZES[storage]
            vector = f"{storage}3"
            third = f"offset + {2 * size}u"
            if kind == "load":
                first = _load("buffer", storage, 2, "offset")
                last = _load("buffer", storage, 1, third)
                code += (
                    f"{vector} {name}({buffer_type} buffer, uint offset)\n{{\n"
                    f"    return {vector}({first}, {last});\n}}\n\n"
                )
            else:
                first = _store("buffer", storage, 2, "offset", "value.xy")
                last = _store("buffer", storage, 1, third, "value.z")
                code += (
                    f"void {name}({buffer_type} buffer, uint offset, {vector} value)\n{{\n"
                    f"    {first};\n    {last};\n}}\n\n"
                )
        return code


def slang_byte_address_vector_call(codegen, func_name, args, statement_context):
    """Slang for CrossGL ``buffer_loadN(buffer, offset)`` or ``buffer_storeN(buffer, offset, value)``, or ``None``."""
    if func_name in _VECTOR_LOADS:
        return _vector_load(codegen, func_name, _VECTOR_LOADS[func_name], args)
    if func_name in _VECTOR_STORES:
        return _vector_store(
            codegen, func_name, _VECTOR_STORES[func_name], args, statement_context
        )
    return None


def _vector_load(codegen, func_name, components, args):
    result_type = f"uint{components}"
    if len(args) < 2:
        return codegen.unsupported_byte_address_buffer_call(
            func_name, "requires buffer and offset arguments", result_type
        )
    buffer_type = codegen.structured_buffer_resource_type(args[0])
    if not codegen.is_byte_address_buffer_resource_type(buffer_type):
        return codegen.unsupported_byte_address_buffer_call(
            func_name, "requires a byte-address buffer resource", result_type
        )
    if codegen.buffer_resource_access(args[0]) == "writeonly":
        return codegen.unsupported_byte_address_buffer_call(
            func_name, "requires readable byte-address buffer resource", result_type
        )
    reason = codegen.byte_address_load_expected_type_unsupported_reason(
        func_name, result_type
    )
    if reason is not None:
        return codegen.unsupported_byte_address_buffer_call(
            func_name,
            reason,
            codegen.byte_address_load_diagnostic_fallback_type(result_type),
        )
    buffer = codegen.generate_expression(args[0])
    offset = codegen.generate_expression(args[1])
    return codegen.slang_byteaddress_word_load(buffer, components, offset)


def _vector_store(codegen, func_name, components, args, statement_context):
    if len(args) < 3:
        return codegen.unsupported_byte_address_buffer_call(
            func_name, "requires buffer, offset and value arguments"
        )
    buffer_type = codegen.structured_buffer_resource_type(args[0])
    if not codegen.is_byte_address_buffer_resource_type(buffer_type):
        return codegen.unsupported_byte_address_buffer_call(
            func_name, "requires a byte-address buffer resource"
        )
    if codegen.buffer_resource_access(args[0]) == "readonly":
        return codegen.unsupported_byte_address_buffer_call(
            func_name, "requires writable byte-address buffer resource"
        )
    if not codegen.is_writable_byte_address_buffer_resource_type(buffer_type):
        return codegen.unsupported_byte_address_buffer_call(
            func_name, "requires RWByteAddressBuffer resource"
        )
    if not statement_context:
        return codegen.unsupported_byte_address_buffer_call(
            func_name, "cannot be used as a value expression", "uint"
        )
    buffer = codegen.generate_expression(args[0])
    offset = codegen.generate_expression(args[1])
    value = codegen.generate_expression(args[2])
    return codegen.slang_byteaddress_word_store(buffer, components, offset, value)
