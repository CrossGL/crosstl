"""DXC's ``[[vk::constant_id(N)]]`` on an HLSL global, imported as a CrossGL specialization constant.

DXC's SPIR-V back end compiles ``[[vk::constant_id(N)]] const int count = 3;`` (with or without ``const``) to
``OpSpecConstant %int 3`` decorated ``SpecId N``: the host may override the value at pipeline creation and
otherwise reads the initializer. Its DXIL back end ignores the attribute (``-Wignored-attributes``) and the
variable becomes a ``$Globals`` member whose initializer DXC drops. The attribute only exists for Vulkan, so the
importer keeps the meaning it names there and writes ``const int count @constant_id(N) = 3;``, the form the GLSL
importer writes for ``layout(constant_id = N)``, which every CrossGL generator reads as a specialization constant with that default (GLSL ``layout(constant_id = N)``, SPIR-V ``SpecId``,
Metal ``[[function_constant(N)]]``, HLSL ``[[vk::constant_id(N)]]``).

DXC accepts the attribute only on an externally visible global of scalar type with a default value; the importer
refuses the cases DXC rejects (a missing initializer, ``static``, a vector, matrix, array or structure) so that no
generator receives a constant without a value or with a shape no target can specialize.
"""

VK_CONSTANT_ID_ATTRIBUTE = "vk::constant_id"

# The scalar types DXC's SPIR-V back end specializes (SpirvEmitter::createSpecConstant).
_SCALAR_TYPES = frozenset(
    {
        "bool",
        "double",
        "dword",
        "float",
        "float16_t",
        "float32_t",
        "float64_t",
        "half",
        "int",
        "int16_t",
        "int32_t",
        "int64_t",
        "min16float",
        "min16int",
        "min16uint",
        "uint",
        "uint16_t",
        "uint32_t",
        "uint64_t",
    }
)


def vk_constant_id_attribute(node):
    """The node's ``[[vk::constant_id]]`` attribute, or ``None``."""
    for attr in getattr(node, "attributes", None) or []:
        if str(getattr(attr, "name", "")) == VK_CONSTANT_ID_ATTRIBUTE:
            return attr
    return None


def has_vk_constant_id(node):
    return vk_constant_id_attribute(node) is not None


def vk_constant_id_suffix(codegen, node):
    """`` @constant_id(N)`` for a ``[[vk::constant_id(N)]]`` global, else ``""``.

    A global DXC's SPIR-V back end rejects is refused through ``codegen.refuse_import``.
    """
    attr = vk_constant_id_attribute(node)
    if attr is None:
        return ""
    name = getattr(node, "name", "<unnamed>")
    arguments = list(
        getattr(attr, "args", None) or getattr(attr, "arguments", None) or []
    )
    index = (
        getattr(arguments[0], "value", arguments[0]) if len(arguments) == 1 else None
    )
    if isinstance(index, str) and index.strip().isdigit():
        index = int(index.strip())
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        codegen.refuse_import(
            f"[[vk::constant_id]] on '{name}' needs one non-negative integer literal"
        )
        return ""
    qualifiers = {str(q).lower() for q in getattr(node, "qualifiers", None) or []}
    if qualifiers & {"static", "groupshared"}:
        codegen.refuse_import(
            f"[[vk::constant_id({index})]] on '{name}': DXC requires a specialization constant to be "
            "externally visible, so it cannot be static or groupshared"
        )
        return ""
    type_name = str(getattr(node, "vtype", "") or "").strip()
    if type_name not in _SCALAR_TYPES or getattr(node, "array_sizes", None):
        codegen.refuse_import(
            f"[[vk::constant_id({index})]] on '{name}' of type '{type_name}': DXC specializes only "
            "global variables of scalar type"
        )
        return ""
    if getattr(node, "value", None) is None:
        codegen.refuse_import(
            f"[[vk::constant_id({index})]] on '{name}': a specialization constant needs a default value"
        )
        return ""
    return f" @constant_id({index})"
