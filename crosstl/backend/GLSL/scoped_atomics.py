"""Import GL_KHR_memory_scope_semantics atomic calls as CrossGL's plain atomic built-ins.

GL_KHR_memory_scope_semantics adds overloads that take a memory scope, storage semantics and memory semantics after
the usual operands: ``atomicAdd(mem, data, scope, storage, sem)``, ``atomicCompSwap(mem, compare, data, scope,
storageEqual, semEqual, storageUnequal, semUnequal)``, ``atomicLoad(mem, scope, storage, sem)``,
``atomicStore(mem, data, scope, storage, sem)``, and the same trailers on ``imageAtomic*``. The importer passed them
through, so the OpenGL and DirectX generators refused the arity, SPIR-V named ``gl_ScopeDevice`` as an unknown
variable, and Metal dropped the trailing operands without checking them.

CrossGL's atomic built-ins are relaxed and device scoped, as GLSL's plain atomics are (glslang lowers them with
``Scope Device`` and ``Semantics None``). :func:`import_scoped_atomic_call` therefore drops the trailer when every
memory semantics operand is ``gl_SemanticsRelaxed`` and the scope is no wider than ``gl_ScopeDevice``: a wider scope
for a relaxed atomic still provides the atomicity a narrower one asks for, and relaxed storage semantics order
nothing. It raises :class:`ScopedAtomicUnsupportedError` for acquire, release, availability, visibility or volatile
semantics, which CrossGL has no form for, for ``gl_ScopeCrossDevice`` (Vulkan forbids it: "Scope for memory must be
limited to Device, QueueFamily, Workgroup, ShaderCallKHR, Subgroup, or Invocation"), for an operand it cannot
evaluate to a constant, and for ``imageAtomicLoad`` and ``imageAtomicStore``, which CrossGL has no image form of.
"""

from __future__ import annotations

from crosstl.translator.diagnostics import CODE_PREFIX, TranslationDiagnostic

from .OpenglAst import BinaryOpNode, FunctionCallNode, NumberNode, VariableNode

__all__ = ["ScopedAtomicUnsupportedError", "import_scoped_atomic_call"]

# GL_KHR_memory_scope_semantics constants (the extension's "Built-in Constants"), equal to the SPIR-V enumerants.
SCOPE_CONSTANTS = {
    "gl_ScopeCrossDevice": 0,
    "gl_ScopeDevice": 1,
    "gl_ScopeWorkgroup": 2,
    "gl_ScopeSubgroup": 3,
    "gl_ScopeInvocation": 4,
    "gl_ScopeQueueFamily": 5,
    "gl_ScopeShaderCallEXT": 6,
}
STORAGE_SEMANTICS_CONSTANTS = {
    "gl_StorageSemanticsNone": 0x0,
    "gl_StorageSemanticsBuffer": 0x40,
    "gl_StorageSemanticsShared": 0x100,
    "gl_StorageSemanticsImage": 0x800,
    "gl_StorageSemanticsOutput": 0x1000,
}
SEMANTICS_CONSTANTS = {
    "gl_SemanticsRelaxed": 0x0,
    "gl_SemanticsAcquire": 0x2,
    "gl_SemanticsRelease": 0x4,
    "gl_SemanticsAcquireRelease": 0x8,
    "gl_SemanticsMakeAvailable": 0x2000,
    "gl_SemanticsMakeVisible": 0x4000,
    "gl_SemanticsVolatile": 0x8000,
}
_SEMANTICS_NAMES = tuple(sorted(SEMANTICS_CONSTANTS.items(), key=lambda item: item[1]))
_CROSS_DEVICE = SCOPE_CONSTANTS["gl_ScopeCrossDevice"]

_READ_MODIFY_WRITE = (
    "Add",
    "Min",
    "Max",
    "And",
    "Or",
    "Xor",
    "Exchange",
)
# Name -> (operands before the scope, number of semantics operands in the trailer). The trailer is the scope
# followed by one (storage semantics, semantics) pair, or two for compare-swap (equal, then unequal).
_BUFFER_FORMS = {f"atomic{op}": (2, 1) for op in _READ_MODIFY_WRITE}
_BUFFER_FORMS.update(
    {"atomicCompSwap": (3, 2), "atomicLoad": (1, 1), "atomicStore": (2, 1)}
)
# Image forms take a coordinate, and a sample index for multisample images, so the operand count is one of two.
_IMAGE_FORMS = {f"imageAtomic{op}": ((3, 4), 1) for op in _READ_MODIFY_WRITE}
_IMAGE_FORMS["imageAtomicCompSwap"] = ((4, 5), 2)
_IMAGE_WITHOUT_CROSSGL_FORM = {
    "imageAtomicLoad": ((2, 3), 1),
    "imageAtomicStore": ((3, 4), 1),
}


class ScopedAtomicUnsupportedError(TranslationDiagnostic):
    """A GL_KHR_memory_scope_semantics atomic call with no CrossGL form."""

    def __init__(self, name, reason):
        self.function_name = name
        self.missing_capabilities = (f"opengl.import.{name}.memory_semantics",)
        super().__init__(
            f"GLSL {name} with explicit memory scope and semantics cannot be imported: {reason}",
            target="cgl",
            code=f"{CODE_PREFIX}opengl-import-scoped-atomic-unsupported",
        )


def _identifier(node):
    if isinstance(node, VariableNode) and node.value is None and not node.vtype:
        return node.name
    if isinstance(node, str):
        return node
    return None


def _constant(node, constants):
    """The integer value of a scope or semantics operand, or ``None`` when it is not a constant this module reads."""
    name = _identifier(node)
    if name is not None:
        return constants.get(name)
    if isinstance(node, NumberNode):
        text = str(node.value).rstrip("uU")
        try:
            return int(text, 0)
        except ValueError:
            return None
    if isinstance(node, BinaryOpNode) and node.op == "|":
        left = _constant(node.left, constants)
        right = _constant(node.right, constants)
        if left is None or right is None:
            return None
        return left | right
    return None


def _semantics_names(value):
    names = [name for name, bit in _SEMANTICS_NAMES if bit and value & bit]
    unknown = value & ~sum(bit for _, bit in _SEMANTICS_NAMES)
    if unknown:
        names.append(hex(unknown))
    return " | ".join(names)


def _check_trailer(name, trailer, pairs):
    scope = _constant(trailer[0], SCOPE_CONSTANTS)
    if scope is None or scope not in SCOPE_CONSTANTS.values():
        raise ScopedAtomicUnsupportedError(
            name, "the memory scope is not a gl_Scope* constant"
        )
    if scope == _CROSS_DEVICE:
        raise ScopedAtomicUnsupportedError(
            name,
            "gl_ScopeCrossDevice is wider than any target's atomics, and Vulkan forbids it",
        )
    for index in range(pairs):
        storage = _constant(trailer[1 + 2 * index], STORAGE_SEMANTICS_CONSTANTS)
        semantics = _constant(trailer[2 + 2 * index], SEMANTICS_CONSTANTS)
        if storage is None or semantics is None:
            raise ScopedAtomicUnsupportedError(
                name,
                "the storage and memory semantics must be gl_StorageSemantics* and gl_Semantics* constants",
            )
        if semantics:
            raise ScopedAtomicUnsupportedError(
                name,
                f"memory semantics {_semantics_names(semantics)} have no CrossGL form, whose atomics are "
                "relaxed",
            )


def _operand_count(args, counts, pairs):
    trailer = 1 + 2 * pairs
    for count in counts:
        if len(args) == count + trailer:
            return count
    return None


def import_scoped_atomic_call(codegen, name, args):
    """The CrossGL text of a scoped GLSL atomic call, or ``None`` when ``name(args)`` is not one.

    A user function of the same name and arity shadows the built-in.
    """
    args = list(args or [])
    if (name, len(args)) in getattr(codegen, "user_function_arities", ()):
        return None
    if name in _BUFFER_FORMS:
        operands, pairs = _BUFFER_FORMS[name]
        if len(args) != operands + 1 + 2 * pairs:
            return None
    elif name in _IMAGE_FORMS:
        counts, pairs = _IMAGE_FORMS[name]
        operands = _operand_count(args, counts, pairs)
        if operands is None:
            return None
    elif name in _IMAGE_WITHOUT_CROSSGL_FORM:
        counts, pairs = _IMAGE_WITHOUT_CROSSGL_FORM[name]
        if _operand_count(args, counts, pairs) is None:
            return None
        raise ScopedAtomicUnsupportedError(
            name, "CrossGL has no atomic image load or store"
        )
    else:
        return None
    _check_trailer(name, args[operands:], pairs)
    call = FunctionCallNode(VariableNode("", name), args[:operands])
    return codegen.generate_function_call(call)
