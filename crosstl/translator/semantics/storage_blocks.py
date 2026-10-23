"""SPIR-V storage blocks in CrossGL: rewritten to the buffer forms every emitter carries.

SPIR-V declares a storage buffer as one variable whose type is a ``Block``-decorated struct (``StorageBuffer``
storage class, or ``Uniform`` with ``BufferBlock`` before SPIR-V 1.3). The struct's last member may be a runtime
array, whose length the bound buffer's size sets and ``OpArrayLength`` reads (SPIR-V 1.6, section 2.16.2 and
``OpArrayLength``). The SPIR-V importer writes such a variable as a one-element structured buffer of the block
struct: ``RWStructuredBuffer<Block> v``, every access through ``v[0]``, and ``spirvArrayLength(v, k)`` for
``OpArrayLength`` on member ``k``. No target has that form. HLSL structured-buffer elements have a fixed size of at
most 2048 bytes, so the DirectX emitter gave the runtime array 1024 elements and DXC rejected the result, and no
emitter knew ``spirvArrayLength``.

For translations from SPIR-V this pass rewrites each such variable into the forms the GLSL importer produces for
the same declarations (GLSL 4.60 section 4.3.9, interface blocks, and section 4.1.9, ``length()``):

* a block whose only member is a runtime array of a 32-bit scalar, or of a two- or four-component vector of one,
  becomes a structured buffer of that element; those element strides are the same under std430 and in HLSL
  structured buffers. ``v[0].a[i]`` becomes ``v[i]``, ``v[0].a.length()`` becomes ``int(buffer_dimensions(v))``
  and ``spirvArrayLength(v, 0)`` becomes ``uint(buffer_dimensions(v))``;
* any other block becomes a std430 buffer block, ``Block v @glsl_buffer_block(std430)``, keeping its set and
  binding and gaining ``@readonly`` when it was a read-only ``StructuredBuffer``. ``v[0]`` becomes ``v`` and
  ``spirvArrayLength(v, k)`` becomes ``uint(v.m.length())`` for the ``k``-th member ``m``.

The block is laid out by std430, the layout glslang and the Vulkan environment's default use for storage blocks
(Vulkan 1.3, section 15.6.4); the importer does not yet carry the module's ``Offset`` decorations.

An array of storage blocks, ``RWStructuredBuffer<Block> v[N]``, is rewritten the same way element by element:
``v[k][0]`` takes the place of ``v[0]`` and ``spirvArrayLength(v[k], m)`` that of ``spirvArrayLength(v, m)``, and the
variable keeps its array size (``RWStructuredBuffer<T> v[N]`` or ``Block v[N] @glsl_buffer_block(std430)``, the GLSL
importer's forms for ``buffer B { ... } v[N]``).

A variable used other than through ``v[0]`` (``v[k][0]`` for an array) or ``spirvArrayLength``, or whose name a local
declaration shadows, is left as it is. A ``spirvArrayLength`` call that is not rewritten is refused with
:class:`SpirvArrayLengthUnsupportedError`, since no target defines that name.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import call_name, iter_functions, iter_structs, type_from_node
from .rewrite import call, child_slots, clone, get_child, set_child

__all__ = [
    "SPIRV_SOURCES",
    "SpirvArrayLengthUnsupportedError",
    "normalize_spirv_storage_blocks",
]

SPIRV_SOURCES = frozenset({"vulkan", "spirv"})
ARRAY_LENGTH = "spirvArrayLength"
_STRUCTURED = {"StructuredBuffer": True, "RWStructuredBuffer": False}


class SpirvArrayLengthUnsupportedError(TranslationDiagnostic):
    """An ``OpArrayLength`` whose buffer could not be rewritten into a form the target can query."""

    def __init__(self, target, detail, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.spirv-array-length",)
        super().__init__(
            f"OpArrayLength {detail}; it is only carried for a storage block variable accessed through "
            "its single block instance",
            target=target,
            code=f"{CODE_PREFIX}{target}-spirv-array-length-unsupported",
        )


def _is_runtime_array(type_node):
    return isinstance(type_node, A.ArrayType) and type_node.size is None


def _literal_int(node):
    value = getattr(node, "value", node) if isinstance(node, A.LiteralNode) else node
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    try:
        return int(str(value).rstrip("uU"), 0)
    except ValueError:
        return None


def _flat_element(type_node) -> bool:
    """Whether a runtime array of this element has the same stride in std430 and in an HLSL structured buffer."""
    value = type_from_node(type_node)
    return (
        value is not None
        and value.kind in ("i", "u", "f")
        and value.bits == 32
        and value.lanes in (1, 2, 4)
    )


def _parents(program):
    parents = {}
    stack = [program]
    while stack:
        node = stack.pop()
        for field, key in child_slots(node):
            child = get_child(node, field, key)
            if id(child) not in parents:
                parents[id(child)] = (node, field, key)
                stack.append(child)
    return parents


def _replace(parents, old, new):
    owner, field, key = parents[id(old)]
    set_child(owner, field, key, new)
    parents[id(new)] = (owner, field, key)


def _walk(node):
    """Every AST node under ``node`` (a node or a list of them), each once."""
    stack, seen = [node], set()
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, A.ASTNode) and id(item) not in seen:
            seen.add(id(item))
            yield item
            stack.extend(get_child(item, f, k) for f, k in child_slots(item))


def _declared_names(program):
    """Names that parameters and local variables declare, which would shadow a global of the same name."""
    names = set()
    for function in iter_functions(program):
        names.update(getattr(p, "name", None) for p in function.parameters or [])
        names.update(
            n.name for n in _walk(function.body) if isinstance(n, A.VariableNode)
        )
    return names


class _Block:
    """One SPIR-V storage block variable and every use found of it."""

    def __init__(self, variable, struct, readonly, array=None):
        self.variable = variable
        self.struct = struct
        self.readonly = readonly
        self.array = array  # the ArrayType of an array of blocks, else None
        self.instances = []  # the ``v[0]`` (``v[k][0]``) nodes
        self.lengths = (
            []
        )  # the ``spirvArrayLength(v, k)`` calls, with their member index
        self.other = False

    @property
    def name(self):
        return self.variable.name

    def declared_type(self, element):
        """``element`` as the variable's type, keeping the array size of an array of blocks."""
        if self.array is None:
            return element
        return A.ArrayType(element, self.array.size)

    def flat_member(self):
        members = self.struct.members
        if len(members) != 1 or not _is_runtime_array(members[0].member_type):
            return None
        member = members[0]
        return member if _flat_element(member.member_type.element_type) else None


class StorageBlockNormalizer:
    def __init__(self, program, target):
        self.program = program
        self.target = target
        self.parents = _parents(program)
        self.structs = {s.name: s for s in iter_structs(program)}

    def run(self) -> bool:
        blocks = self.find_blocks()
        if blocks:
            self.collect_uses(blocks)
        changed = False
        for block in blocks.values():
            if block.other or not (block.instances or block.lengths):
                continue
            member = block.flat_member()
            if member is not None and self.flat_uses(block, member):
                self.flatten(block, member)
            else:
                self.to_buffer_block(block)
            changed = True
        self.refuse_remaining_lengths()
        return changed

    def find_blocks(self):
        shadowed = _declared_names(self.program)
        blocks = {}
        for variable in getattr(self.program, "global_variables", None) or []:
            var_type = getattr(variable, "var_type", None)
            array = None
            if isinstance(var_type, A.ArrayType) and isinstance(
                var_type.element_type, A.NamedType
            ):
                array, var_type = var_type, var_type.element_type
            if not (
                isinstance(variable, A.VariableNode)
                and isinstance(var_type, A.NamedType)
                and var_type.name in _STRUCTURED
                and len(var_type.generic_args) == 1
                and isinstance(var_type.generic_args[0], A.NamedType)
                and variable.name not in shadowed
            ):
                continue
            struct = self.structs.get(var_type.generic_args[0].name)
            if isinstance(struct, A.StructNode) and struct.members:
                blocks[variable.name] = _Block(
                    variable, struct, _STRUCTURED[var_type.name], array
                )
        return blocks

    def collect_uses(self, blocks):
        for function in iter_functions(self.program):
            for node in _walk(function.body):
                if isinstance(node, A.IdentifierNode) and node.name in blocks:
                    self.classify(blocks[node.name], node)

    def classify(self, block, identifier):
        receiver = identifier
        if block.array is not None:
            # ``v[k]``, one block of the array, takes the place of ``v``.
            receiver = self.parent(identifier)
            if not (
                isinstance(receiver, A.ArrayAccessNode)
                and receiver.array_expr is identifier
            ):
                block.other = True
                return
        owner = self.parent(receiver)
        if (
            isinstance(owner, A.ArrayAccessNode)
            and owner.array_expr is receiver
            and _literal_int(owner.index_expr) == 0
        ):
            block.instances.append(owner)
            return
        if (
            isinstance(owner, A.FunctionCallNode)
            and call_name(owner) == ARRAY_LENGTH
            and len(owner.arguments) == 2
            and owner.arguments[0] is receiver
        ):
            index = _literal_int(owner.arguments[1])
            if index is not None and 0 <= index < len(block.struct.members):
                if _is_runtime_array(block.struct.members[index].member_type):
                    block.lengths.append((owner, index))
                    return
        block.other = True

    def parent(self, node):
        return self.parents.get(id(node), (None, None, None))[0]

    def flat_uses(self, block, member):
        """Whether every ``v[0]`` use is ``v[0].a[i]`` or ``v[0].a.length()``."""
        for instance in block.instances:
            access = self.parent(instance)
            if not (
                isinstance(access, A.MemberAccessNode) and access.member == member.name
            ):
                return False
            use = self.parent(access)
            if isinstance(use, A.ArrayAccessNode) and use.array_expr is access:
                continue
            if self.length_call(access) is None:
                return False
        return True

    def length_call(self, access):
        method = self.parent(access)
        if not (isinstance(method, A.MemberAccessNode) and method.member == "length"):
            return None
        invocation = self.parent(method)
        if (
            isinstance(invocation, A.FunctionCallNode)
            and invocation.function is method
            and not invocation.arguments
        ):
            return invocation
        return None

    def flatten(self, block, member):
        variable = block.variable
        element = clone(member.member_type.element_type)
        wrapper = (
            variable.var_type.element_type if block.array else variable.var_type
        ).name
        flat_type = block.declared_type(A.NamedType(wrapper, [element]))
        variable.var_type = variable.vtype = flat_type
        for instance in block.instances:
            access = self.parent(instance)
            use = self.parent(access)
            receiver = self.receiver(block, instance.array_expr)
            if isinstance(use, A.ArrayAccessNode) and use.array_expr is access:
                index = use.index_expr
                element = A.ArrayAccessNode(receiver, index)
                _replace(self.parents, use, element)
                self.parents[id(receiver)] = (element, "array_expr", None)
                self.parents[id(index)] = (element, "index_expr", None)
            else:
                dimensions = call("buffer_dimensions", [receiver])
                _replace(
                    self.parents, self.length_call(access), call("int", [dimensions])
                )
        for length, _ in block.lengths:
            receiver = self.receiver(block, length.arguments[0])
            dimensions = call("buffer_dimensions", [receiver])
            _replace(self.parents, length, call("uint", [dimensions]))
        self.drop_unused_struct(block.struct)

    def receiver(self, block, node):
        """The buffer ``node`` names: a fresh ``v``, or the ``v[k]`` node itself for an array of blocks."""
        return node if block.array is not None else A.IdentifierNode(block.name)

    def to_buffer_block(self, block):
        variable = block.variable
        block_type = block.declared_type(A.NamedType(block.struct.name))
        variable.var_type = variable.vtype = block_type
        attributes = [
            A.AttributeNode("glsl_buffer_block", [A.IdentifierNode("std430")]),
            *(variable.attributes or []),
        ]
        if block.readonly:
            attributes.append(A.AttributeNode("readonly"))
        variable.attributes = attributes
        for instance in block.instances:
            _replace(self.parents, instance, self.receiver(block, instance.array_expr))
        for length, index in block.lengths:
            member = block.struct.members[index].name
            receiver = self.receiver(block, length.arguments[0])
            method = A.MemberAccessNode(A.MemberAccessNode(receiver, member), "length")
            _replace(
                self.parents, length, call("uint", [A.FunctionCallNode(method, [])])
            )

    def drop_unused_struct(self, struct):
        """Remove a block struct that no declaration names any more, so emitters do not declare it."""
        if self.mentions_struct(struct.name):
            return
        for owner in [self.program, *self.stages()]:
            for field in ("structs", "local_structs"):
                items = getattr(owner, field, None)
                if isinstance(items, list) and struct in items:
                    items.remove(struct)

    def stages(self):
        stages = getattr(self.program, "stages", None) or {}
        return list(stages.values()) if isinstance(stages, dict) else list(stages)

    def mentions_struct(self, name):
        def names(type_node):
            if isinstance(type_node, A.NamedType):
                yield type_node.name
                for argument in type_node.generic_args:
                    yield from names(argument)
            elif isinstance(type_node, A.ArrayType):
                yield from names(type_node.element_type)
            elif isinstance(type_node, (A.PointerType, A.ReferenceType)):
                yield from names(
                    getattr(type_node, "pointee_type", None)
                    or getattr(type_node, "referenced_type", None)
                )
            elif isinstance(type_node, str):
                yield type_node

        stack = [self.program]
        seen = set()
        while stack:
            node = stack.pop()
            if not isinstance(node, A.ASTNode) or id(node) in seen:
                continue
            seen.add(id(node))
            for field in ("var_type", "param_type", "return_type", "member_type"):
                if name in names(getattr(node, field, None)):
                    return True
            if isinstance(node, A.IdentifierNode) and node.name == name:
                return True
            stack.extend(get_child(node, f, k) for f, k in child_slots(node))
            for field in ("structs", "local_structs", "global_variables"):
                stack.extend(getattr(node, field, None) or [])
        return False

    def refuse_remaining_lengths(self):
        for function in iter_functions(self.program):
            for node in _walk(function.body):
                if (
                    isinstance(node, A.FunctionCallNode)
                    and call_name(node) == ARRAY_LENGTH
                ):
                    raise SpirvArrayLengthUnsupportedError(
                        self.target,
                        f"in '{function.name}' reads a buffer that is not a storage block variable",
                        source_location=getattr(node, "source_location", None),
                    )


def normalize_spirv_storage_blocks(program, source, target) -> bool:
    """Rewrite ``program``'s SPIR-V storage block variables for ``target``; returns whether any changed."""
    if str(source or "").lower() not in SPIRV_SOURCES:
        return False
    return StorageBlockNormalizer(program, str(target or "")).run()
