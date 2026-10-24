"""Unsized arrays in CrossGL: storage buffers keep them, every other declaration is refused.

An array declared without an element count only has a size the target can know in two places:

* a storage buffer, whose bound size sets the element count of its last member (GLSL 4.60 section 4.1.9, SPIR-V
  ``OpTypeRuntimeArray`` in a ``Block``, WGSL runtime-sized arrays, HLSL structured buffers, MSL buffer pointers);
* an array of resources, which binds a descriptor range at run time.

The hub writes the first as ``buffer T name[]`` (with an optional ``layout(std430|std140|scalar, ...)``), as a
buffer block (``Block v @glsl_buffer_block(std430)`` whose struct ends in ``T m[]``) or as a structured buffer.
The emitters disagreed on ``buffer T name[]``: Metal used a buffer pointer and WGSL a storage array, but HLSL
declared a ``static`` private array of 1024 elements, GLSL a ``uniform`` array, and SPIR-V a runtime array of
structs outside any block. This pass rewrites the declaration, for every target, into the forms all emitters carry:

* a runtime array of a 32-bit scalar, or of a two- or four-component vector of one, in std430 or scalar layout
  (or none, which the hub reads as std430, the Vulkan default for storage buffers) becomes
  ``RWStructuredBuffer<T> name``, or ``StructuredBuffer<T>`` when ``readonly``. Those strides are the same in
  std430, HLSL structured buffers and MSL. ``name.length()`` becomes ``int(buffer_dimensions(name))``;
* any other element (three-component vectors, matrices, other widths) or an std140 layout becomes a
  buffer block ``{Name}Block {name}Storage @glsl_buffer_block(layout)`` holding ``T name[]``, which the emitters
  lay out member by member. ``name[i]`` becomes ``{name}Storage.name[i]``, ``buffer_load(name, i)`` an element
  read, ``buffer_store(name, i, v)`` an assignment statement and ``buffer_dimensions(name)`` a ``length()``.

``buffer S name[]`` with a struct ``S`` is one storage buffer of ``S`` elements, as Metal's ``device S*`` always
read it; GLSL had printed an unsized array of buffer blocks, which glslang rejects, SPIR-V a runtime descriptor
array and HLSL a refusal. A struct of 32-bit scalars and two- and four-component vectors without std430 padding
takes the structured-buffer form, since its offsets agree in std430, HLSL and MSL; any other struct takes the
buffer block form. GLSL's array of buffer blocks, ``buffer B { ... } name[N];``, which the parser marks with
``@glsl_buffer_block``, keeps its meaning of one binding per element and is left alone.

``buffer T name[N]`` is likewise one storage buffer holding ``N`` elements; HLSL had declared a ``static``
private array, so stores never reached memory, GLSL a ``uniform`` array, and struct elements gave ``N`` blocks.
It takes the same two forms. A flat element drops the count from the type, and ``name.length()`` and
``buffer_dimensions(name)`` become ``N``, the count the source declared; a block keeps ``T name[N]``.

Stage interface arrays that GLSL sizes implicitly (geometry inputs, tessellation control outputs, mesh outputs,
``pervertex`` inputs, ``gl_ClipDistance`` and ``gl_CullDistance``) first get the count their stage implies, and
tessellation inputs, which the pipeline sizes, are kept unsized (:mod:`.stage_io_arrays`).

Every other unsized array of values has no size any target can give it, and is refused with
:class:`UnsizedArrayUnsupportedError`: a global, stage or local variable without an initializer, a parameter, an
unsized struct member other than the last, and a struct ending in an unsized member that is used other than as a
storage buffer. HLSL allows unsized arrays only for resources; MSL is C++14, whose array bounds are constant
expressions; GLSL and WGSL allow runtime-sized arrays only as the last member of a storage block. The emitters had
given such arrays 1024 elements or printed them unchanged, so the result was either silently different or rejected
by the target compiler.
"""

from __future__ import annotations

import re

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import call_name, iter_functions, iter_structs, type_from_node
from .rewrite import call, clone
from .stage_io_arrays import size_stage_io_arrays
from .storage_blocks import _flat_element, _parents, _replace, _walk

__all__ = ["UnsizedArrayUnsupportedError", "legalize_unsized_arrays"]

_LAYOUTS = ("std430", "std140", "scalar")
_ACCESS = frozenset({"buffer", "readonly", "writeonly"})
_STORAGE_WRAPPERS = frozenset(
    {"StructuredBuffer", "RWStructuredBuffer", "RasterizerOrderedStructuredBuffer"}
)
_ARRAY_CONSTRUCTOR = re.compile(r"\w+\s*\[\s*\d*\s*\]")
_MATRIX = re.compile(
    r"(?:d?mat[234](?:x[234])?|(?:float|half|double|int|uint)[234]x[234])"
)


class UnsizedArrayUnsupportedError(TranslationDiagnostic):
    """An array without an element count outside a storage buffer, which no target can size."""

    def __init__(self, target, name, where, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.unsized-value-array",)
        super().__init__(
            f"unsized array '{name}' {where}: only a storage buffer's last member or an array of resources may "
            "omit its element count (HLSL allows unsized arrays only for resources, MSL array bounds are "
            "constant expressions, and GLSL and WGSL allow runtime-sized arrays only as the last member of a "
            "storage block); give the array a size or declare it as 'buffer T name[]'",
            target=target,
            code=f"{CODE_PREFIX}{target}-unsized-array-unsupported",
        )


_LAYOUT_MEMBER_ATTRIBUTES = frozenset({"offset", "align", "packoffset", "packed"})


def _flat_struct(struct) -> bool:
    """Whether a struct of 32-bit scalars and two- and four-component vectors has no padding in std430.

    Such a struct has the same member offsets and stride in std430, the scalar layout, an HLSL structured buffer
    (which packs to 4 bytes) and MSL (whose alignments for these types match std430), so a runtime array of it can
    be a structured buffer. Any padding, or a member that is not such a vector, leaves it to the buffer block.
    """
    members = list(getattr(struct, "members", None) or [])
    offset, alignment = 0, 4
    for member in members:
        member_type = getattr(member, "member_type", None)
        if not _flat_element(member_type) or (
            _attribute_names(member) & _LAYOUT_MEMBER_ATTRIBUTES
        ):
            return False
        size = 4 * type_from_node(member_type).lanes
        if offset % size:
            return False
        offset += size
        alignment = max(alignment, size)
    return bool(members) and offset % alignment == 0


def _unsized(type_node):
    return isinstance(type_node, A.ArrayType) and type_node.size is None


def _qualifiers(node):
    return {str(q).lower() for q in getattr(node, "qualifiers", None) or []}


def _attribute_names(node):
    return {
        str(getattr(a, "name", "")).lower()
        for a in getattr(node, "attributes", None) or []
    }


def _is_storage(node):
    """Whether a declaration is a storage buffer: ``buffer``-qualified or a ``@glsl_buffer_block``."""
    return "buffer" in _qualifiers(node) or "glsl_buffer_block" in _attribute_names(
        node
    )


# A parameter with a resource binding, or any parameter of a stage entry point, is bound by the host: kernel
# importers spell CUDA, HIP and OpenCL pointer parameters as ``@group(0) @binding(n) x: array<T>``.
_BINDING_ATTRIBUTES = frozenset(
    {"binding", "group", "set", "buffer", "register", "space"}
)
_ENTRY_ATTRIBUTES = frozenset(
    {
        "compute",
        "vertex",
        "fragment",
        "kernel",
        "stage_entry",
        "mesh",
        "task",
        "amplification",
    }
)


def _is_bound_parameter(function, parameter):
    return bool(
        _attribute_names(parameter) & _BINDING_ATTRIBUTES
        or _attribute_names(function) & _ENTRY_ATTRIBUTES
    )


def _type_names(type_node):
    """Every type name a type mentions, its generic arguments and elements included."""
    if isinstance(type_node, A.NamedType):
        yield type_node.name
        for argument in type_node.generic_args:
            yield from _type_names(argument)
    elif isinstance(type_node, (A.ArrayType, A.VectorType, A.MatrixType)):
        yield from _type_names(type_node.element_type)
    elif isinstance(type_node, A.PointerType):
        yield from _type_names(type_node.pointee_type)
    elif isinstance(type_node, A.ReferenceType):
        yield from _type_names(type_node.referenced_type)
    elif isinstance(type_node, str):
        yield type_node.split("[", 1)[0].strip()


class _Legalizer:
    def __init__(self, program, target):
        self.program = program
        self.target = target
        self.structs = {
            s.name: s for s in iter_structs(program) if isinstance(s, A.StructNode)
        }
        self.parents = None
        self.taken = None
        self.pipeline_sized = set()

    def is_value(self, type_node) -> bool:
        """Whether a type is a value (scalar, vector, matrix, struct or array of them), not a resource."""
        if isinstance(type_node, A.ArrayType):
            return self.is_value(type_node.element_type)
        if isinstance(type_node, (A.VectorType, A.MatrixType)):
            return True
        if isinstance(type_node, A.PrimitiveType):
            return type_from_node(type_node) is not None or bool(
                _MATRIX.fullmatch(type_node.name)
            )
        if isinstance(type_node, A.NamedType) and not type_node.generic_args:
            name = type_node.name
            return (
                type_from_node(type_node) is not None
                or bool(_MATRIX.fullmatch(name))
                or isinstance(self.structs.get(name), A.StructNode)
            )
        return False

    def is_struct(self, type_node):
        return isinstance(type_node, A.NamedType) and type_node.name in self.structs

    def refuse(self, name, where, node):
        raise UnsizedArrayUnsupportedError(
            self.target,
            name,
            where,
            source_location=getattr(node, "source_location", None),
        )

    def scopes(self):
        """``(owner, field)`` for every list of module- and stage-scope variables."""
        yield self.program, "global_variables"
        stages = getattr(self.program, "stages", None) or {}
        for stage in stages.values() if isinstance(stages, dict) else stages:
            yield stage, "local_variables"

    def run(self):
        self.pipeline_sized = size_stage_io_arrays(self.program)
        buffers = [
            (owner, field, variable)
            for owner, field in self.scopes()
            for variable in list(getattr(owner, field, None) or [])
            if isinstance(variable, A.VariableNode)
            and "buffer" in _qualifiers(variable)
            and isinstance(variable.var_type, A.ArrayType)
            and self.is_value(variable.var_type.element_type)
            and "glsl_buffer_block" not in _attribute_names(variable)
        ]
        if buffers:
            self.parents = _parents(self.program)
            self.taken = self.declared_names()
            for owner, field, variable in buffers:
                self.legalize_buffer(owner, field, variable)
        self.refuse_unsized_values()

    # Storage buffers declared as ``buffer T name[]``.

    def declared_names(self):
        names = set(self.structs)
        for owner, field in self.scopes():
            names.update(
                getattr(v, "name", None) for v in getattr(owner, field, None) or []
            )
        for function in iter_functions(self.program):
            names.add(function.name)
            names.update(getattr(p, "name", None) for p in function.parameters or [])
            names.update(
                n.name for n in _walk(function.body) if isinstance(n, A.VariableNode)
            )
        return names

    def fresh(self, base):
        name, index = base, 1
        while name in self.taken:
            index += 1
            name = f"{base}{index}"
        self.taken.add(name)
        return name

    def uses(self, name):
        """Identifiers naming the buffer in function bodies, and whether a function that declares the same name
        (so its identifiers may mean that declaration) mentions it."""
        found, ambiguous = [], False
        for function in iter_functions(self.program):
            identifiers = []
            shadowed = any(
                getattr(p, "name", None) == name for p in function.parameters or []
            )
            for node in _walk(function.body):
                if isinstance(node, A.VariableNode) and node.name == name:
                    shadowed = True
                elif isinstance(node, A.IdentifierNode) and node.name == name:
                    identifiers.append(node)
            if shadowed:
                ambiguous = ambiguous or bool(identifiers)
            else:
                found.extend(identifiers)
        return found, ambiguous

    def parent(self, node):
        entry = self.parents.get(id(node))
        return entry[0] if entry else None

    def legalize_buffer(self, owner, field, variable):
        layouts = [
            a for a in variable.attributes or [] if str(a.name).lower() in _LAYOUTS
        ]
        layout = str(layouts[0].name).lower() if layouts else "std430"
        readonly = "readonly" in _qualifiers(variable)
        element = variable.var_type.element_type
        size = variable.var_type.size
        uses, ambiguous = self.uses(variable.name)
        flat = layout != "std140" and (
            _flat_element(element)
            or (self.is_struct(element) and _flat_struct(self.structs[element.name]))
        )
        if ambiguous and not flat:
            self.refuse(
                variable.name,
                "is a storage buffer used where a local declaration shadows it",
                variable,
            )
        variable.qualifiers = [
            q for q in variable.qualifiers if str(q).lower() not in _ACCESS
        ]
        variable.attributes = [a for a in variable.attributes or [] if a not in layouts]
        if flat:
            wrapper = "StructuredBuffer" if readonly else "RWStructuredBuffer"
            variable.var_type = variable.vtype = A.NamedType(wrapper, [element])
            for identifier in uses:
                self.rewrite_flat_count(identifier, variable.name, size)
            return
        self.to_block(owner, field, variable, element, size, layout, readonly, uses)

    def rewrite_flat_count(self, identifier, name, size):
        """``name.length()`` of a structured buffer becomes its bound element count, or the declared count of
        ``buffer T name[N]``, which ``buffer_dimensions(name)`` also gives; the declared count is what the source
        array had, whatever size of buffer the host binds."""
        length = self.length_call(identifier)
        if length is not None:
            count = (
                clone(size)
                if size is not None
                else call("buffer_dimensions", [A.IdentifierNode(name)])
            )
            _replace(self.parents, length, call("int", [count]))
            return
        user = self.parent(identifier)
        if (
            size is not None
            and isinstance(user, A.FunctionCallNode)
            and call_name(user) == "buffer_dimensions"
            and len(user.arguments or []) == 1
            and user.arguments[0] is identifier
        ):
            _replace(self.parents, user, call("uint", [clone(size)]))

    def length_call(self, identifier):
        """The ``name.length()`` call around ``identifier``, if that is how it is used."""
        access = self.parent(identifier)
        if isinstance(access, A.MemberAccessNode) and access.member == "length":
            length = self.parent(access)
            if isinstance(length, A.FunctionCallNode) and length.function is access:
                return length
        return None

    def to_block(self, owner, field, variable, element, size, layout, readonly, uses):
        name = variable.name
        struct_name = self.fresh(f"{name[:1].upper()}{name[1:]}Block")
        instance = self.fresh(f"{name}Storage")
        count = clone(size) if size is not None else None
        member = A.StructMemberNode(name, A.ArrayType(clone(element), count))
        self.program.structs.append(A.StructNode(struct_name, [member]))
        self.structs[struct_name] = self.program.structs[-1]
        attributes = [A.AttributeNode("glsl_buffer_block", [A.IdentifierNode(layout)])]
        attributes += variable.attributes
        if readonly:
            attributes.append(A.AttributeNode("readonly"))
        block = A.VariableNode(
            instance,
            A.NamedType(struct_name),
            attributes=attributes,
            qualifiers=list(variable.qualifiers),
            source_location=getattr(variable, "source_location", None),
        )
        variables = getattr(owner, field)
        variables[variables.index(variable)] = block
        for identifier in uses:
            self.rewrite_block_use(identifier, instance, name)

    def rewrite_block_use(self, identifier, instance, name):
        array = A.MemberAccessNode(A.IdentifierNode(instance), name)
        user = self.parent(identifier)
        builtin = call_name(user) if isinstance(user, A.FunctionCallNode) else None
        arguments = getattr(user, "arguments", None) or []
        if builtin and arguments and arguments[0] is identifier:
            if builtin == "buffer_load" and len(arguments) == 2:
                _replace(self.parents, user, A.ArrayAccessNode(array, arguments[1]))
                return
            if builtin == "buffer_dimensions" and len(arguments) == 1:
                length = A.FunctionCallNode(A.MemberAccessNode(array, "length"), [])
                _replace(self.parents, user, call("uint", [length]))
                return
            statement = self.parent(user)
            if (
                builtin == "buffer_store"
                and len(arguments) == 3
                and isinstance(statement, A.ExpressionStatementNode)
                and statement.expression is user
            ):
                target = A.ArrayAccessNode(array, arguments[1])
                _replace(
                    self.parents, statement, A.AssignmentNode(target, arguments[2])
                )
                return
        if isinstance(user, (A.ArrayAccessNode, A.MemberAccessNode)) and (
            getattr(user, "array_expr", None) is identifier
            or getattr(user, "object_expr", None) is identifier
        ):
            _replace(self.parents, identifier, array)
            return
        self.refuse(
            name,
            "is a storage buffer of non-flat elements used other than by indexing, length(), buffer_load, "
            "buffer_store or buffer_dimensions",
            identifier,
        )

    # Every other unsized array of values.

    def refuse_unsized_values(self):
        for owner, field in self.scopes():
            for variable in getattr(owner, field, None) or []:
                self.check_declaration(variable, "is declared at module or stage scope")
        for function in iter_functions(self.program):
            for parameter in function.parameters or []:
                if not (
                    _is_storage(parameter) or _is_bound_parameter(function, parameter)
                ):
                    self.check_type(
                        parameter,
                        getattr(parameter, "param_type", None),
                        "is a parameter",
                    )
            for node in _walk(function.body):
                if isinstance(node, A.VariableNode):
                    self.check_declaration(
                        node, f"is a local variable of '{function.name}'"
                    )
        self.check_structs()

    def check_declaration(self, variable, where):
        if not isinstance(variable, A.VariableNode):
            return
        if getattr(variable, "initial_value", None) is not None:
            self.size_from_initializer(variable)
            return
        if _is_storage(variable) or id(variable) in self.pipeline_sized:
            return
        if isinstance(variable, A.ArrayNode) and variable.size is None:
            element = getattr(variable, "element_type", None)
            if self.is_value(
                element if not isinstance(element, str) else A.NamedType(element)
            ):
                self.refuse(variable.name, where, variable)
            return
        self.check_type(variable, getattr(variable, "var_type", None), where)

    def size_from_initializer(self, variable):
        """Give ``T a[] = {x, y}`` and ``T a[] = T[](x, y)`` the element count of their initializer, as C and
        GLSL 4.60 section 4.1.9 do, so that every target declares the same array. A GLSL array constructor that
        initializes an array declaration becomes the equivalent initializer list, which every emitter renders.
        """
        var_type = getattr(variable, "var_type", None)
        if not isinstance(var_type, A.ArrayType) or isinstance(variable, A.ArrayNode):
            return
        value = variable.initial_value
        if isinstance(value, A.FunctionCallNode) and _ARRAY_CONSTRUCTOR.fullmatch(
            call_name(value)
        ):
            value = A.ArrayLiteralNode(list(value.arguments or []))
            variable.initial_value = value
        if not isinstance(value, A.ArrayLiteralNode) or var_type.size is not None:
            return
        count = len(value.elements or [])
        if count:
            sized = A.ArrayType(
                var_type.element_type, A.LiteralNode(count, A.PrimitiveType("int"))
            )
            variable.var_type = variable.vtype = sized

    def check_type(self, node, type_node, where):
        current = type_node
        while isinstance(current, A.ArrayType):
            if current.size is None and self.is_value(current.element_type):
                self.refuse(getattr(node, "name", "?"), where, node)
            current = current.element_type

    def check_structs(self):
        unsized_last = {}
        for struct in self.structs.values():
            members = [
                m for m in struct.members or [] if isinstance(m, A.StructMemberNode)
            ]
            for index, member in enumerate(members):
                member_type = member.member_type
                if not (
                    _unsized(member_type) and self.is_value(member_type.element_type)
                ):
                    continue
                if index != len(members) - 1:
                    self.refuse(
                        f"{struct.name}.{member.name}",
                        "is not the last member of its struct",
                        member,
                    )
                unsized_last[struct.name] = member
        if unsized_last:
            for name, (node, use) in self.non_storage_uses(set(unsized_last)).items():
                member = unsized_last[name]
                self.refuse(f"{name}.{member.name}", f"ends a struct that {use}", node)

    def non_storage_uses(self, names):
        """For each struct in ``names``, the first declaration that uses it other than as a storage buffer."""
        found = {}

        def note(node, type_node, use, allowed):
            for type_name in _type_names(type_node):
                if (
                    type_name in names
                    and type_name not in found
                    and not allowed(type_name)
                ):
                    found[type_name] = (node, use)

        def wrapped(type_node):
            def allowed(type_name):
                return (
                    isinstance(type_node, A.NamedType)
                    and type_node.name in _STORAGE_WRAPPERS
                    and type_name != type_node.name
                )

            return allowed

        for owner, field in self.scopes():
            for variable in getattr(owner, field, None) or []:
                var_type = getattr(variable, "var_type", None)
                if _is_storage(variable):
                    continue
                note(
                    variable,
                    var_type,
                    f"variable '{variable.name}' declares",
                    wrapped(var_type),
                )
        for struct in self.structs.values():
            for member in struct.members or []:
                member_type = getattr(member, "member_type", None)
                note(
                    member,
                    member_type,
                    f"member '{struct.name}.{member.name}' nests",
                    lambda _: False,
                )
        for function in iter_functions(self.program):
            note(
                function,
                function.return_type,
                f"'{function.name}' returns",
                lambda _: False,
            )
            for parameter in function.parameters or []:
                if not (
                    _is_storage(parameter) or _is_bound_parameter(function, parameter)
                ):
                    param_type = parameter.param_type
                    note(
                        parameter,
                        param_type,
                        f"parameter '{parameter.name}' takes",
                        wrapped(param_type),
                    )
            for node in _walk(function.body):
                if isinstance(node, A.VariableNode) and not _is_storage(node):
                    var_type = node.var_type
                    note(
                        node,
                        var_type,
                        f"local '{node.name}' declares",
                        wrapped(var_type),
                    )
        return found


def legalize_unsized_arrays(program, target) -> None:
    """Rewrite ``buffer T name[]`` storage buffers and refuse every other unsized array of values in ``program``."""
    if isinstance(program, A.ShaderNode):
        _Legalizer(program, str(target or "")).run()
