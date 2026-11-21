"""Storage address-space variables in CrossGL, rewritten to the storage-buffer forms every generator carries.

CrossGL also writes a storage buffer the WGSL way, as a variable in the ``storage`` address space with a group and a
binding, ``@group(0) @binding(1) var<storage, read_write> x: T``, at module scope or as a parameter of a stage entry
point. The variable is the buffer's whole content: ``T`` is a runtime-sized array, a fixed-size array, a structure
(whose last member may be a runtime-sized array), a scalar, a vector or a matrix, and the access mode is ``read`` or
``read_write`` (WGSL, "Address Spaces" and "Var Declarations"). Only the runtime-sized array parameter reached a form
the generators bound. Every other one was silently wrong or rejected by the target compiler:

* a module-scope fixed-size array or scalar became an HLSL global, which DXC places in the read-only ``$Globals``
  constant buffer, and a GLSL ``uniform``, so stores never reached the buffer and the binding was lost;
* a module-scope runtime-sized array was refused as an unsized array, and a structure printed ``array<float>``
  members that no target compiler accepts;
* an entry parameter holding a fixed-size array was copied into a private array by the Metal and WGSL generators,
  and a scalar one was passed by value by the CUDA, HIP, WGSL, Mojo and Rust generators, so stores never reached the
  buffer. HLSL received an entry parameter without a semantic, GLSL an undeclared name and SPIR-V an ``Input``
  variable.

For every target that does not write the declaration itself, :func:`lower_storage_variables` rewrites:

* an entry parameter ``var<storage, A> x: array<T, N>`` into the runtime-sized ``array<T>`` parameter, the form
  every generator binds, with ``x.length()`` becoming ``N``;
* an entry parameter of a scalar, vector, matrix or structure ``T`` into ``array<T>``, with every reference to
  ``x`` becoming ``x[0]``. A structure ending in a runtime-sized array cannot be an array element, so that parameter
  moves to module scope with its group and binding instead;
* a module-scope ``var<storage, A> x: array<T, N>`` (or ``array<T>``) into ``buffer T x[N]`` (or ``buffer T x[]``),
  which :mod:`.unsized_arrays` lowers for every target, so ``x.length()`` is ``N`` (or the bound element count);
* a module-scope variable of a structure ``S`` into the buffer block ``S x @glsl_buffer_block(std430)``, whose
  members keep their names; WGSL's storage layout of a structure is std430's;
* a module-scope variable of a scalar, vector or matrix ``T`` into the one-element buffer ``buffer T x[1]``, with
  every reference to ``x`` becoming ``x[0]``.

The buffer is ``readonly`` (``@readonly`` for a block) when the access mode is ``read``. CUDA and HIP declare a
module-scope storage variable natively, as a ``__device__`` variable, and WGSL as written, so for them only entry
parameters are rewritten. A declaration this pass cannot place is refused with :class:`StorageVariableUnsupportedError`: an element
or value of ``atomic<T>`` type, whose loads and stores the buffer forms do not carry; an opaque type; a variable
rewritten to ``x[0]`` in a function that also declares a local of the same name; and a hoisted parameter whose name or
binding another module declaration, or a different parameter of another entry, already uses.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..stage_utils import STAGE_QUALIFIER_NAMES, normalize_stage_name
from .compute_entry_values import _binding_of, _global_names
from .hub_types import iter_functions, iter_structs, type_from_node
from .rewrite import call, clone
from .storage_blocks import _literal_int, _parents, _replace, _walk
from .unsized_arrays import _MATRIX

__all__ = [
    "KEEPING_TARGETS",
    "MODULE_KEEPING_TARGETS",
    "StorageVariableUnsupportedError",
    "lower_storage_variables",
]

# CrossGL output keeps every declaration as written.
KEEPING_TARGETS = frozenset({"", "cgl", "crossgl"})
# Targets that declare a module-scope storage variable natively: CUDA and HIP as ``__device__`` memory, WGSL as
# written.
MODULE_KEEPING_TARGETS = KEEPING_TARGETS | {"cuda", "hip", "wgsl"}

_ADDRESS_SPACE = frozenset({"storage", "read", "read_write", "write"})


class StorageVariableUnsupportedError(TranslationDiagnostic):
    """A ``var<storage>`` declaration that no storage-buffer form carries without changing its meaning."""

    def __init__(self, target, name, why, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.storage-variable",)
        super().__init__(
            f"storage variable '{name}' {why}",
            target=target,
            code=f"{CODE_PREFIX}{target}-storage-variable-unsupported",
        )


def _lower(values) -> list[str]:
    return [str(value).lower() for value in values or []]


def _space_qualifiers(node) -> list[str]:
    """The address-space qualifiers of a module variable or an entry parameter."""
    if isinstance(node, A.ParameterNode):
        return _lower(getattr(node, "resource_qualifiers", None))
    return _lower(getattr(node, "address_space_qualifiers", None)) or (
        _lower(getattr(node, "qualifiers", None))
        if getattr(node, "is_var_address_space", False)
        else []
    )


def _is_storage(node) -> bool:
    return "storage" in _space_qualifiers(node)


def _is_read_only(node) -> bool:
    qualifiers = _space_qualifiers(node)
    return "read" in qualifiers and not {"read_write", "write"} & set(qualifiers)


def _is_entry(function) -> bool:
    names = _lower(getattr(function, "qualifiers", None))[:1] + _lower(
        getattr(attribute, "name", attribute)
        for attribute in getattr(function, "attributes", None) or []
    )
    return any(normalize_stage_name(name) in STAGE_QUALIFIER_NAMES for name in names)


def _is_atomic(type_node) -> bool:
    name = str(getattr(type_node, "name", "")).lower()
    return isinstance(type_node, (A.NamedType, A.PrimitiveType)) and (
        name == "atomic" or name.startswith(("atomic<", "atomic_"))
    )


def _is_matrix(type_node) -> bool:
    return isinstance(type_node, A.MatrixType) or (
        isinstance(type_node, (A.PrimitiveType, A.NamedType))
        and bool(_MATRIX.fullmatch(str(type_node.name)))
    )


def _int(value):
    return A.LiteralNode(value, A.PrimitiveType("int"))


class _Lowerer:
    def __init__(self, program, target):
        self.program = program
        self.target = target
        self.structs = {
            struct.name: struct
            for struct in iter_structs(program)
            if isinstance(struct, A.StructNode)
            and not getattr(struct, "is_cbuffer", False)
        }
        self.parents = _parents(program)

    def refuse(self, node, why):
        raise StorageVariableUnsupportedError(
            self.target,
            getattr(node, "name", "?"),
            why,
            source_location=getattr(node, "source_location", None),
        )

    def struct_of(self, type_node):
        if isinstance(type_node, A.NamedType) and not type_node.generic_args:
            return self.structs.get(type_node.name)
        return None

    def check_value(self, node, declared):
        """The element (or value) type a buffer holds for ``declared``, refusing what no buffer can hold."""
        element = (
            declared.element_type if isinstance(declared, A.ArrayType) else declared
        )
        if _is_atomic(element):
            self.refuse(
                node,
                "holds atomic values; declare the buffer with integer elements and update them with the "
                "atomic built-in functions",
            )
        if (
            self.struct_of(element) is None
            and type_from_node(element) is None
            and not _is_matrix(element)
        ):
            self.refuse(
                node, f"has type {declared}, which is not a value a buffer can hold"
            )
        return element

    def literal_count(self, node, declared):
        """``array<T, N>`` keeps its count as written; the buffer forms need it as an expression."""
        size = declared.size
        if isinstance(size, (int, str)) and not isinstance(size, bool):
            count = _literal_int(size)
            if count is None:
                self.refuse(
                    node, f"has element count {size!r}, which is not a constant"
                )
            declared.size = _int(count)
        return declared.size

    def references(self, function, name, owner):
        """The identifiers in ``function`` naming ``owner``, refusing a function that also declares ``name``."""
        if any(
            p is not owner and getattr(p, "name", None) == name
            for p in function.parameters or []
        ):
            return []
        found, shadowed = [], False
        for node in _walk(getattr(function, "body", None)):
            if isinstance(node, A.VariableNode) and node.name == name:
                shadowed = True
            elif isinstance(node, A.IdentifierNode) and node.name == name:
                parent = self.parents.get(id(node))
                if not (
                    parent is not None
                    and isinstance(parent[0], A.FunctionCallNode)
                    and parent[1] == "function"
                ):
                    found.append(node)
        if shadowed and found:
            self.refuse(
                owner,
                f"is used in function {function.name}, which declares another '{name}'; its references "
                "cannot be told apart",
            )
        return found

    def index_first(self, identifiers):
        for identifier in identifiers:
            _replace(self.parents, identifier, A.ArrayAccessNode(identifier, _int(0)))

    def fold_lengths(self, identifiers, count):
        """``x.length()`` of a fixed-size array becomes its declared count."""
        for identifier in identifiers:
            access = self.parents.get(id(identifier))
            if not (
                access is not None
                and isinstance(access[0], A.MemberAccessNode)
                and access[0].member == "length"
            ):
                continue
            method = self.parents.get(id(access[0]))
            if (
                method is not None
                and isinstance(method[0], A.FunctionCallNode)
                and method[0].function is access[0]
            ):
                _replace(self.parents, method[0], call("int", [clone(count)]))

    # Entry parameters.

    def lower_parameters(self):
        hoisting = []
        for function in getattr(self.program, "functions", None) or []:
            if not isinstance(function, A.FunctionNode) or not _is_entry(function):
                continue
            kept = []
            for parameter in function.parameters or []:
                if _is_storage(parameter) and self.lower_parameter(function, parameter):
                    hoisting.append((function, parameter))
                    continue
                kept.append(parameter)
            function.parameters = kept
        if hoisting and self.target in MODULE_KEEPING_TARGETS:
            self.refuse(
                hoisting[0][1],
                "holds a structure ending in a runtime-sized array, which a kernel parameter cannot pass",
            )
        if hoisting:
            self.hoist(hoisting)

    def lower_parameter(self, function, parameter) -> bool:
        """Rewrite one storage parameter in place; ``True`` when it has to move to module scope instead."""
        declared = getattr(parameter, "param_type", None)
        if isinstance(declared, A.ArrayType) and declared.size is None:
            return False
        self.check_value(parameter, declared)
        if isinstance(declared, A.ArrayType):
            count = self.literal_count(parameter, declared)
            self.fold_lengths(
                self.references(function, parameter.name, parameter), count
            )
            declared.size = None
            return False
        struct = self.struct_of(declared)
        if struct is not None and any(
            isinstance(m.member_type, A.ArrayType) and m.member_type.size is None
            for m in struct.members or []
        ):
            return True
        identifiers = self.references(function, parameter.name, parameter)
        parameter.param_type = A.ArrayType(declared, None)
        self.index_first(identifiers)
        return False

    def hoist(self, hoisting):
        """Move entry parameters holding a structure that ends in a runtime-sized array to module scope."""
        program = self.program
        names = _global_names(program)
        taken = {
            _binding_of(declaration)
            for declaration in list(program.global_variables or []) + list(
                getattr(program, "cbuffers", None) or []
            )
            if _binding_of(declaration)[1] is not None
        }
        hoisted: dict = {}
        for _, parameter in hoisting:
            binding = _binding_of(parameter)
            signature = (repr(parameter.param_type), binding, _is_read_only(parameter))
            if parameter.name in hoisted:
                if hoisted[parameter.name] != signature:
                    self.refuse(
                        parameter,
                        "is a parameter of two entry points with different types, bindings or access "
                        "modes, which one module-scope buffer cannot be",
                    )
                continue
            if parameter.name in names or (binding[1] is not None and binding in taken):
                self.refuse(
                    parameter,
                    f"binds group {binding[0]}, binding {binding[1]}, but a module declaration already "
                    "uses that name or binding",
                )
            hoisted[parameter.name] = signature
            names.add(parameter.name)
            taken.add(binding)
            variable = A.VariableNode(
                parameter.name,
                parameter.param_type,
                attributes=list(getattr(parameter, "attributes", None) or []),
                qualifiers=list(getattr(parameter, "resource_qualifiers", None) or []),
            )
            variable.vtype = parameter.param_type
            variable.is_var_address_space = True
            variable.address_space_qualifiers = list(variable.qualifiers)
            variable.source_location = getattr(parameter, "source_location", None)
            program.global_variables.append(variable)

    # Module-scope variables.

    def lower_globals(self):
        for variable in list(self.program.global_variables or []):
            if isinstance(variable, A.VariableNode) and _is_storage(variable):
                self.lower_global(variable)

    def lower_global(self, variable):
        declared = variable.var_type
        element = self.check_value(variable, declared)
        read_only = _is_read_only(variable)
        variable.is_var_address_space = False
        variable.address_space_qualifiers = []
        variable.qualifiers = [
            q for q in variable.qualifiers or [] if str(q).lower() not in _ADDRESS_SPACE
        ]
        if isinstance(declared, A.ArrayType):
            self.literal_count(variable, declared)
        elif self.struct_of(element) is not None:
            attributes = [
                A.AttributeNode("glsl_buffer_block", [A.IdentifierNode("std430")]),
                *(variable.attributes or []),
            ]
            if read_only:
                attributes.append(A.AttributeNode("readonly"))
            variable.attributes = attributes
            return
        else:
            identifiers = [
                identifier
                for function in iter_functions(self.program)
                for identifier in self.references(function, variable.name, variable)
            ]
            variable.var_type = variable.vtype = A.ArrayType(declared, _int(1))
            self.index_first(identifiers)
        variable.qualifiers = (
            (["readonly"] if read_only else []) + ["buffer"] + list(variable.qualifiers)
        )


def lower_storage_variables(program, target) -> None:
    """Rewrite ``program``'s ``var<storage>`` declarations into storage-buffer forms for ``target``."""
    target = str(target or "")
    if target in KEEPING_TARGETS or not isinstance(program, A.ShaderNode):
        return
    if getattr(program, "global_variables", None) is None:
        program.global_variables = []
    lowerer = _Lowerer(program, target)
    lowerer.lower_parameters()
    if target not in MODULE_KEEPING_TARGETS:
        lowerer.lower_globals()
