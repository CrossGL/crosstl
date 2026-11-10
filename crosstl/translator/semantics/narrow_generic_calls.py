"""Generic calls whose type arguments are narrow integers, on targets that widen narrow registers.

Narrow-integer legalization (:mod:`.narrow`) gives every 8- and 16-bit register a 32-bit ``int``. A generic call
``bump(a)`` with ``uint8 a`` deduces ``T = uint8`` (C++ [temp.deduct.call]/1, [temp.deduct.type]), but the
generators deduce after that widening and bound ``T = int``: an explicit specialization
``@explicit_specialization(bump, uint8)`` was silently skipped, and the primary ran in 32 bits, so its arithmetic
never wrapped at the width of ``T``.

The legalizer therefore resolves such a call before it widens anything. A call whose deduced or explicit type
arguments include a narrow type names the explicit specialization for those arguments when the program declares
one ([temp.expl.spec]/1). Otherwise it names a concrete instantiation of the primary ([temp.inst]), cloned from the
primary's definition as it was before legalization, with every type parameter replaced; the legalizer then
legalizes that instantiation like any function with narrow parameters, including the generic calls in its body that
only become narrow there. A call the pass cannot resolve is refused: a type parameter it cannot deduce, two
arguments that deduce different types, an overloaded name, or a primary declared without a definition.
"""

from __future__ import annotations

import copy
import re

from .. import ast as A
from ..explicit_specializations import explicit_specializations
from .hub_types import HubType, call_name, promote, type_from_node
from .rewrite import clone, set_child

__all__ = ["NarrowGenericCalls", "call_names_narrow_type_arguments"]

_SCALAR_SPELLINGS = {
    ("i", 32): "int",
    ("u", 32): "uint",
    ("f", 32): "float",
    ("f", 16): "half",
    ("f", 64): "double",
    ("b", 1): "bool",
}
_VECTOR_PREFIXES = {"int": "ivec", "uint": "uvec", "float": "vec", "bool": "bvec"}
_TYPE_PARAMETER_FIELDS = ("generic_args",)


def _is_narrow(value):
    return isinstance(value, HubType) and value.kind in "iu" and value.bits < 32


def _template_arguments(text):
    """The comma-separated arguments of ``<...>``, split at the top level only."""
    arguments, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char in "<([":
            depth += 1
        elif char in ">)]":
            depth -= 1
        elif char == "," and depth == 0:
            arguments.append(text[start:index].strip())
            start = index + 1
    arguments.append(text[start:].strip())
    return [argument for argument in arguments if argument]


def explicit_call_parts(node):
    """``(callee name, explicit template arguments)``: the parser spells ``f<uint8>(x)`` as the callee ``f<uint8>``."""
    name = call_name(node)
    explicit = list(getattr(node, "generic_args", None) or [])
    head, bracket, tail = name.partition("<")
    if bracket and tail.endswith(">") and head.strip():
        return head.strip(), _template_arguments(tail[:-1]) + explicit
    return name, explicit


def call_names_narrow_type_arguments(node, narrow_of) -> bool:
    """``f<uint8>(x)``: a call whose explicit generic arguments name a narrow type."""
    if not isinstance(node, A.FunctionCallNode):
        return False
    _, explicit = explicit_call_parts(node)
    return any(
        isinstance(argument, (str, A.TypeNode)) and narrow_of(argument) is not None
        for argument in explicit
    )


class _Binding:
    """A deduced type argument: its value type (``None`` for aggregates), type node and CrossGL spelling."""

    __slots__ = ("value", "node", "spelling")

    def __init__(self, value, node, spelling):
        self.value, self.node, self.spelling = value, node, spelling

    def same(self, other) -> bool:
        if self.value is not None or other.value is not None:
            return self.value == other.value
        return self.spelling == other.spelling


def _spelled_value(value: HubType):
    """``(type node, spelling)`` of a scalar or vector value type in CrossGL."""
    if _is_narrow(value):
        scalar = f"{'uint' if value.kind == 'u' else 'int'}{value.bits}"
    else:
        scalar = _SCALAR_SPELLINGS.get((value.kind, value.bits))
        if scalar is None:
            return None
    element = A.PrimitiveType(scalar)
    if value.lanes == 1:
        return element, scalar
    prefix = _VECTOR_PREFIXES.get(scalar)
    spelling = f"{prefix}{value.lanes}" if prefix else f"vec{value.lanes}<{scalar}>"
    return A.VectorType(element, value.lanes), spelling


def _binding_from_value(value):
    spelled = _spelled_value(value) if value is not None else None
    return _Binding(value, *spelled) if spelled is not None else None


def _type_spelling(node):
    if isinstance(node, (A.PrimitiveType, A.NamedType)) and not getattr(
        node, "generic_args", None
    ):
        return str(node.name)
    return None


def _substituted(node, bindings):
    """``node`` with each type parameter replaced by its binding; ``None`` when nothing changes."""
    if isinstance(node, (A.PrimitiveType, A.NamedType)):
        arguments = getattr(node, "generic_args", None) or []
        if not arguments:
            bound = bindings.get(str(node.name))
            return copy.copy(bound.node) if bound is not None else None
        replaced = [_substituted(argument, bindings) for argument in arguments]
        if all(item is None for item in replaced):
            return None
        concrete = [
            new if new is not None else old for new, old in zip(replaced, arguments)
        ]
        match = re.fullmatch(r"vec([234])", str(node.name))
        if match and len(concrete) == 1:
            # ``vec2<T>`` with ``T = uint8`` is the vector type ``vec2<uint8>``.
            return A.VectorType(concrete[0], int(match.group(1)))
        duplicate = copy.copy(node)
        duplicate.generic_args = concrete
        return duplicate
    for field in ("element_type", "pointee_type", "referenced_type"):
        inner = getattr(node, field, None)
        if isinstance(inner, A.TypeNode):
            replaced = _substituted(inner, bindings)
            if replaced is None:
                return None
            duplicate = copy.copy(node)
            setattr(duplicate, field, replaced)
            return duplicate
    return None


def _substitute_callee(node, bindings):
    """``T(1)`` constructs the bound type; ``helper<T>(x)`` names the bound arguments."""
    function = getattr(node, "function", None)
    if not isinstance(function, A.IdentifierNode):
        return
    name = str(function.name)
    if name in bindings:
        concrete = bindings[name].spelling
    elif "<" in name:
        head, _, tail = name.partition("<")
        tail = re.sub(
            r"\b\w+\b",
            lambda m: (
                bindings[m.group(0)].spelling if m.group(0) in bindings else m.group(0)
            ),
            tail,
        )
        concrete = f"{head}<{tail}"
    else:
        return
    if concrete != name:
        set_child(node, "function", None, A.IdentifierNode(concrete))


def instantiate(template, name, bindings):
    """A concrete copy of the generic function ``template`` for ``bindings``."""
    function = clone(template)
    function.name = name
    function.generic_params = []
    function.annotations.pop("explicit_specialization", None)
    for node in [function, *function.walk()]:
        for field, value in list(vars(node).items()):
            if field in ("parent", "annotations", "source_location"):
                continue
            if isinstance(value, A.TypeNode):
                replaced = _substituted(value, bindings)
                if replaced is not None:
                    for alias, other in list(vars(node).items()):
                        if other is value:
                            setattr(node, alias, replaced)
            elif field in _TYPE_PARAMETER_FIELDS and isinstance(value, list):
                setattr(
                    node,
                    field,
                    [
                        (
                            (_substituted(item, bindings) or item)
                            if isinstance(item, A.TypeNode)
                            else item
                        )
                        for item in value
                    ],
                )
        if isinstance(node, A.FunctionCallNode):
            _substitute_callee(node, bindings)
    return function


class NarrowGenericCalls:
    """Resolve generic calls with narrow type arguments for a :class:`~.narrow.NarrowLegalizer`."""

    def __init__(self, legalizer, containers, narrow_of):
        self.legalizer = legalizer
        self.narrow_of = narrow_of
        # ``containers``: ``(function list, stage or None)`` in program order.
        self.generics = {}
        self.homes = {}
        self.templates = {}
        names, self.concrete = set(), set()
        for functions, stage in containers:
            for function in functions:
                if not isinstance(function, A.FunctionNode):
                    continue
                names.add(function.name)
                if not function.generic_params:
                    self.concrete.add(function.name)
                else:
                    self.generics.setdefault(function.name, []).append(function)
                    self.homes[id(function)] = (functions, stage)
                    # The definition as written, before the legalizer rewrites its body.
                    self.templates[id(function)] = clone(function)
        self.taken = names
        self.specializations = [
            entry
            for functions, _ in containers
            for entry in explicit_specializations(functions)
        ]
        self.instances = {}
        self.last_instance = {}
        self.pending = []

    # Deduction ----------------------------------------------------------------------------------------------
    def _value_of(self, type_or_spelling):
        """The value type a type denotes; a minimum-precision ``min16int`` is not narrow and keeps its spelling."""
        narrow = self.narrow_of(type_or_spelling)
        if narrow is not None:
            return narrow
        value = type_from_node(type_or_spelling)
        return None if value is None or _is_narrow(value) else value

    def _type_binding(self, type_node):
        value = self._value_of(type_node)
        if value is not None:
            return _binding_from_value(value)
        spelling = _type_spelling(type_node)
        return _Binding(None, type_node, spelling) if spelling is not None else None

    def _argument_binding(self, argument):
        legalizer = self.legalizer
        narrow = legalizer.narrow_value(argument)
        if narrow is not None:
            return _binding_from_value(narrow)
        declared = legalizer.scope.node_type(argument)
        if declared is not None:
            return self._type_binding(declared)
        value = legalizer.scope.infer(argument)
        # An integer narrower than 32 bits that is not a narrow value here has been promoted.
        return _binding_from_value(promote(value)) if value is not None else None

    def _deduce(self, parameter_type, argument, parameters, bindings, node):
        """Bind the type parameters ``parameter_type`` names from ``argument`` (C++ [temp.deduct.call])."""
        name = _type_spelling(parameter_type)
        if isinstance(parameter_type, A.ReferenceType):
            return self._deduce(
                parameter_type.referenced_type, argument, parameters, bindings, node
            )
        if name in parameters:
            found = self._argument_binding(argument)
            if found is None:
                return
            self._bind(bindings, name, found, node)
            return
        if isinstance(parameter_type, A.ArrayType):
            element = _type_spelling(parameter_type.element_type)
            declared = self.legalizer.scope.node_type(argument)
            if element in parameters and isinstance(declared, A.ArrayType):
                found = self._type_binding(declared.element_type)
                if found is not None:
                    self._bind(bindings, element, found, node)
            return
        if isinstance(parameter_type, A.VectorType):
            element = _type_spelling(parameter_type.element_type)
            if element not in parameters:
                return
            found = self._argument_binding(argument)
            if found is None or found.value is None or found.value.lanes == 1:
                return
            self._bind(bindings, element, _binding_from_value(found.value.scalar), node)

    def _bind(self, bindings, name, found, node):
        previous = bindings.get(name)
        if previous is not None and not previous.same(found):
            self.legalizer.refuse(
                f"the generic call '{call_name(node)}', whose arguments deduce both "
                f"{previous.spelling} and {found.spelling} for {name}",
                "generic-deduction",
                node,
            )
        bindings.setdefault(name, found)

    def _candidates(self, name, count):
        generics = [
            function
            for function in self.generics.get(name, [])
            if self._accepts(function, count)
        ]
        defined = [function for function in generics if function.body is not None]
        # A body-less generic of the same signature is a forward declaration ([temp.over.link]/1).
        return defined or generics

    @staticmethod
    def _accepts(function, count):
        parameters = list(function.parameters or [])
        if count > len(parameters):
            return False
        return all(
            getattr(parameter, "default_value", None) is not None
            for parameter in parameters[count:]
        )

    def _bindings(self, primary, node, explicit):
        parameters = [str(getattr(p, "name", p)) for p in primary.generic_params]
        if len(explicit) > len(parameters):
            return None
        bindings = {}
        for name, argument in zip(parameters, explicit):
            if isinstance(argument, str):
                argument = A.NamedType(argument)
            bound = (
                self._type_binding(argument)
                if isinstance(argument, A.TypeNode)
                else None
            )
            if bound is None:
                return None
            bindings[name] = bound
        # An explicitly specified parameter is not deduced; its argument converts ([temp.arg.explicit]/6).
        deduced = set(parameters) - set(bindings)
        for parameter, argument in zip(primary.parameters or [], node.arguments or []):
            declared = getattr(parameter, "param_type", None)
            self._deduce(declared, argument, deduced, bindings, node)
        return parameters, bindings

    # Resolution ---------------------------------------------------------------------------------------------
    def resolve(self, node):
        """The concrete callee for the generic call ``node`` when a narrow type argument is involved, else ``None``."""
        name, explicit = explicit_call_parts(node)
        if name not in self.generics or not isinstance(
            getattr(node, "function", None), A.IdentifierNode
        ):
            return None
        candidates = self._candidates(name, len(node.arguments or []))
        if not candidates:
            return None
        deduced = [self._bindings(primary, node, explicit) for primary in candidates]
        narrow = [
            entry
            for entry in deduced
            if entry is not None and any(_is_narrow(b.value) for b in entry[1].values())
        ]
        if not narrow:
            return None
        if len(candidates) > 1 or name in self.concrete:
            self.legalizer.refuse(
                f"the call of overloaded generic '{name}' with a narrow-integer type argument",
                "overload",
                node,
            )
        primary = candidates[0]
        parameters, bindings = narrow[0]
        missing = [parameter for parameter in parameters if parameter not in bindings]
        if missing:
            self.legalizer.refuse(
                f"the generic call '{name}', whose type parameter {missing[0]} is not deduced "
                "while another binds a narrow-integer type",
                "generic-deduction",
                node,
            )
        arguments = tuple(bindings[parameter] for parameter in parameters)
        callee = self._specialization(name, arguments) or self._instance(
            primary, arguments, bindings, node
        )
        set_child(node, "function", None, A.IdentifierNode(callee))
        node.generic_args = []
        node.generic_suffix_present = False
        return callee

    def _specialization(self, name, arguments):
        for function, primary, spellings in self.specializations:
            if primary != name or len(spellings) != len(arguments):
                continue
            if all(
                bound.same(self._spelled_binding(spelling))
                for spelling, bound in zip(spellings, arguments)
            ):
                return function.name
        return None

    def _spelled_binding(self, spelling):
        return _Binding(self._value_of(spelling), None, str(spelling).replace(" ", ""))

    def _instance(self, primary, arguments, bindings, node):
        key = (id(primary), tuple((b.value, b.spelling) for b in arguments))
        if key in self.instances:
            return self.instances[key]
        if primary.body is None:
            self.legalizer.refuse(
                f"the call of '{primary.name}<{', '.join(b.spelling for b in arguments)}>': the generic "
                "is declared without a definition and no explicit specialization covers it",
                "generic-definition",
                node,
            )
        stem = "_".join(
            [primary.name, *(re.sub(r"\W+", "_", b.spelling) for b in arguments)]
        )
        name, ordinal = stem, 1
        while name in self.taken:
            name, ordinal = f"{stem}_{ordinal}", ordinal + 1
        self.taken.add(name)
        self.instances[key] = name
        function = instantiate(self.templates[id(primary)], name, bindings)
        functions, stage = self.homes[id(primary)]
        # Instantiations follow their primary in the order the calls reach them.
        previous = self.last_instance.get(id(primary), primary)
        functions.insert(functions.index(previous) + 1, function)
        self.last_instance[id(primary)] = function
        self.legalizer.functions.setdefault(name, []).append(function)
        self.pending.append((function, stage))
        return name

    def drain(self, stage):
        """Legalize the pending instantiations that live in ``stage`` (``None``: the program's functions)."""
        while True:
            ready = [entry for entry in self.pending if entry[1] is stage]
            if not ready:
                return
            for entry in ready:
                self.pending.remove(entry)
                self.legalizer.function(entry[0])

    def verify_drained(self):
        if self.pending:
            function = self.pending[0][0]
            self.legalizer.refuse(
                f"the instantiation '{function.name}', reached from outside its stage",
                "generic-scope",
                function,
            )
