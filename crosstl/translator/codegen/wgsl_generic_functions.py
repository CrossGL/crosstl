"""Monomorphize CrossGL generic functions for the WGSL target.

WGSL has no user-declared generic functions: every function declaration names
concrete parameter and return types (WGSL, "Function Declaration"). Each
instantiation a call site needs therefore becomes its own concrete function.
Module-scope declarations may appear in any order in WGSL, so specializations
are discovered while the module is generated and emitted afterwards.
"""

from __future__ import annotations

import copy
import re

from ..ast import (
    ArrayType,
    FunctionCallNode,
    IdentifierNode,
    LiteralNode,
    MatrixType,
    NamedType,
    PrimitiveType,
    UnaryOpNode,
    VectorType,
)
from .enum_utils import generic_type_parts, sanitize_type_name

# CrossGL primitive spellings for the scalar types WGSL renders.
_WGSL_SCALAR_SOURCE_NAMES = {
    "f32": "float",
    "i32": "int",
    "u32": "uint",
    "bool": "bool",
}
# Attributes whose plain-string value spells a type rather than a name.
_TYPE_ATTRIBUTE_NAMES = frozenset(
    {
        "var_type",
        "param_type",
        "return_type",
        "target_type",
        "constructor_type",
        "element_type",
        "vtype",
    }
)
_WALK_EXCLUDED_ATTRIBUTES = frozenset({"parent", "source_location", "annotations"})
# WGSL converts these scalars to one another at user-function call sites.
_NUMERIC_SCALARS = frozenset({"f32", "i32", "u32"})
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# A bound on the instantiations one module may need, so a generic that calls
# itself with an ever larger type fails closed instead of growing forever.
_MAX_SPECIALIZATIONS = 256


def generic_parameter_names(func):
    """Return the type parameter names a function declares, in order."""
    return [
        name
        for name in (
            getattr(param, "name", None)
            for param in getattr(func, "generic_params", None) or []
        )
        if name
    ]


class WGSLGenericFunctionSpecializer:
    """Create the concrete WGSL functions a module's generic call sites need."""

    def __init__(self, generator, functions):
        self.generator = generator
        self.definitions = {}
        for func in functions or []:
            if generic_parameter_names(func) and getattr(func, "name", None):
                self.definitions.setdefault(func.name, []).append(func)
        self.specializations = {}
        self.pending = []

    def is_generic(self, func):
        return bool(generic_parameter_names(func))

    def _reject_resource_parameters(self, func):
        """Reject generics whose parameters need per-function resource lowering."""
        generator = self.generator
        resource_maps = (
            generator._function_texture_parameters,
            generator._function_pointer_parameters,
            generator._function_resource_member_parameters,
        )
        keys = (generator.function_signature_key(func), func.name)
        if any(key in mapping for key in keys for mapping in resource_maps):
            raise ValueError(
                "WGSL target cannot specialize generic function "
                f"'{func.name}' with texture, sampler or buffer parameters; "
                "declare a concrete helper for resource arguments"
            )

    def rewrite_call(self, node):
        """Return ``node``, or a copy that calls the concrete specialization."""
        if not self.definitions or not isinstance(node, FunctionCallNode):
            return node
        callee = getattr(node, "function", None)
        if not isinstance(callee, IdentifierNode):
            return node
        base_name, explicit_args = generic_type_parts(callee.name)
        generics = self.definitions.get(base_name)
        if not generics:
            return node
        arguments = list(getattr(node, "arguments", None) or [])
        if not explicit_args and self._concrete_overload_matches(base_name, arguments):
            return node

        explicit = [self._render(arg) for arg in explicit_args]
        argument_types = [self._argument_type(arg) for arg in arguments]
        literal_flags = [_is_numeric_literal(arg) for arg in arguments]
        matches = []
        for func in generics:
            self._reject_resource_parameters(func)
            match = self._match(func, explicit, argument_types, literal_flags)
            if match is not None:
                matches.append(match)
        if not matches:
            if not explicit_args and self._concrete_overloads(base_name, arguments):
                return node
            raise ValueError(
                "WGSL target cannot infer concrete type arguments for generic "
                f"function '{base_name}' ({', '.join(self._parameter_list(generics))})"
                f" from call arguments ({self._argument_list(argument_types)}); "
                "pass arguments of known type or name the type arguments"
            )
        matches.sort(key=lambda match: match[0], reverse=True)
        if len(matches) > 1 and matches[0][0] == matches[1][0]:
            raise ValueError(
                "WGSL target found an ambiguous call to generic function "
                f"'{base_name}' with arguments ({self._argument_list(argument_types)})"
            )
        _rank, func, concrete = matches[0]
        specialized_name = self._specialization_name(func, concrete)
        rewritten = copy.copy(node)
        rewritten_callee = IdentifierNode(specialized_name)
        rewritten.function = rewritten_callee
        if hasattr(rewritten, "name"):
            rewritten.name = rewritten_callee
        return rewritten

    def generate_pending(self):
        """Emit every pending specialization, including ones they request."""
        emitted = []
        while self.pending:
            clone = self.pending.pop(0)
            emitted.append(self.generator.generate_function(clone))
        return emitted

    def _render(self, type_value):
        if type_value is None:
            return None
        try:
            rendered = self.generator.type_name_string(type_value)
        except ValueError:
            return None
        return str(rendered).strip() if rendered else None

    def _argument_type(self, argument):
        return self._render(self.generator.expression_type(argument))

    def _concrete_overloads(self, name, arguments):
        return [
            func
            for func in self.generator._function_overloads_by_name.get(name, ())
            if not self.is_generic(func)
            and len(getattr(func, "parameters", None) or []) == len(arguments)
        ]

    def _concrete_overload_matches(self, name, arguments):
        candidates = self._concrete_overloads(name, arguments)
        if not candidates:
            return False
        if not self.definitions.get(name):
            return True
        argument_types = [
            self.generator.expression_type(argument) for argument in arguments
        ]
        for func in candidates:
            parameters = getattr(func, "parameters", None) or []
            if all(
                self.generator.function_types_compatible(
                    argument_type, getattr(parameter, "param_type", None)
                )
                for argument_type, parameter in zip(argument_types, parameters)
            ):
                return True
        return False

    def _match(self, func, explicit, argument_types, literal_flags):
        """Return ``(rank, func, concrete_args)`` when the call binds every parameter.

        Arguments deduce in order, as the HLSL, GLSL and MSL generators do, except
        that numeric literals deduce last: ``add(x, 1)`` and ``add(1, x)`` both
        take ``T`` from ``x``. A later scalar argument whose numeric type differs
        from the deduced one is converted at the call site rather than rejected.
        """
        params = generic_parameter_names(func)
        parameters = list(getattr(func, "parameters", None) or [])
        if len(explicit) > len(params) or len(parameters) != len(argument_types):
            return None
        if any(arg is None for arg in explicit):
            return None
        bindings = dict(zip(params, explicit))
        explicit_names = set(bindings)
        specificity = 0
        ordered = sorted(
            zip(parameters, argument_types, literal_flags),
            key=lambda item: item[2],
        )
        for parameter, actual, _literal in ordered:
            expected = self._render(getattr(parameter, "param_type", None))
            if expected is None:
                return None
            # A parameter whose type is concrete, or fixed by explicit type
            # arguments, converts its argument instead of deducing from it.
            if all(
                name in explicit_names
                for name in _spelling_parameters(expected, params)
            ):
                continue
            if actual is None:
                return None
            bound = bindings.get(expected.strip())
            if bound in _NUMERIC_SCALARS and actual in _NUMERIC_SCALARS:
                specificity += _spelling_structure(expected, params)
                continue
            if not _unify(expected, actual, bindings, params):
                return None
            specificity += _spelling_structure(expected, params)
        for param in getattr(func, "generic_params", None) or []:
            name = getattr(param, "name", None)
            default_type = getattr(param, "default_type", None)
            if name in bindings or default_type is None:
                continue
            default = self._render(default_type)
            if default is None:
                return None
            bindings[name] = _substitute_spelling(default, bindings)
        if any(param not in bindings for param in params):
            return None
        concrete = tuple(bindings[param] for param in params)
        if not all(_spelling_is_concrete(arg, params) for arg in concrete):
            return None
        return (specificity, func, concrete)

    def _specialization_name(self, func, concrete):
        key = (id(func), concrete)
        existing = self.specializations.get(key)
        if existing is not None:
            return existing.name
        if len(self.specializations) >= _MAX_SPECIALIZATIONS:
            raise ValueError(
                "WGSL target stopped specializing generic function "
                f"'{func.name}' after {_MAX_SPECIALIZATIONS} instantiations; "
                "the generic instantiates itself without bound"
            )
        generator = self.generator
        candidate = "_".join(
            [sanitize_type_name(func.name)]
            + [sanitize_type_name(arg) for arg in concrete]
        )
        name = generator.unique_function_module_name(candidate, self._used_names())
        clone = self._instantiate(func, concrete, name)
        self.specializations[key] = clone
        self.pending.append(clone)
        signature_key = generator.function_signature_key(clone)
        generator._function_overloads_by_name[name] = [clone]
        generator._function_identifier_names[name] = name
        generator._function_signature_identifier_names[signature_key] = name
        generator._function_return_types[name] = clone.return_type
        generator._function_return_types_by_signature[signature_key] = clone.return_type
        return name

    def _used_names(self):
        generator = self.generator
        used = set()
        for mapping in (
            generator._type_identifier_names,
            generator._function_identifier_names,
            generator._function_signature_identifier_names,
            generator._module_identifier_names,
        ):
            used.update(str(value) for value in mapping.values())
        used.update(clone.name for clone in self.specializations.values())
        return used

    def _instantiate(self, func, concrete, name):
        params = generic_parameter_names(func)
        bindings = dict(zip(params, concrete))
        memo = {}
        parent = getattr(func, "parent", None)
        if parent is not None:
            memo[id(parent)] = parent
        clone = copy.deepcopy(func, memo)
        clone.name = name
        clone.generic_params = []
        type_nodes = {
            param: _type_node(spelling) for param, spelling in bindings.items()
        }
        substituter = _GenericSubstituter(bindings, type_nodes, self.generator)
        for attribute in ("return_type", "parameters", "body"):
            setattr(
                clone,
                attribute,
                substituter.substitute(getattr(clone, attribute, None)),
            )
        return clone

    def _parameter_list(self, generics):
        names = []
        for func in generics:
            for name in generic_parameter_names(func):
                if name not in names:
                    names.append(name)
        return names

    def _argument_list(self, argument_types):
        return ", ".join(arg or "unknown" for arg in argument_types)


class _GenericSubstituter:
    """Replace type parameters inside a cloned function, in place."""

    def __init__(self, bindings, type_nodes, generator):
        self.bindings = bindings
        self.type_nodes = type_nodes
        self.generator = generator
        self.visited = set()

    def substitute(self, value):
        if isinstance(value, (PrimitiveType, NamedType)) and not getattr(
            value, "generic_args", None
        ):
            if value.name in self.type_nodes:
                return copy.deepcopy(self.type_nodes[value.name])
        if isinstance(value, IdentifierNode):
            value.name = self._substitute_name(value.name)
            return value
        if isinstance(value, list):
            for index, item in enumerate(value):
                value[index] = self.substitute(item)
            return value
        if isinstance(value, tuple):
            return tuple(self.substitute(item) for item in value)
        if isinstance(value, dict):
            for key, item in list(value.items()):
                value[key] = self.substitute(item)
            return value
        if not hasattr(value, "__dict__") or isinstance(value, type):
            return value
        if id(value) in self.visited:
            return value
        self.visited.add(id(value))
        for attribute, item in list(vars(value).items()):
            if attribute in _WALK_EXCLUDED_ATTRIBUTES:
                continue
            if isinstance(item, str):
                if attribute in _TYPE_ATTRIBUTE_NAMES:
                    setattr(value, attribute, _substitute_spelling(item, self.bindings))
                continue
            setattr(value, attribute, self.substitute(item))
        return value

    def _substitute_name(self, name):
        if name in self.bindings:
            return _constructor_spelling(self.bindings[name], self.generator)
        _base, args = generic_type_parts(name)
        if not args:
            return name
        return _substitute_spelling(name, self.bindings)


def _is_numeric_literal(node):
    if isinstance(node, UnaryOpNode) and getattr(node, "operator", None) in "+-":
        node = getattr(node, "operand", None)
    if not isinstance(node, LiteralNode):
        return False
    literal_type = getattr(node, "literal_type", None)
    return getattr(literal_type, "name", None) in {"int", "uint", "float"}


def _unify(expected, actual, bindings, params):
    expected = expected.strip()
    actual = actual.strip()
    if expected in params:
        bound = bindings.get(expected)
        if bound is None:
            bindings[expected] = actual
            return True
        return bound == actual
    expected_base, expected_args = generic_type_parts(expected)
    actual_base, actual_args = generic_type_parts(actual)
    if not expected_args:
        return expected == actual
    if expected_base != actual_base or len(expected_args) != len(actual_args):
        return False
    return all(
        _unify(expected_arg, actual_arg, bindings, params)
        for expected_arg, actual_arg in zip(expected_args, actual_args)
    )


def _spelling_parameters(spelling, params):
    spelling = spelling.strip()
    if spelling in params:
        return [spelling]
    _base, args = generic_type_parts(spelling)
    found = []
    for arg in args:
        found.extend(_spelling_parameters(arg, params))
    return found


def _spelling_is_concrete(spelling, params):
    return not _spelling_parameters(spelling, params)


def _spelling_structure(spelling, params):
    spelling = spelling.strip()
    if spelling in params:
        return 0
    _base, args = generic_type_parts(spelling)
    return 1 + sum(_spelling_structure(arg, params) for arg in args)


def _substitute_spelling(spelling, bindings):
    spelling = str(spelling).strip()
    if spelling in bindings:
        return bindings[spelling]
    base, args = generic_type_parts(spelling)
    if not args:
        return spelling
    return (
        f"{base}<"
        + ", ".join(_substitute_spelling(arg, bindings) for arg in args)
        + ">"
    )


def _type_node(spelling):
    """Return the CrossGL type node for a WGSL type spelling."""
    spelling = spelling.strip()
    scalar = _WGSL_SCALAR_SOURCE_NAMES.get(spelling)
    if scalar is not None:
        return PrimitiveType(scalar)
    base, args = generic_type_parts(spelling)
    vector = re.fullmatch(r"vec([234])", base)
    if vector and len(args) == 1:
        return VectorType(_type_node(args[0]), int(vector.group(1)))
    matrix = re.fullmatch(r"mat([234])x([234])", base)
    if matrix and len(args) == 1:
        return MatrixType(
            _type_node(args[0]),
            rows=int(matrix.group(2)),
            cols=int(matrix.group(1)),
        )
    if base == "array" and len(args) in (1, 2):
        size = None
        if len(args) == 2:
            size = int(args[1]) if args[1].isdigit() else args[1]
        return ArrayType(_type_node(args[0]), size)
    if not args and _IDENTIFIER_RE.fullmatch(spelling):
        return NamedType(spelling)
    return spelling


def _constructor_spelling(spelling, generator):
    """Return the callee name that constructs a value of ``spelling``."""
    if generator.is_type_constructor_name(spelling):
        return spelling
    base, args = generic_type_parts(spelling)
    if re.fullmatch(r"mat[234]x[234]", base) and args == ["f32"]:
        return base
    return spelling
