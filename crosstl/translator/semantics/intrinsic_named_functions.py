"""User functions that share a name with a CrossGL texture or image built-in.

GLSL lets a program overload a built-in function with its own (GLSL 4.60 section 6.1, "Function Definitions"), so
``float textureProj(vec4 coord, vec2 offset)`` can sit beside the built-in ``textureProj(sampler2D, vec3)``.
Every CrossTL generator, and the rule passes before them, read a call named like a texture or image built-in as
that built-in, so a call to such a user overload was lowered as a texture operation on a value and failed. GLSL ES
forbids the overload outright (GLSL ES 3.20 section 6.1), so keeping the name would not be portable either.

:func:`resolve_intrinsic_named_functions` therefore gives every user function named like a texture or image
built-in a fresh name and redirects each call that resolves to it. A call resolves to the user overloads when one
of them takes its number of arguments with a value (not a texture, sampler or image) first, and the call's first
argument is a value; the built-ins all take a resource first, so overload resolution cannot pick them for such a
call. A call whose first argument is a resource stays a built-in, unless a user overload of the same arity also
takes a resource first: telling the two apart needs every argument's type, so that call is refused with
:class:`IntrinsicOverloadAmbiguousError`.
"""

from __future__ import annotations

from .. import ast as A
from ..codegen.image_access_contracts import texture_image_resource_operation_names
from ..diagnostics import CODE_PREFIX, SourceSpan, TranslationDiagnostic
from ..validation import (
    IMAGE_RESOURCE_INTRINSIC_NAMES,
    _is_sampler_descriptor_type,
    _is_texture_descriptor_type,
    _type_base_name,
)
from .hub_types import call_name, iter_structs
from .scoped_walk import ScopedWalker, parameter_type, program_stages

__all__ = [
    "IntrinsicOverloadAmbiguousError",
    "RESOURCE_BUILTIN_NAMES",
    "resolve_intrinsic_named_functions",
]

RESOURCE_BUILTIN_NAMES = frozenset(
    texture_image_resource_operation_names(IMAGE_RESOURCE_INTRINSIC_NAMES)
    | {"textureGatherCompareOffsets"}
)
# Resource type spellings the validation predicates do not cover: combined samplers beyond the 1D/2D/3D/cube
# shapes (buffer, rectangle, external), Metal depth textures and subpass inputs.
_RESOURCE_PREFIXES = (
    "sampler",
    "isampler",
    "usampler",
    "depth",
    "subpassinput",
    "isubpassinput",
    "usubpassinput",
)
# Calls that return their first argument's resource unchanged.
_RESOURCE_PASSTHROUGH_CALLS = frozenset({"nonuniformEXT"})
_RENAME_SUFFIX = "_fn"


class IntrinsicOverloadAmbiguousError(TranslationDiagnostic):
    """A call that may resolve to a texture or image built-in or to a user overload of the same name."""

    def __init__(self, name, arity, *, source_location=None):
        self.function_name = name
        self.reason_code = "intrinsic-overload-ambiguous"
        self.source_location = source_location
        super().__init__(
            f"call to '{name}' with {arity} argument(s) and a resource first matches both the built-in and a user "
            f"function of that name taking a resource first; give the user function another name",
            span=_span(source_location),
            code=f"{CODE_PREFIX}intrinsic-overload-ambiguous",
        )


def resolve_intrinsic_named_functions(program) -> dict[str, str]:
    """Rename user functions named like texture or image built-ins, and the calls that resolve to them.

    The AST is rewritten in place. Returns the renames as ``{old: new}``. Raises
    :class:`IntrinsicOverloadAmbiguousError` for a call that could resolve to either.
    """
    if not isinstance(program, A.ShaderNode):
        return {}
    entries = {
        id(stage.entry_point)
        for stage in program_stages(program)
        if getattr(stage, "entry_point", None) is not None
    }
    overloads = {}
    for function in _user_functions(program):
        if function.name in RESOURCE_BUILTIN_NAMES and id(function) not in entries:
            overloads.setdefault(function.name, []).append(function)
    if not overloads:
        return {}
    renames = _fresh_names(program, sorted(overloads))
    _CallResolver(program, overloads, renames).run_program()
    for name, functions in overloads.items():
        for function in functions:
            function.name = renames[name]
    return renames


class _CallResolver(ScopedWalker):
    """Redirects the calls that resolve to a renamed user overload."""

    def __init__(self, program, overloads, renames):
        super().__init__(program)
        self.overloads = overloads
        self.renames = renames
        self.struct_names = {struct.name for struct in iter_structs(program)}
        self.resource_members = {
            member_name
            for members in self.scope.structs.values()
            for member_name, member_type in members.items()
            if self.is_resource_type(member_type)
        }
        # A helper outside every stage may still name a stage's resource, so every resource is also known by name.
        self.resource_names = {
            variable.name
            for variable in getattr(program, "global_variables", None) or []
            if self.is_resource_type(variable.var_type)
        }
        for stage in program_stages(program):
            for variable in getattr(stage, "local_variables", None) or []:
                if self.is_resource_type(getattr(variable, "var_type", None)):
                    self.resource_names.add(variable.name)

    def is_resource_type(self, type_node):
        base = _type_base_name(type_node)
        if not base or base in self.struct_names:
            return False
        return (
            _is_texture_descriptor_type(type_node)
            or _is_sampler_descriptor_type(type_node)
            or base.lower().startswith(_RESOURCE_PREFIXES)
        )

    def visit(self, node):
        if isinstance(node, A.FunctionCallNode):
            self._visit_call(node)

    def _visit_call(self, call):
        name = call_name(call)
        overloads = self.overloads.get(name)
        if not overloads:
            return
        arguments = list(call.arguments or [])
        candidates = [
            function for function in overloads if _accepts(function, len(arguments))
        ]
        if not candidates:
            return
        first_is_resource = bool(arguments) and self.is_resource_expression(
            arguments[0]
        )
        takes_resource = [
            bool(function.parameters)
            and self.is_resource_type(parameter_type(function.parameters[0]))
            for function in candidates
        ]
        if first_is_resource:
            if any(takes_resource):
                raise IntrinsicOverloadAmbiguousError(
                    name,
                    len(arguments),
                    source_location=getattr(call, "source_location", None),
                )
            return
        if all(takes_resource):
            return
        _rename_call(call, self.renames[name])

    def is_resource_expression(self, expression):
        if isinstance(expression, A.TernaryOpNode):
            return self.is_resource_expression(
                expression.true_expr
            ) or self.is_resource_expression(expression.false_expr)
        if isinstance(expression, A.FunctionCallNode):
            name = call_name(expression)
            arguments = expression.arguments or []
            if name in _RESOURCE_PASSTHROUGH_CALLS and arguments:
                return self.is_resource_expression(arguments[0])
            if name in self.scope.functions:
                return self.is_resource_type(self.scope.functions[name])
            return self.is_resource_type(A.NamedType(name)) if name else False
        if not isinstance(
            expression, (A.IdentifierNode, A.ArrayAccessNode, A.MemberAccessNode)
        ):
            return False
        declared = self.scope.node_type(expression)
        if declared is not None:
            return self.is_resource_type(declared)
        if isinstance(expression, A.IdentifierNode):
            return expression.name in self.resource_names
        if isinstance(expression, A.ArrayAccessNode):
            return self.is_resource_expression(expression.array_expr)
        return expression.member in self.resource_members


def _accepts(function, count):
    parameters = list(function.parameters or [])
    required = sum(
        1
        for parameter in parameters
        if getattr(parameter, "default_value", None) is None
    )
    return required <= count <= len(parameters)


def _rename_call(call, new_name):
    function = call.function
    if isinstance(function, A.IdentifierNode):
        function.name = new_name
        function.identifier = new_name
        return
    call.function = new_name
    call.name = new_name


def _fresh_names(program, names):
    used = _used_names(program)
    renames = {}
    for name in names:
        candidate = f"{name}{_RENAME_SUFFIX}"
        suffix = 2
        while candidate in used:
            candidate = f"{name}{_RENAME_SUFFIX}{suffix}"
            suffix += 1
        used.add(candidate)
        renames[name] = candidate
    return renames


def _used_names(program):
    used = set()
    seen = set()
    stack = [program]
    while stack:
        node = stack.pop()
        if isinstance(node, (list, tuple)):
            stack.extend(node)
            continue
        if isinstance(node, dict):
            stack.extend(node.values())
            continue
        if not isinstance(node, A.ASTNode) or id(node) in seen:
            continue
        seen.add(id(node))
        for field, value in vars(node).items():
            if field in ("parent", "annotations", "source_location"):
                continue
            if field in ("name", "identifier", "member") and isinstance(value, str):
                used.add(value)
            elif isinstance(value, (A.ASTNode, list, tuple, dict)):
                stack.append(value)
    return used


def _user_functions(program):
    for function in getattr(program, "functions", None) or []:
        if isinstance(function, A.FunctionNode) and function.body is not None:
            yield function
    for stage in program_stages(program):
        for function in getattr(stage, "local_functions", None) or []:
            if isinstance(function, A.FunctionNode) and function.body is not None:
                yield function


def _span(location):
    if location is None:
        return None
    if isinstance(location, dict):
        line, column, file = (location.get(key) for key in ("line", "column", "file"))
    else:
        line, column, file = (
            getattr(location, key, None) for key in ("line", "column", "file")
        )
    if not isinstance(line, int):
        return None
    return SourceSpan(
        file, line, column if isinstance(column, int) and column > 0 else 1
    )
