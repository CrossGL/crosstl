"""Undeclared identifiers: a CrossGL program that reads or writes a name it never declares is refused.

A plain name in a function body (``r`` in ``dst[0] = r;``) is a parameter, a local declared earlier in an enclosing
block, a stage or module variable, a constant, a constant-buffer member, an enumerator, a function, a generic
parameter, a type name, or a value CrossGL defines: a ``gl_`` built-in (GLSL reserves the prefix, GLSL 4.60 section
3.7, "Identifiers"), a standard mathematical or floating-point limit constant, or a ray-tracing constant. Any other
name has no meaning. The HLSL, GLSL and Metal importers kept such a name, and the generators then copied it into the
target, where it is at best a compile error and at worst a different variable of the same spelling (a target
built-in or one of the generator's own helpers). :func:`refuse_undeclared_identifiers` raises
:class:`UndeclaredIdentifierError` for the first such name, so the translation fails with a typed diagnostic.

Scopes follow CrossGL, which keeps GLSL's rule that a local's scope begins after its initializer (GLSL 4.60 section
4.2.2, "Scoping"); :mod:`.declaration_scopes` carries that rule to targets with C++ scoping. A ``for`` loop's
initializer is visible in its condition, update and body, a ``switch`` shares one scope across its cases as C++ and
GLSL do, and a lambda sees its parameters and captures.

Only function bodies a stage entry can reach are checked (every function of a program without entries), as for
:mod:`.declared_types`: the generators drop the others. Names with namespace or member qualifiers (``metal::x``,
``Kind::A``) are resolved by the importers and generators themselves, and so are callee names, member names,
attribute arguments, types and ``static_assert`` conditions. Whether a ``gl_`` built-in exists and fits its stage is
checked by :mod:`.builtin_names` and :mod:`.builtin_stage_identifiers`.
"""

from __future__ import annotations

import contextlib
import contextvars

from .. import ast as A
from ..codegen.metal_ray_flags import RAY_FLAG_CONSTANTS
from ..codegen.spirv_ray_builtins import ray_constant_value
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from ..float_limit_constants import float_limit_constant
from ..ray_query_vocabulary import ray_query_constant
from ..refusals import supersedes_pending_refusals
from ..standard_constants import standard_math_constant
from .builtin_type_names import BUILTIN_TYPE_NAMES
from .declared_types import dead_function_ids, declared_type_names
from .rewrite import child_slots, get_child

__all__ = [
    "UndeclaredIdentifierError",
    "deferring_undeclared_identifier_refusals",
    "generate_refusing_undeclared_identifiers",
    "is_typed_diagnostic",
    "record_undeclared_identifiers",
    "recorded_undeclared_identifier",
    "refuse_recorded_undeclared_identifier",
    "refuse_undeclared_identifiers",
    "undeclared_identifier_uses",
]

# The program annotation that holds the refusal :func:`record_undeclared_identifiers` defers.
_RECORDED = "undeclared_identifier_refusal"

# Set by :func:`deferring_undeclared_identifier_refusals`: the list successful generations add their refusal to.
_DEFERRED = contextvars.ContextVar(
    "crosstl_deferred_undeclared_identifiers", default=None
)

# Values every CrossGL program may name without declaring them: the boolean literals, the null pointer the C++ family
# importers keep, the ``discard`` statement some importers keep as an expression statement, and ``None``, the empty
# value of CrossGL's ``Option`` (the Rust importer's ``Option<T>``; ``Some(x)`` is a call).
_LANGUAGE_VALUES = frozenset({"true", "false", "nullptr", "discard", "None"})

# The memory flags, memory orders and thread scopes that barrier and fence calls take as arguments, in Metal's
# spelling, which CrossGL keeps (Metal Shading Language 3.2, sections 6.9.1 "Threadgroup and SIMD-group
# Synchronization Functions" and 6.15.1 "Memory Order" and "Thread Scope"). The importers write them without the
# ``mem_flags::`` qualifier.
_FENCE_ARGUMENTS = frozenset(
    {
        "mem_none",
        "mem_device",
        "mem_threadgroup",
        "mem_texture",
        "mem_threadgroup_imageblock",
        "mem_object_data",
        "memory_order_relaxed",
        "memory_order_acquire",
        "memory_order_release",
        "memory_order_acq_rel",
        "memory_order_seq_cst",
        "thread_scope_thread",
        "thread_scope_simdgroup",
        "thread_scope_threadgroup",
        "thread_scope_device",
        "thread_scope_system",
    }
)


class UndeclaredIdentifierError(TranslationDiagnostic):
    """A plain name a function body uses that is neither declared in scope nor a CrossGL built-in value."""

    def __init__(self, target, name, function=None, *, source_location=None):
        self.source_location = source_location
        self.identifier = name
        self.function_name = function
        where = f" in function '{function}'" if function else ""
        super().__init__(
            f"identifier '{name}'{where} is not declared: it is neither a local, parameter, stage or module "
            "variable, constant, constant-buffer member, enumerator, function or generic parameter of the "
            "program nor a CrossGL built-in value",
            target=target,
            code=f"{CODE_PREFIX}undeclared-identifier",
        )


def _is_builtin_value(name):
    return (
        name in _LANGUAGE_VALUES
        or name in _FENCE_ARGUMENTS
        or name in RAY_FLAG_CONSTANTS
        or name.startswith("gl_")
        or standard_math_constant(name) is not None
        or float_limit_constant(name) is not None
        or ray_query_constant(name) is not None
        or ray_constant_value(name) is not None
    )


def _members(node):
    return getattr(node, "members", None) or []


def _struct_members(program):
    """Member names of every structure, keyed by the structure's name, for unnamed interface blocks."""
    structs = {}
    for struct in getattr(program, "structs", None) or []:
        if isinstance(struct, A.StructNode) and isinstance(struct.name, str):
            structs[struct.name] = [
                member.name
                for member in _members(struct)
                if isinstance(getattr(member, "name", None), str)
            ]
    for stage in _stages(program):
        for struct in getattr(stage, "local_structs", None) or []:
            if isinstance(struct, A.StructNode) and isinstance(struct.name, str):
                structs.setdefault(
                    struct.name,
                    [
                        member.name
                        for member in _members(struct)
                        if isinstance(getattr(member, "name", None), str)
                    ],
                )
    return structs


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values()) if isinstance(stages, dict) else list(stages)


def _attribute_arguments(node, name):
    for attribute in getattr(node, "attributes", None) or []:
        if getattr(attribute, "name", None) == name:
            return [
                str(
                    getattr(argument, "name", None)
                    or getattr(argument, "value", argument)
                )
                for argument in getattr(attribute, "arguments", None) or []
            ]
    return None


def _interface_block_names(program):
    """The names GLSL interface blocks declare: an instance name, or the members of a block without one.

    The GLSL importer keeps a block as a structure carrying its qualifiers (``@glsl_interface_block(out)
    @glsl_interface_instance(vout) struct Block``, :mod:`.interface_block_instances`); without an instance name the
    members are visible as bare names (GLSL 4.60 section 4.3.9, "Interface Blocks").
    """
    names = set()
    structs = list(getattr(program, "structs", None) or [])
    for stage in _stages(program):
        structs += list(getattr(stage, "local_structs", None) or [])
    for struct in structs:
        if _attribute_arguments(struct, "glsl_interface_block") is None:
            continue
        instance = _attribute_arguments(struct, "glsl_interface_instance")
        if instance:
            names.update(name for name in instance if name)
        else:
            names.update(
                member.name
                for member in _members(struct)
                if isinstance(getattr(member, "name", None), str)
            )
    return names


def _type_name(type_node):
    while isinstance(type_node, A.ArrayType):
        type_node = type_node.element_type
    name = getattr(type_node, "name", None)
    return name if isinstance(name, str) else None


def _variable_names(variables, structs):
    """The names a list of module or stage variables declares, with the members of unnamed blocks.

    The members of a variable of structure type are visible as bare names, as :class:`~.hub_types.TypeScope` reads
    them: that is how an interface block without an instance name declares its members (GLSL 4.60 section 4.3.9,
    "Interface Blocks").
    """
    names = set()
    for variable in variables or []:
        name = getattr(variable, "name", None)
        if not isinstance(name, str):
            continue
        names.add(name)
        names.update(structs.get(_type_name(getattr(variable, "var_type", None)), ()))
    return names


def _cbuffer_names(cbuffers):
    names = set()
    for cbuffer in cbuffers or []:
        name = getattr(cbuffer, "name", None)
        if isinstance(name, str):
            names.add(name)
        names.update(
            member.name
            for member in _members(cbuffer)
            if isinstance(getattr(member, "name", None), str)
        )
    return names


def _function_names(functions):
    return {
        function.name
        for function in functions or []
        if isinstance(function, A.FunctionNode) and isinstance(function.name, str)
    }


def _enumerators(program):
    names = set()
    for node in _walk_all(program):
        if isinstance(node, A.EnumVariantNode) and isinstance(node.name, str):
            names.add(node.name)
    return names


def _walk_all(node):
    seen = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if not isinstance(current, A.ASTNode) or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for field, key in child_slots(current):
            stack.append(get_child(current, field, key))


def _module_names(program, structs):
    names = set(declared_type_names(program)) | _enumerators(program)
    names.update(
        constant.name
        for constant in getattr(program, "constants", None) or []
        if isinstance(getattr(constant, "name", None), str)
    )
    names |= _variable_names(getattr(program, "global_variables", None), structs)
    names |= _cbuffer_names(getattr(program, "cbuffers", None))
    names |= _function_names(getattr(program, "functions", None))
    names |= _interface_block_names(program)
    return names


def _stage_names(stage, structs):
    """The names a stage declares for the functions in it and for the module functions it calls.

    The entry point's parameters are among them, and so are the locals at the top of its body that hold its return
    type: the GLSL importer gathers a stage's inputs into the entry's ``input`` parameter and its outputs into an
    ``output`` local, and leaves the helpers that use them as ``input.uv`` and ``output.color``, which the
    generators resolve.
    """
    names = _variable_names(getattr(stage, "local_variables", None), structs)
    names |= _cbuffer_names(getattr(stage, "local_cbuffers", None))
    names |= _function_names(getattr(stage, "local_functions", None))
    entry = getattr(stage, "entry_point", None)
    if isinstance(entry, A.FunctionNode) and isinstance(entry.name, str):
        names.add(entry.name)
        names.update(
            parameter.name
            for parameter in entry.parameters or []
            if isinstance(getattr(parameter, "name", None), str)
        )
        returned = _type_name(getattr(entry, "return_type", None))
        body = getattr(entry, "body", None)
        statements = (
            getattr(body, "statements", None) if isinstance(body, A.BlockNode) else body
        )
        for statement in statements or []:
            if (
                isinstance(statement, A.VariableNode)
                and isinstance(statement.name, str)
                and returned is not None
                and _type_name(statement.var_type) == returned
            ):
                names.add(statement.name)
    return names


def _pattern_names(pattern):
    """Every name a match pattern may bind: identifier patterns and the field names of structure patterns."""
    names = set()
    for node in _walk_all(pattern):
        if isinstance(node, A.IdentifierPatternNode) and isinstance(node.name, str):
            names.add(node.name)
        elif isinstance(node, A.StructPatternNode):
            fields = node.field_patterns
            if isinstance(fields, dict):
                names.update(name for name in fields if isinstance(name, str))
            elif isinstance(fields, (list, tuple)):
                for item in fields:
                    if (
                        isinstance(item, (list, tuple))
                        and item
                        and isinstance(item[0], str)
                    ):
                        names.add(item[0])
    return names


def _for_in_names(pattern):
    if isinstance(pattern, str):
        return {part.strip() for part in pattern.strip("()").split(",") if part.strip()}
    if isinstance(pattern, A.ASTNode):
        return _pattern_names(pattern)
    return set()


class _Resolver:
    """Walks one function body with the names in scope and records the plain names it cannot resolve."""

    def __init__(self, outer, function):
        self.frames = [set(outer)]
        self.function = function
        self.uses = []

    def declared(self, name):
        return any(name in frame for frame in self.frames)

    def declare(self, names):
        self.frames[-1].update(name for name in names if isinstance(name, str))

    def run(self):
        function = self.function
        self.frames.append(set())
        self.declare(
            getattr(parameter, "name", None) for parameter in function.parameters or []
        )
        self.declare(
            getattr(param, "name", None)
            for param in getattr(function, "generic_params", None) or []
        )
        self.walk(function.body)
        self.frames.pop()
        return self.uses

    def scoped(self, nodes):
        self.frames.append(set())
        for node in nodes:
            self.walk(node)
        self.frames.pop()

    def walk_children(self, node, skip=()):
        for field, key in list(child_slots(node)):
            if field in skip:
                continue
            self.walk(get_child(node, field, key))

    def walk(self, node):
        if isinstance(node, (list, tuple)):
            # A ``for`` header keeps several declarations or updates as a list (``int a = 0, b = n``), in order.
            for item in node:
                self.walk(item)
            return
        if not isinstance(node, A.ASTNode):
            return
        if isinstance(
            node,
            (
                A.TypeNode,
                A.AttributeNode,
                A.FunctionNode,
                A.StructNode,
                A.StaticAssertNode,
                A.PreprocessorNode,
                A.LayoutQualifierNode,
            ),
        ):
            return
        if isinstance(node, A.IdentifierNode):
            self.check(node)
        elif isinstance(node, A.BlockNode):
            self.scoped(node.statements or [])
        elif isinstance(node, A.VariableNode):
            self.walk_children(node, skip=("attributes", "var_type"))
            self.declare([node.name])
        elif isinstance(node, A.ForNode):
            self.scoped([node.init, node.condition, node.update, node.body])
        elif isinstance(node, A.ForInNode):
            self.walk(node.iterable)
            self.frames.append(_for_in_names(node.pattern))
            self.walk(node.body)
            self.frames.pop()
        elif isinstance(node, A.SwitchNode):
            self.walk(node.expression)
            self.frames.append(set())
            for case in list(node.cases or []) + [node.default_case]:
                if isinstance(case, A.CaseNode):
                    self.walk(case.value)
                    for statement in case.statements or []:
                        self.walk(statement)
                else:
                    self.walk(case)
            self.frames.pop()
        elif isinstance(node, A.MatchArmNode):
            self.frames.append(_pattern_names(node.pattern))
            self.walk(node.guard)
            self.walk(node.body)
            self.frames.pop()
        elif isinstance(node, A.LambdaNode):
            self.frames.append(set())
            self.declare(
                getattr(parameter, "name", None) for parameter in node.parameters or []
            )
            self.declare(node.captures or [])
            self.walk(node.body)
            self.frames.pop()
        elif isinstance(node, A.FunctionCallNode):
            callee = node.function
            if isinstance(callee, (A.MemberAccessNode, A.PointerAccessNode)):
                self.walk(
                    getattr(callee, "object_expr", None)
                    or getattr(callee, "object", None)
                )
            elif not isinstance(callee, (A.IdentifierNode, str)):
                self.walk(callee)
            for argument in list(node.arguments or []) + list(node.generic_args or []):
                self.walk(argument)
        elif isinstance(node, (A.MemberAccessNode, A.PointerAccessNode)):
            self.walk(
                getattr(node, "object_expr", None) or getattr(node, "object", None)
            )
        elif isinstance(node, (A.ConstructorNode, A.CastNode)):
            self.walk_children(
                node, skip=("constructor_type", "target_type", "expression_type")
            )
        else:
            self.walk_children(node)

    def check(self, node):
        name = node.name
        if not isinstance(name, str) or not name:
            return
        if "::" in name or "." in name or "<" in name:
            return
        if self.declared(name) or name in BUILTIN_TYPE_NAMES or _is_builtin_value(name):
            return
        self.uses.append((name, node, getattr(self.function, "name", None)))


def _checked_functions(program):
    """``(function, stage)`` for every function body in source order: module functions, then each stage's."""
    for function in getattr(program, "functions", None) or []:
        if isinstance(function, A.FunctionNode):
            yield function, None
    for stage in _stages(program):
        for function in getattr(stage, "local_functions", None) or []:
            if isinstance(function, A.FunctionNode):
                yield function, stage
        entry = getattr(stage, "entry_point", None)
        if isinstance(entry, A.FunctionNode):
            yield entry, stage


def undeclared_identifier_uses(program, target=None):
    """``(name, node, function_name)`` for each plain name a live function body uses without a declaration in scope.

    ``target`` is accepted for symmetry with the other refusal passes; the rule does not depend on it.
    """
    del target
    structs = _struct_members(program)
    module = _module_names(program, structs)
    every_stage = set()
    stage_names = {}
    for stage in _stages(program):
        stage_names[id(stage)] = _stage_names(stage, structs)
        # A module function may be called from any stage, and importers place helpers there that read the
        # variables of the stage that calls them.
        every_stage |= stage_names[id(stage)]
    dead = dead_function_ids(program)
    uses = []
    seen = set()
    for function, stage in _checked_functions(program):
        if (
            id(function) in seen
            or id(function) in dead
            or getattr(function, "body", None) is None
        ):
            continue
        seen.add(id(function))
        outer = module | (stage_names[id(stage)] if stage is not None else every_stage)
        uses.extend(_Resolver(outer, function).run())
    return uses


def refuse_undeclared_identifiers(program, target):
    """Raise :class:`UndeclaredIdentifierError` for the first undeclared plain name in a live function body."""
    for name, node, function in undeclared_identifier_uses(program, target):
        raise UndeclaredIdentifierError(
            target,
            name,
            function,
            source_location=getattr(node, "source_location", None),
        )


def record_undeclared_identifiers(program, target):
    """Record on ``program`` the refusal of its first undeclared plain name, for generation to raise.

    The refusal is held back so that a later pass or the generator can refuse the same program with a diagnostic of
    its own that names the construct it cannot lower (an unresolved generic member call, a host construct an
    importer left in place). The annotation is a plain tuple, so copies of the program keep it. A program without
    annotations is refused at once. Returns the recorded error or ``None``.
    """
    annotations = getattr(program, "annotations", None)
    if not isinstance(annotations, dict):
        refuse_undeclared_identifiers(program, target)
        return None
    annotations.pop(_RECORDED, None)
    for name, node, function in undeclared_identifier_uses(program, target):
        annotations[_RECORDED] = (
            target,
            name,
            function,
            getattr(node, "source_location", None),
        )
        return recorded_undeclared_identifier(program)
    return None


def recorded_undeclared_identifier(program):
    """The :class:`UndeclaredIdentifierError` recorded on ``program``, if any."""
    annotations = getattr(program, "annotations", None)
    record = annotations.get(_RECORDED) if isinstance(annotations, dict) else None
    if record is None:
        return None
    target, name, function, location = record
    return UndeclaredIdentifierError(target, name, function, source_location=location)


def refuse_recorded_undeclared_identifier(program, cause=None):
    """Raise the recorded :class:`UndeclaredIdentifierError`, chained to ``cause``; do nothing when none is recorded."""
    error = recorded_undeclared_identifier(program)
    if error is not None:
        raise error from cause


def is_typed_diagnostic(error) -> bool:
    """Whether ``error`` is a refusal that names its own construct, which takes precedence over the recorded one.

    A site refusal (:class:`~crosstl.translator.refusals.TranslationRefusal`) does not: a generator raises one when
    it meets the undeclared name, so the recorded refusal is the better diagnostic.
    """
    return supersedes_pending_refusals(error)


def generate_refusing_undeclared_identifiers(program, generate, *, refuse_after=True):
    """Return ``generate()``, refusing ``program``'s recorded undeclared name unless the generator refuses first.

    A typed diagnostic from ``generate`` propagates. Any other exception is replaced by the recorded refusal,
    because that is how a generator fails on a name with no declaration. With ``refuse_after`` false the caller
    raises the refusal itself after its own checks of the output.
    """
    if recorded_undeclared_identifier(program) is None:
        return generate()
    try:
        code = generate()
    except Exception as error:
        if is_typed_diagnostic(error):
            raise
        refuse_recorded_undeclared_identifier(program, error)
        raise
    deferred = _DEFERRED.get()
    if deferred is not None:
        deferred.append(recorded_undeclared_identifier(program))
    elif refuse_after:
        refuse_recorded_undeclared_identifier(program)
    return code


@contextlib.contextmanager
def deferring_undeclared_identifier_refusals():
    """Within the block, a generation that succeeds despite a recorded undeclared name returns its output and adds
    the refusal to the yielded list instead of raising it.

    For a caller that checks the output for constructs an importer left in place before it refuses the artifact;
    the caller must raise the refusal itself.
    """
    deferred = []
    token = _DEFERRED.set(deferred)
    try:
        yield deferred
    finally:
        _DEFERRED.reset(token)
