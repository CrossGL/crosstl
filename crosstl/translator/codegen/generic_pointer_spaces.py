"""Give each CrossGL generic pointer ``ptr<T>`` the address space it points into, specializing functions per call.

CrossGL spells a pointer without an address space ``ptr<T>``. The CUDA and HIP importers produce it for every pointer
of device code, because a CUDA pointer is generic: it may address global, shared, local or constant memory, and the
space is resolved when the pointer is dereferenced (PTX ISA, "Generic Addressing"). A Metal pointer type names its
address space (MSL 4.0 section 4, "Address Spaces"), and HLSL and GLSL have no pointers at all: their generators lower a
pointer into a view of the buffer, workgroup array or local array it points into, which needs to know that object. The
GLSL, HLSL, Metal and Slang generators copied ``ptr<float>`` into their output unchanged, which no compiler for those
targets accepts.

:func:`assign_generic_pointer_spaces` infers each space from the program, as compilers of OpenCL's generic address
space do for targets without one. Every function an entry point reaches is specialized for the spaces its ``ptr<T>``
parameters receive, starting from the entry points:

- A pointer argument points into the memory of the declaration it is derived from: a storage buffer is ``device``
  memory, a uniform, constant block member or module constant is ``constant`` memory, a workgroup variable is
  ``threadgroup`` memory, and a local, a parameter or a module-scope private variable is ``thread`` memory. An array
  decays to a pointer into its own memory, ``&x`` points into the memory of ``x``, and a typed pointer into the space
  it names.
- Pointer arithmetic, casts and conditional expressions keep the space. A ``ptr<T>`` local takes the space of its
  initializer and of every value assigned to it, and a function returning ``ptr<T>`` the space of its return values.
- Read-only memory (a read-only storage buffer or constant memory) makes the parameter ``const``, and a write through
  such a pointer is refused.

The specialized parameter, local, cast or return type becomes the typed pointer ``PointerType(T, space)`` the
generators already lower, as the CrossGL spelling ``device T* p`` parses. A function called with one combination of
spaces keeps its name; one called with several gets one copy per combination, named after the spaces, and each call is
renamed to its copy.

A pointer whose space cannot be derived (one read from a structure member, an array of pointers or an opaque call), a
local or conditional expression that points into two spaces, recursion, an overloaded call the arity does not resolve,
a pointer to a generic pointer and an entry point with a generic pointer parameter are refused through ``refuse``. A
function no entry point reaches is dropped when it uses a generic pointer, or calls a function that does, because
nothing fixes its spaces and the target's pipeline cannot call it (a CUDA host launcher, for example); a program
without an entry point that uses one is refused. An unused ``ptr<T>`` local that nothing initializes or assigns is
removed. :func:`refuse_written_generic_pointers` refuses any ``ptr<T>`` an output still spells.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import NamedTuple

from .. import ast as A
from ..semantics.hub_types import call_name
from ..semantics.local_bindings import local_bindings
from ..semantics.rewrite import child_slots, clone, get_child
from .generic_pointer_deduction import structured_buffer_element_type
from .stage_utils import STAGE_QUALIFIER_NAMES, function_stage_name

__all__ = [
    "assign_generic_pointer_spaces",
    "function_body_nodes",
    "is_generic_pointer_type",
    "program_function_nodes",
    "refuse_written_generic_pointers",
    "written_generic_pointer_spellings",
]

# Declaration qualifiers and pointer address spaces, by the address space they name.
_SPACE_OF_QUALIFIER = {
    "device": "device",
    "global": "device",
    "storage": "device",
    "buffer": "device",
    "constant": "constant",
    "uniform": "constant",
    "threadgroup": "threadgroup",
    "workgroup": "threadgroup",
    "shared": "threadgroup",
    "groupshared": "threadgroup",
    "thread": "thread",
    "private": "thread",
    "function": "thread",
}
_READONLY_QUALIFIERS = frozenset({"read", "readonly", "const"})
_WRITABLE_QUALIFIERS = frozenset({"read_write", "readwrite", "write", "writeonly"})
_POINTER_STEPS = frozenset({"+", "-"})
_INCREMENTS = frozenset({"++", "--"})


class Space(NamedTuple):
    """The address space a pointer points into, and whether the memory there is read-only."""

    name: str
    readonly: bool = False

    @property
    def tag(self):
        return f"const_{self.name}" if self.readonly else self.name


def is_generic_pointer_type(type_node):
    """Whether ``type_node`` is the CrossGL generic pointer ``ptr<T>``."""
    return (
        isinstance(type_node, A.NamedType)
        and type_node.name == "ptr"
        and len(type_node.generic_args or ()) == 1
    )


def _contains_generic_pointer(type_node):
    if type_node is None or isinstance(type_node, str):
        return False
    if is_generic_pointer_type(type_node):
        return True
    for attribute in ("pointee_type", "element_type", "referenced_type"):
        if _contains_generic_pointer(getattr(type_node, attribute, None)):
            return True
    return any(
        _contains_generic_pointer(argument)
        for argument in getattr(type_node, "generic_args", None) or ()
        if isinstance(argument, A.TypeNode)
    )


def _declared_type(declaration):
    for attribute in ("param_type", "var_type", "const_type", "member_type"):
        declared = getattr(declaration, attribute, None)
        if declared is not None:
            return declared
    return None


def _qualifier_names(declaration):
    names = set()
    for attribute in ("qualifiers", "resource_qualifiers", "address_space_qualifiers"):
        names.update(
            str(q).lower() for q in getattr(declaration, attribute, None) or []
        )
    declared = _declared_type(declaration)
    names.update(str(q).lower() for q in getattr(declared, "qualifiers", None) or [])
    return names


def _space_named(names):
    spaces = {
        _SPACE_OF_QUALIFIER[name] for name in names if name in _SPACE_OF_QUALIFIER
    }
    return spaces.pop() if len(spaces) == 1 else None


def _is_readonly(names):
    return bool(names & _READONLY_QUALIFIERS) and not names & _WRITABLE_QUALIFIERS


def _walk(node):
    """Pre-order walk of the structural nodes under ``node``, not entering nested functions."""
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(reversed(current))
            continue
        if not isinstance(current, A.ASTNode):
            continue
        yield current
        if isinstance(current, (A.FunctionNode, A.LambdaNode)) and current is not node:
            continue
        children = [
            get_child(current, field, key) for field, key in child_slots(current)
        ]
        stack.extend(reversed(children))


def _body_nodes(function):
    return _walk(getattr(function, "body", None))


def _parameters(function):
    return list(getattr(function, "parameters", None) or [])


def _has_generic_parameters(function):
    return any(
        is_generic_pointer_type(_declared_type(p)) for p in _parameters(function)
    )


def _mentions_generic_pointer(function):
    if any(_contains_generic_pointer(_declared_type(p)) for p in _parameters(function)):
        return True
    if _contains_generic_pointer(getattr(function, "return_type", None)):
        return True
    for node in _body_nodes(function):
        if isinstance(node, A.VariableNode) and _contains_generic_pointer(
            node.var_type
        ):
            return True
        if isinstance(
            node, (A.CastNode, A.PointerReinterpretNode)
        ) and _contains_generic_pointer(node.target_type):
            return True
    return False


def _typed_pointer(generic, space):
    return A.PointerType(
        generic.generic_args[0],
        address_space=space.name,
        access_mode="read" if space.readonly else None,
    )


def _set_call_name(call, name):
    identifier = A.IdentifierNode(name)
    call.function = identifier
    call.name = identifier


class _Spec:
    """One specialization of a function for the spaces its generic pointer parameters receive."""

    def __init__(self, original, function, spaces, stage, is_entry):
        self.original = original
        self.function = function
        self.spaces = spaces
        self.stage = stage
        self.is_entry = is_entry
        self.return_space = None
        self.calls = []
        self.in_progress = True


class _Context:
    """Resolves the identifiers of one specialization and the address spaces its expressions point into."""

    def __init__(self, program, spec):
        self.program = program
        self.spec = spec
        self.function = spec.function
        self.bindings = local_bindings(getattr(spec.function, "body", None))
        self.parameters = {p.name: p for p in _parameters(spec.function)}

    def resolve(self, identifier):
        bound = self.bindings.get(id(identifier))
        if bound is not None:
            return bound[1]
        name = identifier.name
        if name in self.parameters:
            return self.parameters[name]
        return self.program.module_declaration(name, self.spec.stage)

    def where(self, declaration):
        where = self.program.where.get(id(declaration))
        if where is not None:
            return where
        if self.parameters.get(getattr(declaration, "name", None)) is declaration:
            return "entry_parameter" if self.spec.is_entry else "parameter"
        return "local"

    def kind(self, declaration):
        declared = _declared_type(declaration)
        if is_generic_pointer_type(declared) or isinstance(declared, A.PointerType):
            return "pointer"
        if isinstance(declared, A.ReferenceType):
            return "reference"
        if isinstance(declared, A.ArrayType):
            return "array"
        if isinstance(declared, A.NamedType) and (
            declared.name == "array"
            or structured_buffer_element_type(self.program.type_text(declared))
        ):
            return "array"
        return "value"

    def declaration_storage(self, declaration):
        """The space the object ``declaration`` declares lives in."""
        where = self.where(declaration)
        if where == "constant":
            return Space("constant", True)
        names = _qualifier_names(declaration)
        declared = _declared_type(declaration)
        if isinstance(declared, A.ReferenceType):
            names.add(str(getattr(declared, "address_space", "") or "").lower())
        space = _space_named(names)
        if space is None and isinstance(declared, A.NamedType):
            element = structured_buffer_element_type(self.program.type_text(declared))
            if element is not None:
                space = "device"
                names.add(
                    "read" if declared.name == "StructuredBuffer" else "read_write"
                )
        if space is None and where == "global" and "const" in names:
            space = "constant"
        if space is None:
            space = "thread"
        return Space(space, _is_readonly(names) or space == "constant")

    def declaration_pointee(self, declaration):
        """The space the pointer ``declaration`` declares points into."""
        declared = _declared_type(declaration)
        if is_generic_pointer_type(declared):
            return self.program.spaces.get(id(declaration))
        if not isinstance(declared, A.PointerType):
            return None
        names = _qualifier_names(declaration)
        address_space = str(getattr(declared, "address_space", "") or "").lower()
        if address_space:
            if address_space not in _SPACE_OF_QUALIFIER:
                return None
            names.add(address_space)
        names.add(str(getattr(declared, "access_mode", "") or "").lower())
        space = _space_named(names) or "thread"
        return Space(space, _is_readonly(names) or space == "constant")

    def points(self, expression):
        """Whether ``expression`` is a pointer value rather than an object."""
        if isinstance(expression, A.IdentifierNode):
            declaration = self.resolve(expression)
            return declaration is not None and self.kind(declaration) == "pointer"
        if (
            isinstance(expression, A.BinaryOpNode)
            and expression.operator in _POINTER_STEPS
        ):
            return self.points(expression.left) or self.points(expression.right)
        if isinstance(expression, A.UnaryOpNode):
            if expression.operator == "&":
                return True
            return expression.operator in _INCREMENTS and self.points(
                expression.operand
            )
        if isinstance(expression, (A.CastNode, A.PointerReinterpretNode)):
            target = expression.target_type
            return is_generic_pointer_type(target) or isinstance(target, A.PointerType)
        if isinstance(expression, A.TernaryOpNode):
            return self.points(expression.true_expr)
        if isinstance(expression, A.FunctionCallNode):
            callee = self.program.callee_of(expression, self.spec.stage)
            return callee is not None and (
                is_generic_pointer_type(callee.return_type)
                or isinstance(callee.return_type, A.PointerType)
            )
        return False

    def storage(self, expression, tentative=False):
        """The space the object ``expression`` designates lives in, or ``None`` when it cannot be derived."""
        if isinstance(expression, A.IdentifierNode):
            declaration = self.resolve(expression)
            if declaration is None:
                return None
            if self.kind(declaration) == "pointer":
                return Space("thread")
            return self.declaration_storage(declaration)
        if isinstance(expression, A.ArrayAccessNode):
            base = expression.array_expr
            if self.points(base):
                return self.pointee(base, tentative)
            return self.storage(base, tentative)
        if isinstance(expression, A.MemberAccessNode):
            base = expression.object_expr
            if self.points(base):
                return self.pointee(base, tentative)
            return self.storage(base, tentative)
        if isinstance(expression, A.PointerAccessNode):
            return self.pointee(expression.pointer_expr, tentative)
        if isinstance(expression, A.SwizzleNode):
            return self.storage(expression.vector_expr, tentative)
        if isinstance(expression, A.UnaryOpNode) and expression.operator == "*":
            return self.pointee(expression.operand, tentative)
        return None

    def pointee(self, expression, tentative=False):
        """The space the pointer value ``expression`` points into, or ``None`` when it cannot be derived."""
        if isinstance(expression, A.IdentifierNode):
            declaration = self.resolve(expression)
            if declaration is None:
                return None
            kind = self.kind(declaration)
            if kind == "pointer":
                return self.declaration_pointee(declaration)
            if kind == "array":
                return self.declaration_storage(declaration)
            return None
        if (
            isinstance(expression, A.BinaryOpNode)
            and expression.operator in _POINTER_STEPS
        ):
            left = self.points(expression.left)
            right = self.points(expression.right)
            if left and not right:
                return self.pointee(expression.left, tentative)
            if right and not left and expression.operator == "+":
                return self.pointee(expression.right, tentative)
            if not (left or right):
                left_space = self.pointee(expression.left, tentative)
                right_space = self.pointee(expression.right, tentative)
                if (left_space is None) != (right_space is None):
                    return left_space or right_space
            return None
        if isinstance(expression, A.UnaryOpNode):
            if expression.operator == "&":
                return self.storage(expression.operand, tentative)
            if expression.operator in _INCREMENTS:
                return self.pointee(expression.operand, tentative)
            return None
        if isinstance(expression, (A.CastNode, A.PointerReinterpretNode)):
            return self.cast_pointee(expression, tentative)
        if isinstance(expression, A.TernaryOpNode):
            return self.either_pointee(expression, tentative)
        if isinstance(expression, A.AssignmentNode):
            return self.pointee(expression.value, tentative)
        if isinstance(expression, A.CommaExpressionNode):
            expressions = getattr(expression, "expressions", None) or []
            return self.pointee(expressions[-1], tentative) if expressions else None
        if isinstance(expression, A.FunctionCallNode):
            spec = self.program.call_spec(expression, self, tentative)
            return None if spec is None else spec.return_space
        return None

    def cast_pointee(self, expression, tentative):
        target = expression.target_type
        if isinstance(target, A.PointerType) and getattr(target, "address_space", None):
            space = _space_named({str(target.address_space).lower()})
            return None if space is None else Space(space, space == "constant")
        return self.pointee(expression.expression, tentative)

    def either_pointee(self, expression, tentative):
        first = self.pointee(expression.true_expr, tentative)
        second = self.pointee(expression.false_expr, tentative)
        if first is None or second is None:
            return None
        if first.name != second.name:
            if not tentative:
                self.program.refuse(
                    f"generic pointer: a conditional expression in '{self.function.name}' "
                    f"points into {first.name} or {second.name} memory"
                )
            return None
        return Space(first.name, first.readonly or second.readonly)


class _Program:
    """The whole-program specialization of the functions that use generic pointers."""

    def __init__(self, ast, refuse):
        self.ast = ast
        self.refuse = refuse
        self.spaces = {}
        self.where = {}
        self.specs = {}
        self.order = []
        self.call_specs = {}
        self.module = {}
        self.stage_scopes = {}
        self.containers = (
            [(ast.functions, None)] if isinstance(ast.functions, list) else []
        )
        self.entries = []
        for constant in getattr(ast, "constants", None) or []:
            self.declare_module(self.module, constant, "constant")
        for variable in getattr(ast, "global_variables", None) or []:
            self.declare_module(self.module, variable, "global")
        for cbuffer in getattr(ast, "cbuffers", None) or []:
            for member in getattr(cbuffer, "members", None) or []:
                self.declare_module(self.module, member, "constant")
        for stage in self.stages():
            scope = {}
            for variable in getattr(stage, "local_variables", None) or []:
                self.declare_module(scope, variable, "global")
            for cbuffer in getattr(stage, "local_cbuffers", None) or []:
                for member in getattr(cbuffer, "members", None) or []:
                    self.declare_module(scope, member, "constant")
            self.stage_scopes[id(stage)] = scope
            if isinstance(getattr(stage, "local_functions", None), list):
                self.containers.append((stage.local_functions, stage))
            entry = getattr(stage, "entry_point", None)
            if isinstance(entry, A.FunctionNode):
                self.entries.append((entry, stage))
        self.stage_of = {}
        for container, stage in self.containers:
            for function in container:
                if not isinstance(function, A.FunctionNode):
                    continue
                self.stage_of[id(function)] = stage
                if function_stage_name(function) in STAGE_QUALIFIER_NAMES:
                    self.entries.append((function, stage))

    def stages(self):
        stages = getattr(self.ast, "stages", None) or {}
        if isinstance(stages, A.StageMap):
            return [s for key in stages.keys() for s in stages.all_for_stage(key)]
        return list(stages.values()) if isinstance(stages, dict) else list(stages)

    def declare_module(self, scope, declaration, where):
        name = getattr(declaration, "name", None)
        if name is None:
            return
        scope[name] = declaration
        self.where[id(declaration)] = where

    def module_declaration(self, name, stage):
        if stage is not None:
            declaration = self.stage_scopes.get(id(stage), {}).get(name)
            if declaration is not None:
                return declaration
        return self.module.get(name)

    def type_text(self, type_node):
        name = getattr(type_node, "name", "")
        arguments = getattr(type_node, "generic_args", None) or []
        if not arguments:
            return str(name)
        return f"{name}<{', '.join(self.type_text(a) for a in arguments)}>"

    def functions_named(self, name, stage):
        if stage is not None:
            local = [
                f
                for f in getattr(stage, "local_functions", None) or []
                if isinstance(f, A.FunctionNode) and f.name == name
            ]
            if local:
                return local
        return [
            f
            for f in getattr(self.ast, "functions", None) or []
            if isinstance(f, A.FunctionNode) and f.name == name
        ]

    def callee_of(self, call, stage):
        arguments = list(getattr(call, "arguments", None) or [])
        candidates = [
            f
            for f in self.functions_named(call_name(call), stage)
            if len(_parameters(f)) == len(arguments)
        ]
        return candidates[0] if len(candidates) == 1 else None

    def call_spec(self, call, context, tentative=False):
        """The specialization ``call`` reaches, specializing the callee for the spaces of its pointer arguments."""
        if id(call) in self.call_specs:
            return self.call_specs[id(call)]
        name = call_name(call)
        stage = context.spec.stage
        candidates = self.functions_named(name, stage) if name else []
        if not candidates:
            return None
        arguments = list(getattr(call, "arguments", None) or [])
        matching = [f for f in candidates if len(_parameters(f)) == len(arguments)]
        if len(matching) != 1:
            if tentative:
                return None
            self.call_specs[id(call)] = None
            if any(_mentions_generic_pointer(f) for f in candidates):
                self.refuse(
                    f"generic pointer: the call to '{name}' in '{context.function.name}' "
                    "matches no single overload, so the spaces of its pointers are unknown"
                )
            else:
                for function in candidates:
                    self.specialize(
                        function, (), self.stage_of.get(id(function), stage)
                    )
            return None
        callee = matching[0]
        spaces = []
        for index, (parameter, argument) in enumerate(
            zip(_parameters(callee), arguments)
        ):
            if not is_generic_pointer_type(_declared_type(parameter)):
                continue
            space = context.pointee(argument, tentative)
            if space is None:
                if tentative:
                    return None
                self.call_specs[id(call)] = None
                self.refuse(
                    f"generic pointer: the address space that argument {index + 1} of the "
                    f"call to '{name}' in '{context.function.name}' points into is unknown"
                )
                return None
            spaces.append(space)
        spec = self.specialize(
            callee, tuple(spaces), self.stage_of.get(id(callee), stage)
        )
        self.call_specs[id(call)] = spec
        if spec is not None:
            context.spec.calls.append((call, spec))
        return spec

    def specialize(self, function, spaces, stage, is_entry=False):
        key = (id(function), spaces)
        spec = self.specs.get(key)
        if spec is not None:
            if spec.in_progress and spaces:
                self.refuse(
                    f"generic pointer: '{function.name}' calls itself with a generic pointer, "
                    "which a target without generic pointers cannot specialize"
                )
                return None
            return spec
        specialized = clone(function) if _has_generic_parameters(function) else function
        spec = _Spec(function, specialized, spaces, stage, is_entry)
        self.specs[key] = spec
        self.order.append(spec)
        pending = list(spaces)
        for parameter in _parameters(specialized):
            declared = _declared_type(parameter)
            if not is_generic_pointer_type(declared):
                continue
            space = pending.pop(0)
            if _contains_generic_pointer(declared.generic_args[0]):
                self.refuse(
                    f"generic pointer: '{function.name}.{parameter.name}' points to a "
                    "generic pointer, whose address space is unknown"
                )
            parameter.param_type = _typed_pointer(declared, space)
            qualifiers = [
                q for q in parameter.qualifiers or [] if str(q).lower() != "const"
            ]
            parameter.qualifiers = (
                qualifiers + [space.name] + (["const"] if space.readonly else [])
            )
            self.spaces[id(parameter)] = space
        self.process(spec)
        spec.in_progress = False
        return spec

    def process(self, spec):
        context = _Context(self, spec)
        self.resolve_pointer_locals(context)
        for node in _body_nodes(spec.function):
            if isinstance(node, A.FunctionCallNode):
                self.call_spec(node, context)
            elif isinstance(
                node, (A.CastNode, A.PointerReinterpretNode)
            ) and is_generic_pointer_type(node.target_type):
                space = context.cast_pointee(node, False)
                if space is None:
                    self.refuse(
                        f"generic pointer: the address space a pointer cast in "
                        f"'{spec.function.name}' points into is unknown"
                    )
                else:
                    node.target_type = _typed_pointer(node.target_type, space)
        self.refuse_readonly_writes(context)
        self.resolve_return(context)

    def resolve_pointer_locals(self, context):
        body = getattr(context.function, "body", None)
        pointer_locals = [
            node
            for node in _walk(body)
            if isinstance(node, A.VariableNode)
            and is_generic_pointer_type(node.var_type)
        ]
        if not pointer_locals:
            return
        sources = {id(local): [] for local in pointer_locals}
        uses = {id(local): 0 for local in pointer_locals}
        for local in pointer_locals:
            if local.initial_value is not None:
                sources[id(local)].append(local.initial_value)
        for node in _walk(body):
            if isinstance(node, A.IdentifierNode):
                bound = context.bindings.get(id(node))
                if bound is not None and id(bound[1]) in uses:
                    uses[id(bound[1])] += 1
            elif isinstance(node, A.AssignmentNode) and isinstance(
                node.target, A.IdentifierNode
            ):
                bound = context.bindings.get(id(node.target))
                if (
                    bound is not None
                    and id(bound[1]) in sources
                    and node.operator in (None, "=")
                ):
                    sources[id(bound[1])].append(node.value)
        pending = list(pointer_locals)
        progress = True
        while pending and progress:
            progress = False
            for local in list(pending):
                if self.resolve_pointer_local(context, local, sources[id(local)]):
                    pending.remove(local)
                    progress = True
        for local in pending:
            if not sources[id(local)] and not uses[id(local)]:
                self.remove_declaration(body, local)
                continue
            self.refuse(
                f"generic pointer: the address space that '{context.function.name}.{local.name}' "
                "points into is unknown"
            )
        for local in pointer_locals:
            space = self.spaces.get(id(local))
            if space is not None:
                local.var_type = _typed_pointer(local.var_type, space)
                local.vtype = local.var_type
                qualifiers = [
                    q for q in local.qualifiers or [] if str(q).lower() != "const"
                ]
                local.qualifiers = (
                    qualifiers + [space.name] + (["const"] if space.readonly else [])
                )

    def resolve_pointer_local(self, context, local, sources):
        """Settle ``local``'s space when every value it takes resolves; ``False`` leaves it for a later round."""
        guess = None
        for source in sources:
            guess = context.pointee(source, tentative=True)
            if guess is not None:
                break
        if guess is None:
            return False
        self.spaces[id(local)] = guess
        values = [context.pointee(source, tentative=True) for source in sources]
        if any(value is None for value in values):
            del self.spaces[id(local)]
            return False
        names = sorted({value.name for value in values})
        if len(names) > 1:
            self.refuse(
                f"generic pointer: '{context.function.name}.{local.name}' points into "
                f"{' and '.join(names)} memory"
            )
            return True
        self.spaces[id(local)] = Space(
            guess.name, any(value.readonly for value in values)
        )
        return True

    @staticmethod
    def remove_declaration(body, declaration):
        for node in _walk(body):
            for attribute in ("statements", "body"):
                statements = getattr(node, attribute, None)
                if isinstance(statements, list) and any(
                    s is declaration for s in statements
                ):
                    statements[:] = [s for s in statements if s is not declaration]
                    return
        if isinstance(body, list):
            body[:] = [s for s in body if s is not declaration]

    def refuse_readonly_writes(self, context):
        for node in _body_nodes(context.function):
            if isinstance(node, A.AssignmentNode):
                target = node.target
            elif isinstance(node, A.UnaryOpNode) and node.operator in _INCREMENTS:
                target = node.operand
            else:
                continue
            if isinstance(target, A.IdentifierNode):
                continue
            root = target
            while isinstance(
                root,
                (
                    A.ArrayAccessNode,
                    A.MemberAccessNode,
                    A.PointerAccessNode,
                    A.SwizzleNode,
                    A.UnaryOpNode,
                ),
            ):
                root = (
                    getattr(root, "array_expr", None)
                    or getattr(root, "object_expr", None)
                    or getattr(root, "pointer_expr", None)
                    or getattr(root, "vector_expr", None)
                    or getattr(root, "operand", None)
                )
            if not isinstance(root, A.IdentifierNode):
                continue
            declaration = context.resolve(root)
            space = self.spaces.get(id(declaration))
            if space is not None and space.readonly:
                self.refuse(
                    f"generic pointer: '{context.function.name}' writes through "
                    f"'{root.name}', which points into read-only {space.name} memory"
                )

    def resolve_return(self, context):
        function = context.function
        declared = getattr(function, "return_type", None)
        if not is_generic_pointer_type(declared):
            return
        values = [
            node.value
            for node in _body_nodes(function)
            if isinstance(node, A.ReturnNode) and node.value is not None
        ]
        spaces = [context.pointee(value) for value in values]
        if not spaces or any(space is None for space in spaces):
            self.refuse(
                f"generic pointer: the address space the pointer '{function.name}' returns "
                "points into is unknown"
            )
            return
        names = sorted({space.name for space in spaces})
        if len(names) > 1:
            self.refuse(
                f"generic pointer: '{function.name}' returns pointers into "
                f"{' and '.join(names)} memory"
            )
            return
        context.spec.return_space = Space(
            names[0], any(space.readonly for space in spaces)
        )
        function.return_type = _typed_pointer(declared, context.spec.return_space)

    def run(self):
        if not self.entries:
            for container, _stage in self.containers:
                for function in container:
                    if isinstance(
                        function, A.FunctionNode
                    ) and _mentions_generic_pointer(function):
                        self.refuse(
                            f"generic pointer: the program has no entry point, so the address spaces "
                            f"the generic pointers of '{function.name}' point into are unknown"
                        )
            return
        for entry, stage in self.entries:
            for parameter in _parameters(entry):
                if _contains_generic_pointer(_declared_type(parameter)):
                    self.refuse(
                        f"generic pointer: entry point '{entry.name}' has the generic "
                        f"pointer parameter '{parameter.name}'"
                    )
            self.specialize(entry, (), stage, is_entry=True)
        self.assemble()

    def assemble(self):
        """Put each specialization in place of its function, name the copies and drop the unreached users."""
        taken = {
            f.name
            for container, _stage in self.containers
            for f in container
            if isinstance(f, A.FunctionNode)
        }
        taken.update(self.module)
        by_original = {}
        for spec in self.order:
            by_original.setdefault(id(spec.original), []).append(spec)
        replaced = {}
        renamed = set()
        for container, _stage in self.containers:
            for function in container:
                if not isinstance(
                    function, A.FunctionNode
                ) or not _has_generic_parameters(function):
                    continue
                specs = by_original.get(id(function), [])
                replaced[id(function)] = [spec.function for spec in specs]
                if len(specs) > 1:
                    renamed.add(function.name)
                    for spec in specs:
                        spec.function.name = self.fresh_name(
                            function.name, spec.spaces, taken
                        )
        for spec in self.order:
            for call, callee in spec.calls:
                if call_name(call) != callee.function.name:
                    _set_call_name(call, callee.function.name)
        dropped = self.unreached_generic_users(by_original, renamed)
        for container, _stage in self.containers:
            rebuilt = []
            for function in container:
                if id(function) in dropped:
                    continue
                rebuilt.extend(replaced.get(id(function), [function]))
            container[:] = rebuilt

    @staticmethod
    def fresh_name(name, spaces, taken):
        base = f"{name}_{'_'.join(space.tag for space in spaces)}"
        candidate = base
        suffix = 2
        while candidate in taken:
            candidate = f"{base}_{suffix}"
            suffix += 1
        taken.add(candidate)
        return candidate

    def unreached_generic_users(self, by_original, renamed):
        """The ids of the unreached functions that use generic pointers or call a function that does."""
        unreached = [
            function
            for container, _stage in self.containers
            for function in container
            if isinstance(function, A.FunctionNode) and id(function) not in by_original
        ]
        dropped = {id(f) for f in unreached if _mentions_generic_pointer(f)}
        dropped_names = {f.name for f in unreached if id(f) in dropped} | renamed
        changed = True
        while changed:
            changed = False
            for function in unreached:
                if id(function) in dropped:
                    continue
                if any(
                    isinstance(node, A.FunctionCallNode)
                    and call_name(node) in dropped_names
                    for node in _body_nodes(function)
                ):
                    dropped.add(id(function))
                    dropped_names.add(function.name)
                    changed = True
        return dropped


def program_function_nodes(ast):
    """Every function of ``ast``: the module's, then each stage's local functions and entry point."""
    functions = list(getattr(ast, "functions", None) or [])
    stages = getattr(ast, "stages", None) or {}
    stage_list = (
        [s for key in stages.keys() for s in stages.all_for_stage(key)]
        if isinstance(stages, A.StageMap)
        else list(stages.values()) if isinstance(stages, dict) else list(stages)
    )
    for stage in stage_list:
        functions.extend(getattr(stage, "local_functions", None) or [])
        entry = getattr(stage, "entry_point", None)
        if entry is not None and not any(entry is f for f in functions):
            functions.append(entry)
    return [f for f in functions if isinstance(f, A.FunctionNode)]


def function_body_nodes(function):
    """The structural nodes of ``function``'s body, in source order, not entering nested functions."""
    return _body_nodes(function)


def _program_mentions_generic_pointer(ast):
    return any(_mentions_generic_pointer(f) for f in program_function_nodes(ast))


def assign_generic_pointer_spaces(ast, refuse):
    """Return ``ast``, or a copy whose generic pointers name the address spaces they point into.

    ``refuse(reason)`` records each generic pointer whose space cannot be derived; the copy then still spells it.
    """
    if not isinstance(
        getattr(ast, "functions", None), list
    ) or not _program_mentions_generic_pointer(ast):
        return ast
    program = deepcopy(ast)
    _Program(program, refuse).run()
    return program


# Comments and string literals, blanked before an output is searched.
_NON_CODE = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"', re.S)
_GENERIC_POINTER_SPELLING = re.compile(
    r"(?<![\w.:>])ptr\s*<\s*[A-Za-z_][\w:]*(?:\s*<[^<>;{}()]*>)?\s*\**\s*>"
)


def written_generic_pointer_spellings(code):
    """The ``ptr<T>`` type spellings ``code`` uses outside comments and string literals, sorted."""
    if not isinstance(code, str) or "ptr" not in code:
        return []
    text = _NON_CODE.sub(" ", code)
    return sorted(
        {
            re.sub(r"\s+", "", match.group(0))
            for match in _GENERIC_POINTER_SPELLING.finditer(text)
        }
    )


def refuse_written_generic_pointers(code, refuse, target_label):
    """Refuse each generic pointer type the rendered ``code`` still spells, which ``target_label`` has no type for."""
    for spelling in written_generic_pointer_spellings(code):
        refuse(
            f"generic pointer: the output spells '{spelling}', a pointer without an "
            f"address space, which {target_label} has no type for"
        )
