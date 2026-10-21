"""Structs with opaque members, flattened into separate resources for targets that cannot hold them.

CrossGL, GLSL, HLSL and MSL argument buffers let a struct hold samplers, textures and images::

    struct Material { vec3 albedo; sampler2D albedoMap; }
    uniform Material materials[4];
    vec4 shade(Material m, vec2 uv) { return texture(m.albedoMap, uv) * vec4(m.albedo, 1.0); }

A SPIR-V module for Vulkan cannot: a ``Uniform`` block holds only data, opaque values live in their own
``UniformConstant`` variables, and a ``Function`` variable cannot hold one (Vulkan specification, "Shader Resource
Interface"; GL_KHR_vulkan_glsl allows opaque types only as uniforms outside blocks and as parameters). The SPIR-V
emitter refused every such access ("texture requires a sampled image operand"). This pass rewrites the program the
way DXC's legalization does for HLSL structs of resources:

* each opaque member of a uniform struct becomes its own uniform, named after its path (``materials_albedoMap``),
  with one array dimension per struct array crossed on the way, so ``materials[i].albedoMap`` becomes
  ``materials_albedoMap[i]``; the struct keeps its data members, and a uniform left without any is dropped;
* a parameter of such a struct becomes the data parameter followed by one parameter per opaque member, and every
  call passes the matching flattened resources;
* a local copy (``Material m = materials[i];``) records its indices in ``int`` locals, so ``m.albedoMap`` becomes
  ``materials_albedoMap[m_index0]`` with the index the copy was made with; reassigning it from the same uniform
  updates those locals.

The resource selected is the one the source selected, and descriptor-array indexing has the same dynamic-uniformity
rule as indexing the struct array did (GLSL 4.60 section 4.1.7.1), so the rewrite does not change behaviour.
Forms with no such rewrite are refused with :class:`OpaqueStructUnsupportedError`: a non-uniform global, a local
without an initializer or copied from anything but a uniform, parameter or local copy, a copy reassigned from a
different uniform, an ``out`` parameter, a function returning the struct, a struct constructor, and an argument
whose indices have side effects.
"""

from __future__ import annotations

import re

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import call_name, iter_functions, iter_structs
from .rewrite import call, child_slots, clone, get_child, is_pure, set_child

__all__ = ["OpaqueStructUnsupportedError", "legalize_opaque_struct_members"]

TARGETS = frozenset({"vulkan", "spirv"})
_DIMENSIONS = r"(?:1d|2d|3d|cube|2drect|buffer|2dms)(?:_?array)?(?:shadow)?(?:ms)?"
_OPAQUE = re.compile(
    rf"(?:(?:rw|rasterizerordered)?[iu]?(?:sampler|texture|image){_DIMENSIONS}|depth{_DIMENSIONS}"
    r"|sampler(?:shadow)?|samplerstate|samplercomparisonstate|[iu]?subpassinput(?:ms)?"
    r"|accelerationstructure(?:ext)?|raytracingaccelerationstructure|atomic_uint"
    r"|(?:rw|append|consume)?structuredbuffer|(?:rw)?byteaddressbuffer|(?:rw)?buffer)",
    re.IGNORECASE,
)
_WRITABLE = frozenset({"out", "inout"})


class OpaqueStructUnsupportedError(TranslationDiagnostic):
    """A struct holding a sampler, texture or other resource is used in a way that cannot be flattened."""

    def __init__(self, target, detail, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.opaque-struct-member",)
        super().__init__(
            f"{detail}; {target} cannot hold a sampler, texture or image inside a struct, so such a struct is only "
            "carried as a uniform, a parameter or a local copy of one, with its resources accessed by member",
            target=target,
            code=f"{CODE_PREFIX}{target}-opaque-struct-unsupported",
        )


def _type_name(node):
    return str(getattr(node, "name", None) or node)


def _split(type_node):
    """``(element type, array sizes in index order)``: ``T a[2][3]`` is ``ArrayType(ArrayType(T, 3), 2)``."""
    sizes = []
    while isinstance(type_node, A.ArrayType):
        sizes.append(type_node.size)
        type_node = type_node.element_type
    return type_node, sizes


def _copy(size):
    return clone(size) if isinstance(size, A.ASTNode) else size


def _literal_int(size):
    value = getattr(size, "value", size)
    if isinstance(value, bool):
        return None
    try:
        return int(str(value).rstrip("uU"), 0)
    except ValueError:
        return None


def _int(value):
    return A.LiteralNode(value, A.PrimitiveType("int"))


def _descriptor_type(base, sizes):
    """One array dimension at most: Vulkan descriptor arrays are one-dimensional (VUID-04655)."""
    if len(sizes) <= 1:
        return A.ArrayType(base, _copy(sizes[0])) if sizes else base
    literals = [_literal_int(size) for size in sizes]
    if all(value is not None for value in literals):
        total = 1
        for value in literals:
            total *= value
        return A.ArrayType(base, _int(total))
    total = _copy(sizes[0])
    for size in sizes[1:]:
        total = A.BinaryOpNode(total, "*", _copy(size))
    return A.ArrayType(base, total)


def _is_opaque(type_node) -> bool:
    return bool(_OPAQUE.fullmatch(_type_name(type_node)))


def _names(node, field):
    return {
        str(getattr(i, "name", i)).lower() for i in getattr(node, field, None) or []
    }


def _chain(expr):
    """``(head identifier, [("m", member) | ("i", index node)])`` for a member/index chain, else ``(None, None)``."""
    steps = []
    while True:
        if isinstance(expr, A.MemberAccessNode):
            steps.append(("m", expr.member))
            expr = expr.object_expr
        elif isinstance(expr, A.ArrayAccessNode):
            steps.append(("i", expr.index_expr))
            expr = expr.array_expr
        elif isinstance(expr, A.IdentifierNode):
            return expr.name, steps[::-1]
        else:
            return None, None


def _index(expr, index):
    node = A.ArrayAccessNode(expr, index)
    node.array, node.index = node.array_expr, node.index_expr
    return node


class _Leaf:
    """An opaque member reached from a struct: member ``path``, crossed struct-array ``dims``, its own type."""

    def __init__(self, path, dims, base, own):
        self.path, self.dims, self.base, self.own = path, dims, base, own


class _Origin:
    """A uniform or parameter of an opaque struct: the place every flattened name is derived from."""

    def __init__(self, key, prefix, struct, dims):
        self.key, self.prefix, self.struct, self.dims = key, prefix, struct, dims


class _Ref:
    """Where an expression points: an origin, the member path from it, the array sizes crossed and the indices
    consumed so far (``pending`` sizes of the current struct array are not indexed yet).
    """

    def __init__(self, origin, struct, pending, path, indices, sizes):
        self.origin, self.struct, self.pending = origin, struct, pending
        self.path, self.indices, self.sizes = path, indices, sizes


class _Legalizer:
    def __init__(self, program, target):
        self.program, self.target = program, target
        self.structs = {
            s.name: s for s in iter_structs(program) if isinstance(s, A.StructNode)
        }
        self.members = {
            name: [m for m in s.members or [] if hasattr(m, "member_type")]
            for name, s in self.structs.items()
        }
        self._leaves, self._data = {}, {}
        self.opaque = {name for name in self.structs if self.leaves(name)}
        self.flat, self.taken = {}, self._declared_names()
        self.expanded = {}

    def refuse(self, detail, node=None):
        raise OpaqueStructUnsupportedError(
            self.target, detail, source_location=getattr(node, "source_location", None)
        )

    # Struct analysis -------------------------------------------------------------------------------------------

    def leaves(self, name):
        if name not in self._leaves:
            self._leaves[name] = []
            found = []
            for member in self.members.get(name, []):
                base, sizes = _split(member.member_type)
                if _is_opaque(base):
                    found.append(_Leaf((member.name,), [], base, sizes))
                elif _type_name(base) in self.structs:
                    for sub in self.leaves(_type_name(base)):
                        path = (member.name,) + sub.path
                        found.append(_Leaf(path, sizes + sub.dims, sub.base, sub.own))
            self._leaves[name] = found
        return self._leaves[name]

    def has_data(self, name):
        if name not in self._data:
            self._data[name] = any(self._is_data(m) for m in self.members.get(name, []))
        return self._data[name]

    def _is_data(self, member):
        base, _ = _split(member.member_type)
        if _is_opaque(base):
            return False
        return _type_name(base) not in self.opaque or self.has_data(_type_name(base))

    def struct_of(self, type_node):
        base, sizes = _split(type_node)
        name = _type_name(base)
        return (name, sizes) if name in self.opaque else (None, sizes)

    def _declared_names(self):
        names = set(self.structs)
        stack = [self.program]
        while stack:
            node = stack.pop()
            if isinstance(node, (A.VariableNode, A.ParameterNode, A.FunctionNode)):
                names.add(str(node.name))
            stack.extend(get_child(node, f, k) for f, k in child_slots(node))
        return names

    def unique(self, candidate):
        while candidate in self.taken:
            candidate += "_"
        self.taken.add(candidate)
        return candidate

    def flat_name(self, origin, path):
        key = (origin.key, tuple(path))
        if key not in self.flat:
            self.flat[key] = self.unique(f"{origin.prefix}_{'_'.join(path)}")
        return self.flat[key]

    # Resolution ------------------------------------------------------------------------------------------------

    def resolve(self, expr, scope):
        """``("leaf", ref)``, ``("struct", ref)`` or ``None`` for a chain that does not start at an origin."""
        head, steps = _chain(expr)
        entry = scope.get(head) if head is not None else None
        if entry is None:
            return None
        if isinstance(entry, _Origin):
            ref = _Ref(entry, entry.struct, len(entry.dims), [], [], list(entry.dims))
        else:
            ref = _Ref(
                entry.origin,
                entry.struct,
                entry.pending,
                list(entry.path),
                [clone(i) for i in entry.indices],
                list(entry.sizes),
            )
        for position, (kind, value) in enumerate(steps):
            if kind == "i":
                if ref.pending == 0:
                    return None
                ref.pending -= 1
                ref.indices.append(value)
                continue
            if ref.pending:
                self.refuse(
                    f"member '{value}' of an array of struct {ref.struct}", expr
                )
            member = next(
                (m for m in self.members[ref.struct] if m.name == value), None
            )
            if member is None:
                return None
            base, sizes = _split(member.member_type)
            ref.path.append(value)
            ref.sizes.extend(sizes)
            if _is_opaque(base):
                return self.leaf(expr, ref, steps[position + 1 :], len(sizes))
            if _type_name(base) not in self.opaque:
                return None
            ref.struct, ref.pending = _type_name(base), len(sizes)
        return ("struct", ref)

    def leaf(self, expr, ref, rest, own):
        """The resource an opaque member chain selects, once its own array dimensions are indexed too."""
        if len(rest) > own or any(kind != "i" for kind, _ in rest):
            return None
        if len(rest) < own:
            if len(rest) == 0 and len(ref.sizes) == own:
                return ("leaf", ref)
            self.refuse(
                "an array of resources inside a struct array is used as a whole array",
                expr,
            )
        ref.indices.extend(value for _, value in rest)
        return ("leaf", ref)

    def flattened(self, ref):
        expr = A.IdentifierNode(self.flat_name(ref.origin, ref.path))
        if len(ref.indices) == 1:
            return _index(expr, ref.indices[0])
        if not ref.indices:
            return expr
        # Row-major linear index into the one-dimensional descriptor array.
        linear = call("int", [ref.indices[0]])
        for size, index in zip(ref.sizes[1:], ref.indices[1:]):
            scaled = A.BinaryOpNode(linear, "*", _copy(size))
            linear = A.BinaryOpNode(scaled, "+", call("int", [index]))
        return _index(expr, linear)

    # Declarations ----------------------------------------------------------------------------------------------

    def legalize_globals(self, declarations, scope):
        result = []
        for node in declarations:
            if not isinstance(node, A.VariableNode):
                result.append(node)
                continue
            struct, sizes = self.struct_of(node.var_type)
            if struct is None:
                result.append(node)
                continue
            qualifiers = _names(node, "qualifiers") | _names(node, "attributes")
            if "uniform" not in qualifiers or "buffer" in qualifiers:
                self.refuse(f"'{node.name}' of struct {struct} is not a uniform", node)
            if node.initial_value is not None:
                self.refuse(
                    f"uniform '{node.name}' of struct {struct} has an initializer", node
                )
            origin = _Origin(("uniform", node.name, struct), node.name, struct, sizes)
            scope[node.name] = origin
            if self.has_data(struct):
                result.append(node)
            for leaf in self.leaves(struct):
                result.append(
                    A.VariableNode(
                        self.flat_name(origin, leaf.path),
                        _descriptor_type(leaf.base, sizes + leaf.dims + leaf.own),
                        qualifiers=["uniform"],
                    )
                )
        return result

    def expand_signature(self, function):
        """Replace each opaque-struct parameter by its data parameter and one parameter per opaque member."""
        struct, _ = self.struct_of(getattr(function, "return_type", None))
        if struct is not None:
            self.refuse(f"function '{function.name}' returns struct {struct}", function)
        parameters, expansion, origins = [], [], {}
        for position, parameter in enumerate(function.parameters or []):
            struct, sizes = self.struct_of(parameter.param_type)
            if struct is None:
                parameters.append(parameter)
                continue
            if _names(parameter, "qualifiers") & _WRITABLE:
                self.refuse(
                    f"parameter '{parameter.name}' of struct {struct} is written",
                    parameter,
                )
            key = ("parameter", id(function), parameter.name)
            origin = _Origin(key, parameter.name, struct, sizes)
            origins[parameter.name] = origin
            expansion.append((position, struct, len(sizes)))
            if self.has_data(struct):
                parameters.append(parameter)
            for leaf in self.leaves(struct):
                if sizes or leaf.dims or leaf.own:
                    # A descriptor array cannot be sliced, so only single resources are passed on.
                    self.refuse(
                        f"parameter '{parameter.name}' of struct {struct} holds an array of resources",
                        parameter,
                    )
                name = self.flat_name(origin, leaf.path)
                parameters.append(A.ParameterNode(name, leaf.base))
        function.parameters = parameters
        if hasattr(function, "params"):
            function.params = parameters
        return expansion, origins

    # Statements ------------------------------------------------------------------------------------------------

    def statements(self, statements, scope):
        result = []
        for statement in statements:
            expression = getattr(statement, "expression", None)
            if (
                isinstance(statement, A.VariableNode)
                and self.struct_of(statement.var_type)[0]
            ):
                result.extend(self.local_copy(statement, scope))
            elif isinstance(expression, A.AssignmentNode) and self.is_copy_target(
                expression, scope
            ):
                result.extend(self.reassign(statement, expression, scope))
            elif isinstance(statement, A.AssignmentNode) and self.is_copy_target(
                statement, scope
            ):
                result.extend(self.reassign(statement, statement, scope))
            else:
                result.append(self.visit(statement, scope))
        return result

    def is_copy_target(self, assignment, scope):
        target = assignment.target
        return isinstance(target, A.IdentifierNode) and isinstance(
            scope.get(target.name), _Ref
        )

    def capture(self, ref, name, scope):
        """Declare one ``int`` local per index of ``ref``, evaluated once; return them and their names."""
        declarations, identifiers = [], []
        for position, index in enumerate(ref.indices):
            temp = self.unique(f"{name}_index{position}")
            value = call("int", [self.expr(clone(index), scope)])
            declarations.append(A.VariableNode(temp, A.PrimitiveType("int"), value))
            identifiers.append(temp)
        return declarations, identifiers

    def local_copy(self, node, scope):
        struct, sizes = self.struct_of(node.var_type)
        resolved = (
            self.resolve(node.initial_value, scope) if node.initial_value else None
        )
        if resolved is None or resolved[0] != "struct":
            self.refuse(
                f"local '{node.name}' of struct {struct} is not a copy of a uniform, parameter or local copy",
                node,
            )
        ref = resolved[1]
        if ref.struct != struct or ref.pending != len(sizes):
            self.refuse(
                f"local '{node.name}' of struct {struct} copies a {ref.struct}", node
            )
        originals = list(ref.indices)
        declarations, temps = self.capture(ref, node.name, scope)
        replacements = {id(i): A.IdentifierNode(t) for i, t in zip(originals, temps)}
        ref.indices = [A.IdentifierNode(t) for t in temps]
        if self.has_data(struct):
            node.initial_value = self.expr(
                self.substitute(node.initial_value, replacements), scope
            )
            declarations.append(node)
        scope[node.name] = ref
        return declarations

    def reassign(self, statement, assignment, scope):
        name = assignment.target.name
        copy = scope[name]
        resolved = self.resolve(assignment.value, scope)
        if (
            str(getattr(assignment, "operator", "=")) != "="
            or resolved is None
            or resolved[0] != "struct"
            or resolved[1].origin is not copy.origin
            or resolved[1].path != copy.path
            or resolved[1].pending != copy.pending
        ):
            self.refuse(
                f"local copy '{name}' of struct {copy.struct} is reassigned from a different resource",
                statement,
            )
        result = []
        for temp, index in zip(copy.indices, resolved[1].indices):
            value = call("int", [self.expr(clone(index), scope)])
            update = A.AssignmentNode(A.IdentifierNode(temp.name), value, "=")
            result.append(A.ExpressionStatementNode(update))
        if self.has_data(copy.struct):
            replacements = {
                id(i): A.IdentifierNode(t.name)
                for i, t in zip(resolved[1].indices, copy.indices)
            }
            assignment.value = assignment.right = self.expr(
                self.substitute(assignment.value, replacements), scope
            )
            result.append(statement)
        return result

    def substitute(self, expr, replacements):
        """Replace the index nodes of a chain by the locals that captured them."""
        node = expr
        while isinstance(node, (A.MemberAccessNode, A.ArrayAccessNode)):
            if (
                isinstance(node, A.ArrayAccessNode)
                and id(node.index_expr) in replacements
            ):
                set_child(node, "index_expr", None, replacements[id(node.index_expr)])
            node = (
                node.object_expr
                if isinstance(node, A.MemberAccessNode)
                else node.array_expr
            )
        return expr

    def visit(self, node, scope):
        if isinstance(node, A.BlockNode):
            node.statements = self.statements(node.statements or [], dict(scope))
            return node
        if isinstance(node, A.VariableNode):
            if self.struct_of(node.var_type)[0]:
                self.refuse(
                    f"'{node.name}' of a struct of resources is declared here", node
                )
            if node.initial_value is not None:
                node.initial_value = self.expr(node.initial_value, scope)
            return node
        if isinstance(node, A.ExpressionNode):
            return self.expr(node, scope)
        for field, key in list(child_slots(node)):
            set_child(node, field, key, self.visit(get_child(node, field, key), scope))
        return node

    # Expressions -----------------------------------------------------------------------------------------------

    def expr(self, node, scope):
        if isinstance(node, (A.MemberAccessNode, A.ArrayAccessNode)):
            resolved = self.resolve(node, scope)
            if resolved is not None and resolved[0] == "leaf":
                flattened = self.flattened(resolved[1])
                target = flattened
                while isinstance(target, A.ArrayAccessNode):
                    set_child(
                        target, "index_expr", None, self.expr(target.index_expr, scope)
                    )
                    target = target.array_expr
                return flattened
        if isinstance(node, A.IdentifierNode):
            entry = scope.get(node.name)
            if entry is not None and not self.has_data(entry.struct):
                self.refuse(
                    f"'{node.name}' holds only resources and is used as a value", node
                )
            return node
        if isinstance(node, (A.FunctionCallNode, A.ConstructorNode)):
            name = call_name(node) if isinstance(node, A.FunctionCallNode) else None
            name = name or _type_name(getattr(node, "constructor_type", ""))
            if name in self.opaque:
                self.refuse(f"struct {name} is constructed", node)
            if name in self.expanded and isinstance(node, A.FunctionCallNode):
                self.expand_call(node, self.expanded[name], scope)
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            set_child(node, field, key, self.expr(child, scope))
        return node

    def expand_call(self, node, expansion, scope):
        arguments, by_position = list(node.arguments or []), {
            p: (s, d) for p, s, d in expansion
        }
        result = []
        for position, argument in enumerate(arguments):
            if position not in by_position:
                result.append(argument)
                continue
            struct, dims = by_position[position]
            resolved = self.resolve(argument, scope)
            ref = resolved[1] if resolved and resolved[0] == "struct" else None
            if ref is None or ref.struct != struct or ref.pending != dims:
                self.refuse(
                    f"argument {position + 1} of '{call_name(node)}' is not a {struct} uniform, "
                    "parameter or local copy",
                    node,
                )
            if not all(is_pure(index) for index in ref.indices):
                self.refuse(
                    "an index with side effects selects a struct of resources", node
                )
            if self.has_data(struct):
                result.append(argument)
            for leaf in self.leaves(struct):
                indices = [clone(index) for index in ref.indices]
                path = ref.path + list(leaf.path)
                leaf_ref = _Ref(ref.origin, None, 0, path, indices, list(ref.sizes))
                result.append(self.flattened(leaf_ref))
        node.arguments = node.args = result

    # Driver ----------------------------------------------------------------------------------------------------

    def run(self):
        if not self.opaque:
            return
        shader_scope = {}
        self.program.global_variables = self.legalize_globals(
            list(getattr(self.program, "global_variables", None) or []), shader_scope
        )
        stage_scopes = []
        for stage in _stage_list(self.program):
            scope = dict(shader_scope)
            stage.local_variables = self.legalize_globals(
                list(getattr(stage, "local_variables", None) or []), scope
            )
            stage_scopes.append((stage, scope))
        signatures, functions = {}, []
        for function in iter_functions(self.program):
            if any(function is seen for seen in functions):
                continue
            functions.append(function)
            expansion, origins = self.expand_signature(function)
            signature = [(p, s, d) for p, s, d in expansion]
            if function.name in signatures and signatures[function.name] != signature:
                self.refuse(
                    f"overloads of '{function.name}' take structs of resources differently"
                )
            signatures[function.name] = signature
            function.crosstl_opaque_origins = origins
        self.expanded = {name: sig for name, sig in signatures.items() if sig}
        for function in getattr(self.program, "functions", None) or []:
            self.function_body(function, shader_scope)
        for stage, scope in stage_scopes:
            entry = getattr(stage, "entry_point", None)
            if entry is not None and getattr(entry, "crosstl_opaque_origins", None):
                self.refuse(
                    f"entry point '{entry.name}' takes a struct of resources", entry
                )
            for function in getattr(stage, "local_functions", None) or []:
                self.function_body(function, scope)
            if entry is not None:
                self.function_body(entry, scope)
        for name in self.opaque:
            struct = self.structs[name]
            struct.members = [
                m
                for m in struct.members
                if not hasattr(m, "member_type") or self._is_data(m)
            ]

    def function_body(self, function, scope):
        if not hasattr(function, "crosstl_opaque_origins"):
            return
        origins = function.crosstl_opaque_origins
        del function.crosstl_opaque_origins
        body = getattr(function, "body", None)
        if body is None:
            return
        local_scope = dict(scope)
        for parameter in function.parameters or []:
            local_scope.pop(parameter.name, None)
        local_scope.update(origins)
        if isinstance(body, A.BlockNode):
            body.statements = self.statements(body.statements or [], local_scope)
        elif isinstance(body, list):
            function.body = self.statements(body, local_scope)


def _stage_list(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values()) if isinstance(stages, dict) else list(stages)


def legalize_opaque_struct_members(program, target) -> None:
    """Flatten structs holding resources into separate resources for targets in :data:`TARGETS`."""
    target = str(target or "").lower()
    if target in TARGETS and isinstance(program, A.ShaderNode):
        _Legalizer(program, target).run()
