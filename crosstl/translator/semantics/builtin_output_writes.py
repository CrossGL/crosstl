"""Built-in outputs written by name, and ``in``/``out`` stage variables, as the outputs HLSL and MSL return.

GLSL writes a stage's built-in outputs as global variables, anywhere in the stage, helpers included: a vertex shader's
``gl_Position``, ``gl_PointSize``, ``gl_ClipDistance[]`` and ``gl_CullDistance[]`` (GLSL 4.60 section 7.1.1),
``gl_Layer`` and ``gl_ViewportIndex`` (ARB_shader_viewport_layer_array), and a fragment shader's ``gl_FragDepth``
and ``gl_SampleMask[]`` (section 7.1.5), ``gl_FragColor`` and ``gl_FragData[]`` (section 7.1.7) and
``gl_FragStencilRefEXT`` (ARB_shader_stencil_export). The CrossGL import keeps the fragment ones, and CrossGL lets a
stage declare its interface as ``in`` and ``out`` variables. HLSL and MSL have neither: an entry point returns its
outputs, built-in ones included, as a structure carrying system-value semantics or attributes. The generators emitted
the stores verbatim, as undeclared identifiers, and the Metal one declared ``in`` and ``out`` stage variables as
uninitialized locals of the entry point, which compiles and reads and writes nothing.

:func:`declare_builtin_outputs` rewrites each vertex and fragment stage for those targets, before built-in attributes
are spelled for the target:

* a stage-level variable declared ``in`` or ``out`` becomes an ``@input`` or ``@output`` variable;
* each built-in output that the stage's functions write or read by name becomes a stage-level ``@output`` variable
  carrying the built-in, under a fresh name, unless the stage already declares one. ``gl_ClipDistance`` and
  ``gl_CullDistance`` take as many elements as their largest index, which must be a constant, as GLSL requires of an
  array it does not see redeclared. ``gl_FragData[n]`` becomes the output at location ``n``. ``gl_SampleMask[0]``
  becomes a scalar, because ``SV_Coverage`` and ``[[sample_mask]]`` carry one 32-bit word; any other index is refused.

:func:`convert_signed_builtin_outputs` then retypes each ``int`` stage-level output that carries a built-in HLSL and
MSL deliver as ``uint`` (``SV_Coverage``, ``SV_StencilRef``, ``SV_RenderTargetArrayIndex``, ``SV_ViewportArrayIndex``;
``[[sample_mask]]``, ``[[stencil]]``, ``[[render_target_array_index]]``, ``[[viewport_array_index]]``) as ``uint``.
A store converts its value with ``uint()``, so the bits GLSL writes are kept, and a read converts back with ``int()``.
An ``int[1]`` sample mask becomes a scalar the same way. :mod:`.stage_interface_structs` then gathers the stage-level
variables into the structures the entry point takes and returns. A construct this cannot keep exactly is refused with
:class:`~.stage_interface_structs.StageInterfaceUnsupportedError`.
"""

from __future__ import annotations

from .. import ast as A
from .builtin_values import glsl_builtin_name
from .hub_types import call_name
from .rewrite import child_slots, get_child, set_child
from .stage_interface_structs import StageInterfaceUnsupportedError
from .stage_varyings import int_literal, literal_int

__all__ = ["convert_signed_builtin_outputs", "declare_builtin_outputs"]

TARGETS = frozenset({"metal", "directx", "hlsl", "dx11", "dx12", "d3d11", "d3d12"})
_FLOAT, _INT, _VEC4 = "float", "int", "vec4"
_OUTPUTS = {
    "vertex": {
        "gl_Position": _VEC4,
        "gl_PointSize": _FLOAT,
        "gl_ClipDistance": _FLOAT,
        "gl_CullDistance": _FLOAT,
        "gl_Layer": _INT,
        "gl_ViewportIndex": _INT,
    },
    "fragment": {
        "gl_FragDepth": _FLOAT,
        "gl_SampleMask": _INT,
        "gl_FragColor": _VEC4,
        "gl_FragData": _VEC4,
        "gl_FragStencilRefEXT": _INT,
        "gl_FragStencilRefARB": _INT,
    },
}
# The built-in each spelling names: itself, or the EXT spelling an ARB extension shares.
_CANONICAL = {name: name for outputs in _OUTPUTS.values() for name in outputs}
_CANONICAL["gl_FragStencilRefARB"] = "gl_FragStencilRefEXT"
# Arrays sized by the largest constant index the stage uses.
_SIZED_ARRAYS = frozenset({"gl_ClipDistance", "gl_CullDistance"})
# Built-in outputs HLSL and MSL deliver as uint and GLSL types int.
_UNSIGNED = frozenset(
    {"gl_SampleMask", "gl_FragStencilRefEXT", "gl_Layer", "gl_ViewportIndex"}
)
_STAGE_QUALIFIERS = {"in": "input", "out": "output"}


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values() if isinstance(stages, dict) else stages)


def _stage_name(stage):
    value = getattr(stage, "stage", None)
    name = str(getattr(value, "value", value) or "").lower()
    return "fragment" if name == "pixel" else name


def _attribute_names(node):
    return {
        str(getattr(attribute, "name", "")).lower()
        for attribute in getattr(node, "attributes", None) or []
    }


def _builtin(node):
    for attribute in getattr(node, "attributes", None) or []:
        name = glsl_builtin_name(attribute)
        if name is not None:
            return name
    return None


def _walk(node, seen=None):
    seen = set() if seen is None else seen
    if not isinstance(node, A.ASTNode) or id(node) in seen:
        return
    seen.add(id(node))
    yield node
    for field, key in list(child_slots(node)):
        yield from _walk(get_child(node, field, key), seen)


def _function_names(function):
    """The names ``function`` declares: its parameters and its local variables."""
    names = {
        getattr(p, "name", None) for p in getattr(function, "parameters", None) or []
    }
    for node in _walk(getattr(function, "body", None)):
        if isinstance(node, A.VariableNode):
            names.add(node.name)
    return names


def _taken_names(program):
    names = set()
    for variable in getattr(program, "global_variables", None) or []:
        names.add(variable.name)
    for struct in getattr(program, "structs", None) or []:
        names.add(struct.name)
    functions = list(getattr(program, "functions", None) or [])
    for stage in _stages(program):
        for variable in getattr(stage, "local_variables", None) or []:
            names.add(variable.name)
        functions += list(getattr(stage, "local_functions", None) or [])
        if getattr(stage, "entry_point", None) is not None:
            functions.append(stage.entry_point)
    for function in functions:
        names.add(function.name)
        names |= _function_names(function)
    return names


def _fresh(preferred, taken):
    name, index = preferred, 0
    while name in taken:
        index += 1
        name = f"{preferred}_{index}"
    taken.add(name)
    return name


def _stage_functions(program, stage):
    """The stage's entry point first, then its helpers and the module's, all with bodies."""
    entry = stage.entry_point
    helpers = list(getattr(program, "functions", None) or []) + list(
        getattr(stage, "local_functions", None) or []
    )
    return [entry] + [
        function
        for function in helpers
        if function is not entry
        and isinstance(getattr(function, "body", None), A.ASTNode)
    ]


def _refuse(message, target, reason):
    raise StageInterfaceUnsupportedError(message, target=target, reason=reason)


def _normalize_stage_qualifiers(stage):
    changed = 0
    for variable in getattr(stage, "local_variables", None) or []:
        if not isinstance(variable, A.VariableNode):
            continue
        qualifiers = [
            str(q).lower() for q in getattr(variable, "qualifiers", None) or []
        ]
        directions = [q for q in qualifiers if q in _STAGE_QUALIFIERS]
        if len(directions) != 1 or _attribute_names(variable) & {"input", "output"}:
            continue
        variable.qualifiers = [
            q for q in variable.qualifiers if str(q).lower() not in _STAGE_QUALIFIERS
        ]
        variable.attributes = list(variable.attributes or []) + [
            A.AttributeNode(_STAGE_QUALIFIERS[directions[0]])
        ]
        changed += 1
    return changed


def _uses(function, candidates):
    """``(parent, field, key, node)`` for each by-name use of a candidate built-in in ``function``'s body."""
    found = []
    local = _function_names(function)

    def visit(node):
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            if isinstance(child, A.IdentifierNode):
                if child.name in candidates and child.name not in local:
                    found.append((node, field, key, child))
            elif isinstance(child, A.ArrayAccessNode) and isinstance(
                child.array_expr, A.IdentifierNode
            ):
                name = child.array_expr.name
                if name in candidates and name not in local:
                    found.append((node, field, key, child))
                visit(child.index_expr)
            elif isinstance(child, A.ASTNode):
                visit(child)

    body = getattr(function, "body", None)
    if isinstance(body, A.ASTNode):
        visit(body)
    return found


def _index_of(node, builtin, target):
    if not isinstance(node, A.ArrayAccessNode):
        _refuse(
            f"'{builtin}' is used without an index; {target} entry points return each element as a structure member",
            target,
            "builtin-output-array",
        )
    index = literal_int(node.index_expr)
    if index is None or index < 0:
        _refuse(
            f"'{builtin}' is indexed by an expression that is not a constant; GLSL sizes an array it does not see "
            f"redeclared by its constant indices, and {target} needs that size",
            target,
            "builtin-output-index",
        )
    return index


def _existing_output(program, stage, builtin):
    for variable in getattr(stage, "local_variables", None) or []:
        if "output" in _attribute_names(variable) and _builtin(variable) == builtin:
            return variable.name
    for variable in getattr(program, "global_variables", None) or []:
        qualifiers = {
            str(q).lower() for q in getattr(variable, "qualifiers", None) or []
        }
        if "out" in qualifiers and _builtin(variable) == builtin:
            return variable.name
    return None


def _type_node(name, size=None):
    element = (
        A.VectorType(A.PrimitiveType("float"), 4)
        if name == _VEC4
        else A.PrimitiveType(name)
    )
    return element if size is None else A.ArrayType(element, int_literal(size))


def _declare(stage, name, type_node, attributes):
    stage.local_variables = list(getattr(stage, "local_variables", None) or []) + [
        A.VariableNode(
            name=name,
            var_type=type_node,
            attributes=[A.AttributeNode("output")] + attributes,
        )
    ]


def declare_builtin_outputs(program, target) -> int:
    """Declare the stage variables and built-in outputs ``program``'s stages use by name; returns the stages changed."""
    target = str(target or "").lower()
    if target not in TARGETS or not isinstance(program, A.ShaderNode):
        return 0
    taken = None
    names = {}
    changed = 0
    for stage in _stages(program):
        outputs = _OUTPUTS.get(_stage_name(stage))
        if outputs is None or not isinstance(
            getattr(getattr(stage, "entry_point", None), "body", None), A.ASTNode
        ):
            continue
        stage_changed = _normalize_stage_qualifiers(stage) > 0
        if taken is None:
            taken = _taken_names(program)
        candidates = set(outputs) - _declared_variables(program)
        uses = {}
        for function in _stage_functions(program, stage):
            for use in _uses(function, candidates):
                name = _use_name(use[3])
                uses.setdefault(_CANONICAL[name], []).append(use)
        for builtin in sorted(uses):
            context = (names, taken, target)
            _rewrite_uses(
                program, stage, builtin, outputs[builtin], uses[builtin], context
            )
            stage_changed = True
        changed += int(stage_changed)
    return changed


def _use_name(node):
    return node.name if isinstance(node, A.IdentifierNode) else node.array_expr.name


def _declared_variables(program):
    names = {v.name for v in getattr(program, "global_variables", None) or []}
    for stage in _stages(program):
        names |= {v.name for v in getattr(stage, "local_variables", None) or []}
    return names


def _rewrite_uses(program, stage, builtin, element, uses, context):
    names, taken, target = context
    if builtin == "gl_FragData":
        locations = {}
        for parent, field, key, node in uses:
            index = _index_of(node, builtin, target)
            if index not in locations:
                locations[index] = _fresh(f"cgl_FragData{index}", taken)
                _declare(
                    stage,
                    locations[index],
                    _type_node(element),
                    [A.AttributeNode("location", [int_literal(index)])],
                )
            set_child(parent, field, key, A.IdentifierNode(locations[index]))
        return
    if builtin == "gl_SampleMask":
        for _parent, _field, _key, node in uses:
            if _index_of(node, builtin, target) != 0:
                _refuse(
                    f"'{builtin}[{literal_int(node.index_expr)}]' covers samples beyond the first 32; {target} "
                    "carries a single 32-bit sample mask",
                    target,
                    "builtin-output-index",
                )
    name = _existing_output(program, stage, builtin)
    if name is None:
        if builtin not in names:
            names[builtin] = _fresh(f"cgl_{builtin[3:]}", taken)
        name = names[builtin]
        size = None
        if builtin in _SIZED_ARRAYS:
            size = (
                max(_index_of(node, builtin, target) for _p, _f, _k, node in uses) + 1
            )
        _declare(stage, name, _type_node(element, size), [A.AttributeNode(builtin)])
    for parent, field, key, node in uses:
        if builtin == "gl_SampleMask":
            set_child(parent, field, key, A.IdentifierNode(name))
        elif isinstance(node, A.ArrayAccessNode):
            set_child(node, "array_expr", None, A.IdentifierNode(name))
        else:
            set_child(parent, field, key, A.IdentifierNode(name))


def _type_name(type_node):
    return str(getattr(type_node, "name", type_node) or "")


def _is_int(type_node):
    return isinstance(type_node, A.PrimitiveType) and _type_name(type_node) == "int"


def _single_int_array(type_node):
    return (
        isinstance(type_node, A.ArrayType)
        and _is_int(type_node.element_type)
        and literal_int(type_node.size) == 1
    )


def convert_signed_builtin_outputs(program, target) -> int:
    """Retype the ``int`` stage-level outputs carrying a ``uint`` built-in; returns how many changed."""
    target = str(target or "").lower()
    if target not in TARGETS or not isinstance(program, A.ShaderNode):
        return 0
    changed = 0
    for stage in _stages(program):
        if (
            _stage_name(stage) not in _OUTPUTS
            or getattr(stage, "entry_point", None) is None
        ):
            continue
        for variable in getattr(stage, "local_variables", None) or []:
            if (
                "output" not in _attribute_names(variable)
                or _builtin(variable) not in _UNSIGNED
            ):
                continue
            scalarize = _single_int_array(variable.var_type)
            if not (scalarize or _is_int(variable.var_type)):
                continue
            variable.var_type = A.PrimitiveType("uint")
            for function in _stage_functions(program, stage):
                if variable.name not in _function_names(function):
                    _Converter(variable.name, scalarize, target, program).visit(
                        function.body
                    )
            changed += 1
    return changed


class _Converter:
    """Converts stores to and reads of one ``uint`` output so the code around it keeps seeing GLSL's ``int``."""

    def __init__(self, name, scalarize, target, program):
        self.name = name
        self.scalarize = scalarize
        self.target = target
        self.out_parameters = _out_parameters(program)

    def is_ref(self, node):
        if self.scalarize:
            if not isinstance(node, A.ArrayAccessNode) or not self._named(
                node.array_expr
            ):
                return False
            if literal_int(node.index_expr) != 0:
                _refuse(
                    f"'{self.name}' is indexed by an expression other than 0; {self.target} carries a single "
                    "32-bit sample mask",
                    self.target,
                    "builtin-output-index",
                )
            return True
        return self._named(node)

    def _named(self, node):
        return isinstance(node, A.IdentifierNode) and node.name == self.name

    def convert(self, expression):
        if self.is_ref(expression):
            return self.read()
        self.visit(expression)
        return expression

    def visit(self, node):
        if not isinstance(node, A.ASTNode):
            return
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            if isinstance(child, A.AssignmentNode) and self.is_ref(child.target):
                value = self.convert(child.value)
                operator = str(child.operator or "=")
                if operator != "=":
                    value = A.BinaryOpNode(self.read(), operator[:-1], value)
                set_child(child, "target", None, A.IdentifierNode(self.name))
                set_child(child, "value", None, _call("uint", value))
                child.operator = "="
            elif (
                isinstance(child, A.UnaryOpNode)
                and str(child.operator) in {"++", "--"}
                and self.is_ref(child.operand)
            ):
                _refuse(
                    f"'{self.name}' is incremented in place; {self.target} delivers it as uint, so write the "
                    "store as an assignment",
                    self.target,
                    "builtin-output-increment",
                )
            elif self.is_ref(child):
                set_child(node, field, key, self.read())
            elif self.scalarize and self._named(child):
                _refuse(
                    f"the sample mask '{self.name}' is used without an index; {self.target} carries it as one "
                    "32-bit word",
                    self.target,
                    "builtin-output-array",
                )
            elif isinstance(child, A.ASTNode):
                if isinstance(child, A.FunctionCallNode):
                    self.check_out_arguments(child)
                self.visit(child)

    def read(self):
        return _call("int", A.IdentifierNode(self.name))

    def check_out_arguments(self, call):
        qualifiers = self.out_parameters.get(call_name(call))
        if not qualifiers:
            return
        for index, argument in enumerate(call.arguments or []):
            if index < len(qualifiers) and qualifiers[index] and self.is_ref(argument):
                _refuse(
                    f"'{self.name}' is passed to an out parameter of '{call_name(call)}'; {self.target} delivers "
                    "it as uint, so pass an int copy and store it afterwards",
                    self.target,
                    "builtin-output-argument",
                )


def _call(name, argument):
    return A.FunctionCallNode(A.IdentifierNode(name), [argument])


def _out_parameters(program):
    """Each function's parameters, ``True`` where the parameter is ``out`` or ``inout``."""
    functions = list(getattr(program, "functions", None) or [])
    for stage in _stages(program):
        functions += list(getattr(stage, "local_functions", None) or [])
    result = {}
    for function in functions:
        result[function.name] = [
            bool(
                {str(q).lower() for q in getattr(p, "qualifiers", None) or []}
                & {"out", "inout"}
            )
            for p in getattr(function, "parameters", None) or []
        ]
    return result
