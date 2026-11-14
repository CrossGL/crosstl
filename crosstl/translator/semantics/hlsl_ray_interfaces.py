"""GLSL-style ray tracing interfaces lowered to the DXR forms HLSL declares.

CrossGL carries ray tracing stages in two spellings, the one each importer reads:

* HLSL (DXR 1.0): the payload, hit attributes and callable data are entry parameters of struct type marked
  ``@payload``, ``@hit_attribute`` and ``@callable_data``; ``TraceRay(as, flags, mask, offset, stride, miss, ray,
  payload)`` takes a ``RayDesc`` and the payload variable, ``CallShader(index, data)`` the callable data variable,
  ``ReportHit(t, kind, attributes)`` the attributes; ray state is read through intrinsic calls (``WorldRayOrigin()``).
* GLSL (``GL_EXT_ray_tracing``) and SPIR-V: they are module-scope variables marked ``@rayPayloadEXT``,
  ``@rayPayloadInEXT``, ``@hitAttributeEXT``, ``@callableDataEXT`` and ``@callableDataInEXT`` with a ``location``;
  ``TraceRay`` takes the ray's origin, ``tmin``, direction and ``tmax`` and names the payload by its location, as
  ``CallShader`` names the callable data; ``ReportHit(t, kind)`` reports the module's hit attribute variable; ray
  state is a built-in variable (``gl_WorldRayOriginEXT``).

:func:`lower_glsl_ray_interfaces` rewrites the GLSL spelling into the HLSL one, keeping each value's meaning:

* every interface variable becomes a ``static`` module variable (DXR shaders may keep private module state, and
  ``TraceRay``/``CallShader``/``ReportHit`` accept one as their argument); a variable whose type is not a struct is
  wrapped in a one-member struct, because DXR requires struct payloads, attributes and callable data, and its uses
  read the member;
* the entry point receives the incoming payload (closest-hit, any-hit, miss), the hit attributes (closest-hit,
  any-hit) and the incoming callable data (callable) as parameters, copies them into those variables first and
  copies the ``inout`` ones back before every ``return``, ``IgnoreHit()`` and ``AcceptHitAndEndSearch()``, and at the
  end; a stage that declares none still receives the parameter DXR requires, of a 4-byte placeholder struct;
* ``TraceRay`` gets a ``RayDesc`` built by a helper function and the payload variable of its location;
  ``CallShader`` the callable data variable of its location, ``ReportHit`` the hit attribute variable, or a
  placeholder like the entry parameter's when the stage reports no attributes;
* the built-in variables become the DXR intrinsics with the same value, converted to the GLSL type where the two
  differ (``gl_PrimitiveID`` is an ``int``, ``PrimitiveIndex()`` a ``uint``); ray flag and hit kind constants become
  the DXR names of the same values.

What has no DXR form, or no lowering yet, is refused with :class:`RayInterfaceUnsupportedError`: the object/world
transform matrices (their orientation in HLSL is not settled yet), ``gl_CullMaskEXT`` and
``gl_HitTriangleVertexPositionsEXT`` (no DXR equivalent before shader model 6.9), a payload location no variable
declares, ``IgnoreHit``/``AcceptHitAndEndSearch`` outside the entry point of a stage with an ``inout`` parameter,
and module-scope interface variables shared by several ray stages of one program.
"""

from __future__ import annotations

import copy

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .rewrite import child_slots, get_child, set_child

__all__ = ["RayInterfaceUnsupportedError", "lower_glsl_ray_interfaces"]

S = A.ShaderStage
_RAY_STAGES = frozenset(
    {
        S.RAY_GENERATION,
        S.RAY_INTERSECTION,
        S.RAY_CLOSEST_HIT,
        S.RAY_MISS,
        S.RAY_ANY_HIT,
        S.RAY_CALLABLE,
    }
)
# Interface variable attribute -> kind.
_KINDS = {
    "raypayloadext": "payload",
    "raypayloadnv": "payload",
    "raypayloadinext": "payload_in",
    "raypayloadinnv": "payload_in",
    "hitattributeext": "hit_attribute",
    "hitattributenv": "hit_attribute",
    "callabledataext": "callable",
    "callabledatanv": "callable",
    "callabledatainext": "callable_in",
    "callabledatainnv": "callable_in",
}
_STRUCT_PREFIX = {
    "payload": "CrossGLRayPayload",
    "payload_in": "CrossGLRayPayload",
    "hit_attribute": "CrossGLHitAttributes",
    "callable": "CrossGLCallableData",
    "callable_in": "CrossGLCallableData",
}
# Entry parameters each stage receives: (kind of the variable it fills, parameter attribute, qualifier).
_ENTRY_PARAMETERS = {
    S.RAY_CLOSEST_HIT: (
        ("payload_in", "payload", "inout"),
        ("hit_attribute", "hit_attribute", "in"),
    ),
    S.RAY_ANY_HIT: (
        ("payload_in", "payload", "inout"),
        ("hit_attribute", "hit_attribute", "in"),
    ),
    S.RAY_MISS: (("payload_in", "payload", "inout"),),
    S.RAY_CALLABLE: (("callable_in", "callable_data", "inout"),),
}
# GLSL built-in -> (DXR intrinsic, GLSL type when it differs from the intrinsic's).
_BUILTINS = {
    "gl_LaunchIDEXT": ("DispatchRaysIndex", None),
    "gl_LaunchIDNV": ("DispatchRaysIndex", None),
    "gl_LaunchSizeEXT": ("DispatchRaysDimensions", None),
    "gl_LaunchSizeNV": ("DispatchRaysDimensions", None),
    "gl_WorldRayOriginEXT": ("WorldRayOrigin", None),
    "gl_WorldRayDirectionEXT": ("WorldRayDirection", None),
    "gl_ObjectRayOriginEXT": ("ObjectRayOrigin", None),
    "gl_ObjectRayDirectionEXT": ("ObjectRayDirection", None),
    "gl_RayTminEXT": ("RayTMin", None),
    "gl_RayTmaxEXT": ("RayTCurrent", None),
    "gl_HitTEXT": ("RayTCurrent", None),
    "gl_HitKindEXT": ("HitKind", None),
    "gl_IncomingRayFlagsEXT": ("RayFlags", None),
    "gl_PrimitiveID": ("PrimitiveIndex", "int"),
    "gl_InstanceID": ("InstanceIndex", "int"),
    "gl_InstanceCustomIndexEXT": ("InstanceID", "int"),
    "gl_GeometryIndexEXT": ("GeometryIndex", "int"),
}
_UNSUPPORTED_BUILTINS = {
    "gl_ObjectToWorldEXT": "the object-to-world matrix",
    "gl_WorldToObjectEXT": "the world-to-object matrix",
    "gl_ObjectToWorld3x4EXT": "the object-to-world matrix",
    "gl_WorldToObject3x4EXT": "the world-to-object matrix",
    "gl_CullMaskEXT": "the cull mask",
    "gl_HitTriangleVertexPositionsEXT": "the hit triangle's vertex positions",
}
# GL_EXT_ray_tracing constants and the DXR constants with the same values.
_CONSTANTS = {
    "gl_RayFlagsNoneEXT": "RAY_FLAG_NONE",
    "gl_RayFlagsOpaqueEXT": "RAY_FLAG_FORCE_OPAQUE",
    "gl_RayFlagsNoOpaqueEXT": "RAY_FLAG_FORCE_NON_OPAQUE",
    "gl_RayFlagsTerminateOnFirstHitEXT": "RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH",
    "gl_RayFlagsSkipClosestHitShaderEXT": "RAY_FLAG_SKIP_CLOSEST_HIT_SHADER",
    "gl_RayFlagsCullBackFacingTrianglesEXT": "RAY_FLAG_CULL_BACK_FACING_TRIANGLES",
    "gl_RayFlagsCullFrontFacingTrianglesEXT": "RAY_FLAG_CULL_FRONT_FACING_TRIANGLES",
    "gl_RayFlagsCullOpaqueEXT": "RAY_FLAG_CULL_OPAQUE",
    "gl_RayFlagsCullNoOpaqueEXT": "RAY_FLAG_CULL_NON_OPAQUE",
    "gl_RayFlagsSkipTrianglesEXT": "RAY_FLAG_SKIP_TRIANGLES",
    "gl_RayFlagsSkipAABBEXT": "RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES",
    "gl_HitKindFrontFacingTriangleEXT": "HIT_KIND_TRIANGLE_FRONT_FACE",
    "gl_HitKindBackFacingTriangleEXT": "HIT_KIND_TRIANGLE_BACK_FACE",
}
_TERMINATORS = frozenset({"IgnoreHit", "AcceptHitAndEndSearch"})
_RAY_DESC_HELPER = "crossgl_ray_desc"


class RayInterfaceUnsupportedError(TranslationDiagnostic):
    """A GLSL-style ray tracing construct with no DXR form, or none CrossTL lowers yet."""

    def __init__(self, stage, capability, detail):
        self.stage = stage
        self.missing_capabilities = (f"directx.{capability}",)
        super().__init__(
            f"DirectX {stage} stage: {detail}",
            target="directx",
            code=f"{CODE_PREFIX}directx-ray-interface-unsupported",
        )


def _stages(program):
    stages = getattr(program, "stages", None) or {}
    return list(stages.values()) if isinstance(stages, dict) else list(stages)


def _attribute_names(node):
    return [
        str(getattr(attribute, "name", ""))
        for attribute in getattr(node, "attributes", None) or []
    ]


def _kind(variable):
    for name in _attribute_names(variable):
        kind = _KINDS.get(name.lower())
        if kind:
            return kind
    return None


def _literal_int(node):
    if isinstance(node, A.LiteralNode) and not isinstance(node.value, bool):
        text = str(node.value).strip().rstrip("uU")
        try:
            return int(text, 0)
        except ValueError:
            return None
    return None


def _location(variable):
    for attribute in getattr(variable, "attributes", None) or []:
        if (
            str(getattr(attribute, "name", "")).lower() == "location"
            and attribute.arguments
        ):
            return _literal_int(attribute.arguments[0])
    return None


def _stage_label(stage):
    return getattr(stage.stage, "value", str(stage.stage))


def _functions(program):
    yield from getattr(program, "functions", None) or []
    for stage in _stages(program):
        yield from getattr(stage, "local_functions", None) or []
        if getattr(stage, "entry_point", None) is not None:
            yield stage.entry_point


def _transform(node, visit):
    """Replace, bottom-up, every structural descendant ``d`` of ``node`` with ``visit(d)`` when that is not None."""
    for field, key in list(child_slots(node)):
        child = get_child(node, field, key)
        _transform(child, visit)
        replacement = visit(child)
        if replacement is not None and replacement is not child:
            set_child(node, field, key, replacement)


def _walk(node):
    yield node
    for field, key in list(child_slots(node)):
        yield from _walk(get_child(node, field, key))


def _uses_glsl_spelling(program, variables):
    if variables:
        return True
    for function in _functions(program):
        for node in _walk(function):
            if isinstance(node, A.IdentifierNode) and (
                node.name in _BUILTINS
                or node.name in _CONSTANTS
                or node.name in _UNSUPPORTED_BUILTINS
            ):
                return True
            if (
                isinstance(node, A.RayTracingOpNode)
                and node.operation == "TraceRay"
                and len(node.arguments) == 11
            ):
                return True
    return False


class _Lowering:
    def __init__(self, program, stage):
        self.program = program
        self.stage = stage
        self.label = _stage_label(stage)
        self.taken = {struct.name for struct in getattr(program, "structs", None) or []}
        self.taken.update(function.name for function in _functions(program))
        self.locals = set()
        for function in _functions(program):
            for node in _walk(function):
                if isinstance(node, A.VariableNode):
                    self.locals.add(node.name)
            self.locals.update(
                parameter.name for parameter in function.parameters or []
            )
        self.taken.update(self.locals)
        for variable in [
            *(program.global_variables or []),
            *(stage.local_variables or []),
        ]:
            self.taken.add(getattr(variable, "name", None))
        self.structs = {
            struct.name for struct in getattr(program, "structs", None) or []
        }
        self.by_kind = {}
        self.wrapped = set()
        self.unused_attributes = None

    def refuse(self, capability, detail):
        raise RayInterfaceUnsupportedError(self.label, capability, detail)

    def unused(self, base):
        name, suffix = base, 1
        while name in self.taken:
            name, suffix = f"{base}_{suffix}", suffix + 1
        self.taken.add(name)
        return name

    def add_struct(self, base, members):
        name = self.unused(base)
        self.program.structs.append(A.StructNode(name, members))
        self.structs.add(name)
        return name

    # Interface variables -------------------------------------------------------------------------------------
    def declare(self, variable, kind):
        if variable.name in self.locals:
            self.refuse(
                "ray-interface-variable",
                f"ray interface variable '{variable.name}' is shadowed by a local or parameter of the same name",
            )
        location = _location(variable)
        var_type = variable.var_type
        if (
            isinstance(var_type, A.NamedType)
            and var_type.name in self.structs
            and not var_type.generic_args
        ):
            struct_name = var_type.name
        else:
            struct_name = self.add_struct(
                f"{_STRUCT_PREFIX[kind]}_{variable.name}",
                [A.StructMemberNode("value", copy.deepcopy(var_type))],
            )
            self.wrapped.add(variable.name)
        variable.var_type = variable.vtype = A.NamedType(struct_name)
        variable.attributes = [
            attribute
            for attribute in variable.attributes or []
            if str(getattr(attribute, "name", "")).lower() not in _KINDS
            and str(getattr(attribute, "name", "")).lower() != "location"
        ]
        variable.semantic = None
        variable.qualifiers = [
            "static",
            *[q for q in variable.qualifiers or [] if str(q) != "static"],
        ]
        self.by_kind.setdefault(kind, []).append((variable, location))

    def variable_at(self, kinds, location, call):
        if location is None:
            self.refuse(
                "ray-interface-location",
                f"{call} needs a constant payload or callable data location",
            )
        for kind in kinds:
            for variable, declared in self.by_kind.get(kind, []):
                if declared == location:
                    return variable
        self.refuse(
            "ray-interface-location",
            f"{call} names location {location}, which no variable declares",
        )

    def single(self, kind):
        variables = self.by_kind.get(kind, [])
        if len(variables) > 1:
            names = ", ".join(variable.name for variable, _ in variables)
            self.refuse(
                "ray-interface-variable",
                f"several {kind.replace('_', ' ')} variables: {names}",
            )
        return variables[0][0] if variables else None

    def placeholder_struct(self, kind):
        return self.add_struct(
            f"{_STRUCT_PREFIX[kind]}Unused",
            [A.StructMemberNode("unused", A.PrimitiveType("uint"))],
        )

    def placeholder_attributes(self):
        """The attributes a stage without a hitAttributeEXT variable reports: a 4-byte struct no stage reads."""
        if self.unused_attributes is None:
            self.unused_attributes = A.VariableNode(
                self.unused("crossgl_hit_attributes"),
                A.NamedType(self.placeholder_struct("hit_attribute")),
                qualifiers=["static"],
            )
            self.program.global_variables.append(self.unused_attributes)
        return self.unused_attributes

    # Expressions ---------------------------------------------------------------------------------------------
    def ray_desc_helper(self):
        if any(
            function.name == _RAY_DESC_HELPER
            for function in self.program.functions or []
        ):
            return _RAY_DESC_HELPER
        ray = A.IdentifierNode("ray")
        parameters = [
            A.ParameterNode("origin", A.VectorType(A.PrimitiveType("float"), 3)),
            A.ParameterNode("tmin", A.PrimitiveType("float")),
            A.ParameterNode("direction", A.VectorType(A.PrimitiveType("float"), 3)),
            A.ParameterNode("tmax", A.PrimitiveType("float")),
        ]
        statements = [A.VariableNode("ray", A.NamedType("RayDesc"))]
        for member, parameter in (
            ("Origin", "origin"),
            ("TMin", "tmin"),
            ("Direction", "direction"),
            ("TMax", "tmax"),
        ):
            statements.append(
                A.AssignmentNode(
                    A.MemberAccessNode(ray, member), A.IdentifierNode(parameter)
                )
            )
        statements.append(A.ReturnNode(A.IdentifierNode("ray")))
        self.program.functions.insert(
            0,
            A.FunctionNode(
                _RAY_DESC_HELPER,
                A.NamedType("RayDesc"),
                parameters,
                A.BlockNode(statements),
            ),
        )
        self.taken.add(_RAY_DESC_HELPER)
        return _RAY_DESC_HELPER

    def rewrite_call(self, node):
        operation, arguments = node.operation, node.arguments
        if operation == "TraceRay" and len(arguments) == 11:
            payload = self.variable_at(
                ("payload", "payload_in"), _literal_int(arguments[10]), "TraceRay"
            )
            desc = A.FunctionCallNode(
                A.IdentifierNode(self.ray_desc_helper()), list(arguments[6:10])
            )
            node.arguments[:] = [*arguments[:6], desc, A.IdentifierNode(payload.name)]
        elif (
            operation == "CallShader"
            and len(arguments) == 2
            and _literal_int(arguments[1]) is not None
        ):
            data = self.variable_at(
                ("callable", "callable_in"), _literal_int(arguments[1]), "CallShader"
            )
            node.arguments[1] = A.IdentifierNode(data.name)
        elif operation == "ReportHit" and len(arguments) == 2:
            attributes = self.single("hit_attribute") or self.placeholder_attributes()
            node.arguments.append(A.IdentifierNode(attributes.name))

    def rewrite_expression(self, node):
        if isinstance(node, A.RayTracingOpNode):
            self.rewrite_call(node)
            return None
        if not isinstance(node, A.IdentifierNode):
            return None
        name = node.name
        if name in self.wrapped:
            return A.MemberAccessNode(A.IdentifierNode(name), "value")
        if name in _CONSTANTS:
            return A.IdentifierNode(_CONSTANTS[name])
        if name in _UNSUPPORTED_BUILTINS:
            self.refuse(
                "ray-builtin",
                f"{name} ({_UNSUPPORTED_BUILTINS[name]}) has no DXR lowering in CrossTL yet",
            )
        if name in _BUILTINS:
            intrinsic, glsl_type = _BUILTINS[name]
            value = A.FunctionCallNode(A.IdentifierNode(intrinsic), [])
            return (
                A.FunctionCallNode(A.IdentifierNode(glsl_type), [value])
                if glsl_type
                else value
            )
        return None

    # Entry point ---------------------------------------------------------------------------------------------
    def entry_parameters(self, entry):
        roles = _ENTRY_PARAMETERS.get(self.stage.stage, ())
        if not roles:
            return [], []
        if entry.parameters:
            self.refuse(
                "ray-entry-parameters",
                "the entry declares parameters as well as module-scope ray interface variables",
            )
        copy_in, copy_out = [], []
        for kind, role, qualifier in roles:
            variable = self.single(kind)
            if variable is None:
                type_name = self.placeholder_struct(kind)
            else:
                type_name = variable.var_type.name
            parameter = A.ParameterNode(
                self.unused(f"crossgl_{role}"),
                A.NamedType(type_name),
                attributes=[A.AttributeNode(role)],
                qualifiers=[qualifier],
            )
            entry.parameters.append(parameter)
            if variable is None:
                continue
            copy_in.append(
                A.AssignmentNode(
                    A.IdentifierNode(variable.name), A.IdentifierNode(parameter.name)
                )
            )
            if qualifier == "inout":
                copy_out.append((parameter.name, variable.name))
        return copy_in, copy_out

    def copies_out(self, copy_out):
        return [
            A.AssignmentNode(A.IdentifierNode(param), A.IdentifierNode(name))
            for param, name in copy_out
        ]

    def is_terminator(self, node):
        expression = getattr(node, "expression", None)
        return (
            isinstance(node, A.ExpressionStatementNode)
            and isinstance(expression, A.RayTracingOpNode)
            and expression.operation in _TERMINATORS
        )

    def guard_exits(self, entry, copy_out):
        def visit(node):
            if isinstance(node, A.ReturnNode) or self.is_terminator(node):
                return A.BlockNode([*self.copies_out(copy_out), node])
            return None

        statements = entry.body.statements
        ends_with_return = bool(statements) and isinstance(statements[-1], A.ReturnNode)
        _transform(entry.body, visit)
        if not ends_with_return:
            statements.extend(self.copies_out(copy_out))

    def refuse_helper_terminators(self, entry):
        for function in _functions(self.program):
            if function is entry:
                continue
            for node in _walk(function):
                if (
                    isinstance(node, A.RayTracingOpNode)
                    and node.operation in _TERMINATORS
                ):
                    self.refuse(
                        "ray-terminator-in-helper",
                        f"{node.operation} in function '{function.name}' ends the shader where the payload "
                        "parameter is not visible, and copying the payload back there is not implemented yet",
                    )

    def drop_builtin_declarations(self):
        self.stage.local_variables = [
            variable
            for variable in self.stage.local_variables or []
            if not (
                isinstance(variable, A.VariableNode)
                and (
                    variable.name in _BUILTINS or variable.name in _UNSUPPORTED_BUILTINS
                )
                and "input" in {name.lower() for name in _attribute_names(variable)}
            )
        ]

    def run(self, variables):
        for variable, kind in variables:
            self.declare(variable, kind)
        self.drop_builtin_declarations()
        for function in list(_functions(self.program)):
            _transform(function, self.rewrite_expression)
        entry = self.stage.entry_point
        if not variables and entry.parameters:
            return  # HLSL-spelled entry that only reads GLSL-spelled built-ins
        copy_in, copy_out = self.entry_parameters(entry)
        if copy_out:
            self.refuse_helper_terminators(entry)
            self.guard_exits(entry, copy_out)
        entry.body.statements[:0] = copy_in


def lower_glsl_ray_interfaces(program):
    """Rewrite GLSL-style ray tracing interfaces of ``program`` into their DXR form (see the module docstring)."""
    ray_stages = [
        stage
        for stage in _stages(program)
        if getattr(stage, "stage", None) in _RAY_STAGES
    ]
    if not ray_stages:
        return
    stage_variables = {
        id(stage): [(v, _kind(v)) for v in stage.local_variables or [] if _kind(v)]
        for stage in ray_stages
    }
    global_variables = [
        (v, _kind(v)) for v in program.global_variables or [] if _kind(v)
    ]
    if global_variables and len(ray_stages) > 1:
        raise RayInterfaceUnsupportedError(
            _stage_label(ray_stages[0]),
            "ray-interface-variable",
            "module-scope ray interface variables are shared by several ray stages of one program; "
            "declare them in each stage",
        )
    for stage in ray_stages:
        variables = [*global_variables, *stage_variables[id(stage)]]
        if _uses_glsl_spelling(program, variables) and stage.entry_point is not None:
            _Lowering(program, stage).run(variables)
