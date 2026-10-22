"""DXR ray tracing interfaces lowered to the GL_EXT_ray_tracing forms GLSL declares.

This is the inverse of :mod:`.hlsl_ray_interfaces`, which describes both spellings CrossGL carries. HLSL (DXR 1.0)
passes the payload, hit attributes and callable data as entry parameters marked ``@payload``, ``@hit_attribute``
and ``@callable_data``, hands ``TraceRay``/``CallShader``/``ReportHit`` a value of the payload, callable data or
attribute struct, and reads ray state through intrinsic calls. GLSL (GLSL_EXT_ray_tracing, sections 4.3.x and 8.19)
declares module-scope ``rayPayloadEXT``, ``rayPayloadInEXT``, ``hitAttributeEXT``, ``callableDataEXT`` and
``callableDataInEXT`` variables, names a payload or callable data by its ``location`` and reads ray state from
built-in variables. :func:`lower_dxr_ray_interfaces` rewrites the first spelling into the second for GLSL and for
SPIR-V, whose generator declares the GLSL variables in the ``IncomingRayPayloadKHR``, ``RayPayloadKHR``,
``HitAttributeKHR``, ``CallableDataKHR`` and ``IncomingCallableDataKHR`` storage classes and the built-ins with
their ``BuiltIn`` decorations (SPV_KHR_ray_tracing). Without it the SPIR-V generator declared an HLSL payload as a
stage input, so a hit or miss shader wrote its result to a private copy the caller never saw:

* an ``inout`` payload or callable data entry parameter becomes the ``rayPayloadInEXT`` or ``callableDataInEXT``
  variable of its type, under the parameter's name, so every read and write reaches the caller's value as DXR's
  copy-in/copy-out does by the time the shader ends (``IgnoreHit`` keeps the writes made before it in both);
* the ``in`` hit attribute parameter becomes a local copy of the ``hitAttributeEXT`` variable, which GLSL makes
  read-only in hit shaders while an HLSL ``in`` parameter may be written;
* ``TraceRay(as, flags, mask, offset, stride, miss, ray, payload)`` copies ``payload`` into a ``rayPayloadEXT``
  variable of its type, traces with the ray's origin, ``TMin``, direction and ``TMax`` and that variable's location,
  and copies the result back; ``CallShader(index, data)`` does the same with a ``callableDataEXT`` variable;
  ``ReportHit(t, kind, attributes)`` stores the attributes in the stage's ``hitAttributeEXT`` variable first;
* the ray state intrinsics become the built-in variables with the same value, converted to the DXR type where the
  two differ (``PrimitiveIndex()`` is a ``uint``, ``gl_PrimitiveID`` an ``int``), the DXR ray flag and hit kind
  constants the GLSL names of the same values, and ``RayDesc`` a struct with DXR's members;
* the transform intrinsics become ``gl_ObjectToWorldEXT`` and ``gl_WorldToObjectEXT``. CrossGL spells a matrix
  with GLSL's columns-by-rows name and keeps its mathematical shape, so HLSL's 3x4 matrix ``float3x4`` is CrossGL's
  ``mat4x3``, the type of those built-ins (both the 3x4 affine transform), and ``ObjectToWorld4x3()``, its transpose,
  is ``transpose(gl_ObjectToWorldEXT)``, GLSL's ``gl_ObjectToWorld3x4EXT``. DXC's SPIR-V back end reads the same
  ``ObjectToWorldKHR`` built-in for both.

What has no lowering yet is refused with :class:`DxrRayInterfaceUnsupportedError`: a payload, callable data or attribute argument whose struct type
the pass cannot tell, a trace, callable or hit report that is not a statement of its own or the operand of one,
several DXR ray stages in one program, and a program that mixes both spellings.
"""

from __future__ import annotations

import copy

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hlsl_ray_interfaces import (
    _CONSTANTS,
    _RAY_STAGES,
    _functions,
    _kind,
    _literal_int,
    _stage_label,
    _stages,
    _transform,
    _walk,
)
from .stage_varyings import GLSL_TARGETS, SPIRV_TARGETS

__all__ = ["DxrRayInterfaceUnsupportedError", "lower_dxr_ray_interfaces"]

S = A.ShaderStage
# Entry parameter attribute -> (GLSL storage qualifier of its variable, whether it has a location).
_ENTRY_STORAGE = {
    "payload": ("rayPayloadInEXT", True),
    "callable_data": ("callableDataInEXT", True),
    "hit_attribute": ("hitAttributeEXT", False),
}
# DXR intrinsic -> (GLSL built-in, DXR type when it differs from the built-in's).
_INTRINSICS = {
    "DispatchRaysIndex": ("gl_LaunchIDEXT", None),
    "DispatchRaysDimensions": ("gl_LaunchSizeEXT", None),
    "WorldRayOrigin": ("gl_WorldRayOriginEXT", None),
    "WorldRayDirection": ("gl_WorldRayDirectionEXT", None),
    "ObjectRayOrigin": ("gl_ObjectRayOriginEXT", None),
    "ObjectRayDirection": ("gl_ObjectRayDirectionEXT", None),
    "RayTMin": ("gl_RayTminEXT", None),
    "RayTCurrent": ("gl_RayTmaxEXT", None),
    "HitKind": ("gl_HitKindEXT", None),
    "RayFlags": ("gl_IncomingRayFlagsEXT", None),
    "PrimitiveIndex": ("gl_PrimitiveID", "uint"),
    "InstanceIndex": ("gl_InstanceID", "uint"),
    "InstanceID": ("gl_InstanceCustomIndexEXT", "uint"),
    "GeometryIndex": ("gl_GeometryIndexEXT", "uint"),
}
# DXR transform intrinsic -> (GLSL built-in, whether the DXR matrix is its transpose).
_TRANSFORMS = {
    "ObjectToWorld": ("gl_ObjectToWorldEXT", False),
    "ObjectToWorld3x4": ("gl_ObjectToWorldEXT", False),
    "ObjectToWorld4x3": ("gl_ObjectToWorldEXT", True),
    "WorldToObject": ("gl_WorldToObjectEXT", False),
    "WorldToObject3x4": ("gl_WorldToObjectEXT", False),
    "WorldToObject4x3": ("gl_WorldToObjectEXT", True),
}
TARGETS = GLSL_TARGETS | SPIRV_TARGETS
_TARGET_LABELS = {
    "opengl": "OpenGL",
    "glsl": "OpenGL",
    "vulkan": "Vulkan",
    "spirv": "Vulkan",
}
_DXR_CONSTANTS = {dxr: glsl for glsl, dxr in _CONSTANTS.items()}
_RAY_DESC_MEMBERS = (
    ("Origin", A.VectorType(A.PrimitiveType("float"), 3)),
    ("TMin", A.PrimitiveType("float")),
    ("Direction", A.VectorType(A.PrimitiveType("float"), 3)),
    ("TMax", A.PrimitiveType("float")),
)
# DXR's built-in structs (HLSL 6.3 ray tracing, "RayDesc" and "BuiltInTriangleIntersectionAttributes") -> members.
# GLSL and SPIR-V have no such types; the triangle attributes are the two barycentrics GL_EXT_ray_tracing reads
# from a ``hitAttributeEXT vec2``, which a one-member struct of a ``vec2`` lays out the same way.
_BUILTIN_STRUCTS = {
    "RayDesc": _RAY_DESC_MEMBERS,
    "BuiltInTriangleIntersectionAttributes": (
        ("barycentrics", A.VectorType(A.PrimitiveType("float"), 2)),
    ),
}


class DxrRayInterfaceUnsupportedError(TranslationDiagnostic):
    """A DXR ray tracing construct CrossTL does not lower to GL_EXT_ray_tracing or SPV_KHR_ray_tracing yet."""

    def __init__(self, stage, capability, detail, target="opengl"):
        target = "vulkan" if str(target).lower() in SPIRV_TARGETS else "opengl"
        self.stage = stage
        self.missing_capabilities = (f"{target}.{capability}",)
        super().__init__(
            f"{_TARGET_LABELS[target]} {stage} stage: {detail}",
            target=target,
            code=f"{CODE_PREFIX}{target}-ray-interface-unsupported",
        )


def _attribute(node, names):
    for attribute in getattr(node, "attributes", None) or []:
        name = str(getattr(attribute, "name", ""))
        if name in names:
            return name
    return None


def _int(value):
    return A.LiteralNode(value, A.PrimitiveType("int"))


def _type_name(type_node):
    if isinstance(type_node, A.NamedType) and not type_node.generic_args:
        return type_node.name
    return None


def _is_dxr_call(node):
    if isinstance(node, A.RayTracingOpNode):
        count = len(node.arguments or [])
        return (
            (node.operation == "TraceRay" and count == 8)
            or (node.operation == "ReportHit" and count == 3)
            or (
                node.operation == "CallShader"
                and count == 2
                and _literal_int(node.arguments[1]) is None
            )
        )
    if isinstance(node, A.FunctionCallNode):
        function = getattr(node, "function", None)
        name = getattr(function, "name", function)
        return name in _INTRINSICS or name in _TRANSFORMS
    return isinstance(node, A.IdentifierNode) and node.name in _DXR_CONSTANTS


def _dxr_stages(program):
    found = []
    for stage in _stages(program):
        entry = getattr(stage, "entry_point", None)
        if getattr(stage, "stage", None) not in _RAY_STAGES or entry is None:
            continue
        if any(
            _attribute(parameter, _ENTRY_STORAGE)
            for parameter in entry.parameters or []
        ):
            found.append(stage)
    if found:
        return found
    ray_stages = [
        stage
        for stage in _stages(program)
        if getattr(stage, "stage", None) in _RAY_STAGES
        and getattr(stage, "entry_point", None) is not None
    ]
    for function in _functions(program):
        if any(_is_dxr_call(node) for node in _walk(function)):
            return ray_stages
    return []


class _Lowering:
    def __init__(self, program, stage, target="opengl"):
        self.program = program
        self.target = target
        self.stage = stage
        self.label = _stage_label(stage)
        self.taken = {struct.name for struct in program.structs or []}
        self.taken.update(function.name for function in _functions(program))
        self.types = {}
        for function in _functions(program):
            for node in _walk(function):
                if isinstance(node, A.VariableNode):
                    self.taken.add(node.name)
                    self.types.setdefault(node.name, _type_name(node.var_type))
            for parameter in function.parameters or []:
                self.taken.add(parameter.name)
                self.types.setdefault(parameter.name, _type_name(parameter.param_type))
        for variable in [
            *(program.global_variables or []),
            *(stage.local_variables or []),
        ]:
            self.taken.add(getattr(variable, "name", None))
        self.outgoing = {}  # (storage, struct name) -> module variable name
        self.locations = {"rayPayloadEXT": 0, "callableDataEXT": 0}
        self.hit_attributes = None
        # Calls already in the GLSL or SPIR-V form; a SPIR-V CallShader still has two operands afterwards.
        self.lowered_calls = set()

    def refuse(self, capability, detail):
        raise DxrRayInterfaceUnsupportedError(
            self.label, capability, detail, self.target
        )

    def unused(self, base):
        name, suffix = base, 1
        while name in self.taken:
            name, suffix = f"{base}_{suffix}", suffix + 1
        self.taken.add(name)
        return name

    def declare(self, name, type_node, storage, location=None):
        attributes = [A.AttributeNode(storage)]
        if location is not None:
            attributes.insert(0, A.AttributeNode("location", [_int(location)]))
        variable = A.VariableNode(name, copy.deepcopy(type_node), attributes=attributes)
        self.program.global_variables.append(variable)
        return variable

    # Entry point ---------------------------------------------------------------------------------------------
    def lower_entry(self, entry):
        kept, copies = [], []
        for parameter in entry.parameters or []:
            role = _attribute(parameter, _ENTRY_STORAGE)
            if role is None:
                kept.append(parameter)
                continue
            storage, located = _ENTRY_STORAGE[role]
            if _type_name(parameter.param_type) is None:
                self.refuse(
                    "ray-entry-parameter",
                    f"the {role.replace('_', ' ')} parameter '{parameter.name}' is not of a struct type",
                )
            if role == "hit_attribute":
                variable = self.declare(
                    self.unused("crossgl_hit_attributes"), parameter.param_type, storage
                )
                self.hit_attributes = variable
                copies.append(
                    A.VariableNode(
                        parameter.name,
                        copy.deepcopy(parameter.param_type),
                        initial_value=A.IdentifierNode(variable.name),
                    )
                )
                continue
            if any(
                getattr(variable, "name", None) == parameter.name
                for variable in self.program.global_variables or []
            ):
                self.refuse(
                    "ray-entry-parameter",
                    f"the {role.replace('_', ' ')} parameter '{parameter.name}' has the name of a module variable",
                )
            self.declare(parameter.name, parameter.param_type, storage, 0)
            if storage == "rayPayloadInEXT":
                self.locations["rayPayloadEXT"] = 1
            else:
                self.locations["callableDataEXT"] = 1
        entry.parameters[:] = kept
        entry.body.statements[:0] = copies

    # Calls ---------------------------------------------------------------------------------------------------
    def argument_type(self, expression, what, call):
        name = expression.name if isinstance(expression, A.IdentifierNode) else None
        struct = self.types.get(name) if name else None
        if struct is None:
            self.refuse(
                "ray-call-argument",
                f"the {what} argument of {call} is not a variable of a known struct type",
            )
        return struct

    def outgoing_variable(self, storage, struct):
        key = (storage, struct)
        if key not in self.outgoing:
            location = self.locations[storage]
            self.locations[storage] += 1
            base = (
                "crossgl_ray_payload"
                if storage == "rayPayloadEXT"
                else "crossgl_callable_data"
            )
            variable = self.declare(
                self.unused(base), A.NamedType(struct), storage, location
            )
            self.outgoing[key] = (variable.name, location)
        return self.outgoing[key]

    def payload_operand(self, name, location):
        """GLSL's traceRayEXT and executeCallableEXT name the variable by its location; OpTraceRayKHR and
        OpExecuteCallableKHR take the variable itself."""
        if self.target in SPIRV_TARGETS:
            return A.IdentifierNode(name)
        return _int(location)

    def copied_call(self, call, argument, storage, what, prefix=()):
        struct = self.argument_type(argument, what, call.operation)
        name, location = self.outgoing_variable(storage, struct)
        return A.BlockNode(
            [
                *prefix,
                A.AssignmentNode(A.IdentifierNode(name), copy.deepcopy(argument)),
                A.ExpressionStatementNode(call),
                A.AssignmentNode(copy.deepcopy(argument), A.IdentifierNode(name)),
            ]
        )

    def trace_ray(self, call):
        arguments = call.arguments
        ray, payload, prefix = arguments[6], arguments[7], []
        if not isinstance(ray, A.IdentifierNode):
            temporary = self.unused("crossgl_ray")
            prefix.append(
                A.VariableNode(temporary, A.NamedType("RayDesc"), initial_value=ray)
            )
            ray = A.IdentifierNode(temporary)
        struct = self.argument_type(payload, "payload", "TraceRay")
        name, location = self.outgoing_variable("rayPayloadEXT", struct)
        call.arguments = call.args = [
            *arguments[:6],
            *(
                A.MemberAccessNode(copy.deepcopy(ray), member)
                for member, _ in _RAY_DESC_MEMBERS
            ),
            self.payload_operand(name, location),
        ]
        return self.copied_call(call, payload, "rayPayloadEXT", "payload", prefix)

    def call_shader(self, call):
        data = call.arguments[1]
        struct = self.argument_type(data, "callable data", "CallShader")
        name, location = self.outgoing_variable("callableDataEXT", struct)
        call.arguments = call.args = [
            call.arguments[0],
            self.payload_operand(name, location),
        ]
        self.lowered_calls.add(id(call))
        return self.copied_call(call, data, "callableDataEXT", "callable data")

    def report_hits(self, statement):
        """Store the attributes of every ``ReportHit`` in ``statement``'s own expressions; returns the stores."""
        stores = []
        for field in ("expression", "condition", "initial_value", "value"):
            expression = getattr(statement, field, None)
            if not isinstance(expression, A.ASTNode):
                continue
            for node in _walk(expression):
                if not (
                    isinstance(node, A.RayTracingOpNode)
                    and node.operation == "ReportHit"
                    and len(node.arguments) == 3
                ):
                    continue
                attributes = node.arguments.pop()
                node.args = node.arguments
                struct = self.argument_type(attributes, "attributes", "ReportHit")
                if self.hit_attributes is None:
                    self.hit_attributes = self.declare(
                        self.unused("crossgl_hit_attributes"),
                        A.NamedType(struct),
                        "hitAttributeEXT",
                    )
                elif _type_name(self.hit_attributes.var_type) != struct:
                    self.refuse(
                        "ray-hit-attributes",
                        "ReportHit reports attributes of several struct types; GLSL declares one hitAttributeEXT variable",
                    )
                stores.append(
                    A.AssignmentNode(
                        A.IdentifierNode(self.hit_attributes.name), attributes
                    )
                )
        return stores

    def rewrite_statement(self, node):
        if isinstance(node, (A.WhileNode, A.DoWhileNode, A.ForNode)):
            return None
        if isinstance(node, A.ExpressionStatementNode) and isinstance(
            node.expression, A.RayTracingOpNode
        ):
            call = node.expression
            if call.operation == "TraceRay" and len(call.arguments) == 8:
                return self.trace_ray(call)
            if call.operation == "CallShader" and _is_dxr_call(call):
                return self.call_shader(call)
        if isinstance(node, (A.StatementNode, A.VariableNode)):
            stores = self.report_hits(node)
            if stores:
                return A.BlockNode([*stores, node])
        return None

    # Expressions ---------------------------------------------------------------------------------------------
    def rewrite_expression(self, node):
        if isinstance(node, A.IdentifierNode) and node.name in _DXR_CONSTANTS:
            return A.IdentifierNode(_DXR_CONSTANTS[node.name])
        if not isinstance(node, A.FunctionCallNode):
            return None
        function = getattr(node, "function", None)
        name = getattr(function, "name", function)
        if name in _TRANSFORMS and not node.arguments:
            builtin, transposed = _TRANSFORMS[name]
            value = A.IdentifierNode(builtin)
            return (
                A.FunctionCallNode(A.IdentifierNode("transpose"), [value])
                if transposed
                else value
            )
        if name not in _INTRINSICS or node.arguments:
            return None
        builtin, dxr_type = _INTRINSICS[name]
        value = A.IdentifierNode(builtin)
        return (
            A.FunctionCallNode(A.IdentifierNode(dxr_type), [value])
            if dxr_type
            else value
        )

    def add_builtin_structs(self):
        """Declare the DXR built-in structs the program uses and does not declare itself, in a fixed order."""
        used = set()
        for function in _functions(self.program):
            used.add(_type_name(getattr(function, "return_type", None)))
            used.update(
                _type_name(parameter.param_type)
                for parameter in function.parameters or []
            )
            for node in _walk(function):
                if isinstance(node, A.VariableNode):
                    used.add(_type_name(node.var_type))
        used.update(
            _type_name(getattr(variable, "var_type", None))
            for variable in self.program.global_variables or []
        )
        declared = {struct.name for struct in self.program.structs or []}
        for name in reversed(tuple(_BUILTIN_STRUCTS)):
            if name not in used or name in declared:
                continue
            members = [
                A.StructMemberNode(member, copy.deepcopy(kind))
                for member, kind in _BUILTIN_STRUCTS[name]
            ]
            self.program.structs.insert(0, A.StructNode(name, members))

    def run(self):
        self.lower_entry(self.stage.entry_point)
        for function in list(_functions(self.program)):
            _transform(function, self.rewrite_statement)
            _transform(function, self.rewrite_expression)
        self.add_builtin_structs()
        for function in _functions(self.program):
            for node in _walk(function):
                if (
                    isinstance(node, A.RayTracingOpNode)
                    and id(node) not in self.lowered_calls
                    and _is_dxr_call(node)
                ):
                    self.refuse(
                        "ray-call-position",
                        f"{node.operation} is not a statement of its own or an operand of one",
                    )


def lower_dxr_ray_interfaces(program, target) -> int:
    """Rewrite DXR-style ray tracing stages of ``program`` for GLSL or SPIR-V; returns how many stages changed."""
    target = str(target or "").lower()
    if target not in TARGETS or not isinstance(program, A.ShaderNode):
        return 0
    stages = _dxr_stages(program)
    if not stages:
        return 0
    label = _stage_label(stages[0])
    if len(stages) > 1:
        raise DxrRayInterfaceUnsupportedError(
            label,
            "ray-library",
            "several DXR ray stages in one program; each ray stage is lowered as a shader of its own",
            target,
        )
    variables = [
        variable
        for variable in [
            *(program.global_variables or []),
            *(stages[0].local_variables or []),
        ]
        if _kind(variable)
    ]
    if variables:
        raise DxrRayInterfaceUnsupportedError(
            label,
            "ray-mixed-spelling",
            "the stage mixes DXR entry parameters or calls with module-scope GLSL ray interface variables",
            target,
        )
    _Lowering(program, stages[0], target).run()
    return 1
