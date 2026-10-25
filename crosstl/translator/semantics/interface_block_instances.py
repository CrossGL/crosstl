"""GLSL interface block instances declared as SPIR-V stage interface variables.

GLSL declares the user interface of a geometry, tessellation or mesh stage with blocks (GLSL 4.60 section 4.3.9):
``layout(location = 0) out Block { vec2 uv; flat uint id; } vout[];``. Those stages keep their interface at module
scope, and the GLSL importer keeps such a block as a structure that carries the block's qualifiers, layout and
instance (``@glsl_interface_block(out) @location(0) @glsl_interface_instance(vout) @glsl_interface_array struct
Block``), while the stage's functions read and write ``vout[i].uv``. The GLSL emitter prints the block back. The
SPIR-V emitter declared no variable for the instance, so it refused every use ("it does not name storage",
"Unknown variable").

:func:`declare_interface_block_instances` declares, for SPIR-V, one module-scope variable per instance the program
uses: ``out Block vout @location(0)[]``. The variable takes the block's direction and its ``patch``,
``perprimitive`` and interpolation qualifiers, the block's ``location`` and the block's array extents; an
unsized instance array then gets its count from its stage like any other GLSL stage array (:mod:`.stage_io_arrays`).
A structure-typed ``Input`` or ``Output`` variable with a ``Location`` gives its members consecutive locations
(Vulkan 1.3 section 15.1.5, "Location Assignment"), the locations GLSL gives the block's members, so it matches the
adjacent stage's block or variables. Interpolation decorations on outputs need not match the next stage's inputs
(Vulkan 1.3 section 15.1.3, "Interface Matching").

A block whose members carry their own ``location`` or ``component``, or which carries a layout the variable cannot
take (transform feedback, ``stream``), is refused with :class:`InterfaceBlockInstanceUnsupportedError`: a
non-``Block`` structure has no member locations, and the SPIR-V emitter does not lower transform feedback.
"""

from __future__ import annotations

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .hub_types import iter_functions, iter_structs
from .rewrite import clone
from .storage_blocks import _walk

__all__ = [
    "InterfaceBlockInstanceUnsupportedError",
    "declare_interface_block_instances",
]

_TARGETS = frozenset({"vulkan", "spirv"})
_BLOCK = "glsl_interface_block"
_INSTANCE = "glsl_interface_instance"
_ARRAY = "glsl_interface_array"
# Block attributes the declared variable carries or that only describe the GLSL spelling.
_CARRIED = frozenset(
    {_BLOCK, _INSTANCE, _ARRAY, "glsl_interface_block_name", "location"}
)
_MEMBER_LAYOUT = frozenset({"location", "component"})
_DIRECTIONS = frozenset({"in", "out"})
_PER_PRIMITIVE = frozenset({"perprimitive", "perprimitiveext", "perprimitivenv"})
_KEPT_QUALIFIERS = frozenset(
    {
        "patch",
        "flat",
        "smooth",
        "noperspective",
        "centroid",
        "sample",
        "invariant",
        "precise",
    }
)


class InterfaceBlockInstanceUnsupportedError(TranslationDiagnostic):
    """A GLSL interface block instance that no SPIR-V stage interface variable can declare."""

    def __init__(self, target, block, reason, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = (f"{target}.interface-block-instance",)
        super().__init__(
            f"GLSL interface block '{block}' cannot be declared as a {target} stage interface variable: {reason}",
            target=target,
            code=f"{CODE_PREFIX}{target}-interface-block-instance-unsupported",
        )


def _argument_names(attribute):
    names = []
    for argument in getattr(attribute, "arguments", None) or []:
        value = getattr(argument, "name", None)
        if value is None:
            value = getattr(argument, "value", argument)
        names.append(str(value))
    return names


def _attribute(node, name):
    for attribute in getattr(node, "attributes", None) or []:
        if getattr(attribute, "name", None) == name:
            return attribute
    return None


def _instance_name(struct):
    instance = _attribute(struct, _INSTANCE)
    if instance is None or _attribute(struct, _BLOCK) is None:
        return None
    names = _argument_names(instance)
    return names[0] if len(names) == 1 and names[0] else None


def _used_names(program):
    names = set()
    for function in iter_functions(program):
        names.update(
            node.name
            for node in _walk(function.body)
            if isinstance(node, A.IdentifierNode)
        )
    return names


def _declared_names(program):
    names = {
        getattr(v, "name", None)
        for v in getattr(program, "global_variables", None) or []
    }
    stages = getattr(program, "stages", None) or {}
    for stage in stages.values() if isinstance(stages, dict) else stages:
        names.update(
            getattr(v, "name", None)
            for v in getattr(stage, "local_variables", None) or []
        )
    return names


class _Declarer:
    def __init__(self, program, target):
        self.program = program
        self.target = target

    def refuse(self, struct, reason):
        raise InterfaceBlockInstanceUnsupportedError(
            self.target,
            getattr(struct, "name", "?"),
            reason,
            source_location=getattr(struct, "source_location", None),
        )

    def qualifiers(self, struct):
        qualifiers = []
        direction = None
        for name in _argument_names(_attribute(struct, _BLOCK)):
            lowered = name.lower()
            if lowered in _DIRECTIONS:
                direction = lowered
            elif lowered in _PER_PRIMITIVE:
                qualifiers.append("perprimitive")
            elif lowered in _KEPT_QUALIFIERS:
                qualifiers.append(lowered)
            else:
                self.refuse(
                    struct,
                    f"the block qualifier '{name}' has no stage interface variable form",
                )
        if direction is None:
            self.refuse(struct, "the block is neither an 'in' nor an 'out' block")
        return qualifiers + [direction]

    def variable_type(self, struct):
        var_type = A.NamedType(struct.name)
        array = _attribute(struct, _ARRAY)
        if array is None:
            return var_type
        sizes = list(getattr(array, "arguments", None) or []) or [None]
        for size in reversed(sizes):
            var_type = A.ArrayType(var_type, None if size is None else clone(size))
        return var_type

    def check(self, struct):
        for attribute in getattr(struct, "attributes", None) or []:
            name = getattr(attribute, "name", None)
            if name not in _CARRIED:
                self.refuse(
                    struct, f"its layout '{name}' has no stage interface variable form"
                )
        for member in getattr(struct, "members", None) or []:
            for attribute in getattr(member, "attributes", None) or []:
                if getattr(attribute, "name", None) in _MEMBER_LAYOUT:
                    self.refuse(
                        struct,
                        f"member '{getattr(member, 'name', '?')}' declares its own "
                        f"{attribute.name}, which a non-Block structure member cannot carry",
                    )

    def declare(self, struct, name):
        self.check(struct)
        attributes = [
            A.AttributeNode("location", [clone(a) for a in location.arguments])
            for location in [_attribute(struct, "location")]
            if location is not None
        ]
        var_type = self.variable_type(struct)
        variable = A.VariableNode(
            name,
            var_type,
            qualifiers=self.qualifiers(struct),
            attributes=attributes,
            source_location=getattr(struct, "source_location", None),
        )
        return variable

    def run(self):
        used = None
        declared = None
        added = []
        for struct in iter_structs(self.program):
            if not isinstance(struct, A.StructNode):
                continue
            name = _instance_name(struct)
            if name is None or str(getattr(struct, "name", "")).startswith("gl_"):
                continue
            if used is None:
                used, declared = _used_names(self.program), _declared_names(
                    self.program
                )
            if name not in used or name in declared:
                continue
            added.append(self.declare(struct, name))
            declared.add(name)
        if added:
            self.program.global_variables = (
                list(self.program.global_variables or []) + added
            )
        return len(added)


def declare_interface_block_instances(program, target) -> int:
    """Declare the GLSL interface block instances ``program`` uses as SPIR-V interface variables; returns how many."""
    target = str(target or "").lower()
    if target not in _TARGETS or not isinstance(program, A.ShaderNode):
        return 0
    return _Declarer(program, target).run()
